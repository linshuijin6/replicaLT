"""
adapter_v2/embedding_extractor.py
===================================
从预缓存的 patch tokens 提取三种 image embedding：

  (1) mean_pool     : patch_token.mean(dim=1)            → (B, 768)
  (2) attn_pool     : TokenAttentionPool(patch_token)     → (B, 512)  [ckpt frozen]
  (3) attn_pool_proj: proj_img(attn_pool)                 → (B, 512)  [ckpt frozen]

对于检索任务，所有 embedding 需统一到 512 维空间：
  - mean_pool 使用 token_pool.value (768→512) 做固定线性映射（来自 ckpt，不训练），
    得到 mean_pool_512，用于和文本 embedding 做 cosine similarity。
  - attn_pool / attn_pool_proj 本身已是 512 维，直接使用。

说明：
  - TokenAttentionPool 和 proj_img 均从 CoCoOp checkpoint 加载，eval() + requires_grad=False
  - 不加载 ContextNet / ClassBranch / PlasmaBranch
  - 直接读取预缓存的 .vision.pt 文件（region_token），无需重新运行 ImageEncoder
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 将 CLIP-MRI2PET 加入 sys.path ──────────────────────────────────────────
CLIP_MRI2PET_ROOT = Path(__file__).resolve().parents[2] / "CLIP-MRI2PET"
for _p in [str(CLIP_MRI2PET_ROOT), str(CLIP_MRI2PET_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from clip_mri2pet.models.biomedclip_image import load_biomedclip_model, VisionTransformerAdapter
from models import TokenAttentionPool, ProjectionHead  # adapter_v2/models.py


# ============================================================================
# EmbeddingExtractor
# ============================================================================

class EmbeddingExtractor(nn.Module):
    """
    从预缓存的 patch tokens [B, N, 768] 提取三种 image embedding。

    所有层固定（eval + no_grad），从各自权重来源加载。

    Attributes
    ----------
    token_dim             : 输入 patch token 维度，通常 768
    D_pool                : attention pooling 输出维度，通常 512
    token_pool            : TokenAttentionPool，映射 [B,N,768] → [B,512]（CoCoOp ckpt）
    proj_img              : ProjectionHead，映射 [B,512] → [B,512]（CoCoOp ckpt）
    biomedclip_fc_norm    : BiomedCLIP trunk.fc_norm（LayerNorm 768），可能为 None
    biomedclip_visual_proj: BiomedCLIP 原生 visual_projection Linear(768→512)，
                            与 CLS token 投影路径完全一致

    mean_pool_512 说明
    ------------------
    使用 BiomedCLIP 的原生投影（fc_norm → visual_proj）而非 CoCoOp 的 token_pool.value：
    - visual_proj 权重来自 BiomedCLIP 预训练，与文本侧在同一 CLIP 嵌入空间
    - token_pool.value 是 attention pooling 机制的组成部分，在 CoCoOp 任务损失下
      联合优化，语义上不适合用于 mean pooling 的投影
    - 保证 mean_pool 方案作为纯 BiomedCLIP baseline 的公平性
    """

    def __init__(
        self,
        ckpt_path: str | Path,
        device: torch.device | str = "cpu",
        token_dim: int = 768,
        D_pool: int = 512,
    ):
        """
        Parameters
        ----------
        ckpt_path : CoCoOpTAUModel 训练好的 checkpoint 路径（.pt）
        device    : 推理设备
        token_dim : patch token 维度（默认 768，ViT-B/16）
        D_pool    : tokenpool 输出/proj 输入维度（默认 512）
        """
        super().__init__()

        self.token_dim = token_dim
        self.D_pool = D_pool

        # ── CoCoOp 权重容器（attn_pool / attn_pool_proj 用）────────────────
        self.token_pool = TokenAttentionPool(
            token_dim=token_dim,
            output_dim=D_pool,
            hidden_dim=D_pool,
            dropout=0.0,
        )
        self.proj_img = ProjectionHead(
            in_dim=D_pool,
            out_dim=D_pool,
            hidden_dim=D_pool,
            dropout=0.0,
        )

        # ── BiomedCLIP 原生投影（mean_pool_512 用）────────────────────────
        # fc_norm: trunk.fc_norm，LayerNorm(768)，与 CLS 处理路径一致
        # visual_proj: Linear(768, 512, bias=False)，BiomedCLIP 预训练权重
        self.biomedclip_fc_norm: Optional[nn.LayerNorm] = None
        self.biomedclip_visual_proj: nn.Linear = nn.Linear(token_dim, D_pool, bias=False)

        # ── 从 CoCoOp checkpoint 加载 attn_pool / proj_img ────────────────
        self._load_from_ckpt(ckpt_path, device)

        # ── 从 BiomedCLIP 加载原生 visual_projection ─────────────────────
        self._load_biomedclip_proj(device)

        # ── 固定推理，不更新梯度 ──────────────────────────────────────────
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

        self.to(device)

    # -------------------------------------------------------------------------
    def _load_from_ckpt(self, ckpt_path: str | Path, device) -> None:
        """从 CoCoOpTAUModel checkpoint 提取 token_pool / proj_img 权重"""
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint 不存在: {ckpt_path}")

        state = torch.load(str(ckpt_path), map_location="cpu")

        # checkpoint 可能是 {"model_state_dict": ..., ...} 或直接是 state_dict
        if "model_state_dict" in state:
            full_sd = state["model_state_dict"]
        elif "state_dict" in state:
            full_sd = state["state_dict"]
        else:
            full_sd = state

        # ── 提取 token_pool 权重 ──────────────────────────────────────────
        pool_sd = {
            k[len("token_pool."):]: v
            for k, v in full_sd.items()
            if k.startswith("token_pool.")
        }
        if pool_sd:
            self.token_pool.load_state_dict(pool_sd, strict=True)
            print(f"[EmbeddingExtractor] Loaded token_pool ({len(pool_sd)} tensors)")
        else:
            print("[EmbeddingExtractor] WARN: token_pool not found in ckpt, using random init")

        # ── 提取 proj_img 权重 ────────────────────────────────────────────
        proj_sd = {
            k[len("proj_img."):]: v
            for k, v in full_sd.items()
            if k.startswith("proj_img.")
        }
        if proj_sd:
            self.proj_img.load_state_dict(proj_sd, strict=True)
            print(f"[EmbeddingExtractor] Loaded proj_img ({len(proj_sd)} tensors)")
        else:
            print("[EmbeddingExtractor] WARN: proj_img not found in ckpt, using random init")

    # -------------------------------------------------------------------------
    def _load_biomedclip_proj(self, device) -> None:
        """
        从 BiomedCLIP 加载原生视觉投影层，用于 mean_pool_512。

        使用与 CLS token 完全相同的路径：
            fc_norm（trunk.fc_norm，若存在）→ visual_proj（head, 768→512）

        open_clip 版本（CLIP-MRI2PET 实际使用）中：
            visual.head   - timm TimmModel.head, Linear(768→512)
            trunk.fc_norm - timm trunk 的 LayerNorm(768)
        """
        try:
            clip_model, _ = load_biomedclip_model()
            visual_adapter = VisionTransformerAdapter(clip_model.visual)

            if visual_adapter.mode == "timm":
                # ── open_clip / timm 路径 ─────────────────────────────────
                # head: timm TimmModel.head，即 768→512 的 Linear
                head = visual_adapter.head
                # 提取最后一个 Linear 层（有些 head 是 Sequential）
                if isinstance(head, nn.Linear):
                    proj_weight = head.weight.data.clone()   # (512, 768)
                    proj_bias   = head.bias.data.clone() if head.bias is not None else None
                else:
                    # Sequential: 取最后一个 Linear
                    linear_layers = [m for m in head.modules() if isinstance(m, nn.Linear)]
                    if not linear_layers:
                        raise ValueError("visual.head 中未找到 Linear 层")
                    last_linear  = linear_layers[-1]
                    proj_weight  = last_linear.weight.data.clone()
                    proj_bias    = last_linear.bias.data.clone() if last_linear.bias is not None else None

                # 更新 biomedclip_visual_proj
                out_dim, in_dim = proj_weight.shape
                self.biomedclip_visual_proj = nn.Linear(in_dim, out_dim, bias=(proj_bias is not None))
                self.biomedclip_visual_proj.weight.data.copy_(proj_weight)
                if proj_bias is not None:
                    self.biomedclip_visual_proj.bias.data.copy_(proj_bias)

                # fc_norm: trunk.fc_norm (LayerNorm)
                fc_norm = getattr(visual_adapter.trunk, "fc_norm", None)
                if fc_norm is not None:
                    import copy
                    self.biomedclip_fc_norm = copy.deepcopy(fc_norm)
                    print("[EmbeddingExtractor] Loaded BiomedCLIP fc_norm (LayerNorm 768)")
                else:
                    self.biomedclip_fc_norm = None
                    print("[EmbeddingExtractor] INFO: BiomedCLIP trunk has no fc_norm, skip")

                print(
                    f"[EmbeddingExtractor] Loaded BiomedCLIP visual_proj "
                    f"(timm mode): Linear({in_dim}→{out_dim})"
                )

            elif visual_adapter.mode == "clip":
                # ── 标准 CLIP ViT 路径（proj 矩阵）───────────────────────
                proj = visual_adapter.proj   # (768, 512) weight matrix (not nn.Linear)
                if proj is not None:
                    # CLIP 的 proj 是直接的矩阵乘法参数，转成 nn.Linear
                    out_dim = proj.shape[1]
                    in_dim  = proj.shape[0]
                    self.biomedclip_visual_proj = nn.Linear(in_dim, out_dim, bias=False)
                    self.biomedclip_visual_proj.weight.data.copy_(proj.t())
                    print(
                        f"[EmbeddingExtractor] Loaded BiomedCLIP visual_proj "
                        f"(clip mode): Linear({in_dim}→{out_dim})"
                    )

            # 释放 clip_model 节省显存
            del clip_model, visual_adapter

        except Exception as e:
            raise RuntimeError(
                "[EmbeddingExtractor] ERROR: 无法加载 BiomedCLIP visual_proj，"
                "mean_pool_512 需要纯 BiomedCLIP 图像空间，不允许回退到 checkpoint 路径。"
                f" 原始错误: {e}"
            )

    # -------------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, patch_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        patch_tokens : (B, N, token_dim)  来自预缓存的 region_token

        Returns
        -------
        dict:
            "mean_pool"      : (B, token_dim=768) – raw mean over patches
            "mean_pool_512"  : (B, 512)           – mean_pool 经 BiomedCLIP 原生投影
                                                    fc_norm(mean) → visual_proj(mean)
                                                    与 CLS token 投影路径完全一致
            "attn_pool"      : (B, D_pool=512)    – attention-pooled（CoCoOp ckpt）
            "attn_pool_proj" : (B, D_pool=512)    – attn_pool + proj_img（CoCoOp ckpt）
        """
        B, N, D = patch_tokens.shape
        dtype = next(iter(self.token_pool.parameters())).dtype
        x = patch_tokens.to(dtype)

        # (1) mean_pool ──────────────────────────────────────────────────────
        mean_pool = x.mean(dim=1)                     # (B, 768)

        # mean_pool_512: BiomedCLIP 原生投影路径 fc_norm → visual_proj
        #   与 CLS token 路径完全一致，保证在 BiomedCLIP CLIP 嵌入空间中
        mp = mean_pool
        if self.biomedclip_fc_norm is not None:
            mp = self.biomedclip_fc_norm(mp)      # LayerNorm(768)
        mean_pool_512 = self.biomedclip_visual_proj(mp)  # Linear(768→512)

        # (2) attn_pool ──────────────────────────────────────────────────────
        attn_pool = self.token_pool(x)                 # (B, 512)

        # (3) attn_pool_proj ─────────────────────────────────────────────────
        attn_pool_proj = self.proj_img(attn_pool)      # (B, 512)

        return {
            "mean_pool":       mean_pool.float(),
            "mean_pool_512":   mean_pool_512.float(),
            "attn_pool":       attn_pool.float(),
            "attn_pool_proj":  attn_pool_proj.float(),
        }

    # -------------------------------------------------------------------------
    @staticmethod
    def load_cache_file(cache_path: str | Path) -> Dict[str, torch.Tensor]:
        """
        加载单个 .vision.pt 缓存文件。

        Returns
        -------
        {"cls_token": (D,), "region_token": (N, D)}
        """
        payload = torch.load(str(cache_path), map_location="cpu")
        cls_token = payload.get("cls_token")
        region_token = payload.get("region_token")
        if cls_token is None or region_token is None:
            raise ValueError(f"缓存文件格式错误（缺少字段）: {cache_path}")
        return {"cls_token": cls_token.float(), "region_token": region_token.float()}

    # -------------------------------------------------------------------------
    @torch.no_grad()
    def extract_from_dataloader(
        self,
        dataloader,
        device: torch.device | str | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        遍历 dataloader（batch 含 "patch_emb" 字段）提取全量 embedding。

        Parameters
        ----------
        dataloader : 产出含 "patch_emb"(B,N,D)、"label_idx"(B,)、
                     "plasma_vals"(B,K)、"plasma_mask"(B,K) 的 DataLoader

        Returns
        -------
        {
            "mean_pool":       (N_total, 768),
            "mean_pool_512":   (N_total, 512),
            "attn_pool":       (N_total, 512),
            "attn_pool_proj":  (N_total, 512),
            "labels":          (N_total,)  LongTensor
            "plasma_vals":     (N_total, K)
            "plasma_mask":     (N_total, K)
        }
        """
        _dev = device or next(iter(self.parameters())).device
        self.to(_dev)

        mean_pools, mean_pool_512s, attn_pools, attn_pool_projs = [], [], [], []
        labels_list, plasma_vals_list, plasma_mask_list = [], [], []

        for batch in dataloader:
            patch_emb = batch["patch_emb"].to(_dev)          # (B, N, 768)
            label_idx = batch["label_idx"]                    # (B,)
            plasma_v  = batch.get("plasma_vals",
                                   batch.get("plasma_values"))
            plasma_m  = batch.get("plasma_mask")

            embs = self.forward(patch_emb)

            mean_pools.append(embs["mean_pool"].cpu())
            mean_pool_512s.append(embs["mean_pool_512"].cpu())
            attn_pools.append(embs["attn_pool"].cpu())
            attn_pool_projs.append(embs["attn_pool_proj"].cpu())
            labels_list.append(label_idx.cpu())
            if plasma_v is not None:
                plasma_vals_list.append(plasma_v.cpu())
            if plasma_m is not None:
                plasma_mask_list.append(plasma_m.cpu())

        result: Dict[str, torch.Tensor] = {
            "mean_pool":      torch.cat(mean_pools, dim=0),
            "mean_pool_512":  torch.cat(mean_pool_512s, dim=0),
            "attn_pool":      torch.cat(attn_pools, dim=0),
            "attn_pool_proj": torch.cat(attn_pool_projs, dim=0),
            "labels":         torch.cat(labels_list, dim=0),
        }
        if plasma_vals_list:
            result["plasma_vals"] = torch.cat(plasma_vals_list, dim=0)
        if plasma_mask_list:
            result["plasma_mask"] = torch.cat(plasma_mask_list, dim=0)

        return result
