"""
adapter_v2/retrieval_eval.py
=============================
RetrievalEvaluator：Image→Text 检索 Recall@K 评估。

文本 embedding (text_emb) 定义（所有方法统一）：
  1. 用 BiomedCLIP TextEncoder 对 5 条固定 plasma 提示文本编码 → (5, 512)
  2. 用每个样本的实际 plasma 值计算 softmax 权重 → (5,)
  3. 加权汇聚 → (512,)  [不使用 CoCoOp Context_net，不含图像信息]
  4. 应用 checkpoint 中的 proj_plasma → (512,)  [确保与 proj_img 在同一 CLIP 空间]
  5. L2 归一化

GT（ground truth）定义：每张图像的 GT text 是**同一样本**的 plasma text embedding。
计算 Recall@K：对每个 image_emb，在全量 text_emb 中按 cosine similarity 排序，
检查 GT 是否出现在 top-K 中。

说明：
  - TextEncoder / proj_plasma 均冻结（eval + requires_grad=False）
  - 支持批量文本编码（避免重复计算）
  - 支持多 seed（对 bootstrap 置信区间可选）
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

# ── 将 CLIP-MRI2PET 加入 sys.path ──────────────────────────────────────────
CLIP_MRI2PET_ROOT = Path(__file__).resolve().parents[2] / "CLIP-MRI2PET"
for _p in [str(CLIP_MRI2PET_ROOT), str(CLIP_MRI2PET_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from clip_mri2pet.models.biomedclip_text import BiomedCLIPTextEncoder
from clip_mri2pet.models.biomedclip_image import load_biomedclip_model
from models import ProjectionHead           # adapter_v2/models.py

# tracer → (ctx_init_str, class_prompt_template) 映射（与 train.py 保持一致）
_TRACER_MAP = {
    "tau":  {
        "ctx_init": "a tau pet scan of",
        "class_prompt_template": "This is a TAU PET scan of a subject diagnosed with {label}.",
    },
    "av45": {
        "ctx_init": "an av45 pet scan of",
        "class_prompt_template": "This is an AV45 PET scan of a subject diagnosed with {label}.",
    },
    "fdg":  {
        "ctx_init": "an fdg pet scan of",
        "class_prompt_template": "This is an FDG PET scan of a subject diagnosed with {label}.",
    },
}


# ============================================================================
# PlasmaTextEncoder（固定，不训练）
# ============================================================================

class PlasmaTextEncoder(nn.Module):
    """
    用固定 plasma 提示文本 + proj_plasma（来自 ckpt）生成每个样本的 text_emb。

    流程：
      text_features = TextEncoder(plasma_prompts)    # (K, 512)，K=5，只计算一次
      For each sample i:
        weights_i = softmax(plasma_vals_i / T)       # (K,)
        text_emb_i = proj_plasma( sum_k(w_k * text_features[k]) )  # (512,)
        text_emb_i = L2_normalize(text_emb_i)
    """

    # 默认 plasma 提示文本（与 CoCoOpTAUModel 中保持一致）
    DEFAULT_PLASMA_PROMPTS: List[str] = [
        "plasma biomarker indicates amyloid-beta 42/40 ratio pathology",
        "plasma biomarker indicates phosphorylated tau 217 pathology",
        "plasma biomarker indicates combined pTau217 and amyloid pathology",
        "plasma biomarker indicates neurofilament light chain neurodegeneration",
        "plasma biomarker indicates glial fibrillary acidic protein astrogliosis",
    ]

    def __init__(
        self,
        ckpt_path: str | Path,
        biomedclip_path: Optional[str | Path] = None,
        plasma_prompts: Optional[List[str]] = None,
        plasma_temperature: float = 1.0,
        device: torch.device | str = "cpu",
        proj_dim: int = 512,
        tracer: str = "tau",
    ):
        """
        Parameters
        ----------
        ckpt_path         : CoCoOpTAUModel checkpoint（用于加载 proj_plasma）
        biomedclip_path   : BiomedCLIP 本地路径（可选，若环境变量已配置可 None）
        plasma_prompts    : 5 条 plasma 提示文本；None 使用默认
        plasma_temperature: plasma 权重 softmax 温度
        device            : 推理设备
        proj_dim          : 投影维度（默认 512）
        tracer            : 示踪剂类型 tau/av45/fdg，影响内部 ctx_init 字符串
        """
        super().__init__()

        self.plasma_prompts = plasma_prompts or self.DEFAULT_PLASMA_PROMPTS
        self.plasma_temperature = plasma_temperature
        self._dev = torch.device(device) if isinstance(device, str) else device
        _tcfg = _TRACER_MAP.get(tracer.lower(), _TRACER_MAP["tau"])

        # ── 加载 BiomedCLIP TextEncoder ──────────────────────────────────
        clip_model, _ = load_biomedclip_model()
        self.text_encoder = BiomedCLIPTextEncoder(clip_model)
        self.tokenizer = self.text_encoder.tokenizer if hasattr(
            self.text_encoder, "tokenizer"
        ) else None

        # tokenizer 在 BiomedCLIPPromptLearner 中，这里需要单独获取
        # 参考 models.py 中的做法
        try:
            from clip_mri2pet.models.biomedclip_text import BiomedCLIPPromptLearner
            _pl = BiomedCLIPPromptLearner(
                clip_model=clip_model,
                classnames=["CN"],
                num_groups=1,
                ctx_init=_tcfg["ctx_init"],
            )
            self.tokenizer = _pl.tokenizer
            self.context_length = _pl.context_length
            del _pl
        except Exception as e:
            print(f"[PlasmaTextEncoder] WARN: could not get tokenizer: {e}")
            self.tokenizer = None
            self.context_length = 77

        # 冻结 text_encoder
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

        # ── 构建 proj_plasma，从 ckpt 加载权重 ──────────────────────────
        text_width = self.text_encoder.output_dim  # 通常 512
        self.proj_plasma = ProjectionHead(
            in_dim=text_width,
            out_dim=proj_dim,
            hidden_dim=proj_dim,
            dropout=0.0,
        )
        self._load_proj_plasma(ckpt_path)

        # 冻结 proj_plasma
        for p in self.proj_plasma.parameters():
            p.requires_grad_(False)

        self.eval()
        self.to(self._dev)

        # ── 预计算固定 plasma 文本特征 ────────────────────────────────────
        self._plasma_text_features: Optional[torch.Tensor] = None  # (K, text_width)
        self._precompute_text_features()

    # -------------------------------------------------------------------------
    def _load_proj_plasma(self, ckpt_path: str | Path) -> None:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint 不存在: {ckpt_path}")

        state = torch.load(str(ckpt_path), map_location="cpu")
        if "model_state_dict" in state:
            full_sd = state["model_state_dict"]
        elif "state_dict" in state:
            full_sd = state["state_dict"]
        else:
            full_sd = state

        proj_sd = {
            k[len("proj_plasma."):]: v
            for k, v in full_sd.items()
            if k.startswith("proj_plasma.")
        }
        if proj_sd:
            self.proj_plasma.load_state_dict(proj_sd, strict=True)
            print(f"[PlasmaTextEncoder] Loaded proj_plasma ({len(proj_sd)} tensors)")
        else:
            raise RuntimeError(
                "[PlasmaTextEncoder] ERROR: checkpoint 中未找到 proj_plasma 参数。"
                "plasma text retrieval 必须依赖 checkpoint 学到的投影参数。"
            )

    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _precompute_text_features(self) -> None:
        """一次性编码 K 条 plasma 提示文本 → (K, text_width)"""
        if self.tokenizer is None:
            print("[PlasmaTextEncoder] WARN: no tokenizer, skipping precompute")
            return

        encoded = self.tokenizer(
            self.plasma_prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.context_length,
        )
        input_ids = encoded["input_ids"].to(self._dev)
        seq_len = input_ids.shape[1]

        # TextEncoder 接受 (B, seq_len, ctx_dim) 或 (B, seq_len) token_ids
        # 查看 BiomedCLIPTextEncoder 的接口
        # 参考 models.py: text_encoder(prompts, token_ids)
        # 这里直接用 token ids 调用
        embedding = self.text_encoder.embedding
        text_emb = embedding(input_ids)  # (K, seq_len, ctx_dim)

        # text_encoder.forward 需要 (prompts, token_ids)
        # prompts shape: (B_outer, num_texts, seq_len, ctx_dim)
        # 但我们这里把 K 条文本当做 batch
        # 用 _encode_flat 接口更简单：直接把 (K, seq_len, ctx_dim) 传入
        feats = self._encode_text_flat(text_emb, input_ids)  # (K, text_width)
        self._plasma_text_features = feats.cpu()
        print(f"[PlasmaTextEncoder] Precomputed plasma text features: {feats.shape}")

    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _encode_text_flat(
        self,
        text_emb: torch.Tensor,   # (K, seq_len, ctx_dim)
        token_ids: torch.Tensor,  # (K, seq_len)
    ) -> torch.Tensor:
        """
        调用 BiomedCLIPTextEncoder，将字符串 embedding 和 token_ids 传入。
        参照 models.py 中的调用方式，这里以 B=1 批次处理。
        """
        # BiomedCLIPTextEncoder.forward(prompts, token_ids)
        #   prompts:   (B, num_texts, seq_len, ctx_dim)
        #   token_ids: (B, num_texts, seq_len)
        # 我们令 B=1, num_texts=K
        K = text_emb.shape[0]
        prompts  = text_emb.unsqueeze(0)     # (1, K, seq_len, ctx_dim)
        tok_ids  = token_ids.unsqueeze(0)    # (1, K, seq_len)

        # 返回 (1, K, text_width)
        feats = self.text_encoder(prompts, tok_ids)  # (1, K, text_width)
        return feats.squeeze(0)  # (K, text_width)

    # -------------------------------------------------------------------------
    @torch.no_grad()
    def encode_samples(
        self,
        plasma_vals: torch.Tensor,   # (N, K)
        plasma_mask: torch.Tensor,   # (N, K)  bool
    ) -> torch.Tensor:
        """
        为每个样本生成 text_emb（L2 归一化后）。

        Parameters
        ----------
        plasma_vals : (N, K) 归一化后的 plasma 值（来自 dataset）
        plasma_mask : (N, K) 有效 mask（True=有效）

        Returns
        -------
        text_embs : (N, 512)，L2 归一化
        """
        if self._plasma_text_features is None:
            raise RuntimeError("plasma text features 未预计算，检查 tokenizer 是否可用")

        N = plasma_vals.shape[0]
        device = self._dev

        # 固定 plasma 文本特征（K, text_width）
        pf = self._plasma_text_features.to(device)  # (K, text_width)

        # ── 计算每样本 plasma 权重 ────────────────────────────────────────
        vals = plasma_vals.float().to(device)   # (N, K)
        mask = plasma_mask.bool().to(device)    # (N, K)

        T = max(self.plasma_temperature, 1e-6)
        logits = vals / T
        logits = logits.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(logits, dim=-1)  # (N, K)

        # 全缺失时用均匀权重
        all_missing = ~mask.any(dim=-1, keepdim=True)
        uniform     = torch.full_like(weights, 1.0 / weights.shape[1])
        weights     = torch.where(all_missing.expand_as(weights), uniform, weights)
        weights     = torch.nan_to_num(weights, nan=1.0 / weights.shape[1])

        # ── 加权汇聚 ─────────────────────────────────────────────────────
        # plasma_summary: (N, text_width)
        plasma_summary = torch.einsum("nk,kd->nd", weights, pf)

        # ── proj_plasma → (N, 512) ───────────────────────────────────────
        dtype = next(iter(self.proj_plasma.parameters())).dtype
        text_embs = self.proj_plasma(plasma_summary.to(dtype)).float()

        # ── L2 归一化 ────────────────────────────────────────────────────
        text_embs = F.normalize(text_embs, dim=-1)

        return text_embs


# ============================================================================
# ClassTextEncoder（固定，不训练）
# ============================================================================

class ClassTextEncoder(nn.Module):
    """
    用固定诊断类别文本 + proj_class（来自 ckpt）生成 3 条固定的 class_text_emb。

    流程：
      text_features = TextEncoder(class_prompts)    # (3, 512)，只计算一次
      class_embs    = proj_class(text_features)     # (3, 512)
      class_embs    = L2_normalize(class_embs)
    注：不使用 CoCoOp Context_net，仅依赖固定文本编码。
    """

    # 与 models.py CoCoOpTAUModel 中保持一致
    DEFAULT_CLASS_NAMES: List[str] = ["CN", "MCI", "AD"]
    DEFAULT_CLASS_PROMPT_TEMPLATE: str = (
        "This is a TAU PET scan of a subject diagnosed with {label}."
    )

    def __init__(
        self,
        ckpt_path: str | Path,
        class_names: Optional[List[str]] = None,
        class_prompt_template: Optional[str] = None,
        device: torch.device | str = "cpu",
        proj_dim: int = 512,
        use_ckpt_proj_class: bool = True,
        tracer: str = "tau",
    ):
        """
        Parameters
        ----------
        ckpt_path              : CoCoOpTAUModel checkpoint（用于加载 proj_class）
        class_names            : 诊断类别名，默认 ["CN", "MCI", "AD"]
        class_prompt_template  : 文本模板，需含 {label} 占位符；None 则从 tracer 自动派生
        device                 : 推理设备
        proj_dim               : 投影维度（默认 512）
        use_ckpt_proj_class    : 是否使用 checkpoint 的 proj_class。
                     True=checkpoint空间；False=纯BiomedCLIP空间。
        tracer                 : 示踪剂类型 tau/av45/fdg，影响默认 prompt 文本和 ctx_init
        """
        super().__init__()

        _tcfg = _TRACER_MAP.get(tracer.lower(), _TRACER_MAP["tau"])
        self.class_names = class_names or self.DEFAULT_CLASS_NAMES
        self.class_prompt_template = (
            class_prompt_template or _tcfg["class_prompt_template"]
        )
        self.class_prompts: List[str] = [
            self.class_prompt_template.format(label=name)
            for name in self.class_names
        ]
        self._dev = torch.device(device) if isinstance(device, str) else device
        self.use_ckpt_proj_class = use_ckpt_proj_class

        # ── 加载 BiomedCLIP TextEncoder（与 PlasmaTextEncoder 相同方式）──
        clip_model, _ = load_biomedclip_model()
        self.text_encoder = BiomedCLIPTextEncoder(clip_model)

        # 取 tokenizer（复用 PlasmaTextEncoder 的逻辑）
        try:
            from clip_mri2pet.models.biomedclip_text import BiomedCLIPPromptLearner
            _pl = BiomedCLIPPromptLearner(
                clip_model=clip_model,
                classnames=["CN"],
                num_groups=1,
                ctx_init=_tcfg["ctx_init"],
            )
            self.tokenizer = _pl.tokenizer
            self.context_length = _pl.context_length
            del _pl
        except Exception as e:
            print(f"[ClassTextEncoder] WARN: could not get tokenizer: {e}")
            self.tokenizer = None
            self.context_length = 77

        # 冻结 text_encoder
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

        # ── 可选构建 proj_class（checkpoint 空间）────────────────────────
        text_width = self.text_encoder.output_dim  # 通常 512
        self.proj_class: Optional[ProjectionHead] = None
        if self.use_ckpt_proj_class:
            self.proj_class = ProjectionHead(
                in_dim=text_width,
                out_dim=proj_dim,
                hidden_dim=proj_dim,
                dropout=0.0,
            )
            self._load_proj_class(ckpt_path)

            # 冻结 proj_class
            for p in self.proj_class.parameters():
                p.requires_grad_(False)

        self.eval()
        self.to(self._dev)

        # ── 预计算固定 class text features ───────────────────────────────
        self._class_text_embs: Optional[torch.Tensor] = None  # (3, proj_dim)
        self._precompute_class_embs()

    # -------------------------------------------------------------------------
    def _load_proj_class(self, ckpt_path: str | Path) -> None:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint 不存在: {ckpt_path}")

        state = torch.load(str(ckpt_path), map_location="cpu")
        if "model_state_dict" in state:
            full_sd = state["model_state_dict"]
        elif "state_dict" in state:
            full_sd = state["state_dict"]
        else:
            full_sd = state

        # 搜索包含 "proj" 且 "class" 的键前缀（兼容不同命名）
        candidate_prefixes = ["proj_class."]
        # 回退：遍历所有键，找包含 'proj' + 'class' 的模块
        for key in full_sd:
            parts = key.split(".")
            if len(parts) >= 2:
                prefix = parts[0] + "."
                if "proj" in prefix.lower() and "class" in prefix.lower():
                    if prefix not in candidate_prefixes:
                        candidate_prefixes.append(prefix)

        proj_sd = {}
        used_prefix = None
        for prefix in candidate_prefixes:
            candidate = {
                k[len(prefix):]: v
                for k, v in full_sd.items()
                if k.startswith(prefix)
            }
            if candidate:
                proj_sd = candidate
                used_prefix = prefix
                break

        if proj_sd:
            self.proj_class.load_state_dict(proj_sd, strict=True)
            print(
                f"[ClassTextEncoder] Loaded proj_class from prefix '{used_prefix}' "
                f"({len(proj_sd)} tensors)"
            )
        else:
            print(
                "[ClassTextEncoder] WARN: proj_class not found in ckpt "
                "(keys scanned: "
                + str([k for k in full_sd.keys() if "proj" in k.lower()][:10])
                + "), using random init"
            )

    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _encode_text_flat(
        self,
        text_emb: torch.Tensor,   # (K, seq_len, ctx_dim)
        token_ids: torch.Tensor,  # (K, seq_len)
    ) -> torch.Tensor:
        """调用 BiomedCLIPTextEncoder，复用 PlasmaTextEncoder 中的逻辑。"""
        prompts = text_emb.unsqueeze(0)    # (1, K, seq_len, ctx_dim)
        tok_ids = token_ids.unsqueeze(0)   # (1, K, seq_len)
        feats   = self.text_encoder(prompts, tok_ids)  # (1, K, text_width)
        return feats.squeeze(0)  # (K, text_width)

    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _precompute_class_embs(self) -> None:
        """预计算 3 条诊断类别文本的 embedding → (3, proj_dim)，已 L2 归一化。"""
        if self.tokenizer is None:
            print("[ClassTextEncoder] WARN: no tokenizer, skipping precompute")
            return

        encoded = self.tokenizer(
            self.class_prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.context_length,
        )
        input_ids = encoded["input_ids"].to(self._dev)

        embedding = self.text_encoder.embedding
        text_emb = embedding(input_ids)  # (3, seq_len, ctx_dim)

        feats = self._encode_text_flat(text_emb, input_ids)  # (3, text_width)

        if self.use_ckpt_proj_class:
            if self.proj_class is None:
                raise RuntimeError("use_ckpt_proj_class=True 但 proj_class 未初始化")
            # proj_class → (3, proj_dim)
            dtype = next(iter(self.proj_class.parameters())).dtype
            embs = self.proj_class(feats.to(dtype)).float()
            space_name = "ckpt-proj_class"
        else:
            # 纯 BiomedCLIP 文本空间（不使用 checkpoint 投影）
            embs = feats.float()
            space_name = "biomedclip-raw"

        # L2 归一化
        embs = F.normalize(embs, dim=-1)
        self._class_text_embs = embs.cpu()

        print(
            f"[ClassTextEncoder] Precomputed class text embs: {embs.shape}  "
            f"labels={self.class_names}  space={space_name}"
        )

    # -------------------------------------------------------------------------
    @property
    def class_text_embs(self) -> torch.Tensor:
        """返回 (3, proj_dim) 已归一化的 class text embeddings。"""
        if self._class_text_embs is None:
            raise RuntimeError(
                "class text embs 未预计算，检查 tokenizer 是否可用"
            )
        return self._class_text_embs


# ============================================================================
# DiagnosisTextRetrievalEvaluator
# ============================================================================

class DiagnosisTextRetrievalEvaluator:
    """
    Zero-shot 诊断文本检索评估（候选 text 数量固定为 3：CN / MCI / AD）。

    评估方式：
      - 将 image_emb (N,512) 与 class_text_embs (3,512) 做 cosine similarity
      - top-1 / top-2 命中作为 Recall@1 / Recall@2（K=3 时 Recall@3=1，不报告）
      - 输出 micro / macro Recall@1 / Recall@2 以及可选 confusion matrix

    Parameters
    ----------
    ckpt_path              : CoCoOpTAUModel checkpoint（加载 proj_class）
    class_names            : 类别名，默认 ["CN", "MCI", "AD"]
    class_prompt_template  : 文本模板，默认与训练一致
    device                 : 推理设备
    """

    def __init__(
        self,
        ckpt_path: str | Path,
        class_names: Optional[List[str]] = None,
        class_prompt_template: Optional[str] = None,
        device: torch.device | str = "cpu",
        use_ckpt_proj_class: bool = True,
        tracer: str = "tau",
    ):
        self._dev         = torch.device(device) if isinstance(device, str) else device
        self.class_names  = class_names or ClassTextEncoder.DEFAULT_CLASS_NAMES
        self.use_ckpt_proj_class = use_ckpt_proj_class
        self.class_encoder = ClassTextEncoder(
            ckpt_path=ckpt_path,
            class_names=class_names,
            class_prompt_template=class_prompt_template,
            device=device,
            use_ckpt_proj_class=use_ckpt_proj_class,
            tracer=tracer,
        )

    # -------------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(
        self,
        image_embs: torch.Tensor,      # (N, 512)  需为 512 维，否则报错
        y_true:     torch.Tensor,      # (N,)      整数标签，与 class_names 下标对应
        emb_key:    str = "unknown",
        print_confusion_matrix: bool = True,
    ) -> Dict[str, float]:
        """
        Parameters
        ----------
        image_embs : (N, 512) image embedding（会 L2 normalize）
        y_true     : (N,)  整数标签 0=CN, 1=MCI, 2=AD（与 class_names 顺序一致）
        emb_key    : 用于日志标识
        print_confusion_matrix : 是否打印混淆矩阵

        Returns
        -------
        dict 包含:
          diag_recall1_micro, diag_recall2_micro,
          diag_recall1_macro, diag_recall2_macro
        """
        N = image_embs.shape[0]
        d_img = image_embs.shape[-1]
        num_cls = len(self.class_names)

        if d_img != 512:
            raise ValueError(
                f"image_emb 维度为 {d_img}，diagnosis retrieval 要求 512 维。"
                "请使用 mean_pool_512 / attn_pool / attn_pool_proj，"
                "或先通过 BiomedCLIP visual projection 将 768 → 512。"
            )

        # 移到 CPU 进行计算，避免 OOM
        img_n   = F.normalize(image_embs.float().cpu(), dim=-1)   # (N, 512)
        text_n  = self.class_encoder.class_text_embs.cpu()        # (3, 512)
        # text_n 已 L2 归一化

        # sim: (N, 3)
        sim = img_n @ text_n.t()

        labels = y_true.long().cpu()  # (N,)

        # ------------------------------------------------------------------
        # top-1 / top-2 预测
        # ------------------------------------------------------------------
        # topk 返回的 indices: (N, k)
        top2_indices = sim.topk(min(2, num_cls), dim=-1).indices  # (N, 2)
        top1_indices = top2_indices[:, 0]                          # (N,)

        hit1 = (top1_indices == labels).float()                    # (N,)
        hit2 = (top2_indices == labels.unsqueeze(1)).any(dim=-1).float()  # (N,)

        # ------------------------------------------------------------------
        # Micro metrics（直接平均全样本）
        # ------------------------------------------------------------------
        recall1_micro = hit1.mean().item()
        recall2_micro = hit2.mean().item()

        # ------------------------------------------------------------------
        # Macro metrics（先按类算 recall 再平均，忽略缺失类）
        # ------------------------------------------------------------------
        per_class_recall1: List[float] = []
        per_class_recall2: List[float] = []
        per_class_counts: List[int]    = []

        for c in range(num_cls):
            mask = labels == c
            n_c  = mask.sum().item()
            per_class_counts.append(n_c)
            if n_c == 0:
                per_class_recall1.append(float("nan"))
                per_class_recall2.append(float("nan"))
            else:
                per_class_recall1.append(hit1[mask].mean().item())
                per_class_recall2.append(hit2[mask].mean().item())

        valid1 = [x for x in per_class_recall1 if not (x != x)]  # 过滤 nan
        valid2 = [x for x in per_class_recall2 if not (x != x)]
        recall1_macro = float(sum(valid1) / len(valid1)) if valid1 else 0.0
        recall2_macro = float(sum(valid2) / len(valid2)) if valid2 else 0.0

        # ------------------------------------------------------------------
        # 打印结果
        # ------------------------------------------------------------------
        print(
            f"  [{emb_key}] diag-text Recall@1 "
            f"micro={recall1_micro:.4f}  macro={recall1_macro:.4f}"
        )
        print(
            f"  [{emb_key}] diag-text Recall@2 "
            f"micro={recall2_micro:.4f}  macro={recall2_macro:.4f}  (N={N})"
        )
        for c, cname in enumerate(self.class_names):
            r1 = per_class_recall1[c]
            r2 = per_class_recall2[c]
            n_c = per_class_counts[c]
            r1_str = f"{r1:.4f}" if r1 == r1 else "N/A"
            r2_str = f"{r2:.4f}" if r2 == r2 else "N/A"
            print(f"    {cname}: n={n_c:4d}  recall@1={r1_str}  recall@2={r2_str}")

        # ------------------------------------------------------------------
        # Confusion matrix（行=真实类，列=预测类）
        # ------------------------------------------------------------------
        if print_confusion_matrix:
            conf = torch.zeros(num_cls, num_cls, dtype=torch.long)
            for i in range(N):
                gt  = labels[i].item()
                pr  = top1_indices[i].item()
                if 0 <= gt < num_cls and 0 <= pr < num_cls:
                    conf[gt][pr] += 1

            col_w    = 10
            hdr_cols = "".join(f"{'pred_' + n:>{col_w}}" for n in self.class_names)
            print(f"\n  [{emb_key}] Confusion matrix (rows=true, cols=pred):")
            print(f"  {'':12s}" + hdr_cols)
            for r, rname in enumerate(self.class_names):
                row_str = "".join(f"{conf[r][c].item():>{col_w}d}" for c in range(num_cls))
                print(f"  {'true_' + rname:<12s}" + row_str)
            print()

        metrics: Dict[str, float] = {
            "diag_recall1_micro": recall1_micro,
            "diag_recall2_micro": recall2_micro,
            "diag_recall1_macro": recall1_macro,
            "diag_recall2_macro": recall2_macro,
        }

        # ── AUC 计算（One-vs-Rest）────────────────────────────────────
        auc_metrics = self.compute_auc(sim, labels, emb_key=emb_key)
        metrics.update(auc_metrics)

        return metrics


    # -------------------------------------------------------------------------
    def compute_auc(
        self,
        sim: torch.Tensor,
        y_true: torch.Tensor,
        emb_key: str = "unknown",
    ) -> Dict[str, float]:
        """
        计算 One-vs-Rest AUC（macro / micro / per-class）。

        Parameters
        ----------
        sim    : (N, num_cls) 连续 logits（cosine similarity），不做 argmax
        y_true : (N,) 整数标签 {0, 1, 2}
        emb_key: 用于日志标识

        Returns
        -------
        dict 包含:
          diag_auc_macro, diag_auc_micro,
          diag_auc_CN, diag_auc_MCI, diag_auc_AD
        """
        sim_np    = sim.detach().cpu().numpy()       # (N, num_cls)
        y_true_np = y_true.detach().cpu().numpy()    # (N,)
        num_cls   = len(self.class_names)

        # ── per-class AUC (One-vs-Rest) ──────────────────────────────────
        auc_per_class: List[float] = []
        for c in range(num_cls):
            scores_c = sim_np[:, c]
            labels_c = (y_true_np == c).astype(int)
            try:
                auc_c = roc_auc_score(labels_c, scores_c)
            except ValueError:
                warnings.warn(
                    f"[diag-text] WARN: class '{self.class_names[c]}' 在验证集中"
                    f"缺少正样本或负样本，AUC 设为 NaN"
                )
                auc_c = float("nan")
            auc_per_class.append(auc_c)

        # ── macro AUC ────────────────────────────────────────────────────
        valid_aucs = [a for a in auc_per_class if a == a]  # 过滤 nan
        macro_auc  = float(np.mean(valid_aucs)) if valid_aucs else float("nan")

        # ── micro AUC（OvR + one-hot，直接使用 sim logits）─────────────────
        try:
            y_true_oh = label_binarize(y_true_np, classes=np.arange(num_cls))  # (N, C)
            micro_auc = roc_auc_score(
                y_true_oh,
                sim_np,
                average="micro",
            )
        except ValueError:
            warnings.warn(
                "[diag-text] WARN: micro AUC 计算失败（可能某类缺少样本），设为 NaN"
            )
            micro_auc = float("nan")

        # ── 打印 ─────────────────────────────────────────────────────────
        print(f"  [{emb_key}] [diag-text] AUC per class:")
        for c, cname in enumerate(self.class_names):
            auc_str = f"{auc_per_class[c]:.4f}" if auc_per_class[c] == auc_per_class[c] else "N/A"
            print(f"    {cname}: {auc_str}")
        print(f"  [{emb_key}] [diag-text] macro AUC = {macro_auc:.4f}")
        print(f"  [{emb_key}] [diag-text] micro AUC = {micro_auc:.4f}")

        # ── 组装返回值 ───────────────────────────────────────────────────
        result: Dict[str, float] = {
            "diag_auc_macro": macro_auc,
            "diag_auc_micro": micro_auc,
        }
        for c, cname in enumerate(self.class_names):
            result[f"diag_auc_{cname}"] = auc_per_class[c]

        return result


# ============================================================================
# RetrievalEvaluator
# ============================================================================

class RetrievalEvaluator:
    """
    评估 Image→Text Recall@K。

    所有三种 image_emb 方案均使用**同一套** text_embs，
    由 PlasmaTextEncoder 一次性预计算后缓存，避免重复编码。

    Parameters
    ----------
    ckpt_path         : CoCoOpTAUModel checkpoint（用于加载 proj_plasma）
    biomedclip_path   : BiomedCLIP 路径（可选）
    plasma_prompts    : 可选自定义 plasma 提示
    plasma_temperature: plasma 权重温度
    device            : 推理设备
    k_list            : Recall@K 中的 K 值列表
    """

    def __init__(
        self,
        ckpt_path: str | Path,
        biomedclip_path: Optional[str | Path] = None,
        plasma_prompts: Optional[List[str]] = None,
        plasma_temperature: float = 1.0,
        device: torch.device | str = "cpu",
        k_list: List[int] = None,
    ):
        self.k_list = k_list or [1, 5, 10]
        self._dev = torch.device(device) if isinstance(device, str) else device

        self.text_encoder_module = PlasmaTextEncoder(
            ckpt_path=ckpt_path,
            biomedclip_path=biomedclip_path,
            plasma_prompts=plasma_prompts,
            plasma_temperature=plasma_temperature,
            device=device,
        )

        # 缓存的 text_embs（避免重复计算）
        self._cached_text_embs: Optional[torch.Tensor] = None

    # -------------------------------------------------------------------------
    @torch.no_grad()
    def precompute_text_embeddings(
        self,
        plasma_vals: torch.Tensor,   # (N, K)
        plasma_mask: torch.Tensor,   # (N, K)
    ) -> torch.Tensor:
        """
        预计算并缓存所有样本的 text_embs。

        Parameters
        ----------
        plasma_vals, plasma_mask : 全量验证集的 plasma 数据

        Returns
        -------
        text_embs : (N, 512)
        """
        text_embs = self.text_encoder_module.encode_samples(plasma_vals, plasma_mask)
        self._cached_text_embs = text_embs.cpu()
        print(f"[RetrievalEvaluator] text_embs computed: {text_embs.shape}")
        return self._cached_text_embs

    # -------------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(
        self,
        image_embs:  torch.Tensor,              # (N, D_img)  已 L2 归一化
        text_embs:   Optional[torch.Tensor] = None,  # (N, 512)  若 None 用缓存
        emb_key:     str = "unknown",
    ) -> Dict[str, float]:
        """
        计算 Image→Text Recall@K。

        GT 定义：第 i 个 image 的 GT 是第 i 个 text（同一样本）。
        image_embs 和 text_embs 均做 L2 归一化后计算 cosine similarity。

        Parameters
        ----------
        image_embs : (N, D_img)  可以是任意维度（会自动 L2 normalize）
        text_embs  : (N, 512)   若 None 则使用 precompute_text_embeddings 的缓存
        emb_key    : 用于日志

        Returns
        -------
        dict: {"recall@1": float, "recall@5": float, "recall@10": float}
        """
        if text_embs is None:
            if self._cached_text_embs is None:
                raise RuntimeError("需先调用 precompute_text_embeddings() 或传入 text_embs")
            text_embs = self._cached_text_embs

        N = image_embs.shape[0]
        assert text_embs.shape[0] == N, (
            f"image_embs 和 text_embs 数量不一致: {N} vs {text_embs.shape[0]}"
        )

        # 维度对齐检查
        d_img  = image_embs.shape[-1]
        d_text = text_embs.shape[-1]
        if d_img != d_text:
            raise ValueError(
                f"image_emb dim ({d_img}) ≠ text_emb dim ({d_text})。"
                f"请确保使用 'mean_pool_512' 而非 'mean_pool'（768-dim）进行检索，"
                f"或为 mean_pool 提供合适的投影。"
            )

        # L2 归一化
        img_n  = F.normalize(image_embs.float(),  dim=-1)   # (N, D)
        text_n = F.normalize(text_embs.float(),   dim=-1)   # (N, D)

        # Cosine similarity matrix: (N, N)
        sim_matrix = img_n @ text_n.t()   # (N, N)

        # GT: 对角线（第 i 行的 GT = index i）
        gt_indices = torch.arange(N)

        metrics: Dict[str, float] = {}
        for k in self.k_list:
            k_eff = min(k, N)
            # 每行取 top-k 索引
            top_k = sim_matrix.topk(k_eff, dim=-1).indices   # (N, k)
            # 检查 GT 是否在 top-k 中
            hit = (top_k == gt_indices.unsqueeze(1)).any(dim=-1).float()  # (N,)
            recall_k = hit.mean().item()
            metrics[f"recall@{k}"] = recall_k

        print(
            f"  [{emb_key}] Retrieval Recall@1={metrics.get('recall@1', 0):.4f} "
            f"@5={metrics.get('recall@5', 0):.4f} "
            f"@10={metrics.get('recall@10', 0):.4f}  (N={N})"
        )
        return metrics

    # -------------------------------------------------------------------------
    @staticmethod
    def compute_recall_at_k(
        queries: torch.Tensor,    # (N, D) 已归一化
        gallery: torch.Tensor,    # (M, D) 已归一化
        gt_indices: torch.Tensor, # (N,)   每个 query 对应的 GT gallery index
        k_list: List[int] = None,
    ) -> Dict[str, float]:
        """
        通用 Recall@K 计算（queries 和 gallery 可不同大小）。
        """
        k_list = k_list or [1, 5, 10]
        N = queries.shape[0]

        sim = F.normalize(queries.float(), dim=-1) @ F.normalize(gallery.float(), dim=-1).t()
        # sim: (N, M)

        metrics = {}
        for k in k_list:
            k_eff = min(k, gallery.shape[0])
            top_k = sim.topk(k_eff, dim=-1).indices  # (N, k)
            hit = (top_k == gt_indices.unsqueeze(1)).any(dim=-1).float()
            metrics[f"recall@{k}"] = hit.mean().item()

        return metrics
