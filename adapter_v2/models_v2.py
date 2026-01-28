"""
adapter_v2/models_v2.py
=======================
TAU 单示踪剂 CoCoOp 模型 V2 实现

核心改进（修复 image-conditioned shortcut 问题）：
1. Context Net 仅用于 Class 分支，采用低秩 bottleneck（rank ≤ 8）
2. Plasma 分支完全 image-agnostic，不使用任何 image-conditioned context
3. 增强的 Plasma 投影层（2 层）
4. 支持生成固定的 class prototype（验证时不含 image-conditioned context）
5. 添加冗余约束输出（T_ctx 与 g 的相似度）

架构：
1. ImageBranch: patch tokens -> pooling -> img_emb
2. Context Net (低秩): g -> T_ctx（仅用于 Class 分支）
3. ClassBranch: class prompts + base_ctx + T_ctx -> class_emb
4. PlasmaBranch: plasma prompts + base_ctx_plasma (无 T_ctx) -> plasma_emb
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# 添加 CLIP-MRI2PET 到 path
CLIP_MRI2PET_ROOT = Path(__file__).resolve().parents[2] / "CLIP-MRI2PET"
if str(CLIP_MRI2PET_ROOT) not in sys.path:
    sys.path.insert(0, str(CLIP_MRI2PET_ROOT))
if str(CLIP_MRI2PET_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(CLIP_MRI2PET_ROOT / "src"))

# 复用 CLIP-MRI2PET 组件
from clip_mri2pet.models.biomedclip_text import (
    BiomedCLIPTextEncoder,
    BiomedCLIPPromptLearner,
)
from clip_mri2pet.models.biomedclip_image import load_biomedclip_model


# ============================================================================
# Token Pooling
# ============================================================================

class TokenAttentionPool(nn.Module):
    """
    可学习的 attention pooling，将多个 patch tokens 聚合为单一向量
    
    Input: (B, N, D) - B=batch, N=num_tokens, D=token_dim
    Output: (B, D_out)
    """
    
    def __init__(
        self,
        token_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden = hidden_dim or token_dim
        # 注意力计算
        self.query = nn.Linear(token_dim, hidden, bias=True)
        self.score = nn.Linear(hidden, 1, bias=False)
        # 值投影
        self.value = nn.Linear(token_dim, output_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, N, D) - patch tokens
            
        Returns:
            pooled: (B, D_out) - 聚合后的表征
        """
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
        
        # Attention 计算
        attn_hidden = torch.tanh(self.query(tokens))
        attn_logits = self.score(attn_hidden).squeeze(-1)
        attn_weights = torch.softmax(attn_logits, dim=-1)
        
        # 值投影
        values = self.value(tokens)
        
        # 加权聚合
        pooled = torch.einsum("bn,bnd->bd", attn_weights, values)
        
        return self.dropout(pooled)


# ============================================================================
# 低秩 Context Net (Bottleneck 约束)
# ============================================================================

class LowRankContextNet(nn.Module):
    """
    低秩 Context Net：生成 image-conditioned context
    
    使用 bottleneck 结构限制表达能力（等效 rank ≤ bottleneck_dim）
    
    结构: Linear(input_dim -> bottleneck_dim) -> ReLU -> Linear(bottleneck_dim -> output_dim)
    
    bottleneck_dim 默认为 8，确保 Context Net 只能提供有限的 instance-conditioned adjustment
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bottleneck_dim: int = 8,
    ):
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        
        self.down = nn.Linear(input_dim, bottleneck_dim, bias=True)
        self.up = nn.Linear(bottleneck_dim, output_dim, bias=False)
        
        # 初始化：small scale
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.down.bias)
        nn.init.normal_(self.up.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) - image global feature g
            
        Returns:
            ctx: (B, output_dim) - context tokens (flattened)
        """
        h = F.relu(self.down(x))
        return self.up(h)


# ============================================================================
# Projection Head
# ============================================================================

class ProjectionHead(nn.Module):
    """
    投影头：将特征映射到统一的 CLIP 空间
    
    结构：Linear -> GELU -> Dropout -> Linear
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 512,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden = hidden_dim or max(in_dim, out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim, bias=False),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EnhancedProjectionHead(nn.Module):
    """
    增强版投影头（2 层 MLP）：用于 Plasma 分支
    
    结构：Linear -> GELU -> Dropout -> Linear -> GELU -> Dropout -> Linear
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 512,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden = hidden_dim or max(in_dim, out_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim, bias=False),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# Plasma 权重计算
# ============================================================================

def compute_plasma_weights(
    plasma_values: torch.Tensor,
    plasma_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    计算 plasma 加权权重：softmax(plasma_z / T) with mask
    
    Args:
        plasma_values: (B, 5) - z-score 归一化的 plasma 值
        plasma_mask: (B, 5) - 有效 mask
        temperature: softmax 温度
        
    Returns:
        weights: (B, 5) - 归一化权重，缺失位置为 0
    """
    logits = plasma_values / max(temperature, 1e-6)
    logits = logits.masked_fill(~plasma_mask, float("-inf"))
    
    weights = F.softmax(logits, dim=-1)
    
    all_missing = ~plasma_mask.any(dim=-1, keepdim=True)
    uniform = torch.ones_like(weights) / 5.0
    weights = torch.where(all_missing.expand_as(weights), uniform, weights)
    weights = torch.nan_to_num(weights, nan=0.2)
    
    return weights


# ============================================================================
# CoCoOp V2 主模型
# ============================================================================

class CoCoOpTAUModelV2(nn.Module):
    """
    TAU 单示踪剂 CoCoOp 模型 V2
    
    核心改进：
    1. Context Net 仅用于 Class 分支，采用低秩约束
    2. Plasma 分支完全 image-agnostic
    3. 支持生成 class prototype（验证时）
    
    前向流程：
    1. Image Branch:
       - tau_tokens: (B, N, Dv) -> TokenAttentionPool -> g: (B, D_pool)
       - g -> ProjectionHead -> img_emb: (B, 512)
       - g -> LowRankContextNet -> T_ctx: (B, ctx_len, ctx_dim) [仅用于 Class]
       
    2. Class Branch:
       - class_prompts + base_ctx_class + T_ctx -> TextEncoder -> class_emb
       
    3. Plasma Branch (image-agnostic):
       - plasma_prompts + base_ctx_plasma -> TextEncoder -> plasma_emb
       - 不使用任何 T_ctx！
    """
    
    def __init__(
        self,
        biomedclip_path: str | Path,
        class_names: List[str] = None,
        class_prompt_template: str = "This is a TAU PET scan of a subject diagnosed with {label}.",
        plasma_prompts: List[str] = None,
        ctx_len: int = 4,
        proj_dim: int = 512,
        ctx_bottleneck_dim: int = 8,
        plasma_temperature: float = 1.0,
    ):
        """
        Args:
            biomedclip_path: BiomedCLIP 本地路径
            class_names: 诊断类别名，如 ["CN", "MCI", "AD"]
            class_prompt_template: 类别提示模板
            plasma_prompts: 5 条 plasma 语义提示
            ctx_len: CoCoOp context 长度
            proj_dim: projection 输出维度
            ctx_bottleneck_dim: Context Net 的 bottleneck 维度，控制表达能力
            plasma_temperature: plasma 权重温度
        """
        super().__init__()
        
        self.class_names = class_names or ["CN", "MCI", "AD"]
        self.plasma_temperature = plasma_temperature
        self.ctx_len = ctx_len
        
        # =====================================================================
        # 加载 BiomedCLIP
        # =====================================================================
        clip_model, _ = load_biomedclip_model()
        
        # 冻结 text encoder
        self.text_encoder = BiomedCLIPTextEncoder(clip_model)
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        
        # 获取维度信息
        self.text_width = self.text_encoder.output_dim  # 512
        self.ctx_dim = self.text_encoder.embedding.embedding_dim  # 768
        
        # =====================================================================
        # Prompt Learner
        # =====================================================================
        self.prompt_learner = BiomedCLIPPromptLearner(
            clip_model=clip_model,
            classnames=self.class_names,
            num_groups=ctx_len,
            ctx_init="a tau pet scan of",
        )
        self.tokenizer = self.prompt_learner.tokenizer
        self.context_length = self.prompt_learner.context_length
        
        # =====================================================================
        # 文本模板
        # =====================================================================
        self.class_prompt_template = class_prompt_template
        self.class_prompts = [
            class_prompt_template.format(label=name)
            for name in self.class_names
        ]
        
        self.plasma_prompts = plasma_prompts or [
            "plasma biomarker indicates amyloid-beta 42/40 ratio pathology",
            "plasma biomarker indicates phosphorylated tau 217 pathology",
            "plasma biomarker indicates combined pTau217 and amyloid pathology",
            "plasma biomarker indicates neurofilament light chain neurodegeneration",
            "plasma biomarker indicates glial fibrillary acidic protein astrogliosis",
        ]
        
        # =====================================================================
        # 可学习 Context Base Tokens
        # =====================================================================
        dtype = self.prompt_learner.token_prefix.dtype
        
        # Class 分支的 base context
        self.base_ctx_class = nn.Parameter(
            torch.empty(ctx_len, self.ctx_dim, dtype=dtype)
        )
        nn.init.normal_(self.base_ctx_class, std=0.02)
        
        # Plasma 分支的 base context（独立，不受 image 影响）
        self.base_ctx_plasma = nn.Parameter(
            torch.empty(ctx_len, self.ctx_dim, dtype=dtype)
        )
        nn.init.normal_(self.base_ctx_plasma, std=0.02)
        
        # =====================================================================
        # Image Branch: Pooling + Projection
        # =====================================================================
        self.token_dim = 768  # BiomedCLIP ViT token dim
        self.D_pool = proj_dim  # 512
        
        self.token_pool = TokenAttentionPool(
            token_dim=self.token_dim,
            output_dim=self.D_pool,
            hidden_dim=self.D_pool,
            dropout=0.1,
        )
        
        # Image projection
        self.proj_img = ProjectionHead(
            in_dim=self.D_pool,
            out_dim=proj_dim,
            hidden_dim=proj_dim,
            dropout=0.1,
        )
        
        # =====================================================================
        # 低秩 Context Net（仅用于 Class 分支）
        # =====================================================================
        self.context_net = LowRankContextNet(
            input_dim=self.D_pool,
            output_dim=ctx_len * self.ctx_dim,
            bottleneck_dim=ctx_bottleneck_dim,
        )
        
        # =====================================================================
        # Projection Heads
        # =====================================================================
        # Class projection: 简单单层
        self.proj_class = ProjectionHead(
            in_dim=self.text_width,
            out_dim=proj_dim,
            hidden_dim=proj_dim,
            dropout=0.1,
        )
        
        # Plasma projection: 增强版（2 层 MLP）
        self.proj_plasma = EnhancedProjectionHead(
            in_dim=self.text_width,
            out_dim=proj_dim,
            hidden_dim=proj_dim,
            dropout=0.1,
        )
        
        # =====================================================================
        # Logit Scale
        # =====================================================================
        self.logit_scale = nn.Parameter(torch.tensor(2.6593))
        
        # =====================================================================
        # T_ctx 投影层（用于计算冗余约束）
        # =====================================================================
        # 将 T_ctx (ctx_dim=768) 投影到 D_pool (512) 以计算与 g 的相似度
        self.ctx_to_g_proj = nn.Linear(self.ctx_dim, self.D_pool, bias=False)
        nn.init.orthogonal_(self.ctx_to_g_proj.weight)  # 正交初始化，保持距离关系
        self.ctx_to_g_proj.requires_grad_(False)  # 冻结，不参与梯度更新        
        # =====================================================================
        # 缓存 Plasma Prototype Embedding（image-agnostic）
        # =====================================================================
        self._cached_plasma_proto = None  # 将在第一次 forward 后缓存
    
    def _tokenize(self, texts: List[str], device: torch.device) -> torch.LongTensor:
        """Tokenize 文本列表"""
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.context_length,
        )
        return encoded["input_ids"].to(device)
    
    def _build_prompts_with_context(
        self,
        texts: List[str],
        context: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将 context 插入到文本 embedding 中
        
        Args:
            texts: 文本列表
            context: (B, ctx_len, ctx_dim) - 条件 context
            device: 设备
            
        Returns:
            prompts: (B, num_texts, seq_len, ctx_dim)
            token_ids: (B, num_texts, seq_len)
        """
        B = context.shape[0]
        num_texts = len(texts)
        
        token_ids = self._tokenize(texts, device)
        seq_len = token_ids.shape[1]
        
        embedding = self.text_encoder.embedding
        dtype = context.dtype
        text_emb = embedding(token_ids).to(dtype)
        
        text_emb = text_emb.unsqueeze(0).expand(B, -1, -1, -1)
        
        prefix = text_emb[:, :, :1, :]
        ctx = context.unsqueeze(1).expand(-1, num_texts, -1, -1)
        
        suffix_start = min(1 + self.ctx_len, seq_len)
        suffix = text_emb[:, :, suffix_start:, :]
        
        prompts = torch.cat([prefix, ctx, suffix], dim=2)
        token_ids = token_ids.unsqueeze(0).expand(B, -1, -1)
        
        return prompts, token_ids
    
    def _build_prompts_no_context(
        self,
        texts: List[str],
        base_ctx: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建不含 image-conditioned context 的 prompt（仅使用 base context）
        
        用于：
        1. Plasma 分支（完全 image-agnostic）
        2. Class prototype 生成（验证时）
        
        Args:
            texts: 文本列表
            base_ctx: (ctx_len, ctx_dim) - 基础 learnable context
            batch_size: batch 大小
            device: 设备
            
        Returns:
            prompts: (B, num_texts, seq_len, ctx_dim)
            token_ids: (B, num_texts, seq_len)
        """
        num_texts = len(texts)
        
        token_ids = self._tokenize(texts, device)
        seq_len = token_ids.shape[1]
        
        embedding = self.text_encoder.embedding
        dtype = base_ctx.dtype
        text_emb = embedding(token_ids).to(dtype)
        
        # text_emb: (num_texts, seq_len, ctx_dim)
        # 扩展到 batch
        text_emb = text_emb.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        prefix = text_emb[:, :, :1, :]
        
        # base_ctx: (ctx_len, ctx_dim) -> (1, 1, ctx_len, ctx_dim) -> (B, num_texts, ctx_len, ctx_dim)
        ctx = base_ctx.unsqueeze(0).unsqueeze(0).expand(batch_size, num_texts, -1, -1)
        
        suffix_start = min(1 + self.ctx_len, seq_len)
        suffix = text_emb[:, :, suffix_start:, :]
        
        prompts = torch.cat([prefix, ctx, suffix], dim=2)
        token_ids = token_ids.unsqueeze(0).expand(batch_size, -1, -1)
        
        return prompts, token_ids
    
    def forward(
        self,
        tau_tokens: torch.Tensor,
        diagnosis_id: torch.Tensor,
        plasma_values: torch.Tensor,
        plasma_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            tau_tokens: (B, N, Dv) - patch tokens
            diagnosis_id: (B,) - 诊断类别 ID [0, 1, 2]
            plasma_values: (B, 5) - z-score 归一化的 plasma 值
            plasma_mask: (B, 5) - 有效 mask
            
        Returns:
            Dict:
                img_emb: (B, 512) - 图像 embedding，L2 归一化
                class_emb: (B, 512) - 类别 embedding（含 image-conditioned context）
                plasma_emb: (B, 512) - plasma embedding（image-agnostic）
                plasma_weights: (B, 5) - plasma 权重
                logit_scale: () - 可学习的 logit scale
                ctx_img_sim: (B,) - T_ctx 与 g 的余弦相似度（用于冗余约束）
        """
        B = tau_tokens.shape[0]
        device = tau_tokens.device
        dtype = self.base_ctx_class.dtype
        
        # =====================================================================
        # 1. Image Branch
        # =====================================================================
        # Token pooling -> g
        g = self.token_pool(tau_tokens.to(dtype))  # (B, D_pool)
        
        # Image embedding
        img_emb = self.proj_img(g)  # (B, 512)
        img_emb = F.normalize(img_emb, dim=-1)
        
        # 低秩 Context Net -> T_ctx（仅用于 Class）
        T_ctx_flat = self.context_net(g)  # (B, ctx_len * ctx_dim)
        T_ctx = T_ctx_flat.view(B, self.ctx_len, self.ctx_dim)
        
        # 计算 T_ctx 与 g 的相似度（用于冗余约束）
        # 将 T_ctx 均值 pooling 后投影到 g 的维度
        T_ctx_mean = T_ctx.mean(dim=1)  # (B, ctx_dim=768)
        T_ctx_proj = self.ctx_to_g_proj(T_ctx_mean)  # (B, D_pool=512)
        # 计算余弦相似度
        ctx_img_sim = (F.normalize(T_ctx_proj, dim=-1) * F.normalize(g, dim=-1)).sum(dim=-1)  # (B,)
        
        # =====================================================================
        # 2. Class Branch（含 image-conditioned context）
        # =====================================================================
        # ctx_class = base_ctx_class + T_ctx
        ctx_class = self.base_ctx_class.unsqueeze(0) + T_ctx
        
        class_prompts, class_token_ids = self._build_prompts_with_context(
            self.class_prompts, ctx_class, device
        )
        
        class_features = self.text_encoder(class_prompts, class_token_ids)  # (B, 3, text_width)
        
        # 选择对应诊断的特征
        batch_indices = torch.arange(B, device=device)
        valid_diag_id = diagnosis_id.clamp(min=0)
        class_feat_pos = class_features[batch_indices, valid_diag_id]  # (B, text_width)
        
        class_emb = self.proj_class(class_feat_pos)  # (B, 512)
        class_emb = F.normalize(class_emb, dim=-1)
        
        # =====================================================================
        # 3. Plasma Branch（完全 image-agnostic）
        # =====================================================================
        # 使用 base_ctx_plasma，不添加任何 T_ctx
        plasma_prompts_emb, plasma_token_ids = self._build_prompts_no_context(
            self.plasma_prompts, self.base_ctx_plasma, B, device
        )
        
        plasma_features = self.text_encoder(plasma_prompts_emb, plasma_token_ids)  # (B, 5, text_width)
        
        # 计算 plasma 权重
        plasma_weights = compute_plasma_weights(
            plasma_values, plasma_mask, self.plasma_temperature
        )
        
        # 加权汇聚
        plasma_summary = torch.einsum("bk,bkd->bd", plasma_weights, plasma_features)  # (B, text_width)
        
        plasma_emb = self.proj_plasma(plasma_summary)  # (B, 512)
        plasma_emb = F.normalize(plasma_emb, dim=-1)
        
        # =====================================================================
        # 返回
        # =====================================================================
        return {
            "img_emb": img_emb,
            "class_emb": class_emb,
            "plasma_emb": plasma_emb,
            "plasma_weights": plasma_weights,
            "logit_scale": self.logit_scale.exp().clamp(max=100.0),
            "ctx_img_sim": ctx_img_sim,  # 用于冗余约束损失
            "class_features_all": class_features,
            "plasma_features_all": plasma_features,
            # 额外输出用于调试
            "g": g,  # (B, D_pool)
            "T_ctx": T_ctx,  # (B, ctx_len, ctx_dim)
        }
    
    @torch.no_grad()
    def get_class_prototypes(self, device: torch.device) -> torch.Tensor:
        """
        获取 Class Prototypes（不含 image-conditioned context）
        
        用于验证阶段：每类一个固定 embedding
        
        Returns:
            prototypes: (num_classes, 512) - L2 归一化
        """
        dtype = self.base_ctx_class.dtype
        num_classes = len(self.class_names)
        
        # 使用 base_ctx_class 构建 prompt（无 T_ctx）
        prompts, token_ids = self._build_prompts_no_context(
            self.class_prompts, self.base_ctx_class, batch_size=1, device=device
        )
        # prompts: (1, num_classes, seq_len, ctx_dim)
        # token_ids: (1, num_classes, seq_len)
        
        class_features = self.text_encoder(prompts, token_ids)  # (1, num_classes, text_width)
        class_features = class_features.squeeze(0)  # (num_classes, text_width)
        
        prototypes = self.proj_class(class_features)  # (num_classes, 512)
        prototypes = F.normalize(prototypes, dim=-1)
        
        return prototypes
    
    @torch.no_grad()
    def get_plasma_prototypes(self, device: torch.device) -> torch.Tensor:
        """
        获取 Plasma Prototypes（5 个 biomarker 各一个）
        
        由于 Plasma 分支是 image-agnostic，这些 prototype 是固定的
        
        Returns:
            prototypes: (5, 512) - L2 归一化
        """
        dtype = self.base_ctx_plasma.dtype
        
        # 使用 base_ctx_plasma 构建 prompt
        prompts, token_ids = self._build_prompts_no_context(
            self.plasma_prompts, self.base_ctx_plasma, batch_size=1, device=device
        )
        # prompts: (1, 5, seq_len, ctx_dim)
        
        plasma_features = self.text_encoder(prompts, token_ids)  # (1, 5, text_width)
        plasma_features = plasma_features.squeeze(0)  # (5, text_width)
        
        prototypes = self.proj_plasma(plasma_features)  # (5, 512)
        prototypes = F.normalize(prototypes, dim=-1)
        
        return prototypes
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """返回所有可训练参数"""
        params = []
        # Context base tokens
        params.append(self.base_ctx_class)
        params.append(self.base_ctx_plasma)
        # Context Net（低秩）
        params.extend(self.context_net.parameters())
        # Token pool
        params.extend(self.token_pool.parameters())
        # Projection heads
        params.extend(self.proj_img.parameters())
        params.extend(self.proj_class.parameters())
        params.extend(self.proj_plasma.parameters())
        # Logit scale
        params.append(self.logit_scale)
        return params
