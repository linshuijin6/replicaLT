"""
adapter_v2/models.py
====================
TAU 单示踪剂 CoCoOp 模型实现

核心架构：
1. ImageBranch: patch tokens -> pooling -> ContextNet -> context tokens
2. ClassBranch: class prompts + context -> TextEncoder -> Class Embedding
3. PlasmaBranch: plasma prompts + context -> TextEncoder -> 加权汇聚 Plasma Embedding
4. Projection Heads: 三路 embedding 对齐到 512 维 CLIP 空间

复用 CLIP-MRI2PET 组件：
- BiomedCLIPTextEncoder / BiomedCLIPPromptLearner
- ContextNet
- TextPromptEncoder
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
from clip_mri2pet.models.prompt_ccp_vit import ContextNet


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
        # tokens 形状: (B, N, D)
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)  # (1, N, D)
        
        # Attention 计算
        # attn_hidden: (B, N, hidden)
        attn_hidden = torch.tanh(self.query(tokens))
        # attn_logits: (B, N)
        attn_logits = self.score(attn_hidden).squeeze(-1)
        # attn_weights: (B, N) - softmax 归一化
        attn_weights = torch.softmax(attn_logits, dim=-1)
        
        # 值投影
        # values: (B, N, D_out)
        values = self.value(tokens)
        
        # 加权聚合
        # pooled: (B, D_out)
        pooled = torch.einsum("bn,bnd->bd", attn_weights, values)
        
        return self.dropout(pooled)


# ============================================================================
# Projection Head
# ============================================================================

class ProjectionHead(nn.Module):
    """
    投影头：将特征映射到统一的 CLIP 空间（512 维）
    
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
        """
        Args:
            x: (..., in_dim)
        Returns:
            (..., out_dim)
        """
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
    # 对无效位置设置 -inf
    logits = plasma_values / max(temperature, 1e-6)
    # mask: True 表示有效，False 表示缺失
    logits = logits.masked_fill(~plasma_mask, float("-inf"))
    
    # Softmax
    weights = F.softmax(logits, dim=-1)
    
    # 处理全缺失情况（softmax 后全为 nan）
    # 若某样本全缺失，使用均匀权重
    all_missing = ~plasma_mask.any(dim=-1, keepdim=True)  # (B, 1)
    uniform = torch.ones_like(weights) / 5.0
    weights = torch.where(all_missing.expand_as(weights), uniform, weights)
    
    # 确保无 nan
    weights = torch.nan_to_num(weights, nan=0.2)
    
    return weights


# ============================================================================
# CoCoOp 主模型
# ============================================================================

class CoCoOpTAUModel(nn.Module):
    """
    TAU 单示踪剂 CoCoOp 模型
    
    前向流程：
    1. Image Branch:
       - tau_tokens: (B, N, Dv) -> TokenAttentionPool -> g: (B, D_pool)
       - g -> ContextNet -> T_ctx: (B, ctx_len, ctx_dim)
       - g -> ProjectionHead -> img_emb: (B, 512)
       
    2. Class Branch:
       - class_prompts + T_ctx -> TextEncoder -> class_features: (B, 3, D_text)
       - 取对应 diagnosis 的特征 -> ProjectionHead -> class_emb: (B, 512)
       
    3. Plasma Branch:
       - plasma_prompts + T_ctx -> TextEncoder -> plasma_features: (B, 5, D_text)
       - plasma_weights 加权汇聚 -> ProjectionHead -> plasma_emb: (B, 512)
    
    输出：
    - img_emb, class_emb, plasma_emb 各 (B, 512)
    - plasma_weights: (B, 5)
    """
    
    def __init__(
        self,
        biomedclip_path: str | Path,
        class_names: List[str] = None,
        class_prompt_template: str = "This is a TAU PET scan of a subject diagnosed with {label}.",
        plasma_prompts: List[str] = None,
        ctx_len: int = 4,
        proj_dim: int = 512,
        ctx_hidden_dim: int = 1024,
        share_ctx_base: bool = False,
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
            ctx_hidden_dim: ContextNet 隐藏层维度
            share_ctx_base: 是否共享 class/plasma 的 base context
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
        # text_width: text encoder 的 embedding 维度（ctx_dim）
        # proj_dim: 最终对齐维度（512）
        self.text_width = self.text_encoder.output_dim  # 通常 512
        self.ctx_dim = self.text_encoder.embedding.embedding_dim  # 通常 768
        
        # =====================================================================
        # Prompt Learner（用于 tokenize 和获取 embedding）
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
        # Class prompts
        self.class_prompt_template = class_prompt_template
        self.class_prompts = [
            class_prompt_template.format(label=name)
            for name in self.class_names
        ]
        # 文本: ["This is a TAU PET scan of a subject diagnosed with CN.", ...]
        
        # Plasma prompts
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
        # base_ctx_class: (ctx_len, ctx_dim) - 类别分支的基础 context
        # base_ctx_plasma: (ctx_len, ctx_dim) - plasma 分支的基础 context
        dtype = self.prompt_learner.token_prefix.dtype
        self.base_ctx_class = nn.Parameter(
            torch.empty(ctx_len, self.ctx_dim, dtype=dtype)
        )
        nn.init.normal_(self.base_ctx_class, std=0.02)
        
        if share_ctx_base:
            # 共享 context base
            self.base_ctx_plasma = self.base_ctx_class
        else:
            self.base_ctx_plasma = nn.Parameter(
                torch.empty(ctx_len, self.ctx_dim, dtype=dtype)
            )
            nn.init.normal_(self.base_ctx_plasma, std=0.02)
        
        # =====================================================================
        # Image Branch: Pooling + ContextNet
        # =====================================================================
        # 假设 token_dim = 768（BiomedCLIP ViT）
        # 如果缓存的 region_token 维度不同，需要调整
        self.token_dim = 768  # 可从实际缓存推断
        
        # Token Attention Pool: (B, N, token_dim) -> (B, D_pool)
        self.D_pool = proj_dim  # 对齐到 512
        self.token_pool = TokenAttentionPool(
            token_dim=self.token_dim,
            output_dim=self.D_pool,
            hidden_dim=self.D_pool,
            dropout=0.1,
        )
        
        # ContextNet: (B, D_pool) -> (B, ctx_len * ctx_dim)
        self.context_net = ContextNet(
            input_dim=self.D_pool,
            output_dim=ctx_len * self.ctx_dim,
            hidden_dim=ctx_hidden_dim,
        )
        
        # =====================================================================
        # Projection Heads
        # =====================================================================
        # Image projection: D_pool -> proj_dim
        self.proj_img = ProjectionHead(
            in_dim=self.D_pool,
            out_dim=proj_dim,
            hidden_dim=proj_dim,
            dropout=0.1,
        )
        
        # Class projection: text_width -> proj_dim
        self.proj_class = ProjectionHead(
            in_dim=self.text_width,
            out_dim=proj_dim,
            hidden_dim=proj_dim,
            dropout=0.1,
        )
        
        # Plasma projection: text_width -> proj_dim
        self.proj_plasma = ProjectionHead(
            in_dim=self.text_width,
            out_dim=proj_dim,
            hidden_dim=proj_dim,
            dropout=0.1,
        )
        
        # =====================================================================
        # Logit Scale（可学习）
        # =====================================================================
        # 初始化为 ln(1/0.07) ≈ 2.659
        self.logit_scale = nn.Parameter(torch.tensor(2.6593))
    
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
            prompts: (B, num_texts, seq_len, ctx_dim) - 完整 prompt embedding
            token_ids: (B, num_texts, seq_len) - token ids
        """
        B = context.shape[0]
        num_texts = len(texts)
        
        # Tokenize
        token_ids = self._tokenize(texts, device)  # (num_texts, seq_len)
        seq_len = token_ids.shape[1]
        
        # 获取 embedding
        embedding = self.text_encoder.embedding
        dtype = context.dtype
        text_emb = embedding(token_ids).to(dtype)  # (num_texts, seq_len, ctx_dim)
        
        # 扩展到 batch
        # text_emb: (num_texts, seq_len, ctx_dim) -> (B, num_texts, seq_len, ctx_dim)
        text_emb = text_emb.unsqueeze(0).expand(B, -1, -1, -1)
        
        # 分离 prefix (第一个 token) 和 suffix (context 之后的 tokens)
        # prefix: (B, num_texts, 1, ctx_dim)
        prefix = text_emb[:, :, :1, :]
        
        # context: (B, ctx_len, ctx_dim) -> (B, 1, ctx_len, ctx_dim) -> (B, num_texts, ctx_len, ctx_dim)
        ctx = context.unsqueeze(1).expand(-1, num_texts, -1, -1)
        
        # suffix 从 1+ctx_len 开始
        suffix_start = min(1 + self.ctx_len, seq_len)
        suffix = text_emb[:, :, suffix_start:, :]
        
        # 拼接: prefix + context + suffix
        prompts = torch.cat([prefix, ctx, suffix], dim=2)  # (B, num_texts, new_seq_len, ctx_dim)
        
        # token_ids 扩展到 batch
        token_ids = token_ids.unsqueeze(0).expand(B, -1, -1)  # (B, num_texts, seq_len)
        
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
                class_emb: (B, 512) - 类别 embedding，L2 归一化
                plasma_emb: (B, 512) - plasma embedding，L2 归一化
                plasma_weights: (B, 5) - plasma 权重
                logit_scale: () - 可学习的 logit scale
        """
        B = tau_tokens.shape[0]
        device = tau_tokens.device
        dtype = self.base_ctx_class.dtype
        
        # =====================================================================
        # 1. Image Branch
        # =====================================================================
        # tau_tokens: (B, N, Dv) 例如 (B, 196, 768) 或 (B, num_rois, 768)
        
        # Token pooling
        # g: (B, D_pool) 例如 (B, 512)
        g = self.token_pool(tau_tokens.to(dtype))
        
        # ContextNet: 生成条件 context
        # T_ctx_flat: (B, ctx_len * ctx_dim)
        T_ctx_flat = self.context_net(g)
        # T_ctx: (B, ctx_len, ctx_dim) 例如 (B, 4, 768)
        T_ctx = T_ctx_flat.view(B, self.ctx_len, self.ctx_dim)
        
        # Image embedding
        # img_emb: (B, 512)
        img_emb = self.proj_img(g)
        img_emb = F.normalize(img_emb, dim=-1)
        
        # =====================================================================
        # 2. Class Branch
        # =====================================================================
        # 组合 context: base_ctx_class + T_ctx
        # ctx_class: (B, ctx_len, ctx_dim)
        ctx_class = self.base_ctx_class.unsqueeze(0) + T_ctx
        
        # 构建 prompt embedding
        # class_prompts: (B, 3, seq_len, ctx_dim)
        # class_token_ids: (B, 3, seq_len)
        class_prompts, class_token_ids = self._build_prompts_with_context(
            self.class_prompts, ctx_class, device
        )
        
        # Text encoder
        # class_features: (B, 3, text_width) 例如 (B, 3, 512)
        class_features = self.text_encoder(class_prompts, class_token_ids)
        
        # 根据 diagnosis_id 选择对应的类别特征
        # indices: (B,)
        batch_indices = torch.arange(B, device=device)
        # 处理无效 diagnosis_id（-1 变为 0 以避免索引错误）
        valid_diag_id = diagnosis_id.clamp(min=0)
        # class_feat_pos: (B, text_width)
        class_feat_pos = class_features[batch_indices, valid_diag_id]
        
        # Project
        # class_emb: (B, 512)
        class_emb = self.proj_class(class_feat_pos)
        class_emb = F.normalize(class_emb, dim=-1)
        
        # =====================================================================
        # 3. Plasma Branch
        # =====================================================================
        # 组合 context: base_ctx_plasma + T_ctx
        # ctx_plasma: (B, ctx_len, ctx_dim)
        ctx_plasma = self.base_ctx_plasma.unsqueeze(0) + T_ctx
        
        # 构建 prompt embedding
        # plasma_prompts_emb: (B, 5, seq_len, ctx_dim)
        # plasma_token_ids: (B, 5, seq_len)
        plasma_prompts_emb, plasma_token_ids = self._build_prompts_with_context(
            self.plasma_prompts, ctx_plasma, device
        )
        
        # Text encoder
        # plasma_features: (B, 5, text_width)
        plasma_features = self.text_encoder(plasma_prompts_emb, plasma_token_ids)
        
        # 计算 plasma 权重
        # plasma_weights: (B, 5)
        plasma_weights = compute_plasma_weights(
            plasma_values, plasma_mask, self.plasma_temperature
        )
        
        # 加权汇聚
        # plasma_summary: (B, text_width)
        plasma_summary = torch.einsum("bk,bkd->bd", plasma_weights, plasma_features)
        
        # Project
        # plasma_emb: (B, 512)
        plasma_emb = self.proj_plasma(plasma_summary)
        plasma_emb = F.normalize(plasma_emb, dim=-1)
        
        # =====================================================================
        # 返回
        # =====================================================================
        return {
            "img_emb": img_emb,           # (B, 512) - L2 归一化
            "class_emb": class_emb,       # (B, 512) - L2 归一化
            "plasma_emb": plasma_emb,     # (B, 512) - L2 归一化
            "plasma_weights": plasma_weights,  # (B, 5)
            "logit_scale": self.logit_scale.exp().clamp(max=100.0),  # ()
            # 额外输出用于调试
            "class_features_all": class_features,  # (B, 3, text_width)
            "plasma_features_all": plasma_features,  # (B, 5, text_width)
        }
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """返回所有可训练参数"""
        params = []
        # Context base tokens
        params.append(self.base_ctx_class)
        if not (self.base_ctx_plasma is self.base_ctx_class):
            params.append(self.base_ctx_plasma)
        # ContextNet
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
