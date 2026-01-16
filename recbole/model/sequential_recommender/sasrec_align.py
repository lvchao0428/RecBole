# -*- coding: utf-8 -*-
# Align-only variant of SASRec with optional text embedding alignment

import os
import json
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder, MLPLayers
from recbole.model.loss import BPRLoss


class DCNV2Cross(nn.Module):
    """DCN-V2 cross network (non-mix) over dense features.

    Follows the original implementation in recbole.model.context_aware_recommender.dcnv2
    with the update rule: x_{l+1} = x_l + x_0 ⊙ (W_l x_l + b_l).
    """

    def __init__(self, input_dim: int, num_layers: int = 3):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_layers = int(max(0, num_layers))
        # W: (in_feature_num, in_feature_num) per layer
        self.cross_layer_w = nn.ParameterList(
            nn.Parameter(torch.randn(self.input_dim, self.input_dim))
            for _ in range(self.num_layers)
        )
        # b: (in_feature_num, 1) per layer
        self.bias = nn.ParameterList(
            nn.Parameter(torch.zeros(self.input_dim, 1))
            for _ in range(self.num_layers)
        )

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        if self.num_layers == 0:
            return x0
        # x0: [batch, in_feature_num]
        x0_u = x0.unsqueeze(dim=2)  # [B, D, 1]
        xl = x0_u
        for i in range(self.num_layers):
            xl_w = torch.matmul(self.cross_layer_w[i], xl)  # [B, D, 1]
            xl_w = xl_w + self.bias[i]
            xl_dot = torch.mul(x0_u, xl_w)
            xl = xl_dot + xl
        xl = xl.squeeze(dim=2)  # [B, D]
        return xl


class SASRecAlign(SequentialRecommender):
    def __init__(self, config, dataset):
        super(SASRecAlign, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        # additional regularization / scoring configs
        self.label_smoothing = float(config["label_smoothing"]) if "label_smoothing" in config else 0.0
        self.cosine_score = bool(config["cosine_score"]) if "cosine_score" in config else False
        self.cosine_scale = float(config["cosine_scale"]) if "cosine_scale" in config else 10.0
        self.token_dropout_prob = float(config["token_dropout_prob"]) if "token_dropout_prob" in config else 0.0

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            # label smoothing supported in torch>=1.10
            try:
                self.loss_fct = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
            except TypeError:
                # fallback if torch version doesn't support label_smoothing
                self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # --- text-alignment settings & feature fusion ---
        self.alignment_weight = config["alignment_weight"] if "alignment_weight" in config else 0.0
        self.temperature = config["temperature"] if "temperature" in config else 0.07
        self.normalize_text = config["normalize_text"] if "normalize_text" in config else True
        self.detach_text_emb = config["detach_text_emb"] if "detach_text_emb" in config else True
        self.use_llm = config["use_llm"] if "use_llm" in config else False
        self.use_cross = config["use_cross"] if "use_cross" in config else False
        self.use_seq_text_cross = bool(config["use_seq_text_cross"]) if "use_seq_text_cross" in config else False
        self.seq_text_cross_dropout_rate = (
            float(config["seq_text_cross_dropout"]) if "seq_text_cross_dropout" in config else 0.1
        )
        self.use_text_view_cross = bool(config["use_text_view_cross"]) if "use_text_view_cross" in config else False
        self.text_view_residual_weight = (
            float(config["text_view_residual_weight"]) if "text_view_residual_weight" in config else 1.0
        )
        self.use_align = config["use_align"] if "use_align" in config else True
        self.text_cross_layer_num = config["text_cross_layer_num"] if "text_cross_layer_num" in config else 3
        # Cross-output dropout and learnable text gate configs
        self.cross_dropout_prob = float(config["cross_dropout_prob"]) if "cross_dropout_prob" in config else 0.0
        self.text_gate_init = float(config["text_gate_init"]) if "text_gate_init" in config else 0.5
        self.text_gate_reg_l2 = float(config["text_gate_reg_l2"]) if "text_gate_reg_l2" in config else 0.0
        self.text_gate_reg_entropy = float(config["text_gate_reg_entropy"]) if "text_gate_reg_entropy" in config else 0.0
        # learnable global gate alpha in [0,1] via sigmoid
        self.text_gate_param = nn.Parameter(torch.tensor(self.text_gate_init, dtype=torch.float32))
        # Explicit switch to fully disable text features and mimic pure SASRec
        self.disable_text_feature = bool(config["disable_text_feature"]) if "disable_text_feature" in config else False
        # Training utilities
        self.freeze_backbone = bool(config["freeze_backbone"]) if "freeze_backbone" in config else False
        # Exclude Top-K nearest neighbors in InfoNCE negatives to mitigate false negatives
        self.align_exclude_topk = int(config["align_exclude_topk"]) if "align_exclude_topk" in config else 0
        
        # [Cold-Start Alignment Boost] 冷启动对齐权重增强
        # 问题: 对齐损失被高频商品主导，冷启动商品的对齐信号弱
        # 解决: 给低频商品更高的对齐损失权重
        # 权重公式: weight = 1.0 + boost * max(0, threshold - popularity) / threshold
        # - cold_start_align_boost: 增强系数，0.0表示关闭，建议2.0-5.0
        # - cold_start_align_threshold: 低于此popularity的商品获得额外权重
        self.cold_start_align_boost = float(config["cold_start_align_boost"]) if "cold_start_align_boost" in config else 0.0
        self.cold_start_align_threshold = int(config["cold_start_align_threshold"]) if "cold_start_align_threshold" in config else 10
        
        # [Inference Cold Text Boost] 推理时冷启动文本权重增强
        # 问题: 训练时的 cold_start_align_boost 只影响对齐损失，推理时所有商品使用相同的 text_weight
        #       导致模型对新品"有能力但不敢推"，HR_new 下降但 MRR_new 提升
        # 解决: 推理时动态增加低频商品的文本权重
        # 权重公式: effective_text_weight *= 1.0 + boost * max(0, threshold - popularity) / threshold
        # - inference_cold_text_boost: 推理增强系数，0.0表示关闭，建议0.5-2.0
        # 效果: 低频商品推理时获得更强的文本信号，提升 HR_new 而不损害 MRR_frequent
        self.inference_cold_text_boost = float(config["inference_cold_text_boost"]) if "inference_cold_text_boost" in config else 0.0
        
        # [IPW] Inverse Propensity Weighting for alignment loss
        # - use_ipw_weighting: 是否启用 IPW 权重（优先于 cold_start_align_boost）
        # - ipw_threshold: 高频阈值，pop >= threshold 时 weight ≈ 1.0（与分层评估 frequent 阈值对齐）
        # - ipw_alpha: 控制曲线形状，>1 更陡峭（低频物品权重增长更快），<1 更平缓
        # - ipw_max_weight: 最大权重（对应 pop=0），类似 1+boost
        # - ipw_clip_min/max: 权重裁剪范围，防止极端值
        self.use_ipw_weighting = bool(config["use_ipw_weighting"]) if "use_ipw_weighting" in config else False
        self.ipw_threshold = int(config["ipw_threshold"]) if "ipw_threshold" in config else 10  # 与 frequent 阈值对齐
        self.ipw_alpha = float(config["ipw_alpha"]) if "ipw_alpha" in config else 0.5
        self.ipw_max_weight = float(config["ipw_max_weight"]) if "ipw_max_weight" in config else 4.0
        self.ipw_clip_min = float(config["ipw_clip_min"]) if "ipw_clip_min" in config else 0.5
        self.ipw_clip_max = float(config["ipw_clip_max"]) if "ipw_clip_max" in config else 10.0
        # New: simple non-cross enhancements
        self.text_weight = float(config["text_weight"]) if "text_weight" in config else 1.0
        self.text_tail_threshold = int(config["text_tail_threshold"]) if "text_tail_threshold" in config else 0
        # Control whether text features participate in item embedding fusion (alignment can still run)
        self.fuse_text_feature = (
            bool(config["fuse_text_feature"]) if "fuse_text_feature" in config else True
        )
        # Text MLP normalization
        self.text_mlp_bn = bool(config["text_mlp_bn"]) if "text_mlp_bn" in config else False
        # Normalization toggles for projections and fused item embeddings
        self.text_proj_norm_flag = bool(config["text_proj_norm"]) if "text_proj_norm" in config else True
        self.fused_item_norm_flag = bool(config["fused_item_norm"]) if "fused_item_norm" in config else True
        # Chunk size for memory-friendly full item fusion (0 disables chunking)
        self.fusion_chunk_size = int(config["fusion_chunk_size"]) if "fusion_chunk_size" in config else 0
        # For backward compatibility: accept single path as base
        item_text_emb_path_base = (
            config["item_text_emb_path_base"] if "item_text_emb_path_base" in config else None
        )
        item_text_emb_path_llm = (
            config["item_text_emb_path_llm"] if "item_text_emb_path_llm" in config else None
        )
        if item_text_emb_path_base is None and item_text_emb_path_llm is None:
            # Fallback to legacy key
            item_text_emb_path_base = config["item_text_emb_path"] if "item_text_emb_path" in config else None

        if self.disable_text_feature:
            emb_base = None
            emb_llm = None
        else: emb_base = self._load_text_embeddings(item_text_emb_path_base, self.n_items); emb_llm = self._load_text_embeddings(item_text_emb_path_llm, self.n_items)

        if self.normalize_text:
            with torch.no_grad():
                if emb_base is not None:
                    norms = torch.norm(emb_base, p=2, dim=1, keepdim=True)
                    emb_base = emb_base / norms.clamp_min(1e-8)
                    emb_base[torch.isnan(emb_base)] = 0.0
                if emb_llm is not None:
                    norms = torch.norm(emb_llm, p=2, dim=1, keepdim=True)
                    emb_llm = emb_llm / norms.clamp_min(1e-8)
                    emb_llm[torch.isnan(emb_llm)] = 0.0

        self.register_buffer("item_text_emb_base", emb_base if emb_base is not None else None)
        self.register_buffer("item_text_emb_llm", emb_llm if emb_llm is not None else None)

        # Optional multi-view text splits (e.g., per-prompt embeddings)
        self.item_text_views = []
        self.text_view_prompts = []
        self.text_view_cross_layers = None
        self.text_view_deep_layers = None
        self.text_view_predictor_layers = None
        self.text_view_cross_dropouts = None
        self.text_view_gate_params = None

        multiview_dirs = []
        for key in ("item_text_multiview_dir", "item_text_multiview_dir_llm", "item_text_multiview_dir_base"):
            if key in config and config[key]:
                multiview_dirs.append(os.path.abspath(os.path.expanduser(config[key])))

        seen_dirs = set()
        for dir_path in multiview_dirs:
            if dir_path in seen_dirs:
                continue
            seen_dirs.add(dir_path)
            views, prompts = self._load_text_views_from_dir(dir_path, self.n_items)
            if len(views) == 0:
                continue
            for tensor in views:
                buffer_name = f"text_view_tensor_{len(self.item_text_views)}"
                self.register_buffer(buffer_name, tensor)
                self.item_text_views.append(getattr(self, buffer_name))
            if prompts:
                self.text_view_prompts.extend(prompts)
        if len(self.item_text_views) > 0:
            self.logger.info(
                "SASRecAlign: loaded %d multi-view splits from %d directories (use_text_view_cross=%s).",
                len(self.item_text_views),
                len(seen_dirs),
                str(self.use_text_view_cross),
            )
        if self.use_text_view_cross and len(self.item_text_views) > 0:
            self._build_text_view_cross_modules()

        if self.disable_text_feature:
            # Safety: when text branch is explicitly disabled, never fuse text embeddings
            self.fuse_text_feature = False
        else:
            if self.use_llm:
                if self.item_text_emb_llm is None:
                    raise ValueError(
                        "SASRecAlign: use_llm=True but item_text_emb_path_llm is missing "
                        "or points to an invalid file."
                    )
            else:
                if self.item_text_emb_base is None:
                    raise ValueError(
                        "SASRecAlign: text features are enabled but item_text_emb_path_base "
                        "is missing or invalid."
                    )

        # Precompute item popularity for optional tail gating (no extra IO during training)
        pop_counts = None
        try:
            inter_iids = dataset.inter_feat[dataset.iid_field].numpy()
            pop_counts = np.bincount(inter_iids, minlength=self.n_items)
        except Exception:
            pop_counts = np.zeros((self.n_items,), dtype=np.int64)
        self.register_buffer("item_popularity", torch.from_numpy(pop_counts).long())
        # [IPW] 预计算最大 popularity 用于 IPW 权重归一化
        self.max_popularity = int(pop_counts.max()) if pop_counts.max() > 0 else 1
        if self.text_tail_threshold > 0:
            gate = (self.item_popularity <= int(self.text_tail_threshold)).float()
        else:
            gate = None
        # shape [n_items], 1.0 means enable text for that item
        self.register_buffer("text_item_gate_all", gate)

        # Determine text input formation according to flags and availability
        base_dim = int(self.item_text_emb_base.shape[1]) if self.item_text_emb_base is not None else 0
        llm_dim = int(self.item_text_emb_llm.shape[1]) if self.item_text_emb_llm is not None else 0
        self._text_mode = "none"
        if self.use_llm:
            if base_dim > 0 and llm_dim > 0:
                self._text_mode = "both"
                text_in_dim = base_dim + llm_dim
            elif llm_dim > 0:
                self._text_mode = "llm"
                text_in_dim = llm_dim
            elif base_dim > 0:
                self._text_mode = "base"
                text_in_dim = base_dim
            else:
                text_in_dim = 0
        else:
            if base_dim > 0:
                self._text_mode = "base"
                text_in_dim = base_dim
            else:
                text_in_dim = 0

        # Track the total raw text dimension for diagnostics
        self._text_input_dim = text_in_dim

        # --- New: Text Amplifier / SENet configuration ---
        self.num_text_views = int(config["num_text_views"]) if "num_text_views" in config else 1
        self.text_use_senet = bool(config["text_use_senet"]) if "text_use_senet" in config else False
        self.text_amplifier = None
        
        if text_in_dim > 0 and self.text_use_senet:
             # Import dynamically to avoid top-level dependency if not needed
             try:
                 from recbole.model.sequential_recommender.text_amplifier import TextFeatureAmplifier
                 self.logger.info(f"SASRecAlign: initializing TextFeatureAmplifier with {self.num_text_views} views, SENet=True.")
                 self.text_amplifier = TextFeatureAmplifier(
                     input_dim=text_in_dim,
                     output_dim=self.hidden_size,
                     num_views=self.num_text_views,
                     apply_senet=True
                 )
             except ImportError:
                 self.logger.warning("SASRecAlign: TextFeatureAmplifier not found. SENet disabled.")
                 self.text_use_senet = False

        # Build text projection/fusion modules
        self.text_cross = None
        self.text_deep = None
        self.text_predictor = None
        self.item_text_proj = None
        self.item_concat_predictor = None
        self.text_proj_norm = None
        self.fused_item_norm = None
        self.text_cross_dropout = None
        self.item_fusion_cross_dropout = None
        
        # Item-side fusion modules for integrating text into scoring
        self.item_fusion_cross = None
        self.item_fusion_deep = None
        self.item_fusion_predictor = None
        self.item_emb_norm = None  # New: LN for item embedding before fusion
        
        self.seq_text_cross = None
        self.seq_text_cross_dropout = None
        self.seq_text_residual_gate = None
        
        if text_in_dim > 0:
            if self.fused_item_norm_flag:
                # Use same eps as backbone
                self.item_emb_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

            # Logic:
            # 1. If use_cross=True, use DCN branch.
            # 2. Else, if text_amplifier is active, use it as the 'linear' projection replacement.
            # 3. Else, use standard linear item_text_proj.
            
            # Determine input dimension for fusion/projection modules
            # If SENet is active, it transforms raw text (text_in_dim) -> hidden_size
            current_text_dim = text_in_dim
            if self.text_amplifier is not None:
                current_text_dim = self.hidden_size

            if self.use_cross:
                self.text_cross = DCNV2Cross(current_text_dim, num_layers=self.text_cross_layer_num)
                # Simple deep tower to the model hidden size (no dropout; dropout only on cross outputs)
                self.text_deep = MLPLayers([current_text_dim, self.hidden_size], dropout=0.0, bn=self.text_mlp_bn)
                self.text_predictor = nn.Linear(current_text_dim + self.hidden_size, self.hidden_size)
                
                # Item-side fusion: combine item embedding with text features
                # Input: [item_emb(hidden_size), text_features(current_text_dim)]
                fusion_input_dim = self.hidden_size + current_text_dim
                self.item_fusion_cross = DCNV2Cross(fusion_input_dim, num_layers=self.text_cross_layer_num)
                # Item fusion deep tower (no dropout; dropout only on cross outputs)
                self.item_fusion_deep = MLPLayers([fusion_input_dim, self.hidden_size], dropout=0.0, bn=self.text_mlp_bn)
                self.item_fusion_predictor = nn.Linear(fusion_input_dim + self.hidden_size, self.hidden_size)
                # dropout on cross outputs
                if self.cross_dropout_prob > 0.0:
                    self.text_cross_dropout = nn.Dropout(self.cross_dropout_prob)
                    self.item_fusion_cross_dropout = nn.Dropout(self.cross_dropout_prob)
            else:
                # Non-Cross Branch
                if self.text_amplifier is not None:
                    # Use amplifier (SENet) instead of simple linear
                    # It serves the same role: text_in_dim -> hidden_size
                    # We assign it to self.item_text_proj so standard forward logic works (it's a Module)
                    self.item_text_proj = self.text_amplifier
                else:
                    self.item_text_proj = nn.Linear(current_text_dim, self.hidden_size)

                # Concatenation-based fusion (no-cross): [item_emb, text_proj] -> hidden_size
                self.item_concat_predictor = nn.Linear(self.hidden_size * 2, self.hidden_size)


            if self.text_proj_norm_flag:
                self.text_proj_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

            if self.fused_item_norm_flag:
                self.fused_item_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

        if self.use_seq_text_cross and text_in_dim > 0:
            seq_cross_in_dim = self.hidden_size + text_in_dim
            self.seq_text_cross = DCNV2Cross(seq_cross_in_dim, num_layers=max(1, self.text_cross_layer_num))
            self.seq_text_cross_dropout = nn.Dropout(config.get("seq_text_cross_dropout", 0.1))
            self.seq_text_residual_gate = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        self._align_debug_logged = False
        # Gate-alpha logging flag (first training step)
        self._gate_debug_logged = False
        
        # Cache for fused item embeddings to improve efficiency
        self._fused_item_emb_cache = None
        self._fusion_chunk_warned = False

        # Logging for alignment availability
        if self.alignment_weight > 0.0:
            if not self._has_item_text() or (
                (self.use_cross and (self.text_cross is None or self.text_deep is None or self.text_predictor is None))
                or (not self.use_cross and self.item_text_proj is None)
            ):
                self.logger.warning(
                    "SASRecAlign: alignment_weight>0 but no valid text modules/embeddings (expected rows=%d). Alignment will be disabled.",
                    self.n_items,
                )
            else:
                try:
                    # Prefer to check the union emb if both are available
                    check_emb = None
                    if self.item_text_emb_llm is not None:
                        check_emb = self.item_text_emb_llm
                    elif self.item_text_emb_base is not None:
                        check_emb = self.item_text_emb_base
                    nan_rows = int(torch.isnan(check_emb).any(dim=1).sum().item()) if check_emb is not None else -1
                    zero_rows = int((torch.norm(check_emb, p=2, dim=1) == 0).sum().item()) if check_emb is not None else -1
                except Exception:
                    nan_rows = zero_rows = -1
                self.logger.info(
                    "SASRecAlign: text mode=%s base_dim=%d llm_dim=%d -> hidden_size=%d; sanitized_nan_rows=%d zero_rows=%d",
                    self._text_mode, base_dim, llm_dim, self.hidden_size, nan_rows, zero_rows,
                )

        # parameters initialization
        self.apply(self._init_weights)
        # apply freezing if needed
        self.set_freeze(self.freeze_backbone)

    def set_freeze(self, freeze: bool) -> None:
        """Freeze or unfreeze backbone (ID/position embeddings + transformer encoder + layer norms)."""
        # item & position embeddings
        self.item_embedding.weight.requires_grad_(not freeze)
        self.position_embedding.weight.requires_grad_(not freeze)
        # encoder and input LayerNorm
        for p in self.trm_encoder.parameters():
            p.requires_grad_(not freeze)
        for p in self.LayerNorm.parameters():
            p.requires_grad_(not freeze)

    def get_optimizer_grouped_parameters(self, config):
        """Build optimizer param groups for lightweight, per-module learning rates.
        
        Groups:
          - text_head: text projection modules used for alignment/projection
          - dnn_cross: item-side fusion (cross/deep/predictor) and global text gate
          - backbone: SASRec backbone (ID/position embeddings + Transformer encoder + LayerNorm)
        
        Config keys (optional, fall back to global learning_rate/weight_decay):
          - lr_text_head, lr_dnn_cross, lr_backbone
          - wd_text_head, wd_dnn_cross, wd_backbone
        """
        # Base fallbacks
        base_lr = float(config["learning_rate"])
        base_wd = float(config["weight_decay"])
        lr_text_head = float(config["lr_text_head"]) if "lr_text_head" in config else base_lr
        lr_dnn_cross = float(config["lr_dnn_cross"]) if "lr_dnn_cross" in config else base_lr
        lr_backbone = float(config["lr_backbone"]) if "lr_backbone" in config else base_lr
        wd_text_head = float(config["wd_text_head"]) if "wd_text_head" in config else base_wd
        wd_dnn_cross = float(config["wd_dnn_cross"]) if "wd_dnn_cross" in config else base_wd
        wd_backbone = float(config["wd_backbone"]) if "wd_backbone" in config else base_wd

        def collect_params(modules, extra_params=None):
            params = []
            for m in modules:
                if m is None:
                    continue
                for p in m.parameters(recurse=True):
                    if p.requires_grad:
                        params.append(p)
            if extra_params:
                for p in extra_params:
                    if p is not None and getattr(p, "requires_grad", False):
                        params.append(p)
            return params

        # text_head: projection from raw/base+llm text to hidden_size
        text_head_modules = []
        if self.text_amplifier is not None:
            text_head_modules.append(self.text_amplifier)

        if self.use_cross:
            # cross-side text projection tower
            text_head_modules.extend([self.text_cross, self.text_deep, self.text_predictor])
        else:
            # non-cross single linear projection
            if self.item_text_proj is not self.text_amplifier:
                text_head_modules.append(self.item_text_proj)
        if self.text_proj_norm is not None:
            text_head_modules.append(self.text_proj_norm)

        # dnn/cross: item-side fusion modules + global gate + fused norm
        dnn_cross_modules = []
        if self.use_cross:
            dnn_cross_modules.extend([self.item_fusion_cross, self.item_fusion_deep, self.item_fusion_predictor])
        else:
            dnn_cross_modules.append(self.item_concat_predictor)
        if self.fused_item_norm is not None:
            dnn_cross_modules.append(self.fused_item_norm)
        if self.use_text_view_cross and self.text_view_predictor_layers is not None:
            dnn_cross_modules.extend(
                [self.text_view_cross_layers, self.text_view_deep_layers, self.text_view_predictor_layers]
            )
        dnn_extra_params = [self.text_gate_param]
        if self.text_view_gate_params is not None:
            dnn_extra_params.append(self.text_view_gate_params)

        # backbone: SASRec core
        backbone_modules = [self.item_embedding, self.position_embedding, self.trm_encoder, self.LayerNorm]

        g_text = collect_params(text_head_modules)
        g_dnn = collect_params(dnn_cross_modules, extra_params=dnn_extra_params)
        g_backbone = collect_params(backbone_modules)

        groups = []
        if len(g_text) > 0:
            groups.append({"params": g_text, "lr": lr_text_head, "weight_decay": wd_text_head})
        if len(g_dnn) > 0:
            groups.append({"params": g_dnn, "lr": lr_dnn_cross, "weight_decay": wd_dnn_cross})
        if len(g_backbone) > 0:
            groups.append({"params": g_backbone, "lr": lr_backbone, "weight_decay": wd_backbone})

        # Any leftover trainable params fall back to base lr/wd
        covered = {id(p) for g in groups for p in g["params"]}
        rest = [p for p in self.parameters() if p.requires_grad and id(p) not in covered]
        if len(rest) > 0:
            groups.append({"params": rest, "lr": base_lr, "weight_decay": base_wd})
        return groups

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _load_text_embeddings(self, path, expected_rows):
        if path is None or (isinstance(path, str) and path.strip() == ""):
            return None
        if not isinstance(path, str) or not os.path.exists(path):
            return None
        emb = None
        try:
            if path.endswith(".npy"):
                emb_np = np.load(path)
                emb = torch.from_numpy(emb_np).float()
            else:
                loaded = torch.load(path, map_location="cpu")
                if isinstance(loaded, torch.Tensor):
                    emb = loaded.float()
                elif isinstance(loaded, np.ndarray):
                    emb = torch.from_numpy(loaded).float()
                elif isinstance(loaded, dict) and "emb" in loaded:
                    emb = loaded["emb"].float()
        except Exception:
            emb = None
        if emb is None or emb.dim() != 2:
            return None
        if emb.size(0) != expected_rows:
            return None
        return emb

    def _load_text_views_from_dir(self, dir_path: str, expected_rows: int):
        views = []
        prompts = []
        if dir_path is None or not os.path.isdir(dir_path):
            return views, prompts
        metadata_entries = []
        meta_path = os.path.join(dir_path, "views.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                for entry in meta.get("prompts", []):
                    file_name = entry.get("file")
                    if not file_name:
                        continue
                    full_path = file_name if os.path.isabs(file_name) else os.path.join(dir_path, file_name)
                    prompt = entry.get("prompt") or f"view_{entry.get('index', len(metadata_entries))}"
                    metadata_entries.append((full_path, prompt))
            except Exception as e:
                self.logger.warning("SASRecAlign: failed to parse %s (%s); falling back to *.npy listing.", meta_path, e)
        if len(metadata_entries) == 0:
            try:
                npy_files = sorted(f for f in os.listdir(dir_path) if f.endswith(".npy"))
            except Exception:
                npy_files = []
            metadata_entries = [(os.path.join(dir_path, fname), f"view_{idx}") for idx, fname in enumerate(npy_files)]
        for file_path, prompt in metadata_entries:
            if not os.path.exists(file_path):
                self.logger.warning("SASRecAlign: multi-view file missing: %s", file_path)
                continue
            try:
                arr = np.load(file_path)
            except Exception as e:
                self.logger.warning("SASRecAlign: failed to load %s (%s)", file_path, e)
                continue
            if arr.ndim != 2 or arr.shape[0] != expected_rows:
                self.logger.warning(
                    "SASRecAlign: invalid multi-view shape %s (expected rows=%d).", file_path, expected_rows
                )
                continue
            tensor = torch.from_numpy(arr).float()
            views.append(tensor)
            prompts.append(prompt)
        return views, prompts

    def _has_text_views(self) -> bool:
        return hasattr(self, "item_text_views") and len(self.item_text_views) > 0

    def _gather_text_views(self, ids_flat: torch.Tensor):
        if not self._has_text_views():
            return []
        gathered = []
        for tensor in self.item_text_views:
            gathered.append(tensor[ids_flat])
        return gathered

    def _build_text_view_cross_modules(self):
        num_views = len(self.item_text_views)
        if num_views == 0:
            return
        self.text_view_cross_layers = nn.ModuleList()
        self.text_view_deep_layers = nn.ModuleList()
        self.text_view_predictor_layers = nn.ModuleList()
        if self.cross_dropout_prob > 0.0:
            self.text_view_cross_dropouts = nn.ModuleList()
        else:
            self.text_view_cross_dropouts = None
        self.text_view_gate_params = nn.Parameter(
            torch.full((num_views,), float(self.text_gate_init), dtype=torch.float32)
        )
        for tensor in self.item_text_views:
            view_dim = tensor.size(1)
            fusion_input_dim = self.hidden_size + view_dim
            self.text_view_cross_layers.append(DCNV2Cross(fusion_input_dim, num_layers=self.text_cross_layer_num))
            self.text_view_deep_layers.append(
                MLPLayers([fusion_input_dim, self.hidden_size], dropout=0.0, bn=self.text_mlp_bn)
            )
            self.text_view_predictor_layers.append(nn.Linear(fusion_input_dim + self.hidden_size, self.hidden_size))
            if self.text_view_cross_dropouts is not None:
                self.text_view_cross_dropouts.append(nn.Dropout(self.cross_dropout_prob))

    def _fuse_with_text_views(self, item_emb: torch.Tensor, all_ids: torch.Tensor) -> torch.Tensor:
        if not self.use_text_view_cross or not self._has_text_views() or self.text_view_predictor_layers is None:
            return item_emb
        if self.detach_text_emb:
            view_features = [view.detach() for view in self._gather_text_views(all_ids)]
        else:
            view_features = self._gather_text_views(all_ids)
        if len(view_features) == 0:
            return item_emb

        device = item_emb.device
        if self.text_item_gate_all is not None:
            gate = self.text_item_gate_all[all_ids].to(device).unsqueeze(1)
        else:
            gate = None

        item_emb_for_fusion = item_emb
        if self.item_emb_norm is not None:
            item_emb_for_fusion = self.item_emb_norm(item_emb)

        # Compute align_scale and temp_scale for multi-view fusion (consistent with main fusion path)
        align_scale = (1.0 + self.alignment_weight) if self.alignment_weight > 0 else 1.0
        temp_scale = (0.07 / self.temperature) if self.temperature > 0 else 1.0
        
        contributions = []
        for idx, view_feat in enumerate(view_features):
            view_feat = view_feat.to(device)
            fusion_input = torch.cat([item_emb_for_fusion, view_feat], dim=1)
            cross_out = self.text_view_cross_layers[idx](fusion_input)
            if self.text_view_cross_dropouts is not None and len(self.text_view_cross_dropouts) > idx:
                cross_out = self.text_view_cross_dropouts[idx](cross_out)
            deep_out = self.text_view_deep_layers[idx](fusion_input)
            fused = torch.cat([cross_out, deep_out], dim=1)
            view_out = self.text_view_predictor_layers[idx](fused)
            if self.fused_item_norm is not None:
                view_out = self.fused_item_norm(view_out)
            delta = view_out - item_emb
            alpha = torch.sigmoid(self.text_view_gate_params[idx]) if self.text_view_gate_params is not None else 1.0
            # Include align_scale and temp_scale so alignment_weight and temperature affect inference
            scaled = self.text_weight * alpha * align_scale * temp_scale * delta
            if gate is not None:
                scaled = scaled * gate
            contributions.append(scaled)

        if len(contributions) == 0:
            return item_emb
        stacked = torch.stack(contributions, dim=0).mean(dim=0)
        fused_emb = item_emb + self.text_view_residual_weight * stacked
        return fused_emb

    def _has_item_text(self) -> bool:
        return (
            hasattr(self, "item_text_emb_base")
            and hasattr(self, "item_text_emb_llm")
            and (self.item_text_emb_base is not None or self.item_text_emb_llm is not None)
        )

    def _gather_text_raw(self, ids_flat: torch.Tensor) -> torch.Tensor:
        parts = []
        if self._text_mode in ("base", "both") and self.item_text_emb_base is not None:
            parts.append(self.item_text_emb_base[ids_flat])
        if self._text_mode in ("llm", "both") and self.item_text_emb_llm is not None:
            parts.append(self.item_text_emb_llm[ids_flat])
        if len(parts) == 0:
            return torch.zeros((ids_flat.size(0), 0), device=ids_flat.device)
        return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

    def _project_text(self, raw: torch.Tensor) -> torch.Tensor:
        if raw.size(1) == 0:
            return torch.zeros((raw.size(0), self.hidden_size), device=raw.device)
        if self.use_cross and self.text_cross is not None and self.text_deep is not None and self.text_predictor is not None:
            # Apply SENet if active (amplifier) before cross network
            if self.text_amplifier is not None:
                raw = self.text_amplifier(raw)
            
            cross_out = self.text_cross(raw)
            if self.text_cross_dropout is not None:
                cross_out = self.text_cross_dropout(cross_out)
            deep_out = self.text_deep(raw)
            fused = torch.cat([cross_out, deep_out], dim=1)
            proj = self.text_predictor(fused)
        elif (not self.use_cross) and self.item_text_proj is not None:
            proj = self.item_text_proj(raw)
        else:
            proj = torch.zeros((raw.size(0), self.hidden_size), device=raw.device)
        if self.text_proj_norm is not None:
            proj = self.text_proj_norm(proj)
        return proj

    def _compute_cold_start_weights(self, item_ids: torch.Tensor) -> torch.Tensor:
        """计算冷启动商品的对齐损失权重。
        
        权重公式: weight = 1.0 + boost * max(0, threshold - popularity) / threshold
        - popularity >= threshold: weight = 1.0 (正常权重)
        - popularity = 0: weight = 1.0 + boost (最大权重)
        - popularity 介于两者之间: 线性插值
        """
        if self.cold_start_align_boost <= 0:
            # 关闭冷启动权重，返回全1
            return torch.ones(item_ids.size(0), device=item_ids.device)
        
        # 获取 item popularity
        item_pop = self.item_popularity[item_ids].float()  # [B]
        threshold = float(self.cold_start_align_threshold)
        
        # 计算权重: 1.0 + boost * max(0, threshold - pop) / threshold
        cold_factor = torch.clamp(threshold - item_pop, min=0) / threshold  # [0, 1]
        weights = 1.0 + self.cold_start_align_boost * cold_factor  # [1.0, 1.0 + boost]
        
        return weights

    def _compute_ipw_weights(self, item_ids: torch.Tensor) -> torch.Tensor:
        """[IPW] 计算 Inverse Propensity Weighting 权重。
        
        使用对数平滑的 IPW，确保：
        - popularity >= ipw_threshold 时，weight ≈ 1.0（与分层评估 frequent 阈值对齐）
        - popularity = 0 时，weight = ipw_max_weight
        - 中间平滑过渡，无硬阈值
        
        公式:
            threshold = ipw_threshold  # 默认 10，与 frequent 阈值对齐
            log_threshold = log(threshold + 1)
            log_pop = log(pop + 1)
            normalized = clamp((log_threshold - log_pop) / log_threshold, 0, 1)
            weight = 1.0 + (max_weight - 1.0) * normalized ** alpha
        
        Args:
            item_ids: Item IDs [B]
            
        Returns:
            weights: Per-item weights [B], 高频物品(pop>=threshold)权重≈1.0，低频物品权重更高
        """
        if not self.use_ipw_weighting:
            return torch.ones(item_ids.size(0), device=item_ids.device)
        
        # 获取 item popularity
        item_pop = self.item_popularity[item_ids].float()  # [B]
        threshold = float(self.ipw_threshold)  # 高频阈值，与分层评估的 frequent 阈值对齐
        
        # 对数平滑 IPW（基于 threshold 而非 max_pop）
        # 让 pop >= threshold 时 weight ≈ 1.0
        log_threshold = torch.log(torch.tensor(threshold + 1.0, device=item_ids.device))
        log_pop = torch.log(item_pop + 1.0)  # +1 防止 log(0)
        log_ratio = log_threshold - log_pop  # pop >= threshold 时 <= 0，pop = 0 时 = log_threshold
        
        # 归一化到 [0, 1]：pop >= threshold 时为 0，pop = 0 时为 1
        # 使用 clamp 确保 pop > threshold 时 normalized = 0（而非负值）
        normalized = torch.clamp(log_ratio / log_threshold.clamp_min(1e-6), min=0.0, max=1.0)
        
        # 应用 alpha 控制曲线形状，然后映射到 [1.0, max_weight]
        # alpha > 1: 曲线更陡峭，低频物品权重增长更快
        # alpha < 1: 曲线更平缓，权重分布更均匀
        weight_factor = normalized ** self.ipw_alpha  # [0, 1]
        weights = 1.0 + (self.ipw_max_weight - 1.0) * weight_factor  # [1.0, max_weight]
        
        # 裁剪防止极端值
        weights = torch.clamp(weights, min=self.ipw_clip_min, max=self.ipw_clip_max)
        
        return weights

    def _compute_align_weights(self, item_ids: torch.Tensor) -> torch.Tensor:
        """统一的对齐权重计算接口。
        
        优先使用 IPW，如果未启用则回退到 cold_start_align_boost。
        
        Args:
            item_ids: Item IDs [B]
            
        Returns:
            weights: Per-item weights [B]
        """
        if self.use_ipw_weighting:
            return self._compute_ipw_weights(item_ids)
        elif self.cold_start_align_boost > 0:
            return self._compute_cold_start_weights(item_ids)
        else:
            return torch.ones(item_ids.size(0), device=item_ids.device)

    def _info_nce_align_weighted(self, a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """带权重的InfoNCE对齐损失，用于冷启动增强。
        
        Args:
            a: ID embeddings [B, D]
            b: Text embeddings [B, D]
            weights: 每个样本的损失权重 [B]
            
        Note: 
            与 MultiViewV2 实现保持一致，使用 sum(loss*w)/sum(w) 而非 mean(loss*w)
            以确保权重和为正确的归一化因子
        """
        if a.size(0) == 0 or b.size(0) == 0:
            return torch.zeros(1, device=a.device)
        
        a = F.normalize(a, dim=1)
        b = F.normalize(b, dim=1)
        sim = torch.matmul(a, b.t())  # [B, B]
        
        # InfoNCE: 计算缩放后的相似度矩阵
        sim_scaled = sim / self.temperature
        labels = torch.arange(a.size(0), device=a.device)
        
        # 计算每个样本的 cross-entropy loss
        per_sample_loss = F.cross_entropy(sim_scaled, labels, reduction='none')  # [B]
        
        # 应用权重并按权重和归一化（与 MultiViewV2 保持一致）
        weighted_loss = (per_sample_loss * weights).sum() / weights.sum().clamp_min(1e-6)
        
        return weighted_loss

    def _info_nce_align(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.size(0) == 0 or b.size(0) == 0:
            return torch.zeros(1, device=a.device)
        a = F.normalize(a, dim=1)
        b = F.normalize(b, dim=1)
        sim = torch.matmul(a, b.t())
        # optional exclusion of Top-K nearest neighbors (except diagonal)
        if getattr(self, "align_exclude_topk", 0) and self.align_exclude_topk > 0:
            with torch.no_grad():
                sim_for_topk = sim.clone()
                sim_for_topk.fill_diagonal_(-1e9)
                # top-k indices to exclude per row
                _, topk_idx = torch.topk(sim_for_topk, k=min(self.align_exclude_topk, sim_for_topk.size(1) - 1), dim=1, largest=True)
                exclude_mask = torch.zeros_like(sim, dtype=torch.bool)
                exclude_mask.scatter_(1, topk_idx, True)
                exclude_mask.fill_diagonal_(False)
            sim = sim.masked_fill(exclude_mask, -1e9)
        logits = sim / self.temperature
        labels = torch.arange(a.size(0), device=a.device)
        return nn.CrossEntropyLoss()(logits, labels)
    
    def _get_fused_item_embeddings(self, item_ids: torch.Tensor = None) -> torch.Tensor:
        """Get item embeddings fused with text features.
        
        Args:
            item_ids: Specific item IDs to get embeddings for. If None, returns all items.
            
        Returns:
            Fused item embeddings of shape [n_items, hidden_size] or [batch_size, hidden_size]
        """
        if item_ids is None:
            # Get all item embeddings (chunked fusion disabled due to instability)
            all_ids = torch.arange(self.n_items, device=self.item_embedding.weight.device)
            item_emb = self.item_embedding.weight  # [n_items, hidden_size]
        else:
            all_ids = item_ids
            item_emb = self.item_embedding(item_ids)
        
        if (
            item_ids is None
            and self.fuse_text_feature
            and self.fusion_chunk_size <= 0
            and not self._fusion_chunk_warned
        ):
            try:
                approx_dim = max(self.hidden_size, int(self._text_input_dim or 0))
                approx_mb = (self.n_items * approx_dim * 4.0) / (1024 ** 2)
            except Exception:
                approx_mb = -1
            msg = (
                "SASRecAlign: fusion_chunk_size is 0 while fusing text for all items "
                "(loss=CE). This builds dense tensors for %d items each step%s. "
                "Consider setting fusion_chunk_size>0 or disabling fuse_text_feature "
                "during diagnostics."
            )
            suffix = (
                f" (~{approx_mb:.1f} MB per tensor)" if approx_mb > 0 else ""
            )
            self.logger.warning(msg, self.n_items, suffix)
            self._fusion_chunk_warned = True

        if not self.fuse_text_feature:
            return item_emb

        if self.use_text_view_cross and self._has_text_views():
            return self._fuse_with_text_views(item_emb, all_ids)

        # If no (aggregated) text features or fusion modules, return original embeddings
        if (
            (not self._has_item_text())
            or (self.use_cross and self.item_fusion_predictor is None)
            or ((not self.use_cross) and (self.item_text_proj is None or self.item_concat_predictor is None))
        ):
            return item_emb

        # Get text features for items
        text_raw = self._gather_text_raw(all_ids)
        if self.detach_text_emb:
            text_raw = text_raw.detach()
        
        # Compute effective text fusion weight incorporating alignment_weight and temperature for grid search
        # This allows alignment_weight and temperature to affect inference, not just training loss
        alpha = torch.sigmoid(self.text_gate_param)
        # Scale by alignment_weight so different grid values produce different inference results
        align_scale = (1.0 + self.alignment_weight) if self.alignment_weight > 0 else 1.0
        # Scale by temperature: higher temperature -> softer/weaker text influence
        # Use inverse relationship: lower temp means sharper/stronger text signal
        # Normalize around default temp=0.07: temp_scale = 0.07 / temperature
        temp_scale = (0.07 / self.temperature) if self.temperature > 0 else 1.0
        effective_text_weight = alpha * self.text_weight * align_scale * temp_scale
        
        # [CHANGE-9] Inference-time cold-start text boost
        # 给低频商品更强的文本信号，提升 HR_new 而不损害 MRR_frequent
        # 与 sasrecalignmultiviewv2.py 保持一致
        if self.inference_cold_text_boost > 0 and all_ids is not None:
            item_pop = self.item_popularity[all_ids].float()  # [B] or [n_items]
            threshold = float(self.cold_start_align_threshold)
            cold_factor = torch.clamp(threshold - item_pop, min=0) / threshold  # [0, 1]
            cold_boost = 1.0 + self.inference_cold_text_boost * cold_factor  # [1.0, 1.0 + boost]
            cold_boost = cold_boost.unsqueeze(1).to(item_emb.device)  # [B, 1] or [n_items, 1]
            effective_text_weight = effective_text_weight * cold_boost
        
        if self.use_cross and self.item_fusion_predictor is not None:
            # Apply SENet if active (amplifier) before cross fusion
            if self.text_amplifier is not None:
                text_raw = self.text_amplifier(text_raw)

            # Apply gating/weighting to text features before cross fusion for stability
            if self.text_item_gate_all is not None:
                if item_ids is None:
                    gate = self.text_item_gate_all
                else:
                    gate = self.text_item_gate_all[all_ids]
                gate = gate.to(item_emb.device).unsqueeze(1)
                scaled_text = (effective_text_weight * gate) * text_raw
            else:
                scaled_text = effective_text_weight * text_raw

            # Fuse item embeddings with scaled text features using cross network
            # Stabilize: normalize item_emb before concatenation if configured
            item_emb_for_fusion = item_emb
            if self.item_emb_norm is not None:
                item_emb_for_fusion = self.item_emb_norm(item_emb)

            fusion_input = torch.cat([item_emb_for_fusion, scaled_text], dim=1)
            cross_out = self.item_fusion_cross(fusion_input)
            if self.item_fusion_cross_dropout is not None:
                cross_out = self.item_fusion_cross_dropout(cross_out)
            deep_out = self.item_fusion_deep(fusion_input)
            fused = torch.cat([cross_out, deep_out], dim=1)
            fused_emb = self.item_fusion_predictor(fused)
        else:
            # Concatenation fusion (no-cross): [item_emb, scaled text_proj] -> predictor -> hidden_size
            text_proj = self._project_text(text_raw)
            if self.text_item_gate_all is not None:
                if item_ids is None:
                    gate = self.text_item_gate_all
                else:
                    gate = self.text_item_gate_all[all_ids]
                gate = gate.to(item_emb.device).unsqueeze(1)
                scaled_text = (effective_text_weight * gate) * text_proj
            else:
                scaled_text = effective_text_weight * text_proj
            
            # Stabilize here too
            item_emb_for_fusion = item_emb
            if self.item_emb_norm is not None:
                item_emb_for_fusion = self.item_emb_norm(item_emb)
                
            concat = torch.cat([item_emb_for_fusion, scaled_text], dim=1)
            fused_emb = self.item_concat_predictor(concat)
        if self.fused_item_norm is not None:
            fused_emb = self.fused_item_norm(fused_emb)
        return fused_emb

    def forward(self, item_seq, item_seq_len):
        # Token Dropout for sequence augmentation (training only)
        if self.training and self.token_dropout_prob > 0.0:
            # mask of real tokens (non-padding)
            real_token_mask = item_seq.ne(0)
            if real_token_mask.any():
                # random dropout mask
                dropout_mask = torch.rand_like(item_seq, dtype=torch.float) < float(self.token_dropout_prob)
                dropout_mask = dropout_mask & real_token_mask
                # keep the last valid token for each sequence
                try:
                    batch_index = torch.arange(item_seq.size(0), device=item_seq.device)
                    last_pos = (item_seq_len - 1).clamp(min=0)
                    dropout_mask[batch_index, last_pos] = False
                except Exception:
                    pass
                item_seq = item_seq.masked_fill(dropout_mask, 0)
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)

        if self.use_seq_text_cross and self.seq_text_cross is not None:
            # Gather corresponding text features for sequence tokens (use last item ids)
            seq_item_ids = self.gather_indexes(item_seq, item_seq_len - 1)
            text_raw = self._gather_text_raw(seq_item_ids)
            if self.detach_text_emb:
                text_raw = text_raw.detach()
            text_proj = self._project_text(text_raw)
            concat = torch.cat([output, text_proj], dim=1)
            cross_out = self.seq_text_cross(concat)
            cross_out = self.seq_text_cross_dropout(cross_out)
            gate = torch.sigmoid(self.seq_text_residual_gate)
            output = output + gate * (cross_out - output)
        return output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            # Use fused embeddings for positive and negative items
            pos_items_emb = self._get_fused_item_embeddings(pos_items)
            neg_items_emb = self._get_fused_item_embeddings(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
        else:  # CE
            # Use fused embeddings for all items
            test_item_emb = self._get_fused_item_embeddings()
            if self.cosine_score:
                seq_n = F.normalize(seq_output, dim=1)
                item_n = F.normalize(test_item_emb, dim=1)
                logits = self.cosine_scale * torch.matmul(seq_n, item_n.transpose(0, 1))
            else:
                logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)

        # optional alignment loss (align ID item embeddings with text embeddings)
        if (
            self.use_align
            and self.alignment_weight > 0.0
            and self._has_item_text()
            and ((self.use_cross and self.text_predictor is not None) or ((not self.use_cross) and self.item_text_proj is not None))
        ):
            pos_ids_flat = pos_items.view(-1)
            id_item_e = self.item_embedding(pos_ids_flat)
            txt_raw = self._gather_text_raw(pos_ids_flat)
            if self.detach_text_emb:
                txt_raw = txt_raw.detach()
            txt_item_e = self._project_text(txt_raw)
            
            # [Cold-Start Alignment Boost] 计算冷启动权重并使用带权重的对齐损失
            cold_start_weights = self._compute_cold_start_weights(pos_ids_flat)
            use_weighted_align = self.cold_start_align_boost > 0
            
            if use_weighted_align:
                align_loss = self._info_nce_align_weighted(id_item_e, txt_item_e, cold_start_weights)
            else:
                align_loss = self._info_nce_align(id_item_e, txt_item_e)
            loss = loss + self.alignment_weight * align_loss

            if not self._align_debug_logged:
                try:
                    self.logger.info(
                        "SASRecAlign: first-step align_loss=%.6f, batch_pos=%d, proj_norm=%.6f",
                        align_loss.item(), int(pos_ids_flat.numel()), float(
                            (self.text_predictor.weight if (self.use_cross and self.text_predictor is not None) else self.item_text_proj.weight).norm().item()
                        ),
                    )
                    # [Cold-Start Alignment Boost] 记录冷启动权重信息
                    if use_weighted_align:
                        min_w = cold_start_weights.min().item()
                        max_w = cold_start_weights.max().item()
                        mean_w = cold_start_weights.mean().item()
                        cold_count = (cold_start_weights > 1.01).sum().item()
                        self.logger.info(
                            "  cold_start_align: boost=%.2f, threshold=%d, weights=[min=%.2f, max=%.2f, mean=%.2f], cold_items=%d/%d",
                            self.cold_start_align_boost, self.cold_start_align_threshold,
                            min_w, max_w, mean_w, cold_count, cold_start_weights.size(0)
                        )
                    else:
                        self.logger.info("  cold_start_align: disabled (boost=0)")
                except Exception:
                    pass
                self._align_debug_logged = True

        # Regularization on text gate alpha
        if self.text_gate_reg_l2 > 0.0 or self.text_gate_reg_entropy > 0.0:
            alpha = torch.sigmoid(self.text_gate_param)
            if self.text_gate_reg_l2 > 0.0:
                loss = loss + self.text_gate_reg_l2 * (alpha ** 2)
            if self.text_gate_reg_entropy > 0.0:
                eps = 1e-8
                entropy = -(alpha * torch.log(alpha + eps) + (1.0 - alpha) * torch.log(1.0 - alpha + eps))
                loss = loss + self.text_gate_reg_entropy * entropy
            if self.text_view_gate_params is not None:
                view_alpha = torch.sigmoid(self.text_view_gate_params)
                if self.text_gate_reg_l2 > 0.0:
                    loss = loss + self.text_gate_reg_l2 * torch.sum(view_alpha ** 2)
                if self.text_gate_reg_entropy > 0.0:
                    eps = 1e-8
                    view_entropy = -(
                        view_alpha * torch.log(view_alpha + eps)
                        + (1.0 - view_alpha) * torch.log(1.0 - view_alpha + eps)
                    )
                    loss = loss + self.text_gate_reg_entropy * torch.sum(view_entropy)

        # Log gate alpha on first training step (independent of alignment branch)
        if (not self._gate_debug_logged) and self.training:
            try:
                alpha_val = float(torch.sigmoid(self.text_gate_param).detach().cpu().item())
                self.logger.info(
                    "SASRecAlign: first-step text_gate_alpha=%.6f (use_llm=%s, use_cross=%s, use_align=%s, text_mode=%s)",
                    alpha_val, str(self.use_llm), str(self.use_cross), str(self.use_align), getattr(self, "_text_mode", "unknown")
                )
            except Exception:
                pass
            self._gate_debug_logged = True

        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        # Use fused embeddings for test items
        test_item_emb = self._get_fused_item_embeddings(test_item)
        if self.cosine_score:
            seq_n = F.normalize(seq_output, dim=1)
            item_n = F.normalize(test_item_emb, dim=1)
            scores = self.cosine_scale * torch.mul(seq_n, item_n).sum(dim=1)
        else:
            scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        # Use fused embeddings for all items
        test_items_emb = self._get_fused_item_embeddings()
        if self.cosine_score:
            seq_n = F.normalize(seq_output, dim=1)
            item_n = F.normalize(test_items_emb, dim=1)
            scores = self.cosine_scale * torch.matmul(seq_n, item_n.transpose(0, 1))
        else:
            scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores



 # Alias to enable model name 'SASRec_Align' to load this module
SASRec_Align = SASRecAlign
