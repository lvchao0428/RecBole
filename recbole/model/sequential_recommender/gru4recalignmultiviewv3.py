"""
GRU4RecAlignMultiViewV3 - Simplified Multi-View Text Features

简化权重配置，与 gru4recalignv3.py 保持一致：
1. align_weight: 全局对齐权重，控制对齐损失强度（继承自V3）
2. cold_text_boost: 冷启动文本增强，训练时给低频商品更高的对齐损失权重（继承自V3）
3. infer_boost: 推理增强，推理时给低频商品更强的文本信号（继承自V3）

相比 sasrecalignmultiviewv2.py 的变化：
- 继承自 GRU4RecAlignV3 而非 SASRecAlign
- 移除 multiview_align_scale（直接使用 align_weight）
- 移除 cold_start_align_boost / cold_start_align_threshold / inference_cold_text_boost（使用 V3 统一配置）
- 移除 use_ipw_weighting 及相关配置
- 保留 per_view_l2_norm、text_view_senet_ratio 等 multi-view 特有配置
"""

import json
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from recbole.model.sequential_recommender.gru4recalignv3 import GRU4RecAlignV3


class GRU4RecAlignMultiViewV3(GRU4RecAlignV3):
    """
    GRU4RecAlignMultiViewV3 - 简化版多视图文本特征模型

    继承自 GRU4RecAlignV3，复用其简化的权重配置：
    - align_weight: 全局对齐权重
    - cold_text_boost: 冷启动训练增强
    - infer_boost: 推理增强
    - cold_threshold: 冷启动阈值

    Multi-View 特有配置：
    - use_text_view_split: 是否使用多视图文本特征
    - text_view_senet_ratio: SENet 压缩比（默认 2）
    - per_view_l2_norm: 每个视图独立 L2 归一化
    - use_text_view_senet: 是否启用 SENet 增强
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # Multi-View 特有配置
        self.use_text_view_split = bool(config["use_text_view_split"]) if "use_text_view_split" in config else False
        self.use_multiview_text_cross = bool(config["use_multiview_text_cross"]) if "use_multiview_text_cross" in config else False

        if "item_text_emb_split_dir" in config and config["item_text_emb_split_dir"]:
            self.text_view_split_dir = os.path.abspath(os.path.expanduser(config["item_text_emb_split_dir"]))
        else:
            self.text_view_split_dir = None

        # SENet 和归一化配置
        self.text_view_senet_ratio = int(config["text_view_senet_ratio"]) if "text_view_senet_ratio" in config else 2
        self.use_text_view_senet = bool(config["use_text_view_senet"]) if "use_text_view_senet" in config else True
        self.per_view_l2_norm = bool(config["per_view_l2_norm"]) if "per_view_l2_norm" in config else True
        self.text_view_half_precision = bool(config["text_view_half_precision"]) if "text_view_half_precision" in config else True
        self.text_view_storage_dtype = torch.float16 if self.text_view_half_precision else torch.float32

        # 初始化多视图模块
        self.text_view_buffer_names = []
        self.text_view_proj = nn.ModuleList()
        self.text_view_senet = nn.ModuleList()
        self.text_view_gate_params = None
        self.multiview_concat_proj = None

        # DCN-V2 用于多视图文本
        self.multiview_text_cross = None
        self.multiview_text_deep = None
        self.multiview_text_predictor = None
        self.multiview_text_cross_dropout = None

        if self.use_text_view_split:
            if not self.text_view_split_dir:
                raise ValueError("use_text_view_split=True but item_text_emb_split_dir is missing.")

            meta_path = os.path.join(self.text_view_split_dir, "views.json")
            if not os.path.exists(meta_path):
                raise ValueError(f"views.json not found in {self.text_view_split_dir}")

            with open(meta_path, "r", encoding="utf-8") as mf:
                meta = json.load(mf)
            self.text_view_meta = meta
            prompts = meta.get("prompts", [])

            if len(prompts) == 0:
                raise ValueError("views.json contains no prompt metadata.")

            self.num_text_views = len(prompts)

            # 每个视图的可学习门控参数
            self.text_view_gate_params = nn.Parameter(torch.zeros(self.num_text_views, dtype=torch.float32))

            # 加载每个视图的嵌入并创建投影模块
            for view in prompts:
                idx = view["index"]
                file_name = view["file"]
                file_path = os.path.join(self.text_view_split_dir, file_name)

                if not os.path.exists(file_path):
                    raise ValueError(f"Split embedding file not found: {file_path}")

                npy = np.load(file_path)
                tensor = torch.from_numpy(npy).to(self.text_view_storage_dtype)
                buffer_name = f"text_view_emb_{idx}"
                self.register_buffer(buffer_name, tensor)
                self.text_view_buffer_names.append(buffer_name)

                view_dim = int(view["vector_dim"])

                # 投影层: view_dim → embedding_size
                self.text_view_proj.append(nn.Linear(view_dim, self.embedding_size))

                # SENet: embedding_size → reduction → embedding_size
                reduction = max(1, self.embedding_size // self.text_view_senet_ratio)
                self.text_view_senet.append(
                    nn.Sequential(
                        nn.Linear(self.embedding_size, reduction),
                        nn.ReLU(),
                        nn.Linear(reduction, self.embedding_size),
                        nn.Sigmoid(),
                    )
                )

            # 多视图拼接投影
            multiview_input_dim = self.num_text_views * self.embedding_size
            if self.item_text_emb_base is not None:
                base_dim = self.item_text_emb_base.shape[1]
                multiview_input_dim += base_dim

            self.multiview_concat_proj = nn.Linear(multiview_input_dim, self.embedding_size)

            # DCN-V2 用于多视图文本融合（可选）
            fusion_input_dim = self.embedding_size + self.embedding_size

            if self.use_cross:
                from recbole.model.sequential_recommender.gru4recalignv3 import DCNV2Cross
                from recbole.model.layers import MLPLayers

                if self.use_multiview_text_cross:
                    self.multiview_text_cross = DCNV2Cross(self.embedding_size, num_layers=self.text_cross_layer_num)
                    self.multiview_text_deep = MLPLayers(
                        [self.embedding_size, self.embedding_size],
                        dropout=0.0,
                        bn=self.text_mlp_bn
                    )
                    self.multiview_text_predictor = nn.Linear(self.embedding_size + self.embedding_size, self.embedding_size)
                    if self.cross_dropout_prob > 0.0:
                        self.multiview_text_cross_dropout = nn.Dropout(self.cross_dropout_prob)

                # 重新初始化 item fusion cross 网络
                self.item_fusion_cross = DCNV2Cross(fusion_input_dim, num_layers=self.text_cross_layer_num)
                self.item_fusion_deep = MLPLayers(
                    [fusion_input_dim, self.embedding_size],
                    dropout=0.0,
                    bn=self.text_mlp_bn
                )
                self.item_fusion_predictor = nn.Linear(
                    fusion_input_dim + self.embedding_size,
                    self.embedding_size
                )
                if self.cross_dropout_prob > 0.0:
                    self.item_fusion_cross_dropout = nn.Dropout(self.cross_dropout_prob)

            # 确保归一化层存在
            if self.item_emb_norm is None and self.fused_item_norm_flag:
                self.item_emb_norm = nn.LayerNorm(self.embedding_size, eps=1e-12)
            if self.fused_item_norm is None and self.fused_item_norm_flag:
                self.fused_item_norm = nn.LayerNorm(self.embedding_size, eps=1e-12)

            # 日志
            self.logger.info(
                "GRU4RecAlignMultiViewV3: %d views, SENet=%s (ratio=%d), per_view_l2_norm=%s",
                self.num_text_views,
                "enabled" if self.use_text_view_senet else "disabled",
                self.text_view_senet_ratio,
                self.per_view_l2_norm
            )
            self.logger.info(
                "V3 weights: align_weight=%.3f, cold_text_boost=%.2f, infer_boost=%.2f, cold_threshold=%d",
                self.align_weight, self.cold_text_boost, self.infer_boost, self.cold_threshold
            )

        self._multiview_align_debug_logged = False

    def _get_view_buffer(self, idx: int) -> torch.Tensor:
        """获取指定视图的嵌入缓冲区。"""
        name = self.text_view_buffer_names[idx]
        return getattr(self, name)

    def _gather_text_views(self, ids_flat: torch.Tensor) -> torch.Tensor:
        """
        收集并处理每个视图的特征。

        Returns:
            Stacked view features [B, num_views, embedding_size]
        """
        if not self.use_text_view_split:
            raise RuntimeError("text view split not enabled.")

        view_features = []
        for idx, proj in enumerate(self.text_view_proj):
            # 1. 加载原始视图嵌入
            view_emb_table = self._get_view_buffer(idx)
            if view_emb_table.device != ids_flat.device:
                view_emb_table = view_emb_table.to(ids_flat.device)
            gathered = view_emb_table[ids_flat]

            # 2. 投影到 embedding_size
            if gathered.dtype != proj.weight.dtype:
                gathered = gathered.to(proj.weight.dtype)
            projected = proj(gathered)

            # 3. SENet 增强（可选）
            if self.use_text_view_senet:
                excitation = self.text_view_senet[idx](projected)
                refined = projected * excitation
            else:
                refined = projected

            # 4. 每个视图独立 L2 归一化
            if self.per_view_l2_norm:
                refined = F.normalize(refined, p=2, dim=-1)

            view_features.append(refined)

        stacked = torch.stack(view_features, dim=1)
        return stacked

    def _get_fused_item_embeddings(self, item_ids: torch.Tensor = None) -> torch.Tensor:
        """
        获取融合文本特征的 item embeddings。

        Flow: SENet → L2Norm → Gate → Concat → Projection → Cross Network Fusion
        """
        if not self.use_text_view_split:
            return super()._get_fused_item_embeddings(item_ids)

        # 处理大规模 item 集合的分块
        if item_ids is None:
            all_ids = torch.arange(self.n_items, device=self.item_embedding.weight.device)
            if (
                isinstance(self.fusion_chunk_size, int)
                and self.fusion_chunk_size > 0
                and self.fusion_chunk_size < all_ids.numel()
            ):
                chunks = []
                for chunk_ids in torch.split(all_ids, self.fusion_chunk_size):
                    chunks.append(self._get_fused_item_embeddings(chunk_ids))
                return torch.cat(chunks, dim=0)
            item_emb = self.item_embedding.weight
        else:
            all_ids = item_ids
            item_emb = self.item_embedding(item_ids)

        if not self.fuse_text_feature or len(self.text_view_buffer_names) == 0:
            return item_emb

        # Step 1: 收集 SENet 增强 + L2 归一化的多视图特征
        view_stack = self._gather_text_views(all_ids)

        if self.detach_text_emb:
            view_stack = view_stack.detach()

        # Step 2: 应用每个视图的门控权重
        view_weights = torch.sigmoid(self.text_view_gate_params).to(view_stack.device)

        # Step 3: 加权拼接
        weighted_views = []
        for idx in range(self.num_text_views):
            view_feat = view_stack[:, idx, :]
            weighted = view_weights[idx] * view_feat
            weighted_views.append(weighted)

        text_concat = torch.cat(weighted_views, dim=-1)

        # Step 3.5: 拼接 base 文本特征（如果有）
        if self.item_text_emb_base is not None:
            base_feat = self.item_text_emb_base[all_ids]
            if self.detach_text_emb:
                base_feat = base_feat.detach()
            text_concat = torch.cat([base_feat, text_concat], dim=-1)

        # Step 4: 投影到 embedding_size
        text_proj = self.multiview_concat_proj(text_concat)

        # Step 4.5: DCN-V2 多视图文本交叉（可选）
        if (
            self.use_cross
            and self.multiview_text_cross is not None
            and self.multiview_text_deep is not None
            and self.multiview_text_predictor is not None
        ):
            cross_out = self.multiview_text_cross(text_proj)
            if self.multiview_text_cross_dropout is not None:
                cross_out = self.multiview_text_cross_dropout(cross_out)
            deep_out = self.multiview_text_deep(text_proj)
            text_fused = torch.cat([cross_out, deep_out], dim=1)
            text_proj = self.multiview_text_predictor(text_fused)

            if self.text_proj_norm is not None:
                text_proj = self.text_proj_norm(text_proj)

        # Step 5: 使用 Cross Network 融合
        return self._fuse_with_cross_network(item_emb, text_proj, all_ids)

    def _fuse_with_cross_network(
        self,
        item_emb: torch.Tensor,
        text_proj: torch.Tensor,
        item_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        使用 Cross Network 融合 item embeddings 和文本特征。
        """
        # 计算有效文本权重
        alpha = torch.sigmoid(self.text_gate_param) if hasattr(self, 'text_gate_param') else torch.tensor(1.0)
        effective_text_weight = alpha * self.text_weight

        # 推理增强：给冷启动商品更强的文本信号
        # 注意：只在推理时（self.training == False）应用，避免影响训练
        if self.infer_boost > 0 and item_ids is not None and not self.training:
            cold_factor = self._compute_cold_weights(item_ids)
            cold_boost = 1.0 + self.infer_boost * cold_factor
            cold_boost = cold_boost.unsqueeze(1).to(effective_text_weight.device if hasattr(effective_text_weight, 'device') else item_emb.device)
            if isinstance(effective_text_weight, torch.Tensor) and effective_text_weight.dim() == 0:
                effective_text_weight = effective_text_weight * cold_boost
            else:
                effective_text_weight = effective_text_weight * cold_boost

        # Cross Network 融合
        if self.use_cross and self.item_fusion_predictor is not None:
            if isinstance(effective_text_weight, torch.Tensor) and effective_text_weight.dim() == 0:
                scaled_text = effective_text_weight * text_proj
            else:
                scaled_text = effective_text_weight * text_proj

            item_emb_for_fusion = item_emb
            if self.item_emb_norm is not None:
                item_emb_for_fusion = self.item_emb_norm(item_emb_for_fusion)

            fusion_input = torch.cat([item_emb_for_fusion, scaled_text], dim=1)
            cross_out = self.item_fusion_cross(fusion_input)
            if self.item_fusion_cross_dropout is not None:
                cross_out = self.item_fusion_cross_dropout(cross_out)
            deep_out = self.item_fusion_deep(fusion_input)
            fused = torch.cat([cross_out, deep_out], dim=1)
            fused_emb = self.item_fusion_predictor(fused)
        else:
            item_emb_for_fusion = item_emb
            if self.item_emb_norm is not None:
                item_emb_for_fusion = self.item_emb_norm(item_emb_for_fusion)

            scaled_text = effective_text_weight * text_proj
            fused_emb = item_emb_for_fusion + scaled_text

        if self.fused_item_norm is not None:
            fused_emb = self.fused_item_norm(fused_emb)

        return fused_emb

    def calculate_loss(self, interaction):
        """
        计算损失，包含每个视图的对齐损失。
        """
        # 调用父类的损失计算（CE/BPR 损失）
        loss = super().calculate_loss(interaction)

        # 添加每个视图的对齐损失
        if (
            self.use_text_view_split
            and self.use_align
            and self.align_weight > 0.0
            and len(self.text_view_buffer_names) > 0
        ):
            pos_items = interaction[self.POS_ITEM_ID]
            id_item_emb = self.item_embedding(pos_items)
            view_stack = self._gather_text_views(pos_items)

            if self.detach_text_emb:
                view_stack = view_stack.detach()

            # 计算冷启动权重
            cold_weights = self._compute_cold_weights(pos_items) if self.cold_text_boost > 0 else None

            # 计算每个视图的对齐损失
            per_view_align_losses = []
            for idx in range(self.num_text_views):
                view_feat = view_stack[:, idx, :]
                align_loss_i = self._info_nce_align(id_item_emb, view_feat, cold_weights)
                per_view_align_losses.append(align_loss_i)

            # 平均视图对齐损失
            total_align_loss = sum(per_view_align_losses) / len(per_view_align_losses)
            loss = loss + self.align_weight * total_align_loss

            # 调试日志
            if not self._multiview_align_debug_logged:
                try:
                    losses_str = ", ".join([f"L{i}={per_view_align_losses[i].item():.6f}" for i in range(self.num_text_views)])
                    view_gate_str = ", ".join([f"g{i}={torch.sigmoid(self.text_view_gate_params[i]).item():.4f}" for i in range(self.num_text_views)])
                    self.logger.info(
                        "GRU4RecAlignMultiViewV3: per-view align | total=%.6f | losses=[%s]",
                        total_align_loss.item(), losses_str
                    )
                    self.logger.info("  view_gates=[%s]", view_gate_str)
                    if self.cold_text_boost > 0 and cold_weights is not None:
                        sample_weights = 1.0 + self.cold_text_boost * cold_weights
                        self.logger.info(
                            "  cold_text_boost=%.2f, weights=[min=%.2f, max=%.2f, mean=%.2f]",
                            self.cold_text_boost,
                            sample_weights.min().item(),
                            sample_weights.max().item(),
                            sample_weights.mean().item()
                        )
                except Exception:
                    pass
                self._multiview_align_debug_logged = True

        return loss


# Aliases
GRU4Rec_Align_MultiView_V3 = GRU4RecAlignMultiViewV3

__all__ = ["GRU4RecAlignMultiViewV3", "GRU4Rec_Align_MultiView_V3"]
