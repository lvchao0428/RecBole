# -*- coding: utf-8 -*-
"""
UniSRecAlignMultiViewV3 - UniSRec backbone with V3 Multi-View text pipeline

Design:
- Inherits SASRecAlignMultiViewV3 for multi-view + cross + align logic
- Adds MoE adaptor from UniSRec for PLM embedding adaptation
- forward() injects MoE-adapted text embedding into Transformer input
  (same as original UniSRec: input = ID + MoE(text) + position)
- V3 scaffold (DCN-V2 cross, InfoNCE align, cold boost, SENet gate) works on top

This is the third row: UniSRec + Cross + Align + Multi-View.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.sequential_recommender.sasrecalignmultiviewv3 import SASRecAlignMultiViewV3


class PWLayer(nn.Module):
    """Single Parametric Whitening Layer (from UniSRec)."""

    def __init__(self, input_size, output_size, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = nn.Linear(input_size, output_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)


class MoEAdaptorLayer(nn.Module):
    """MoE-enhanced Adaptor for text embeddings (from UniSRec)."""

    def __init__(self, n_exps, layers, dropout=0.0, noise=True):
        super().__init__()
        self.n_exps = n_exps
        self.noisy_gating = noise
        self.experts = nn.ModuleList(
            [PWLayer(layers[0], layers[1], dropout) for _ in range(n_exps)]
        )
        self.w_gate = nn.Parameter(
            torch.zeros(layers[0], n_exps), requires_grad=True
        )
        self.w_noise = nn.Parameter(
            torch.zeros(layers[0], n_exps), requires_grad=True
        )

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = F.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            logits = noisy_logits
        else:
            logits = clean_logits
        gates = F.softmax(logits, dim=-1)
        return gates

    def forward(self, x):
        gates = self.noisy_top_k_gating(x, self.training)
        expert_outputs = [
            self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)
        ]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        return multiple_outputs.sum(dim=-2)


class UniSRecAlignMultiViewV3(SASRecAlignMultiViewV3):
    """
    UniSRec + V3 Multi-View text pipeline.

    Differences from SASRecAlignMultiViewV3:
    - Loads a PLM embedding table and adapts it via MoE adaptor
    - forward() adds MoE(text) to item_embedding before Transformer
    - Multi-view cross/align/gate/SENet all operate normally on top
    """

    def __init__(self, config, dataset):
        self._unisrec_n_exps = int(config["n_exps"]) if "n_exps" in config else 8
        self._unisrec_adaptor_dropout = float(config["adaptor_dropout_prob"]) if "adaptor_dropout_prob" in config else 0.2
        self._unisrec_plm_path = config["item_plm_emb_path"] if "item_plm_emb_path" in config else None
        if self._unisrec_plm_path is None:
            self._unisrec_plm_path = config["item_text_emb_path_plm"] if "item_text_emb_path_plm" in config else None

        super().__init__(config, dataset)

        plm_emb = self._load_plm_embeddings()
        if plm_emb is None:
            raise ValueError(
                "UniSRecAlignMultiViewV3 requires PLM embeddings. "
                "Set item_plm_emb_path or item_text_emb_path_plm in config."
            )
        self.register_buffer("plm_embedding_weight", plm_emb)
        plm_dim = int(plm_emb.shape[1])

        self.moe_adaptor = MoEAdaptorLayer(
            self._unisrec_n_exps,
            [plm_dim, self.hidden_size],
            dropout=self._unisrec_adaptor_dropout,
        )

        self.logger.info(
            "UniSRecAlignMultiViewV3: MoE adaptor (%d experts, plm_dim=%d -> hidden=%d)",
            self._unisrec_n_exps, plm_dim, self.hidden_size,
        )

    def _load_plm_embeddings(self):
        """Load PLM embeddings for MoE adaptor."""
        path = self._unisrec_plm_path
        if path is None or (isinstance(path, str) and path.strip() == ""):
            return None
        if not os.path.exists(path):
            return None
        try:
            if path.endswith(".npy"):
                emb = torch.from_numpy(np.load(path)).float()
            else:
                loaded = torch.load(path, map_location="cpu")
                if isinstance(loaded, torch.Tensor):
                    emb = loaded.float()
                elif isinstance(loaded, np.ndarray):
                    emb = torch.from_numpy(loaded).float()
                else:
                    return None
        except Exception:
            return None
        if emb.dim() != 2 or emb.size(0) != self.n_items:
            return None
        return emb

    def _get_plm_embedding(self, item_ids):
        """Get MoE-adapted PLM embeddings for given item IDs."""
        raw = self.plm_embedding_weight[item_ids]
        return self.moe_adaptor(raw)

    def forward(self, item_seq, item_seq_len):
        """
        UniSRec-style forward: input = ID + MoE(PLM) + position.
        Then standard Transformer encoding.
        """
        if self.training and self.token_dropout_prob > 0.0:
            real_token_mask = item_seq.ne(0)
            if real_token_mask.any():
                dropout_mask = torch.rand_like(item_seq, dtype=torch.float) < float(self.token_dropout_prob)
                dropout_mask = dropout_mask & real_token_mask
                try:
                    batch_index = torch.arange(item_seq.size(0), device=item_seq.device)
                    last_pos = (item_seq_len - 1).clamp(min=0)
                    dropout_mask[batch_index, last_pos] = False
                except Exception:
                    pass
                item_seq = item_seq.masked_fill(dropout_mask, 0)

        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        plm_emb = self._get_plm_embedding(item_seq)

        input_emb = item_emb + plm_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output
