# -*- coding: utf-8 -*-
"""
UniSRec - Adapted for RecBole framework

Reference:
    Yupeng Hou et al. "Towards Universal Sequence Representation Learning for Recommender Systems."
    In KDD 2022.

Adapted from: https://github.com/RUCAIBox/UniSRec

Simplifications:
- No cross-domain pretraining; single-domain training only (transductive mode)
- Uses precomputed text embeddings (.npy) instead of runtime PLM encoding
- Full-ranking evaluation via full_sort_predict
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss


class PWLayer(nn.Module):
    """Single Parametric Whitening Layer."""

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
    """MoE-enhanced Adaptor for text embeddings."""

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


class UniSRec(SequentialRecommender):
    """
    UniSRec adapted for our framework.

    Uses precomputed text embeddings + MoE adaptor + SASRec backbone.
    Operates in transductive mode: item_embedding (ID) + adapted text embedding.
    """

    def __init__(self, config, dataset):
        super(UniSRec, self).__init__(config, dataset)

        # SASRec backbone params
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

        # UniSRec-specific params
        self.temperature = float(config["temperature"]) if "temperature" in config else 0.07
        self.n_exps = int(config["n_exps"]) if "n_exps" in config else 8
        self.adaptor_dropout = float(config["adaptor_dropout_prob"]) if "adaptor_dropout_prob" in config else 0.2

        # Backbone layers
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

        # Load precomputed text embeddings
        item_text_emb_path = config["item_text_emb_path"] if "item_text_emb_path" in config else None
        if item_text_emb_path is None:
            item_text_emb_path = config["item_text_emb_path_base"] if "item_text_emb_path_base" in config else None
        if item_text_emb_path is None:
            item_text_emb_path = config["item_text_emb_path_llm"] if "item_text_emb_path_llm" in config else None

        plm_emb = self._load_text_embeddings(item_text_emb_path, self.n_items)
        if plm_emb is None:
            raise ValueError(
                "UniSRec requires text embeddings. Set item_text_emb_path, "
                "item_text_emb_path_base, or item_text_emb_path_llm."
            )

        self.register_buffer("plm_embedding_weight", plm_emb)
        text_dim = int(plm_emb.shape[1])

        # MoE Adaptor: text_dim -> hidden_size
        self.moe_adaptor = MoEAdaptorLayer(
            self.n_exps,
            [text_dim, self.hidden_size],
            dropout=self.adaptor_dropout,
        )

        # Loss
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.apply(self._init_weights)

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

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _get_plm_embedding(self, item_ids):
        """Get adapted text embeddings via MoE adaptor."""
        raw = self.plm_embedding_weight[item_ids]
        return self.moe_adaptor(raw)

    def forward(self, item_seq, item_seq_len):
        # Text embeddings through MoE adaptor
        text_emb = self._get_plm_embedding(item_seq)

        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        # Transductive: ID embedding + adapted text embedding
        input_emb = text_emb + self.item_embedding(item_seq) + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(
            input_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output

    def _get_all_item_emb(self):
        """Get transductive item embeddings for scoring: ID + adapted text."""
        text_emb = self.moe_adaptor(self.plm_embedding_weight)
        return text_emb + self.item_embedding.weight

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            test_item_emb = self._get_all_item_emb()
            pos_items_emb = test_item_emb[pos_items]
            neg_items_emb = test_item_emb[neg_items]
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
        else:  # CE
            test_item_emb = self._get_all_item_emb()

            seq_output = F.normalize(seq_output, dim=-1)
            test_item_emb = F.normalize(test_item_emb, dim=-1)

            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1)) / self.temperature
            loss = self.loss_fct(logits, pos_items)

        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self._get_all_item_emb()
        test_item_emb = test_item_emb[test_item]

        seq_output = F.normalize(seq_output, dim=-1)
        test_item_emb = F.normalize(test_item_emb, dim=-1)

        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self._get_all_item_emb()

        seq_output = F.normalize(seq_output, dim=-1)
        test_items_emb = F.normalize(test_items_emb, dim=-1)

        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores
