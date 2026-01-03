import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

class FiBiNETLayer(nn.Module):
    """
    FiBiNET-style interaction layer for processing multi-view text features.
    
    Args:
        num_fields (int): Number of text views (e.g. 5).
        embedding_size (int): Dimension of each view's embedding.
        reduction_ratio (int): Reduction ratio for SENet.
    """
    def __init__(self, num_fields, embedding_size, reduction_ratio=2):
        super(FiBiNETLayer, self).__init__()
        self.num_fields = num_fields
        self.embedding_size = embedding_size
        self.reduction_size = max(1, num_fields // reduction_ratio)
        
        # SENet Layers
        self.senet_layer = nn.Sequential(
            nn.Linear(num_fields, self.reduction_size),
            nn.ReLU(),
            nn.Linear(self.reduction_size, num_fields),
            nn.Sigmoid()
        )
        
        # Bilinear Interaction
        # Types: "Field-All", "Field-Each", "Field-Interaction"
        # Here we implement a simplified interaction: element-wise product + linear projection
        # To keep it simple and effective for "amplification", we can use a combination:
        # 1. SENet re-weighted original features
        # 2. Bilinear interaction features
        
        self.W = nn.Parameter(torch.Tensor(num_fields, embedding_size, embedding_size))
        nn.init.xavier_normal_(self.W)

    def forward(self, x):
        """
        x: [Batch, Num_Fields, Emb_Size]
        """
        # 1. SENet
        # Squeeze: [B, F, D] -> [B, F] (mean pooling over D)
        z = torch.mean(x, dim=-1) 
        # Excitation
        w = self.senet_layer(z) # [B, F]
        # Re-weight
        x_senet = x * w.unsqueeze(-1) # [B, F, D]
        
        # 2. Bilinear Interaction (Optional/Simplified)
        # For now, we can just return the flattened SENet output to be processed by MLP
        # Or implement full interaction if needed.
        # Let's stick to Flatten(SENet_Output) for now as it directly fits into the "Amplifier" concept
        
        return x_senet.reshape(x.size(0), -1) # [B, F*D]

class TextFeatureAmplifier(nn.Module):
    """
    Wrapper to handle loading and processing of amplified text features.
    Supports both 'concat' (long vector) and 'stack' (tensor) inputs if we were to extend loading.
    Current implementation focuses on processing the LOADED long vector (concat mode) 
    and applying SENet/FiBiNET logic if configured.
    """
    def __init__(self, input_dim, output_dim, num_views=1, apply_senet=False):
        super().__init__()
        self.input_dim = input_dim
        self.num_views = num_views
        self.apply_senet = apply_senet
        
        if apply_senet and num_views > 1:
            # Assume input is [B, Num_Views * View_Dim]
            view_dim = input_dim // num_views
            self.fibinet = FiBiNETLayer(num_views, view_dim)
            self.proj = nn.Linear(input_dim, output_dim) # Proj after flatten
        else:
            self.fibinet = None
            self.proj = nn.Linear(input_dim, output_dim)
            
    def forward(self, x):
        if self.fibinet is not None:
            # Reshape to [B, F, D]
            B = x.size(0)
            view_dim = self.input_dim // self.num_views
            x_reshaped = x.view(B, self.num_views, view_dim)
            x_refined = self.fibinet(x_reshaped) # [B, F*D]
            return self.proj(x_refined)
        else:
            return self.proj(x)

