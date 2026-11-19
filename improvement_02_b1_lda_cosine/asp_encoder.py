import os
import torch
import torch.nn as nn
from typing import Tuple, List, Optional
import numpy as np

# OpenMP fix
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class LayerWeightedSum(nn.Module):
    """Computes a learned/weighted sum across multiple Transformer hidden layers.
    
    Given hidden representations H = [H^(0), H^(1), ..., H^(L)] from L layers:
        H_weighted = sum_{l=0}^L softmax(w)_l * H^(l)
    """
    def __init__(self, num_layers: int):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(num_layers))

    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            hidden_states: List of L tensors, each of shape (batch, time, dim)
        Returns:
            Weighted tensor of shape (batch, time, dim)
        """
        # Shape: (num_layers, batch, time, dim)
        stacked = torch.stack(hidden_states, dim=0)
        norm_weights = torch.softmax(self.weights, dim=0).view(-1, 1, 1, 1)
        weighted_sum = torch.sum(norm_weights * stacked, dim=0)
        return weighted_sum


class AttentiveStatisticsPooling(nn.Module):
    """Attentive Statistics Pooling (ASP) over acoustic frame sequences.
    
    Computes attentive weighted mean (mu) and weighted standard deviation (sigma)
    across the temporal axis, producing a fixed-size utterance embedding:
        e_t = v^T * tanh(W * h_t + b)
        alpha_t = softmax(e_t)
        mu = sum_t alpha_t * h_t
        sigma = sqrt( sum_t alpha_t * (h_t - mu)^2 )
        embedding = [mu, sigma] (2 * in_dim)
    """
    def __init__(self, in_dim: int, attn_dim: int = 128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, time, in_dim)
        Returns:
            Pooled embedding of shape (batch, 2 * in_dim)
        """
        # Attention scores: (batch, time, 1)
        scores = self.attn(x)
        alpha = torch.softmax(scores, dim=1)  # (batch, time, 1)

        # Attentive Mean: (batch, in_dim)
        mean = torch.sum(alpha * x, dim=1)

        # Attentive Variance & Std: (batch, in_dim)
        diff_sq = (x - mean.unsqueeze(1)) ** 2
        var = torch.sum(alpha * diff_sq, dim=1)
        std = torch.sqrt(torch.clamp(var, min=1e-9))

        # Concatenate: (batch, 2 * in_dim)
        return torch.cat([mean, std], dim=1)


class FrozenAudioEncoder(nn.Module):
    """End-to-End Frozen Transformer Encoder with Layer-wise Weighting & Attentive Stats Pooling."""
    def __init__(self, model_type: str = "whisper", model_name: str = "openai/whisper-tiny"):
        super().__init__()
        self.model_type = model_type.lower()
        self.model_name = model_name

        if "whisper" in self.model_type:
            from transformers import WhisperModel
            whisper = WhisperModel.from_pretrained(model_name)
            self.backbone = whisper.encoder
            hidden_dim = self.backbone.config.d_model
            num_layers = self.backbone.config.encoder_layers + 1
        elif "wav2vec" in self.model_type:
            from transformers import Wav2Vec2Model
            self.backbone = Wav2Vec2Model.from_pretrained(model_name)
            hidden_dim = self.backbone.config.hidden_size
            num_layers = self.backbone.config.num_hidden_layers + 1
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        # Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        self.layer_weighter = LayerWeightedSum(num_layers=num_layers)
        self.asp_pooler = AttentiveStatisticsPooling(in_dim=hidden_dim)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """Extracts utterance-level embedding from input acoustic features."""
        with torch.no_grad():
            outputs = self.backbone(input_features, output_hidden_states=True)
            hidden_states = list(outputs.hidden_states)

        # Weighted combination of transformer layers
        fused_seq = self.layer_weighter(hidden_states)

        # Temporal attentive stats pooling
        embedding = self.asp_pooler(fused_seq)
        return embedding
