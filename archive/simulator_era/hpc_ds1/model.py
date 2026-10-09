"""HPC-DS1 Neural Network Model Architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class HardwareEncoder(nn.Module):
    """Encodes hardware specifications to embeddings."""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        # 7 hardware features: cpu_cores, cpu_freq_ghz, l3_cache_mb,
        # gpu_memory_gb, ram_gb, interconnect (0-2), gpu_present (binary)
        self.encoder = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, hw_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hw_features: [batch, 7] hardware features
        Returns:
            [batch, hidden_dim] hardware embeddings
        """
        return self.encoder(hw_features)


class WorkloadEncoder(nn.Module):
    """Encodes workload specifications."""

    def __init__(self, hidden_dim: int = 256, num_workloads: int = 5):
        super().__init__()
        self.workload_emb = nn.Embedding(num_workloads, hidden_dim)
        self.workload_feat = nn.Sequential(
            nn.Linear(3, hidden_dim),  # problem_size, algorithm_type, flops_estimate
            nn.GELU(),
        )
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, workload_idx: torch.Tensor, workload_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            workload_idx: [batch] workload type indices
            workload_features: [batch, 3] workload features
        Returns:
            [batch, hidden_dim] workload embeddings
        """
        emb = self.workload_emb(workload_idx)
        feat = self.workload_feat(workload_features)
        fused = torch.cat([emb, feat], dim=-1)
        return torch.relu(self.fusion(fused))


class ConfigEncoder(nn.Module):
    """Encodes configuration parameters."""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        # 4 config features: mpi_ranks, omp_threads, gpu_enabled, cuda_block_size
        self.encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, config_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            config_features: [batch, 4] configuration features
        Returns:
            [batch, hidden_dim] configuration embeddings
        """
        return self.encoder(config_features)


class TelemetryEncoder(nn.Module):
    """Encodes telemetry data (time-series if multiple timesteps)."""

    def __init__(self, hidden_dim: int = 256, num_timesteps: int = 1):
        super().__init__()
        # 8 telemetry metrics per timestep
        self.lstm = nn.LSTM(8, hidden_dim, batch_first=True, num_layers=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, telemetry_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            telemetry_seq: [batch, seq_len, 8] telemetry features
        Returns:
            [batch, hidden_dim] telemetry embeddings
        """
        lstm_out, _ = self.lstm(telemetry_seq)  # [batch, seq_len, hidden_dim]
        pooled = lstm_out.mean(dim=1)  # Mean pooling: [batch, hidden_dim]
        return pooled


class HistoryEncoder(nn.Module):
    """Encodes experiment history using transformer."""

    def __init__(self, hidden_dim: int = 256, num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            activation="gelu",
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, history_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            history_emb: [batch, num_history, hidden_dim] experiment history embeddings
        Returns:
            [batch, hidden_dim] compressed history
        """
        if history_emb.size(1) == 0:
            # Empty history; return zero vector
            return torch.zeros(history_emb.size(0), history_emb.size(-1), device=history_emb.device)

        transformer_out = self.transformer(history_emb)  # [batch, num_history, hidden_dim]
        pooled = transformer_out.mean(dim=1)  # [batch, hidden_dim]
        return pooled


class CandidateEncoder(nn.Module):
    """Encodes candidate actions."""

    def __init__(self, hidden_dim: int = 256, num_candidates: int = 8):
        super().__init__()
        # Each candidate has 4 features (change in mpi, omp, gpu, cuda_block)
        self.encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.num_candidates = num_candidates

    def forward(self, candidate_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            candidate_features: [batch, num_candidates, 4] candidate features
        Returns:
            [batch, num_candidates, hidden_dim] candidate embeddings
        """
        batch_size, num_cands, feat_dim = candidate_features.shape
        flat = candidate_features.view(batch_size * num_cands, feat_dim)
        encoded = self.encoder(flat)  # [batch*num_cands, hidden_dim]
        return encoded.view(batch_size, num_cands, -1)


class MultiHeadAttention(nn.Module):
    """Cross-attention between candidates and state."""

    def __init__(self, hidden_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1,
        )

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [batch, num_candidates, hidden_dim]
            key: [batch, hidden_dim] → expanded to [batch, 1, hidden_dim]
            value: [batch, hidden_dim] → expanded to [batch, 1, hidden_dim]
        Returns:
            [batch, num_candidates, hidden_dim] attended representations
        """
        key = key.unsqueeze(1)  # [batch, 1, hidden_dim]
        value = value.unsqueeze(1)

        attn_out, _ = self.attention(query, key, value)  # [batch, num_candidates, hidden_dim]
        return attn_out


class HPC_DS1(nn.Module):
    """
    Non-autoregressive decision model for HPC tuning.

    Inputs:
    - Hardware specification
    - Workload specification
    - Current configuration
    - Telemetry (single or sequence)
    - Experiment history
    - Candidate actions

    Outputs:
    - Value scores for each candidate [0, 1]
    - Confidence for each candidate [0, 1]
    - Stop decision [0, 1]
    - Bottleneck classification [0, 1] × 4
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_workloads: int = 5,
        num_candidates: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_candidates = num_candidates

        # Modular encoders
        self.hw_encoder = HardwareEncoder(hidden_dim)
        self.workload_encoder = WorkloadEncoder(hidden_dim, num_workloads)
        self.config_encoder = ConfigEncoder(hidden_dim)
        self.telemetry_encoder = TelemetryEncoder(hidden_dim)
        self.history_encoder = HistoryEncoder(hidden_dim, num_layers=2, num_heads=num_heads)
        self.candidate_encoder = CandidateEncoder(hidden_dim, num_candidates)

        # Fusion layer
        self.fusion_attention = MultiHeadAttention(hidden_dim, num_heads)
        fusion_input_dim = hidden_dim * 5  # hw + workload + config + telemetry + history
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Candidate-specific scoring
        self.candidate_context_attention = MultiHeadAttention(hidden_dim, num_heads)

        # Decision heads
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.stop_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.bottleneck_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 4),
        )

        # Temperature scaling (learned during training, fixed for inference)
        self.register_parameter("temperature", nn.Parameter(torch.ones(1)))

    def forward(
        self,
        hw_features: torch.Tensor,  # [batch, 7]
        workload_idx: torch.Tensor,  # [batch]
        workload_features: torch.Tensor,  # [batch, 3]
        config_features: torch.Tensor,  # [batch, 4]
        telemetry_seq: torch.Tensor,  # [batch, seq_len, 8]
        history_emb: torch.Tensor,  # [batch, num_history, hidden_dim] or [batch, 0, hidden_dim]
        candidate_features: torch.Tensor,  # [batch, num_candidates, 4]
    ) -> dict:
        """
        Forward pass through HPC-DS1.

        Args:
            hw_features: Hardware specifications
            workload_idx: Workload type index
            workload_features: Workload parameters
            config_features: Current configuration
            telemetry_seq: Performance telemetry (sequence)
            history_emb: Experiment history embeddings
            candidate_features: Candidate action features

        Returns:
            dict with keys: action_values, action_confidence, stop_prob, bottleneck_logits
        """
        # Step 1: Encode each modality
        hw_emb = self.hw_encoder(hw_features)  # [batch, hidden]
        wl_emb = self.workload_encoder(workload_idx, workload_features)  # [batch, hidden]
        cfg_emb = self.config_encoder(config_features)  # [batch, hidden]
        tel_emb = self.telemetry_encoder(telemetry_seq)  # [batch, hidden]
        hist_emb = self.history_encoder(history_emb)  # [batch, hidden]
        cand_emb = self.candidate_encoder(candidate_features)  # [batch, num_candidates, hidden]

        # Step 2: Fuse all modalities
        fused_context = torch.cat([hw_emb, wl_emb, cfg_emb, tel_emb, hist_emb], dim=-1)  # [batch, 5*hidden]
        state_repr = self.fusion_layer(fused_context)  # [batch, hidden]

        # Step 3: Score each candidate with attention to state
        candidate_context = self.candidate_context_attention(
            cand_emb, state_repr, state_repr
        )  # [batch, num_candidates, hidden]

        # Concatenate candidate with context for each candidate
        fused_candidates = torch.cat(
            [cand_emb, candidate_context], dim=-1
        )  # [batch, num_candidates, 2*hidden]

        # Step 4: Apply decision heads
        batch_size = fused_candidates.size(0)
        flat_candidates = fused_candidates.view(batch_size * self.num_candidates, -1)  # [batch*num_cands, 2*hidden]

        # Value and confidence heads (candidate-specific)
        action_values = self.value_head(flat_candidates)  # [batch*num_cands, 1]
        action_confidence = self.confidence_head(flat_candidates)  # [batch*num_cands, 1]

        action_values = action_values.view(batch_size, self.num_candidates)  # [batch, num_cands]
        action_confidence = action_confidence.view(batch_size, self.num_candidates)  # [batch, num_cands]

        # Stop and bottleneck heads (shared state)
        stop_logits = self.stop_head(state_repr)  # [batch, 1]
        bottleneck_logits = self.bottleneck_head(state_repr)  # [batch, 4]

        return {
            "action_values": action_values,  # [batch, num_candidates]
            "action_confidence": action_confidence,  # [batch, num_candidates]
            "stop_prob": stop_logits.squeeze(-1),  # [batch]
            "bottleneck_logits": bottleneck_logits,  # [batch, 4]
        }
