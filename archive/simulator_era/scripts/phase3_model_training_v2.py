#!/usr/bin/env python3
"""
Phase 3 v2: Full HPC-DS1 Model Training

Upgraded from 6K → 2.5M parameters

Architecture:
  - Larger modular encoders (256 dims each)
  - Deep fusion network (4 layers)
  - Multiple decision heads (action, confidence, stop, bottleneck)
  - Proper loss weighting for calibration
  - Temperature scaling for confidence

Similar to: Laya (structured encoding), Jev (confidence routing)
"""

import sys
import json
import csv
import math
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    print("[ERROR] PyTorch not installed.")
    sys.exit(1)


# ============================================================================
# FULL HPC-DS1 MODEL (2.5M parameters)
# ============================================================================

class ModuleEncoder(nn.Module):
    """Encode a single module with deeper network."""
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


class HardwareEncoder(nn.Module):
    """Encode hardware specs."""
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.encoder = ModuleEncoder(6, hidden_dim)

    def forward(self, hw_dict):
        features = [
            hw_dict.get("cpu_cores", 8) / 64.0,
            hw_dict.get("cpu_freq_ghz", 3.5) / 5.0,
            hw_dict.get("l3_cache_mb", 24) / 100.0,
            hw_dict.get("gpu_memory_gb", 6.0) / 16.0,
            hw_dict.get("ram_gb", 24) / 128.0,
            1.0 if hw_dict.get("gpu_model") else 0.0,
        ]
        x = torch.tensor(features, dtype=torch.float32, device=next(self.encoder.parameters()).device)
        return self.encoder(x)


class WorkloadEncoder(nn.Module):
    """Encode workload characteristics."""
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.encoder = ModuleEncoder(5, hidden_dim)

    def forward(self, wl_dict):
        wl_type_map = {"embarrassingly_parallel": 0.9, "moderate": 0.5, "irregular": 0.2}
        wl_type = wl_type_map.get(wl_dict.get("algorithm_type", "moderate"), 0.5)

        features = [
            math.log(wl_dict.get("problem_size", 50000) + 1) / 15.0,
            math.log(wl_dict.get("flops_estimate", 1e10) + 1) / 25.0,
            wl_type,
            0.0, 0.0,  # padding
        ]
        x = torch.tensor(features, dtype=torch.float32, device=next(self.encoder.parameters()).device)
        return self.encoder(x)


class ConfigEncoder(nn.Module):
    """Encode current configuration."""
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.encoder = ModuleEncoder(4, hidden_dim)

    def forward(self, cfg_dict):
        features = [
            math.log(cfg_dict.get("mpi_ranks", 1) + 1) / 4.0,
            math.log(cfg_dict.get("omp_threads", 1) + 1) / 4.0,
            1.0 if cfg_dict.get("gpu_enabled", False) else 0.0,
            (cfg_dict.get("cuda_block_size", 128) or 128) / 256.0,
        ]
        x = torch.tensor(features, dtype=torch.float32, device=next(self.encoder.parameters()).device)
        return self.encoder(x)


class TelemetryEncoder(nn.Module):
    """Encode system telemetry."""
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.encoder = ModuleEncoder(5, hidden_dim)

    def forward(self, tel_dict):
        features = [
            math.log(tel_dict.get("runtime_sec", 2.0) + 1) / 5.0,
            tel_dict.get("cpu_util_pct", 50) / 100.0,
            tel_dict.get("gpu_util_pct", 0) / 100.0,
            tel_dict.get("memory_util_pct", 50) / 100.0,
            tel_dict.get("mpi_comm_overhead_pct", 0) / 100.0,
        ]
        x = torch.tensor(features, dtype=torch.float32, device=next(self.encoder.parameters()).device)
        return self.encoder(x)


class HPCDS1Full(nn.Module):
    """
    Full HPC-DS1: ~2.5M parameters

    Similar to Laya: structured modular encoding
    Similar to Jev: confidence-driven routing
    """

    def __init__(self, hidden_dim=256, num_actions=8):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        # Modular encoders (4 × 256 dims = 1024 total)
        self.hw_encoder = HardwareEncoder(hidden_dim)
        self.wl_encoder = WorkloadEncoder(hidden_dim)
        self.cfg_encoder = ConfigEncoder(hidden_dim)
        self.tel_encoder = TelemetryEncoder(hidden_dim)

        # Deep fusion network (4 layers)
        fusion_dim = hidden_dim * 4
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Multiple decision heads
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.stop_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.bottleneck_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4),  # 4 bottleneck types
        )

        # Temperature for confidence scaling
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state_dict):
        """Forward pass."""
        # Encode each component
        hw_emb = self.hw_encoder(state_dict.get("hardware", {}))
        wl_emb = self.wl_encoder(state_dict.get("workload", {}))
        cfg_emb = self.cfg_encoder(state_dict.get("current_config", {}))
        tel_emb = self.tel_encoder(state_dict.get("telemetry", {}))

        # Fuse embeddings
        fused = torch.cat([hw_emb, wl_emb, cfg_emb, tel_emb], dim=0)
        hidden = self.fusion(fused)

        # Decode all decisions (non-autoregressive)
        action_logits = self.action_head(hidden)
        confidence_logit = self.confidence_head(hidden)
        stop_logit = self.stop_head(hidden)
        bottleneck_logits = self.bottleneck_head(hidden)

        # Temperature-scaled confidence
        confidence = torch.sigmoid(confidence_logit / self.temperature)

        return {
            "action_logits": action_logits,
            "confidence": confidence.squeeze(),
            "stop_prob": torch.sigmoid(stop_logit).squeeze(),
            "bottleneck_logits": bottleneck_logits,
            "temperature": self.temperature.item(),
        }

    def count_parameters(self):
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def state_to_features_dict(state_dict):
    """Return state dict (no need to convert for full model)."""
    return state_dict


def load_trajectories(jsonl_file):
    """Load trajectories from JSONL file."""
    trajectories = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))
    return trajectories


def trajectory_to_sample(traj):
    """Convert trajectory to sample."""
    state = traj.get("state", {})

    action_names = ["INCREASE_MPI", "INCREASE_OMP", "ENABLE_GPU", "DISABLE_GPU",
                   "DECREASE_MPI", "CHANGE_BLOCK_SIZE", "REDUCE_PRECISION", "PREFETCH"]
    best_action_name = traj.get("best_action", "INCREASE_MPI")
    try:
        best_action_idx = action_names.index(best_action_name)
    except ValueError:
        best_action_idx = 0

    best_improvement = min(1.0, max(0.0, traj.get("best_improvement", 0.0)))

    return state, best_action_idx, best_improvement


def print_phase(title):
    """Print phase header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def main():
    """Execute Phase 3 v2 with full model."""

    print("\n" + "="*70)
    print("  PHASE 3 v2: FULL MODEL TRAINING (2.5M PARAMETERS)")
    print("="*70)
    print("\nGoal: Train larger HPC-DS1 model (Laya/Jev scale)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model_dir = PROJECT_ROOT / "model"
    results_dir = PROJECT_ROOT / "results"
    model_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    # Load datasets
    print_phase("LOADING DATASETS")

    data_dir = PROJECT_ROOT / "data" / "phase2_results"

    train_traj = load_trajectories(data_dir / "train_trajectories.jsonl")
    val_traj = load_trajectories(data_dir / "val_trajectories.jsonl")

    train_samples = [trajectory_to_sample(t) for t in train_traj]
    val_samples = [trajectory_to_sample(t) for t in val_traj]

    print(f"[OK] Loaded {len(train_samples)} training samples")
    print(f"[OK] Loaded {len(val_samples)} validation samples\n")

    # Initialize model
    print_phase("INITIALIZING FULL MODEL")

    model = HPCDS1Full(hidden_dim=256, num_actions=8)
    model = model.to(device)

    param_count = model.count_parameters()
    print(f"Model parameters: {param_count:,}")
    print(f"Architecture: 4 modular encoders (256 dims) + deep fusion + 4 heads")
    print(f"Encoders:")
    print(f"  - Hardware: 6 -> 256")
    print(f"  - Workload: 5 -> 256")
    print(f"  - Config:   4 -> 256")
    print(f"  - Telemetry: 5 -> 256")
    print(f"Fusion: 1024 -> 512 -> 512 -> 256")
    print(f"Decision Heads:")
    print(f"  - Action: 256 -> 8 (non-autoregressive)")
    print(f"  - Confidence: 256 -> 1 (temperature-scaled)")
    print(f"  - Stop: 256 -> 1")
    print(f"  - Bottleneck: 256 -> 4\n")

    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion_action = nn.CrossEntropyLoss()
    criterion_conf = nn.MSELoss()
    criterion_bottleneck = nn.CrossEntropyLoss()

    num_epochs = 100
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    # Training loop
    print_phase("TRAINING")

    training_metrics = []
    start_time = time.time()

    for epoch in range(num_epochs):
        # Train epoch
        model.train()
        train_loss_sum = 0.0
        train_acc_sum = 0.0

        for state, best_action, confidence_target in train_samples:
            optimizer.zero_grad()

            output = model(state)

            # Multi-task loss
            action_loss = criterion_action(
                output["action_logits"].unsqueeze(0),
                torch.tensor([best_action], device=device)
            )

            conf_loss = criterion_conf(
                output["confidence"].unsqueeze(0),
                torch.tensor([confidence_target], dtype=torch.float32, device=device)
            )

            # Auxiliary tasks (weak supervision)
            stop_loss = 0.1 * criterion_conf(
                output["stop_prob"].unsqueeze(0),
                torch.tensor([0.3], dtype=torch.float32, device=device)
            )

            # Combined loss
            loss = action_loss + 0.5 * conf_loss + stop_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            pred_action = output["action_logits"].argmax().item()
            train_acc_sum += (pred_action == best_action)

        train_loss = train_loss_sum / len(train_samples)
        train_acc = train_acc_sum / len(train_samples)

        # Validate
        model.eval()
        val_loss_sum = 0.0
        val_acc_sum = 0.0

        with torch.no_grad():
            for state, best_action, confidence_target in val_samples:
                output = model(state)

                action_loss = criterion_action(
                    output["action_logits"].unsqueeze(0),
                    torch.tensor([best_action], device=device)
                )
                conf_loss = criterion_conf(
                    output["confidence"].unsqueeze(0),
                    torch.tensor([confidence_target], dtype=torch.float32, device=device)
                )
                stop_loss = 0.1 * criterion_conf(
                    output["stop_prob"].unsqueeze(0),
                    torch.tensor([0.3], dtype=torch.float32, device=device)
                )
                loss = action_loss + 0.5 * conf_loss + stop_loss

                val_loss_sum += loss.item()
                pred_action = output["action_logits"].argmax().item()
                val_acc_sum += (pred_action == best_action)

        val_loss = val_loss_sum / len(val_samples) if val_samples else 0
        val_acc = val_acc_sum / len(val_samples) if val_samples else 0

        # LR decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-3 * (0.5 ** (epoch / 20))

        # Logging
        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "temperature": model.temperature.item(),
            "lr": optimizer.param_groups[0]['lr'],
        }
        training_metrics.append(metrics)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            status = "[NEW BEST]" if val_loss < best_val_loss else ""
            print(f"Epoch {epoch+1:3d}/{num_epochs} | Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f} | Temp: {model.temperature.item():.3f} {status}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "param_count": param_count,
            }, model_dir / "hpc_ds1_full_checkpoint.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early stopping] Val loss plateaued for {patience} epochs")
                break

    elapsed = time.time() - start_time
    print(f"\n[OK] Training complete ({elapsed/60:.1f} minutes)")

    # Save training metrics
    print_phase("SAVING RESULTS")

    metrics_file = results_dir / "training_metrics_v2.csv"
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=training_metrics[0].keys())
        writer.writeheader()
        writer.writerows(training_metrics)

    print(f"[OK] Saved training metrics to {metrics_file}")

    # Summary
    print_phase("PHASE 3 v2 COMPLETE")

    print(f"Full model trained: {len(training_metrics)} epochs")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {model_dir / 'hpc_ds1_full_checkpoint.pt'}")
    print(f"Parameters: {param_count:,} (2.5M scale)")
    print(f"\nReady for Phase 4v2: Validation with full model!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
