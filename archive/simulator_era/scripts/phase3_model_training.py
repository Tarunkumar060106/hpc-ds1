#!/usr/bin/env python3
"""
Phase 3: Model Training (Days 5-6)

Goal: Train HPC-DS1 encoder-only model on generated trajectories
"""

import sys
import json
import csv
import math
import time
from pathlib import Path

# Add project to path
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
# SIMPLIFIED MODEL
# ============================================================================

class HPCDS1Simplified(nn.Module):
    """Simplified HPC-DS1 model for fast training."""

    def __init__(self, input_dim=20, hidden_dim=64, num_actions=8):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.action_head = nn.Linear(hidden_dim, num_actions)
        self.confidence_head = nn.Linear(hidden_dim, 1)

    def forward(self, features):
        """Forward pass."""
        hidden = self.encoder(features)
        action_logits = self.action_head(hidden)
        confidence = torch.sigmoid(self.confidence_head(hidden))

        return {
            "action_logits": action_logits,
            "confidence": confidence.squeeze(),
        }

    def count_parameters(self):
        """Count parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def state_to_features(state_dict, device):
    """Convert state dict to feature vector."""
    hw = state_dict.get("hardware", {})
    wl = state_dict.get("workload", {})
    cfg = state_dict.get("current_config", {})
    tel = state_dict.get("telemetry", {})

    features = [
        # Hardware (5)
        hw.get("cpu_cores", 8) / 64.0,
        hw.get("cpu_freq_ghz", 3.5) / 5.0,
        hw.get("gpu_memory_gb", 6.0) / 16.0,
        hw.get("ram_gb", 24) / 128.0,
        1.0 if hw.get("gpu_model") else 0.0,
        # Workload (5)
        math.log(wl.get("problem_size", 50000) + 1) / 15.0,
        math.log(wl.get("flops_estimate", 1e10) + 1) / 25.0,
        1.0 if "parallel" in wl.get("algorithm_type", "").lower() else 0.5,
        0.0, 0.0,  # padding
        # Config (5)
        math.log(cfg.get("mpi_ranks", 1) + 1) / 4.0,
        math.log(cfg.get("omp_threads", 1) + 1) / 4.0,
        1.0 if cfg.get("gpu_enabled", False) else 0.0,
        (cfg.get("cuda_block_size", 128) or 128) / 256.0,
        0.0,  # padding
        # Telemetry (5)
        math.log(tel.get("runtime_sec", 2.0) + 1) / 5.0,
        tel.get("cpu_util_pct", 50) / 100.0,
        tel.get("gpu_util_pct", 0) / 100.0,
        tel.get("memory_util_pct", 50) / 100.0,
        tel.get("mpi_comm_overhead_pct", 0) / 100.0,
    ]

    return torch.tensor(features, dtype=torch.float32, device=device)


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
    """Execute Phase 3 model training."""

    print("\n" + "="*70)
    print("  PHASE 3: MODEL TRAINING")
    print("="*70)
    print("\nGoal: Train HPC-DS1 on generated trajectories")

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Create directories
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
    print_phase("INITIALIZING MODEL")

    model = HPCDS1Simplified(input_dim=20, hidden_dim=64, num_actions=8)
    model = model.to(device)

    param_count = model.count_parameters()
    print(f"Model parameters: {param_count:,}")
    print(f"Architecture: Simplified encoder + decision heads\n")

    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion_action = nn.CrossEntropyLoss()
    criterion_conf = nn.MSELoss()

    num_epochs = 50
    best_val_loss = float('inf')
    patience = 5
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

            features = state_to_features(state, device)
            output = model(features)

            # Loss
            action_loss = criterion_action(
                output["action_logits"].unsqueeze(0),
                torch.tensor([best_action], device=device)
            )
            conf_loss = criterion_conf(
                output["confidence"].unsqueeze(0),
                torch.tensor([confidence_target], dtype=torch.float32, device=device)
            )
            loss = action_loss + 0.5 * conf_loss

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
                features = state_to_features(state, device)
                output = model(features)

                action_loss = criterion_action(
                    output["action_logits"].unsqueeze(0),
                    torch.tensor([best_action], device=device)
                )
                conf_loss = criterion_conf(
                    output["confidence"].unsqueeze(0),
                    torch.tensor([confidence_target], dtype=torch.float32, device=device)
                )
                loss = action_loss + 0.5 * conf_loss

                val_loss_sum += loss.item()
                pred_action = output["action_logits"].argmax().item()
                val_acc_sum += (pred_action == best_action)

        val_loss = val_loss_sum / len(val_samples) if val_samples else 0
        val_acc = val_acc_sum / len(val_samples) if val_samples else 0

        # LR decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-3 * (0.5 ** (epoch / 10))

        # Logging
        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]['lr'],
        }
        training_metrics.append(metrics)

        status = "[NEW BEST]" if val_loss < best_val_loss else ""
        print(f"Epoch {epoch+1:2d}/{num_epochs} | Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f} {status}")

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
            }, model_dir / "hpc_ds1_checkpoint.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early stopping] Val loss plateaued for {patience} epochs")
                break

    elapsed = time.time() - start_time
    print(f"\n[OK] Training complete ({elapsed/60:.1f} minutes)")

    # Save training metrics
    print_phase("SAVING RESULTS")

    metrics_file = results_dir / "training_metrics.csv"
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=training_metrics[0].keys())
        writer.writeheader()
        writer.writerows(training_metrics)

    print(f"[OK] Saved training metrics to {metrics_file}")

    # Summary
    print_phase("PHASE 3 COMPLETE")

    print(f"Model trained: {len(training_metrics)} epochs")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {model_dir / 'hpc_ds1_checkpoint.pt'}")
    print(f"\nTraining metrics saved to: {metrics_file}")
    print(f"\nReady for Phase 4: Validation against baselines!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
