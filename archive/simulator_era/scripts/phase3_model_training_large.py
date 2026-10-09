#!/usr/bin/env python3
"""
Phase 3 v3: LARGEST SMOOTH MODEL

Transformer-fusion HPC-DS1: modular residual encoders + multi-head
self-attention fusion across the 4 state components + 4 decision heads.

Target: as many parameters as comfortably fit + train on a 6GB laptop GPU
(RTX 4050) without VRAM pressure, while still training in seconds given
our current (small) dataset.

Also exports a JSON trace of internal activations for every test sample,
used by the model-internals visualizer (visualizer/model_internals.html).
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


ACTION_NAMES = ["INCREASE_MPI", "INCREASE_OMP", "ENABLE_GPU", "DISABLE_GPU",
                "DECREASE_MPI", "CHANGE_BLOCK_SIZE", "REDUCE_PRECISION", "PREFETCH"]
BOTTLENECK_NAMES = ["COMPUTE", "MEMORY", "COMMUNICATION", "IO"]


# ============================================================================
# LARGE MODEL: residual encoders + transformer fusion
# ============================================================================

class ResidualBlock(nn.Module):
    """Pre-norm residual MLP block."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.net(self.norm(x))


class WideEncoder(nn.Module):
    """Deep residual encoder for one state component."""
    def __init__(self, input_dim, hidden_dim, num_blocks):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.out_norm(h)


def hw_features(hw_dict, device):
    f = [
        hw_dict.get("cpu_cores", 8) / 64.0,
        hw_dict.get("cpu_freq_ghz", 3.5) / 5.0,
        hw_dict.get("l3_cache_mb", 24) / 100.0,
        hw_dict.get("gpu_memory_gb", 6.0) / 16.0,
        hw_dict.get("ram_gb", 24) / 128.0,
        1.0 if hw_dict.get("gpu_model") else 0.0,
    ]
    return torch.tensor(f, dtype=torch.float32, device=device)


def wl_features(wl_dict, device):
    wl_type_map = {"embarrassingly_parallel": 0.9, "moderate": 0.5, "irregular": 0.2}
    wl_type = wl_type_map.get(wl_dict.get("algorithm_type", "moderate"), 0.5)
    f = [
        math.log(wl_dict.get("problem_size", 50000) + 1) / 15.0,
        math.log(wl_dict.get("flops_estimate", 1e10) + 1) / 25.0,
        wl_type,
        0.0, 0.0,
    ]
    return torch.tensor(f, dtype=torch.float32, device=device)


def cfg_features(cfg_dict, device):
    f = [
        math.log(cfg_dict.get("mpi_ranks", 1) + 1) / 4.0,
        math.log(cfg_dict.get("omp_threads", 1) + 1) / 4.0,
        1.0 if cfg_dict.get("gpu_enabled", False) else 0.0,
        (cfg_dict.get("cuda_block_size", 128) or 128) / 256.0,
    ]
    return torch.tensor(f, dtype=torch.float32, device=device)


def tel_features(tel_dict, device):
    f = [
        math.log(tel_dict.get("runtime_sec", 2.0) + 1) / 5.0,
        tel_dict.get("cpu_util_pct", 50) / 100.0,
        tel_dict.get("gpu_util_pct", 0) / 100.0,
        tel_dict.get("memory_util_pct", 50) / 100.0,
        tel_dict.get("mpi_comm_overhead_pct", 0) / 100.0,
    ]
    return torch.tensor(f, dtype=torch.float32, device=device)


class HPCDS1Large(nn.Module):
    """
    Large HPC-DS1: modular residual encoders -> transformer fusion
    (4 module tokens, multi-head self-attention) -> 4 decision heads.
    """

    def __init__(self, hidden_dim=768, encoder_blocks=3, fusion_layers=4,
                 nhead=8, ffn_dim=3072, num_actions=8):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        self.hw_encoder = WideEncoder(6, hidden_dim, encoder_blocks)
        self.wl_encoder = WideEncoder(5, hidden_dim, encoder_blocks)
        self.cfg_encoder = WideEncoder(4, hidden_dim, encoder_blocks)
        self.tel_encoder = WideEncoder(5, hidden_dim, encoder_blocks)

        # Learned "which module is this token" embedding (like position ids)
        self.type_embed = nn.Parameter(torch.randn(4, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=ffn_dim,
            dropout=0.1, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=fusion_layers)

        self.pool_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_actions),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.stop_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.bottleneck_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 4),
        )

        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, state_dict, return_internals=False):
        device = self.type_embed.device

        hw_in = hw_features(state_dict.get("hardware", {}), device)
        wl_in = wl_features(state_dict.get("workload", {}), device)
        cfg_in = cfg_features(state_dict.get("current_config", {}), device)
        tel_in = tel_features(state_dict.get("telemetry", {}), device)

        hw_emb = self.hw_encoder(hw_in)
        wl_emb = self.wl_encoder(wl_in)
        cfg_emb = self.cfg_encoder(cfg_in)
        tel_emb = self.tel_encoder(tel_in)

        tokens = torch.stack([hw_emb, wl_emb, cfg_emb, tel_emb], dim=0) + self.type_embed
        tokens = tokens.unsqueeze(0)  # (1, 4, hidden_dim)

        fused = self.transformer(tokens)  # (1, 4, hidden_dim)
        pooled = fused.mean(dim=1).squeeze(0)
        pooled = self.pool_proj(pooled)

        action_logits = self.action_head(pooled)
        confidence_logit = self.confidence_head(pooled)
        stop_logit = self.stop_head(pooled)
        bottleneck_logits = self.bottleneck_head(pooled)

        confidence = torch.sigmoid(confidence_logit / self.temperature)

        out = {
            "action_logits": action_logits,
            "confidence": confidence.squeeze(),
            "stop_prob": torch.sigmoid(stop_logit).squeeze(),
            "bottleneck_logits": bottleneck_logits,
            "temperature": self.temperature.item(),
        }

        if return_internals:
            out["internals"] = {
                "encoder_norms": {
                    "hardware": hw_emb.norm().item(),
                    "workload": wl_emb.norm().item(),
                    "config": cfg_emb.norm().item(),
                    "telemetry": tel_emb.norm().item(),
                },
                "fused_token_norms": fused.squeeze(0).norm(dim=-1).tolist(),
                "pooled_norm": pooled.norm().item(),
            }

        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_trajectories(jsonl_file):
    trajectories = []
    with open(jsonl_file, "r") as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))
    return trajectories


def trajectory_to_sample(traj):
    state = traj.get("state", {})
    best_action_name = traj.get("best_action", "INCREASE_MPI")
    try:
        best_action_idx = ACTION_NAMES.index(best_action_name)
    except ValueError:
        best_action_idx = 0
    best_improvement = min(1.0, max(0.0, traj.get("best_improvement", 0.0)))
    return state, best_action_idx, best_improvement


def print_phase(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def main():
    print("\n" + "=" * 70)
    print("  PHASE 3 v3: LARGEST SMOOTH MODEL (transformer fusion)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU: {torch.cuda.get_device_name(0)} ({total_vram:.1f} GB total VRAM)")

    model_dir = PROJECT_ROOT / "model"
    results_dir = PROJECT_ROOT / "results"
    viz_dir = PROJECT_ROOT / "visualizer"
    model_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    viz_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    print_phase("LOADING DATASETS")

    data_dir = PROJECT_ROOT / "data" / "phase2_results"
    train_traj = load_trajectories(data_dir / "train_trajectories.jsonl")
    val_traj = load_trajectories(data_dir / "val_trajectories.jsonl")
    test_traj = load_trajectories(data_dir / "test_trajectories.jsonl")

    train_samples = [trajectory_to_sample(t) for t in train_traj]
    val_samples = [trajectory_to_sample(t) for t in val_traj]
    test_samples = [trajectory_to_sample(t) for t in test_traj]

    print(f"[OK] Train: {len(train_samples)}  Val: {len(val_samples)}  Test: {len(test_samples)}\n")

    # ------------------------------------------------------------------
    print_phase("INITIALIZING LARGE MODEL")

    HIDDEN_DIM = 1280
    ENCODER_BLOCKS = 5
    FUSION_LAYERS = 6
    NHEAD = 16
    FFN_DIM = 5120

    model = HPCDS1Large(
        hidden_dim=HIDDEN_DIM, encoder_blocks=ENCODER_BLOCKS,
        fusion_layers=FUSION_LAYERS, nhead=NHEAD, ffn_dim=FFN_DIM,
        num_actions=8,
    ).to(device)

    param_count = model.count_parameters()
    weight_bytes = param_count * 4
    train_state_bytes = param_count * 4 * 4  # weights + grad + adam(m,v)

    print(f"Hidden dim:        {HIDDEN_DIM}")
    print(f"Encoder blocks:    {ENCODER_BLOCKS} per module x 4 modules")
    print(f"Fusion layers:     {FUSION_LAYERS} (transformer, {NHEAD} heads, ffn={FFN_DIM})")
    print(f"Decision heads:    4 (action, confidence, stop, bottleneck)")
    print(f"\nTotal parameters:  {param_count:,}")
    print(f"Weights (fp32):    {weight_bytes/1e6:.1f} MB")
    print(f"Full train state:  {train_state_bytes/1e6:.1f} MB (weights+grad+adam)")

    if device.type == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        budget_gb = vram_gb * 0.6  # leave headroom for OS/display/CUDA context
        print(f"VRAM budget (60% of {vram_gb:.1f} GB): {budget_gb:.2f} GB")
        print(f"Estimated usage:   {train_state_bytes/1e9:.3f} GB "
              f"({100*train_state_bytes/1e9/budget_gb:.1f}% of budget)")

    # ------------------------------------------------------------------
    print_phase("TRAINING")

    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion_action = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()

    num_epochs = 60
    best_val_loss = float("inf")
    patience = 8
    patience_counter = 0

    training_metrics = []
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        train_loss_sum, train_acc_sum = 0.0, 0.0

        for state, best_action, confidence_target in train_samples:
            optimizer.zero_grad()
            output = model(state)

            action_loss = criterion_action(
                output["action_logits"].unsqueeze(0),
                torch.tensor([best_action], device=device),
            )
            conf_loss = criterion_reg(
                output["confidence"].unsqueeze(0),
                torch.tensor([confidence_target], dtype=torch.float32, device=device),
            )
            stop_loss = 0.1 * criterion_reg(
                output["stop_prob"].unsqueeze(0),
                torch.tensor([0.3], dtype=torch.float32, device=device),
            )
            loss = action_loss + 0.5 * conf_loss + stop_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            train_acc_sum += (output["action_logits"].argmax().item() == best_action)

        train_loss = train_loss_sum / len(train_samples)
        train_acc = train_acc_sum / len(train_samples)

        model.eval()
        val_loss_sum, val_acc_sum = 0.0, 0.0
        with torch.no_grad():
            for state, best_action, confidence_target in val_samples:
                output = model(state)
                action_loss = criterion_action(
                    output["action_logits"].unsqueeze(0),
                    torch.tensor([best_action], device=device),
                )
                conf_loss = criterion_reg(
                    output["confidence"].unsqueeze(0),
                    torch.tensor([confidence_target], dtype=torch.float32, device=device),
                )
                loss = action_loss + 0.5 * conf_loss
                val_loss_sum += loss.item()
                val_acc_sum += (output["action_logits"].argmax().item() == best_action)

        val_loss = val_loss_sum / len(val_samples) if val_samples else 0.0
        val_acc = val_acc_sum / len(val_samples) if val_samples else 0.0

        for pg in optimizer.param_groups:
            pg["lr"] = 5e-4 * (0.5 ** (epoch / 15))

        metrics = {
            "epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "temperature": model.temperature.item(),
            "lr": optimizer.param_groups[0]["lr"],
        }
        training_metrics.append(metrics)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            tag = "[NEW BEST]" if val_loss < best_val_loss else ""
            print(f"Epoch {epoch+1:3d}/{num_epochs} | Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f} | Temp: {model.temperature.item():.3f} {tag}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "param_count": param_count,
                "hidden_dim": HIDDEN_DIM,
                "encoder_blocks": ENCODER_BLOCKS,
                "fusion_layers": FUSION_LAYERS,
                "nhead": NHEAD,
                "ffn_dim": FFN_DIM,
            }, model_dir / "hpc_ds1_large_checkpoint.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early stopping] Val loss plateaued for {patience} epochs")
                break

    elapsed = time.time() - start_time
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device.type == "cuda" else 0.0

    print(f"\n[OK] Training complete ({elapsed:.1f} seconds, {len(training_metrics)} epochs)")
    if device.type == "cuda":
        print(f"[OK] Peak VRAM used during training: {peak_vram_mb:.1f} MB")

    # ------------------------------------------------------------------
    print_phase("SAVING TRAINING METRICS")

    metrics_file = results_dir / "training_metrics_large.csv"
    with open(metrics_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=training_metrics[0].keys())
        writer.writeheader()
        writer.writerows(training_metrics)
    print(f"[OK] Saved {metrics_file}")

    # ------------------------------------------------------------------
    print_phase("EXPORTING MODEL INTERNALS FOR VISUALIZER")

    # Reload best checkpoint
    checkpoint = torch.load(model_dir / "hpc_ds1_large_checkpoint.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    traces = []
    correct = 0
    for i, (state, best_action_idx, best_action_name) in enumerate(test_samples, 1):
        with torch.no_grad():
            output = model(state, return_internals=True)

        logits = output["action_logits"].cpu().numpy().tolist()
        probs = torch.softmax(output["action_logits"], dim=0).cpu().numpy().tolist()
        pred_idx = int(np.argmax(logits))
        bottleneck_probs = torch.softmax(output["bottleneck_logits"], dim=0).cpu().numpy().tolist()

        is_correct = pred_idx == best_action_idx
        correct += is_correct

        traces.append({
            "sample": i,
            "state": state,
            "best_action_idx": best_action_idx,
            "best_action": ACTION_NAMES[best_action_idx],
            "predicted_action_idx": pred_idx,
            "predicted_action": ACTION_NAMES[pred_idx],
            "correct": bool(is_correct),
            "action_logits": logits,
            "action_probs": probs,
            "confidence": float(output["confidence"].item()),
            "stop_prob": float(output["stop_prob"].item()),
            "bottleneck_probs": bottleneck_probs,
            "temperature": output["temperature"],
            "internals": output["internals"],
        })

    internals_export = {
        "model_name": "HPC-DS1 Large",
        "param_count": param_count,
        "hidden_dim": HIDDEN_DIM,
        "encoder_blocks": ENCODER_BLOCKS,
        "fusion_layers": FUSION_LAYERS,
        "nhead": NHEAD,
        "ffn_dim": FFN_DIM,
        "action_names": ACTION_NAMES,
        "bottleneck_names": BOTTLENECK_NAMES,
        "test_accuracy": correct / len(test_samples) if test_samples else 0.0,
        "training_epochs": len(training_metrics),
        "training_time_sec": elapsed,
        "peak_vram_mb": peak_vram_mb,
        "best_val_loss": best_val_loss,
        "device": str(device),
        "training_curve": training_metrics,
        "test_traces": traces,
    }

    internals_file = viz_dir / "model_internals.json"
    with open(internals_file, "w") as f:
        json.dump(internals_export, f, indent=2)

    print(f"[OK] Exported internals trace to {internals_file}")

    # ------------------------------------------------------------------
    print_phase("PHASE 3 v3 COMPLETE")

    print(f"Model:            HPC-DS1 Large")
    print(f"Parameters:       {param_count:,}")
    print(f"Test accuracy:    {100*correct/len(test_samples):.1f}%")
    print(f"Peak VRAM:        {peak_vram_mb:.1f} MB")
    print(f"Training time:    {elapsed:.1f} sec")
    print(f"\nInternals exported for visualizer -> {internals_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
