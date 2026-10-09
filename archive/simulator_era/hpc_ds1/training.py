"""Training utilities for HPC-DS1."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Optional
import numpy as np


def compute_multi_task_loss(
    output: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    loss_weights: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute multi-task loss combining all decision heads.

    Tasks:
    1. Action value prediction (MSE)
    2. Confidence calibration (MSE)
    3. Stop criterion (BCE)
    4. Bottleneck classification (CrossEntropy)

    Args:
        output: Model output dict with keys: action_values, action_confidence, stop_prob, bottleneck_logits
        targets: Target dict with keys: action_values, action_confidence, stop_labels, bottleneck_class
        loss_weights: Relative weights for each loss term

    Returns:
        (total_loss, loss_dict)
    """

    if loss_weights is None:
        loss_weights = {
            "value": 1.0,
            "confidence": 1.0,
            "stop": 0.5,
            "bottleneck": 0.3,
            "entropy": -0.1,
        }

    losses = {}

    # 1. Value loss: MSE between predicted and target action values
    value_loss = F.mse_loss(output["action_values"], targets["action_values"])
    losses["value"] = value_loss

    # 2. Confidence loss: MSE between predicted confidence and ground truth
    #    GT confidence = 1 if action improved performance, 0 otherwise
    conf_loss = F.mse_loss(output["action_confidence"], targets["action_confidence"])
    losses["confidence"] = conf_loss

    # 3. Stop criterion loss: BCE for binary stop/continue decision
    stop_loss = F.binary_cross_entropy(output["stop_prob"], targets["stop_labels"])
    losses["stop"] = stop_loss

    # 4. Bottleneck classification loss: CrossEntropy for resource bottleneck
    bottleneck_loss = F.cross_entropy(output["bottleneck_logits"], targets["bottleneck_class"])
    losses["bottleneck"] = bottleneck_loss

    # 5. Entropy regularizer: encourage diversity in action scores
    action_probs = torch.softmax(output["action_values"], dim=-1)
    entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1).mean()
    losses["entropy"] = entropy

    # Weighted sum
    total_loss = (
        loss_weights["value"] * losses["value"] +
        loss_weights["confidence"] * losses["confidence"] +
        loss_weights["stop"] * losses["stop"] +
        loss_weights["bottleneck"] * losses["bottleneck"] +
        loss_weights["entropy"] * losses["entropy"]
    )

    return total_loss, losses


def expected_calibration_error(predictions: torch.Tensor, targets: torch.Tensor, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures: E[|P(correct) - accuracy|]
    Lower ECE = better calibrated confidence.

    Args:
        predictions: [N] predicted probabilities in [0, 1]
        targets: [N] binary ground truth (0 or 1)
        n_bins: Number of bins for calibration calculation

    Returns:
        ECE score (0 = perfect calibration)
    """

    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])

        if mask.sum() == 0:
            continue

        bin_acc = targets[mask].mean()
        bin_conf = predictions[mask].mean()
        bin_weight = mask.sum() / len(predictions)

        ece += bin_weight * abs(bin_conf - bin_acc)

    return ece


def fit_temperature_scaling(
    logits: torch.Tensor,
    targets: torch.Tensor,
    initial_temp: float = 1.0,
    lr: float = 0.01,
    num_iterations: int = 100,
) -> Tuple[float, float]:
    """
    Find optimal temperature T for confidence calibration.

    Minimizes NLL on validation set:
    T* = argmin_T sum(-log(softmax(logits/T)[targets]))

    Args:
        logits: [N, K] raw model logits
        targets: [N] ground truth class indices
        initial_temp: Starting temperature
        lr: Learning rate for temperature optimization
        num_iterations: Number of iterations

    Returns:
        (best_temperature, best_nll)
    """

    temperature = torch.tensor([initial_temp], dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=num_iterations)

    def closure():
        optimizer.zero_grad()
        scaled_logits = logits / temperature
        nll = F.cross_entropy(scaled_logits, targets)
        nll.backward()
        return nll

    optimizer.step(closure)
    best_nll = closure().item()
    best_temp = temperature.item()

    return best_temp, best_nll


class EarlyStopping:
    """Early stopping callback."""

    def __init__(self, patience: int = 5, verbose: bool = False):
        """
        Args:
            patience: Number of epochs without improvement before stopping
            verbose: Print messages
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.

        Returns:
            True if should stop
        """

        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss < self.best_score:
            self.best_score = val_loss
            self.counter = 0
            if self.verbose:
                print(f"Validation loss improved to {val_loss:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement for {self.counter}/{self.patience} epochs")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered")

        return self.early_stop


def train_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: HPC_DS1 model
        train_loader: DataLoader for training data
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Dictionary of loss values
    """

    model.train()
    total_losses = {
        "total": 0.0,
        "value": 0.0,
        "confidence": 0.0,
        "stop": 0.0,
        "bottleneck": 0.0,
    }
    num_batches = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Move to device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}

        # Forward pass
        outputs = model(**inputs)

        # Compute loss
        loss, loss_dict = compute_multi_task_loss(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Accumulate losses
        total_losses["total"] += loss.item()
        for key in loss_dict:
            total_losses[key] += loss_dict[key].item()
        num_batches += 1

    # Average losses
    for key in total_losses:
        total_losses[key] /= num_batches

    return total_losses


def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Validate model on validation set.

    Args:
        model: HPC_DS1 model
        val_loader: DataLoader for validation data
        device: Device to validate on

    Returns:
        Dictionary of metrics
    """

    model.eval()
    metrics = {
        "loss": 0.0,
        "ece": 0.0,
        "action_accuracy": 0.0,
    }
    num_batches = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            # Move to device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            targets = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}

            # Forward pass
            outputs = model(**inputs)

            # Compute loss
            loss, _ = compute_multi_task_loss(outputs, targets)
            metrics["loss"] += loss.item()

            # Compute ECE
            preds = outputs["action_confidence"].flatten()
            ground_truth = targets["action_confidence"].flatten()
            ece = expected_calibration_error(preds, ground_truth)
            metrics["ece"] += ece

            # Compute action accuracy (top-1)
            top_preds = torch.argmax(outputs["action_values"], dim=-1)
            top_targets = torch.argmax(targets["action_values"], dim=-1)
            acc = (top_preds == top_targets).float().mean().item()
            metrics["action_accuracy"] += acc

            num_batches += 1

    # Average metrics
    for key in metrics:
        metrics[key] /= num_batches

    return metrics
