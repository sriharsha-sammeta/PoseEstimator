"""
Pose prediction metrics for EgoDex hand joint evaluation.

All functions accept tensors of shape (B, T, J, 3) in metres and return
scalar floats in millimetres (unless noted).

Primary checkpoint metric: val_mpjpe
    Mean Per-Joint Position Error over all timesteps and all 50 hand joints.
    Lower is better.
"""

from __future__ import annotations

import torch


def compute_mpjpe(
    pred: torch.Tensor,
    gt: torch.Tensor,
) -> float:
    """
    Mean Per-Joint Position Error (MPJPE) — primary checkpoint metric.

    Args:
        pred: (B, T, J, 3) predicted joint positions in metres.
        gt:   (B, T, J, 3) ground-truth joint positions in metres.

    Returns:
        Scalar float in millimetres.
    """
    # L2 per joint: (B, T, J)
    per_joint_err = torch.linalg.norm(pred - gt, dim=-1)
    # Mean over B, T, J → scalar, convert m → mm
    return (per_joint_err.mean() * 1000.0).item()


def compute_mean_l2(
    pred: torch.Tensor,
    gt: torch.Tensor,
) -> float:
    """
    Mean L2 distance across all predicted keypoints (identical to MPJPE).
    Kept as a separate named metric for logging clarity.

    Returns: scalar float in millimetres.
    """
    return compute_mpjpe(pred, gt)


def compute_final_step_error(
    pred: torch.Tensor,
    gt: torch.Tensor,
) -> float:
    """
    MPJPE at the last predicted timestep only.
    Measures how well the model predicts the endpoint of the trajectory.

    Args:
        pred: (B, T, J, 3) in metres.
        gt:   (B, T, J, 3) in metres.

    Returns: scalar float in millimetres.
    """
    # Slice to last timestep: (B, J, 3)
    per_joint_err = torch.linalg.norm(pred[:, -1] - gt[:, -1], dim=-1)  # (B, J)
    return (per_joint_err.mean() * 1000.0).item()


def compute_per_timestep_error(
    pred: torch.Tensor,
    gt: torch.Tensor,
) -> list[float]:
    """
    MPJPE at each individual timestep.

    Returns: list of T floats in millimetres, one per timestep.
    """
    # (B, T, J, 3) → per-joint error (B, T, J) → mean over B,J → (T,)
    per_joint_err = torch.linalg.norm(pred - gt, dim=-1)  # (B, T, J)
    per_timestep  = per_joint_err.mean(dim=(0, 2))         # (T,)
    return (per_timestep * 1000.0).tolist()


def compute_pct_within_threshold(
    pred: torch.Tensor,
    gt: torch.Tensor,
    threshold_mm: float = 10.0,
) -> float:
    """
    Percentage of joint predictions within *threshold_mm* of ground truth.

    Args:
        pred:           (B, T, J, 3) in metres.
        gt:             (B, T, J, 3) in metres.
        threshold_mm:   Distance threshold in millimetres (default 10 mm).

    Returns: scalar float in [0, 100].
    """
    threshold_m  = threshold_mm / 1000.0
    per_joint_err = torch.linalg.norm(pred - gt, dim=-1)  # (B, T, J)
    within = (per_joint_err < threshold_m).float().mean()
    return (within * 100.0).item()


# ---------------------------------------------------------------------------
# Convenience: compute all metrics at once
# ---------------------------------------------------------------------------

def compute_all_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    threshold_mm: float = 10.0,
) -> dict[str, float]:
    """
    Compute all pose metrics and return as a dict.

    Keys: val_mpjpe, val_mean_l2, val_final_step_error, val_pct_within_Xmm
    """
    with torch.no_grad():
        pred = pred.float()
        gt   = gt.float()
        return {
            "val_mpjpe":             compute_mpjpe(pred, gt),
            "val_mean_l2":           compute_mean_l2(pred, gt),
            "val_final_step_error":  compute_final_step_error(pred, gt),
            f"val_pct_within_{int(threshold_mm)}mm": compute_pct_within_threshold(
                pred, gt, threshold_mm
            ),
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def test_metrics_smoke():
    """Quick sanity check — run with: python -m utils.metrics"""
    torch.manual_seed(0)
    B, T, J = 2, 16, 50
    pred = torch.randn(B, T, J, 3) * 0.1    # ~100 mm scale
    gt   = torch.randn(B, T, J, 3) * 0.1

    metrics = compute_all_metrics(pred, gt)
    for k, v in metrics.items():
        assert isinstance(v, float) and torch.isfinite(torch.tensor(v)), \
            f"Metric {k} is not a finite float: {v}"
        print(f"  {k}: {v:.2f}")

    per_ts = compute_per_timestep_error(pred, gt)
    assert len(per_ts) == T
    print(f"  per_timestep_error[0]: {per_ts[0]:.2f} mm")
    print("test_metrics_smoke PASSED")


if __name__ == "__main__":
    test_metrics_smoke()
