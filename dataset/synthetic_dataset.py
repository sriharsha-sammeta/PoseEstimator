"""
Synthetic dataset for dry-run / smoke-test mode.

Returns random tensors with the same shapes as EgoDexDataset.__getitem__,
so the full train.py pipeline can be exercised without any real data.
The 1B Being-H0 checkpoint is still loaded (real model loading is tested).
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class SyntheticEgoDexDataset(Dataset):
    """
    Random-data stand-in for EgoDexDataset.  Used when --dry_run is set.

    Output shapes match EgoDexDataset exactly:
        image:       (3, image_size, image_size) float32
        instruction: str
        hand_joints: (pred_horizon, 50, 3) float32  — values in [−0.5, 0.5] metres
        confidence:  (pred_horizon, 50) float32      — values in [0.8, 1.0]
        camera_K:    (3, 3) float32                  — plausible intrinsics
    """

    # Plausible instruction templates so the tokeniser sees real text
    _INSTRUCTIONS = [
        "Pick up the cup and place it on the shelf.",
        "Grasp the bottle and put it in the box.",
        "Pick up the block and set it down on the table.",
        "Grab the object and transfer it to the target location.",
        "Lift the item and place it carefully.",
    ]

    def __init__(
        self,
        n_samples: int = 32,
        pred_horizon: int = 16,
        image_size: int = 448,
    ):
        super().__init__()
        self.n_samples    = n_samples
        self.pred_horizon = pred_horizon
        self.image_size   = image_size

        # Plausible camera intrinsics (1080p-ish, fx≈fy≈1500, cx=540, cy=960)
        self._camera_K = torch.tensor(
            [[1500.0, 0.0, 540.0],
             [0.0, 1500.0, 960.0],
             [0.0,    0.0,   1.0]],
            dtype=torch.float32,
        )

        print(
            f"[SyntheticEgoDexDataset] dry-run mode: "
            f"{n_samples} synthetic samples, T={pred_horizon}, img={image_size}x{image_size}"
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        # Deterministic per-index so dataloader workers reproduce the same samples
        rng = torch.Generator()
        rng.manual_seed(idx)

        image = torch.randn(3, self.image_size, self.image_size, generator=rng)

        # Joint positions: small random values around origin, in metres
        hand_joints = (torch.rand(self.pred_horizon, 50, 3, generator=rng) - 0.5) * 0.4

        # Confidence: mostly high to mimic real data
        confidence = 0.8 + 0.2 * torch.rand(self.pred_horizon, 50, generator=rng)

        instruction = self._INSTRUCTIONS[idx % len(self._INSTRUCTIONS)]

        return {
            "image":       image,
            "instruction": instruction,
            "hand_joints": hand_joints,
            "confidence":  confidence,
            "camera_K":    self._camera_K.clone(),
        }
