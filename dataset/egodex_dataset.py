"""
EgoDex dataset loader for Being-H0 pick-and-place finetuning.

EgoDex HDF5 structure (per sequence file):
    f['transforms']               # (N, 68, 4, 4) — SE(3) joint transforms at 30 Hz
    f['confidences']              # (N, 68)        — ARKit confidence 0–1
    f['camera']                   # (3, 3)         — camera intrinsic matrix
    f.attrs['llm_description']    # str            — GPT-4 task description
    f.attrs.get('llm_description2', '')  # str     — second description (reversible tasks)

Joint index mapping (within the 68-joint hierarchy):
    Joints  0–17: body / head / arms
    Joints 18–42: right hand (25 joints)
    Joints 43–67: left  hand (25 joints)

Output per sample:
    image:       (3, H, W) float32 — context frame, ImageNet-normalised
    instruction: str               — task description
    hand_joints: (T, 50, 3) float32 in metres — right[0:25] + left[25:50]
    confidence:  (T, 50)   float32 — per-joint ARKit confidence
    camera_K:    (3, 3)    float32 — camera intrinsics
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# EgoDex joint slice indices (inclusive start, exclusive end)
RIGHT_HAND_SLICE = slice(18, 43)   # joints 18–42 → 25 joints
LEFT_HAND_SLICE  = slice(43, 68)   # joints 43–67 → 25 joints
NUM_HAND_JOINTS  = 50              # 25 right + 25 left

# Default pick-and-place keyword list for llm_description filtering.
# NOTE: EgoDex does not expose clean task-type labels — this keyword approach
# is a documented fallback. Adjust keywords based on your inspection of the data.
DEFAULT_PICK_PLACE_KEYWORDS: List[str] = [
    "pick up", "pick", "place", "grasp", "put down", "set down",
    "grab", "lift", "transfer", "move", "carry",
]

# ImageNet normalisation (matches Being-H0 preprocessing)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_image_transform(image_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def _load_video_frame(mp4_path: Path, frame_idx: int) -> Optional[np.ndarray]:
    """
    Load a single video frame as (H, W, 3) uint8 RGB.
    Tries decord → opencv. Returns None if both fail.
    """
    try:
        import decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(str(mp4_path), ctx=decord.cpu(0))
        frame = vr[frame_idx].asnumpy()  # (H, W, 3) BGR in decord native
        return frame
    except Exception:
        pass

    try:
        import cv2
        cap = cv2.VideoCapture(str(mp4_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Sequence metadata
# ---------------------------------------------------------------------------

def _read_sequence_metadata(hdf5_path: Path) -> Optional[Dict]:
    """
    Read only the metadata fields from an HDF5 file (fast, no video/transform loading).
    Returns None if the file is unreadable or malformed.
    """
    try:
        with h5py.File(hdf5_path, "r") as f:
            desc1 = str(f.attrs.get("llm_description", ""))
            desc2 = str(f.attrs.get("llm_description2", ""))

            transforms_shape = f["transforms"].shape  # (N, 68, 4, 4)
            n_frames = transforms_shape[0]

            # Derive mp4 path (same stem, same folder)
            mp4_path = hdf5_path.with_suffix(".mp4")

        return {
            "hdf5_path": hdf5_path,
            "mp4_path": mp4_path,
            "description": desc1,
            "description2": desc2,
            "n_frames": n_frames,
            "has_video": mp4_path.exists(),
        }
    except Exception as e:
        warnings.warn(f"Could not read {hdf5_path}: {e}")
        return None


def scan_sequences(root: Path) -> List[Dict]:
    """
    Recursively scan *root* for all .hdf5 files and return their metadata.
    Does NOT load video or joint transforms.
    """
    root = Path(root)
    hdf5_files = sorted(root.rglob("*.hdf5"))
    sequences = []
    for hdf5_path in hdf5_files:
        meta = _read_sequence_metadata(hdf5_path)
        if meta is not None and meta["n_frames"] > 0:
            sequences.append(meta)
    return sequences


def filter_pick_place(
    sequences: List[Dict],
    keywords: Optional[List[str]] = None,
    task_filter: str = "pick_place",
) -> List[Dict]:
    """
    Filter sequences to pick-and-place subset using keyword matching on llm_description.

    Args:
        sequences:   List of sequence metadata dicts from scan_sequences().
        keywords:    Override default keyword list.
        task_filter: "pick_place" to apply keyword filter, "all" to return everything.

    Returns filtered list.
    NOTE: This is a keyword-based heuristic because EgoDex does not expose
    structured task-type labels. Some non-pick-and-place sequences may slip
    through; some pick-and-place sequences may be excluded.
    """
    if task_filter == "all":
        return sequences

    kws = [k.lower() for k in (keywords or DEFAULT_PICK_PLACE_KEYWORDS)]

    filtered = []
    for seq in sequences:
        text = (seq["description"] + " " + seq["description2"]).lower()
        if any(kw in text for kw in kws):
            filtered.append(seq)
    return filtered


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EgoDexDataset(Dataset):
    """
    PyTorch Dataset for EgoDex pick-and-place sequences.

    Expects *root* to contain .hdf5 (and optionally .mp4) files produced by EgoDex.
    The dataset will work in metadata-only mode (no video) if MP4 files are absent,
    returning a zero image tensor in that case.

    Args:
        root:          Path to directory containing EgoDex HDF5 (+ optional MP4) files.
        split:         "train" | "val" | "test". Train/val are split 90/10 from the
                       same file list; "test" uses all sequences (for eval-only mode).
        task_filter:   "pick_place" | "all".
        keywords:      Custom keyword list for pick-and-place filtering.
        pred_horizon:  Number of future frames T to predict (default 16 ≈ 0.53s @ 30Hz).
        image_size:    Resize input frame to this square size.
        val_fraction:  Fraction of sequences held out for validation (default 0.1).
        seed:          RNG seed for deterministic train/val split.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        task_filter: str = "pick_place",
        keywords: Optional[List[str]] = None,
        pred_horizon: int = 16,
        image_size: int = 448,
        val_fraction: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        assert split in ("train", "val", "test"), f"Unknown split: {split}"

        self.root = Path(root)
        self.split = split
        self.pred_horizon = pred_horizon
        self.image_size = image_size
        self.transform = _build_image_transform(image_size)

        # Scan all sequences
        all_seqs = scan_sequences(self.root)
        if not all_seqs:
            raise RuntimeError(f"No .hdf5 files found under {self.root}")

        # Filter to pick-and-place
        filtered = filter_pick_place(all_seqs, keywords=keywords, task_filter=task_filter)

        # Keep only sequences long enough for pred_horizon
        min_frames = pred_horizon + 1
        filtered = [s for s in filtered if s["n_frames"] >= min_frames]

        # Train / val split (deterministic, by sorted path order)
        rng = np.random.default_rng(seed)
        indices = np.arange(len(filtered))
        rng.shuffle(indices)

        n_val = max(1, int(len(filtered) * val_fraction))
        if split == "val":
            self.sequences = [filtered[i] for i in indices[:n_val]]
        elif split == "train":
            self.sequences = [filtered[i] for i in indices[n_val:]]
        else:  # "test" — use everything
            self.sequences = filtered

        self._print_stats(task_filter, all_seqs, filtered)

    def _print_stats(self, task_filter: str, all_seqs: List[Dict], filtered: List[Dict]):
        if task_filter == "pick_place":
            print(
                f"[EgoDexDataset] Pick-and-place filter ({self.split}): "
                f"{len(self.sequences)} / {len(filtered)} filtered / {len(all_seqs)} total sequences"
            )
        else:
            print(
                f"[EgoDexDataset] task_filter=all ({self.split}): "
                f"{len(self.sequences)} sequences"
            )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict:
        seq = self.sequences[idx]
        hdf5_path: Path = seq["hdf5_path"]
        mp4_path: Path  = seq["mp4_path"]
        n_frames: int   = seq["n_frames"]

        # Random start frame
        max_start = n_frames - self.pred_horizon - 1
        t0 = int(np.random.randint(0, max(1, max_start)))
        t_end = t0 + self.pred_horizon

        # Load joint transforms from HDF5 (only the slice we need)
        with h5py.File(hdf5_path, "r") as f:
            # Extract translations (last column of SE(3)) for both hands
            # transforms: (N, 68, 4, 4)  → slice → (T, 50, 3)
            right = f["transforms"][t0:t_end, RIGHT_HAND_SLICE, :3, 3]  # (T, 25, 3)
            left  = f["transforms"][t0:t_end, LEFT_HAND_SLICE,  :3, 3]  # (T, 25, 3)
            hand_joints = np.concatenate([right, left], axis=1).astype(np.float32)  # (T, 50, 3)

            right_conf = f["confidences"][t0:t_end, RIGHT_HAND_SLICE]  # (T, 25)
            left_conf  = f["confidences"][t0:t_end, LEFT_HAND_SLICE]   # (T, 25)
            confidence = np.concatenate([right_conf, left_conf], axis=1).astype(np.float32)

            camera_K = f["camera"][:].astype(np.float32)  # (3, 3)

        # Load context image (frame at t0)
        image_tensor = self._load_frame(mp4_path, t0)

        instruction = seq["description"] or seq["description2"] or "pick and place"

        return {
            "image":       image_tensor,                          # (3, H, W)
            "instruction": instruction,                            # str
            "hand_joints": torch.from_numpy(hand_joints),         # (T, 50, 3) metres
            "confidence":  torch.from_numpy(confidence),          # (T, 50)
            "camera_K":    torch.from_numpy(camera_K),            # (3, 3)
        }

    def _load_frame(self, mp4_path: Path, frame_idx: int) -> torch.Tensor:
        """Load and preprocess a single video frame. Returns zero tensor if unavailable."""
        if mp4_path.exists():
            frame_np = _load_video_frame(mp4_path, frame_idx)
            if frame_np is not None:
                img = Image.fromarray(frame_np.astype(np.uint8))
                return self.transform(img)

        # Metadata-only mode: return zero image
        return torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)
