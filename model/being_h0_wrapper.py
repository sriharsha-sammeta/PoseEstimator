"""
Being-H0 wrapper for EgoDex pick-and-place finetuning.

Architecture
───────────
  BeingVLAModel (pretrained backbone)
      ↓  forward(pixel_values, input_ids, …, output_hidden_states=True)
      ↓  outputs.hidden_states[-1]        # (B, seq_len, hidden_dim)
      ↓  mean-pool over sequence tokens   # (B, hidden_dim)
  JointPredictionHead (trainable)
      ↓  LayerNorm → Linear → GELU → Linear → GELU → Linear
      ↓  reshape                          # (B, T, 50, 3)

Finetuning strategy (v1)
────────────────────────
  - Backbone is frozen by default (freeze_backbone=True)
  - Only JointPredictionHead is trained
  - Set freeze_backbone=False to also finetune the backbone (use LoRA if possible)
  - Being-H0's native motion tokeniser is NOT used in this pipeline; we train
    a continuous regression head directly on EgoDex joint positions.

Baseline evaluation (--eval_only)
──────────────────────────────────
  Running eval_only with a randomly-initialised head + pretrained backbone gives
  a "pretrained representations, zero-shot probe" baseline — measuring how much
  structural hand-pose information is already in the VLM features before training.

Model checkpoints
─────────────────
  Dry run  →  BeingBeyond/Being-H0-1B          (1B params, fits on MacBook)
  Default  →  BeingBeyond/Being-H0-8B-2508     (8B params, needs GPU)
  14B      →  BeingBeyond/Being-H0-14B-2508    (pass via --model_name)

Installation
────────────
  pip install git+https://github.com/BeingBeyond/Being-H0.git
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Ensure beingvla is importable — add third_party/Being-H0 to sys.path
# ---------------------------------------------------------------------------
_REPO_ROOT    = Path(__file__).resolve().parent.parent
_BEING_H0_DIR = _REPO_ROOT / "third_party" / "Being-H0"
if _BEING_H0_DIR.exists() and str(_BEING_H0_DIR) not in sys.path:
    sys.path.insert(0, str(_BEING_H0_DIR))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DRY_RUN_MODEL  = "BeingBeyond/Being-H0-1B-2508"
DEFAULT_MODEL  = "BeingBeyond/Being-H0-8B-2508"

# Approximate hidden dims for auto-fallback if config is unavailable
_KNOWN_HIDDEN_DIMS = {
    "1b":  2048,
    "8b":  4096,
    "14b": 5120,
}

# ---------------------------------------------------------------------------
# Regression head
# ---------------------------------------------------------------------------

class JointPredictionHead(nn.Module):
    """
    Maps a pooled backbone feature vector to (T, 50, 3) joint positions.

    The last linear layer is zero-initialised for stable training start:
    predictions begin near zero, which is a reasonable prior for relative
    hand poses in a centred coordinate frame.
    """

    def __init__(self, in_dim: int, pred_horizon: int = 16, num_joints: int = 50):
        super().__init__()
        hidden = min(1024, in_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, pred_horizon * num_joints * 3),
        )
        self.pred_horizon = pred_horizon
        self.num_joints   = num_joints

        # Zero-init output layer
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, in_dim).  Returns: (B, T, num_joints, 3)."""
        B = x.shape[0]
        return self.net(x).reshape(B, self.pred_horizon, self.num_joints, 3)


# ---------------------------------------------------------------------------
# Backbone loader
# ---------------------------------------------------------------------------

def _ensure_local_checkpoint(model_name: str, hf_token: Optional[str]) -> str:
    """
    beingvla's from_pretrained only loads weights from local directories.
    If model_name is a HuggingFace repo ID (not a local path), download it
    to the HF cache using snapshot_download and return the local path.
    """
    if os.path.isdir(model_name):
        return model_name  # already local

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("huggingface_hub is required. pip install huggingface_hub")

    print(f"  Downloading checkpoint from HuggingFace: {model_name}")
    print(f"  (this may take a while on first run — weights are cached afterwards)")
    try:
        local_path = snapshot_download(
            repo_id=model_name,
            token=hf_token,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
        )
        print(f"  Checkpoint cached at: {local_path}")
        return local_path
    except Exception as e:
        err = str(e)
        if "401" in err or "403" in err or "gated" in err.lower() or "access" in err.lower():
            print(
                f"\nERROR: Cannot download {model_name}\n"
                f"  The model may be gated. Make sure you:\n"
                f"  1. Have a HuggingFace account and accepted the model licence\n"
                f"  2. Set the HF_TOKEN environment variable (or --hf_token_env)\n"
                f"  Model page: https://huggingface.co/{model_name}\n"
            )
        raise


def _try_import_beingvla():
    """Import beingvla, raising a friendly error if not installed."""
    try:
        from beingvla.models.vla.being_vla_model import BeingVLAModel
        from beingvla.models.vla.config import BeingVLAConfig
        return BeingVLAModel, BeingVLAConfig
    except ImportError:
        raise ImportError(
            "Could not import beingvla.  Install it with:\n"
            "  pip install git+https://github.com/BeingBeyond/Being-H0.git\n"
            "On GPU machines also run:\n"
            "  pip install flash-attn --no-build-isolation"
        )


def _get_device_and_dtype(device: Optional[str]) -> Tuple[torch.device, torch.dtype]:
    """
    Choose device and dtype automatically if not specified.
    Mac (CPU / MPS) → float32.  CUDA → bfloat16.
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    dev = torch.device(device)
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    return dev, dtype


def _resolve_hidden_dim(model, model_name: str) -> int:
    """Extract hidden_dim from model config, with fallback to known values."""
    # Try BeingVLAConfig path
    try:
        cfg = model.config
        # BeingVLAConfig wraps llm_config
        if hasattr(cfg, "llm_config") and hasattr(cfg.llm_config, "hidden_size"):
            return cfg.llm_config.hidden_size
        if hasattr(cfg, "hidden_size"):
            return cfg.hidden_size
    except Exception:
        pass

    # Try language model config
    try:
        lm = model.language_model
        if hasattr(lm, "config") and hasattr(lm.config, "hidden_size"):
            return lm.config.hidden_size
    except Exception:
        pass

    # Fallback: probe by model name substring
    name_lower = model_name.lower()
    for key, dim in _KNOWN_HIDDEN_DIMS.items():
        if key in name_lower:
            warnings.warn(
                f"Could not read hidden_dim from model config; "
                f"falling back to {dim} for '{key}' checkpoint."
            )
            return dim

    # Last resort: forward a dummy batch and check shape
    warnings.warn(
        "Could not determine hidden_dim from config or model name. "
        "Probing with a dummy forward pass (slow, runs once)."
    )
    with torch.no_grad():
        dummy_ids = torch.zeros(1, 4, dtype=torch.long)
        dummy_pv  = torch.zeros(1, 3, 224, 224)
        try:
            out = model(pixel_values=dummy_pv, input_ids=dummy_ids, output_hidden_states=True)
            return out.hidden_states[-1].shape[-1]
        except Exception as e:
            raise RuntimeError(
                f"Cannot determine hidden_dim automatically: {e}. "
                "Pass hidden_dim explicitly to BeingH0Wrapper."
            )


# ---------------------------------------------------------------------------
# Tokeniser helper
# ---------------------------------------------------------------------------

def _build_tokenizer(model_name: str, hf_token: Optional[str]):
    """Load the tokeniser matching the BeingVLA checkpoint."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=hf_token,
    )


def _tokenize_instructions(
    instructions: list[str],
    tokenizer,
    device: torch.device,
    max_length: int = 64,
) -> Dict[str, torch.Tensor]:
    """Tokenise a list of instruction strings into padded input_ids + attention_mask."""
    enc = tokenizer(
        instructions,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {k: v.to(device) for k, v in enc.items()}


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------

class BeingH0Wrapper(nn.Module):
    """
    Trainable wrapper around BeingVLAModel for EgoDex joint-position prediction.

    Forward pass:
        1. Tokenise instruction strings
        2. Run BeingVLAModel.forward() with output_hidden_states=True
        3. Mean-pool the last hidden state over sequence tokens
        4. Pass through JointPredictionHead → (B, T, 50, 3)

    Args:
        model_name:       HuggingFace model ID or local path to checkpoint.
        hf_token:         HuggingFace auth token (needed for private/gated models).
        freeze_backbone:  If True, backbone parameters are frozen; only the head trains.
        pred_horizon:     Number of future frames T to predict.
        num_joints:       Total hand joints (50 = 25 right + 25 left).
        device:           "cuda", "mps", "cpu", or None (auto-detect).
        use_flash_attn:   Enable flash attention (GPU only; set False on Mac).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        hf_token: Optional[str] = None,
        freeze_backbone: bool = True,
        pred_horizon: int = 16,
        num_joints: int = 50,
        device: Optional[str] = None,
        use_flash_attn: bool = False,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.num_joints   = num_joints

        self.dev, self.dtype = _get_device_and_dtype(device)
        use_flash_attn = use_flash_attn and (self.dev.type == "cuda")

        print(f"[BeingH0Wrapper] Loading checkpoint: {model_name}")
        print(f"  device={self.dev}, dtype={self.dtype}, flash_attn={use_flash_attn}")

        # -- Download checkpoint to local cache (beingvla only loads from local dirs) --
        local_model_path = _ensure_local_checkpoint(model_name, hf_token)

        # -- Load BeingVLAModel backbone --
        BeingVLAModel, _ = _try_import_beingvla()
        self.backbone = BeingVLAModel.from_pretrained(
            local_model_path,
            torch_dtype=self.dtype,   # beingvla uses torch_dtype kwarg internally
            low_cpu_mem_usage=True,
            use_flash_attn=use_flash_attn,
            build_motion_model=False,  # not needed for regression head
        ).to(self.dev)

        # -- Load tokeniser --
        self.tokenizer = _build_tokenizer(model_name, hf_token)

        # -- Set img_context_token_id (required before any forward pass) --
        # BeingVLA uses <IMG_CONTEXT> tokens as placeholders in input_ids for image patches.
        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        img_ctx_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        if img_ctx_id == self.tokenizer.unk_token_id or img_ctx_id is None:
            # Token not in vocab — add it and resize embeddings
            self.tokenizer.add_tokens([IMG_CONTEXT_TOKEN], special_tokens=True)
            self.backbone.resize_token_embeddings(len(self.tokenizer))
            img_ctx_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.backbone.img_context_token_id = img_ctx_id

        # -- Resolve hidden dim --
        _hd = hidden_dim or _resolve_hidden_dim(self.backbone, model_name)
        print(f"  hidden_dim={_hd}")

        # -- Optionally freeze backbone --
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()
            print(f"  backbone frozen — only head is trainable")
        else:
            trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
            print(f"  backbone unfrozen — {trainable/1e6:.1f}M trainable params in backbone")

        # -- Regression head (always trainable) --
        self.head = JointPredictionHead(_hd, pred_horizon, num_joints).to(self.dev)
        head_params = sum(p.numel() for p in self.head.parameters())
        print(f"  head trainable params: {head_params/1e3:.1f}K")

    # -----------------------------------------------------------------------

    def _build_prompts(self, instructions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build tokenised input_ids for a batch of instructions.

        BeingVLAModel expects the image tokens embedded in the text prompt as:
            <img><IMG_CONTEXT>×(num_image_token * num_patches)</img>  {instruction}

        We use 1 patch per image (no dynamic tiling in v1).
        Returns (input_ids, attention_mask) each (B, seq_len).
        """
        num_ctx = getattr(self.backbone, "num_image_token", 256)  # default for InternVL
        img_placeholder = "<IMG_CONTEXT>" * num_ctx
        prompts = [f"<img>{img_placeholder}</img> {instr}" for instr in instructions]

        enc = self.tokenizer(
            prompts,
            padding="longest",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(self.dev)
        attention_mask = enc["attention_mask"].to(self.dev)
        return input_ids, attention_mask

    def forward(
        self,
        images: torch.Tensor,
        instructions: list[str],
        camera_K: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            images:       (B, 3, H, W) float tensor, ImageNet-normalised.
            instructions: List of B task-description strings.
            camera_K:     (B, 3, 3) intrinsics (unused in v1, reserved for v2).

        Returns:
            dict with key "pred_joints": (B, T, 50, 3) in metres.
        """
        B = images.shape[0]
        # pixel_values: BeingVLAModel treats each row as one patch.
        # With 1 patch per image and batch size B → (B, 3, H, W)
        pixel_values = images.to(self.dev, dtype=self.dtype)

        # image_flags: 1 per patch, all active (shape: (B,))
        image_flags = torch.ones(B, dtype=torch.long, device=self.dev)

        # Tokenise: embed <IMG_CONTEXT> placeholders so the model knows patch positions
        input_ids, attention_mask = self._build_prompts(instructions)

        # Forward through backbone with hidden states
        outputs = self.backbone(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            output_hidden_states=True,
            return_dict=True,
        )

        # Mean-pool last hidden state over non-padding tokens → (B, D)
        last_hidden = outputs.hidden_states[-1]              # (B, S, D)
        mask        = attention_mask.unsqueeze(-1).float()   # (B, S, 1)
        features    = (last_hidden.float() * mask).sum(1) / mask.sum(1).clamp(min=1)

        # Predict joint positions through regression head
        pred_joints = self.head(features)  # (B, T, 50, 3)
        return {"pred_joints": pred_joints}

    # -----------------------------------------------------------------------

    def trainable_parameters(self):
        """Yields only the parameters that require gradients."""
        return (p for p in self.parameters() if p.requires_grad)

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def load_being_h0(
    model_name: str,
    hf_token: Optional[str] = None,
    dry_run: bool = False,
    freeze_backbone: bool = True,
    pred_horizon: int = 16,
    device: Optional[str] = None,
) -> BeingH0Wrapper:
    """
    Factory function used by train.py.

    In dry_run mode the 1B checkpoint is loaded (unless model_name was
    explicitly overridden by the user) so the full model loading path is
    tested on a MacBook without requiring the larger 8B weights.
    """
    if dry_run and model_name == DEFAULT_MODEL:
        model_name = DRY_RUN_MODEL
        print(f"[load_being_h0] dry_run=True → using 1B checkpoint: {model_name}")

    # On Mac, disable flash attention automatically
    use_flash_attn = torch.cuda.is_available()

    return BeingH0Wrapper(
        model_name=model_name,
        hf_token=hf_token,
        freeze_backbone=freeze_backbone,
        pred_horizon=pred_horizon,
        device=device,
        use_flash_attn=use_flash_attn,
    )
