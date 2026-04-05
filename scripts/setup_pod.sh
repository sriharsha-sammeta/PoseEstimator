#!/bin/bash
# =============================================================
# Runpod Pod Setup Script
# Run this once after creating a new pod attached to a volume.
# Assumes: /workspace is the mounted network volume.
# =============================================================
set -e

echo "=========================================="
echo "  PoseEstimator — Pod Setup"
echo "=========================================="

# ---- 1. Clone repo (if not already present) ----
if [ ! -d /workspace/PoseEstimator ]; then
    echo "[1/5] Cloning repo..."
    git clone https://github.com/sriharsha-sammeta/PoseEstimator.git /workspace/PoseEstimator
else
    echo "[1/5] Repo already exists, pulling latest..."
    cd /workspace/PoseEstimator && git pull origin main 2>/dev/null || true
fi
cd /workspace/PoseEstimator

# ---- 2. Install Python dependencies ----
echo "[2/5] Installing Python dependencies..."
pip install -q -r requirements.txt 2>&1 | tail -3
pip install -q einx timm opencv-python-headless 2>&1 | tail -3

# ---- 3. Clone beingvla (if not already present) ----
if [ ! -d third_party/Being-H0 ]; then
    echo "[3/5] Cloning Being-H0 repo..."
    git clone --depth=1 https://github.com/BeingBeyond/Being-H0.git third_party/Being-H0
else
    echo "[3/5] Being-H0 repo already exists, skipping clone..."
fi

# ---- 4. Patch beingvla internvl_adapter.py ----
echo "[4/5] Patching beingvla..."
ADAPTER_FILE="third_party/Being-H0/beingvla/models/vlm/internvl_adapter.py"
if grep -q "# Convert to target dtype if needed" "$ADAPTER_FILE" 2>/dev/null; then
    python3 << 'PATCH_EOF'
import pathlib, re

p = pathlib.Path("third_party/Being-H0/beingvla/models/vlm/internvl_adapter.py")
code = p.read_text()

# The bug: the raise NotImplementedError for unknown arch is inside the
# dtype-conversion else branch instead of an arch else branch.
# Find the section between "model = Qwen2ForCausalLM" and "return model"
# and restructure it.

old_pattern = (
    "            model = Qwen2ForCausalLM(llm_config_obj)\n"
    "        \n"
    "        # Convert to target dtype if needed\n"
    "        if self.target_dtype != torch.float32:\n"
    "            model = model.to(self.target_dtype)\n"
    "        else:\n"
    "            raise NotImplementedError"
)

new_pattern = (
    "            model = Qwen2ForCausalLM(llm_config_obj)\n"
    "\n"
    "        else:\n"
    "            raise NotImplementedError"
    "(f'{arch} is not implemented. BeingVLA currently supports LlamaForCausalLM and Qwen2ForCausalLM.')\n"
    "\n"
    "        # Convert to target dtype if needed\n"
    "        if self.target_dtype != torch.float32:\n"
    "            model = model.to(self.target_dtype)"
)

if old_pattern in code:
    code = code.replace(old_pattern, new_pattern)
    p.write_text(code)
    print("  Patch applied successfully.")
else:
    print("  Pattern not found (may already be patched or file structure changed).")
    # Try a more lenient match
    if "# Convert to target dtype" in code and "raise NotImplementedError" in code:
        print("  WARNING: File contains both markers but exact pattern didn't match.")
        print("  Please verify the patch manually.")
PATCH_EOF
else
    echo "  Already patched (or file structure changed)."
fi

# ---- 5. Install flash-attn (GPU only) ----
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    if python3 -c "import flash_attn" 2>/dev/null; then
        echo "[5/5] flash-attn already installed."
    else
        echo "[5/5] Installing flash-attn (this takes a few minutes)..."
        pip install flash-attn --no-build-isolation 2>&1 | tail -3
    fi
else
    echo "[5/5] No CUDA — skipping flash-attn."
fi

# ---- Verify ----
echo ""
echo "=== Verification ==="
python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
import sys; sys.path.insert(0, 'third_party/Being-H0')
from beingvla.models.vla.being_vla_model import BeingVLAModel
print('beingvla: OK')
try:
    import flash_attn; print(f'flash-attn: {flash_attn.__version__}')
except: print('flash-attn: not installed')
import h5py, wandb, tqdm, transformers
print('All deps: OK')
"

echo ""
echo "=========================================="
echo "  Setup complete!"
echo "=========================================="

# ---- Create directories on the persistent volume ----
mkdir -p /workspace/egodex
mkdir -p /workspace/results
mkdir -p /workspace/huggingface_cache
export HF_HOME=/workspace/huggingface_cache

echo "Persistent dirs created:"
echo "  /workspace/egodex/       ← dataset storage"
echo "  /workspace/results/      ← checkpoints & logs"
echo "  /workspace/huggingface_cache/ ← model weight cache"
echo ""
echo "Set HF_HOME=/workspace/huggingface_cache to cache models on volume."
