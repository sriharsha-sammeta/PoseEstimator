# Runpod Setup & Training Instructions

Step-by-step guide to go from a blank Runpod GPU pod to a running Being-H0 × EgoDex finetuning job.

---

## 1. Pod Setup

**Recommended pod specs:**
- GPU: A100 40GB (or H100 for faster training)
- Disk: 200GB minimum (50GB for model + 16GB for EgoDex test split + workspace)
- For full training with all EgoDex splits: 2TB disk
- Template: PyTorch 2.1+ (any recent CUDA image with Python 3.10+)

---

## 2. Clone the Repo

```bash
cd /workspace
git clone <your-repo-url> PoseEstimator
cd PoseEstimator
```

Or if uploading manually:
```bash
cd /workspace
# upload/scp the PoseEstimator folder here
cd PoseEstimator
```

---

## 3. Create Environment & Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Core dependencies
pip install -r requirements.txt

# beingvla (no setup.py — clone manually)
mkdir -p third_party
git clone --depth=1 https://github.com/BeingBeyond/Being-H0.git third_party/Being-H0

# GPU-only: flash attention (significantly speeds up the backbone)
pip install flash-attn --no-build-isolation

# Linux-only: fast video decoding
pip install decord

# Fix a bug in beingvla (float32 dtype check crashes on CPU path)
# This sed command patches the issue in internvl_adapter.py
cd third_party/Being-H0
python3 -c "
import pathlib
p = pathlib.Path('beingvla/models/vlm/internvl_adapter.py')
code = p.read_text()
old = '''            model = Qwen2ForCausalLM(llm_config_obj)
        
        # Convert to target dtype if needed
        if self.target_dtype != torch.float32:
            model = model.to(self.target_dtype)
        else:
            raise NotImplementedError(f'{arch} is not implemented. BeingVLA currently supports LlamaForCausalLM and Qwen2ForCausalLM.')
            
        return model'''
new = '''            model = Qwen2ForCausalLM(llm_config_obj)

        else:
            raise NotImplementedError(f'{arch} is not implemented. BeingVLA currently supports LlamaForCausalLM and Qwen2ForCausalLM.')

        # Convert to target dtype if needed (float32 is fine as-is on CPU/MPS)
        if self.target_dtype != torch.float32:
            model = model.to(self.target_dtype)

        return model'''
if old in code:
    p.write_text(code.replace(old, new))
    print('Patch applied successfully.')
else:
    print('Patch already applied or file structure changed.')
"
cd /workspace/PoseEstimator
```

---

## 4. Set API Keys

```bash
# HuggingFace token (required — model won't load without it)
export HF_TOKEN=<your_hf_token>

# Weights & Biases (optional but recommended for remote monitoring)
export WANDB_API_KEY=<your_wandb_key>
```

---

## 5. Verify Setup (Dry Run on GPU with Real 8B Model)

This runs 5 training steps with synthetic data using the **real 8B model** to verify
GPU, flash attention, and the full-size model all work before downloading EgoDex data.
First run downloads ~16GB of model weights (cached for all subsequent runs).

```bash
source .venv/bin/activate
python train.py \
    --dry_run \
    --model_name BeingBeyond/Being-H0-8B-2508 \
    --output_dir ./runs/gpu_dry_run
```

**Expected output:**
```
[dry_run] Overrides: batch_size=2, max_steps=5, eval_every=5
[auth] HF token found in $HF_TOKEN
[SyntheticEgoDexDataset] dry-run mode: 32 synthetic samples, T=16, img=448x448
[BeingH0Wrapper] Loading checkpoint: BeingBeyond/Being-H0-8B-2508
  device=cuda, dtype=torch.bfloat16, flash_attn=True
  Downloading checkpoint from HuggingFace: BeingBeyond/Being-H0-8B-2508
  (this may take a while on first run — weights are cached afterwards)
  Fetching XX files: 100%|██████████| ...
  Checkpoint cached at: /root/.cache/huggingface/hub/models--BeingBeyond--Being-H0-8B-2508/...
  hidden_dim=4096
  backbone frozen — only head is trainable
  head trainable params: 12546.0K

[train] Starting: total_steps=5, ...
  step     1/5  ...  loss=0.01xx
  step     2/5  ...
  step     3/5  ...
  step     4/5  ...
  step     5/5  ...
  [eval @ step 5]  val_loss=0.01xx  val_mpjpe=~190 mm  val_final_step=~190 mm
  ★ new best val_mpjpe=... mm

[train] Done. best val_mpjpe=... mm → runs/gpu_dry_run/best_model.pt
```

**What to check:**
- `device=cuda` — confirms GPU is being used
- `dtype=torch.bfloat16` — confirms efficient GPU dtype
- `flash_attn=True` — confirms flash attention is active
- `hidden_dim=4096` — confirms the 8B model loaded (not 1B which has 896)
- `head trainable params: 12546.0K` — larger head than 1B because hidden_dim is 4096
- 5 steps complete without error
- Checkpoint file exists: `ls -la runs/gpu_dry_run/best_model.pt`

If this fails, debug here before proceeding.

---

## 6. Download EgoDex Data

### Option A: Test split only (16 GB) — for eval-only baseline

```bash
mkdir -p /workspace/egodex
cd /workspace/egodex
curl -L "https://ml-site.cdn-apple.com/datasets/egodex/test.zip" -o test.zip
unzip test.zip -d test
cd /workspace/PoseEstimator
```

### Option B: Full training data (~1.5 TB) — for finetuning

```bash
mkdir -p /workspace/egodex
cd /workspace/egodex
for i in 0 1 2 3 4; do
  echo "Downloading train_${i}.zip ..."
  curl -L "https://ml-site.cdn-apple.com/datasets/egodex/train_${i}.zip" -o "train_${i}.zip"
  unzip "train_${i}.zip" -d train
  rm "train_${i}.zip"   # delete zip after extracting to save disk
done
cd /workspace/PoseEstimator
```

---

## 7. Run Eval-Only Baseline (Raw Pretrained Model)

This evaluates the **raw pretrained 8B Being-H0** on EgoDex pick-and-place data with no training.
Gives you the baseline numbers to improve over.

```bash
source .venv/bin/activate
python train.py \
    --eval_only \
    --model_name BeingBeyond/Being-H0-8B-2508 \
    --dataset_root /workspace/egodex/test \
    --output_dir ./runs/baseline_8b \
    --task_filter pick_place \
    --batch_size 4
```

**Expected output:**
```
[EgoDexDataset] Pick-and-place filter (train): N / M filtered / K total sequences
[EgoDexDataset] Pick-and-place filter (val):   ...
[BeingH0Wrapper] Loading checkpoint: BeingBeyond/Being-H0-8B-2508
  device=cuda, dtype=torch.bfloat16, flash_attn=True
  hidden_dim=4096
  backbone frozen — only head is trainable

[eval_only] Evaluating raw pretrained model on val split...
[eval_only] Results:
  val_mpjpe:             XXX.X mm       <-- record this as your baseline
  val_mean_l2:           XXX.X mm
  val_final_step_error:  XXX.X mm
  val_pct_within_10mm:   X.X %
  val_loss:              X.XXX
```

**Save these numbers.** After finetuning, val_mpjpe should be lower.

---

## 8. Run Finetuning

### 8a. Train on test split (quick experiment, 16 GB data)

Good for verifying training actually reduces the loss before committing to the full dataset.

```bash
source .venv/bin/activate
python train.py \
    --model_name BeingBeyond/Being-H0-8B-2508 \
    --dataset_root /workspace/egodex/test \
    --output_dir ./runs/finetune_test_split \
    --batch_size 8 \
    --num_workers 4 \
    --epochs 5 \
    --learning_rate 1e-4 \
    --weight_decay 1e-2 \
    --eval_every 100 \
    --save_every 500 \
    --task_filter pick_place \
    --wandb_project being_h0_egodex \
    --wandb_run_name finetune_8b_test_split
```

### 8b. Train on full training data (~1.5 TB)

```bash
source .venv/bin/activate
python train.py \
    --model_name BeingBeyond/Being-H0-8B-2508 \
    --dataset_root /workspace/egodex/train \
    --output_dir ./runs/finetune_full \
    --batch_size 8 \
    --num_workers 4 \
    --epochs 10 \
    --learning_rate 1e-4 \
    --weight_decay 1e-2 \
    --eval_every 500 \
    --save_every 1000 \
    --task_filter pick_place \
    --wandb_project being_h0_egodex \
    --wandb_run_name finetune_8b_full_v1
```

**Expected output during training:**
```
[EgoDexDataset] Pick-and-place filter (train): XXXX / ... sequences
[BeingH0Wrapper] Loading checkpoint: BeingBeyond/Being-H0-8B-2508
  device=cuda, dtype=torch.bfloat16, flash_attn=True
  hidden_dim=4096
  head trainable params: 12546.0K

[train] Starting: total_steps=XXXX, epochs=10, trainable_params=12.55M
  step     1/XXXX  epoch 1/10  loss=0.XXXX  lr=1.00e-04
  step    10/XXXX  ...  loss=0.XXXX
  ...
  [eval @ step 500]  val_loss=0.XXXX  val_mpjpe=XXX.X mm
  ★ new best val_mpjpe=XXX.X mm
  [checkpoint] saved → runs/finetune_full/best_model.pt
  ...
```

**What to monitor in W&B:**
- `train/loss` should decrease over time
- `val_mpjpe` should decrease (lower = better)
- `train/lr` should show cosine decay with warmup

---

## 9. Resume Training (if interrupted)

If the pod dies or you need to restart:

```bash
source .venv/bin/activate
python train.py \
    --model_name BeingBeyond/Being-H0-8B-2508 \
    --dataset_root /workspace/egodex/train \
    --output_dir ./runs/finetune_full \
    --resume_from ./runs/finetune_full/step_005000.pt \
    --batch_size 8 \
    --num_workers 4 \
    --epochs 10 \
    --learning_rate 1e-4 \
    --eval_every 500 \
    --save_every 1000 \
    --task_filter pick_place \
    --wandb_project being_h0_egodex \
    --wandb_run_name finetune_8b_full_v1_resumed
```

---

## 10. Evaluate Finetuned Model

After training completes, evaluate the best checkpoint:

```bash
source .venv/bin/activate
python train.py \
    --eval_only \
    --model_name BeingBeyond/Being-H0-8B-2508 \
    --dataset_root /workspace/egodex/test \
    --output_dir ./runs/eval_finetuned \
    --resume_from ./runs/finetune_full/best_model.pt \
    --task_filter pick_place \
    --batch_size 4
```

**Compare against baseline (step 7):**
```
                          Baseline    Finetuned
val_mpjpe (mm):           XXX.X       XXX.X      <-- should be lower
val_final_step (mm):      XXX.X       XXX.X
val_pct_within_10mm (%):  X.X         X.X        <-- should be higher
```

---

## 11. Download Results

Before terminating the pod, download your checkpoints and logs:

```bash
# Tar up the best checkpoint and training logs
cd /workspace/PoseEstimator
tar czf results.tar.gz runs/

# Download via Runpod UI, or scp to your machine:
# scp -P <port> root@<pod-ip>:/workspace/PoseEstimator/results.tar.gz .
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'beingvla'` | `third_party/Being-H0` not cloned. Run step 3 again. |
| `No module named 'einx'` | `pip install einx timm` |
| `CUDA out of memory` | Reduce `--batch_size` to 4 or 2 |
| `Model download failed / 401 / 403` | Check `echo $HF_TOKEN` is set correctly |
| `No checkpoint found, model initialized from config only` | HF_TOKEN is missing or wrong. Re-export it. |
| `FlashAttention2 is not installed` | Run `pip install flash-attn --no-build-isolation` (needs CUDA toolkit) |
| `No .hdf5 files found` | Check `--dataset_root` path. Run `find /workspace/egodex -name "*.hdf5" | head` |
| W&B not logging | Check `echo $WANDB_API_KEY` or run `wandb login` |
| Training loss not decreasing | Expected for first few hundred steps with frozen backbone. If flat after 1000 steps, try `--unfreeze_backbone` |

---

## Quick Reference: All Commands in Order

```bash
# --- ONE-TIME SETUP ---
cd /workspace
git clone <repo> PoseEstimator && cd PoseEstimator
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -r requirements.txt
mkdir -p third_party && git clone --depth=1 https://github.com/BeingBeyond/Being-H0.git third_party/Being-H0
pip install flash-attn --no-build-isolation
pip install decord
export HF_TOKEN=<your_hf_token>
export WANDB_API_KEY=<your_key>

# --- VERIFY (real 8B model, synthetic data) ---
python train.py --dry_run --model_name BeingBeyond/Being-H0-8B-2508 --output_dir ./runs/gpu_dry_run

# --- DOWNLOAD DATA ---
mkdir -p /workspace/egodex && cd /workspace/egodex
curl -L "https://ml-site.cdn-apple.com/datasets/egodex/test.zip" -o test.zip && unzip test.zip -d test
cd /workspace/PoseEstimator

# --- BASELINE ---
python train.py --eval_only --model_name BeingBeyond/Being-H0-8B-2508 --dataset_root /workspace/egodex/test --output_dir ./runs/baseline_8b --batch_size 4

# --- TRAIN ---
python train.py --model_name BeingBeyond/Being-H0-8B-2508 --dataset_root /workspace/egodex/test --output_dir ./runs/finetune_v1 --batch_size 8 --epochs 5 --eval_every 100 --save_every 500 --wandb_project being_h0_egodex

# --- EVAL FINETUNED ---
python train.py --eval_only --resume_from ./runs/finetune_v1/best_model.pt --model_name BeingBeyond/Being-H0-8B-2508 --dataset_root /workspace/egodex/test --output_dir ./runs/eval_finetuned --batch_size 4
```
