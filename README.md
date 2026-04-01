# Being-H0 × EgoDex Pick-and-Place Finetuning

Supervised finetuning pipeline for predicting hand pose trajectories from egocentric images and language instructions, using **Being-H0** as the pretrained backbone and **EgoDex** as the dataset.

**Task:** `image + language instruction → future hand pose sequence (T=16 frames, 50 joints)`

---

## Quickstart

```bash
# 1. Create venv  (Python 3.10 required — matches beingvla's environment.yml)
python3.10 -m venv .venv
source .venv/bin/activate

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Being-H0 package from source
pip install git+https://github.com/BeingBeyond/Being-H0.git

# 5. (GPU machines only) Flash attention
pip install flash-attn --no-build-isolation

# 6. (Optional) MANO hand model — only needed for --eval_only native inference mode
#    Download MANO model files from https://mano.is.tue.mpg.de/ (requires free registration)
#    and place in:  beingvla/models/motion/mano/
pip install git+https://github.com/lixiny/manotorch.git
pip install git+https://github.com/mattloper/chumpy.git
```

---

## Hugging Face Auth

Being-H0 checkpoints are hosted at `BeingBeyond/Being-H0-*` on Hugging Face.
If the model is public, no token is needed. If it becomes gated:

```bash
# Option 1: environment variable (recommended)
export HF_TOKEN=hf_your_token_here

# Option 2: huggingface-cli
huggingface-cli login

# Then pass the env var name to train.py
python train.py --hf_token_env HF_TOKEN ...
```

If download fails with a 401/403 error, train.py will print a clear message pointing here.

---

## W&B Setup

```bash
# One-time login
wandb login          # or: export WANDB_API_KEY=your_key

# Use in training
python train.py --wandb_project being_h0_egodex --wandb_run_name run_01 ...
```

Omit `--wandb_project` to disable W&B logging entirely.

---

## MacBook Dry Run

Tests the entire pipeline end-to-end with **synthetic data** and the **1B checkpoint**.
No EgoDex data download required.

```bash
python train.py \
    --dry_run \
    --output_dir ./runs/dry_run
```

Expected output:
- Loads Being-H0 1B checkpoint (~2GB download on first run)
- Prints synthetic dataset sizes (train=32, val=8)
- Runs 5 training steps, printing loss at each step
- Runs 1 validation pass, printing `val_mpjpe`, `val_mean_l2`, `val_final_step_error`
- Saves best checkpoint to `./runs/dry_run/best_model.pt`
- Completes in under 2 minutes on M-series Mac (CPU/MPS)

---

## Eval-Only Baseline (Raw Pretrained Model)

Evaluate the **raw pretrained Being-H0** on EgoDex pick-and-place val split
**without any finetuning**. This gives your baseline to improve over.

```bash
# Requires EgoDex test split downloaded (~16 GB)
# See "EgoDex Download" section below

python train.py \
    --eval_only \
    --model_name BeingBeyond/Being-H0-8B-2508 \
    --dataset_root /data/egodex/test \
    --output_dir   ./runs/baseline \
    --task_filter  pick_place \
    --batch_size   4
```

Prints a metric table like:
```
val_loss:              0.047
val_mpjpe:           143.2 mm
val_mean_l2:         143.2 mm
val_final_step_error: 171.4 mm
val_pct_within_10mm:    2.1 %
```

> **What this measures**: pretrained backbone representations + randomly-initialised
> regression head, on EgoDex pick-and-place. After finetuning, you should see
> `val_mpjpe` drop significantly.

---

## Single-GPU Runpod Training

```bash
python train.py \
    --model_name   BeingBeyond/Being-H0-8B-2508 \
    --dataset_root /data/egodex/train \
    --output_dir   ./runs/finetune_v1 \
    --batch_size   8 \
    --num_workers  4 \
    --epochs       10 \
    --learning_rate 1e-4 \
    --weight_decay  1e-2 \
    --eval_every   500 \
    --save_every   1000 \
    --task_filter  pick_place \
    --wandb_project being_h0_egodex \
    --wandb_run_name finetune_8b_v1
```

**Recommended GPU**: A100 40GB or H100 for 8B model. RTX 4090 may work with `--batch_size 4`.

To also finetune the backbone (not just the head):
```bash
python train.py ... --unfreeze_backbone
```

To resume from a checkpoint:
```bash
python train.py ... --resume_from ./runs/finetune_v1/step_005000.pt
```

---

## Multi-GPU Runpod (Optional — v2)

> **Note**: Multi-GPU training with DDP/FSDP is **not implemented in v1**.
> Being-H0's official training code uses FSDP via torchrun.
> The following is a reference command for v2:

```bash
torchrun \
    --nproc_per_node 4 \
    --master_port 29500 \
    train.py \
    --model_name BeingBeyond/Being-H0-8B-2508 \
    --dataset_root /data/egodex/train \
    --output_dir   ./runs/finetune_multigpu \
    --batch_size   4 \
    --epochs       10 \
    ...
```

---

## EgoDex Download

EgoDex **does not support streaming or partial downloads** per shard.
Available splits and sizes:

| Split | Files | Size | Use case |
|---|---|---|---|
| `test`  | `test.zip`             | ~16 GB   | eval-only baseline, quick experiments |
| `train` | `train_0.zip` … `train_4.zip` | ~1.5 TB total | full training |
| `additional` | `additional.zip` | ~200 GB  | extra data |

```bash
# Test split (recommended first download — 16 GB)
curl "https://ml-site.cdn-apple.com/datasets/egodex/test.zip" -o test.zip
unzip test.zip -d /data/egodex/test

# Full training data (1.5 TB — only for Runpod)
for i in 0 1 2 3 4; do
  curl "https://ml-site.cdn-apple.com/datasets/egodex/train_${i}.zip" -o "train_${i}.zip"
  unzip "train_${i}.zip" -d /data/egodex/train
done
```

**Dry run requires zero data download** — synthetic tensors are used instead.

---

## Pick-and-Place Subset Filtering

EgoDex does **not** expose structured task-type labels. Each sequence has a
`llm_description` attribute (GPT-4 generated) stored in the HDF5 file.

Our filter matches any sequence whose description contains one of these keywords:

```
"pick up", "pick", "place", "grasp", "put down", "set down",
"grab", "lift", "transfer", "move", "carry"
```

The filter is applied in `dataset/egodex_dataset.py: filter_pick_place()`.
Sample counts are printed at startup:

```
[EgoDexDataset] Pick-and-place filter (train): 4823 / 6120 filtered / 10240 total sequences
```

**Limitation**: This is a heuristic — some non-pick-and-place sequences may pass
the filter (false positives), and some pick-and-place sequences may be excluded
(false negatives). To use a different filter, pass `--task_filter all` or
customise the keyword list in `dataset/egodex_dataset.py: DEFAULT_PICK_PLACE_KEYWORDS`.

---

## Metrics Explained

| Metric | Description | Unit | Role |
|---|---|---|---|
| `val_mpjpe` | Mean Per-Joint Position Error across all 50 joints × 16 timesteps | mm | **Primary — checkpoint selection** |
| `val_mean_l2` | Mean L2 over all predicted joints (same as MPJPE here) | mm | Secondary |
| `val_final_step_error` | MPJPE at the last predicted frame (t+16) | mm | Diagnostic |
| `val_loss` | MSE loss on validation set | — | Training signal |
| `val_pct_within_10mm` | % of joint predictions within 10 mm of ground truth | % | Optional quality gate |

**Lower is better** for all error metrics. **Higher is better** for `val_pct_within_Nmm`.

**Joint layout** (`hand_joints` shape: `(T, 50, 3)`):
- Indices `0–24`: right hand (25 joints, EgoDex joints 18–42)
- Indices `25–49`: left hand (25 joints, EgoDex joints 43–67)

**Coordinate system**: metres, in EgoDex camera frame.

---

## Architecture Notes

### Finetuning strategy (v1)

```
BeingVLAModel (pretrained, frozen by default)
    ↓  forward(pixel_values, input_ids, output_hidden_states=True)
    ↓  mean-pool last hidden state  →  (B, hidden_dim)
JointPredictionHead (trainable — LayerNorm → Linear → GELU → Linear → GELU → Linear)
    ↓
(B, T=16, J=50, 3)  predicted joint positions in metres
    ↓
MSE loss vs EgoDex ground truth
```

Being-H0's native **GRVQ-8K motion tokeniser** is **not used** in this pipeline.
We bypass it and train a continuous regression head directly on EgoDex SE(3) joint translations.

This avoids the MANO parameter fitting step (out of scope for v1) and gives a
clean, debuggable baseline. A v2 could use the native motion tokeniser with MANO fitting.

### Model checkpoints

| ID | Params | VRAM | Use |
|---|---|---|---|
| `BeingBeyond/Being-H0-1B-2508`  | ~1B  | ~4 GB  | Dry run (auto-selected) |
| `BeingBeyond/Being-H0-8B-2508`  | ~8B  | ~20 GB | Default training         |
| `BeingBeyond/Being-H0-14B-2508` | ~14B | ~35 GB | Pass via `--model_name`  |

---

## Project Structure

```
PoseEstimator/
├── train.py                    ← main entrypoint
├── requirements.txt
├── README.md
├── dataset/
│   ├── egodex_dataset.py       ← EgoDex HDF5 + MP4 loader, pick-and-place filter
│   └── synthetic_dataset.py    ← random-data stand-in for dry run
├── model/
│   └── being_h0_wrapper.py     ← BeingVLAModel + JointPredictionHead
├── utils/
│   └── metrics.py              ← MPJPE, mean L2, final-step error, pct-within-threshold
└── sprints/
    └── v1/
        ├── PRD.md
        └── TASKS.md
```
