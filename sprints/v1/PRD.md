# Sprint v1 — PRD: Being-H0 × EgoDex Pick-and-Place Finetuning Pipeline

## Overview

Build a supervised finetuning pipeline (`train.py`) that loads the pretrained **Being-H0** VLA model, filters the **EgoDex** dataset to pick-and-place sequences, and trains the model to predict future hand pose sequences from egocentric images and language instructions. The first milestone is a clean baseline + finetuning loop with W&B logging that runs end-to-end on a MacBook (dry run) and on a single Runpod GPU (full training).

---

## Goals

- **Done** = `train.py --dry_run` completes 5 steps + 1 val pass on a MacBook with metrics printed and W&B logged
- **Done** = `train.py --eval_only` runs the raw pretrained Being-H0 on EgoDex pick-and-place val split and prints baseline metrics
- **Done** = `train.py` trains for N epochs on Runpod, checkpoints best model, logs loss curves to W&B
- **Done** = Pick-and-place subset is automatically filtered from EgoDex `llm_description` metadata; sample counts are printed
- **Done** = `README.md` explains full setup, all modes, and how to reproduce

---

## User Stories

- As a researcher, I want to run `python train.py --eval_only` and see baseline metrics for Being-H0 on EgoDex pick-and-place data, so I know what I'm improving over.
- As a researcher, I want to run `python train.py --dry_run` on my MacBook and have the full pipeline verified in under 2 minutes, so I can catch bugs before launching an expensive GPU job.
- As a researcher, I want to run `python train.py` on Runpod with a single command and have training logged to W&B automatically, so I can monitor progress remotely.
- As a researcher, I want the pick-and-place subset size printed at startup, so I can verify filtering is working correctly.
- As a researcher, I want to resume training from a checkpoint with `--resume_from`, so I don't lose progress if a GPU job is interrupted.

---

## Technical Architecture

### Tech Stack

| Component | Choice | Notes |
|---|---|---|
| Model | Being-H0 (`BeingBeyond/Being-H0-8B-2508`) | VLA transformer, class `BeingVLAModel` in `beingvla` package |
| Dataset | EgoDex | HDF5 + MP4 via Apple CDN; `h5py` + `decord` |
| Training | PyTorch (native loop) | No Trainer abstraction — keeps it debuggable |
| Logging | Weights & Biases (`wandb`) | Optional; skipped if not configured |
| Setup | Python venv + pip | `python3.10 -m venv .venv` |
| Python | **3.10** | Required by `beingvla` / `environment.yml` |

### Being-H0 Architecture (key facts)

- GitHub: `https://github.com/BeingBeyond/Being-H0` — package name `beingvla`
- Model class: `BeingVLAModel` — inherits `PreTrainedModel` + training/generation mixins
- Backbone: InternVL (vision) + Qwen (language)
- Native output: MANO hand pose parameters via GRVQ-8K motion tokeniser (discrete codes)
- **Our pipeline**: bypasses motion tokeniser; uses `output_hidden_states=True` → regression head
- `BeingVLAModel.forward()` supports `output_hidden_states=True` → `outputs.hidden_states[-1]`
- Checkpoints: 1B (dry run), 8B (default: `BeingBeyond/Being-H0-8B-2508`), 14B
- **Training scripts**: listed as "planned for future implementation" in the repo — we implement our own loop
- `training_mixins.py` exists with cross-entropy motion token loss (for v2 native finetuning)

### EgoDex Dataset Format (key facts)

- **Scale**: 829 hours @ 30 Hz, 1080p egocentric video
- **Splits**: train (~725h, 5 zip files ≈ 1.5TB), test (16GB), additional (200GB)
- **Per-sequence files**: `{idx}.hdf5` (annotations) + `{idx}.mp4` (video)
- **HDF5 structure**:
  ```
  f['transforms']         # shape: (N, 68, 4, 4)  — SE(3) transforms for 68 joints at 30Hz
  f['confidences']        # shape: (N, 68)          — ARKit confidence 0-1
  f.attrs['llm_description']   # str — GPT-4-generated task description
  f.attrs['llm_description2']  # str — second description (for reversible tasks)
  f['camera']             # (3, 3) intrinsic matrix
  ```
- **Hand joints**: indices 18–42 (right, 25 joints) and 43–67 (left, 25 joints) within the 68-joint hierarchy — **both hands used as prediction target**
- **Language filtering**: `llm_description` contains free-text task descriptions — filter by pick/place keywords

### Representation Alignment: EgoDex → Being-H0

Being-H0 outputs MANO parameters; EgoDex stores SE(3) joint transforms. Alignment strategy for v1:

1. **Extract 3D joint positions** from EgoDex SE(3) matrices (take the translation column: `transforms[:, 18:68, :3, 3]`)
2. **Target format**: `(T, 50, 3)` — T future frames, 50 hand joints (right 0–24 = EgoDex joints 18–42, left 25–49 = EgoDex joints 43–67), 3D position
3. **Prediction format**: run Being-H0's MANO forward pass to get joint positions from predicted MANO params for both hands
4. **Loss**: MSE between predicted and target joint positions (in meters)
5. **Main checkpoint metric**: `val_mpjpe` (mean per-joint position error, mm)

This avoids the need for MANO parameter fitting in v1. Note in README that a v2 could add MANO fitting for tighter alignment.

### Data Flow

```
EgoDex HDF5 files
    │
    ▼
EgoDexDataset (PyTorch Dataset)
    │  filter by llm_description keywords → pick-and-place subset
    │  load: RGB frame(s) + language instruction + future hand joint positions (T, 25, 3)
    ▼
DataLoader (batch_size, num_workers)
    │
    ▼
Being-H0 model (BeingHPolicy)
    │  inputs:  RGB image tensor, tokenized instruction
    │  outputs: MANO params → 3D joint positions (T, 25, 3)
    ▼
Loss: MSE(pred_joints, gt_joints)
    │
    ▼
Optimizer (AdamW, cosine LR schedule)
    │
    ▼
Metrics: val_mpjpe, val_mean_l2, val_final_step_error
    │
    ▼
W&B logging + checkpoint saving
```

### Component Diagram

```
train.py
├── parse_args()                  ← all CLI flags
├── setup_wandb()                 ← optional W&B init
├── load_model()                  ← Being-H0 via BeingHPolicy, HF download
│   └── handle_gated_model()     ← clear error on auth failure
├── EgoDexDataset                 ← dataset/egodex_dataset.py
│   ├── scan_sequences()         ← index HDF5 files without loading video
│   ├── filter_pick_place()      ← keyword filter on llm_description
│   └── __getitem__()            ← load frame + poses from HDF5/MP4
├── train_loop()                  ← N epochs × steps
│   ├── forward + loss
│   ├── backward + optimizer step
│   └── log to W&B
├── eval_loop()                   ← called every --eval_every steps
│   ├── compute_mpjpe()
│   ├── compute_mean_l2()
│   ├── compute_final_step_error()
│   └── save best checkpoint
└── main()                        ← orchestration, dry_run / eval_only modes
```

---

## Out of Scope (v1)

- MANO parameter fitting / inverse kinematics from EgoDex joint positions
- Robot control, simulation, retargeting
- Multi-GPU / DDP training (mentioned in README but not implemented)
- Data augmentation beyond basic normalization
- Custom tokenizer training or motion tokenizer finetuning
- Streaming / partial video decoding optimization
- v2+ features: confidence-weighted loss, temporal attention, multi-frame context stacking

---

## Dependencies

### External

- `BeingBeyond/Being-H0-8B-2508` on Hugging Face (public)
- `beingvla` package: `pip install git+https://github.com/BeingBeyond/Being-H0.git`
- EgoDex download: Apple CDN (test split ~16GB; dry run needs zero data)
- Python **3.10** (required by `beingvla` / `environment.yml`)
- `decord` for video frame decoding (`cv2` fallback)

### Python packages (key)

```
torch>=2.1
torchvision
torchcodec          # EgoDex video loading
h5py                # EgoDex HDF5 reading
transformers        # Being-H0 backbone dependencies
wandb
tqdm
numpy
Pillow
```

### EgoDex Download Strategy

The full EgoDex training set is ~1.5TB. For v1, we support three modes:

| Mode | What to download | How |
|---|---|---|
| Dry run | Nothing — synthetic data | `--dry_run` generates random tensors matching real shapes |
| Metadata inspection | HDF5 files only (no MP4) | `--metadata_only` scans `llm_description` without video |
| Subset download | Test split only (16GB) | `curl "https://ml-site.cdn-apple.com/datasets/egodex/test.zip" -o test.zip` |
| Full training | All 5 train zips (~1.5TB) | Only needed for real Runpod training |

Note: EgoDex does not support streaming or per-shard partial download. The test split (16GB) is the lightest real-data option for eval-only mode.

---

## Metrics

| Metric | Description | Unit | Primary? |
|---|---|---|---|
| `val_mpjpe` | Mean per-joint position error across all hand joints and timesteps | mm | **YES — checkpoint selection** |
| `val_mean_l2` | Mean L2 distance across all predicted keypoints | mm | secondary |
| `val_final_step_error` | MPJPE at the last predicted timestep only | mm | diagnostic |
| `val_loss` | MSE loss on validation set | — | logged |
| `val_pct_within_10mm` | % of keypoints within 10mm of ground truth | % | optional |

---

## Key Technical Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Being-H0 input format may not match expected EgoDex image shape/normalization | Read Being-H0 `data_config` carefully; add normalization transform matching training config |
| MANO joint ordering in Being-H0 may differ from EgoDex joint ordering | Document joint index mapping explicitly in `egodex_dataset.py`; add assertion |
| EgoDex `llm_description` may not cleanly separate pick-and-place | Use keyword list; log filtered count; note limitation in README |
| Being-H0 forward pass may fail without full observation dict (camera params etc.) | Wrap in try/except; document required observation keys |
| torchcodec install may fail on older macOS | Provide fallback using `decord` or `cv2` for video loading |
