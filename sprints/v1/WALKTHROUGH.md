# Sprint v1 — Walkthrough

## Summary

Built a complete supervised finetuning pipeline for predicting egocentric hand pose trajectories using **Being-H0** (a Vision-Language-Action model) as a pretrained backbone and **EgoDex** (Apple's 829-hour egocentric manipulation dataset) as training data. The pipeline is filtered to pick-and-place sequences, predicts 16 future frames of both-hand joint positions (50 joints × 3D), and was successfully dry-run end-to-end on a MacBook with the real pretrained 1B checkpoint.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  train.py  (orchestrator)                                           │
│                                                                     │
│  ┌──────────────────┐      ┌──────────────────────────────────────┐ │
│  │  EgoDexDataset   │      │         BeingH0Wrapper               │ │
│  │  or              │─────▶│                                      │ │
│  │  SyntheticDataset│      │  ┌─────────────────────────────────┐ │ │
│  └──────────────────┘      │  │  BeingVLAModel  (frozen)        │ │ │
│           │                │  │  InternVL vision + Qwen LM      │ │ │
│           │                │  │  forward(..., output_hs=True)   │ │ │
│   batch:  │                │  │  → hidden_states[-1] (B,S,D)   │ │ │
│   image   │                │  │  → mean-pool → (B, D=896)       │ │ │
│   instr.  │                │  └──────────────┬──────────────────┘ │ │
│   joints  │                │                 │                     │ │
│           │                │  ┌──────────────▼──────────────────┐ │ │
│           │                │  │  JointPredictionHead (trainable)│ │ │
│           │                │  │  LN→Lin→GELU→Lin→GELU→Lin      │ │ │
│           │                │  │  → reshape → (B, 16, 50, 3)    │ │ │
│           │                │  └──────────────────────────────────┘ │ │
│           │                └──────────────────────────────────────┘ │
│           │                           │                              │
│           │         pred_joints       │                              │
│           └──────────────────────────▶│                              │
│                    gt_joints          │                              │
│                         │             │                              │
│                    MSELoss ◀──────────┘                              │
│                         │                                            │
│                    AdamW + cosine LR schedule                        │
│                         │                                            │
│               ┌─────────┴──────────┐                                │
│               │  utils/metrics.py  │   W&B logging                  │
│               │  val_mpjpe ★       │──────────────────────────────▶ │
│               │  val_mean_l2       │                                 │
│               │  val_final_step    │                                 │
│               │  val_pct_10mm      │                                 │
│               └────────────────────┘                                 │
└─────────────────────────────────────────────────────────────────────┘

EgoDex on disk:
  {root}/
    ├── 0.hdf5   transforms:(N,68,4,4)  confidences:(N,68)
    ├── 0.mp4    1080p egocentric video @ 30 Hz
    ├── 1.hdf5
    └── ...
```

---

## Files Created / Modified

---

### `requirements.txt`

**Purpose**: Declares all Python dependencies and documents the two manual install steps (beingvla, flash-attn) that can't be automated via pip.

**Key entries**:
- `torch>=2.1.0`, `torchvision` — training backbone
- `h5py>=3.9.0` — reads EgoDex HDF5 annotation files
- `opencv-python-headless>=4.8.0` — video frame decoding fallback on macOS
- `einx>=0.3.0`, `timm>=0.9.0` — discovered missing deps in beingvla (not listed in their repo)
- `peft>=0.10.0` — LoRA support, ready for v2 backbone finetuning

**How it works**: Standard pip requirements. The file notes that `decord` (faster video decoder) is Linux-only and should be installed separately on Runpod. The beingvla package itself has no `pyproject.toml` so it can't be listed here — it's cloned manually into `third_party/`.

---

### `dataset/egodex_dataset.py`

**Purpose**: PyTorch Dataset that scans EgoDex HDF5 files, filters to pick-and-place sequences via keyword matching, and loads image + both-hand joint positions per sample.

**Key functions/classes**:
- `scan_sequences(root)` — recursively finds all `.hdf5` files, reads only metadata (fast, no video/joints)
- `filter_pick_place(sequences, keywords, task_filter)` — keyword filter on `llm_description`
- `_load_video_frame(mp4_path, frame_idx)` — decord → cv2 fallback chain
- `EgoDexDataset.__getitem__(idx)` — loads one context frame + T future joint frames

**How it works**:

EgoDex stores per-sequence data as paired `.hdf5` + `.mp4` files. Each HDF5 contains a `transforms` array of shape `(N, 68, 4, 4)` — these are SE(3) transformation matrices (4×4 rotation+translation) for 68 body/hand joints at 30 Hz. Joints 18–42 are the right hand, 43–67 are the left hand.

The dataset extracts 3D joint positions by taking the **translation column** of each SE(3) matrix (the last column of the top-left 3×3 block):

```python
right = f["transforms"][t0:t_end, slice(18, 43), :3, 3]  # (T, 25, 3)
left  = f["transforms"][t0:t_end, slice(43, 68), :3, 3]  # (T, 25, 3)
hand_joints = np.concatenate([right, left], axis=1)        # (T, 50, 3) metres
```

Each `__getitem__` call picks a random start frame `t0`, loads the context image at `t0`, and returns the next 16 frames of joint positions as the prediction target. If the `.mp4` is absent (metadata-only mode), a zero image tensor is returned without crashing — useful for inspecting metadata before downloading video.

The train/val split is 90/10, deterministic (seeded shuffle of sorted file paths). The "test" split uses all sequences — intended for `--eval_only` mode against the 16GB test download.

**Pick-and-place filter**: EgoDex has no structured task labels. The `llm_description` attribute is a GPT-4 generated free-text description per sequence. The filter checks if the text contains any keyword from the default list (`"pick up"`, `"place"`, `"grasp"`, etc.). This is a heuristic — expect some false positives and false negatives.

---

### `dataset/synthetic_dataset.py`

**Purpose**: Drops in for `EgoDexDataset` during `--dry_run`, returning random tensors with identical shapes so the pipeline can be exercised without any data download.

**Key class**:
- `SyntheticEgoDexDataset(n_samples, pred_horizon, image_size)` — 32 train / 8 val by default

**How it works**: Each sample is seeded by its index so DataLoader workers reproduce the same data deterministically. Joint positions are uniform random in `[-0.2, 0.2]` metres; confidence scores are near 1.0; instructions are drawn from 5 hardcoded pick-and-place templates so the tokeniser sees real text. The output dict is shape-identical to `EgoDexDataset.__getitem__`.

---

### `model/being_h0_wrapper.py`

**Purpose**: Wraps BeingVLAModel (Being-H0's backbone) for supervised training on EgoDex joint positions, adding a regression head that maps VLM hidden states to `(B, T, 50, 3)` joint coordinates.

**Key classes/functions**:
- `_ensure_local_checkpoint(model_name, hf_token)` — downloads HF checkpoint to local cache
- `JointPredictionHead` — 3-layer MLP, zero-init output for stable start
- `BeingH0Wrapper.__init__` — loads backbone, sets `img_context_token_id`, freezes backbone
- `BeingH0Wrapper._build_prompts()` — formats instructions with `<IMG_CONTEXT>` placeholders
- `BeingH0Wrapper.forward()` — backbone → mean-pool → head → `{pred_joints}`
- `load_being_h0()` — factory used by `train.py`, auto-selects 1B for dry run

**How it works**:

**The checkpoint download problem**: beingvla's `from_pretrained` only loads weights from local directories — it downloads the config from HuggingFace but silently skips the weights for remote repo IDs. The fix is `_ensure_local_checkpoint()`, which calls `huggingface_hub.snapshot_download()` first to materialize the full checkpoint locally, then passes the local path to beingvla:

```python
local_path = snapshot_download(repo_id=model_name, token=hf_token,
    ignore_patterns=["*.msgpack", "*.h5", ...])
self.backbone = BeingVLAModel.from_pretrained(local_path, ...)
```

**The `img_context_token_id` requirement**: BeingVLAModel's `forward()` raises an error if `img_context_token_id` is not set. This token tells the model which positions in `input_ids` correspond to image patch embeddings. After loading the tokenizer, we look up the `<IMG_CONTEXT>` token ID and set it on the model:

```python
img_ctx_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
self.backbone.img_context_token_id = img_ctx_id
```

**The forward pass**: BeingVLAModel expects `pixel_values` as a batch of image patches `(B, 3, H, W)`, `input_ids` that embed `<IMG_CONTEXT>` placeholders, and `image_flags` that mark which patches are active. We format the input accordingly:

```python
img_placeholder = "<IMG_CONTEXT>" * num_image_token  # e.g. ×256
prompt = f"<img>{img_placeholder}</img> {instruction}"
```

After the backbone forward pass, we mean-pool the last hidden state over non-padding tokens and pass it through the regression head to get joint positions.

**The beingvla bug fix**: The third_party `internvl_adapter.py` had a structural bug where `raise NotImplementedError` was placed inside `else: (dtype == float32)` instead of `else: (unknown architecture)`, causing every float32 forward pass (CPU / MPS) to crash. The fix was to restructure the `if/elif/else` so the dtype cast and the unsupported-arch error are separate:

```python
# Before (buggy): dtype else branch contained the error
if self.target_dtype != torch.float32:
    model = model.to(self.target_dtype)
else:
    raise NotImplementedError(...)  # fired on every CPU/Mac run!

# After (fixed): arch else branch contains the error
elif arch == 'Qwen2ForCausalLM':
    ...
else:
    raise NotImplementedError(...)
if self.target_dtype != torch.float32:
    model = model.to(self.target_dtype)
```

---

### `utils/metrics.py`

**Purpose**: All pose evaluation metrics, operating on `(B, T, J, 3)` tensors in metres, returning millimetre scalars.

**Key functions**:
- `compute_mpjpe(pred, gt)` — primary checkpoint metric; mean L2 across all joints and timesteps
- `compute_final_step_error(pred, gt)` — MPJPE at last predicted frame only (trajectory endpoint quality)
- `compute_per_timestep_error(pred, gt)` — list of T per-timestep errors (for diagnosing temporal decay)
- `compute_pct_within_threshold(pred, gt, threshold_mm=10.0)` — fraction of joints within 10mm
- `compute_all_metrics(pred, gt)` — convenience wrapper returning all metrics as a dict
- `test_metrics_smoke()` — runnable sanity check: `python -m utils.metrics`

**How it works**: All metrics use `torch.linalg.norm(..., dim=-1)` to compute per-joint L2 distance, then aggregate across batch, time, and joint dimensions. Everything runs under `torch.no_grad()`. The choice of `val_mpjpe` as the primary checkpoint metric (rather than `val_loss`) is intentional: MSE loss is sensitive to outliers and scale, while MPJPE in mm is directly interpretable as "average finger/wrist positioning error."

---

### `train.py`

**Purpose**: Single entrypoint orchestrating all pipeline components across three modes: dry run, eval-only, and full training.

**Key functions**:
- `parse_args()` — all 18 CLI flags with defaults and help strings
- `setup_wandb()` — optional W&B init; gracefully skips if not configured
- `build_dataloaders()` — switches between real EgoDex and synthetic datasets
- `build_scheduler()` — cosine decay with linear warmup (10% of total steps, max 100)
- `eval_loop()` — `@torch.no_grad()`, accumulates predictions, computes all metrics
- `train_loop()` — epoch × step loop with grad clipping, LR logging, eval triggers, checkpointing
- `save_checkpoint()` / `load_checkpoint()` — saves full state (model + optimizer + step + epoch)

**How it works**:

The three run modes are controlled by two flags:

| Mode | Flags | What happens |
|---|---|---|
| Dry run | `--dry_run` | Synthetic data, 1B model, 5 steps, eval at end |
| Eval only | `--eval_only` | Real data, no optimizer, prints metric table |
| Training | (neither) | Full train loop, saves checkpoints |

The dry run override block enforces `batch_size=2, max_steps=5, eval_every=5, num_workers=0` regardless of what was passed, making it safe to run on a MacBook without knowing the right parameters.

Checkpointing saves `best_model.pt` whenever `val_mpjpe` improves, plus optional periodic checkpoints at `--save_every` steps for recovery. The checkpoint dict stores model state, optimizer state, global step, and epoch so training resumes correctly:

```python
torch.save({
    "model_state_dict":     model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "global_step":          global_step,
    "epoch":                epoch,
    "metrics":              metrics,
}, path)
```

The LR scheduler is fast-forwarded on resume by calling `scheduler.step()` N times, which is correct for `LambdaLR` (no side effects).

W&B logging is fully optional — if `--wandb_project` is not passed, `wandb_module` is `None` and all `log_metrics()` calls are no-ops.

---

### `third_party/Being-H0/` (patched)

**Purpose**: Clone of the Being-H0 GitHub repo, used because beingvla has no pip-installable package. One bug was patched in-place.

**Patched file**: `beingvla/models/vlm/internvl_adapter.py`

The `_create_language_model` method had a structural bug (see model wrapper section above). The patch was applied directly to the cloned file since the repo has no pip install mechanism and the bug blocks all CPU/MPS usage.

---

### `README.md`

**Purpose**: Complete operator manual covering setup, all run modes, EgoDex download strategy, metrics, and architecture notes.

**Sections**: venv setup, HF auth, W&B setup, MacBook dry run, eval-only baseline, single-GPU Runpod training, multi-GPU (v2 note), EgoDex download, pick-and-place filter explanation, metrics table, architecture notes, project structure.

---

### `secrets.md` (gitignored)

**Purpose**: Local file for `HF_TOKEN` and `WANDB_API_KEY`. Added to `.gitignore`. Never committed.

---

### `runpod_instructions.md`

**Purpose**: Step-by-step operator manual to go from a blank Runpod GPU pod to a running finetuning job.

**Sections** (11 steps):
1. Pod setup — GPU/disk recommendations (A100 40GB, 200GB+ disk)
2. Clone repo
3. Install deps — including a scripted patch for the beingvla `internvl_adapter.py` bug
4. Set API keys (`HF_TOKEN`, `WANDB_API_KEY`)
5. **GPU dry run with real 8B model** — `python train.py --dry_run --model_name BeingBeyond/Being-H0-8B-2508` — verifies `device=cuda`, `flash_attn=True`, `bfloat16`, `hidden_dim=4096`
6. Download EgoDex data (test 16GB or full 1.5TB)
7. Eval-only baseline (raw pretrained 8B, no training)
8. Finetuning (test split first, then full)
9. Resume from checkpoint (for interrupted jobs)
10. Evaluate finetuned model vs baseline
11. Download results before terminating pod

Also includes a troubleshooting table (8 common errors with fixes) and a copy-paste quick-reference block with all commands in order.

---

## Data Flow

```
                         EgoDex HDF5 + MP4 files
                                  │
                    scan_sequences() — reads only attrs
                                  │
                    filter_pick_place() — keyword match
                                  │
                     90/10 train/val split (seeded)
                                  │
                    __getitem__(idx)
                      ├─ random t0 in [0, N-T-1]
                      ├─ load frame t0 from MP4 → (3, 448, 448)
                      └─ load transforms[t0:t0+16, 18:68, :3, 3]
                           right: joints 18-42 → (16, 25, 3)
                           left:  joints 43-67 → (16, 25, 3)
                           concat → hand_joints (16, 50, 3) metres
                                  │
                    DataLoader (batch_size=B)
                                  │
                    BeingH0Wrapper.forward(images, instructions)
                      ├─ _build_prompts() → input_ids with <IMG_CONTEXT>×256
                      ├─ BeingVLAModel.forward(pixel_values, input_ids, ...)
                      │    output_hidden_states=True
                      │    hidden_states[-1] → (B, S, 896)
                      ├─ mean-pool over non-padding → (B, 896)
                      └─ JointPredictionHead → pred_joints (B, 16, 50, 3)
                                  │
                    MSELoss(pred_joints, gt_joints) → scalar
                                  │
                    loss.backward() + AdamW.step() + scheduler.step()
                                  │
                    every eval_every steps:
                      └─ eval_loop() → val_mpjpe, val_mean_l2,
                                       val_final_step_error, val_pct_within_10mm
                           → save best_model.pt if val_mpjpe improved
                           → log all metrics to W&B
```

---

## Test Coverage

- **Automated tests**: 0 formal test files. The metrics module has an inline `test_metrics_smoke()` runnable via `python -m utils.metrics`.
- **Integration test**: The `--dry_run` mode serves as the integration test — it exercises every component (dataset, model load, forward pass, loss, backward, eval loop, checkpoint save) in ~3 seconds on a MacBook.
- **Verified**: Dry run confirmed working with real pretrained Being-H0-1B-2508 weights on MPS (Apple Silicon). Hidden dim auto-detected as 896. 5 training steps + 1 val pass completed cleanly.

---

## Security Measures

- `secrets.md` is gitignored; HF token is read from environment variable, never hardcoded.
- HF token is not logged or printed.
- No user-facing web server or network-exposed surface in this sprint.

---

## Known Limitations

**Architecture**

1. **Single context frame only**: The model sees one image at `t0` to predict 16 future frames. No temporal context stack (multiple past frames) is used. This is a significant limitation for pick-and-place which involves continuous motion.

2. **Regression head bypasses motion tokeniser**: Being-H0's native GRVQ-8K motion tokeniser is not used. We add an MLP head and train with MSE loss on raw joint positions. This means the pretrained model's action-space knowledge is not directly exploited — only the visual-language feature representations.

3. **No temporal structure in head**: The `JointPredictionHead` maps a single pooled vector to all 16 future frames independently. It has no recurrence, attention, or autoregressive decoding over time, which will limit multi-step prediction accuracy.

4. **1 image patch per sample**: Being-H0 supports dynamic tiling (1–12 crops for high-res images). We use only 1 patch at 448×448. For 1080p EgoDex frames, this means significant downscaling and loss of fine-grained hand detail.

5. **Backbone always frozen in v1**: Only 3.76M head parameters are trained. The full model benefit of pretrained VLM representations is not yet realised through finetuning.

**Data**

6. **Keyword filter is a heuristic**: EgoDex has no structured task labels. The pick-and-place filter may include non-pick-and-place sequences that mention "move" or "transfer" and exclude valid sequences with unusual phrasing.

7. **No train/val split in EgoDex**: EgoDex only has official train and test splits. We create a val split by holding out 10% of the training files. This means val and train share the same recording sessions and potentially similar sequences.

8. **SE(3) translation only**: We extract only the translation column (3D position) from the SE(3) joint transforms, discarding rotation. This means joint orientations are not supervised — a limitation for tasks requiring precise finger grip orientation.

**Infrastructure**

9. **beingvla has a patched bug**: `third_party/Being-H0/beingvla/models/vlm/internvl_adapter.py` was patched in-place. If the repo is re-cloned, the patch will be lost. This should be tracked or upstreamed.

10. **No multi-GPU support**: DDP/FSDP is not implemented. Single GPU only on Runpod.

11. **resume_from loads checkpoint twice**: The checkpoint file is loaded once for model weights, then loaded again for optimizer state. This is a minor inefficiency.

---

## What's Next (v2 Priorities)

1. **Multi-frame context**: Stack 4–8 past frames as input instead of a single frame. This gives the model temporal motion context for more accurate trajectory prediction.

2. **LoRA backbone finetuning**: Use PEFT/LoRA to partially unfreeze the backbone with manageable GPU memory. The `--unfreeze_backbone` flag is already wired; it just needs LoRA wrapping in the wrapper init.

3. **Native motion token finetuning**: Add MANO parameter fitting from EgoDex SE(3) joints (using `manotorch`), encode to GRVQ-8K tokens, and finetune with Being-H0's native cross-entropy motion token loss. This is the architecturally correct path.

4. **Temporal decoder head**: Replace the flat MLP with a small causal transformer or GRU that autoregressively decodes future frames. This would dramatically improve multi-step prediction.

5. **Joint rotation supervision**: Extract and supervise rotation matrices (or 6D rotation representations) from EgoDex SE(3) transforms, not just translations.

6. **Fix the beingvla patch properly**: Upstream the `internvl_adapter.py` fix to the Being-H0 repo, or pin a commit and apply the patch via a setup script.

7. **Formal test suite**: Add pytest unit tests for dataset filtering, metric computation, and model I/O shapes.
