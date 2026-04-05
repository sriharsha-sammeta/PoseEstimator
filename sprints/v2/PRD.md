# Sprint v2 — PRD: Runpod GPU Validation

## Overview

Deploy the v1 finetuning pipeline to a Runpod GPU pod and verify that the full stack — environment setup, beingvla install, Being-H0-8B model loading, synthetic dry run, and a short training run — works end-to-end on real GPU hardware. No new features; this is a pure infrastructure validation sprint.

---

## Goals

- **Done** = A Runpod pod is created programmatically via MCP or API
- **Done** = All dependencies install cleanly on the pod (torch, beingvla, flash-attn, etc.)
- **Done** = `python train.py --dry_run --model_name BeingBeyond/Being-H0-8B-2508` completes on the pod with `device=cuda`, `flash_attn=True`, `bfloat16`
- **Done** = `python train.py --eval_only --dry_run` also completes, printing baseline metrics
- **Done** = Pod is terminated after validation to avoid runaway costs

---

## User Stories

- As a researcher, I want to know my pipeline runs on Runpod GPU before committing to an expensive multi-hour training job, so I don't waste money on environment issues.
- As a researcher, I want the Runpod setup automated as much as possible, so I can reproduce it without following a 20-step manual.

---

## Technical Architecture

### What exists (from v1)

```
Local MacBook
├── train.py                  ← verified with 1B model, MPS, synthetic data
├── dataset/                  ← EgoDexDataset + SyntheticEgoDexDataset
├── model/being_h0_wrapper.py ← BeingVLAModel + JointPredictionHead
├── utils/metrics.py          ← MPJPE etc.
├── requirements.txt
├── runpod_instructions.md    ← manual step-by-step guide
└── third_party/Being-H0/    ← cloned beingvla repo (patched)
```

### What this sprint adds

```
Runpod GPU Pod (A100 / H100)
├── /workspace/PoseEstimator/ ← cloned from GitHub
├── .venv/                    ← pip install -r requirements.txt
├── third_party/Being-H0/    ← cloned + patched beingvla
└── Validation:
     ├── dry_run with 8B model → device=cuda, bfloat16, flash_attn
     └── eval_only dry_run     → metrics printed
```

### Execution flow

```
Local MacBook
    │
    │  1. Create pod via Runpod MCP / API
    │  2. SSH into pod (or exec commands remotely)
    │  3. Clone repo from GitHub
    │  4. Install deps + beingvla + flash-attn
    │  5. Export HF_TOKEN
    │  6. Run dry_run with 8B model
    │  7. Run eval_only dry_run
    │  8. Collect output / confirm success
    │  9. Terminate pod
    │
    ▼
Runpod Pod (A100)
    device=cuda, dtype=bfloat16, flash_attn=True
    hidden_dim=4096 (8B model)
```

### Environment variables needed

| Variable | Source | Purpose |
|---|---|---|
| `RUNPOD_API_KEY` | `keys_tokens/export_keys.sh` | Create/manage Runpod pods |
| `HF_TOKEN` | `keys_tokens/export_keys.sh` | Download Being-H0 checkpoints |
| `WANDB_API_KEY` | `keys_tokens/export_keys.sh` | (Optional) W&B logging |

---

## Out of Scope (v2)

- Downloading real EgoDex data (test or train splits)
- Running actual finetuning on real data
- Multi-GPU / DDP training
- Any code changes to train.py, model, or dataset
- W&B dashboard setup or monitoring

---

## Dependencies

- **v1 code** pushed to GitHub at `sriharsha-sammeta/PoseEstimator`
- **Runpod MCP** configured in `.mcp.json` with `RUNPOD_API_KEY`
- **HF_TOKEN** for downloading Being-H0-8B-2508 checkpoint
- **Runpod account** with GPU quota (A100 40GB preferred)

---

## Success Criteria

The sprint is successful when the following output is observed from the Runpod pod:

```
[BeingH0Wrapper] Loading checkpoint: BeingBeyond/Being-H0-8B-2508
  device=cuda, dtype=torch.bfloat16, flash_attn=True
  hidden_dim=4096
  backbone frozen — only head is trainable
  head trainable params: 12546.0K

[train] Starting: total_steps=5 ...
  step     1/5  ...  loss=0.01xx
  ...
  step     5/5  ...
  [eval @ step 5]  val_loss=...  val_mpjpe=... mm
  ★ new best val_mpjpe=... mm

[train] Done. best val_mpjpe=... mm → runs/.../best_model.pt
```

Key checks:
- `device=cuda` (not cpu/mps)
- `dtype=torch.bfloat16` (not float32)
- `flash_attn=True` (not False)
- `hidden_dim=4096` (8B model, not 896/1B)
- 5 steps complete without error
- Checkpoint file saved
