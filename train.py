#!/usr/bin/env python3
"""
train.py — Being-H0 × EgoDex pick-and-place finetuning pipeline.

Modes
─────
  Dry run (MacBook smoke test):
      python train.py --dry_run --output_dir ./runs/dry_run

  Eval-only baseline (raw pretrained model, no training):
      python train.py --eval_only \\
          --dataset_root /data/egodex/test \\
          --output_dir   ./runs/baseline

  Full training (single GPU):
      python train.py \\
          --dataset_root /data/egodex/train \\
          --output_dir   ./runs/finetune_v1 \\
          --wandb_project being_h0_egodex \\
          --epochs 10

See README.md for full usage guide.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EgoDexDataset, SyntheticEgoDexDataset
from model import load_being_h0
from utils.metrics import compute_all_metrics, compute_per_timestep_error

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Being-H0 × EgoDex pick-and-place finetuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument("--model_name",    default="BeingBeyond/Being-H0-8B-2508",
                   help="HuggingFace model ID or local path. In dry_run mode, "
                        "auto-switched to the 1B checkpoint unless explicitly set.")
    p.add_argument("--hf_token_env",  default="HF_TOKEN",
                   help="Environment variable name that holds the HuggingFace auth token.")

    # Dataset
    p.add_argument("--dataset_root",  default=None,
                   help="Path to directory containing EgoDex .hdf5 (and .mp4) files. "
                        "Not needed in --dry_run mode.")
    p.add_argument("--task_filter",   default="pick_place",
                   choices=["pick_place", "all"],
                   help="Filter dataset to pick-and-place sequences (pick_place) "
                        "or use all sequences (all).")

    # Training
    p.add_argument("--output_dir",    default="./runs/default")
    p.add_argument("--batch_size",    type=int,   default=8)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--epochs",        type=int,   default=10)
    p.add_argument("--max_steps",     type=int,   default=None,
                   help="If set, stop training after this many gradient steps "
                        "(overrides --epochs).")
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay",  type=float, default=1e-2)
    p.add_argument("--eval_every",    type=int,   default=500,
                   help="Run validation every N gradient steps.")
    p.add_argument("--save_every",    type=int,   default=1000,
                   help="Save a periodic checkpoint every N gradient steps.")
    p.add_argument("--resume_from",   default=None,
                   help="Path to a checkpoint .pt file to resume training from.")
    p.add_argument("--pred_horizon",  type=int,   default=16,
                   help="Number of future frames T to predict.")

    # Modes
    p.add_argument("--dry_run",    action="store_true",
                   help="Run a 5-step smoke test with synthetic data and the 1B model.")
    p.add_argument("--eval_only",  action="store_true",
                   help="Evaluate the (raw or loaded) model on val split only; no training.")

    # Backbone
    p.add_argument("--freeze_backbone", action="store_true", default=True,
                   help="Freeze backbone; train only the regression head (default).")
    p.add_argument("--unfreeze_backbone", dest="freeze_backbone", action="store_false",
                   help="Unfreeze backbone for full finetuning (use with LoRA on GPU).")

    # W&B
    p.add_argument("--wandb_project",  default=None,
                   help="W&B project name. Omit to disable W&B logging.")
    p.add_argument("--wandb_run_name", default=None,
                   help="W&B run name. Auto-generated if not set.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# W&B
# ---------------------------------------------------------------------------

def setup_wandb(args: argparse.Namespace):
    """Initialise W&B if requested. Returns the wandb module or None."""
    if args.wandb_project is None:
        return None

    try:
        import wandb
    except ImportError:
        print("WARNING: wandb not installed — skipping W&B logging.")
        return None

    api_key = os.environ.get("WANDB_API_KEY", "")
    if not api_key:
        print("WARNING: WANDB_API_KEY not set — W&B logging will be offline only.")

    run_name = args.wandb_run_name or (
        f"{'dry_run' if args.dry_run else 'train'}"
        f"_{Path(args.output_dir).name}"
    )

    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args),
        resume="allow",
    )
    print(f"[W&B] Initialised: project={args.wandb_project}, run={run_name}")
    return wandb


def log_metrics(wandb_module, metrics: dict, step: int):
    if wandb_module is not None:
        wandb_module.log(metrics, step=step)


# ---------------------------------------------------------------------------
# Dataset / DataLoader helpers
# ---------------------------------------------------------------------------

def build_dataloaders(args: argparse.Namespace):
    """Return (train_loader, val_loader)."""
    if args.dry_run:
        # Synthetic data — no EgoDex download required
        n_train, n_val = 32, 8
        train_ds = SyntheticEgoDexDataset(n_samples=n_train, pred_horizon=args.pred_horizon)
        val_ds   = SyntheticEgoDexDataset(n_samples=n_val,   pred_horizon=args.pred_horizon)
        print(f"[dry_run] Synthetic datasets: train={n_train}, val={n_val}")
    else:
        if args.dataset_root is None:
            print("ERROR: --dataset_root is required when not in --dry_run mode.")
            sys.exit(1)

        train_ds = EgoDexDataset(
            root=args.dataset_root,
            split="train",
            task_filter=args.task_filter,
            pred_horizon=args.pred_horizon,
        )
        val_ds = EgoDexDataset(
            root=args.dataset_root,
            split="val",
            task_filter=args.task_filter,
            pred_horizon=args.pred_horizon,
        )

    # In dry_run: use fewer workers to avoid overhead on MacBook
    nw = 0 if args.dry_run else args.num_workers

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=nw, pin_memory=False, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=nw, pin_memory=False,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def build_scheduler(optimizer, total_steps: int):
    """Cosine decay with linear warmup."""
    warmup_steps = min(100, max(1, total_steps // 10))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_loop(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    global_step: int,
    wandb_module,
    verbose: bool = True,
) -> dict[str, float]:
    """Run evaluation and return metrics dict."""
    model.eval()

    all_pred, all_gt, total_loss = [], [], 0.0
    loss_fn = nn.MSELoss()
    n_batches = 0

    for batch in tqdm(val_loader, desc="  eval", leave=False):
        images      = batch["image"]
        instructions = batch["instruction"]
        gt_joints   = batch["hand_joints"].to(device)  # (B, T, 50, 3)

        outputs = model(images, instructions)
        pred_joints = outputs["pred_joints"]            # (B, T, 50, 3)

        loss = loss_fn(pred_joints.float(), gt_joints.float())
        total_loss += loss.item()
        n_batches  += 1

        all_pred.append(pred_joints.cpu().float())
        all_gt.append(gt_joints.cpu().float())

    all_pred = torch.cat(all_pred, dim=0)   # (N, T, 50, 3)
    all_gt   = torch.cat(all_gt,   dim=0)

    metrics = compute_all_metrics(all_pred, all_gt)
    metrics["val_loss"] = total_loss / max(1, n_batches)

    if verbose:
        print(
            f"  [eval @ step {global_step}]"
            f"  val_loss={metrics['val_loss']:.4f}"
            f"  val_mpjpe={metrics['val_mpjpe']:.1f} mm"
            f"  val_final_step={metrics['val_final_step_error']:.1f} mm"
        )

    log_metrics(wandb_module, {**metrics, "global_step": global_step}, step=global_step)

    model.train()
    return metrics


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    epoch: int,
    metrics: dict,
    path: Path,
):
    torch.save(
        {
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step":          global_step,
            "epoch":                epoch,
            "metrics":              metrics,
        },
        path,
    )
    print(f"  [checkpoint] saved → {path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
) -> tuple[int, int]:
    """Load checkpoint. Returns (global_step, epoch)."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    global_step = ckpt.get("global_step", 0)
    epoch       = ckpt.get("epoch",       0)
    print(f"[resume] Loaded checkpoint from {path} (step={global_step}, epoch={epoch})")
    return global_step, epoch


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    args: argparse.Namespace,
    wandb_module,
    start_step: int = 0,
    start_epoch: int = 0,
) -> None:
    loss_fn = nn.MSELoss()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step   = start_step
    best_mpjpe    = float("inf")
    best_ckpt_path = output_dir / "best_model.pt"

    # Determine total steps
    steps_per_epoch = len(train_loader)
    if args.max_steps is not None:
        total_steps = args.max_steps
        total_epochs = math.ceil(total_steps / max(1, steps_per_epoch))
    else:
        total_epochs = args.epochs
        total_steps  = total_epochs * steps_per_epoch

    print(
        f"\n[train] Starting: "
        f"total_steps={total_steps}, epochs={total_epochs}, "
        f"steps_per_epoch={steps_per_epoch}, "
        f"trainable_params={model.trainable_param_count()/1e6:.2f}M"
    )

    model.train()
    t_start = time.time()

    for epoch in range(start_epoch, total_epochs):
        for batch in train_loader:
            if global_step >= total_steps:
                break

            images       = batch["image"]
            instructions = batch["instruction"]
            gt_joints    = batch["hand_joints"].to(device)  # (B, T, 50, 3)

            # Forward
            outputs     = model(images, instructions)
            pred_joints = outputs["pred_joints"]            # (B, T, 50, 3)

            loss = loss_fn(pred_joints.float(), gt_joints.float())

            # Backward
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            lr_now = scheduler.get_last_lr()[0]

            # --- logging ---
            if global_step % 10 == 0 or global_step <= 5:
                elapsed = time.time() - t_start
                print(
                    f"  step {global_step:5d}/{total_steps}"
                    f"  epoch {epoch+1}/{total_epochs}"
                    f"  loss={loss.item():.4f}"
                    f"  lr={lr_now:.2e}"
                    f"  t={elapsed:.0f}s"
                )
                log_metrics(
                    wandb_module,
                    {
                        "train/loss": loss.item(),
                        "train/lr":   lr_now,
                        "train/epoch": epoch + global_step / max(1, steps_per_epoch),
                    },
                    step=global_step,
                )

            # --- eval ---
            if global_step % args.eval_every == 0 or global_step == total_steps:
                metrics = eval_loop(
                    model, val_loader, device, global_step, wandb_module
                )
                mpjpe = metrics["val_mpjpe"]

                # Save best checkpoint
                if mpjpe < best_mpjpe:
                    best_mpjpe = mpjpe
                    save_checkpoint(model, optimizer, global_step, epoch, metrics, best_ckpt_path)
                    print(f"  ★ new best val_mpjpe={mpjpe:.1f} mm")
                    log_metrics(wandb_module, {"best_val_mpjpe": best_mpjpe}, step=global_step)

                model.train()

            # --- periodic save ---
            if args.save_every and global_step % args.save_every == 0:
                periodic_path = output_dir / f"step_{global_step:06d}.pt"
                save_checkpoint(model, optimizer, global_step, epoch, {}, periodic_path)

        if global_step >= total_steps:
            break

    print(f"\n[train] Done. best val_mpjpe={best_mpjpe:.1f} mm → {best_ckpt_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Dry-run overrides
    if args.dry_run:
        args.batch_size = 2
        args.max_steps  = 5
        args.eval_every = 5
        args.num_workers = 0
        print("[dry_run] Overrides: batch_size=2, max_steps=5, eval_every=5")

    # HuggingFace token
    hf_token = os.environ.get(args.hf_token_env, None)
    if hf_token:
        print(f"[auth] HF token found in ${args.hf_token_env}")
    else:
        print(f"[auth] No HF token in ${args.hf_token_env} — using unauthenticated download")

    # W&B
    wandb_module = setup_wandb(args)

    # Output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # DataLoaders
    print("\n[data] Building datasets...")
    train_loader, val_loader = build_dataloaders(args)

    # Model
    # Detect if user explicitly passed --model_name on the CLI
    model_name_explicit = any(
        a.startswith("--model_name") for a in sys.argv[1:]
    )
    print("\n[model] Loading Being-H0...")
    model = load_being_h0(
        model_name=args.model_name,
        hf_token=hf_token,
        dry_run=args.dry_run,
        freeze_backbone=args.freeze_backbone,
        pred_horizon=args.pred_horizon,
        model_name_explicitly_set=model_name_explicit,
    )
    device = model.dev

    # Resume
    start_step, start_epoch = 0, 0
    if args.resume_from:
        # Optimizer not built yet — load model weights only here; load opt state below
        ckpt = torch.load(args.resume_from, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        start_step  = ckpt.get("global_step", 0)
        start_epoch = ckpt.get("epoch", 0)
        print(f"[resume] model weights loaded (step={start_step}, epoch={start_epoch})")

    # Optimizer + scheduler (built after potential freeze)
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location="cpu")
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    steps_per_epoch = len(train_loader)
    if args.max_steps is not None:
        total_steps = args.max_steps
    else:
        total_steps = args.epochs * steps_per_epoch

    scheduler = build_scheduler(optimizer, total_steps)
    # Fast-forward scheduler to resume step
    for _ in range(start_step):
        scheduler.step()

    # ── Eval-only mode ──────────────────────────────────────────────────────
    if args.eval_only:
        print("\n[eval_only] Evaluating raw pretrained model on val split...")
        metrics = eval_loop(model, val_loader, device, global_step=0, wandb_module=wandb_module)
        print("\n[eval_only] Results:")
        for k, v in metrics.items():
            unit = " mm" if "error" in k or "mpjpe" in k or "l2" in k else ""
            pct  = " %" if "pct" in k else ""
            print(f"  {k}: {v:.3f}{unit}{pct}")
        if wandb_module is not None:
            wandb_module.finish()
        return

    # ── Training mode ────────────────────────────────────────────────────────
    train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        args=args,
        wandb_module=wandb_module,
        start_step=start_step,
        start_epoch=start_epoch,
    )

    if wandb_module is not None:
        wandb_module.finish()


if __name__ == "__main__":
    main()
