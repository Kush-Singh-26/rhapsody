"""Text LM Training — Muon + AdamW, WSD schedule, text-only pretraining.

This module extracts the LM-specific training utilities from train.py
for use in standalone text-only pretraining (Stage 1).
"""

from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo
from torch.utils.data import DataLoader, IterableDataset
from accelerate import Accelerator

from .lm import TextLM, LmConfig, create_text_only_65m
from .data import get_tokenizer, TextPretrainDataset, DataCollatorWithPadding


# =============================================================================
# Learning Rate Schedules
# =============================================================================

def get_wsd_lr(step: int, total_steps: int, warmup_frac: float = 0.01, decay_frac: float = 0.12) -> float:
    """
    Warmup-Stable-Decay schedule. Returns a [0, 1] multiplier for the peak LR.
      - Linear warmup for the first warmup_frac fraction of steps.
      - Constant peak for the stable middle portion.
      - Cosine decay for the final decay_frac fraction of steps.
    """
    warmup_steps = int(total_steps * warmup_frac)
    decay_start = int(total_steps * (1.0 - decay_frac))

    if step < warmup_steps:
        return step / max(1, warmup_steps)
    elif step < decay_start:
        return 1.0
    else:
        progress = (step - decay_start) / max(1, total_steps - decay_start)
        return (1 + math.cos(math.pi * progress)) / 2


def get_cosine_lr(step: int, total_steps: int, warmup_frac: float = 0.03) -> float:
    """
    Cosine decay schedule. Returns a [0, 1] multiplier for the peak LR.
      - Linear warmup for the first warmup_frac fraction of steps.
      - Cosine decay to zero for the remaining steps.
    """
    warmup_steps = int(total_steps * warmup_frac)
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))


# =============================================================================
# Muon Optimizer
# =============================================================================

class Muon(torch.optim.Optimizer):
    """
    Muon: Momentum + Newton-Schulz orthogonalisation for weight matrices.

    Uses AdamW for scalars, embeddings, and LM head (second param group).

    References:
      - Kosson et al., "Muon" (2024).
      - Original implementation: github.com/KellerJordan/modded-nanogpt
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_params=None,
        adamw_lr: float = 1e-3,
        adamw_betas: tuple = (0.9, 0.95),
        adamw_wd: float = 0.1,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            is_adamw=False,
            betas=adamw_betas,
            weight_decay=0.0,
        )
        super().__init__(list(params), defaults)

        if adamw_params is not None:
            self.add_param_group({
                "params": list(adamw_params),
                "lr": adamw_lr,
                "betas": adamw_betas,
                "weight_decay": adamw_wd,
                "is_adamw": True,
                "momentum": momentum,
                "nesterov": nesterov,
                "ns_steps": ns_steps,
            })

        self._adam_step = 0

    def _newton_schulz(self, M: torch.Tensor) -> torch.Tensor:
        orig_shape = M.shape
        if M.ndim > 2:
            M = M.view(M.shape[0], -1)

        flip = M.shape[0] > M.shape[1]
        X = M.T if flip else M

        X = X / (X.norm() + 1e-7)
        a, b, c = 3.4445, -4.7750, 2.0315
        for _ in range(5):
            A = X @ X.T
            B = b * A + c * (A @ A)
            X = a * X + B @ X

        if flip:
            X = X.T

        return X.view(orig_shape)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._adam_step += 1

        for group in self.param_groups:
            if group.get("is_adamw"):
                beta1, beta2 = group.get("betas", (0.9, 0.95))
                wd = group.get("weight_decay", 0.1)
                lr = group["lr"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    state = self.state[p]
                    if "m" not in state:
                        state["m"] = torch.zeros_like(p)
                        state["v"] = torch.zeros_like(p)
                    state["m"].mul_(beta1).add_(grad, alpha=1 - beta1)
                    state["v"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    m_hat = state["m"] / (1 - beta1 ** self._adam_step)
                    v_hat = state["v"] / (1 - beta2 ** self._adam_step)
                    p.add_(-lr * (m_hat / (v_hat.sqrt() + 1e-8) + wd * p))
            else:
                lr = group["lr"]
                momentum = group.get("momentum", 0.95)
                nesterov = group.get("nesterov", True)
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    state = self.state[p]
                    if "m" not in state:
                        state["m"] = torch.zeros_like(p)
                    state["m"].mul_(momentum).add_(grad)
                    update = grad + momentum * state["m"] if nesterov else state["m"]
                    if p.ndim >= 2:
                        update = self._newton_schulz(update)
                    p.add_(-lr * update)

        return loss

    def state_dict(self):
        d = super().state_dict()
        d['_adam_step'] = self._adam_step
        return d

    def load_state_dict(self, state_dict):
        self._adam_step = state_dict.pop('_adam_step', 0)
        super().load_state_dict(state_dict)


# =============================================================================
# Loss Computation (Text-Only)
# =============================================================================

def compute_lm_loss(model: nn.Module, batch: dict, device: torch.device) -> torch.Tensor:
    """
    Compute next-token prediction loss with auxiliary Z-loss (alpha=1e-4).

    Labels are pre-shifted: labels[t] = next token after input_ids[t].
    """
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    output = model(input_ids, labels=labels)
    loss = output["loss"]
    logits = output["logits"]

    # Auxiliary Z-loss to prevent logit explosion
    flat_logits = logits.view(-1, logits.size(-1))
    flat_labels = labels.view(-1)
    mask = flat_labels != -100
    if mask.any():
        masked_logits = flat_logits[mask]
        log_z = torch.logsumexp(masked_logits, dim=-1)
        z_loss = torch.mean(log_z ** 2)
        loss = loss + 1e-4 * z_loss

    return loss


# =============================================================================
# Checkpoint Helpers
# =============================================================================

def find_latest_checkpoint(output_dir: Path) -> Optional[Path]:
    checkpoints = []
    if not output_dir.exists():
        return None
    for item in output_dir.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            suffix = item.name.removeprefix("checkpoint-")
            if suffix.isdigit():
                checkpoints.append((int(suffix), item))
    return sorted(checkpoints)[-1][1] if checkpoints else None


def save_checkpoint(
    ckpt_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    accelerator: Accelerator,
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(model)
    raw_model = getattr(unwrapped_model, "_orig_mod", unwrapped_model)
    payload = {
        "model": raw_model.state_dict(),
        "step": step,
        "config": raw_model.config.to_config_dict() if hasattr(raw_model, "config") else {},
        "rng_state": torch.get_rng_state(),
        "python_random_state": random.getstate(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

    accelerator.save(payload, ckpt_dir / "model.pt")
    accelerator.save({"optimizer": optimizer.state_dict()}, ckpt_dir / "optimizer.pt")
    accelerator.save({"scheduler": scheduler.state_dict()}, ckpt_dir / "scheduler.pt")


# =============================================================================
# Build LM Optimizer
# =============================================================================

def build_lm_optimizer(
    model: nn.Module,
    muon_lr: float = 0.015,
    adamw_lr: float = 0.0008,
    adamw_betas: tuple = (0.9, 0.95),
    adamw_wd: float = 0.1,
) -> torch.optim.Optimizer:
    """
    Build Muon + AdamW optimizer for TextLM training.

    - Muon: 2D+ weight matrices (attention/FFN projections)
    - AdamW: embeddings, lm_head, norms, biases (no weight decay)
    """
    muon_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad
        and p.ndim >= 2
        and "embed" not in n
        and "lm_head" not in n
        and "norm_scale" not in n
    ]

    adamw_no_decay_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad
        and (p.ndim < 2 or "embed" in n or "lm_head" in n or "norm_scale" in n)
    ]

    all_ids = {id(p) for n, p in model.named_parameters() if p.requires_grad}
    covered_ids = {id(p) for p in muon_params + adamw_no_decay_params}
    assert covered_ids == all_ids, f"Param coverage mismatch! {len(all_ids ^ covered_ids)} params uncovered."

    optimizer = Muon(
        muon_params,
        lr=muon_lr, momentum=0.95, nesterov=True, ns_steps=5,
        adamw_params=None,
    )

    optimizer.add_param_group({
        "params": adamw_no_decay_params,
        "lr": adamw_lr,
        "betas": adamw_betas,
        "weight_decay": 0.0,
        "is_adamw": True,
    })

    return optimizer


# =============================================================================
# Training Loop
# =============================================================================

def train_lm():
    """Train TextLM from scratch (Stage 1 text-only pretraining)."""
    import argparse
    parser = argparse.ArgumentParser(description="Rhapsody Text LM Training")
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0008,
                        help="AdamW LR (for embeddings/norms/biases).")
    parser.add_argument("--muon-lr", type=float, default=0.015,
                        help="Muon LR (for 2D+ weight matrices).")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint directory to resume from.")
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="Enable gradient checkpointing to trade compute for memory.")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile on CUDA.")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Detect best mixed precision
    mixed_precision = "no"
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        mixed_precision = "bf16" if major >= 8 else "fp16"

    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=args.grad_accum,
    )
    device = accelerator.device
    print = accelerator.print

    use_wandb = False
    if accelerator.is_main_process:
        try:
            import wandb
            import hashlib
            if not os.environ.get("WANDB_API_KEY") and not os.environ.get("WANDB_MODE"):
                os.environ["WANDB_MODE"] = "disabled"
            run_id = hashlib.md5(str(Path(args.output_dir).resolve()).encode()).hexdigest()[:8]
            wandb.init(project="rhapsody-lm", config=vars(args), resume="allow", id=f"lm-{run_id}")
            use_wandb = wandb.run is not None
        except Exception:
            use_wandb = False

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"[LM] Device: {device}")
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability()
        print(f"[LM] GPU: {torch.cuda.get_device_name()}")
        if major >= 8:
            torch.set_float32_matmul_precision('high')

    # Model
    model = create_text_only_65m(vocab_size=args.vocab_size)
    model.config.gradient_checkpointing = args.grad_checkpoint
    model = model.to(device)

    # Dataset
    tokenizer = get_tokenizer(symbolic=False)
    dataset = TextPretrainDataset(tokenizer, seq_len=args.seq_len)

    # Optimizer
    optimizer = build_lm_optimizer(model, muon_lr=args.muon_lr, adamw_lr=args.lr)

    # Scheduler (WSD)
    lr_fn = lambda step, n=args.max_steps: get_wsd_lr(step, n)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[lr_fn] * len(optimizer.param_groups),
    )

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume
    start_step = 0
    if args.resume:
        ckpt_path = Path(args.resume)
        model_pt = ckpt_path / "model.pt"
        if model_pt.exists():
            print(f"[LM] Resuming from {ckpt_path}")
            ckpt = torch.load(model_pt, map_location=device, weights_only=True)
            state_dict = ckpt["model"]
            if any(k.startswith("module.") for k in state_dict):
                state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            start_step = ckpt.get("step", 0)
            if "rng_state" in ckpt:
                torch.set_rng_state(ckpt["rng_state"].cpu())
            if "python_random_state" in ckpt:
                random.setstate(ckpt["python_random_state"])
            if torch.cuda.is_available() and "cuda_rng_state_all" in ckpt:
                torch.cuda.set_rng_state_all(ckpt["cuda_rng_state_all"])
            print(f"[LM] Resumed at step {start_step}")

        opt_pt = ckpt_path / "optimizer.pt"
        if opt_pt.exists():
            opt_ckpt = torch.load(opt_pt, map_location=device, weights_only=True)
            optimizer.load_state_dict(opt_ckpt["optimizer"])

        sched_pt = ckpt_path / "scheduler.pt"
        if sched_pt.exists():
            sched_ckpt = torch.load(sched_pt, map_location=device, weights_only=True)
            scheduler.load_state_dict(sched_ckpt["scheduler"])
        else:
            scheduler.last_epoch = start_step

    # DataLoader
    is_iterable = isinstance(dataset, IterableDataset)
    num_workers = 0 if is_iterable else (args.num_workers if args.num_workers is not None else min(4, os.cpu_count() or 1))
    collate_fn = None if is_iterable else DataCollatorWithPadding(tokenizer)

    batches_to_skip = start_step * args.grad_accum
    if not is_iterable and batches_to_skip > 0:
        examples_to_skip = batches_to_skip * args.batch_size * accelerator.num_processes
        print(f"[LM] Fast-forwarding dataset by {examples_to_skip} examples...")
        if examples_to_skip < len(dataset):
            dataset = torch.utils.data.Subset(dataset, range(examples_to_skip, len(dataset)))
        else:
            dataset = torch.utils.data.Subset(dataset, [])
        batches_to_skip = 0

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        persistent_workers=(num_workers > 0),
    )

    # Prepare
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    if args.compile and device.type == "cuda":
        print("[LM] Compiling model with torch.compile...")
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(output_dir / ".inductor_cache"))
        torch._dynamo.config.cache_size_limit = 64
        model = torch.compile(model)

    # Training
    total_steps = args.max_steps
    eff_batch = args.batch_size * args.grad_accum * accelerator.num_processes
    print(f"[LM] Training: {total_steps} steps, batch={args.batch_size}, "
          f"accum={args.grad_accum}, eff_batch={eff_batch}")

    model.train()
    step = start_step
    running_loss = 0.0
    running_grad_norm = 0.0
    tokens_in_window = 0
    micro_steps_in_window = 0
    window_start = time.time()

    if batches_to_skip > 0:
        print(f"[LM] Fast-forwarding dataloader by {batches_to_skip} batches...")

    while step < total_steps:
        for batch in dataloader:
            if batches_to_skip > 0:
                batches_to_skip -= 1
                continue

            if step >= total_steps:
                break

            with accelerator.accumulate(model):
                loss = compute_lm_loss(model, batch, device)
                accelerator.backward(loss)

                loss_val = accelerator.reduce(loss, "mean").item()
                running_loss += loss_val
                micro_steps_in_window += 1

                tokens_in_window += batch["input_ids"].numel()

                grad_norm_val = 0.0
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    if grad_norm is not None:
                        grad_norm_val = grad_norm.item()

                optimizer.step()

                if accelerator.sync_gradients:
                    if not accelerator.optimizer_step_was_skipped:
                        scheduler.step()
                        step += 1
                        running_grad_norm += grad_norm_val
                    else:
                        print("  [LM] Warning: Gradient overflow detected, skipping step.")

                optimizer.zero_grad(set_to_none=True)

                # Logging
                if step > 0 and step % args.log_steps == 0 and accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                    elapsed = time.time() - window_start
                    avg_loss = running_loss / max(1, micro_steps_in_window)
                    avg_grad_norm = running_grad_norm / args.log_steps

                    tokens_tensor = torch.tensor(tokens_in_window, device=device)
                    global_tokens = accelerator.reduce(tokens_tensor, "sum").item()
                    tok_per_sec = global_tokens / elapsed if elapsed > 0 else 0.0

                    muon_lr = scheduler.get_last_lr()[0]
                    adamw_lr_val = scheduler.get_last_lr()[1] if len(scheduler.get_last_lr()) > 1 else args.lr

                    print(
                        f"  Step {step:>6}/{total_steps} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"GradNorm: {avg_grad_norm:.4f} | "
                        f"LR(Muon): {muon_lr:.2e} | "
                        f"LR(AdamW): {adamw_lr_val:.2e} | "
                        f"Tok/s: {tok_per_sec:,.0f}"
                    )

                    if use_wandb and accelerator.is_main_process:
                        wandb.log({
                            "loss": avg_loss, "grad_norm": avg_grad_norm,
                            "lr_muon": muon_lr, "lr_adamw": adamw_lr_val,
                            "tok_per_sec": tok_per_sec, "step": step,
                        })

                    running_loss = 0.0
                    running_grad_norm = 0.0
                    tokens_in_window = 0
                    micro_steps_in_window = 0
                    window_start = time.time()

                # Checkpointing
                if step > 0 and step % args.save_steps == 0 and accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                    ckpt_dir = output_dir / f"checkpoint-{step}"
                    if accelerator.is_main_process:
                        save_checkpoint(ckpt_dir, model, optimizer, scheduler, step, accelerator)
                        print(f"  Checkpoint saved: {ckpt_dir}")

                        try:
                            local_ckpts = []
                            for item in output_dir.iterdir():
                                if item.is_dir() and item.name.startswith("checkpoint-"):
                                    suffix = item.name.removeprefix("checkpoint-")
                                    if suffix.isdigit():
                                        local_ckpts.append((int(suffix), item))
                            local_ckpts.sort()
                            for _, old_ckpt in local_ckpts[:-3]:
                                print(f"  [LM] Pruning old checkpoint: {old_ckpt.name}")
                                import shutil
                                shutil.rmtree(old_ckpt)
                        except Exception as e:
                            print(f"  [LM] WARNING: checkpoint pruning failed: {e}")

                    accelerator.wait_for_everyone()

    # Final save
    accelerator.wait_for_everyone()
    print("[LM] Training complete!")
    final_dir = output_dir / "final"
    if accelerator.is_main_process:
        final_dir.mkdir(parents=True, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        raw_model = getattr(unwrapped_model, "_orig_mod", unwrapped_model)
        torch.save(
            {"model": raw_model.state_dict(),
             "config": raw_model.config.to_config_dict()},
            final_dir / "model.pt",
        )
        print(f"[LM] Final model saved to {final_dir}")


if __name__ == "__main__":
    train_lm()
