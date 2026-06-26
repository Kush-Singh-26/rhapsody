"""Rhapsody TPU Training Script — Muon + AdamW, WSD schedule, text-only pretraining."""

from __future__ import annotations

import contextlib
import math
import os

# Map standard PyTorch distributed environment variables to PJRT variables for TPU v5e
if "LOCAL_RANK" in os.environ:
    os.environ["PJRT_LOCAL_RANK"] = os.environ["LOCAL_RANK"]
if "LOCAL_WORLD_SIZE" in os.environ:
    os.environ["PJRT_LOCAL_WORLD_SIZE"] = os.environ["LOCAL_WORLD_SIZE"]

# Clean up Kaggle default environment variables that conflict with single-host PJRT initialization
os.environ.pop("TPU_PROCESS_ADDRESSES", None)
os.environ.pop("CLOUD_TPU_TASK_ID", None)

# Prevent OpenMP and Eigen thread over-subscription across 8 rank processes
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Enable latency-hiding scheduler and decompose all-gather einsums in PJRT/XLA for TPU
os.environ["LIBTPU_INIT_ARGS"] = (
    "--xla_tpu_enable_latency_hiding_scheduler=true "
    "--xla_tpu_decompose_all_gather_einsum=true"
)


# Block TensorFlow and JAX from being imported to avoid metrics aggregator conflicts on TPU.
import sys
import types
from importlib.machinery import ModuleSpec

class MockModule(types.ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.__path__ = []
        self.__spec__ = ModuleSpec(name, None)

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return MockModule(f"{self.__name__}.{name}")

    def __call__(self, *args, **kwargs):
        return MockModule(f"{self.__name__}.call")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

class MockImportFinder:
    def __init__(self, mock_names):
        self.mock_names = mock_names
    def find_spec(self, fullname, path, target=None):
        parts = fullname.split(".")
        if parts[0] in self.mock_names:
            return ModuleSpec(fullname, self)
        return None
    def create_module(self, spec):
        return MockModule(spec.name)
    def exec_module(self, module):
        pass

sys.meta_path.insert(0, MockImportFinder({"tensorflow", "jax", "jaxlib"}))

sys.modules["tensorflow"] = MockModule("tensorflow")
sys.modules["jax"] = MockModule("jax")
sys.modules["jaxlib"] = MockModule("jaxlib")

# Patch xmp.spawn to force start_method='spawn'
try:
    import torch_xla.distributed.xla_multiprocessing as xmp
    _orig_spawn = xmp.spawn
    def _custom_spawn(fn, args=(), nprocs=None, start_method='spawn', join=True, daemon=False):
        return _orig_spawn(fn, args=args, nprocs=nprocs, start_method='spawn', join=join, daemon=daemon)
    xmp.spawn = _custom_spawn
except Exception as e:
    print(f"[Rhapsody] Warning: failed to patch xmp.spawn: {e}")

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

from .model import create_text_only_65m
from .data import get_tokenizer, TextPretrainDataset, DataCollatorWithPadding, PreTokenizedDataset


# =============================================================================
# WSD Learning Rate Schedule
# =============================================================================

def get_wsd_lr(step: int, total_steps: int, warmup_frac: float = 0.01, decay_frac: float = 0.12) -> float:
    """Warmup-Stable-Decay schedule."""
    warmup_steps = int(total_steps * warmup_frac)
    decay_start = int(total_steps * (1.0 - decay_frac))

    if step < warmup_steps:
        return step / max(1, warmup_steps)
    elif step < decay_start:
        return 1.0
    else:
        progress = (step - decay_start) / max(1, total_steps - decay_start)
        return (1 + math.cos(math.pi * progress)) / 2


# =============================================================================
# Muon Optimizer
# =============================================================================

class Muon(torch.optim.Optimizer):
    """Muon: Momentum + Newton-Schulz orthogonalisation for weight matrices."""

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

def compute_loss(model: nn.Module, batch: dict, device: torch.device) -> torch.Tensor:
    """Compute next-token prediction loss with auxiliary Z-loss."""
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    output = model(input_ids, labels=labels)
    loss = output["loss"]
    logits = output["logits"]

    # Auxiliary Z-loss
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
    elif accelerator.device.type == "xla":
        try:
            import torch_xla.core.xla_model as xm
            payload["xla_rng_state"] = xm.get_rng_state()
        except ImportError:
            pass

    accelerator.save(payload, ckpt_dir / "model.pt")
    accelerator.save({"optimizer": optimizer.state_dict()}, ckpt_dir / "optimizer.pt")
    accelerator.save({"scheduler": scheduler.state_dict()}, ckpt_dir / "scheduler.pt")


# =============================================================================
# Training Loop
# =============================================================================

def train():
    import argparse
    parser = argparse.ArgumentParser(description="Rhapsody TPU TextLM Pre-Training")
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0008,
                        help="AdamW LR (for embeddings/norms/biases).")
    parser.add_argument("--muon-lr", type=float, default=0.015,
                        help="Muon LR (for 2D+ weight matrices). Default: 0.015.")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint directory to resume from.")
    parser.add_argument("--auto-resume", action="store_true",
                        help="Resume from the latest local checkpoint, or Hub checkpoint when --forge-config is set.")
    parser.add_argument("--forge-config", type=str, default=None,
                        help="Optional path to a lm_forge config.yaml for Hub pull/push checkpoint sync.")
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="Enable gradient checkpointing to trade compute for memory.")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile on CUDA/TPU.")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of DataLoader workers.")
    parser.add_argument("--pretok-dir", type=str, default=None,
                        help="Path to pre-tokenized shard directory produced by pretokenize.py.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Detect best mixed precision
    mixed_precision = "no"
    is_tpu = False
    try:
        import torch_xla.core.xla_model as xm
        is_tpu = True
    except ImportError:
        pass

    if is_tpu:
        mixed_precision = "bf16"
    elif torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        mixed_precision = "bf16" if major >= 8 else "fp16"

    # Setup diagnostic logs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    local_rank = 0
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        try:
            import torch_xla.core.xla_model as xm
            local_rank = xm.get_ordinal()
        except Exception:
            pass

    _slog_path = output_dir / f"rank_{local_rank}_setup.log"
    _slog_fh = open(_slog_path, "w", buffering=1)
    def _slog(msg: str):
        ts = time.strftime("%H:%M:%S")
        _slog_fh.write(f"[{ts}][R{local_rank}] {msg}\n")
        _slog_fh.flush()
    _slog("=== Setup start ===")

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
            run_id = hashlib.md5(str(output_dir.resolve()).encode()).hexdigest()[:8]
            wandb.init(project="rhapsody-pretrain-tpu", config=vars(args), resume="allow", id=f"lm-tpu-{run_id}")
            use_wandb = wandb.run is not None
        except Exception as e:
            print(f"[Rhapsody] WARNING: wandb failed: {e}")
            use_wandb = False

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"[Rhapsody] Device: {device}")

    # Tokenizer & Model
    tokenizer = get_tokenizer(symbolic=False)
    vocab_size = len(tokenizer)

    model = create_text_only_65m(vocab_size=vocab_size)
    model.config.gradient_checkpointing = args.grad_checkpoint
    model = model.to(device)

    def tie_weights(model: nn.Module) -> None:
        if hasattr(model, "config") and getattr(model.config, "tie_word_embeddings", False):
            if hasattr(model, "lm_head") and hasattr(model, "embed_tokens"):
                model.lm_head.weight = model.embed_tokens.weight
                print("[Rhapsody] Explicitly tied TextLM word embeddings on device.")

    tie_weights(model)

    # Optimizer (Muon + AdamW)
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
        lr=args.muon_lr, momentum=0.95, nesterov=True, ns_steps=5,
        adamw_params=None,
    )

    optimizer.add_param_group({
        "params": adamw_no_decay_params,
        "lr": args.lr,
        "betas": (0.9, 0.95),
        "weight_decay": 0.0,
        "is_adamw": True,
    })

    # Scheduler (WSD only)
    total_opt_steps = args.max_steps
    print("[Rhapsody] Scheduler: Warmup-Stable-Decay (WSD)")
    lr_fn = lambda step, n=total_opt_steps: get_wsd_lr(step, n)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[lr_fn] * len(optimizer.param_groups),
    )

    # Nomad Checkpointing
    start_step = 0
    hub_manager = None
    if args.forge_config:
        try:
            import yaml
            from lm_forge.forge import HubCheckpointManager
            with open(args.forge_config) as fh:
                forge_cfg = yaml.safe_load(fh)
            hub_manager = HubCheckpointManager(forge_cfg, output_dir)
        except ImportError:
            print("[Rhapsody] lm_forge not installed. Bypassing Hub checkpoint manager.")

    if args.auto_resume:
        if hub_manager is not None:
            if accelerator.is_main_process:
                print("[Rhapsody] Auto-resume: checking HF Hub...")
                try:
                    hub_manager.pull_latest()
                except Exception as e:
                    print(f"[Rhapsody] Warning: pull latest failed: {e}")
            
            _sentinel = output_dir / "hub_sync_complete.sentinel"
            if accelerator.is_main_process:
                _sentinel.touch()
            else:
                while not _sentinel.exists():
                    time.sleep(0.5)
            time.sleep(2)
            if accelerator.is_main_process:
                _sentinel.unlink(missing_ok=True)
        else:
            accelerator.wait_for_everyone()

        latest_local = find_latest_checkpoint(output_dir)
        if latest_local is not None:
            args.resume = str(latest_local)
            print(f"[Rhapsody] Resuming from local checkpoint: {latest_local}")

    if args.resume:
        ckpt_path = Path(args.resume)
        model_pt = ckpt_path / "model.pt"
        if model_pt.exists():
            print(f"[Rhapsody] Resuming from {ckpt_path}")
            ckpt = torch.load(model_pt, map_location="cpu", weights_only=True)
            state_dict = ckpt["model"]
            if any(k.startswith("module.") for k in state_dict):
                state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            start_step = ckpt.get("step", 0)
            if "rng_state" in ckpt:
                try:
                    torch.set_rng_state(ckpt["rng_state"].cpu().byte())
                except Exception as e:
                    print(f"[Rhapsody] WARNING: Failed to restore CPU RNG state: {e}")
            if "python_random_state" in ckpt:
                try:
                    random.setstate(ckpt["python_random_state"])
                except Exception as e:
                    print(f"[Rhapsody] WARNING: Failed to restore Python random state: {e}")
            if torch.cuda.is_available() and "cuda_rng_state_all" in ckpt:
                try:
                    rng_states = [s.cpu().byte() if isinstance(s, torch.Tensor) else s for s in ckpt["cuda_rng_state_all"]]
                    torch.cuda.set_rng_state_all(rng_states)
                except Exception as e:
                    print(f"[Rhapsody] WARNING: Failed to restore CUDA RNG state: {e}")
            elif device.type == "xla" and "xla_rng_state" in ckpt:
                try:
                    import torch_xla.core.xla_model as xm
                    xm.set_rng_state(ckpt["xla_rng_state"])
                except Exception as e:
                    print(f"[Rhapsody] WARNING: Failed to restore XLA RNG state: {e}")
            print(f"[Rhapsody] Resumed at step {start_step}")

    # ── Dataset Loading ──────────────────────────────────────────────────────
    global_batch_size = args.batch_size * args.grad_accum * accelerator.num_processes
    _slog(f"Creating dataset...")
    if args.pretok_dir and Path(args.pretok_dir).exists():
        print(f"[Rhapsody] Using pre-tokenized dataset from {args.pretok_dir}")
        dataset = PreTokenizedDataset(args.pretok_dir, seq_len=args.seq_len)
    else:
        dataset = TextPretrainDataset(
            tokenizer,
            seq_len=args.seq_len,
            resume_step=start_step,
            global_batch_size=global_batch_size
        )
    _slog("Dataset created.")

    # ── DataLoader ────────────────────────────────────────────────────────────
    is_iterable = isinstance(dataset, IterableDataset)
    is_pretok   = isinstance(dataset, PreTokenizedDataset)

    if is_pretok:
        num_workers = 0
        collate_fn  = None
    elif is_iterable:
        num_workers = 0
        collate_fn  = None
    else:
        num_workers = args.num_workers if args.num_workers is not None else min(4, os.cpu_count() or 1)
        collate_fn  = DataCollatorWithPadding(tokenizer)

    batches_to_skip = start_step * args.grad_accum
    if getattr(dataset, 'already_fast_forwarded', False):
        batches_to_skip = 0
        print(f"[Rhapsody] Pre-tokenized dataset starts at resume point (step {start_step}).")
    elif not is_iterable and batches_to_skip > 0:
        examples_to_skip = batches_to_skip * args.batch_size * accelerator.num_processes
        print(f"[Rhapsody] Fast-forwarding map-style dataset by {examples_to_skip} examples...")
        if examples_to_skip < len(dataset):
            dataset = torch.utils.data.Subset(dataset, range(examples_to_skip, len(dataset)))
        else:
            dataset = torch.utils.data.Subset(dataset, [])
        batches_to_skip = 0
    elif is_iterable:
        batches_to_skip = 0

    sampler = None
    if is_pretok and accelerator.num_processes > 1:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
        print(f"[Rhapsody] DistributedSampler created for pretok dataset.")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        persistent_workers=False,
        drop_last=True,
        sampler=sampler,
        shuffle=(sampler is None and not is_iterable and not is_pretok),
    )
    _slog("DataLoader created.")

    model, optimizer = accelerator.prepare(model, optimizer)
    _slog("accelerator.prepare() complete.")

    if device.type == "xla":
        try:
            import torch_xla.distributed.parallel_loader as pl
            print(f"[Rhapsody] Wrapping dataloader in MpDeviceLoader for {device}...")
            dataloader = pl.MpDeviceLoader(dataloader, device)
        except Exception as e:
            print(f"[Rhapsody] Warning: failed to wrap dataloader: {e}")

    # Load Optimizer & Scheduler states after prepare
    if args.resume:
        opt_pt = Path(args.resume) / "optimizer.pt"
        sched_pt = Path(args.resume) / "scheduler.pt"
        if opt_pt.exists():
            print(f"[Rhapsody] Loading optimizer state from {opt_pt}...")
            opt_ckpt = torch.load(opt_pt, map_location="cpu", weights_only=True)
            saved_state_dict = opt_ckpt["optimizer"]
            try:
                saved_param_ids = []
                for group in saved_state_dict["param_groups"]:
                    saved_param_ids.extend(group["params"])
                
                active_params = []
                for group in optimizer.param_groups:
                    active_params.extend(group["params"])
                
                if len(saved_param_ids) == len(active_params):
                    print("[Rhapsody] Aligning optimizer state dict parameter groups...")
                    new_state = {}
                    for active_p, saved_pid in zip(active_params, saved_param_ids):
                        if saved_pid in saved_state_dict["state"]:
                            new_state[active_p] = {
                                k: (v.to(active_p.device) if torch.is_tensor(v) else v)
                                for k, v in saved_state_dict["state"][saved_pid].items()
                            }
                    optimizer.state.clear()
                    optimizer.state.update(new_state)
                    
                    if "_adam_step" in saved_state_dict:
                        optimizer._adam_step = saved_state_dict["_adam_step"]
                    print("[Rhapsody] Optimizer state aligned and loaded successfully.")
                else:
                    print("[Rhapsody] WARNING: Optimizer parameter count mismatch. Fallback to standard load.")
                    optimizer.load_state_dict(saved_state_dict)
            except Exception as e:
                print(f"[Rhapsody] WARNING: Failed to align optimizer: {e}. Fallback to standard load.")
                optimizer.load_state_dict(saved_state_dict)
        if sched_pt.exists():
            print(f"[Rhapsody] Loading scheduler state from {sched_pt}")
            sched_ckpt = torch.load(sched_pt, map_location="cpu", weights_only=True)
            scheduler.load_state_dict(sched_ckpt["scheduler"])
        else:
            scheduler.last_epoch = start_step

    if args.compile:
        print("[Rhapsody] Compiling model with torch.compile...")
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(output_dir / ".inductor_cache"))
        torch._dynamo.config.cache_size_limit = 64
        if device.type == "xla":
            try:
                model = torch.compile(model, backend="openxla")
                print("[Rhapsody] torch.compile: using openxla backend (TPU).")
            except Exception as e:
                print(f"[Rhapsody] openxla backend unavailable ({e}). Falling back to default compile.")
                model = torch.compile(model)
        else:
            model = torch.compile(model)

    # All-ranks barrier
    if device.type == "xla" and accelerator.num_processes > 1:
        _rdy = output_dir / f".rank_{accelerator.process_index}_ready"
        _rdy.touch()
        _slog(f"Ready-file written. Waiting for all {accelerator.num_processes} ranks...")
        print(f"[Rhapsody] Rank {accelerator.process_index}: at all-ranks barrier...")
        _bd = time.time() + 600
        while time.time() < _bd:
            n_rdy = sum(
                1 for i in range(accelerator.num_processes)
                if (output_dir / f".rank_{i}_ready").exists()
            )
            if n_rdy >= accelerator.num_processes:
                break
            time.sleep(0.5)
        time.sleep(2)
        _rdy.unlink(missing_ok=True)
        _slog("All-ranks barrier passed. Entering training loop.")
        if accelerator.is_main_process:
            print("[Rhapsody] All-ranks barrier passed. Starting training loop.")

    # ── Training ─────────────────────────────────────────────────────────────
    total_steps = args.max_steps
    eff_batch = args.batch_size * args.grad_accum * accelerator.num_processes
    print(f"[Rhapsody] Training: {total_steps} optimizer steps, "
          f"batch={args.batch_size}, accum={args.grad_accum}, eff_batch={eff_batch}")

    model.train()
    step = start_step
    running_loss = torch.tensor(0.0, device=device)
    running_grad_norm = torch.tensor(0.0, device=device)
    tokens_in_window = 0
    micro_steps_in_window = 0
    window_start = time.time()

    _rlog_path = output_dir / f"rank_{accelerator.process_index}_train.log"
    _rlog_fh = open(_rlog_path, "w", buffering=1)
    def _rlog(msg: str):
        ts = time.strftime("%H:%M:%S")
        _rlog_fh.write(f"[{ts}][R{accelerator.process_index}] {msg}\n")
        _rlog_fh.flush()

    _rlog(f"=== Training start: step={step}, total_steps={total_steps} ===")
    print(f"[Rhapsody] Process {accelerator.process_index} starting training loop...")

    _xla_sync_fn = None
    if device.type == "xla":
        try:
            import torch_xla as _txla
            _xla_sync_fn = _txla.sync
        except AttributeError:
            import torch_xla.core.xla_model as _xm
            _xla_sync_fn = _xm.mark_step

    while step < total_steps:
        for batch in dataloader:
            if step >= total_steps:
                break

            with accelerator.accumulate(model):
                loss = compute_loss(model, batch, device)
                accelerator.backward(loss)

                loss_val_tensor = loss.detach()

                grad_norm_tensor = None
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    if grad_norm is not None:
                        grad_norm_tensor = grad_norm.detach()

                optimizer.step()

                if accelerator.sync_gradients:
                    if not accelerator.optimizer_step_was_skipped:
                        scheduler.step()
                        step += 1
                    else:
                        print("  [Rhapsody] Warning: Gradient overflow detected, skipping optimizer step.")

                optimizer.zero_grad(set_to_none=True)

                if _xla_sync_fn is not None:
                    _xla_sync_fn()

                running_loss += loss_val_tensor
                micro_steps_in_window += 1

                batch_text_tokens = batch["input_ids"].numel()
                tokens_in_window += batch_text_tokens

                if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                    if grad_norm_tensor is not None:
                        running_grad_norm += grad_norm_tensor

                # ── Logging ──────────────────────────────────────────────────────
                if step > 0 and step % args.log_steps == 0 and accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                    elapsed = time.time() - window_start
                    avg_loss = (running_loss / max(1, micro_steps_in_window)).item()
                    avg_grad_norm = (running_grad_norm / args.log_steps).item()
                    
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
                        wandb.log({"loss": avg_loss, "grad_norm": avg_grad_norm, "lr_muon": muon_lr, "lr_adamw": adamw_lr_val,
                                   "tok_per_sec": tok_per_sec, "step": step})
                                   
                    running_loss.fill_(0.0)
                    running_grad_norm.fill_(0.0)
                    tokens_in_window = 0
                    micro_steps_in_window = 0
                    window_start = time.time()

                # ── Checkpointing ─────────────────────────────────────────────────
                if step > 0 and step % args.save_steps == 0 and accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                    ckpt_dir = output_dir / f"checkpoint-{step}"
                    save_checkpoint(ckpt_dir, model, optimizer, scheduler, step, accelerator)
                    
                    if accelerator.is_main_process:
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
                                print(f"  [Rhapsody] Pruning old local checkpoint: {old_ckpt.name}")
                                import shutil
                                shutil.rmtree(old_ckpt)
                        except Exception as e:
                            print(f"  [Rhapsody] WARNING: local checkpoint pruning failed: {e}")

                        if hub_manager is not None:
                            def upload_and_prune(ckpt_dir, step):
                                try:
                                    hub_manager.upload_checkpoint(ckpt_dir, step)
                                    hub_manager.prune_checkpoints()
                                except Exception as e:
                                    print(f"  [Rhapsody] Hub sync error at step {step}: {e}")

                            import threading
                            threading.Thread(
                                target=upload_and_prune,
                                args=(ckpt_dir, step),
                                daemon=False
                            ).start()
                            
                    if device.type == "xla" and accelerator.num_processes > 1:
                        import time as _t
                        _ckpt_sentinel = output_dir / ".ckpt_sync_ready"
                        if accelerator.is_main_process:
                            _ckpt_sentinel.touch()
                        else:
                            _td = _t.time()
                            while not _ckpt_sentinel.exists():
                                if _t.time() - _td > 120: break
                                _t.sleep(0.3)
                        _t.sleep(1)
                        if accelerator.is_main_process:
                            _ckpt_sentinel.unlink(missing_ok=True)
                    else:
                        accelerator.wait_for_everyone()

    _rlog("=== Training loop complete ===")
    _rlog_fh.close()

    print("[Rhapsody] Training complete!")
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(model)
    raw_model = getattr(unwrapped_model, "_orig_mod", unwrapped_model)
    
    accelerator.save(
        {"model": raw_model.state_dict(),
         "config": raw_model.config.to_config_dict()
         if hasattr(raw_model, "config") else {}},
        final_dir / "model.pt",
    )
    if accelerator.is_main_process:
        print(f"[Rhapsody] Final model saved to {final_dir}")


if __name__ == "__main__":
    train()
