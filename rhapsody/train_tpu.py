"""Rhapsody Training Script — Muon + AdamW, WSD schedule, 3-stage pipeline."""

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

# Block TensorFlow and JAX from being imported to avoid metrics aggregator conflicts on TPU.
# We map them to a dummy MockModule in sys.modules and register a MockImportFinder in sys.meta_path.
# This prevents actual imports (which load libtpu.so) while satisfying any module-level imports
# (like transformers.image_transforms which imports jax.numpy and tensorflow unconditionally on Kaggle).
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

# CRITICAL: Patch xmp.spawn to force start_method='spawn' (defaults to 'fork' in some environments).
# This prevents child processes from inheriting initialized XLA/libtpu C++ states from the parent,
# which causes the "Check failed: reporting_closure_ == nullptr" crash under PJRT.
try:
    import torch_xla.distributed.xla_multiprocessing as xmp
    _orig_spawn = xmp.spawn
    def _custom_spawn(fn, args=(), nprocs=None, start_method='spawn', join=True, daemon=False):
        return _orig_spawn(fn, args=args, nprocs=nprocs, start_method='spawn', join=join, daemon=daemon)
    xmp.spawn = _custom_spawn
except Exception as e:
    print(f"[Rhapsody] Warning: failed to patch xmp.spawn: {e}")

import random
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo
from torch.utils.data import DataLoader, IterableDataset
from accelerate import Accelerator

try:
    from .model import RhapsodyConfig, create_text_only_65m, create_rhapsody_65m
    from .data import get_tokenizer, TextPretrainDataset, AudioTextDataset, SymbolicMusicDataset, DataCollatorWithPadding, PreTokenizedDataset
except ImportError:
    from model import RhapsodyConfig, create_text_only_65m, create_rhapsody_65m
    from data import get_tokenizer, TextPretrainDataset, AudioTextDataset, SymbolicMusicDataset, DataCollatorWithPadding, PreTokenizedDataset


# =============================================================================
# WSD Learning Rate Schedule  (Warmup → Stable → Cosine Decay)
# =============================================================================

def get_wsd_lr(step: int, total_steps: int, warmup_frac: float = 0.01, decay_frac: float = 0.12) -> float:
    """
    Returns a [0, 1] multiplier for the peak LR.
      • Linear warmup for the first warmup_frac fraction of steps.
      • Constant peak for the stable middle portion.
      • Cosine decay for the final decay_frac fraction of steps.
    """
    warmup_steps = int(total_steps * warmup_frac)
    decay_start = int(total_steps * (1.0 - decay_frac))

    if step < warmup_steps:
        return step / max(1, warmup_steps)
    elif step < decay_start:
        return 1.0
    elif step < total_steps:
        progress = (step - decay_start) / max(1, total_steps - decay_start)
        return (1 + math.cos(math.pi * progress)) / 2
    else:
        return 0.0


# =============================================================================
# Cosine Learning Rate Schedule  (Warmup → Cosine Decay)
# =============================================================================

def get_cosine_lr(step: int, total_steps: int, warmup_frac: float = 0.03) -> float:
    """
    Returns a [0, 1] multiplier for the peak LR.
      • Linear warmup for the first warmup_frac fraction of steps.
      • Cosine decay to zero for the remaining steps.
    """
    warmup_steps = int(total_steps * warmup_frac)
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    elif step < total_steps:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        return 0.0


# =============================================================================
# Muon Optimizer  (Newton-Schulz orthogonalised SGD + AdamW for scalars/embeds)
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
        # super().__init__ initialises self.state and self.param_groups correctly
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

        self._adam_step = 0  # step counter for AdamW bias correction

    def _newton_schulz(self, M: torch.Tensor) -> torch.Tensor:
        """
        5-step Newton-Schulz iteration to orthogonalise a 2D matrix.
        Handles 2D weight matrices; for higher-rank tensors, reshapes to 2D.
        """
        orig_shape = M.shape
        if M.ndim > 2:
            M = M.view(M.shape[0], -1)

        # Transpose if tall to optimize compute (FLOPs) and satisfy orthogonal projection requirements
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
                # ── AdamW update (embeddings, LM head, norms, biases) ─────
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
                # ── Muon update (weight matrices: attention, FFN projections) ──
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
# Loss Computation
# =============================================================================

def compute_loss(model: nn.Module, batch: dict, device: torch.device) -> torch.Tensor:
    """
    Compute next-token prediction loss with auxiliary Z-loss (alpha=1e-4) to prevent logit explosion.

    Labels from both TextPretrainDataset and AudioTextDataset are pre-shifted:
    labels[t] = next token after input_ids[t].  No further shift is applied here.
    """
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)                      # pre-shifted
    audio_features = batch.get("audio_features")

    if audio_features is not None:
        # Stage 2 / 3: multimodal batch — model is RhapsodyModel
        output = model(input_ids, audio_features=audio_features.to(device), labels=labels)
    else:
        # Stage 1: text-only batch — model may be TextLM or RhapsodyModel
        # TextLM.forward() has no audio_features param, so call without it
        output = model(input_ids, labels=labels)

    loss = output["loss"]
    logits = output["logits"]

    # Compute auxiliary Z-loss to prevent logit explosion (Issue #13)
    if logits is not None and labels is not None:
        if audio_features is not None:
            # Multimodal: logits has audio prefix, but labels only correspond to the text portion
            audio_len = logits.shape[1] - labels.shape[1]
            active_logits = logits[:, audio_len:, :]
        else:
            active_logits = logits

        flat_logits = active_logits.view(-1, active_logits.size(-1))
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


def extract_text_lm_state(state_dict: dict) -> dict:
    """Accept either a TextLM checkpoint or a RhapsodyModel checkpoint."""
    if any(k.startswith("text_lm.") for k in state_dict):
        return {
            k.removeprefix("text_lm."): v
            for k, v in state_dict.items()
            if k.startswith("text_lm.")
        }
    return state_dict


def load_pretrained_text_lm(model: nn.Module, checkpoint_dir: str | Path) -> None:
    lm_ckpt_path = Path(checkpoint_dir) / "model.pt"
    if not lm_ckpt_path.exists():
        print(f"[Rhapsody] WARNING: --pretrained-lm path not found: {lm_ckpt_path}")
        return

    print(f"[Rhapsody] Loading pretrained LM from {lm_ckpt_path}")
    ckpt = torch.load(lm_ckpt_path, map_location="cpu", weights_only=True)
    raw_state = ckpt.get("model", ckpt)
    lm_state = extract_text_lm_state(raw_state)
    missing, unexpected = model.text_lm.load_state_dict(lm_state, strict=False)
    if unexpected:
        print(f"[Rhapsody] WARNING: ignored unexpected LM keys: {len(unexpected)}")
    if missing:
        print(f"[Rhapsody] WARNING: missing LM keys: {len(missing)}")
    print("[Rhapsody] Pretrained LM weights loaded.")


class FallbackHubManager:
    def __init__(self, config_path: str):
        import yaml
        import json
        import tempfile
        from pathlib import Path
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        
        self.project_name = cfg.get("name", "rhapsody-65m")
        state_cfg = cfg.get("state", {})
        self.repo_id = state_cfg.get("repo_id", "Kush26/rhapsody-65m-checkpoints")
        self.branch = state_cfg.get("branch", "checkpoints")
        self.checkpoint_limit = state_cfg.get("checkpoint_limit", 3)
        self.private = state_cfg.get("private", True)
        
        # Token
        self.token = os.environ.get("HF_TOKEN")
        if not self.token:
            try:
                from dotenv import load_dotenv
                cfg_path = Path(config_path)
                load_dotenv(dotenv_path=cfg_path.parent / ".env")
                self.token = os.environ.get("HF_TOKEN")
            except Exception:
                pass
        
        if not self.token:
            print("[Rhapsody.FallbackHub] WARNING: HF_TOKEN env var not set.")
            
        from huggingface_hub import HfApi
        self.api = HfApi(token=self.token)
        self.remote_prefix = f"checkpoints/{self.project_name}"
        
        # Ensure branch exists
        try:
            self.api.create_branch(repo_id=self.repo_id, branch=self.branch, exist_ok=True)
        except Exception as e:
            print(f"[Rhapsody.FallbackHub] WARNING: Branch creation failed: {e}")

    def upload_checkpoint(self, local_dir: str | Path, step: int, retries: int = 3) -> None:
        import time
        import json
        import tempfile
        local_dir = Path(local_dir)
        path_in_repo = f"{self.remote_prefix}/checkpoint-{step}"
        for attempt in range(retries):
            try:
                print(f"[Rhapsody.FallbackHub] Uploading checkpoint-{step} to '{self.repo_id}' [{self.branch}] (Attempt {attempt+1}/{retries})...")
                self.api.upload_folder(
                    repo_id=self.repo_id,
                    folder_path=str(local_dir),
                    path_in_repo=path_in_repo,
                    commit_message=f"Upload checkpoint-{step}",
                    allow_patterns=["*"],
                    revision=self.branch,
                )
                self._update_latest_pointer(step)
                print(f"[Rhapsody.FallbackHub] Upload complete & verified: {path_in_repo}")
                return
            except Exception as e:
                print(f"[Rhapsody.FallbackHub] ERROR: Upload attempt {attempt+1} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(5 * (attempt + 1))

    def _update_latest_pointer(self, step: int):
        import json
        import tempfile
        pointer_path = f"{self.remote_prefix}/latest.json"
        data = {"latest_step": step, "timestamp": time.time()}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(data, tmp)
            tmp_path = tmp.name
        try:
            self.api.upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo=pointer_path,
                repo_id=self.repo_id,
                revision=self.branch,
                commit_message=f"Update latest pointer to step {step}"
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def get_latest_verified_step(self) -> Optional[int]:
        import json
        pointer_path = f"{self.remote_prefix}/latest.json"
        try:
            from huggingface_hub import hf_hub_download
            local_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=pointer_path,
                revision=self.branch,
                token=self.token
            )
            with open(local_path, "r") as f:
                return json.load(f).get("latest_step")
        except Exception:
            return None

    def prune_checkpoints(self) -> None:
        steps = self.list_remote_checkpoints()
        if len(steps) <= self.checkpoint_limit:
            return
        to_delete = steps[:-self.checkpoint_limit]
        for step in to_delete:
            path_in_repo = f"{self.remote_prefix}/checkpoint-{step}"
            try:
                self.api.delete_folder(
                    repo_id=self.repo_id,
                    path_in_repo=path_in_repo,
                    commit_message=f"Prune old checkpoint-{step}",
                    revision=self.branch,
                )
                print(f"[Rhapsody.FallbackHub] Pruned: {path_in_repo}")
            except Exception as e:
                print(f"[Rhapsody.FallbackHub] ERROR: Deletion failed for {path_in_repo}: {e}")

    def list_remote_checkpoints(self) -> list[int]:
        try:
            repo_tree = self.api.list_repo_tree(
                repo_id=self.repo_id,
                path_in_repo=self.remote_prefix,
                recursive=False,
                revision=self.branch,
            )
            steps = []
            for item in repo_tree:
                path_str = str(item.path)
                if "checkpoint-" in path_str:
                    folder_name = Path(path_str).name
                    step_str = folder_name.split("checkpoint-")[-1]
                    if step_str.isdigit():
                        steps.append(int(step_str))
            return sorted(steps)
        except Exception:
            return []

    def get_local_checkpoints(self, local_root: str | Path) -> list[int]:
        local_root = Path(local_root)
        if not local_root.exists():
            return []
        steps = []
        for item in local_root.iterdir():
            if item.is_dir() and "checkpoint-" in item.name:
                step_str = item.name.split("checkpoint-")[-1]
                if step_str.isdigit():
                    steps.append(int(step_str))
        return sorted(steps)

    def pull_latest(self, local_root: str | Path, force: bool = False) -> Optional[Path]:
        from huggingface_hub import snapshot_download
        import shutil
        local_root = Path(local_root)
        latest_verified = self.get_latest_verified_step()
        remote_steps = self.list_remote_checkpoints()
        if not remote_steps:
            print("[Rhapsody.FallbackHub] No remote checkpoints found.")
            return None
        latest_remote = latest_verified if latest_verified is not None else remote_steps[-1]
        local_steps = self.get_local_checkpoints(local_root)
        latest_local = local_steps[-1] if local_steps else -1
        if latest_remote <= latest_local and not force:
            print(f"[Rhapsody.FallbackHub] Local state (step {latest_local}) is up-to-date with Hub (step {latest_remote}).")
            return local_root / f"checkpoint-{latest_local}"
        if force and local_root.exists():
            print(f"[Rhapsody.FallbackHub] Force pull requested. Clearing {local_root}...")
            shutil.rmtree(local_root)
            local_root.mkdir(parents=True, exist_ok=True)
        
        target_steps = [latest_remote] if latest_remote in remote_steps else reversed(remote_steps)
        for latest_step in target_steps:
            path_in_repo = f"{self.remote_prefix}/checkpoint-{latest_step}"
            target_path = local_root.resolve() / f"checkpoint-{latest_step}"
            print(f"[Rhapsody.FallbackHub] Pulling checkpoint-{latest_step}...")
            try:
                snapshot_download(
                    repo_id=self.repo_id,
                    allow_patterns=[f"{path_in_repo}/*"],
                    local_dir=str(local_root),
                    token=self.token,
                    revision=self.branch,
                    local_dir_use_symlinks=False
                )
                downloaded_path = local_root / path_in_repo
                if downloaded_path.exists():
                    if target_path.exists() and target_path != downloaded_path:
                        shutil.rmtree(target_path)
                    if target_path != downloaded_path:
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(downloaded_path), str(target_path))
                        current = downloaded_path.parent
                        while current != local_root:
                            try:
                                current.rmdir()
                                current = current.parent
                            except OSError:
                                break
                # check model file exists
                has_model = (
                    (target_path / "model.safetensors").exists()
                    or (target_path / "pytorch_model.bin").exists()
                    or (target_path / "model.bin").exists()
                    or (target_path / "model.pt").exists()
                )
                if has_model:
                    print(f"[Rhapsody.FallbackHub] Pull complete: {target_path}")
                    return target_path
            except Exception as e:
                print(f"[Rhapsody.FallbackHub] ERROR: Pull failed: {e}")
        return None


def init_forge_hub(config_path: Optional[str]):
    if not config_path:
        return None
    try:
        from dotenv import load_dotenv
        from forge.config import ForgeConfig
        from forge.state.hub_manager import HubManager

        cfg_path = Path(config_path)
        load_dotenv(dotenv_path=cfg_path.parent / ".env")
        cfg = ForgeConfig.load(cfg_path)
        return HubManager(cfg.state, cfg.name)
    except Exception as e:
        print(f"[Rhapsody] Forge package not found or failed to load. Falling back to self-contained FallbackHubManager: {e}")
        try:
            return FallbackHubManager(config_path)
        except Exception as ex:
            import traceback
            print(f"[Rhapsody] ERROR: FallbackHubManager also failed: {ex}")
            traceback.print_exc()
            return None


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
    parser = argparse.ArgumentParser(description="Rhapsody Training")
    parser.add_argument("--task", type=str, default="audio-captioning",
                        choices=["audio-captioning", "symbolic-music"],
                        help="Task type: audio-captioning (default) | symbolic-music")
    parser.add_argument("--stage", type=str, default="pretrain",
                        choices=["pretrain", "align", "finetune"],
                        help="Training stage: pretrain | align | finetune")
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
                        help="Optional path to a lm_forge forge.yaml for Hub pull/push checkpoint sync.")
    parser.add_argument("--pretrained-lm", type=str, default=None,
                        help="Path to a Stage-1 checkpoint dir to initialise the text LM "
                             "before Stage-2 alignment training.")
    parser.add_argument("--grad-checkpoint", action="store_true",
                        help="Enable gradient checkpointing to trade compute for memory.")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile on CUDA. Useful for longer Colab sessions after warmup.")
    parser.add_argument("--symbolic-dataset", type=str, default=None,
                        help="Path to a local JSONL file for symbolic music training.")
    parser.add_argument("--symbolic-hf", type=str, default="Seeker38/music_abc_notation",
                        help="HuggingFace dataset for symbolic music (default: Seeker38/music_abc_notation).")
    parser.add_argument("--symbolic-max-examples", type=int, default=None,
                        help="Max examples to load from symbolic dataset.")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="Number of DataLoader workers. Defaults to CPU count limited.")
    parser.add_argument("--pretok-dir", type=str, default=None,
                        help="Path to pre-tokenized shard directory produced by pretokenize.py. "
                             "When set, bypasses streaming tokenization for ~3x faster throughput "
                             "on TPU. Only applies to --stage pretrain.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Detect best mixed precision based on device capabilities
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

    # ── Output directory & Per-rank SETUP diagnostic log ────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # We need to know our rank for early logging.
    local_rank = 0
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        try:
            import torch_xla.core.xla_model as xm
            local_rank = xm.get_ordinal()
        except (ImportError, RuntimeError, Exception):
            pass
    
    _slog_path = output_dir / f"rank_{local_rank}_setup.log"
    _slog_fh = open(_slog_path, "w", buffering=1)
    def _slog(msg: str):
        ts = time.strftime("%H:%M:%S")
        _slog_fh.write(f"[{ts}][R{local_rank}] {msg}\n")
        _slog_fh.flush()
    _slog(f"=== Setup start | pid={os.getpid()} ===")

    _slog("Initializing Accelerator...")
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=args.grad_accum,
        step_scheduler_with_optimizer=False
    )
    device = accelerator.device
    print = accelerator.print
    _slog(f"Accelerator initialized. device={device}")


    use_wandb = False
    if accelerator.is_main_process:
        try:
            import wandb
            import hashlib
            # If wandb is installed, default to disabled mode unless explicitly configured
            # via WANDB_API_KEY or WANDB_MODE env vars to prevent blocking in notebooks.
            if not os.environ.get("WANDB_API_KEY") and not os.environ.get("WANDB_MODE"):
                os.environ["WANDB_MODE"] = "disabled"
                print("[Rhapsody] wandb unconfigured: defaulting to disabled mode to prevent interactive prompts.")
            
            run_id = hashlib.md5(str(Path(args.output_dir).resolve()).encode()).hexdigest()[:8]
            wandb.init(project="rhapsody", config=vars(args), resume="allow", id=f"rhapsody-{args.stage}-{run_id}")
            use_wandb = wandb.run is not None
        except Exception as e:
            print(f"[Rhapsody] WARNING: wandb initialization failed or disabled: {e}")
            use_wandb = False

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Device / dtype ──────────────────────────────────────────────────────
    print(f"[Rhapsody] Device: {device}")

    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        print(f"[Rhapsody] GPU: {torch.cuda.get_device_name()}")
        print(f"[Rhapsody] dtype: {dtype}, managed by Accelerator")
        if major >= 8:
            torch.set_float32_matmul_precision('high')
            print("[Rhapsody] Set float32 matmul precision to 'high' (enables TF32)")
    elif device.type == "xla":
        dtype = torch.bfloat16
        print(f"[Rhapsody] TPU: Managed by Accelerator (BF16)")
    else:
        dtype = torch.float32

    # ── Tokenizer ───────────────────────────────────────────────────────────
    is_symbolic = args.task == "symbolic-music"
    tokenizer = get_tokenizer(symbolic=is_symbolic)
    vocab_size = len(tokenizer)

    # ── Model per stage ──────────────────────────────────────────────────────
    if args.stage == "pretrain":
        print("[Rhapsody] Stage 1: Text Pretraining")
        model = create_text_only_65m(vocab_size=vocab_size)
        model.config.gradient_checkpointing = args.grad_checkpoint
        model = model.to(device)

    elif args.stage == "align":
        print("[Rhapsody] Stage 2: Audio-Text Alignment (projector only)")
        model = create_rhapsody_65m(vocab_size=vocab_size)
        model.config.gradient_checkpointing = args.grad_checkpoint
        model.text_lm.config.gradient_checkpointing = args.grad_checkpoint

        # Optionally load pretrained text LM weights from Stage 1
        if args.pretrained_lm:
            load_pretrained_text_lm(model, args.pretrained_lm)

        # Freeze text LM; train projector only
        for param in model.text_lm.parameters():
            param.requires_grad = False
        print("[Rhapsody] Text LM frozen — training projector only.")

        model = model.to(device)

    else:  # finetune
        print("[Rhapsody] Stage 3: Instruction Fine-tuning (full model, encoder stays frozen)")
        model = create_rhapsody_65m(vocab_size=vocab_size)
        model.config.gradient_checkpointing = args.grad_checkpoint
        model.text_lm.config.gradient_checkpointing = args.grad_checkpoint
        model = model.to(device)

    def tie_weights(model: nn.Module) -> None:
        if hasattr(model, "config") and getattr(model.config, "tie_word_embeddings", False):
            if hasattr(model, "lm_head") and hasattr(model, "embed_tokens"):
                model.lm_head.weight = model.embed_tokens.weight
                print("[Rhapsody] Explicitly tied TextLM word embeddings on device.")
        if hasattr(model, "text_lm"):
            tie_weights(model.text_lm)

    tie_weights(model)

    # ── Optimizer ────────────────────────────────────────────────────────────
    # 1. Muon parameters: 2D+ weight matrices for Attention/FFN projections inside text LM.
    #    Excludes embeddings, lm_head, projector (AudioProjector), and norm scales (per-head scalars).
    muon_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad
        and p.ndim >= 2
        and "embed" not in n
        and "lm_head" not in n
        and "projector" not in n
        and "norm_scale" not in n   # q_norm_scale / k_norm_scale — not 2D matrices
    ]

    # 2. AdamW parameters with weight decay (projector weights only)
    adamw_decay_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and "projector" in n and "bias" not in n and p.ndim >= 2
    ]

    # 3. AdamW parameters without weight decay
    #    (biases, norms, embeddings/head, and per-head attention scale parameters)
    adamw_no_decay_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad
        and (p.ndim < 2 or "embed" in n or "lm_head" in n or "norm_scale" in n)
    ]

    # Verify param coverage (Fix #1)
    all_ids = {id(p) for n, p in model.named_parameters() if p.requires_grad}
    covered_ids = {id(p) for p in muon_params + adamw_decay_params + adamw_no_decay_params}
    assert covered_ids == all_ids, f"Param coverage mismatch! {len(all_ids ^ covered_ids)} params uncovered."

    if not muon_params:
        print("[Rhapsody] No Muon parameters detected (frozen text LM). Falling back to standard AdamW optimizer.")
        groups = []
        if adamw_decay_params:
            groups.append({"params": adamw_decay_params, "weight_decay": 0.1, "lr": args.lr, "betas": (0.9, 0.95)})
        groups.append({"params": adamw_no_decay_params, "weight_decay": 0.0, "lr": args.lr, "betas": (0.9, 0.95)})
        optimizer = torch.optim.AdamW(groups)
    else:
        optimizer = Muon(
            muon_params,
            lr=args.muon_lr, momentum=0.95, nesterov=True, ns_steps=5,
            adamw_params=None,  # We add AdamW groups manually below
        )

        # Add AdamW decay group only if parameters are present (Stage 2/3)
        if adamw_decay_params:
            optimizer.add_param_group({
                "params": adamw_decay_params,
                "lr": args.lr,
                "betas": (0.9, 0.95),
                "weight_decay": 0.1,
                "is_adamw": True,
            })

        # Add AdamW no-decay group
        optimizer.add_param_group({
            "params": adamw_no_decay_params,
            "lr": args.lr,
            "betas": (0.9, 0.95),
            "weight_decay": 0.0,
            "is_adamw": True,
        })

    # ── LR Scheduler ─────────────────────────────────────────────────────────
    # Use WSD for Stage 1 pretraining; Cosine Decay with Warmup for Stage 2 & 3
    total_opt_steps = args.max_steps
    if args.stage == "pretrain":
        print("[Rhapsody] Scheduler: Warmup-Stable-Decay (WSD)")
        lr_fn = lambda step, n=total_opt_steps: get_wsd_lr(step, n)
    else:
        print("[Rhapsody] Scheduler: Cosine Decay with Warmup")
        lr_fn = lambda step, n=total_opt_steps: get_cosine_lr(step, n, warmup_frac=0.03)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[lr_fn] * len(optimizer.param_groups),
    )

    # Only initialize the Hub manager on the main process to avoid concurrent HF API requests
    hub_manager = None
    if accelerator.is_main_process:
        _slog("Initialising hub_manager...")
        hub_manager = init_forge_hub(args.forge_config)
        _slog("hub_manager ready.")

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_step = 0
    _slog("Starting auto-resume check...")


    if args.auto_resume and args.resume is None:
        # Step 1: Main process pulls the latest checkpoint from Hub if not locally available
        if accelerator.is_main_process and hub_manager is not None:
            latest_local = find_latest_checkpoint(output_dir)
            if latest_local is None:
                print("[Rhapsody] No local checkpoints found. Pulling from Hub...")
                pulled = hub_manager.pull_latest(output_dir)
                if pulled is not None:
                    print(f"[Rhapsody] Pulled latest checkpoint from Hub: {pulled}")

        # Step 2: Synchronize all processes so non-main ranks wait for any Hub download.
        # CRITICAL: accelerator.wait_for_everyone() calls xm.rendezvous() which is a lazy
        # XLA collective op. Before accelerator.prepare() is called, the XLA distributed
        # runtime is not ready to execute collectives, causing a permanent deadlock on
        # multi-core TPU (v5e-8). We use a file-based sentinel instead, which requires
        # no XLA collective communication.
        if device.type == "xla" and accelerator.num_processes > 1:
            _sentinel = output_dir / ".rank0_resume_ready"
            if accelerator.is_main_process:
                _sentinel.touch()
                sys.stdout.write("[Rhapsody] Wrote resume sentinel. Waiting for ranks to sync...\n")
                sys.stdout.flush()
            else:
                sys.stdout.write(f"[Rank {accelerator.process_index}] Waiting for resume sentinel...\n")
                sys.stdout.flush()
                _t0 = time.time()
                while not _sentinel.exists():
                    if time.time() - _t0 > 300:
                        sys.stdout.write(f"[Rank {accelerator.process_index}] Timed out waiting for sentinel!\n")
                        sys.stdout.flush()
                        break
                    time.sleep(0.5)
                sys.stdout.write(f"[Rank {accelerator.process_index}] Found resume sentinel.\n")
                sys.stdout.flush()
            # Brief pause so all non-main ranks see the file before rank 0 removes it
            time.sleep(2)
            if accelerator.is_main_process:
                _sentinel.unlink(missing_ok=True)
        else:
            # GPU / CPU / single-process: standard barrier is safe here
            accelerator.wait_for_everyone()

        # Step 3: All processes independently scan for the checkpoint
        latest_local = find_latest_checkpoint(output_dir)
        if latest_local is not None:
            args.resume = str(latest_local)
            print(f"[Rhapsody] Found checkpoint to resume: {latest_local}")
    _slog(f"Resume check done. args.resume={args.resume}")

    if args.resume:
        ckpt_path = Path(args.resume)
        model_pt = ckpt_path / "model.pt"
        opt_pt = ckpt_path / "optimizer.pt"
        sched_pt = ckpt_path / "scheduler.pt"
        if model_pt.exists():
            print(f"[Rhapsody] Resuming from {ckpt_path}")
            ckpt = torch.load(model_pt, map_location="cpu", weights_only=True)
            # Support both wrapped and unwrapped checkpoints
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
                    random_state = ckpt["python_random_state"]
                    if isinstance(random_state, list):
                        version, state_vector, gauss_next = random_state
                        if isinstance(state_vector, list):
                            state_vector = tuple(state_vector)
                        random_state = (version, state_vector, gauss_next)
                    random.setstate(random_state)
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
        # Note: optimizer and scheduler states are loaded AFTER accelerator.prepare() to prevent state corruption.
        pass

    # ── Dataset per stage ────────────────────────────────────────────────────
    global_batch_size = args.batch_size * args.grad_accum * accelerator.num_processes
    _slog(f"Creating dataset: stage={args.stage}")
    if args.stage == "pretrain" and args.pretok_dir and Path(args.pretok_dir).exists():
        # ── Fast path: pre-tokenized shards from pretokenize.py ──────────────
        # Dataset only contains the remaining training data (post-resume-step).
        # already_fast_forwarded=True suppresses the Subset-skip logic below.
        print(f"[Rhapsody] Using pre-tokenized dataset from {args.pretok_dir}")
        dataset = PreTokenizedDataset(args.pretok_dir, seq_len=args.seq_len)
    elif args.stage == "pretrain":
        # ── Slow path: live streaming + on-the-fly tokenization ───────────────
        dataset = TextPretrainDataset(
            tokenizer,
            seq_len=args.seq_len,
            resume_step=start_step,
            global_batch_size=global_batch_size
        )
    elif is_symbolic:
        dataset = SymbolicMusicDataset(
            tokenizer,
            seq_len=args.seq_len,
            dataset_path=args.symbolic_dataset,
            hf_dataset=args.symbolic_hf,
            max_examples=args.symbolic_max_examples,
        )
    else:
        dataset = AudioTextDataset(tokenizer, seq_len=args.seq_len)
    _slog(f"Dataset created: {type(dataset).__name__}")

    # ── DataLoader ────────────────────────────────────────────────────────────
    is_iterable = isinstance(dataset, IterableDataset)
    is_pretok   = isinstance(dataset, PreTokenizedDataset)

    if is_pretok:
        # Pre-tokenized: map-style, fixed-length, no padding needed.
        # IMPORTANT: Use num_workers=0 on Kaggle TPU. Setting num_workers>0 causes
        # DataLoader to fork child processes. When 8 ranks each fork 2 workers,
        # you get 16 extra processes all trying to load .pt shards simultaneously,
        # exhausting RAM and crashing the browser tab.
        num_workers = 0
        collate_fn  = None
    elif is_iterable:
        # Streaming IterableDatasets: must use 0 workers to avoid duplicate samples.
        num_workers = 0
        collate_fn  = None
    else:
        # Other map-style datasets (AudioText, SymbolicMusic): need padding collator.
        num_workers = args.num_workers if args.num_workers is not None else min(4, os.cpu_count() or 1)
        collate_fn  = DataCollatorWithPadding(tokenizer)

    # ── Fast-forwarding ───────────────────────────────────────────────────────
    batches_to_skip = start_step * args.grad_accum
    if getattr(dataset, 'already_fast_forwarded', False):
        # PreTokenizedDataset: dataset was generated starting exactly at the resume
        # point — no subset skipping needed.
        batches_to_skip = 0
        print(f"[Rhapsody] Pre-tokenized dataset starts at resume point (step {start_step}). "
              f"Skipping Subset fast-forward.")
    elif not is_iterable and batches_to_skip > 0:
        examples_to_skip = batches_to_skip * args.batch_size * accelerator.num_processes
        print(f"[Rhapsody] Fast-forwarding map-style dataset by {examples_to_skip} examples...")
        if examples_to_skip < len(dataset):
            dataset = torch.utils.data.Subset(dataset, range(examples_to_skip, len(dataset)))
        else:
            dataset = torch.utils.data.Subset(dataset, [])
        batches_to_skip = 0  # Handled via Subset
    elif is_iterable:
        # Iterable datasets fast-forward O(1) inside their __init__ (TextPretrainDataset).
        batches_to_skip = 0

    # For PreTokenizedDataset (map-style), we manually add a DistributedSampler
    # so each of the 8 ranks gets a non-overlapping slice of the data.
    # We do NOT pass the dataloader into accelerator.prepare() because on XLA/TPU
    # that triggers MpDeviceLoader which forks child processes — with 8 ranks and
    # num_workers>0 this creates a process explosion that crashes the notebook.
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
        print(f"[Rhapsody] Rank {accelerator.process_index}: DistributedSampler created "
              f"({len(sampler):,} examples / {accelerator.num_processes} ranks)")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        persistent_workers=False,
        drop_last=True,   # critical: ensures static batch shapes for XLA compilation
        sampler=sampler,  # None for iterable/single-process, DistributedSampler for pretok multi-rank
        shuffle=(sampler is None and not is_iterable and not is_pretok),  # only if no sampler
    )
    _slog("DataLoader created. Starting accelerator.prepare()...")

    # ── Prepare Accelerator ──────────────────────────────────────────────────
    # IMPORTANT: We NEVER pass the dataloader into accelerator.prepare() on XLA/TPU.
    # Doing so wraps it in MpDeviceLoader which spawns prefetch worker processes —
    # on a node with 8 TPU processes, this creates 16+ extra processes and causes
    # an OOM/crash. Instead we use a manual DistributedSampler (above) for sharding.
    model, optimizer = accelerator.prepare(model, optimizer)
    _slog("accelerator.prepare() complete.")

    # ── Load Optimizer & Scheduler state dicts after prepare ─────────────────
    if args.resume:
        opt_pt = Path(args.resume) / "optimizer.pt"
        sched_pt = Path(args.resume) / "scheduler.pt"
        if opt_pt.exists():
            print(f"[Rhapsody] Loading optimizer state from {opt_pt} (after accelerator.prepare)...")
            opt_ckpt = torch.load(opt_pt, map_location="cpu", weights_only=True)
            saved_state_dict = opt_ckpt["optimizer"]
            try:
                # Collect all parameters from saved groups
                saved_param_ids = []
                for group in saved_state_dict["param_groups"]:
                    saved_param_ids.extend(group["params"])
                
                # Collect all parameters from active optimizer
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
                    
                    # Restore Muon step counter if present
                    if "_adam_step" in saved_state_dict:
                        optimizer._adam_step = saved_state_dict["_adam_step"]
                    
                    print("[Rhapsody] Optimizer state aligned and loaded successfully.")
                else:
                    print(f"[Rhapsody] WARNING: Optimizer parameter count mismatch: saved has {len(saved_param_ids)}, active has {len(active_params)}. Fallback to standard load.")
                    optimizer.load_state_dict(saved_state_dict)
            except Exception as e:
                print(f"[Rhapsody] WARNING: Failed to dynamically align optimizer state dict: {e}. Falling back to standard loading.")
                optimizer.load_state_dict(saved_state_dict)
        
        if sched_pt.exists():
            print(f"[Rhapsody] Loading scheduler state from {sched_pt}...")
            sched_ckpt = torch.load(sched_pt, map_location="cpu", weights_only=True)
            scheduler.load_state_dict(sched_ckpt["scheduler"])
            # Force alignment with correct start_step to repair any previously corrupted or scaled last_epoch
            scheduler.last_epoch = start_step
            if hasattr(scheduler, "_step_count"):
                scheduler._step_count = start_step + 1
            print(f"[Rhapsody] Scheduler last_epoch aligned to {start_step}.")
        else:
            # Fallback if scheduler state is missing: set last_epoch manually
            scheduler.last_epoch = start_step
            if hasattr(scheduler, "_step_count"):
                scheduler._step_count = start_step + 1
            print(f"[Rhapsody] Scheduler last_epoch initialized to fallback: {start_step}.")

    # Note: torch.compile is intentionally called after accelerator.prepare.
    # The optimizer holds references to the original model parameters, and DDP wrappers
    # are compiled correctly.
    # On XLA/TPU: uses the openxla backend (torch_xla >= 2.1). Expect a 2-3 min
    # warmup while XLA traces and compiles the graph, then throughput improves ~15-20%.
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

    # ── All-ranks-ready barrier ───────────────────────────────────────────────
    # CRITICAL: Without this barrier, rank 0 enters the training loop and burns
    # CPU on Python tokenisation + XLA dispatch, starving ranks 1-7 from finishing
    # their dataset creation. All ranks signal readiness; none proceed until every
    # rank has written its ready file. File-based: avoids XLA collective deadlocks.
    if device.type == "xla" and accelerator.num_processes > 1:
        _rdy = output_dir / f".rank_{accelerator.process_index}_ready"
        _rdy.touch()
        _slog(f"Ready-file written. Waiting for all {accelerator.num_processes} ranks...")
        print(f"[Rhapsody] Rank {accelerator.process_index}: at all-ranks barrier "
              f"(waiting for all {accelerator.num_processes} ranks)...")
        _bd = time.time() + 600  # 10-minute timeout
        while time.time() < _bd:
            n_rdy = sum(
                1 for i in range(accelerator.num_processes)
                if (output_dir / f".rank_{i}_ready").exists()
            )
            if n_rdy >= accelerator.num_processes:
                break
            time.sleep(0.5)
        else:
            n_rdy = sum(
                1 for i in range(accelerator.num_processes)
                if (output_dir / f".rank_{i}_ready").exists()
            )
            _slog(f"WARNING: barrier timed out! Only {n_rdy}/{accelerator.num_processes} ranks ready.")
            print(f"[Rhapsody] WARNING: barrier timed out! Only {n_rdy}/{accelerator.num_processes} ranks ready.")
        # Brief pause so all ranks see all files, then clean up this rank's file
        time.sleep(2)
        _rdy.unlink(missing_ok=True)
        _slog("All-ranks barrier passed. Entering training loop.")
        if accelerator.is_main_process:
            print("[Rhapsody] All-ranks-ready barrier passed. Starting training loop.")

    # ── Training ─────────────────────────────────────────────────────────────
    total_steps = args.max_steps
    eff_batch = args.batch_size * args.grad_accum * accelerator.num_processes
    print(f"[Rhapsody] Training: {total_steps} optimizer steps, "
          f"batch={args.batch_size}, accum={args.grad_accum}, eff_batch={eff_batch}")

    model.train()

    step = start_step  # tracks optimizer steps
    running_loss = torch.tensor(0.0, device=device)
    running_grad_norm = torch.tensor(0.0, device=device)
    tokens_in_window = 0
    micro_steps_in_window = 0
    window_start = time.time()

    # ── Per-rank diagnostic log file ─────────────────────────────────────────
    # All 8 ranks write here because accelerator.print() only shows rank 0.
    # If a rank hangs at sync(), its log will show the last batch-fetch line
    # but never the "sync DONE" line, pinpointing the deadlock exactly.
    _rlog_path = output_dir / f"rank_{accelerator.process_index}_train.log"
    _rlog_fh = open(_rlog_path, "w", buffering=1)
    def _rlog(msg: str):
        ts = time.strftime("%H:%M:%S")
        _rlog_fh.write(f"[{ts}][R{accelerator.process_index}] {msg}\n")
        _rlog_fh.flush()

    _rlog(f"=== Training start: step={step}, total_steps={total_steps}, device={device} ===")
    _rlog(f"Dataset rank={accelerator.process_index}/{accelerator.num_processes}, "
          f"batch_size={args.batch_size}, grad_accum={args.grad_accum}")
    print(f"[Rhapsody] Process {accelerator.process_index} starting training loop...")
    print(f"[Rhapsody] Per-rank diagnostic logs: {output_dir}/rank_N_train.log")
    while step < total_steps:
        for batch in dataloader:
            if step >= total_steps:
                break

            _t_batch = time.time()  # batch fetch completed (we're now inside the loop)
            batch_shape = tuple(batch["input_ids"].shape)
            _rlog(f"step={step} | batch fetched {batch_shape} | fetch-to-loop: latency tracked externally")

            _t0 = time.time()
            with accelerator.accumulate(model):
                loss = compute_loss(model, batch, device)
                _rlog(f"step={step} | forward DONE ({time.time()-_t0:.3f}s)")

                _t1 = time.time()
                accelerator.backward(loss)
                _rlog(f"step={step} | backward DONE ({time.time()-_t1:.3f}s) -- all_reduce queued")

                # Keep loss as a local TPU tensor (do not call .item() here)
                loss_val_tensor = loss.detach()

                grad_norm_tensor = None
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    if grad_norm is not None:
                        grad_norm_tensor = grad_norm.detach()

                _t2 = time.time()
                optimizer.step()
                _rlog(f"step={step} | optimizer.step DONE ({time.time()-_t2:.3f}s)")

                if accelerator.sync_gradients:
                    if not accelerator.optimizer_step_was_skipped:
                        scheduler.step()
                        step += 1
                    else:
                        print("  [Rhapsody] Warning: Gradient overflow detected, skipping optimizer step.")

                optimizer.zero_grad(set_to_none=True)

                # Execute the XLA lazy graph.
                # torch_xla.sync() (formerly xm.mark_step()) dispatches ALL queued ops
                # including the DDP all_reduce. ALL 8 ranks must reach this call for the
                # all_reduce to complete. If this line never returns, one or more other
                # ranks are stuck in data loading and never queued their all_reduce op.
                if device.type == "xla":
                    _rlog(f"step={step} | calling torch_xla.sync() (blocks until all 8 ranks execute all_reduce)...")
                    _t3 = time.time()
                    try:
                        import torch_xla
                        torch_xla.sync()   # preferred API in torch_xla >= 2.x
                    except AttributeError:
                        import torch_xla.core.xla_model as xm
                        xm.mark_step()     # fallback for older torch_xla
                    _rlog(f"step={step} | torch_xla.sync() DONE ({time.time()-_t3:.3f}s) ✓")

                # NOW accumulate values on device without calling .item() at every step
                _rlog(f"step={step} | step COMPLETE")
                running_loss += loss_val_tensor
                micro_steps_in_window += 1

                # Count actual text + audio tokens processed in the batch
                batch_text_tokens = batch["input_ids"].numel()
                batch_audio_tokens = (64 * batch["input_ids"].shape[0]) if "audio_features" in batch else 0
                tokens_in_window += (batch_text_tokens + batch_audio_tokens)

                if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                    if grad_norm_tensor is not None:
                        running_grad_norm += grad_norm_tensor

                # ── Logging ──────────────────────────────────────────────────────
                if step > 0 and step % args.log_steps == 0 and accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
                    elapsed = time.time() - window_start
                    avg_loss = (running_loss / max(1, micro_steps_in_window)).item()
                    avg_grad_norm = (running_grad_norm / args.log_steps).item()
                    
                    # Gather and sum tokens processed across all processes for logging accurate speed
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
                    # Must call save_checkpoint on all processes to avoid TPU checkpoint saving deadlocks
                    save_checkpoint(ckpt_dir, model, optimizer, scheduler, step, accelerator)
                    
                    if accelerator.is_main_process:
                        print(f"  Checkpoint saved: {ckpt_dir}")

                        # Keep latest 3 checkpoints locally to save disk space
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
                            # Prune after uploading to avoid race condition where we delete what we upload
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
                                daemon=False  # Must be False so upload isn't killed if script exits
                            ).start()
                            
                    # XLA-safe barrier: avoid wait_for_everyone() which deadlocks before
                    # prepare() is done or after it due to XLA collective timing issues.
                    # File-based sentinel is safe for same-node filesystem.
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

    # ── Final save ─────────────────────────────────────────────────────────
    print("[Rhapsody] Training complete!")
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(model)
    raw_model = getattr(unwrapped_model, "_orig_mod", unwrapped_model)
    # Must save on all processes to participate in XLA save synchronization
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
