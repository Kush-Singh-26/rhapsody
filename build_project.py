#!/usr/bin/env python3
"""
Build script for AudioForge project.

WARNING: Running this script will OVERWRITE existing source files.
It is intended only for initial project scaffolding.
Pass --force to confirm you want to regenerate files.
"""

import argparse
import os
import sys

AUDIOFORGE_DIR = os.path.dirname(os.path.abspath(__file__))


def write_file(rel_path: str, content: str, force: bool = False) -> None:
    path = os.path.join(AUDIOFORGE_DIR, rel_path)
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    if os.path.exists(path) and not force:
        print(f"  Skipped (already exists): {rel_path}  — use --force to overwrite")
        return
    with open(path, "w") as f:
        f.write(content)
    print(f"  {'Overwritten' if force else 'Created'}: {rel_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing files (destructive — back up any edits first)."
    )
    args = parser.parse_args()

    if not args.force:
        print("AudioForge build script — dry run (no files will be overwritten).")
        print("Pass --force to actually write files.")
        print()

    print("Building AudioForge project...")
    _write_all(force=args.force)
    print("\nDone!")
    print(f"Files {'written' if args.force else 'checked'} in: {AUDIOFORGE_DIR}")


def _write_all(force: bool = False) -> None:
    # config.yaml
    write_file("config.yaml", '''\
name: "rhapsody-65m"
state:
  # Replace YOUR_USERNAME with your HuggingFace username before running
  repo_id: "Kush26/rhapsody-65m-checkpoints"
  branch: "checkpoints"
  checkpoint_limit: 3
  push_every: 500
  private: true

providers:
  modal:
    gpu: "A100"
    gpu_count: 1
    cpu: 16
    memory: 65536   # MB
    timeout: 86400  # seconds

# NOTE: These profiles are reference values for manual CLI args.
# train.py uses argparse; config.yaml is not loaded automatically.
#   --lr controls the AdamW LR (embeddings/norms).
#   Muon LR is hardcoded at 0.015 in train.py.
profiles:
  colab:
    per_device_train_batch_size: 2
    gradient_accumulation_steps: 16
    learning_rate: 0.0008
    probe_memory: true
    data_cache: false
  modal:
    per_device_train_batch_size: 16
    gradient_accumulation_steps: 32
    learning_rate: 0.015
    data_cache: true
  local:
    per_device_train_batch_size: 8
    gradient_accumulation_steps: 8
    learning_rate: 0.001
''', force=force)

    # requirements.txt
    write_file("requirements.txt", '''\
torch>=2.0.0
transformers>=4.40.0
datasets>=2.18.0
accelerate>=0.20.0
huggingface-hub>=0.20.0
safetensors>=0.4.0
tokenizers>=0.15.0
pyyaml>=6.0
python-dotenv>=1.0.0
numpy>=1.24.0,<2.0.0
torchaudio>=2.0.0
sentencepiece>=0.1.99
protobuf>=3.20.0

# Optional: comment out if not using W&B / TensorBoard logging
wandb>=0.15.0
tensorboard>=2.14.0
''', force=force)


if __name__ == "__main__":
    main()
