# Rhapsody

A small, lightweight decoder-only Language Model pre-training codebase (~65M/84M parameters). Optimized for resource-friendly pre-training on GPU (single/multi-device) and TPU (v5e-8 PJRT/XLA environments) with automated checkpoint syncing across sequential Colab and Kaggle sessions.

---

## Architecture

Rhapsody implements a deep-and-thin Transformer Language Model based on modern design principles:

* **Transformer Structure**: Decoder-only architecture following Llama-like SwiGLU feed-forward networks, RMSNorm, and Rotary Position Embeddings (RoPE).
* **Attention**: Grouped-Query Attention (GQA) with 8 query heads and 4 KV heads for memory-efficient training and high-throughput generation.
* **Tied Embeddings**: Word embeddings are tied with the language model output head.
* **Small Footprint**: Sized at ~65M-84M parameters with a 32K-token vocabulary using the `HuggingFaceTB/cosmo2-tokenizer`.

---

## Repository Structure

* [rhapsody/model.py](file:///home/kush26/Projects/rhapsody/rhapsody/model.py): Core `TextLM` transformer architecture definition and configs.
* [rhapsody/data.py](file:///home/kush26/Projects/rhapsody/rhapsody/data.py): Tokenizer loading, `TextPretrainDataset`, and `PreTokenizedDataset` for shard loading.
* [rhapsody/inference.py](file:///home/kush26/Projects/rhapsody/rhapsody/inference.py): Text generation inference utility with KV-caching.
* [rhapsody/train.py](file:///home/kush26/Projects/rhapsody/rhapsody/train.py): GPU/CPU training script supporting mixed precision (BF16/FP16) via HF Accelerate.
* [rhapsody/train_tpu.py](file:///home/kush26/Projects/rhapsody/rhapsody/train_tpu.py): Performance-tuned pre-training script optimized for TPU v5e-8 (using PJRT/XLA and mock import blockers).
* [test_training.py](file:///home/kush26/Projects/rhapsody/test_training.py) & [test_kv_cache.py](file:///home/kush26/Projects/rhapsody/test_kv_cache.py): Synthetic verification tests for training loops and KV-caching correctness.

---

## Optimizers & Scheduling

* **Muon Optimizer**: Momentum + Newton-Schulz orthogonalisation applied to all 2D+ weight matrices (attention & FFN projections).
* **AdamW**: Applied to scalars, embeddings, layer norms, and biases.
* **WSD Schedule**: Warmup-Stable-Decay learning rate schedule to maximize pre-training throughput and stability.

---

## Setup & Notebook Deployment

### 1. Cloned Deployment in Colab or Kaggle
Run this in a notebook cell to clone and install:

```python
# 1. Clone the repository
!git clone https://github.com/Kush-Singh-26/rhapsody.git
%cd rhapsody

# 2. Install dependencies in editable mode
!pip install -e .
```

---

## Training

### 1. GPU/CPU Pre-training
Run pre-training using Accelerate DDP:

```bash
# Single GPU or CPU training
rhapsody-train --max-steps 100000 --batch-size 4 --grad-accum 16 --lr 0.0008

# Multi-GPU via torchrun (e.g. Kaggle 2x T4)
torchrun --nproc_per_node=2 rhapsody/train.py --max-steps 100000 --batch-size 4 --grad-accum 8 --lr 0.0008
```

### 2. TPU Pre-training
For Kaggle TPU v5e-8 PJRT multi-processing environments, run using:

```bash
python rhapsody/train_tpu.py --max-steps 100000 --batch-size 8 --grad-accum 16 --pretok-dir ./pretokenized_shards
```

*Note: [train_tpu.py](file:///home/kush26/Projects/rhapsody/rhapsody/train_tpu.py) automatically patches subprocess spawning, blocks conflicting TensorFlow/JAX metrics allocators, sets thread affinity, and integrates XLA-friendly file-based checkpoint barriers to eliminate TPU hangs.*

---

## Nomad Checkpointing (`config.yaml`)

Both training scripts support automated, multi-session pre-training resumption via `lm_forge` HubSync:
1. Local checkpoints are periodically saved and pruned.
2. The latest state is uploaded to your Hugging Face Hub repository in a background thread.
3. Upon preemption/session restart, passing `--auto-resume --forge-config ./config.yaml` pulls the latest checkpoint and fast-forwards the dataset to the exact microstep where training left off.

```bash
rhapsody-train --auto-resume --forge-config ./config.yaml --save-steps 500
```

---

## Verification & Testing

Before kicking off training runs, verify model components on CPU:

```bash
# Verify the training update step and Muon orthogonal updates
python test_training.py

# Verify KV-Cache logic and output equivalence
python test_kv_cache.py
```

---

## Inference

Run local autoregressive generation on a pre-trained checkpoint:

```bash
python -m rhapsody.inference \
  --checkpoint ./outputs/final/model.pt \
  --prompt "Deep learning is" \
  --max-new-tokens 128
```
