#!/usr/bin/env python3
"""
pretokenize.py — Rhapsody Pre-tokenization Script
==================================================
Run this in a Kaggle CPU notebook BEFORE TPU training.
Generates ~3.15B tokens (enough to cover steps 58,500 → 100,000 with buffer)
and saves them as int16 .pt shard files ready for PreTokenizedDataset.

Usage (in Kaggle CPU notebook):
    !pip install -q hf_transfer datasets transformers
    !python /kaggle/working/rhapsody/pretokenize.py

After it finishes (~30-45 min), create a Kaggle Dataset:
    See the printed instructions at the end of this script.
"""

import os
import sys
import json
import time
from pathlib import Path

# ── Fast HuggingFace downloads (Rust-based, 3-5x faster) ─────────────────────
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
try:
    import hf_transfer  # noqa: F401
except ImportError:
    os.system("pip install -q hf_transfer")

import torch
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG — edit only if you changed these in config.yaml or train_tpu.py
# ═════════════════════════════════════════════════════════════════════════════
SEQ_LEN        = 1024           # must match --seq-len in training command
SHARD_SIZE     = 50_000_000     # tokens per shard → ~100 MB at int16

# Tokens needed: (100_000 - 58_500) steps × 64 global_batch × 1024 seq_len
# = 41_500 × 65_536 = 2,719,744,000 tokens.  +15% buffer → 3,127,705,600
# Round up to a clean shard boundary:
TARGET_TOKENS  = 3_150_000_000  # 63 shards = 6.3 GB

OUT_DIR        = Path("/kaggle/working/rhapsody_tokens")

# Dataset mix — must match data.py ratios
FINEWEB_RATIO  = 0.70
DCLM_RATIO     = 0.25
COSMO_RATIO    = 0.05

BATCH_DOCS     = 2000           # documents per tokenization batch (Rust tokenizer batches efficiently)

# ═════════════════════════════════════════════════════════════════════════════
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Tokenizer ─────────────────────────────────────────────────────────────────
print("─" * 60)
print("Loading cosmo2-tokenizer...")
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
tok.add_special_tokens({"additional_special_tokens": ["<|pad|>", "<|audio|>", "<|text|>"]})
EOS = tok.eos_token_id
VOCAB = len(tok)
print(f"  Vocab size : {VOCAB}")
print(f"  EOS token  : {EOS}")

# ── Streaming datasets ────────────────────────────────────────────────────────
print("─" * 60)
print("Opening streaming datasets (no full download)...")
fw    = load_dataset("HuggingFaceFW/fineweb-edu",         name="sample-10BT",    split="train", streaming=True)
dclm  = load_dataset("mlfoundations/dclm-baseline-1.0",                          split="train", streaming=True)
cosmo = load_dataset("HuggingFaceTB/cosmopedia-v2",       name="cosmopedia-v2",   split="train", streaming=True)
mixed = interleave_datasets(
    [fw, dclm, cosmo],
    probabilities=[FINEWEB_RATIO, DCLM_RATIO, COSMO_RATIO],
    stopping_strategy="all_exhausted",
)
print("Datasets ready.")

# ── Tokenize + pack into shards ───────────────────────────────────────────────
print("─" * 60)
print(f"Target  : {TARGET_TOKENS / 1e9:.2f}B tokens")
print(f"Shards  : {TARGET_TOKENS // SHARD_SIZE} × {SHARD_SIZE // 1_000_000}M tokens = "
      f"{TARGET_TOKENS // SHARD_SIZE * 200} MB")
print(f"Out dir : {OUT_DIR}")
print("─" * 60)

buffer:      list[int] = []
batch_texts: list[str] = []
shard_idx    = 0
total_toks   = 0
t0           = time.time()
t_last_log   = t0

EXAMPLES_PER_FULL_SHARD = SHARD_SIZE // (SEQ_LEN + 1)  # packed examples


def save_shard(data: list[int], idx: int) -> None:
    """Save a shard as a flat int32 tensor."""
    t = torch.tensor(data, dtype=torch.int32)
    path = OUT_DIR / f"shard_{idx:04d}.pt"
    torch.save(t, path)
    size_mb = t.numel() * 4 / 1024 / 1024
    elapsed = time.time() - t0
    rate_m  = (total_toks + len(data)) / elapsed / 1e6
    print(f"  Shard {idx:04d} saved | {size_mb:.0f} MB | "
          f"total={( total_toks + len(data)) / 1e9:.3f}B tok | "
          f"{rate_m:.2f}M tok/s | {elapsed:.0f}s elapsed")


def flush_buffer() -> None:
    global buffer, shard_idx, total_toks
    while len(buffer) >= SHARD_SIZE:
        save_shard(buffer[:SHARD_SIZE], shard_idx)
        total_toks += SHARD_SIZE
        buffer      = buffer[SHARD_SIZE:]
        shard_idx  += 1


# Main tokenization loop
for example in mixed:
    text = example.get("text", "")
    if not text or len(text) < 100:
        continue

    batch_texts.append(text)

    if len(batch_texts) >= BATCH_DOCS:
        # Batch tokenize: the Rust-based fast tokenizer handles this in parallel internally
        enc = tok(
            batch_texts,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_tensors=None,
        )
        for ids in enc["input_ids"]:
            buffer.extend(ids)
            buffer.append(EOS)   # document boundary signal
        batch_texts = []
        flush_buffer()

    if total_toks >= TARGET_TOKENS:
        print(f"\n✅ Reached target of {TARGET_TOKENS / 1e9:.2f}B tokens.")
        break

# Process any remaining docs in the batch
if batch_texts:
    enc = tok(
        batch_texts,
        truncation=False,
        padding=False,
        return_attention_mask=False,
        return_tensors=None,
    )
    for ids in enc["input_ids"]:
        buffer.extend(ids)
        buffer.append(EOS)
    flush_buffer()

# Save the partial final shard (will be smaller than SHARD_SIZE)
if len(buffer) >= (SEQ_LEN + 1):
    save_shard(buffer, shard_idx)
    last_shard_examples = len(buffer) // (SEQ_LEN + 1)
    total_toks += len(buffer)
    shard_idx  += 1
else:
    last_shard_examples = 0
    print(f"  Discarding {len(buffer)} leftover tokens (< 1 full example).")

# ── Write metadata ────────────────────────────────────────────────────────────
total_examples = (shard_idx - 1) * EXAMPLES_PER_FULL_SHARD + last_shard_examples

meta = {
    "num_shards":            shard_idx,
    "shard_size_tokens":     SHARD_SIZE,
    "examples_per_full_shard": EXAMPLES_PER_FULL_SHARD,
    "last_shard_examples":   last_shard_examples,
    "total_examples":        total_examples,
    "seq_len":               SEQ_LEN,
    "vocab_size":            VOCAB,
    "eos_id":                EOS,
    "total_tokens_approx":   total_toks,
    "mix": {
        "fineweb_edu_sample10bt": FINEWEB_RATIO,
        "dclm_baseline_1.0":     DCLM_RATIO,
        "cosmopedia_v2":         COSMO_RATIO,
    },
}
(OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

elapsed_min = (time.time() - t0) / 60
print("\n" + "═" * 60)
print(f"  Shards created  : {shard_idx}")
print(f"  Total examples  : {total_examples:,}")
print(f"  Total tokens    : {total_toks / 1e9:.3f}B")
print(f"  Disk size       : ~{shard_idx * 100} MB")
print(f"  Time taken      : {elapsed_min:.1f} minutes")
print("═" * 60)

# ── Kaggle Dataset upload instructions ───────────────────────────────────────
print("""
NEXT STEPS — upload to a Kaggle Dataset:
─────────────────────────────────────────────────────────────────────
1. Write dataset metadata (run this cell):

import json
from pathlib import Path
meta_dir = Path("/kaggle/working/rhapsody_tokens")
(meta_dir / "dataset-metadata.json").write_text(json.dumps({
    "title": "rhapsody-pretokenized",
    "id": "kush26/rhapsody-pretokenized",
    "licenses": [{"name": "CC0-1.0"}]
}, indent=2))
print("metadata written.")

2. Create the dataset (run this cell):

!kaggle datasets create -p /kaggle/working/rhapsody_tokens --dir-mode tar

3. In your TPU training notebook, add the dataset:
   Settings → Add Data → Your Datasets → rhapsody-pretokenized

4. It mounts at: /kaggle/input/rhapsody-pretokenized/

5. Launch training with the new flag:
   --pretok-dir /kaggle/input/rhapsody-pretokenized
─────────────────────────────────────────────────────────────────────
""")
