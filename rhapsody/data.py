"""Rhapsody Text-only Data Pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, IterableDataset
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer


def get_tokenizer(symbolic: bool = False) -> AutoTokenizer:
    """
    Load the cosmo2-tokenizer (falls back to gpt2 if unavailable).
    Adds Rhapsody special tokens: <|pad|>, <|audio|>, <|text|>.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
        print(f"[Rhapsody] Loaded cosmo2-tokenizer (vocab={tokenizer.vocab_size})")
    except Exception:
        print("[Rhapsody] cosmo2-tokenizer not found, falling back to gpt2 tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

    special = ["<|pad|>", "<|audio|>", "<|text|>"]
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": special})
    if num_added > 0:
        print(f"[Rhapsody] Added {num_added} special tokens")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


class TextPretrainDataset(IterableDataset):
    """
    Streaming interleaved dataset for Stage-1 text pretraining.
    Mixes FineWeb-Edu, DCLM-Baseline, and Cosmopedia v2.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        seq_len: int = 2048,
        fineweb_ratio: float = 0.70,
        dclm_ratio: float = 0.25,
        cosmopedia_ratio: float = 0.05,
        resume_step: int = 0,
        global_batch_size: int = 64,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        total_tokens_to_skip = resume_step * global_batch_size * seq_len

        try:
            from accelerate import PartialState
            from datasets.distributed import split_dataset_by_node as _split_by_node
            _dist_state = PartialState()
            _rank = _dist_state.process_index
            _world = _dist_state.num_processes
        except Exception:
            _rank = 0
            _world = 1
        _do_shard = _world > 1

        def _fast_forward_dataset(ds, tokens_to_skip, density_factor, total_docs, is_dclm=False):
            if tokens_to_skip <= 0:
                return ds
            docs_to_skip = tokens_to_skip // density_factor
            ex_it = ds._ex_iterable
            if not hasattr(ex_it, "kwargs"):
                return ds

            if is_dclm:
                original_files = ex_it.kwargs.get("original_files", [])
                if original_files:
                    files_len = len(original_files)
                    docs_per_file = total_docs / files_len
                    files_to_skip = int(docs_to_skip // docs_per_file)
                    remaining_docs = int(docs_to_skip % docs_per_file)
                    
                    files_to_skip = min(files_to_skip, files_len - 1)
                    if files_to_skip > 0:
                        ex_it.kwargs["original_files"] = ex_it.kwargs["original_files"][files_to_skip:]
                        ex_it.kwargs["base_files"] = ex_it.kwargs["base_files"][files_to_skip:]
                        ex_it.kwargs["files_iterables"] = ex_it.kwargs["files_iterables"][files_to_skip:]
                        print(f"[Rhapsody] [DCLM] Shard-level fast-forward: skipped {files_to_skip} shards.")
                    if remaining_docs > 0:
                        print(f"[Rhapsody] [DCLM] Doc-level fast-forward: skipping remaining {remaining_docs} documents...")
                        ds = ds.skip(remaining_docs)
            else:
                files = ex_it.kwargs.get("files", [])
                if files:
                    files_len = len(files)
                    docs_per_file = total_docs / files_len
                    files_to_skip = int(docs_to_skip // docs_per_file)
                    remaining_docs = int(docs_to_skip % docs_per_file)
                    
                    files_to_skip = min(files_to_skip, files_len - 1)
                    if files_to_skip > 0:
                        ex_it.kwargs["files"] = ex_it.kwargs["files"][files_to_skip:]
                        if "row_groups_list" in ex_it.kwargs:
                            ex_it.kwargs["row_groups_list"] = ex_it.kwargs["row_groups_list"][files_to_skip:]
                        print(f"[Rhapsody] [Parquet] Shard-level fast-forward: skipped {files_to_skip} shards.")
                    if remaining_docs > 0:
                        print(f"[Rhapsody] [Parquet] Doc-level fast-forward: skipping remaining {remaining_docs} documents...")
                        ds = ds.skip(remaining_docs)
            return ds

        datasets_list = []
        weights = []

        print(f"[Rhapsody] Fast-forwarding streaming datasets for step {resume_step} ({total_tokens_to_skip:,} tokens)...")

        # 1. FineWeb-Edu
        fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        fw = _fast_forward_dataset(fw, total_tokens_to_skip * fineweb_ratio, 1034, 9672101, is_dclm=False)
        if _do_shard:
            fw = _split_by_node(fw, rank=_rank, world_size=_world)
        datasets_list.append(fw)
        weights.append(fineweb_ratio)

        # 2. DCLM
        dclm = load_dataset("mlfoundations/dclm-baseline-1.0", split="train", streaming=True)
        dclm = _fast_forward_dataset(dclm, total_tokens_to_skip * dclm_ratio, 1333, 3000000000, is_dclm=True)
        if _do_shard:
            dclm = _split_by_node(dclm, rank=_rank, world_size=_world)
        datasets_list.append(dclm)
        weights.append(dclm_ratio)

        # 3. Cosmopedia
        cosmo = load_dataset("HuggingFaceTB/cosmopedia-v2", name="cosmopedia-v2", split="train", streaming=True)
        cosmo = _fast_forward_dataset(cosmo, total_tokens_to_skip * cosmopedia_ratio, 803, 39134000, is_dclm=False)
        if _do_shard:
            cosmo = _split_by_node(cosmo, rank=_rank, world_size=_world)
        datasets_list.append(cosmo)
        weights.append(cosmopedia_ratio)

        self.dataset = interleave_datasets(
            datasets_list, probabilities=weights, stopping_strategy="all_exhausted"
        )

        if _do_shard:
            print(f"[Rhapsody] Sharded each sub-dataset independently before interleaving: rank {_rank}/{_world}")

    def __iter__(self):
        buffer: list[int] = []
        eos_id = self.tokenizer.eos_token_id

        while True:
            for example in self.dataset:
                text = example.get("text", "")
                if not text or len(text) < 100:
                    continue

                tokenized = self.tokenizer(
                    text,
                    truncation=False,
                    padding=False,
                    return_tensors=None,
                )
                tokens = tokenized["input_ids"]

                if eos_id is not None:
                    tokens = tokens + [eos_id]

                buffer.extend(tokens)

                while len(buffer) >= self.seq_len + 1:
                    chunk = buffer[: self.seq_len + 1]
                    buffer = buffer[self.seq_len:]

                    input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                    labels    = torch.tensor(chunk[1:],  dtype=torch.long)

                    yield {"input_ids": input_ids, "labels": labels}


class PreTokenizedDataset(Dataset):
    """
    Map-style dataset backed by pre-tokenized int16 shard files.
    """

    already_fast_forwarded: bool = True

    def __init__(self, shard_dir: str, seq_len: int = 1024):
        self.shard_dir = Path(shard_dir)
        self.seq_len   = seq_len
        self.chunk_len = seq_len + 1

        meta_path = self.shard_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"[Rhapsody] PreTokenizedDataset: meta.json not found in {shard_dir}. "
                "Run pretokenize.py first."
            )
        meta = json.loads(meta_path.read_text())
        self.total_tokens = meta["total_tokens"]
        self.shard_size   = meta["shard_size"]
        self.num_samples  = (self.total_tokens - 1) // self.seq_len

        self.shard_paths = sorted(self.shard_dir.glob("shard_*.pt"))
        if not self.shard_paths:
            raise FileNotFoundError(f"[Rhapsody] No shard_*.pt files found in {shard_dir}")

        self.cache = {}

    def __len__(self) -> int:
        return self.num_samples

    def _load_shard(self, shard_idx: int) -> torch.Tensor:
        if shard_idx in self.cache:
            return self.cache[shard_idx]

        if len(self.cache) >= 2:
            self.cache.pop(next(iter(self.cache)))

        shard_path = self.shard_paths[shard_idx]
        data = torch.load(shard_path, map_location="cpu")
        self.cache[shard_idx] = data
        return data

    def __getitem__(self, idx: int) -> dict:
        global_token_offset = idx * self.seq_len

        shard_idx = global_token_offset // self.shard_size
        local_offset = global_token_offset % self.shard_size

        shard_data = self._load_shard(shard_idx)

        if local_offset + self.chunk_len <= len(shard_data):
            chunk = shard_data[local_offset : local_offset + self.chunk_len]
        else:
            chunk = shard_data[local_offset:].clone()
            next_shard_idx = shard_idx + 1
            if next_shard_idx < len(self.shard_paths):
                next_shard_data = self._load_shard(next_shard_idx)
                needed = self.chunk_len - len(chunk)
                chunk = torch.cat([chunk, next_shard_data[:needed]])
            else:
                needed = self.chunk_len - len(chunk)
                pad = torch.full((needed,), 0, dtype=chunk.dtype)
                chunk = torch.cat([chunk, pad])

        chunk_long = chunk.long()
        input_ids = chunk_long[:-1]
        labels    = chunk_long[1:]

        return {"input_ids": input_ids, "labels": labels}


class DataCollatorWithPadding:
    """Collates variable-length sequences in a batch, padding them dynamically to the longest length."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __call__(self, batch: list[dict]) -> dict:
        batch_input_ids = [item["input_ids"] for item in batch]
        max_len = max(len(x) for x in batch_input_ids)

        padded_input_ids = []
        padded_labels = []

        for item in batch:
            input_ids = item["input_ids"]
            labels = item["labels"]
            length = input_ids.shape[0]
            padding_length = max_len - length

            if padding_length > 0:
                padded_input = torch.cat([
                    input_ids,
                    torch.full((padding_length,), self.pad_token_id, dtype=input_ids.dtype)
                ])
                padded_label = torch.cat([
                    labels,
                    torch.full((padding_length,), -100, dtype=labels.dtype)
                ])
            else:
                padded_input = input_ids
                padded_label = labels

            padded_input_ids.append(padded_input)
            padded_labels.append(padded_label)

        return {
            "input_ids": torch.stack(padded_input_ids, dim=0),
            "labels": torch.stack(padded_labels, dim=0),
        }
