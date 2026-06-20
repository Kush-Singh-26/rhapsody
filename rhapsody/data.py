"""Rhapsody Data Pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import json

import torch
from torch.utils.data import Dataset, IterableDataset
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer
import torchaudio


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
    if symbolic:
        special.extend(
            [
                "<|music|>",
                "<|abc_start|>",
                "<|abc_end|>",
                "<|midi_start|>",
                "<|midi_end|>",
                "<|style|>",
                "<|key|>",
                "<|meter|>",
                "<|tempo|>",
            ]
        )
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": special})
    if num_added > 0:
        print(f"[Rhapsody] Added {num_added} special tokens")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


class TextPretrainDataset(IterableDataset):
    """
    Streaming interleaved dataset for Stage-1 text pretraining.

    Mixes FineWeb-Edu, DCLM-Baseline, Stack-Edu, and Cosmopedia v2.
    Uses token-packing: tokens are buffered and sliced into fixed-length
    chunks with no wasted padding. An EOS token is appended at each
    document boundary so the model learns clean end-of-document signals.

    Labels are PRE-SHIFTED:
      input_ids[t]  = token t
      labels[t]     = token t+1   (i.e., the next token to predict)
    This is consistent with AudioTextDataset and avoids double-shifting
    in the loss computation.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        seq_len: int = 2048,
        fineweb_ratio: float = 0.55,
        dclm_ratio: float = 0.25,
        stack_edu_ratio: float = 0.15,
        cosmopedia_ratio: float = 0.05,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        datasets_list = []
        weights = []

        print("[Rhapsody] Loading FineWeb-Edu...")
        fw = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                          split="train", streaming=True)
        datasets_list.append(fw)
        weights.append(fineweb_ratio)

        print("[Rhapsody] Loading DCLM-Baseline...")
        dclm = load_dataset("mlfoundations/dclm-baseline-1.0",
                            split="train", streaming=True)
        datasets_list.append(dclm)
        weights.append(dclm_ratio)

        print("[Rhapsody] Loading Stack-Edu...")
        stack = load_dataset("HuggingFaceTB/stack-edu", split="train", streaming=True)
        datasets_list.append(stack)
        weights.append(stack_edu_ratio)

        print("[Rhapsody] Loading Cosmopedia v2...")
        cosmo = load_dataset("HuggingFaceTB/cosmopedia-v2", split="train", streaming=True)
        datasets_list.append(cosmo)
        weights.append(cosmopedia_ratio)

        self.dataset = interleave_datasets(
            datasets_list, weights=weights, stopping_strategy="all_exhausted"
        )

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

                # Append EOS at document boundary so the model learns when docs end
                if eos_id is not None:
                    tokens = tokens + [eos_id]

                buffer.extend(tokens)

                # Yield fixed-length chunks with pre-shifted labels
                while len(buffer) >= self.seq_len + 1:
                    chunk = buffer[: self.seq_len + 1]
                    buffer = buffer[self.seq_len:]   # advance by seq_len (1-token overlap is intentional)

                    input_ids = torch.tensor(chunk[:-1], dtype=torch.long)  # [seq_len]
                    labels    = torch.tensor(chunk[1:],  dtype=torch.long)  # [seq_len], pre-shifted

                    yield {"input_ids": input_ids, "labels": labels}


class AudioTextDataset(Dataset):
    """
    Map-style dataset for Stage-2 alignment and Stage-3 fine-tuning.

    Stores only lightweight metadata (text + dataset reference + index) during
    __init__. Raw audio is loaded lazily in __getitem__ via HuggingFace Arrow
    memory-mapping, avoiding the ~10 GB RAM spike that would occur from eagerly
    storing all numpy audio arrays.

    Audio is processed through ClapProcessor into mel spectrograms:
        input_features: [1, mel_bins, time_frames]  (batch dim squeezed)
    DataLoader collates these to [batch, 1, mel_bins, time_frames].

    Labels are PRE-SHIFTED (consistent with TextPretrainDataset):
      input_ids[t] = token t
      labels[t]    = token t+1   (-100 for the last position and padding)
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        seq_len: int = 2048,
        encoder_id: str = "laion/clap-htsat-unfused",
        clotho_split: str = "development",
    ):
        from transformers import ClapProcessor

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.processor = ClapProcessor.from_pretrained(encoder_id)

        # Lightweight metadata only — no raw audio arrays in RAM
        # Each entry: {"text": str, "source": "clotho"|"audioset",
        #              "dataset": HF Dataset object, "index": int}
        self.examples: list[dict] = []
        self._datasets: dict[str, object] = {}   # keyed by source name, holds HF dataset

        print(f"[Rhapsody] Loading Clotho ({clotho_split} split)...")
        try:
            # Map soundata/clotho split names to CLAPv2/Clotho split names
            clotho_split_map = {
                "development": "train",
                "validation": "validation",
                "evaluation": "test"
            }
            clotho_hf_split = clotho_split_map.get(clotho_split, clotho_split)

            # CLAPv2/Clotho is public and contains audio bytes natively
            clotho = load_dataset("CLAPv2/Clotho", split=clotho_hf_split)
            self._datasets["clotho"] = clotho
            for i, item in enumerate(clotho):
                raw_text_list = item.get("raw_text") or []
                for text in raw_text_list:
                    if text and len(text) > 20:
                        self.examples.append({
                            "text": text,
                            "source": "clotho",
                            "index": i,
                        })
            print(f"[Rhapsody] Clotho: {len(self.examples)} examples (from {len(clotho)} clips)")
        except Exception as e:
            print(f"[Rhapsody] Clotho load failed: {e}")

        print(f"[Rhapsody] Audio-text dataset total: {len(self.examples)} examples")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        example = self.examples[idx]

        # ── Tokenise text ────────────────────────────────────────────────────
        tokenized = self.tokenizer(
            example["text"],
            truncation=True,
            max_length=self.seq_len,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].squeeze(0)  # [L] (variable length)

        # Pre-shifted labels: labels[t] = input_ids[t+1], -100 at last pos
        labels = torch.full_like(input_ids, -100)
        if len(input_ids) > 1:
            labels[:-1] = input_ids[1:].clone()

        # ── Load audio lazily, extract CLAP mel features ─────────────────────
        audio_array = None
        sampling_rate = 48000

        if example["source"] == "clotho":
            # HuggingFace Arrow dataset supports O(1) random access — no RAM spike
            row = self._datasets["clotho"][example["index"]]
            audio = row.get("audio") or {}
            audio_array = audio.get("array")
            sampling_rate = audio.get("sampling_rate", 48000)

        if audio_array is not None and len(audio_array) > 0:
            try:
                feats = self.processor(
                    audios=audio_array,
                    sampling_rate=sampling_rate,
                    return_tensors="pt",
                )
                # [1, 1, mel_bins, time_frames] → squeeze batch dim → [1, mel_bins, time_frames]
                audio_features = feats.input_features.squeeze(0)
            except Exception as e:
                print(f"[Rhapsody] WARNING: audio processing failed for idx={idx}: {e}")
                audio_features = torch.zeros(1, 64, 1001)   # CLAP silence fallback
        else:
            audio_features = torch.zeros(1, 64, 1001)

        return {
            "input_ids": input_ids,         # [L]
            "labels": labels,               # [L]
            "audio_features": audio_features,  # [1, mel_bins, time_frames]
        }


class SymbolicMusicDataset(Dataset):
    """
    Dataset for symbolic music generation (ABC / MIDI-like tokens).

    Accepts either:
      1) A local JSONL file (`dataset_path`) with entries containing
         symbolic notation and optional conditioning fields.
      2) A Hugging Face dataset (`hf_dataset`) with a text field containing
         symbolic notation.

    Local JSONL expected keys (configurable):
      - symbolic field: required (ABC / REMI / MIDI-like text)
      - prompt field: optional textual control prompt
      - audio_path field: optional path for audio conditioning

    Returns pre-shifted labels, consistent with other Rhapsody datasets.
    """

    _SYMBOLIC_FIELD_CANDIDATES = (
        "symbolic",
        "abc",
        "notation",
        "score",
        "midi_tokens",
        "text",
        "assistant",
    )

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        seq_len: int = 1024,
        encoder_id: str = "laion/clap-htsat-unfused",
        dataset_path: Optional[str] = None,
        hf_dataset: Optional[str] = None,
        hf_split: str = "train",
        symbolic_field: Optional[str] = None,
        prompt_field: str = "prompt",
        audio_path_field: str = "audio_path",
        max_examples: Optional[int] = None,
    ):
        from transformers import ClapProcessor

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.prompt_field = prompt_field
        self.audio_path_field = audio_path_field
        self.symbolic_field = symbolic_field
        self.examples: list[dict] = []
        self.processor = ClapProcessor.from_pretrained(encoder_id)

        if dataset_path is not None:
            self._load_local_jsonl(Path(dataset_path), max_examples=max_examples)
        else:
            ds_name = hf_dataset or "Seeker38/music_abc_notation"
            self._load_hf_dataset(ds_name, hf_split, max_examples=max_examples)

        print(f"[Rhapsody] Symbolic dataset total: {len(self.examples)} examples")

    def _resolve_symbolic_field(self, sample: dict) -> Optional[str]:
        if self.symbolic_field is not None and self.symbolic_field in sample:
            return self.symbolic_field
        keys_lc = {k.lower(): k for k in sample.keys()}
        for candidate in self._SYMBOLIC_FIELD_CANDIDATES:
            if candidate in sample:
                return candidate
            if candidate in keys_lc:
                return keys_lc[candidate]
        return None

    def _load_local_jsonl(self, path: Path, max_examples: Optional[int]) -> None:
        print(f"[Rhapsody] Loading symbolic JSONL: {path}")
        if not path.exists():
            raise FileNotFoundError(f"Symbolic dataset path not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_examples is not None and len(self.examples) >= max_examples:
                    break
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                symbolic_key = self._resolve_symbolic_field(item)
                if symbolic_key is None:
                    continue
                symbolic_text = item.get(symbolic_key)
                if not isinstance(symbolic_text, str) or len(symbolic_text) < 8:
                    continue
                self.examples.append(
                    {
                        "symbolic": symbolic_text,
                        "prompt": item.get(self.prompt_field, ""),
                        "audio_path": item.get(self.audio_path_field),
                    }
                )
                if (i + 1) % 10000 == 0:
                    print(f"[Rhapsody] Parsed {i + 1} lines...")

    def _load_hf_dataset(self, name: str, split: str, max_examples: Optional[int]) -> None:
        print(f"[Rhapsody] Loading symbolic HF dataset: {name} [{split}]")
        ds = load_dataset(name, split=split, streaming=True)
        for row in ds:
            if max_examples is not None and len(self.examples) >= max_examples:
                break
            symbolic_key = self._resolve_symbolic_field(row)
            if symbolic_key is None:
                continue
            symbolic_text = row.get(symbolic_key)
            if not isinstance(symbolic_text, str) or len(symbolic_text) < 8:
                continue
            self.examples.append(
                {
                    "symbolic": symbolic_text,
                    "prompt": row.get(self.prompt_field, ""),
                    "audio_path": row.get(self.audio_path_field),
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def _extract_audio_features(self, audio_path: Optional[str]) -> Optional[torch.Tensor]:
        if not audio_path:
            return None
        try:
            path = Path(audio_path)
            if not path.exists():
                return None
            waveform, sr = torchaudio.load(str(path))
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != 48000:
                import torchaudio.functional as F_audio
                waveform = F_audio.resample(waveform, orig_freq=sr, new_freq=48000)
            waveform = waveform.squeeze(0)
            feats = self.processor(
                audios=waveform.numpy(),
                sampling_rate=48000,
                return_tensors="pt",
            )
            return feats.input_features.squeeze(0)
        except Exception as e:
            print(f"[Rhapsody] WARNING: failed to load or process audio from {audio_path}: {e}")
            return None

    def __getitem__(self, idx: int) -> dict:
        example = self.examples[idx]

        prompt = example.get("prompt") or ""
        symbolic = example["symbolic"]
        if prompt:
            text = f"<|music|> {prompt}\n<|abc_start|> {symbolic} <|abc_end|>"
        else:
            text = f"<|music|> <|abc_start|> {symbolic} <|abc_end|>"

        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.seq_len,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].squeeze(0)

        labels = torch.full_like(input_ids, -100)
        if len(input_ids) > 1:
            labels[:-1] = input_ids[1:].clone()

        item = {"input_ids": input_ids, "labels": labels}
        audio_features = self._extract_audio_features(example.get("audio_path"))
        if audio_features is not None:
            item["audio_features"] = audio_features
        return item


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
        audio_features_list = []

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

            if "audio_features" in item:
                audio_features_list.append(item["audio_features"])

        collated = {
            "input_ids": torch.stack(padded_input_ids, dim=0),
            "labels": torch.stack(padded_labels, dim=0),
        }

        has_audio = any("audio_features" in item for item in batch)
        if has_audio:
            # Re-build audio_features_list to ensure correct ordering and filling of missing values
            audio_features_list = []
            for item in batch:
                if "audio_features" not in item or item["audio_features"] is None:
                    # CLAP default silence features shape is [1, 64, 1001]
                    item["audio_features"] = torch.zeros(1, 64, 1001)
                audio_features_list.append(item["audio_features"])
            collated["audio_features"] = torch.stack(audio_features_list, dim=0)

        return collated
