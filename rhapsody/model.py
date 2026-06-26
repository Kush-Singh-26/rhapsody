"""
Rhapsody Text Language Model (~65M/84M parameters).
Decoder-only Transformer LM: deep-and-thin, SwiGLU, RoPE, RMSNorm, GQA.
Optimized for resource-friendly pre-training and fine-tuning.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class LmConfig:
    """Configuration for the text-only language model."""

    hidden_size: int = 512
    num_hidden_layers: int = 20
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    intermediate_size: int = 1408
    vocab_size: int = 49155
    max_position_embeddings: int = 2048
    rope_theta: float = 100000.0
    tie_word_embeddings: bool = True
    dropout: float = 0.0
    gradient_checkpointing: bool = False

    def to_config_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}

    @classmethod
    def from_config_dict(cls, d: dict) -> "LmConfig":
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid_fields})


# Aliases for backwards compatibility
RhapsodyConfig = LmConfig


# =============================================================================
# RoPE (Rotary Position Embeddings)
# =============================================================================

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE).
    A single shared instance is created at the TextLM level and passed
    to every GroupedQueryAttention layer to save buffer memory.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, past_len: int = 0):
        return (
            self.cos_cached[past_len : past_len + seq_len],
            self.sin_cached[past_len : past_len + seq_len],
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# =============================================================================
# RMSNorm
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


# =============================================================================
# SwiGLU Feed-Forward Network
# =============================================================================

class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, dim: int, intermediate_dim: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


# =============================================================================
# Grouped-Query Attention (GQA)
# =============================================================================

class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention with RoPE."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rotary_emb: RotaryEmbedding,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.num_kv_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)

        self.q_norm_scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.k_norm_scale = nn.Parameter(torch.ones(num_kv_heads, 1, 1))

        self.rotary_emb = rotary_emb
        self.attn_dropout_p = dropout

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        past_len = past_key_value[0].shape[-2] if past_key_value is not None else 0
        cos, sin = self.rotary_emb(seq_len, past_len=past_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        q_scale = torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + 1e-6)
        k_scale = torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + 1e-6)
        q = q * q_scale * self.q_norm_scale
        k = k * k_scale * self.k_norm_scale

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=-2)
            v = torch.cat([past_key_value[1], v], dim=-2)

        present_key_value = (k, v) if use_cache else None

        is_causal = (mask is None and seq_len > 1 and past_key_value is None)
        try:
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.attn_dropout_p if self.training else 0.0,
                is_causal=is_causal,
                enable_gqa=self.num_kv_groups > 1,
            )
        except TypeError:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.attn_dropout_p if self.training else 0.0,
                is_causal=is_causal,
            )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output), present_key_value


# =============================================================================
# Transformer Block
# =============================================================================

class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_dim: int,
        rotary_emb: RotaryEmbedding,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = GroupedQueryAttention(dim, num_heads, num_kv_heads, rotary_emb, dropout)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, intermediate_dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, present_key_value = self.attn(
            self.norm1(x), mask, past_key_value=past_key_value, use_cache=use_cache
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, present_key_value


# =============================================================================
# Text Language Model
# =============================================================================

class TextLM(nn.Module):
    """
    Decoder-only Transformer Language Model.
    Architecture: deep-and-thin design following MobileLLM/SmolLM2 principles.
    """

    def __init__(self, config: LmConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.rotary_emb = RotaryEmbedding(
            dim=config.hidden_size // config.num_attention_heads,
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta,
        )

        self.layers = nn.ModuleList([
            TransformerBlock(
                dim=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                intermediate_dim=config.intermediate_size,
                rotary_emb=self.rotary_emb,
                dropout=config.dropout,
            )
            for _ in range(config.num_hidden_layers)
        ])

        self.norm = RMSNorm(config.hidden_size)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)

        scale = 1.0 / math.sqrt(2 * config.num_hidden_layers)
        for name, module in self.named_modules():
            if name.endswith(".attn.o_proj") or name.endswith(".ffn.down_proj"):
                module.weight.data.mul_(scale)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[Rhapsody] TextLM initialized: {n_params:,} parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> dict:
        if inputs_embeds is not None:
            x = inputs_embeds
        elif input_ids is not None:
            x = self.embed_tokens(input_ids)
        else:
            raise ValueError("Provide either input_ids or inputs_embeds.")

        next_decoder_cache = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            if self.config.gradient_checkpointing and self.training:
                x, present_key_value = torch.utils.checkpoint.checkpoint(
                    layer, x, attention_mask, past_key_value, use_cache, use_reentrant=False
                )
            else:
                x, present_key_value = layer(
                    x, attention_mask, past_key_value=past_key_value, use_cache=use_cache
                )
            if use_cache:
                next_decoder_cache.append(present_key_value)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        res = {"logits": logits, "loss": loss}
        if use_cache:
            res["past_key_values"] = next_decoder_cache
        return res


# Aliases for backwards compatibility
RhapsodyModel = TextLM


# =============================================================================
# Factory Function
# =============================================================================

def create_text_only_65m(vocab_size: int = 49155) -> TextLM:
    """Create the text-only 65M/84M LM."""
    config = LmConfig(
        hidden_size=512,
        num_hidden_layers=20,
        num_attention_heads=8,
        num_key_value_heads=4,
        intermediate_size=1408,
        vocab_size=vocab_size,
        max_position_embeddings=2048,
        rope_theta=100000.0,
        tie_word_embeddings=True,
    )
    return TextLM(config)
