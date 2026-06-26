"""
Rhapsody: A small audio-language model.
Architecture: Frozen audio encoder + projector + ~65M text LM

Based on SmolLM2-135M architecture scaled down for T4-friendly training.
Key principles: deep-and-thin, SwiGLU, RoPE, RMSNorm, GQA.
"""

from __future__ import annotations

import contextlib
import math
import dataclasses
from dataclasses import dataclass
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RhapsodyConfig:
    """Configuration for Rhapsody model."""

    # Text LM dimensions (roughly 65M params at a 32K vocab)
    hidden_size: int = 512
    num_hidden_layers: int = 20
    num_attention_heads: int = 8
    num_key_value_heads: int = 4  # GQA: 2:1 ratio
    intermediate_size: int = 1408  # SwiGLU FFN (Note: intentionally conservative; upgraded to 1408 for optimal capacity scaling)
    vocab_size: int = 32000
    max_position_embeddings: int = 2048
    rope_theta: float = 100000.0
    tie_word_embeddings: bool = True  # Tie at 65M to save ~16M params
    dropout: float = 0.0

    # Audio encoder — CLAP HTSAT-tiny (laion/clap-htsat-unfused)
    # Audio tower: ~31M params, trained on 630K audio-text pairs (music + general audio).
    # Internal hidden size = 768; outputs [B, 768, freq, time] which we flatten to [B, T, 768].
    audio_encoder_dim: int = 768  # HTSAT internal hidden size (before CLAP projection head)
    audio_encoder_type: str = "laion/clap-htsat-unfused"
    freeze_audio_encoder: bool = True

    # Projector dimensions (maps audio_encoder_dim → text hidden_size)
    projector_hidden: int = 512
    projector_layers: int = 2
    projector_dropout: float = 0.1

    # Training
    gradient_checkpointing: bool = False

    def to_config_dict(self) -> dict:
        """Serialize all config fields (text LM + audio + projector + training)."""
        return {
            # Text LM
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.rope_theta,
            "tie_word_embeddings": self.tie_word_embeddings,
            "dropout": self.dropout,
            # Audio encoder
            "audio_encoder_dim": self.audio_encoder_dim,
            "audio_encoder_type": self.audio_encoder_type,
            "freeze_audio_encoder": self.freeze_audio_encoder,
            # Projector
            "projector_hidden": self.projector_hidden,
            "projector_layers": self.projector_layers,
            "projector_dropout": self.projector_dropout,
            # Training
            "gradient_checkpointing": self.gradient_checkpointing,
        }

    @classmethod
    def from_config_dict(cls, d: dict) -> "RhapsodyConfig":
        """Restore config from a serialized dict (ignores unknown keys)."""
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid_fields})


# =============================================================================
# RoPE (Rotary Position Embeddings) — single shared instance per model
# =============================================================================

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE).

    A single shared instance is created at the TextLM level and passed
    to every GroupedQueryAttention layer, saving 24x repeated buffer memory.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin for max_seq_len
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
    """Rotates half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings to q and k."""
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
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
    """
    Grouped-Query Attention with RoPE.

    Accepts a *shared* RotaryEmbedding instance so that all 24 layers
    re-use the same pre-computed cos/sin buffers (24x memory saving).

    Attention mask convention:
    - None → hardware-efficient causal masking via is_causal=True (training).
    - 4D float tensor [B, 1, S, S] → additive mask (0 = attend, -inf = block).
      Used for the multimodal prefix-LM forward pass.
    """

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

        # Learnable QK-Norm scale parameters (Fix #4)
        self.q_norm_scale = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.k_norm_scale = nn.Parameter(torch.ones(num_kv_heads, 1, 1))

        # Shared RoPE — NOT created here, injected from TextLM
        self.rotary_emb = rotary_emb
        self.attn_dropout_p = dropout

        # Detect once whether F.scaled_dot_product_attention supports enable_gqa
        # (added in PyTorch 2.5). Check at init to avoid try/except in the hot path.
        self._sdpa_supports_gqa = True
        try:
            major, minor = map(int, torch.__version__.split(".")[:2])
            if (major < 2) or (major == 2 and minor < 5):
                self._sdpa_supports_gqa = False
        except Exception:
            pass



    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = x.shape

        # Project
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        past_len = past_key_value[0].shape[-2] if past_key_value is not None else 0
        cos, sin = self.rotary_emb(seq_len, past_len=past_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Apply QK-Norm: use F.normalize (single fused kernel) instead of
        # manual rsqrt+pow+mean which generates 3+ separate XLA ops per head.
        # F.normalize sets L2 norm to 1. We must multiply by sqrt(head_dim)
        # to match RMSNorm variance, then apply the learnable scale.
        import math
        q_scale = math.sqrt(self.head_dim)
        q = F.normalize(q, dim=-1) * (q_scale * self.q_norm_scale)
        k = F.normalize(k, dim=-1) * (q_scale * self.k_norm_scale)

        # Update cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=-2)
            v = torch.cat([past_key_value[1], v], dim=-2)

        present_key_value = (k, v) if use_cache else None

        # Scaled dot-product attention (uses Flash Attention where available)
        # - mask is None  → is_causal=True (efficient triangular mask, text-only)
        # - mask provided → 4D additive mask [B,1,S,S] (prefix-LM, multimodal)
        # During decoding step (seq_len == 1) or when past_key_value is not None, is_causal is False.
        is_causal = (mask is None and seq_len > 1 and past_key_value is None)
        if self._sdpa_supports_gqa and self.num_kv_groups > 1:
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.attn_dropout_p if self.training else 0.0,
                is_causal=is_causal,
                enable_gqa=True,
            )
        else:
            if self.num_kv_groups > 1:
                k = k.repeat_interleave(self.num_kv_groups, dim=1)
                v = v.repeat_interleave(self.num_kv_groups, dim=1)
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.attn_dropout_p if self.training else 0.0,
                is_causal=is_causal,
            )

        # Reshape and project (reshape avoids a forced contiguous() copy)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.o_proj(attn_output), present_key_value


# =============================================================================
# Transformer Block
# =============================================================================

class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm (norm-first)."""

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
# Prefix-LM Attention Mask Helper
# =============================================================================

def _build_prefix_causal_mask(
    audio_len: int,
    text_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build a prefix-LM additive attention mask for audio + text sequences:
    - Audio prefix tokens attend to each other fully (bidirectional).
    - Text tokens attend causally to past text tokens + all audio tokens.

    Returns: [1, 1, total_len, total_len] float mask.
             0.0 = attend, -inf = block.
    """
    total_len = audio_len + text_len
    # Standard causal upper-triangular mask (0 on/below diagonal, -inf above)
    mask = torch.triu(
        torch.full((total_len, total_len), float("-inf"), device=device, dtype=dtype),
        diagonal=1,
    )
    # Remove causal restriction within audio prefix → full bidirectional attention
    mask[:audio_len, :audio_len] = 0.0
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, total_len, total_len]


# =============================================================================
# Text Language Model (65M params)
# =============================================================================

class TextLM(nn.Module):
    """
    Decoder-only Transformer Language Model (~65M parameters at 32K vocab).

    Architecture: deep-and-thin design following MobileLLM/SmolLM2 principles.
    - 20 layers x 512 hidden = deep and thin (better at small scale)
    - GQA with 4 KV heads (2:1 ratio)
    - SwiGLU FFN
    - RoPE positional embeddings (one shared RotaryEmbedding across all layers)
    - RMSNorm (pre-norm)
    - Tied embeddings (saves ~16M params at 32K vocab)

    Supports both input_ids (standard) and inputs_embeds (multimodal fusion).

    Label convention:
    - Labels must be pre-shifted: labels[t] = the next token after input_ids[t].
    - Use -100 to ignore padding or the final position.
    - This avoids a double-shift bug when combined with pre-shifted dataset outputs.
    """

    def __init__(self, config: RhapsodyConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Single shared RoPE module — injected into all attention layers
        self.rotary_emb = RotaryEmbedding(
            dim=config.hidden_size // config.num_attention_heads,
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta,
        )

        # Transformer layers — all share the same rotary_emb instance
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

        # Final norm
        self.norm = RMSNorm(config.hidden_size)

        # LM head (optionally tied with embedding)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Scale output projection weights by 1/sqrt(2 * num_layers) to stabilize residual stream
        scale = 1.0 / math.sqrt(2 * config.num_hidden_layers)
        for name, module in self.named_modules():
            if name.endswith(".attn.o_proj") or name.endswith(".ffn.down_proj"):
                module.weight.data.mul_(scale)

        # Log parameter count
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
        """
        Args:
            input_ids: [batch, seq_len]. Mutually exclusive with inputs_embeds.
            inputs_embeds: [batch, seq_len, hidden_size]. Used in multimodal path.
            attention_mask: 4D additive float mask [batch, 1, seq_len, seq_len].
                            None → fast causal masking (is_causal=True) in each layer.
            labels: [batch, seq_len]. Pre-shifted next-token targets.
                    -100 marks positions to ignore in the loss.
            past_key_values: list of [past_k, past_v] caches for each layer.
            use_cache: whether to compute and return the updated KV caches.
        """
        if inputs_embeds is not None:
            x = inputs_embeds
        elif input_ids is not None:
            x = self.embed_tokens(input_ids)
        else:
            raise ValueError("Provide either input_ids or inputs_embeds.")

        # Forward through transformer layers
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

        # Final norm + LM head
        x = self.norm(x)
        logits = self.lm_head(x)

        # Loss: CE over pre-shifted labels, ignoring -100 positions
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


# =============================================================================
# Audio Projector
# =============================================================================

class AudioProjector(nn.Module):
    """
    Projects audio encoder features into the text LM embedding space.
    MLP: num_layers Linear layers with GELU activations and dropout between them.

    For num_layers=1: input_dim → output_dim (single linear).
    For num_layers=2: input_dim → hidden_dim → output_dim (default).
    For num_layers=N: input_dim → (N-1 hidden layers) → output_dim.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers must be at least 1."

        layers = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))

        self.projector = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_features: [batch, audio_seq_len, audio_dim]
        Returns:
            projected: [batch, audio_seq_len, text_hidden_dim]
        """
        return self.projector(audio_features)


# =============================================================================
# Rhapsody Model (Complete)
# =============================================================================

class RhapsodyModel(nn.Module):
    """
    Rhapsody: Multimodal Audio-Language Model.

    Architecture:
        Audio Input -> Audio Encoder (frozen) -> Projector (trainable) -> Text LM (trainable)

    Total trainable: ~65M params for the LM plus the audio projector.
    Total inference: trainable params plus the frozen CLAP audio encoder.
    """

    def __init__(self, config: RhapsodyConfig):
        super().__init__()
        self.config = config

        # 1. Audio Encoder (frozen CLAP HTSAT-tiny)
        self.audio_encoder = self._load_audio_encoder()

        # 2. Audio Projector (trainable)
        self.projector = AudioProjector(
            input_dim=config.audio_encoder_dim,
            output_dim=config.hidden_size,
            hidden_dim=config.projector_hidden,
            num_layers=config.projector_layers,
            dropout=config.projector_dropout,
        )

        # 3. Text Language Model (trainable)
        self.text_lm = TextLM(config)

        # Freeze audio encoder parameters
        if config.freeze_audio_encoder:
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
            print("[Rhapsody] Audio encoder frozen")

        # Log parameter counts
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[Rhapsody] Total parameters:     {total:,}")
        print(f"[Rhapsody] Trainable parameters: {trainable:,}")
        print(f"[Rhapsody] Frozen parameters:    {total - trainable:,}")

    def train(self, mode: bool = True):
        """Override train to keep the audio encoder in eval mode if frozen."""
        super().train(mode)
        if self.config.freeze_audio_encoder and hasattr(self, "audio_encoder"):
            self.audio_encoder.eval()

    def _load_audio_encoder(self):
        """
        Load the CLAP audio tower (HTSAT-tiny, ~31M params).

        We load only the audio model — the text branch of CLAP is discarded.
        CLAP is trained on 630K audio-text pairs covering music, speech, and
        environmental sounds, making it semantically appropriate for Rhapsody.
        """
        from transformers import ClapAudioModel

        print(f"[Rhapsody] Loading CLAP audio encoder: {self.config.audio_encoder_type}")
        audio_model = ClapAudioModel.from_pretrained(self.config.audio_encoder_type)
        return audio_model

    def encode_audio(self, audio_input_features: torch.Tensor) -> torch.Tensor:
        """
        Encode CLAP mel features through the frozen HTSAT encoder + trainable projector.

        HTSAT is a hierarchical vision-transformer applied to mel spectrograms.
        Recent Transformers CLAP returns a sequence [B, T, C]. Older/local
        variants may return spatial features [B, C, F, T], which are flattened.

        Args:
            audio_input_features: CLAP mel features [batch, 1, freq_bins, time_frames]
                                  (as produced by ClapProcessor).
        Returns:
            Projected token sequence [batch, num_audio_tokens, hidden_size]
        """
        # Use no_grad when encoder is frozen, otherwise allow gradients (e.g. unfrozen finetune)
        ctx = torch.no_grad() if self.config.freeze_audio_encoder else contextlib.nullcontext()
        with ctx:
            raw = self.audio_encoder(audio_input_features).last_hidden_state

        if raw.ndim == 4:
            # [B, C, F, T] -> [B, F*T, C]
            B, C, Freq, Time = raw.shape
            encoder_output = raw.permute(0, 2, 3, 1).reshape(B, Freq * Time, C)
        elif raw.ndim == 3:
            # [B, T, C]
            encoder_output = raw
        else:
            raise RuntimeError(f"Unexpected CLAP hidden state shape: {tuple(raw.shape)}")

        projected = self.projector(encoder_output)   # [B, freq*time, hidden_size]
        return projected

    def forward(
        self,
        input_ids: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> dict:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch, text_seq_len]
            audio_features: Optional CLAP mel features [batch, 1, freq_bins, time_frames]
                            (produced by ClapProcessor). None for text-only forward.
            attention_mask: Unused in multimodal mode (computed internally).
                            Forwarded to text_lm in text-only mode.
            labels: Pre-shifted targets [batch, text_seq_len]. -100 for ignored positions.
            past_key_values: list of [past_k, past_v] caches for each layer.
            use_cache: whether to compute and return the updated KV caches.

        Returns:
            {"logits": Tensor, "loss": Tensor or None, "past_key_values": ... (if use_cache is True)}
        """
        if audio_features is not None:
            # ── Multimodal forward ─────────────────────────────────────────
            audio_embeds = self.encode_audio(audio_features)    # [B, audio_len, H]
            text_embeds = self.text_lm.embed_tokens(input_ids)  # [B, text_len, H]

            combined_embeds = torch.cat([audio_embeds, text_embeds], dim=1)
            audio_len = audio_embeds.shape[1]
            text_len = text_embeds.shape[1]

            # Prefix-LM mask: audio tokens attend bidirectionally,
            # text tokens attend causally + see all audio tokens.
            prefix_mask = _build_prefix_causal_mask(
                audio_len, text_len,
                device=audio_embeds.device,
                dtype=audio_embeds.dtype,
            )

            # Forward through TextLM using pre-computed combined embeddings
            output = self.text_lm(
                inputs_embeds=combined_embeds,
                attention_mask=prefix_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            logits = output["logits"]  # [B, audio_len + text_len, vocab]

            # Compute loss only on the text portion (not the audio prefix)
            loss = None
            if labels is not None:
                text_logits = logits[:, audio_len:, :]   # [B, text_len, vocab]
                loss = F.cross_entropy(
                    text_logits.view(-1, self.config.vocab_size),
                    labels.view(-1),
                    ignore_index=-100,
                )

            res = {"logits": logits, "loss": loss}
            if use_cache:
                res["past_key_values"] = output["past_key_values"]
            return res
        else:
            # ── Text-only forward ──────────────────────────────────────────
            return self.text_lm(
                input_ids,
                attention_mask=attention_mask,
                labels=labels,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )


# =============================================================================
# Factory Functions
# =============================================================================

def create_rhapsody_65m(vocab_size: int = 32000) -> RhapsodyModel:
    """
    Create the full Rhapsody multimodal model.

    Audio encoder: CLAP HTSAT-tiny (~31M frozen)
      - Trained on 630K audio-text pairs (music + speech + environmental audio)
      - Internal hidden dim: 768 → projected to 512 (text LM hidden size)
    """
    config = RhapsodyConfig(
        hidden_size=512,
        num_hidden_layers=20,
        num_attention_heads=8,
        num_key_value_heads=4,
        intermediate_size=1408,
        vocab_size=vocab_size,
        max_position_embeddings=2048,
        rope_theta=100000.0,
        tie_word_embeddings=True,
        audio_encoder_dim=768,             # CLAP HTSAT internal hidden size
        audio_encoder_type="laion/clap-htsat-unfused",
        projector_hidden=512,
        projector_layers=2,
    )
    return RhapsodyModel(config)


def create_text_only_65m(vocab_size: int = 49155) -> TextLM:
    """Create the text-only 65M LM (for Stage 1 pretraining)."""
    config = RhapsodyConfig(
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


if __name__ == "__main__":
    # Smoke-test forward pass (text-only, no CLAP download needed)
    print("=" * 60)
    print("Creating TextLM-65M (text-only)")
    print("=" * 60)
    text_model = create_text_only_65m(vocab_size=32000)

    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, 32000, (batch_size, seq_len))
    # Pre-shifted labels: labels[t] = input_ids[t+1], -100 for last position
    labels = torch.full_like(input_ids, -100)
    labels[:, :-1] = input_ids[:, 1:]

    with torch.no_grad():
        output = text_model(input_ids, labels=labels)
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Loss: {output['loss'].item():.4f}")
