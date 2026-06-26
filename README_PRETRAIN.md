# Rhapsody TextLM (Base Model)

Rhapsody TextLM is an 84-million parameter, decoder-only Transformer language model. It is designed as a lightweight, high-performance base model optimized for resource-constrained environments (such as Google Colab T4 GPUs).

## Model Details

- **Developer:** Kush Singh
- **Model Type:** Decoder-only Transformer
- **Language:** English
- **License:** MIT
- **Tokenizer:** `HuggingFaceTB/cosmo2-tokenizer` (vocab size = 49,152) + 3 special tokens (`<|pad|>`, `<|audio|>`, `<|text|>`) = **49,155 total vocab**

## Architecture

Rhapsody TextLM follows modern LLM architecture principles (similar to Llama 3 and SmolLM2), utilizing a deep-and-thin configuration for high capacity at a small parameter scale:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Hidden Size** | 512 | Model dimensionality |
| **Layers** | 20 | Number of transformer blocks |
| **Attention Heads** | 8 | Number of Query heads |
| **Key-Value Heads** | 4 | GQA (Grouped-Query Attention) with 2:1 ratio |
| **Intermediate Size** | 1408 | MLP hidden dimension (SwiGLU activation) |
| **Max Position Embeddings** | 2048 | Context window size |
| **Positional Embeddings** | RoPE | Rotary Position Embeddings (theta = 100,000) |
| **Tied Word Embeddings** | True | Shares weights between embed and LM head |
| **Total Parameters** | 84,170,992 | ~84M parameters |

## Pre-training Details

The model was pre-trained using token-packed autoregressive next-token prediction:
* **Dataset Mix:** An interleaved stream of FineWeb-Edu, DCLM-Baseline, and Cosmopedia v2.
* **Sequence Length:** 2048 tokens.
* **Target Labels:** Pre-shifted next-token targets, fully optimized for causal attention.

## How to Use

You can load and query the base model using the code structure in the Rhapsody repository:

```python
import torch
from rhapsody.inference import load_model, generate_text
from rhapsody.data import get_tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pretrained model
model = load_model("path/to/pretrained_model/model.safetensors", device=device)
tokenizer = get_tokenizer(symbolic=False)

# Run standard autoregressive completion
prompt = "Once upon a time in a distant galaxy,"
output = generate_text(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_new_tokens=100,
    temperature=0.7,
    device=device
)
print(output)
```
