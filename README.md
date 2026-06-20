# Rhapsody

Small, lightweight audio-language model training codebase. Designed to run as a standalone repository and easily sync training checkpoints across sequential Colab and Kaggle sessions.

## Tasks

Rhapsody supports two task families:

| Task | Description |
|---|---|
| **Audio Captioning** | Audio → text description (default) |
| **Symbolic Music Generation** | Audio/text prompt → ABC notation or MIDI-like tokens |

The symbolic music task uses a CLAP embedding as a style/mood conditioner and autoregressively generates structured symbolic sequences (ABC notation). This sidesteps the token-length problem of raw audio generation — symbolic sequences are short, structured, and learnable by a 65M model.

---

## Architecture

```text
Task 1 (Audio Captioning):
  Audio input -> CLAP encoder (frozen) -> projector -> ~65M text LM -> text output

Task 2 (Symbolic Music):
  Audio/text prompt -> CLAP encoder (frozen) -> projector -> ~65M text LM -> ABC notation
```

| Component | Details |
|---|---|
| **Text LM** | 20 layers, 512 hidden size, 8 query heads, 4 KV heads, SwiGLU, RoPE, RMSNorm, tied embeddings |
| **Audio Encoder** | `laion/clap-htsat-unfused` (CLAP audio tower, ~31M parameters), frozen by default |
| **Projector** | MLP mapping CLAP embedding size 768 to LM hidden size 512 |
| **Attention** | PyTorch SDPA with native GQA; custom prefix-LM mask for audio-conditioned text |

The core text model is sized to be roughly 65M parameters with a 32K-token vocabulary (using `HuggingFaceTB/cosmo2-tokenizer`).

---

## Setup & Notebook Deployment

### 1. Push to your GitHub (First Time)
Since you will be running this on remote machines (Colab/Kaggle), you need to push this repository to GitHub so it can be cloned:

```bash
# In your local /home/kush26/Projects/rhapsody directory:
git add .
git commit -m "Initial commit: standalone rhapsody with DDP & Clotho support"
git branch -M main
git remote add origin https://github.com/Kush-Singh-26/rhapsody.git
git push -u origin main
```

---

### 2. Running in Colab or Kaggle (Jupyter Notebooks)

You do **not** need to use `uv` on Colab/Kaggle (though you can). Standard `pip` works out-of-the-box. 

Create a cell at the top of your notebook to clone your repo and install the dependencies:

```python
# 1. Clone the repository
!git clone https://github.com/Kush-Singh-26/rhapsody.git
%cd rhapsody

# 2. Log in to HuggingFace (required for gated weights/pulling checkpoints)
!pip install huggingface_hub
!huggingface-cli login

# 3. Option A: Standard Fast Installation (Direct Pip)
!pip install -e .

# 3. Option B: Blazing Fast Installation (Using UV)
# !pip install uv
# !uv pip install --system -e .
```

*Note: The `-e .` (editable mode) flag installs the package using the `pyproject.toml` dependencies and registers the `rhapsody-train` command globally on the VM.*

---

## Datasets

By default, the training pipeline uses:
* **Audio Captioning (default task):** 
  * **Clotho (`soundata/clotho`):** 4,981 clips. The loader automatically extracts all **5 human captions** for each clip, yielding **~19,000 highly diverse audio-text pairs** during alignment. No YouTube downloading is required as the raw audio bytes stream directly from the Hugging Face cache.
  * **AudioSet (`EleutherAI/...`):** Capped at 2,000 examples with native audio bytes.
* **Symbolic Music Generation:**
  * **ABC Notation (`Seeker38/music_abc_notation`):** 383,000 ABC tunes.

---

## Training

The repository supports distributed multi-GPU training (via Hugging Face Accelerate DDP) out-of-the-box. It automatically handles mixed precision (BF16 on L4/A100, FP16 on T4) and dynamically scales dataset indices and checkpoint loading parameters.

### 1. Single GPU Training (Google Colab 1x T4 / A100)
Run standard sequential training:

```bash
# Stage 1: Text Pretraining (TextLM only)
rhapsody-train --stage pretrain --max-steps 100000 --batch-size 4 --grad-accum 16 --lr 0.0008

# Stage 2: Audio-Text Alignment (Projector only, TextLM frozen)
rhapsody-train --stage align --max-steps 5000 --batch-size 4 --grad-accum 8 --lr 0.0001 --pretrained-lm ./outputs/stage1/final

# Stage 3: Fine-Tuning (Full multimodal model, CLAP frozen)
rhapsody-train --stage finetune --max-steps 3000 --batch-size 2 --grad-accum 16 --lr 0.00005 --pretrained-lm ./outputs/stage2/final
```

### 2. Multi-GPU Distributed Training (Kaggle 2x T4 GPUs)
Launch using `torchrun` to train on both GPUs in parallel:

```bash
# Stage 2: Alignment (Halved grad-accum to keep global batch size at 64)
torchrun --nproc_per_node=2 train.py --stage align --max-steps 5000 --batch-size 4 --grad-accum 8 --pretrained-lm ./outputs/stage1/final
```

*Note: Keeping the global effective batch size (`batch_size * grad_accum * num_gpus`) identical across environments prevents optimization trajectory mismatch and stabilizes learning rate schedules.*

---

## Nomad Checkpointing (`config.yaml`)

This repository uses a `config.yaml` state file linked to `lm_forge` to enable robust, multi-session training. If a Colab or Kaggle session terminates:
1. It automatically saves local checkpoints periodically.
2. It pushes the latest checkpoint state to a designated Hugging Face Hub repository in a daemon thread.
3. Upon restart, it pulls the latest checkpoint from the Hub and **instantly fast-forwards the dataset** (using `torch.utils.data.Subset` for Map datasets) to the exact step where it was preempted.

To run with nomad checkpointing:
```bash
rhapsody-train --stage align --auto-resume --forge-config ./config.yaml --save-steps 500
```

### The `config.yaml` Schema:
```yaml
name: "rhapsody-65m"
state:
  repo_id: "Kush26/rhapsody-65m-checkpoints"
  branch: "checkpoints"
  checkpoint_limit: 3
  push_every: 500
  private: true
```

---

## Inference

### Audio Captioning
```bash
python -m rhapsody.inference \
  --checkpoint ./outputs/final/model.pt \
  --audio path/to/clip.wav \
  --prompt "Describe this audio:" \
  --max-new-tokens 128
```

### Symbolic Music Generation
Autoregressively generate ABC notation:
```bash
python -m rhapsody.inference \
  --checkpoint ./outputs/final/model.pt \
  --task symbolic-music \
  --audio path/to/reference.wav \
  --max-new-tokens 256
```

---

## Notes
* `--max-steps` is the exact number of **optimizer steps** (weight updates) to perform.
* `--lr` sets the base AdamW learning rate (for norms, biases, embeddings, and projector parameters). The Muon orthogonal learning rate for weight matrices is fixed internally at `0.015`.
* Gradient scaling and accumulation parameters are automatically managed by the `Accelerator`. Saved checkpoints automatically strip out DDP `module.` wrappers, enabling cross-compatibility between multi-GPU and single-GPU environments.
