---
license: mit
datasets:
- taucris/haiku_333K
language:
- en
base_model:
- Kush26/rhapsody-84m-pretrain
pipeline_tag: text-generation
---
# Rhapsody Constraint Poet (Haiku Generator)

Rhapsody Constraint Poet is an 84-million parameter Transformer language model fine-tuned to generate structured, topic-constrained 5-7-5 haikus. 

It is fine-tuned from the base **Rhapsody TextLM** (84M) on a subset of the `taucris/haiku_333K` dataset.

## Model Details

- **Developer:** Kush Singh
- **Model Type:** Decoder-only Transformer
- **Fine-tuned From:** `Rhapsody TextLM` (84M base)
- **Language:** English
- **License:** MIT
- **Task:** Topic-constrained 5-7-5 Haiku Generation
- **Inference Mode:** Autoregressive Generation (temperature = 0.7, repetition_penalty = 1.15)

## Fine-tuning Details

The model was fine-tuned for sequence-to-sequence style prompt-response generation:
* **Training Format:**
  `Write a haiku about {topic}.\n{haiku_line_1}\n{haiku_line_2}\n{haiku_line_3}<|endoftext|>`
* **Dataset Size:** 50,000 samples from the `taucris/haiku_333K` dataset.
* **Epochs:** 3
* **Batch Size:** 64
* **Learning Rate:** 5e-5 (AdamW optimizer, weight decay = 0.01)
* **Optimization:** Shifted target labels (`labels[t] = input_ids[t+1]`), with prompt tokens and padding fully masked out (`-100`) so the model only computes loss on the poem contents and the final `<|endoftext|>` token.

## Evaluation Metrics

Evaluated on 50 test samples from the `taucris/haiku_333K` validation set:

* **3-Line Structure Accuracy:** **100.0%** (All samples successfully generated exactly 3 lines).
* **Exact 5-7-5 Syllable Accuracy:** **8.0%** (Generates perfect 5-7-5 syllable structures on unseen topics).
* **Average Syllable Error:** **2.28** syllables off per poem.

## How to Use

You can run interactive generation locally using the model checkpoint:

```python
import torch
from rhapsody.inference import load_model, generate_text
from rhapsody.data import get_tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
model_path = "outputs_poet/poet_model.safetensors"
model = load_model(model_path, device=device)
tokenizer = get_tokenizer(symbolic=False)

# Define your custom topic
topic = "lonely night"
prompt = f"Write a haiku about {topic}.\n"

# Autoregressive generation
completion = generate_text(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_new_tokens=40,
    temperature=0.7,
    repetition_penalty=1.15,
    device=device
)

print(completion.strip())
```

### Example Outputs

* **Topic: `lonely night`**
  ```text
  stars shine in the dark
  their light a beacon of hope
  night's lonely plea
  ```
  *(Syllables: 5-7-5)*

* **Topic: `delhi`**
  ```text
  the chefs knife slices
  delhi's tender flesh and crust
  flavors on the plate
  ```
  *(Syllables: 5-7-5)*

* **Topic: `new york`**
  ```text
  old york's rusted gate
  creaks but never breaks the night
  history's hold tight
  ```
  *(Syllables: 5-7-5)*
