# ck_ner

Named Entity Recognition pipeline for medieval French texts, trained on the PRESTO corpus. Covers the full workflow from raw CoNLL data through two model families — a spaCy transformer model and a fine-tuned d'Alembert (RoBERTa-based) model — with quantitative evaluation reaching **93.5% F1**.

## Overview

This project trains and evaluates NER models on `presto_max_v5.4`, a large annotated corpus of medieval French texts in CoNLL tab-separated format. The corpus contains ~141,000 sentences and over 4.7 million tokens, annotated with eight named entity types using BIO tagging.

Two model approaches are explored in parallel:

1. **spaCy + CamemBERT** — a spaCy v3 pipeline with a `camembert-base` transformer backbone, producing the `ck-model-best` artifact. A second, lighter `ck-model-best-cnn` variant uses a CNN tok2vec encoder initialized from `fr_core_news_md` vectors instead.
2. **d'Alembert (HuggingFace Transformers)** — `pjox/dalembert`, a RoBERTa-style model pre-trained on Old and Middle French, fine-tuned via the HuggingFace `Trainer` API. This approach achieves the highest results.

The notebooks are structured as a sequential pipeline and also serve as annotated learning material, with each notebook ending in a "Lesson Review" section explaining the NLP concepts applied.

## Entity Types

| Label | Description | Token count (approx.) |
|---|---|---|
| `PERSON` | Personal names | 81,541 |
| `LOCATION` | Place names | 37,944 |
| `AMOUNT` | Quantities and numerical expressions | 11,360 |
| `FUNCTION` | Titles, roles, occupations | 9,912 |
| `TIME` | Dates, time expressions | 9,805 |
| `ORGANIZATION` | Institutional names | 3,283 |
| `PRODUCT` | Named objects, works | 975 |
| `EVENT` | Named events | 933 |

The raw corpus uses BIO prefixes (`B-pers`, `I-loc`, etc.); these are normalized to the labels above before training.

## Results

### spaCy transformer model (`ck-model-best` — CamemBERT backbone)

| Entity | Precision | Recall | F1 |
|---|---|---|---|
| PERSON | 0.915 | 0.909 | 0.912 |
| LOCATION | 0.903 | 0.921 | 0.912 |
| AMOUNT | 0.949 | 0.965 | 0.957 |
| FUNCTION | 0.917 | 0.925 | 0.921 |
| TIME | 0.927 | 0.951 | 0.939 |
| ORGANIZATION | 0.881 | 0.918 | 0.899 |
| EVENT | 0.844 | 0.931 | 0.885 |
| PRODUCT | 0.810 | 0.823 | 0.816 |
| **Overall** | **0.912** | **0.917** | **0.915** |

### spaCy CNN model (`ck-model-best-cnn` — tok2vec + French word vectors)

| Entity | Precision | Recall | F1 |
|---|---|---|---|
| PERSON | 0.851 | 0.890 | 0.870 |
| LOCATION | 0.894 | 0.834 | 0.863 |
| AMOUNT | 0.888 | 0.929 | 0.908 |
| FUNCTION | 0.856 | 0.897 | 0.876 |
| TIME | 0.951 | 0.866 | 0.906 |
| ORGANIZATION | 0.841 | 0.711 | 0.771 |
| EVENT | 0.964 | 0.931 | 0.947 |
| PRODUCT | 0.966 | 0.452 | 0.615 |
| **Overall** | **0.869** | **0.871** | **0.870** |

### d'Alembert fine-tuned (`dalembert-ner-finetuned_ep5`)

Evaluated on the HuggingFace validation split (12,700 examples) using `seqeval` entity-level scoring:

| Metric | Score |
|---|---|
| Precision | 92.9% |
| Recall | 94.1% |
| **F1** | **93.5%** |
| Accuracy | 99.7% |

Evaluation loss: `0.0182`. All metrics reported at entity span level — a span is correct only if both boundaries and label match exactly.

## Tech Stack

- **spaCy 3.7** — training framework for the CNN and transformer pipelines
- **spacy-transformers** — CamemBERT integration inside spaCy
- **HuggingFace Transformers 4.41** — `Trainer` API for d'Alembert fine-tuning
- **HuggingFace Datasets** — Arrow-format dataset storage for the transformer pipeline
- **pjox/dalembert** — RoBERTa-based masked language model pre-trained on Old/Middle French
- **camembert-base** — modern French transformer used as the spaCy backbone
- **fr_core_news_md** — spaCy French vectors (300-dim, 500k keys) for the CNN model
- **seqeval** — entity-level precision/recall/F1 evaluation
- **pandas, numpy** — data wrangling
- GPU training: RTX 4080 SUPER (noted in the training notebook); `fp16=True` enabled

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (strongly recommended; the d'Alembert training notebook explicitly targets an RTX 4080 SUPER)
- spaCy 3.7.x (`>=3.7.4,<3.8.0` per model metadata)

## Installation

```bash
# Clone the repo
git clone <repo-url>
cd ck_ner

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install spaCy and transformer support
pip install spacy>=3.7.4,<3.8.0 spacy-transformers

# Download the French pipeline needed for CNN tok2vec vectors
python -m spacy download fr_core_news_md

# Install HuggingFace dependencies
pip install transformers datasets evaluate seqeval

# Standard data-science stack
pip install pandas numpy jupyter
```

## Project Structure

```
ck_ner/
├── notebooks/
│   ├── 01_data_import.ipynb       # Parse PRESTO CoNLL file, build DataFrame
│   ├── 02_prepare_spacy.ipynb     # Convert to char-span training data, serialize to .spacy
│   ├── 03_finetune_dalembert.ipynb# Fine-tune pjox/dalembert via HuggingFace Trainer
│   ├── 04_test_spacy.ipynb        # Qualitative testing of ck-model-best with displacy
│   └── 05_test_dalembert.ipynb    # Quantitative seqeval evaluation of dalembert model
│
├── src/
│   └── hg_functions.py            # Shared helpers: label mappings, tokenization alignment,
│                                  #   compute_metrics (used by notebook 05)
│
├── data/
│   ├── presto_max_v5.4.txt        # Raw CoNLL corpus (medieval French, ~141k sentences)
│   ├── presto_max_as_csv.csv      # Parsed DataFrame cache (tokens, lemmas, pos, ent)
│   ├── ck_ner_dataset_hg/         # HuggingFace Arrow dataset (train/validation/test splits)
│   └── dalembert-ner-finetuned_tokenizer/  # Saved tokenizer for inference
│
├── training/
│   ├── config.cfg                 # spaCy transformer config (CamemBERT backbone)
│   ├── config_cnn_fr.cfg          # spaCy CNN config (tok2vec + fr_core_news_md)
│   ├── train_ck.spacy             # Training split in spaCy DocBin format
│   └── dev_ck.spacy               # Dev split in spaCy DocBin format
│
└── models/
    ├── ck-model-best/             # Best spaCy transformer model (overall F1: 0.915)
    ├── ck-model-best-cnn/         # Best spaCy CNN model (overall F1: 0.870)
    ├── dalembert-ner/             # Training checkpoints (checkpoint-28576, checkpoint-35720)
    ├── dalembert-ner-finetuned_ep1/
    └── dalembert-ner-finetuned_ep5/  # Best d'Alembert checkpoint (F1: 0.935)
```

## Pipeline Walkthrough

The five notebooks run in order and can be executed independently once their inputs exist.

### Step 1 — Data Import (`01_data_import.ipynb`)

Reads `presto_max_v5.4.txt`, a CoNLL-format file with one token per line and blank-line sentence boundaries. Each line has eight tab-separated columns; columns used are: surface token, lemma, POS tag, and NER tag (column 3, broad BIO labels).

The parser builds a sentence-indexed DataFrame with columns `text`, `tokens`, `lemmas`, `pos`, `ent`. A `detokenize()` function reconstructs the original sentence string from tokens, handling French apostrophes and punctuation.

The result is saved to `data/presto_max_as_csv.csv`.

### Step 2 — Prepare spaCy Training Data (`02_prepare_spacy.ipynb`)

Loads the CSV, remaps BIO tags to human-readable labels (`B-pers` → `PERSON`, etc.), and converts token-level labels to character-offset spans `(start_char, end_char, label)` required by spaCy.

Multi-token entities are merged into single spans by walking a character pointer through the sentence text. The data is split 90/10 (train/dev) with `random.seed(44)`, then serialized to `training/train_ck.spacy` and `training/dev_ck.spacy` using spaCy's `DocBin`.

### Step 3 — Fine-tune d'Alembert (`03_finetune_dalembert.ipynb`)

Loads the HuggingFace Arrow dataset from `data/ck_ner_dataset_hg/` and fine-tunes `pjox/dalembert` using `AutoModelForTokenClassification`. Label alignment handles BPE subword tokenization: only the first subword of each original token receives the entity label; continuation subwords are masked with `-100`.

Key hyperparameters:

| Parameter | Value |
|---|---|
| Learning rate | 3e-5 |
| Epochs | 3 (notebook config) / 5 (best checkpoint saved) |
| Batch size | 16 per device |
| Precision | fp16 |
| Weight decay | 0.01 |
| Best model criterion | validation F1 |

The trained model is saved to `models/dalembert-ner-finetuned_ep5/`.

### Step 4 — Test spaCy Model (`04_test_spacy.ipynb`)

Qualitative evaluation using `displacy.render()`. Tests three text types:
- Real medieval French sentences from the dataset
- A modern French news passage (out-of-domain generalisation probe)
- ChatGPT-generated pseudo-medieval French with unseen proper names

### Step 5 — Evaluate d'Alembert (`05_test_dalembert.ipynb`)

Runs a full quantitative evaluation of `dalembert-ner-finetuned_ep5` on the validation split (12,700 examples) using the HuggingFace `Trainer.evaluate()` and `seqeval` entity-level scoring. Uses shared helpers from `src/hg_functions.py`.

## Training the spaCy Models

### Transformer model (CamemBERT backbone)

```bash
cd training
python -m spacy train config.cfg \
  --output ./output \
  --paths.train ./train_ck.spacy \
  --paths.dev ./dev_ck.spacy \
  --gpu-id 0
```

### CNN model (tok2vec + French vectors)

```bash
cd training
python -m spacy train config_cnn_fr.cfg \
  --output ./output \
  --paths.train ./train_ck.spacy \
  --paths.dev ./dev_ck.spacy \
  --paths.vectors fr_core_news_md
```

## Running Inference

### spaCy model

```python
import spacy
from spacy import displacy

nlp = spacy.load("./models/ck-model-best")
doc = nlp("Jehan de Flores, chevalier de la cour, fut contraint quitter la cité de Lyon.")
displacy.render(doc, style="ent")

for ent in doc.ents:
    print(ent.text, ent.label_, ent.start_char, ent.end_char)
```

### d'Alembert model

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("./models/dalembert-ner-finetuned_tokenizer")
model = AutoModelForTokenClassification.from_pretrained("./models/dalembert-ner-finetuned_ep5")

ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
results = ner("Jehan de Flores, chevalier de la cité de Lyon.")
print(results)
```

## Dataset

The corpus is **PRESTO** (`presto_max_v5.4`), an annotated collection of medieval French texts. The dataset is not included in this repository beyond what is committed; the raw file `data/presto_max_v5.4.txt` must be present for the pipeline to run from scratch.

Corpus statistics derived from the parsed data:

| Statistic | Count |
|---|---|
| Total sentences | 141,103 |
| Total tokens | ~4.76 million |
| Entity tokens | ~155,000 (3.3% of corpus) |
| Entity types | 8 |

The HuggingFace dataset at `data/ck_ner_dataset_hg/` contains `train`, `validation`, and `test` splits with `tokens` and `ner_tags` sequence features. The validation split used in notebook 05 contains 12,700 examples.
