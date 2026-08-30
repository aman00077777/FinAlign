# FinAlign — Personal Finance Advisor LLM

Fine-tuning **Mistral-7B-Instruct-v0.2** into a personal finance advisor
using **QLoRA + DPO alignment** as a 5-person team project.

> Domain: budgeting, saving, investing, debt management, credit, retirement planning.

---

## Team

| Contributor | Area | Key Files |
|---|---|---|
| **Aman** | Fine-Tuning Lead | `src/train_qlora.py`, `src/train_dpo.py`, `src/save_adapters.py`, `notebooks/02`, `notebooks/03` |
| **Sharvari** | Dataset Part A | `src/data_pipeline.py` Part A, `data/raw/`, `data/processed/` |
| **Samiksha** | Dataset Part B | `src/data_pipeline.py` Part B, `data/preference_pairs/`, splits |

---

## Project Structure

```
FinAlign/
├── data/
│   ├── raw/                        # Raw HuggingFace downloads
│   │   └── raw_combined.jsonl
│   ├── processed/                  # Cleaned SFT data
│   │   ├── finance_sft_clean.jsonl # Full clean set (10,572+ records)
│   │   ├── train.jsonl             # 80% split (quality-filtered)
│   │   ├── val.jsonl               # 10% split
│   │   └── test.jsonl              # 10% split
│   ├── preference_pairs/           # DPO training data
│   │   ├── train_pref.jsonl        # 400 chosen/rejected pairs
│   │   └── val_pref.jsonl          # 100 chosen/rejected pairs
│   ├── eval_set/
│   │   └── benchmark_100q.jsonl    # 100 held-out eval questions
│   └── data_stats_report.md
│
├── notebooks/
│   ├── 01_dataset_prep.ipynb       # Parts A + B, runnable top-to-bottom
│   ├── 02_qlora_finetune.ipynb     # QLoRA SFT (Aman)
│   └── 03_dpo_alignment.ipynb      # DPO alignment (Aman)
│
├── src/
│   ├── data_pipeline.py            # Full pipeline: A+B functions
│   ├── train_qlora.py              # SFT CLI script
│   ├── train_dpo.py                # DPO CLI script
│   └── save_adapters.py            # Export + merge adapters
│
└── reports/
    └── wandb_run_report.md
```

---

## Data Pipeline Summary

### Sources
- `Akhil-Theerthala/PersonalFinance-Reddit-QA` — 19,984 Reddit Q&A pairs
- `ceadar-ie/FinTalk-19k` — 19,111 finance conversation pairs
- Local `finalign_cleaned_dataset.jsonl` — merged for additional coverage

### Cleaning Steps (Part A)
1. **Length filter** — min 100 chars / 20 words response, max 8000 chars
2. **Exact dedup** — MD5 hash on instruction field
3. **Near-dedup** — character 5-shingle Jaccard ≥ 0.7 within buckets
4. **PII scrub** — emails, phones, SSNs, account numbers, Reddit usernames
5. **Format standardise** → `{instruction, input, output}` + finance relevance check

### Output Stats
| Metric | Value |
|---|---|
| Total raw records | 39,095+ |
| Final clean records | 10,572+ |
| Train pairs | 6,500+ |
| Val pairs | 1,050+ |
| DPO preference pairs | 500 |
| Eval questions | 100 |

---

## Prerequisites & Setup

### Git LFS (Required for Datasets)
All dataset files (`*.jsonl`) are tracked via **Git LFS**. Ensure Git LFS is installed and pulled before running pipelines or training:

```bash
# Install Git LFS hooks and pull all dataset files
git lfs install
git lfs pull
```

---

## Quick Start

### 1. Run Data Pipeline
```bash
python -c "
import sys; sys.path.insert(0,'src')
from data_pipeline import run_part_a_pipeline, run_part_b_pipeline
records, stats = run_part_a_pipeline()
run_part_b_pipeline(records)
"
```

### 2. Install training deps (Colab A100 / Linux)
```bash
pip install -r requirements.txt
```

### 3. SFT Fine-tuning
```bash
python src/train_qlora.py \
    --train_file data/processed/train.jsonl \
    --val_file   data/processed/val.jsonl \
    --output_dir checkpoints/sft_adapter \
    --wandb_project FinAlign-SFT
```

### 4. DPO Alignment
```bash
python src/train_dpo.py \
    --sft_adapter_path checkpoints/sft_adapter \
    --train_pref_file  data/preference_pairs/train_pref.jsonl \
    --val_pref_file    data/preference_pairs/val_pref.jsonl \
    --output_dir       checkpoints/dpo_adapter \
    --wandb_project    FinAlign-DPO
```

### 5. Export adapters
```bash
python src/save_adapters.py export
```

### 6. Merge adapters for deployment
```bash
python src/save_adapters.py merge \
    --dpo_adapter checkpoints/dpo_adapter \
    --output_dir exports/finalign_v1_merged
```

### 7. Run Demo
```bash
python app.py
```

---

## Dataset Formats

**SFT pairs** (`data/processed/train.jsonl`):
```json
{"instruction": "How do I build a 6-month emergency fund?", "input": "", "output": "Start by calculating..."}
```

**Preference pairs** (`data/preference_pairs/train_pref.jsonl`):
```json
{
  "prompt":   "<s>[INST] Should I invest my emergency fund in crypto? [/INST]",
  "chosen":   " No — an emergency fund should be liquid and stable...",
  "rejected": " This strategy is virtually guaranteed to work for everyone.",
  "topic":    "saving"
}
```

**Eval questions** (`data/eval_set/benchmark_100q.jsonl`):
```json
{"question": "What is the 50/30/20 budget rule?", "gold_answer": "...", "topic": "budgeting", "gold_keywords": ["budget","50","30","20","needs","wants"]}
```

---

## Hyperparameters

### SFT (QLoRA)
| Parameter | Value |
|---|---|
| LoRA r / alpha | 64 / 128 |
| Effective batch | 16 |
| LR | 2e-4 (cosine) |
| Epochs | 3 (early stop) |
| Max seq len | 2048 (packed) |

### DPO
| Parameter | Value |
|---|---|
| Beta | 0.1 |
| LR | 5e-5 |
| Epochs | 1 |
| Preference pairs | 500 |
