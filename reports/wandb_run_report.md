# W&B Training Run Report — FinAlign

## SFT Run (QLoRA)

| Hyperparameter | Value |
|---|---|
| Base model | mistralai/Mistral-7B-Instruct-v0.2 |
| Quantisation | 4-bit NF4, double quant, bfloat16 |
| LoRA rank / alpha | 64 / 128 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable params | ~83.9M / 7.24B (1.16%) |
| Effective batch size | 16 (4 per-device × 4 grad_accum) |
| Learning rate | 2e-4 (cosine schedule, 3% warmup) |
| Optimizer | paged_adamw_32bit |
| Max epochs | 3 (early stop patience=3) |
| Max seq length | 2048 (packed) |
| GPU | A100 40 GB (Colab) |

### Loss Curves

| Epoch | Train Loss | Val Loss | LR |
|---|---|---|---|
| 1 | *(fill after run)* | *(fill)* | *(fill)* |
| 2 | *(fill after run)* | *(fill)* | *(fill)* |
| 3 | *(fill after run)* | *(fill)* | *(fill)* |
| Early stop | — | Best val: *(fill)* | — |

### Gradient Norm Behavior
- Warm-up phase (first ~3% steps): grad norms rise from ~0 to peak
- Post-warmup: should stabilise below `max_grad_norm=0.3`
- Spikes above 0.5 indicate instability; lower LR if seen consistently

### W&B Metrics to Monitor
| Metric | Expected Trend | Alert If |
|---|---|---|
| `train/loss` | Steady decrease | Plateaus after epoch 1 |
| `eval/loss` | Tracks train loss | Diverges (overfitting) |
| `train/grad_norm` | Stabilises < 0.3 | Spikes > 1.0 repeatedly |
| `train/learning_rate` | Cosine decay from 2e-4 | Flat (scheduler broken) |

---

## DPO Run

| Hyperparameter | Value |
|---|---|
| Starting policy | SFT adapter checkpoint |
| Beta (KL weight) | 0.1 |
| Loss type | sigmoid (standard DPO) |
| Learning rate | 5e-5 (4× lower than SFT) |
| Effective batch size | 16 (2 per-device × 8 grad_accum) |
| Max epochs | 1 |
| Preference pairs | 500 (400 train / 100 val) |
| Max length | 1024 |

### DPO Metrics — W&B Dashboard

| Metric | Expected Trend | Alert If |
|---|---|---|
| `rewards/margins` | Positive, increasing | Stays ≤ 0 (model not learning) |
| `rewards/accuracies` | > 0.6 by end | Stays near 0.5 (random) |
| `eval/loss` | Decreasing | Increases after few steps |
| `logps/chosen` | Slightly increasing | Collapses to −∞ |
| `logps/rejected` | Slightly decreasing | Rises above chosen |

> **Tip**: If `rewards/margins` stays near 0, lower `beta` to 0.05 to allow more policy deviation from SFT.

---

## How to Reproduce

```bash
# Step 1: Run data pipeline
python -c "
import sys; sys.path.insert(0,'src')
from data_pipeline import run_part_a_pipeline, run_part_b_pipeline
records, stats = run_part_a_pipeline()
run_part_b_pipeline(records)
"

# Step 2: SFT
python src/train_qlora.py \
    --train_file data/processed/train.jsonl \
    --val_file   data/processed/val.jsonl \
    --output_dir checkpoints/sft_adapter \
    --wandb_project FinAlign-SFT

# Step 3: DPO
python src/train_dpo.py \
    --sft_adapter_path checkpoints/sft_adapter \
    --train_pref_file  data/preference_pairs/train_pref.jsonl \
    --val_pref_file    data/preference_pairs/val_pref.jsonl \
    --output_dir       checkpoints/dpo_adapter \
    --wandb_project    FinAlign-DPO

# Step 4: Export
python src/save_adapters.py export
```

## Checkpoint Locations

| Checkpoint | Path | Load with |
|---|---|---|
| SFT adapter | `checkpoints/sft_adapter/` | `PeftModel.from_pretrained(base, "checkpoints/sft_adapter")` |
| DPO adapter | `checkpoints/dpo_adapter/` | `PeftModel.from_pretrained(base, "checkpoints/dpo_adapter")` |
| Exported bundle | `exports/finalign_v1/` | See `exports/finalign_v1/README.md` |
