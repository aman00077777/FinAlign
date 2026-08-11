# FinAlign ? Data Statistics Report

## Part A ? Dataset Collection & Cleaning (Sharvari)

### Sources
| Source | Records Loaded |
|---|---|
| Akhil-Theerthala/PersonalFinance-Reddit-QA | 19,984 |
| ceadar-ie/FinTalk-19k | 19,111 |
| Local: finalign_cleaned_dataset.jsonl | 0 (merged) |
| **Total Raw** | **39,095** |

### Filtering Pipeline
| Step | Before | After | Removed | % Removed |
|---|---|---|---|---|
| Raw collection | ? | 39,095 | ? | ? |
| Length filter | 39,095 | 37,877 | 1,218 | 3.12% |
| Exact deduplication | 37,877 | 36,961 | 916 | 2.42% |
| Near-dedup (Jaccard >= 0.7) | 36,961 | 36,254 | 707 | 1.91% |
| PII scrubbing | ? | 36,254 | 5,181 instances | ? |
| Off-topic filter (< 2 finance kws) | 36,254 | 29,613 | 6,641 | 18.32% |
| **Final clean** | ? | **29,613** | **9,482 total** | **24.25%** |

### Length Filter Breakdown
| Reason | Count |
|---|---|
| Instruction < 20 chars | 412 |
| Response < 100 chars | 520 |
| Response > 8000 chars | 184 |
| Response < 20 words | 102 |

### PII Scrubbing Results
| Pattern | Replacements |
|---|---|
| Email -> [EMAIL] | 39 |
| Phone -> [PHONE] | 744 |
| URL -> [URL] | 4,334 |
| Reddit user -> [USER] | 29 |
| Account# -> [ACCOUNT] | 35 |
| **Total** | **5,181** |

> **Target: 5,000+ clean pairs ? PASSED (29,613 total)** ?

---

## Part B ? Quality Scoring, Splits & Preference Pairs (Samiksha)

### Quality Score Distribution (0-9 scale, n=29,613)
| Score Range | Tier | Action |
|---|---|---|
| 0-3 | Low quality | Filtered out of training |
| 4-6 | Medium quality | Included in training / chosen candidate |
| 7-9 | High quality | Priority chosen response |

Quality filter applied to **train split only** (threshold >= 4/9).

### Train / Val / Test Split (80 / 10 / 10)
| Split | Records | % |
|---|---|---|
| Train | 21,707 | 80% |
| Val | 2,957 | 10% |
| Test | 2,957 | 10% |
| **Total** | **29,613** | **100%** |

**Prompt Leakage Check**:
- Train-Val instruction overlap: 0 (PASS ?)
- Train-Test instruction overlap: 0 (PASS ?)

### Preference Pairs (DPO)
| Split | Pairs |
|---|---|
| Train pref (train_pref.jsonl) | 400 |
| Val pref (val_pref.jsonl) | 100 |
| **Total** | **500** |

### Preference Pair Topic Breakdown (500 pairs)
| Topic | Pairs |
|---|---|
| Budgeting | 46 |
| Credit | 46 |
| Debt | 46 |
| General | 45 |
| Income | 45 |
| Insurance | 45 |
| Investing | 46 |
| Mortgage | 45 |
| Retirement | 45 |
| Saving | 46 |
| Tax | 45 |
| **Total** | **500** |

### Preference Pair Construction Method
- **Chosen**: High/medium quality record (quality score >= 4/9): accurate, specific, safe advice with concrete numbers
- **Rejected**: Plausible but degraded version transformed via:
  1. Replacing percentages with vague placeholders ("some percentage")
  2. Removing named strategies (avalanche, snowball, 50/30/20)
  3. Appending risky guarantees ("This strategy is virtually guaranteed to work for everyone.")
  4. Truncating structure to 60% length
- Prompt format: `<s>[INST] {instruction} [/INST]`

### Eval Set (100 Questions)
- Carved out from held-out test split (zero overlap with train/val)
- 10 topics x 10 questions each
- Format: `{question, gold_answer, topic, gold_keywords}`
- File location: `data/eval_set/benchmark_100q.jsonl`
