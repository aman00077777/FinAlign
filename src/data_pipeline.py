"""
src/data_pipeline.py
====================
FinAlign Data Pipeline -> Part A (Sharvari) + Part B (Samiksha)

Part A:
  - load_huggingface_data()     : Pull raw data from HuggingFace
  - length_filter()             : Drop too-short / too-long responses
  - deduplicate()               : Exact + near-duplicate removal (MinHash)
  - scrub_pii()                 : Remove emails, phone#, SSN, account#, names
  - standardize_format()        : Convert to {instruction, input, output}

Part B:
  - quality_score()             : Heuristic rubric (0-9 score)
  - stratified_split()          : 80/10/10 split with topic balance, no leakage
  - assign_topic()              : Topic labelling for stratification
  - generate_preference_pairs() : 500 chosen/rejected pairs for DPO
  - generate_eval_set()         : 100 held-out questions
"""

import re
import json
import hashlib
import random
import unicodedata
from pathlib import Path
from typing import Optional
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def _resolve_path(path_obj: str | Path) -> Path:
    p = Path(path_obj)
    return p if p.is_absolute() else PROJECT_ROOT / p

# PART A -> Sharvari

# A1. HuggingFace Data Loading 

def load_huggingface_data(cache_dir: str = "data/raw") -> list[dict]:
    """
    Pull raw instruction-response data from:
      - Akhil-Theerthala/PersonalFinance-Reddit-QA
      - ceadar-ie/FinTalk-19k
    Returns list of dicts with keys: instruction, response, source
    Falls back to local finalign_cleaned_dataset.jsonl if HF unavailable.
    """
    from datasets import load_dataset

    cache_dir = str(_resolve_path(cache_dir))
    records = []

    # Source 1: PersonalFinance-Reddit-QA ???????????????????????????????
    print("[Load] Akhil-Theerthala/PersonalFinance-Reddit-QA ...")
    try:
        ds1 = load_dataset(
            "Akhil-Theerthala/PersonalFinance-Reddit-QA",
            split="train",
            cache_dir=cache_dir,
        )
        print(f"       Columns: {ds1.column_names}")
        # Actual columns: category, subreddit, query, answer
        for row in ds1:
            instruction = ""
            response = ""
            # Reddit-QA specific columns first, then generic fallbacks
            for ic in ["query", "title", "question", "instruction", "selftext", "prompt"]:
                if ic in row and row[ic]:
                    instruction = str(row[ic]).strip()
                    break
            for rc in ["answer", "top_answer", "response", "body", "output"]:
                if rc in row and row[rc]:
                    response = str(row[rc]).strip()
                    break
            if instruction and response:
                records.append({
                    "instruction": instruction,
                    "response": response,
                    "source": "reddit_qa",
                })
        print(f"       Loaded: {sum(1 for r in records if r['source']=='reddit_qa')} records")
    except Exception as e:
        print(f"       [WARN] Reddit-QA failed: {e}")

    # Source 2: FinTalk-19k ?????????????????????????????????????????????
    print("[Load] ceadar-ie/FinTalk-19k ...")
    try:
        ds2 = load_dataset(
            "ceadar-ie/FinTalk-19k",
            split="train",
            cache_dir=cache_dir,
        )
        print(f"       Columns: {ds2.column_names}")
        before = len(records)
        for row in ds2:
            instruction = ""
            response = ""
            for ic in ["question", "instruction", "prompt", "input", "human"]:
                if ic in row and row[ic]:
                    instruction = str(row[ic]).strip()
                    break
            for rc in ["answer", "response", "output", "gpt", "assistant"]:
                if rc in row and row[rc]:
                    response = str(row[rc]).strip()
                    break
            if instruction and response:
                records.append({
                    "instruction": instruction,
                    "response": response,
                    "source": "fintalk",
                })
        print(f"       Loaded: {len(records)-before} records")
    except Exception as e:
        print(f"       [WARN] FinTalk-19k failed: {e}")

    # Merge local dataset to ensure 5,000+ training pairs ─────────────────
    # Always merge local data on top of HF data for richer coverage
    local_path = _resolve_path("data/finalign_cleaned_dataset.jsonl")
    if local_path.exists():
        print(f"[Load] Merging local dataset: {local_path}")
        existing_instrs = {r["instruction"].strip().lower() for r in records}
        count = 0
        with open(local_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    instr = obj.get("instruction", "").strip()
                    resp  = obj.get("response", obj.get("output", "")).strip()
                    # Only add if not already in HF data (avoid duplicates)
                    if instr and resp and instr.lower() not in existing_instrs:
                        records.append({
                            "instruction": instr,
                            "response": resp,
                            "source": "local",
                        })
                        existing_instrs.add(instr.lower())
                        count += 1
                except Exception:
                    pass
        print(f"       Merged {count} additional records from local file")
    else:
        print(f"[Load] Local file not found: {local_path}")

    print(f"[Load] Total raw records: {len(records)}")
    return records


# A2. Length Filter ?????????????????????????????????????????????????????????

def length_filter(
    records: list[dict],
    min_instruction_chars: int = 20,
    max_instruction_chars: int = 2000,
    min_response_chars: int = 100,
    max_response_chars: int = 8000,
    min_response_words: int = 20,
) -> tuple[list[dict], dict]:
    """
    Drop records with:
      - Instruction too short (degenerate) or too long
      - Response too short (uninformative) or too long (rambling)
    Returns (filtered_records, stats_dict)
    """
    kept = []
    reasons = Counter()

    for r in records:
        instr = r.get("instruction", "")
        resp  = r.get("response", "")
        words = len(resp.split())

        if len(instr) < min_instruction_chars:
            reasons["instruction_too_short"] += 1
            continue
        if len(instr) > max_instruction_chars:
            reasons["instruction_too_long"] += 1
            continue
        if len(resp) < min_response_chars:
            reasons["response_too_short"] += 1
            continue
        if len(resp) > max_response_chars:
            reasons["response_too_long"] += 1
            continue
        if words < min_response_words:
            reasons["response_too_few_words"] += 1
            continue

        kept.append(r)

    stats = {
        "before": len(records),
        "after": len(kept),
        "removed": len(records) - len(kept),
        "pct_removed": round(100 * (len(records) - len(kept)) / max(1, len(records)), 2),
        "breakdown": dict(reasons),
    }
    print(f"[Filter] {stats['before']} -> {stats['after']} "
          f"(removed {stats['removed']}, {stats['pct_removed']}%)")
    return kept, stats


# A3. Deduplication (Exact + Near-duplicate) ????????????????????????????????

def _shingle(text: str, k: int = 5) -> set[str]:
    """Character k-shingles for Jaccard similarity."""
    text = text.lower().strip()
    return {text[i:i+k] for i in range(len(text) - k + 1)} if len(text) >= k else set()


def deduplicate(
    records: list[dict],
    jaccard_threshold: float = 0.7,
    field: str = "instruction",
) -> tuple[list[dict], dict]:
    """
    Two-pass deduplication:
      Pass 1 -> Exact match (MD5 hash)
      Pass 2 -> Near-duplicate via LSH-lite bucket + Jaccard
    Returns (deduped_records, stats_dict)
    """
    # Pass 1: Exact
    seen_hashes: set[str] = set()
    after_exact = []
    exact_removed = 0
    for r in records:
        h = hashlib.md5(r.get(field, "").strip().lower().encode()).hexdigest()
        if h in seen_hashes:
            exact_removed += 1
        else:
            seen_hashes.add(h)
            after_exact.append(r)

    print(f"[Dedup] Exact: removed {exact_removed} -> {len(after_exact)} remain")

    # Pass 2: Near-duplicate via shingle buckets (LSH-lite)
    # Group by first 3-gram of instruction tokens as bucket key
    def bucket_key(text: str) -> str:
        tokens = text.lower().split()
        return " ".join(tokens[:3]) if len(tokens) >= 3 else text[:15].lower()

    buckets: dict[str, list[int]] = defaultdict(list)
    for idx, r in enumerate(after_exact):
        key = bucket_key(r.get(field, ""))
        buckets[key].append(idx)

    near_dup_indices: set[int] = set()
    for key, indices in buckets.items():
        if len(indices) < 2:
            continue
        shingles = [_shingle(after_exact[i].get(field, "")) for i in indices]
        for a in range(len(indices)):
            if indices[a] in near_dup_indices:
                continue
            for b in range(a + 1, len(indices)):
                if indices[b] in near_dup_indices:
                    continue
                sa, sb = shingles[a], shingles[b]
                union = len(sa | sb)
                if union == 0:
                    continue
                jaccard = len(sa & sb) / union
                if jaccard >= jaccard_threshold:
                    near_dup_indices.add(indices[b])  # Remove the later one

    after_near = [r for i, r in enumerate(after_exact) if i not in near_dup_indices]
    near_removed = len(after_exact) - len(after_near)
    print(f"[Dedup] Near-dup (J>={jaccard_threshold}): removed {near_removed} -> {len(after_near)} remain")

    stats = {
        "before": len(records),
        "exact_removed": exact_removed,
        "near_dup_removed": near_removed,
        "after": len(after_near),
        "pct_removed": round(100 * (len(records) - len(after_near)) / max(1, len(records)), 2),
    }
    return after_near, stats


# A4. PII Scrubber ??????????????????????????????????????????????????????????

_PII_PATTERNS = [
    # Email addresses
    (re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'), "[EMAIL]"),
    # US phone numbers
    (re.compile(r'\b(\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'), "[PHONE]"),
    # SSN
    (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), "[SSN]"),
    # Credit card (basic 4x4 pattern)
    (re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'), "[CARD]"),
    # Long digit strings (account/routing numbers 9-12 digits)
    (re.compile(r'\b\d{9,12}\b'), "[ACCOUNT]"),
    # Reddit usernames  u/username
    (re.compile(r'\bu/[A-Za-z0-9_\-]{3,20}\b'), "[USER]"),
    # URLs (keep domain concept but strip personal paths)
    (re.compile(r'https?://(?!www\.annualcreditreport|www\.irs|www\.sec|investopedia|nerdwallet|bogleheads)[^\s]+'), "[URL]"),
]

def scrub_pii(records: list[dict]) -> tuple[list[dict], dict]:
    """
    Replace PII patterns with placeholders in both instruction and response.
    Returns (scrubbed_records, stats_dict)
    """
    stats = Counter()
    scrubbed = []

    for r in records:
        new_r = dict(r)
        for field in ("instruction", "response"):
            text = new_r.get(field, "")
            for pattern, placeholder in _PII_PATTERNS:
                matches = pattern.findall(text)
                if matches:
                    stats[placeholder] += len(matches)
                text = pattern.sub(placeholder, text)
            new_r[field] = text
        scrubbed.append(new_r)

    total_replacements = sum(stats.values())
    print(f"[PII]   Scrubbed {total_replacements} PII instances across {len(scrubbed)} records")
    print(f"        Breakdown: {dict(stats)}")
    return scrubbed, {"total_replacements": total_replacements, "by_type": dict(stats)}


# A5. Format Standardizer ???????????????????????????????????????????????????

# Finance domain keywords for on-topic check
_FINANCE_KWS = {
    "budget","saving","invest","loan","debt","credit","mortgage","retire",
    "bank","401k","ira","tax","income","expense","fund","stock","bond",
    "insurance","money","financial","interest","payment","salary","portfolio",
    "etf","dividend","compound","inflation","emergency","roth","hsa","asset",
    "liability","net worth","cash flow","yield","apr","apy","deductible",
}

def _is_finance_relevant(text: str, min_hits: int = 2) -> bool:
    text_lower = text.lower()
    hits = sum(1 for kw in _FINANCE_KWS if kw in text_lower)
    return hits >= min_hits

def standardize_format(records: list[dict]) -> tuple[list[dict], dict]:
    """
    Convert every record to {instruction, input, output} (Alpaca format).
    Also drops records with no finance relevance.
    instruction = the question/prompt
    input       = "" (no additional context for most pairs)
    output      = the answer/response
    """
    standardized = []
    off_topic = 0

    for r in records:
        instr = r.get("instruction", "").strip()
        resp  = r.get("response", r.get("output", "")).strip()

        combined = instr + " " + resp
        if not _is_finance_relevant(combined):
            off_topic += 1
            continue

        standardized.append({
            "instruction": instr,
            "input": "",
            "output": resp,
            "source": r.get("source", "unknown"),
        })

    stats = {
        "before": len(records),
        "off_topic_removed": off_topic,
        "after": len(standardized),
        "pct_removed": round(100 * off_topic / max(1, len(records)), 2),
    }
    print(f"[Format] Off-topic removed: {off_topic} | Final: {len(standardized)}")
    return standardized, stats


# A: Full Pipeline Runner ???????????????????????????????????????????????????

def run_part_a_pipeline(
    output_path: str = "data/processed/finance_sft_clean.jsonl",
    raw_output_path: str = "data/raw/raw_combined.jsonl",
    stats_path: str = "data/processed/pipeline_stats.json",
) -> tuple[list[dict], dict]:
    """
    Run the complete Part A pipeline:
    Load -> Length filter -> Deduplicate -> PII scrub -> Format -> Save
    Returns (final_records, all_stats)
    """
    output_path = _resolve_path(output_path)
    raw_output_path = _resolve_path(raw_output_path)
    stats_path = _resolve_path(stats_path)
    all_stats = {}

    # 1. Load
    raw = load_huggingface_data()
    all_stats["raw_count"] = len(raw)

    # Save raw
    Path(raw_output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(raw_output_path, "w", encoding="utf-8") as f:
        for r in raw:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 2. Length filter
    filtered, fstats = length_filter(raw)
    all_stats["length_filter"] = fstats

    # 3. Deduplicate
    deduped, dstats = deduplicate(filtered)
    all_stats["dedup"] = dstats

    # 4. PII scrub
    scrubbed, pstats = scrub_pii(deduped)
    all_stats["pii_scrub"] = pstats

    # 5. Standardize format + topic filter
    final, rstats = standardize_format(scrubbed)
    all_stats["format"] = rstats
    all_stats["final_count"] = len(final)

    # Save processed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Save stats
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)

    print(f"\n[Part A Done] {all_stats['raw_count']} raw -> {len(final)} clean records")
    print(f"              Saved: {output_path}")
    return final, all_stats


# ------------------------------------------------------------
# PART B -> Samiksha
# ------------------------------------------------------------

# Topic taxonomy ????????????????????????????????????????????????????????????

TOPIC_KEYWORDS = {
    "budgeting":    ["budget","spending","expense","track","50/30/20","zero-based","frugal","monthly","category","overspend"],
    "saving":       ["saving","emergency fund","hysa","high-yield","cd","certificate","liquid","rainy day","nest egg","interest rate"],
    "investing":    ["invest","stock","etf","index fund","portfolio","brokerage","vanguard","fidelity","s&p","compound","dividend","mutual fund"],
    "debt":         ["debt","loan","credit card","pay off","avalanche","snowball","interest","minimum payment","balance transfer","refinanc"],
    "credit":       ["credit score","credit report","fico","experian","equifax","transunion","utilization","hard inquiry","dispute","freeze"],
    "mortgage":     ["mortgage","home","house","down payment","refinance","rent","landlord","hoa","property tax","escrow"],
    "retirement":   ["retire","401k","roth","ira","pension","social security","annuity","required minimum","early withdrawal","contribution limit"],
    "tax":          ["tax","irs","deduction","w-2","1099","filing","refund","withholding","bracket","write-off","capital gains"],
    "insurance":    ["insurance","premium","deductible","life insurance","health","disability","coverage","policy","claim","beneficiary"],
    "income":       ["salary","income","raise","negotiat","side hustle","freelance","gig","bonus","overtime","passive income"],
}

def assign_topic(record: dict) -> str:
    """Assign the best-matching personal finance topic."""
    text = (record.get("instruction","") + " " + record.get("output","")).lower()
    scores = {}
    for topic, kws in TOPIC_KEYWORDS.items():
        scores[topic] = sum(1 for kw in kws if kw in text)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


# B1. Quality Scoring ???????????????????????????????????????????????????????

_RISKY_PHRASES = [
    "guaranteed return","you will definitely","risk-free investment","cannot lose",
    "100% safe","always make money","get rich quick","double your money",
    "i guarantee","you will make","no risk","sure profit",
]
_GOOD_SIGNALS = [
    "emergency fund","diversif","consult","professional","depending on","consider",
    "generally","typically","it depends","your situation","individual","specific",
]

def quality_score(record: dict) -> dict:
    """
    Heuristic quality rubric -> scores each record 0-9:
      Relevance    (0-3): domain keyword density
      Specificity  (0-3): contains numbers, percentages, named strategies
      Safety       (0-3): avoids risky guarantees; encourages professional advice
    Returns dict with total score and sub-scores.
    """
    text = (record.get("instruction","") + " " + record.get("output","")).lower()
    resp = record.get("output","").lower()
    resp_words = len(record.get("output","").split())

    # Relevance: finance keyword hits
    kw_hits = sum(1 for kw in _FINANCE_KWS if kw in text)
    relevance = min(3, kw_hits // 3)

    # Specificity: numbers, %, named products
    has_numbers  = bool(re.search(r'\d+\.?\d*\s*%', resp))
    has_amount   = bool(re.search(r'\$[\d,]+', resp))
    has_strategy = any(w in resp for w in ["avalanche","snowball","50/30/20","three-fund",
                                            "roth ladder","dollar-cost","index fund"])
    good_hits    = sum(1 for s in _GOOD_SIGNALS if s in resp)
    specificity  = min(3, int(has_numbers) + int(has_amount) + int(has_strategy) + min(1, good_hits // 2))

    # Safety: no risky guarantees
    risky_hits = sum(1 for p in _RISKY_PHRASES if p in resp)
    safety = max(0, 3 - risky_hits * 2)
    # Penalise very short responses
    if resp_words < 30:
        safety = max(0, safety - 1)

    total = relevance + specificity + safety
    return {
        "relevance": relevance,
        "specificity": specificity,
        "safety": safety,
        "total": total,
    }


# B2. Stratified Train/Val/Test Split ???????????????????????????????????????

def stratified_split(
    records: list[dict],
    train_ratio: float = 0.80,
    val_ratio:   float = 0.10,
    test_ratio:  float = 0.10,
    seed: int = 42,
    quality_threshold: int = 4,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """
    1. Score all records for quality; filter low-quality (< threshold) from train
    2. Assign topics
    3. Stratified shuffle-split by topic to maintain topic balance across splits
    4. Ensure no prompt leakage (exact instruction match check)
    Returns (train, val, test, stats)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    random.seed(seed)

    # Score and assign topics
    scored = []
    for r in records:
        qs = quality_score(r)
        topic = assign_topic(r)
        scored.append({**r, "_quality": qs, "_topic": topic})

    # Group by topic
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for r in scored:
        by_topic[r["_topic"]].append(r)

    train_list, val_list, test_list = [], [], []
    topic_stats = {}

    for topic, items in by_topic.items():
        random.shuffle(items)
        n = len(items)
        n_val  = max(1, int(n * val_ratio))
        n_test = max(1, int(n * test_ratio))
        n_train = n - n_val - n_test

        # Quality filter on train only (val/test keep all for fair evaluation)
        train_candidates = items[:n_train]
        high_quality_train = [r for r in train_candidates if r["_quality"]["total"] >= quality_threshold]
        # If too few pass quality filter, relax threshold
        if len(high_quality_train) < max(10, int(n_train * 0.5)):
            high_quality_train = train_candidates

        train_list.extend(high_quality_train)
        val_list.extend(items[n_train:n_train + n_val])
        test_list.extend(items[n_train + n_val:])

        topic_stats[topic] = {
            "total": n,
            "train": len(high_quality_train),
            "val": n_val,
            "test": n_test,
        }

    # Shuffle final splits
    random.shuffle(train_list)
    random.shuffle(val_list)
    random.shuffle(test_list)

    # Leakage check: ensure val/test instructions not in train
    train_instrs = {r["instruction"].strip().lower() for r in train_list}
    val_list  = [r for r in val_list  if r["instruction"].strip().lower() not in train_instrs]
    test_list = [r for r in test_list if r["instruction"].strip().lower() not in train_instrs]

    stats = {
        "total_input": len(records),
        "train": len(train_list),
        "val": len(val_list),
        "test": len(test_list),
        "topic_breakdown": topic_stats,
        "quality_threshold": quality_threshold,
    }
    print(f"[Split] Train={len(train_list)} | Val={len(val_list)} | Test={len(test_list)}")
    print(f"        No leakage guaranteed. Topics: {list(topic_stats.keys())}")
    return train_list, val_list, test_list, stats


# B3. Preference Pair Generation ???????????????????????????????????????????

REJECT_TRANSFORMS = [
    # Strategy A: Replace specific numbers with vague placeholders
    lambda t: re.sub(r'\d+\.?\d*\s*%', 'some percentage', t),
    # Strategy B: Remove concrete named strategies
    lambda t: t.replace("avalanche method", "paying off debt").replace(
               "snowball method", "paying off debt").replace(
               "three-fund portfolio", "diversified portfolio"),
    # Strategy C: Add risky guarantees
    lambda t: t + " This strategy is virtually guaranteed to work for everyone.",
    # Strategy D: Oversimplify (drop all bullet points/structure)
    lambda t: re.sub(r'\n+[-*?]\s+', ' ', t),
    # Strategy E: Add hedge that undermines advice
    lambda t: "It's hard to say for sure, but " + t[:400] + " though results vary widely.",
]

def _make_rejected(chosen_response: str, rng: random.Random) -> str:
    """Apply 1-2 transforms to degrade a chosen response into a plausible-but-weaker rejected."""
    result = chosen_response
    transforms = rng.sample(REJECT_TRANSFORMS, k=rng.randint(1, 2))
    for transform in transforms:
        result = transform(result)
    # Truncate to ~60% of original length to simulate incomplete answer
    words = result.split()
    if len(words) > 60:
        result = " ".join(words[:int(len(words) * 0.6)]) + "..."
    return result.strip()


def generate_preference_pairs(
    train_records: list[dict],
    n_pairs: int = 500,
    seed: int = 42,
    topic_balance: bool = True,
) -> list[dict]:
    """
    Generate n_pairs chosen/rejected preference pairs for DPO.
    - chosen: high-quality record (quality -> 6/9)
    - rejected: transformed/degraded version of the same or lower-quality response
    - Balanced across topics

    Format per pair:
    {
      "prompt":   "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n",
      "chosen":   "{high_quality_response}<|im_end|>",
      "rejected": "{degraded_response}<|im_end|>",
      "topic":    "{topic}"
    }
    """
    rng = random.Random(seed)

    # Get high-quality records per topic
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for r in train_records:
        topic = r.get("_topic", assign_topic(r))
        score = r.get("_quality", quality_score(r))
        if score["total"] >= 4:  # High/medium quality ones as chosen
            by_topic[topic].append(r)

    # Fallback to any record in topic if pool is smaller than quota needed
    all_by_topic: dict[str, list[dict]] = defaultdict(list)
    for r in train_records:
        all_by_topic[r.get("_topic", assign_topic(r))].append(r)

    # Calculate per-topic quota
    topics = list(TOPIC_KEYWORDS.keys()) + ["general"]
    per_topic = n_pairs // len(topics)
    leftover  = n_pairs - per_topic * len(topics)

    pairs = []
    for i, topic in enumerate(topics):
        quota = per_topic + (1 if i < leftover else 0)
        pool  = by_topic.get(topic, [])
        if len(pool) < quota:
            pool = all_by_topic.get(topic, [])
        rng.shuffle(pool)
        selected = pool[:quota]

        for record in selected:
            instr   = record["instruction"].strip()
            chosen  = record["output"].strip()
            rejected = _make_rejected(chosen, rng)

            # Ensure chosen != rejected and rejected is not empty
            if not rejected or rejected == chosen:
                rejected = "I'm not sure about the specifics, but generally speaking, " + chosen[:200] + "..."

            pairs.append({
                "prompt":   f"<|im_start|>user\n{instr}<|im_end|>\n<|im_start|>assistant\n",
                "chosen":   f"{chosen}<|im_end|>",
                "rejected": f"{rejected}<|im_end|>",
                "topic":    topic,
            })

    # If total pairs < n_pairs due to small topic pools, fill deficit from remaining train records
    if len(pairs) < n_pairs:
        used_instrs = {p["prompt"] for p in pairs}
        remaining_pool = [r for r in train_records if f"<|im_start|>user\n{r['instruction'].strip()}<|im_end|>\n<|im_start|>assistant\n" not in used_instrs]
        rng.shuffle(remaining_pool)
        needed = n_pairs - len(pairs)
        for record in remaining_pool[:needed]:
            instr = record["instruction"].strip()
            chosen = record["output"].strip()
            rejected = _make_rejected(chosen, rng)
            if not rejected or rejected == chosen:
                rejected = "I'm not sure about the specifics, but generally speaking, " + chosen[:200] + "..."
            pairs.append({
                "prompt": f"<|im_start|>user\n{instr}<|im_end|>\n<|im_start|>assistant\n",
                "chosen": f"{chosen}<|im_end|>",
                "rejected": f"{rejected}<|im_end|>",
                "topic": record.get("_topic", assign_topic(record)),
            })

    rng.shuffle(pairs)
    print(f"[Pref]  Generated {len(pairs)} preference pairs across {len(topics)} topics")
    topic_counts = Counter(p["topic"] for p in pairs)
    for t, c in sorted(topic_counts.items()):
        print(f"        {t}: {c}")
    return pairs


# B4. Eval Set Generation ???????????????????????????????????????????????????

def generate_eval_set(
    test_records: list[dict],
    n_questions: int = 100,
    seed: int = 42,
) -> list[dict]:
    """
    Carve out 100 held-out evaluation questions from the test split.
    Format: {question, gold_answer, topic, gold_keywords}
    """
    rng = random.Random(seed)
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for r in test_records:
        topic = r.get("_topic", assign_topic(r))
        by_topic[topic].append(r)

    topics = list(by_topic.keys())
    per_topic = n_questions // max(1, len(topics))
    leftover  = n_questions - per_topic * len(topics)

    eval_items = []
    for i, topic in enumerate(topics):
        quota = per_topic + (1 if i < leftover else 0)
        pool  = by_topic[topic]
        rng.shuffle(pool)
        for r in pool[:quota]:
            # Extract gold keywords from the answer
            resp_words = r.get("output","").lower().split()
            gold_kws = list({w for w in resp_words
                             if w in _FINANCE_KWS and len(w) > 4})[:6]
            eval_items.append({
                "question":      r["instruction"],
                "gold_answer":   r["output"],
                "topic":         topic,
                "gold_keywords": gold_kws,
            })

    rng.shuffle(eval_items)
    eval_items = eval_items[:n_questions]
    print(f"[Eval]  Carved out {len(eval_items)} eval questions")
    return eval_items


# B: Full Pipeline Runner ???????????????????????????????????????????????????

def run_part_b_pipeline(
    clean_records: list[dict],
    output_dir: str = "data",
    seed: int = 42,
) -> dict:
    """
    Run the complete Part B pipeline:
    Score -> Split -> Preference pairs -> Eval set -> Save all
    """
    output_dir = _resolve_path(output_dir)
    (output_dir / "preference_pairs").mkdir(parents=True, exist_ok=True)
    (output_dir / "eval_set").mkdir(parents=True, exist_ok=True)
    (output_dir / "processed").mkdir(parents=True, exist_ok=True)

    # 1. Split
    train, val, test, split_stats = stratified_split(clean_records, seed=seed)

    def _strip_meta(records):
        return [{k: v for k, v in r.items() if not k.startswith("_")} for r in records]

    # Save splits
    for name, records in [("train", train), ("val", val), ("test", test)]:
        path = f"{output_dir}/processed/{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in _strip_meta(records):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[Save]  {path} ({len(records)} records)")

    # 2. Preference pairs
    all_pairs = generate_preference_pairs(train, n_pairs=500, seed=seed)
    n_train_pref = int(len(all_pairs) * 0.8)
    train_pref = all_pairs[:n_train_pref]
    val_pref   = all_pairs[n_train_pref:]

    for name, pairs in [("train_pref", train_pref), ("val_pref", val_pref)]:
        path = f"{output_dir}/preference_pairs/{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for p in pairs:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"[Save]  {path} ({len(pairs)} pairs)")

    # 3. Eval set
    eval_items = generate_eval_set(test, n_questions=100, seed=seed)
    eval_path = f"{output_dir}/eval_set/benchmark_100q.jsonl"
    with open(eval_path, "w", encoding="utf-8") as f:
        for item in eval_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[Save]  {eval_path} ({len(eval_items)} questions)")

    return {
        "split_stats":     split_stats,
        "train_pref_count": len(train_pref),
        "val_pref_count":   len(val_pref),
        "eval_count":       len(eval_items),
    }
