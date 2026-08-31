"""
src/train_qlora.py
==================
QLoRA Supervised Fine-Tuning (SFT) for FinAlign
Model  : Qwen/Qwen2.5-3B-Instruct
Method : 4-bit NF4 QLoRA + PEFT LoRA
GPU    : Colab T4 (16 GB) / A10G / A100
Tracker: Weights & Biases
Output : checkpoints/sft_adapter/

Usage:
    python src/train_qlora.py \
        --train_file data/processed/train.jsonl \
        --val_file   data/processed/val.jsonl \
        --output_dir checkpoints/sft_adapter \
        --wandb_project FinAlign-SFT
"""

import os, sys, json, argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, EarlyStoppingCallback, set_seed,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import wandb

# ── 0. Args ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_file",  default="data/processed/train.jsonl")
    p.add_argument("--val_file",    default="data/processed/val.jsonl")
    p.add_argument("--output_dir",  default="checkpoints/sft_adapter")
    p.add_argument("--model_name",  default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--max_seq_len", type=int,   default=1024)
    p.add_argument("--lora_r",      type=int,   default=32)
    p.add_argument("--lora_alpha",  type=int,   default=64)
    p.add_argument("--lora_dropout",type=float, default=0.05)
    p.add_argument("--num_epochs",  type=int,   default=3)
    p.add_argument("--batch",       type=int,   default=4)
    p.add_argument("--grad_accum",  type=int,   default=4)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--warmup",      type=float, default=0.03)
    p.add_argument("--patience",    type=int,   default=3)
    p.add_argument("--wandb_project", default="FinAlign-SFT")
    p.add_argument("--wandb_run",   default=None)
    p.add_argument("--no_wandb",    action="store_true")
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()

# ── 1. Configs ────────────────────────────────────────────────────────────────

def bnb_config():
    is_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if is_bf16 else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

def lora_config(args):
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none",
        inference_mode=False,
    )

# ── 2. Formatter ──────────────────────────────────────────────────────────────

TEMPLATE = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"

def fmt(ex):
    instr = ex.get("instruction","").strip()
    ctx   = ex.get("input","").strip()
    out   = ex.get("output", ex.get("response","")).strip()
    prompt = f"{instr}\n\n{ctx}" if ctx else instr
    return {"text": TEMPLATE.format(instruction=prompt, output=out)}

def get_attn_implementation():
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:  # Ampere, Ada, Hopper (A100, A10, RTX 3090/4090)
            try:
                import flash_attn  # noqa: F401
                return "flash_attention_2"
            except ImportError:
                pass
    return "sdpa"

# ── 3. Main ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run or f"sft-qwen3b-r{args.lora_r}-lr{args.lr}",
            config=vars(args),
            tags=["sft","qlora","qwen2.5-3b","personal-finance"],
        )

    # Data
    ds = load_dataset("json", data_files={"train": args.train_file, "validation": args.val_file}, split=None)
    ds = ds.map(fmt, remove_columns=ds["train"].column_names)
    print(f"[Data] train={len(ds['train'])} | val={len(ds['validation'])}")

    # Model
    is_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if is_bf16 else torch.float16

    attn_impl = get_attn_implementation()
    print(f"[Model] Using attention: {attn_impl} | Compute dtype: {compute_dtype}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, quantization_config=bnb_config(),
        device_map="auto", dtype=compute_dtype,
        attn_implementation=attn_impl,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, lora_config(args))
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Training args
    ta = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        learning_rate=args.lr,
        weight_decay=0.001,
        max_grad_norm=0.3,
        lr_scheduler_type="cosine",
        fp16=not is_bf16,
        bf16=is_bf16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none" if args.no_wandb else "wandb",
        run_name=args.wandb_run,
        seed=args.seed,
        logging_steps=10,
        dataset_text_field="text",
        max_length=args.max_seq_len,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        args=ta,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience, early_stopping_threshold=1e-4)],
    )

    print("[Train] Starting QLoRA SFT ...")
    trainer.train()

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "sft_meta.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"[Done] SFT adapter: {args.output_dir}")
    print(f"       Next: python src/train_dpo.py --sft_adapter_path {args.output_dir}")
    if not args.no_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
