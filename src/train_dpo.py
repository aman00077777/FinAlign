"""
src/train_dpo.py
================
DPO Alignment Training for FinAlign
Loads SFT adapter checkpoint as starting policy.
Training on 500 chosen/rejected preference pairs.

Usage:
    python src/train_dpo.py \
        --sft_adapter_path checkpoints/sft_adapter \
        --train_pref_file  data/preference_pairs/train_pref.jsonl \
        --val_pref_file    data/preference_pairs/val_pref.jsonl \
        --output_dir       checkpoints/dpo_adapter
"""

import os, json, argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from peft import PeftModel, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
import wandb

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_adapter_path", default="checkpoints/sft_adapter")
    p.add_argument("--train_pref_file",  default="data/preference_pairs/train_pref.jsonl")
    p.add_argument("--val_pref_file",    default="data/preference_pairs/val_pref.jsonl")
    p.add_argument("--output_dir",       default="checkpoints/dpo_adapter")
    p.add_argument("--model_name",       default="mistralai/Mistral-7B-Instruct-v0.2")
    p.add_argument("--beta",      type=float, default=0.1)
    p.add_argument("--loss_type",         default="sigmoid")
    p.add_argument("--num_epochs",type=int,   default=1)
    p.add_argument("--batch",     type=int,   default=2)
    p.add_argument("--grad_accum",type=int,   default=8)
    p.add_argument("--lr",        type=float, default=5e-5)
    p.add_argument("--max_length",       type=int, default=1024)
    p.add_argument("--max_prompt_length",type=int, default=512)
    p.add_argument("--wandb_project", default="FinAlign-DPO")
    p.add_argument("--wandb_run",     default=None)
    p.add_argument("--no_wandb",      action="store_true")
    p.add_argument("--seed",    type=int, default=42)
    return p.parse_args()

def bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

def get_attn_implementation():
    if torch.cuda.is_available():
        try:
            import flash_attn  # noqa: F401
            return "flash_attention_2"
        except ImportError:
            pass
    return "sdpa"

def main():
    args = parse_args()
    set_seed(args.seed)

    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run or f"dpo-beta{args.beta}-lr{args.lr}",
            config=vars(args),
            tags=["dpo","mistral-7b","personal-finance","alignment"],
        )

    # Data ? expects: prompt, chosen, rejected
    ds = load_dataset("json", data_files={
        "train":      args.train_pref_file,
        "validation": args.val_pref_file,
    }, split=None)
    print(f"[Data] train_pref={len(ds['train'])} | val_pref={len(ds['validation'])}")

    attn_impl = get_attn_implementation()
    print(f"[Model] Using attention implementation: {attn_impl}")

    # Policy model (trainable)
    bnb = bnb_config()
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name, quantization_config=bnb,
        device_map="auto", torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    policy = PeftModel.from_pretrained(base, args.sft_adapter_path, is_trainable=True)
    policy = prepare_model_for_kbit_training(policy)

    # Reference model (frozen SFT)
    base_ref = AutoModelForCausalLM.from_pretrained(
        args.model_name, quantization_config=bnb,
        device_map="auto", torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    ref = PeftModel.from_pretrained(base_ref, args.sft_adapter_path, is_trainable=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dpo_cfg = DPOConfig(
        output_dir=args.output_dir, overwrite_output_dir=True,
        beta=args.beta, loss_type=args.loss_type,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_32bit",
        learning_rate=args.lr, weight_decay=0.001, max_grad_norm=1.0,
        lr_scheduler_type="cosine", warmup_ratio=0.1,
        bf16=True, tf32=True, fp16=False,
        max_length=args.max_length, max_prompt_length=args.max_prompt_length,
        eval_strategy="epoch", save_strategy="epoch",
        save_total_limit=1, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        report_to="none" if args.no_wandb else "wandb",
        run_name=args.wandb_run, seed=args.seed,
        logging_steps=5, logging_first_step=True,
        remove_unused_columns=False,
    )

    # W&B tracks automatically: rewards/chosen, rewards/rejected,
    # rewards/margins, rewards/accuracies, logps/chosen, logps/rejected

    trainer = DPOTrainer(
        model=policy, ref_model=ref, args=dpo_cfg,
        train_dataset=ds["train"], eval_dataset=ds["validation"],
        tokenizer=tokenizer,
    )

    print("[Train] Starting DPO alignment ...")
    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "dpo_meta.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"[Done] DPO adapter: {args.output_dir}")
    print(f"       Export: python src/save_adapters.py export")
    if not args.no_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
