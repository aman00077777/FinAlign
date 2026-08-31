"""
src/save_adapters.py  ? Export & merge LoRA adapters
Usage:
  python src/save_adapters.py export --sft_adapter checkpoints/sft_adapter --dpo_adapter checkpoints/dpo_adapter --output_dir exports/finalign_v1
  python src/save_adapters.py merge  --dpo_adapter checkpoints/dpo_adapter --output_dir exports/finalign_v1_merged
"""
import os, json, shutil, argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "Qwen/Qwen2.5-3B-Instruct"

def export_cmd(args):
    os.makedirs(args.output_dir, exist_ok=True)
    for name, path in [("sft_adapter", args.sft_adapter), ("dpo_adapter", args.dpo_adapter)]:
        dest = os.path.join(args.output_dir, name)
        if os.path.exists(path):
            shutil.copytree(path, dest, dirs_exist_ok=True)
            print(f"Copied {name} -> {dest}")
        else:
            print(f"WARN: {path} not found, skipping")
    readme = """# FinAlign Adapter Bundle

## Load DPO model (recommended)
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE = "Qwen/Qwen2.5-3B-Instruct"
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(base, "dpo_adapter", is_trainable=False)
tok = AutoTokenizer.from_pretrained(BASE)

def ask(q):
    messages = [{"role": "user", "content": q}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=512, temperature=0.7, top_p=0.9, do_sample=True)
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
```
"""
    with open(os.path.join(args.output_dir, "README.md"), "w") as f:
        f.write(readme)
    print(f"Bundle ready: {args.output_dir}")

def merge_cmd(args):
    os.makedirs(args.output_dir, exist_ok=True)
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, args.dpo_adapter)
    model = model.merge_and_unload()
    model.save_pretrained(args.output_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(BASE).save_pretrained(args.output_dir)
    print(f"Merged model: {args.output_dir}")

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--sft_adapter", default="checkpoints/sft_adapter")
    e.add_argument("--dpo_adapter", default="checkpoints/dpo_adapter")
    e.add_argument("--output_dir",  default="exports/finalign_v1")
    m = sub.add_parser("merge")
    m.add_argument("--dpo_adapter", default="checkpoints/dpo_adapter")
    m.add_argument("--output_dir",  default="exports/finalign_v1_merged")
    args = p.parse_args()
    if args.cmd == "export": export_cmd(args)
    elif args.cmd == "merge": merge_cmd(args)

if __name__ == "__main__":
    main()
