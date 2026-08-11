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

BASE = "mistralai/Mistral-7B-Instruct-v0.2"

def export_cmd(args):
    os.makedirs(args.output_dir, exist_ok=True)
    for name, path in [("sft_adapter", args.sft_adapter), ("dpo_adapter", args.dpo_adapter)]:
        dest = os.path.join(args.output_dir, name)
        if os.path.exists(path):
            shutil.copytree(path, dest, dirs_exist_ok=True)
            print(f"Copied {name} -> {dest}")
        else:
            print(f"WARN: {path} not found, skipping")
    readme = """# FinAlign Adapter Bundle\n\n## Load DPO model (recommended)\n```python\nimport torch\nfrom transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig\nfrom peft import PeftModel\n\nBASE = "mistralai/Mistral-7B-Instruct-v0.2"\nbnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",\n    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)\nbase = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")\nmodel = PeftModel.from_pretrained(base, "dpo_adapter", is_trainable=False)\ntok = AutoTokenizer.from_pretrained(BASE)\n\ndef ask(q):\n    prompt = f"<s>[INST] {q} [/INST]"\n    inp = tok(prompt, return_tensors="pt").to(model.device)\n    with torch.no_grad():\n        out = model.generate(**inp, max_new_tokens=512, temperature=0.7, top_p=0.9, do_sample=True)\n    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)\n```\n"""
    with open(os.path.join(args.output_dir, "README.md"), "w") as f:
        f.write(readme)
    print(f"Bundle ready: {args.output_dir}")

def merge_cmd(args):
    os.makedirs(args.output_dir, exist_ok=True)
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16, device_map="auto")
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
