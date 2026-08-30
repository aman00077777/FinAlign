"""
app.py — FinAlign Personal Finance Advisor Gradio Demo
======================================================
Interactive Web Interface for FinAlign (Mistral-7B + QLoRA SFT + DPO Alignment).

Supports two backend modes:
1. Local Transformers / PEFT mode (loads model/adapter on GPU/CPU)
2. API Server mode (connects to running vLLM or Ollama endpoint)

Run locally:
    python app.py
"""

import os
import sys
import time
import argparse
import requests
import torch

try:
    import gradio as gr
except ImportError:
    print("[ERROR] Gradio is not installed. Run: pip install gradio")
    sys.exit(1)

# Default paths and API endpoints
DEFAULT_MODEL_PATH = "exports/finalign_v1_merged"
DEFAULT_ADAPTER_PATH = "checkpoints/dpo_adapter"
DEFAULT_BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_VLLM_URL = "http://localhost:8000/v1/chat/completions"

SYSTEM_PROMPT = (
    "You are FinAlign, an expert personal finance advisor. "
    "Provide clear, actionable, and safe financial guidance on budgeting, saving, investing, debt, credit, "
    "retirement, and tax planning. Avoid making unrealistic guarantees or high-risk speculative claims. "
    "Encourage consulting a certified professional for complex personal situations."
)

SAMPLE_QUESTIONS = [
    "How should I start building a 6-month emergency fund?",
    "What is the 50/30/20 budget rule and how do I apply it?",
    "Should I prioritize paying off high-interest debt or investing in a Roth IRA?",
    "What is the difference between index funds and individual stock picking?",
    "How do I improve my credit score from 650 to 750+ within a year?",
    "How much should I be saving for retirement in my 20s and 30s?",
]

# Global model container for local mode
LOCAL_MODEL = None
LOCAL_TOKENIZER = None

def load_local_model(model_path: str, adapter_path: str = None):
    global LOCAL_MODEL, LOCAL_TOKENIZER
    if LOCAL_MODEL is not None:
        return LOCAL_MODEL, LOCAL_TOKENIZER

    merged_exists = os.path.exists(model_path)
    adapter_exists = bool(adapter_path and os.path.exists(adapter_path))

    if not merged_exists and not adapter_exists:
        raise FileNotFoundError(
            f"Neither merged model ('{model_path}') nor adapter checkpoint ('{adapter_path}') was found.\n"
            f"To serve locally, first merge the trained adapters using:\n"
            f"  python src/save_adapters.py merge --dpo_adapter checkpoints/dpo_adapter --output_dir exports/finalign_v1_merged\n"
            f"Or select an API backend (Ollama API / vLLM API) under Advanced Settings."
        )

    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"[Loading] Local model from: {model_path if merged_exists else DEFAULT_BASE_MODEL}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_path if merged_exists else DEFAULT_BASE_MODEL,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if merged_exists:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )
    else:
        print(f"[Fallback] Base model + LoRA adapter ({adapter_path})")
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            DEFAULT_BASE_MODEL,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )
        model = PeftModel.from_pretrained(base, adapter_path)

    model.eval()
    LOCAL_MODEL = model
    LOCAL_TOKENIZER = tokenizer
    return model, tokenizer

def query_ollama(prompt: str, system: str, temperature: float, top_p: float, max_tokens: int, url: str):
    payload = {
        "model": "finalign",
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_tokens
        }
    }
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json().get("response", "")

def query_vllm(prompt: str, system: str, temperature: float, top_p: float, max_tokens: int, url: str):
    payload = {
        "model": "finalign-7b",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens
    }
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]

def generate_response(
    user_question: str,
    system_prompt: str,
    backend_mode: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    api_url: str
):
    if not user_question.strip():
        return "Please enter a personal finance question."

    if system_prompt and system_prompt.strip():
        formatted_prompt = f"<s>[INST] {system_prompt.strip()}\n\n{user_question.strip()} [/INST]"
    else:
        formatted_prompt = f"<s>[INST] {user_question.strip()} [/INST]"

    start_time = time.time()
    try:
        if backend_mode == "Ollama API":
            url = api_url if api_url.strip() else DEFAULT_OLLAMA_URL
            answer = query_ollama(user_question, system_prompt, temperature, top_p, max_tokens, url)

        elif backend_mode == "vLLM API":
            url = api_url if api_url.strip() else DEFAULT_VLLM_URL
            answer = query_vllm(user_question, system_prompt, temperature, top_p, max_tokens, url)

        else: # Local Transformers
            model, tokenizer = load_local_model(DEFAULT_MODEL_PATH, DEFAULT_ADAPTER_PATH)
            inputs = tokenizer(formatted_prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            input_len = inputs["input_ids"].shape[1]
            answer = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

        elapsed = round(time.time() - start_time, 2)
        return f"{answer.strip()}\n\n---\n*Generated in {elapsed}s via {backend_mode}*"

    except Exception as e:
        return f"⚠️ Error generating response: {str(e)}\n\nCheck backend service or model checkpoints."

# Build Gradio UI
def build_ui():
    custom_css = """
    .main-title { text-align: center; color: #1e3a8a; font-size: 2.2rem; font-weight: 700; margin-bottom: 0.2rem; }
    .sub-title { text-align: center; color: #475569; font-size: 1.1rem; margin-bottom: 1.5rem; }
    .finance-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; }
    """

    with gr.Blocks(title="FinAlign — Personal Finance AI Advisor", css=custom_css) as demo:
        gr.Markdown(
            """
            # 💰 FinAlign — Personal Finance AI Advisor
            ### Fine-Tuned Mistral-7B with QLoRA & DPO Preference Alignment
            """,
            elem_classes=["main-title"]
        )

        with gr.Row():
            with gr.Column(scale=2):
                question_input = gr.Textbox(
                    label="Ask a Personal Finance Question",
                    placeholder="e.g. How should I set up my emergency fund and where should I keep it?",
                    lines=3
                )

                with gr.Row():
                    submit_btn = gr.Button("🚀 Ask FinAlign", variant="primary")
                    clear_btn = gr.Button("🧹 Clear")

                gr.Markdown("### 💡 Sample Questions")
                gr.Examples(
                    examples=SAMPLE_QUESTIONS,
                    inputs=question_input
                )

            with gr.Column(scale=3):
                response_output = gr.Markdown(
                    label="FinAlign Advice",
                    value="*Your financial guidance will appear here...*"
                )

        with gr.Accordion("⚙️ Advanced Generation & Backend Settings", open=False):
            with gr.Row():
                backend_dropdown = gr.Dropdown(
                    choices=["Local Transformers / PEFT", "Ollama API", "vLLM API"],
                    value="Local Transformers / PEFT",
                    label="Backend Mode"
                )
                api_url_input = gr.Textbox(
                    label="Custom API Endpoint URL (optional)",
                    placeholder="http://localhost:11434/api/generate or http://localhost:8000/v1/chat/completions"
                )

            with gr.Row():
                temp_slider = gr.Slider(0.1, 1.0, value=0.7, step=0.05, label="Temperature")
                topp_slider = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-P")
                tokens_slider = gr.Slider(64, 1024, value=512, step=32, label="Max New Tokens")

            sys_prompt_input = gr.Textbox(
                label="System Prompt",
                value=SYSTEM_PROMPT,
                lines=2
            )

        submit_btn.click(
            fn=generate_response,
            inputs=[question_input, sys_prompt_input, backend_dropdown, temp_slider, topp_slider, tokens_slider, api_url_input],
            outputs=response_output
        )
        clear_btn.click(lambda: ("", "*Your financial guidance will appear here...*"), outputs=[question_input, response_output])

    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public shareable link")
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)
