#!/usr/bin/env python3
"""
Qwen2.5-1.5B QLoRA Fine-tuning — chạy trên Kaggle T4.
Trigger: thay đổi config/qwen_params.yaml → GitHub Actions → llm_kaggle_trigger.py
"""

import math
import os
import subprocess
import sys

# ── Cài dependencies trên Kaggle ──────────────────────────────────────────────
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "peft", "trl", "bitsandbytes", "accelerate", "datasets", "pyyaml"],
    check=True,
)

import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer

# ── Load config ───────────────────────────────────────────────────────────────
cfg_path = Path("config/qwen_params.yaml")
if not cfg_path.exists():
    cfg_path = Path("/kaggle/working/config/qwen_params.yaml")
cfg = yaml.safe_load(cfg_path.read_text())

BASE_MODEL   = cfg["model"]["base"]
HF_REPO      = cfg["model"]["hf_repo"]
HF_TOKEN = "INJECT_HF_TOKEN"  # replaced by llm_kaggle_trigger.py

LORA_R       = cfg["lora"]["r"]
LORA_ALPHA   = cfg["lora"]["alpha"]
LORA_DROPOUT = cfg["lora"]["dropout"]
TARGET_MODS  = cfg["lora"]["target_modules"]

EPOCHS       = cfg["training"]["epochs"]
BATCH_SIZE   = cfg["training"]["batch_size"]
GRAD_ACC     = cfg["training"]["grad_accumulation"]
LR           = float(cfg["training"]["learning_rate"])
WARMUP       = cfg["training"]["warmup_ratio"]
MAX_SEQ_LEN  = cfg["training"]["max_seq_length"]

DS_NAME      = cfg["dataset"]["name"]
TRAIN_SIZE   = cfg["dataset"]["train_size"]
TEST_SIZE    = cfg["dataset"]["test_size"]

print(f"Base model : {BASE_MODEL}")
print(f"Dataset    : {DS_NAME}  (train={TRAIN_SIZE}, test={TEST_SIZE})")
print(f"LoRA r={LORA_R}, alpha={LORA_ALPHA}, targets={TARGET_MODS}")
print(f"Epochs={EPOCHS}, lr={LR}, batch={BATCH_SIZE}×{GRAD_ACC}")


# ── Dataset ────────────────────────────────────────────────────────────────────

def format_example(row, tokenizer):
    """Chuyển instruction/input/output → Qwen chat template."""
    instruction = row.get("instruction", "").strip()
    inp         = row.get("input", "").strip()
    output      = row.get("output", "").strip()

    user_msg = instruction
    if inp:
        user_msg = f"{instruction}\n\n{inp}"

    messages = [
        {"role": "system",    "content": "Bạn là Jarvis, trợ lý AI thông minh. Trả lời ngắn gọn và chính xác bằng tiếng Việt."},
        {"role": "user",      "content": user_msg},
        {"role": "assistant", "content": output},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


print("\nLoading dataset…")
raw = load_dataset(DS_NAME, split="train")
raw = raw.shuffle(seed=42)
train_ds = raw.select(range(TRAIN_SIZE))
eval_ds  = raw.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))
print(f"  Train: {len(train_ds)}  Eval: {len(eval_ds)}")


# ── Tokenizer ─────────────────────────────────────────────────────────────────
print("\nLoading tokenizer…")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Format datasets
train_ds = train_ds.map(lambda x: {"text": format_example(x, tokenizer)}, remove_columns=train_ds.column_names)
eval_ds  = eval_ds.map(lambda x:  {"text": format_example(x, tokenizer)},  remove_columns=eval_ds.column_names)


# ── Model (4-bit QLoRA) ───────────────────────────────────────────────────────
print("\nLoading base model (4-bit)…")
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_cfg,
    torch_dtype=torch.float16,    # P100 không hỗ trợ bfloat16
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False

# ── LoRA ──────────────────────────────────────────────────────────────────────
lora_cfg = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODS,
    task_type=TaskType.CAUSAL_LM,
    bias="none",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# Cast bfloat16 → float32 cho LoRA params (P100 không hỗ trợ bf16 training)
for param in model.parameters():
    if param.dtype == torch.bfloat16:
        param.data = param.data.to(torch.float32)


# ── Loss tracking callback ────────────────────────────────────────────────────
class LossCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses  = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:
            self.train_losses.append(logs["loss"])
        if "eval_loss" in logs:
            self.eval_losses.append(logs["eval_loss"])


loss_cb = LossCallback()

# ── SFTTrainer ────────────────────────────────────────────────────────────────
print("\nStarting training…")
sft_cfg = SFTConfig(
    output_dir="/kaggle/working/checkpoints",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LR,
    warmup_steps=10,              # thay warmup_ratio (deprecated transformers v5)
    max_length=MAX_SEQ_LEN,       # trl >= 0.12: max_seq_length → max_length
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=False,    # P100 BFloat16 conflict với GradScaler — LoRA dùng float32
    bf16=False,
    report_to="none",
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,   # trl >= 0.12: tokenizer → processing_class
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    args=sft_cfg,
    callbacks=[loss_cb],
)

train_result = trainer.train()
print(f"\nTraining done. Final train loss: {train_result.training_loss:.4f}")

eval_result  = trainer.evaluate()
eval_loss    = eval_result["eval_loss"]
perplexity   = math.exp(eval_loss)
print(f"Eval loss: {eval_loss:.4f}  Perplexity: {perplexity:.2f}")


# ── Merge adapter vào base model → push full model lên HF Hub ────────────────
if HF_TOKEN:
    try:
        print(f"\nMerging LoRA adapter into base model…")
        merged_model = trainer.model.merge_and_unload()
        # Dequantize hoàn toàn về float16 trước khi push
        merged_model = merged_model.to(torch.float16)
        print(f"Pushing merged model → {HF_REPO}")
        merged_model.push_to_hub(HF_REPO, token=HF_TOKEN, safe_serialization=True)
        tokenizer.push_to_hub(HF_REPO, token=HF_TOKEN)
        print("Push done! Model sẵn sàng trên HF Inference API.")
    except Exception as e:
        print(f"\n⚠ HF push thất bại: {e}")
        print("  → Training metrics vẫn được lưu. Kiểm tra HF_TOKEN có quyền Write không.")
else:
    print("\n⚠ HF_TOKEN chưa đặt — bỏ qua push to Hub")


# ── Sample generations ─────────────────────────────────────────
print("\nGenerating samples…")
sample_prompts = [
    "Thủ đô của Việt Nam là gì?",
    "Giải thích ngắn gọn về trí tuệ nhân tạo.",
    "Hôm nay thời tiết Hà Nội thế nào?",
]

model.eval()
samples_text = []
for prompt in sample_prompts:
    messages = [
        {"role": "system",    "content": "Bạn là Jarvis, trợ lý AI thông minh. Trả lời ngắn gọn và chính xác bằng tiếng Việt."},
        {"role": "user",      "content": prompt},
    ]
    raw = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    # apply_chat_template có thể trả về tensor hoặc BatchEncoding tuỳ version
    input_ids = raw if isinstance(raw, torch.Tensor) else raw["input_ids"]
    input_ids = input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids, max_new_tokens=100, temperature=0.7, do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    samples_text.append(f"Q: {prompt}\nA: {response}")
    print(f"  Q: {prompt}\n  A: {response}\n")


# ── Loss curve ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
if loss_cb.train_losses:
    ax.plot(loss_cb.train_losses, label="Train loss", color="#4f8ef7")
if loss_cb.eval_losses:
    eval_x = [
        int((i + 1) * len(loss_cb.train_losses) / len(loss_cb.eval_losses))
        for i in range(len(loss_cb.eval_losses))
    ]
    ax.plot(eval_x, loss_cb.eval_losses, label="Eval loss", color="#f7854f", marker="o")
ax.set_xlabel("Step"); ax.set_ylabel("Loss")
ax.set_title(f"Qwen2.5-1.5B QLoRA — Loss Curve")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("loss_curve.png", dpi=120)
print("Saved loss_curve.png")


# ── metrics.txt ───────────────────────────────────────────────────────────────
metrics = textwrap.dedent(f"""
    ## Qwen2.5-1.5B QLoRA Training Metrics

    | Metric       | Value     |
    |--------------|-----------|
    | Train loss   | {train_result.training_loss:.4f} |
    | Eval loss    | {eval_loss:.4f} |
    | Perplexity   | {perplexity:.2f} |
    | Epochs       | {EPOCHS} |
    | LR           | {LR} |
    | LoRA r       | {LORA_R} |
    | Model repo   | [{HF_REPO}](https://huggingface.co/{HF_REPO}) |
    | Inference API| `https://api-inference.huggingface.co/models/{HF_REPO}` |

    ## Sample Generations

""").lstrip()

for s in samples_text:
    metrics += s + "\n\n"

Path("metrics.txt").write_text(metrics, encoding="utf-8")
print("Saved metrics.txt")
print("\n✅ Pipeline hoàn thành!")
