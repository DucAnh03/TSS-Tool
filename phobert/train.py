#!/usr/bin/env python3
"""
PhoBERT Fine-tuning — Vietnamese Sentiment Classification
----------------------------------------------------------
Chạy trên Kaggle T4 GPU.
Input : uitnlp/vietnamese_students_feedback
Output: model đẩy lên HuggingFace Hub + metrics.txt + confusion_matrix.png
"""

import json
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import torch

# ── Load hyperparams ──────────────────────────────────────────────────────────
with open("config/params.yaml") as f:
    p = yaml.safe_load(f)

MODEL_NAME   = p["model"]["name"]
NUM_LABELS   = p["model"]["num_labels"]
MAX_LEN      = p["model"]["max_length"]
EPOCHS       = p["training"]["epochs"]
BATCH        = p["training"]["batch_size"]
LR           = p["training"]["learning_rate"]
WARMUP       = p["training"]["warmup_ratio"]
WD           = p["training"]["weight_decay"]
PATIENCE     = p["training"]["early_stopping_patience"]
DS_NAME      = p["dataset"]["name"]
TEXT_COL     = p["dataset"]["text_column"]
LABEL_COL    = p["dataset"]["label_column"]
TRAIN_SIZE   = p["dataset"]["train_size"]
TEST_SIZE    = p["dataset"]["test_size"]
LABEL_NAMES  = [p["labels"][i] for i in range(NUM_LABELS)]

HF_TOKEN     = os.getenv("HF_TOKEN", "")
HF_REPO      = os.getenv("HF_MODEL_REPO", "")

print(f"Model   : {MODEL_NAME}")
print(f"Dataset : {DS_NAME}")
print(f"Device  : {'GPU' if torch.cuda.is_available() else 'CPU'}")

# ── Dataset ───────────────────────────────────────────────────────────────────
print("\n1. Loading dataset…")
ds = load_dataset(DS_NAME)
train_ds = ds["train"].select(range(min(TRAIN_SIZE, len(ds["train"]))))
test_ds  = ds["test"].select(range(min(TEST_SIZE,  len(ds["test"]))))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch[TEXT_COL],
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
    )

print("2. Tokenizing…")
train_ds = train_ds.map(tokenize, batched=True)
test_ds  = test_ds.map(tokenize, batched=True)

train_ds = train_ds.rename_column(LABEL_COL, "labels")
test_ds  = test_ds.rename_column(LABEL_COL, "labels")

cols = ["input_ids", "attention_mask", "labels"]
train_ds.set_format("torch", columns=cols)
test_ds.set_format("torch",  columns=cols)

# ── Model ─────────────────────────────────────────────────────────────────────
print("3. Loading model…")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS
)

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1":       f1_score(labels, preds, average="weighted"),
    }

# ── Training ──────────────────────────────────────────────────────────────────
print("4. Training…")
args = TrainingArguments(
    output_dir="./phobert_output",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH,
    per_device_eval_batch_size=BATCH,
    learning_rate=LR,
    warmup_ratio=WARMUP,
    weight_decay=WD,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none",
    fp16=torch.cuda.is_available(),
    logging_steps=50,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
)

trainer.train()

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("5. Evaluating…")
results   = trainer.evaluate()
pred_out  = trainer.predict(test_ds)
preds     = np.argmax(pred_out.predictions, axis=-1)
labels    = pred_out.label_ids

acc = results["eval_accuracy"]
f1  = results["eval_f1"]
print(f"   Accuracy : {acc*100:.2f}%")
print(f"   F1       : {f1:.4f}")

# ── metrics.txt for CML ───────────────────────────────────────────────────────
os.makedirs("artifacts/latest", exist_ok=True)

with open("metrics.txt", "w", encoding="utf-8") as fout:
    fout.write(f"- **Accuracy:** {acc*100:.2f}%\n")
    fout.write(f"- **F1 (weighted):** {f1:.4f}\n")
    fout.write(f"- **Model:** `{MODEL_NAME}`\n")
    fout.write(f"- **Dataset:** `{DS_NAME}`\n")
    fout.write(f"- **Epochs:** {EPOCHS} | **LR:** {LR} | **Batch:** {BATCH}\n")

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm   = confusion_matrix(labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(cmap=plt.cm.Blues, ax=ax)
ax.set_title(f"PhoBERT — {DS_NAME}\nAcc={acc*100:.1f}%  F1={f1:.3f}")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120)
plt.savefig("artifacts/latest/phobert_confusion.png", dpi=120)
plt.close()

# ── Push to HuggingFace Hub ───────────────────────────────────────────────────
if HF_REPO and HF_TOKEN:
    print(f"6. Pushing model → {HF_REPO}")
    trainer.model.push_to_hub(HF_REPO, token=HF_TOKEN)
    tokenizer.push_to_hub(HF_REPO, token=HF_TOKEN)
    # Lưu label mapping
    label_map = {str(i): LABEL_NAMES[i] for i in range(NUM_LABELS)}
    import requests
    requests.put(
        f"https://huggingface.co/api/models/{HF_REPO}",
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={"cardData": {"label_map": label_map}},
    )
    print("   Done!")
else:
    print("6. Skipped HF push (no HF_TOKEN / HF_MODEL_REPO)")

print("\n✅ Training complete!")
