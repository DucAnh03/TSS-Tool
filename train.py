#!/usr/bin/env python3
"""
Emotion Classification — HuggingFace dair-ai/emotion
-----------------------------------------------------
Dataset : dair-ai/emotion  (6 classes: sadness, joy, love, anger, fear, surprise)
Model   : TF-IDF + Logistic Regression  (fast, demo-friendly)
Outputs : metrics.txt, confusion_matrix.png  → picked up by CML
"""

import json
import os

import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import make_pipeline

CLASS_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# ── 1. Load dataset ───────────────────────────────────────────────────────────
print("1. Đang tải dataset từ HuggingFace (dair-ai/emotion)…")
ds = load_dataset("dair-ai/emotion")

TRAIN_SIZE = int(os.getenv("TRAIN_SIZE", "2000"))
TEST_SIZE  = int(os.getenv("TEST_SIZE",  "500"))

train = ds["train"].select(range(TRAIN_SIZE))
test  = ds["test"].select(range(TEST_SIZE))

X_train, y_train = train["text"], train["label"]
X_test,  y_test  = test["text"],  test["label"]

# ── 2. Train ──────────────────────────────────────────────────────────────────
print("2. Đang huấn luyện TF-IDF + Logistic Regression…")
model = make_pipeline(
    TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
    LogisticRegression(max_iter=500, C=1.0),
)
model.fit(X_train, y_train)

# ── 3. Evaluate ───────────────────────────────────────────────────────────────
print("3. Đang đánh giá…")
y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)

print(f"\nAccuracy : {acc * 100:.2f}%")
print(report)

# ── 4. Save metrics.txt (CML reads this) ─────────────────────────────────────
os.makedirs("artifacts/latest", exist_ok=True)

with open("metrics.txt", "w", encoding="utf-8") as f:
    f.write(f"- **Accuracy:** {acc * 100:.2f}%\n")
    f.write(f"- **Train samples:** {TRAIN_SIZE}\n")
    f.write(f"- **Test samples:** {TEST_SIZE}\n")
    f.write(f"- **Dataset:** dair-ai/emotion (HuggingFace)\n")
    f.write(f"- **Model:** TF-IDF (5k features, bigrams) + LogisticRegression\n\n")
    f.write("### Per-class Report\n\n```\n")
    f.write(report)
    f.write("```\n")

# ── 5. Confusion matrix PNG ───────────────────────────────────────────────────
cm   = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
fig, ax = plt.subplots(figsize=(8, 7))
disp.plot(cmap=plt.cm.Oranges, xticks_rotation=45, ax=ax)
ax.set_title("Confusion Matrix — Emotion Classification")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120)
plt.savefig("artifacts/latest/emotion_confusion.png", dpi=120)
plt.close()

# ── 6. Update metrics.json contract ──────────────────────────────────────────
import subprocess
from datetime import datetime, timezone

try:
    sha = subprocess.check_output(
        ["git", "rev-parse", "--short=7", "HEAD"],
        stderr=subprocess.DEVNULL, text=True
    ).strip()
except Exception:
    sha = "nogit"

metrics_path = "artifacts/latest/metrics.json"
try:
    with open(metrics_path) as f:
        data = json.load(f)
except Exception:
    data = {}

data.update({
    "git_sha": sha,
    "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
    "emotion": {
        "dataset": "dair-ai/emotion",
        "model": "TF-IDF+LR",
        "accuracy": round(acc, 4),
        "train_size": TRAIN_SIZE,
        "test_size": TEST_SIZE,
        "confusion_png": "artifacts/latest/emotion_confusion.png",
    },
})
with open(metrics_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"\n✅ Xong! Accuracy = {acc * 100:.2f}%")
print("   metrics.txt, confusion_matrix.png → sẵn sàng cho CML")
