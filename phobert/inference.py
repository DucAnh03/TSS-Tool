#!/usr/bin/env python3
"""
PhoBERT Inference — load model từ HuggingFace Hub, chạy trên CPU VPS.
Được gọi bởi api/server.py sau khi STT trả về text.
"""

import os
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_tokenizer = None
_model     = None
_labels    = None


def _load():
    global _tokenizer, _model, _labels
    if _model is not None:
        return

    repo = os.getenv("HF_MODEL_REPO", "")
    token = os.getenv("HF_TOKEN", "")

    if not repo:
        print("[PhoBERT] HF_MODEL_REPO chưa đặt → bỏ qua inference")
        return

    print(f"[PhoBERT] Loading model từ {repo}…")
    kwargs = {"token": token} if token else {}

    _tokenizer = AutoTokenizer.from_pretrained(repo, **kwargs)
    _model = AutoModelForSequenceClassification.from_pretrained(
        repo, **kwargs
    )
    _model.eval()

    # Label map: đọc từ .env hoặc dùng mặc định
    raw = os.getenv("PHOBERT_LABELS", "0:tiêu cực,1:trung tính,2:tích cực")
    _labels = {}
    for item in raw.split(","):
        idx, name = item.split(":")
        _labels[int(idx)] = name.strip()

    print(f"[PhoBERT] Ready — labels: {_labels}")


def predict(text: str) -> Optional[dict]:
    """
    Trả về: {"label": "tích cực", "label_id": 2, "confidence": 0.93}
    Trả về None nếu model chưa load.
    """
    _load()
    if _model is None:
        return None

    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True,
    )
    with torch.no_grad():
        logits = _model(**inputs).logits
        probs  = torch.softmax(logits, dim=-1)[0]
        label_id = int(probs.argmax())
        confidence = float(probs[label_id])

    return {
        "label":      _labels.get(label_id, str(label_id)),
        "label_id":   label_id,
        "confidence": round(confidence, 3),
    }
