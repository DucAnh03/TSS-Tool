#!/usr/bin/env python3
"""
Trigger Kaggle kernel cho PhoBERT training từ GitHub Actions.
Chạy: python scripts/kaggle_trigger.py
"""

import json
import os
import shutil
import time
import zipfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApiExtended

api = KaggleApiExtended()
api.authenticate()

USERNAME    = os.environ["KAGGLE_USERNAME"]
KERNEL_SLUG = "phobert-vi-training"
KERNEL_ID   = f"{USERNAME}/{KERNEL_SLUG}"
PUSH_DIR    = Path("_kaggle_push")

# ── Chuẩn bị thư mục push lên Kaggle ─────────────────────────────────────────
if PUSH_DIR.exists():
    shutil.rmtree(PUSH_DIR)
PUSH_DIR.mkdir()

# Copy files cần thiết — giữ nguyên cấu trúc thư mục
shutil.copy("phobert/train.py", PUSH_DIR / "train.py")
(PUSH_DIR / "config").mkdir(exist_ok=True)
shutil.copy("config/params.yaml", PUSH_DIR / "config/params.yaml")

# Tạo kernel-metadata.json
meta = {
    "id":           KERNEL_ID,
    "title":        "PhoBERT VI Training",
    "code_file":    "train.py",
    "language":     "python",
    "kernel_type":  "script",
    "is_private":   True,
    "enable_gpu":   True,
    "enable_internet": True,
    "dataset_sources":    [],
    "competition_sources": [],
    "kernel_sources":      [],
    "environment_variables": [
        {"key": "HF_TOKEN",      "value": os.environ.get("HF_TOKEN",      "")},
        {"key": "HF_MODEL_REPO", "value": os.environ.get("HF_MODEL_REPO", "")},
    ],
}
(PUSH_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

# ── Push kernel ───────────────────────────────────────────────────────────────
print(f"Pushing kernel → {KERNEL_ID}")
api.kernels_push(str(PUSH_DIR))
print("Kernel pushed, waiting for run to start…")
time.sleep(15)

# ── Poll trạng thái ───────────────────────────────────────────────────────────
MAX_WAIT = 60 * 60   # 1 giờ tối đa
waited   = 0
interval = 30

while waited < MAX_WAIT:
    status = api.kernel_status(USERNAME, KERNEL_SLUG)
    print(f"  [{waited//60:02d}m] status = {status.status}")

    if status.status == "complete":
        print("✅ Kernel hoàn thành!")
        break
    if status.status in ("error", "cancelAcknowledged", "cancel"):
        raise RuntimeError(f"Kaggle kernel thất bại: {status.status}")

    time.sleep(interval)
    waited += interval
else:
    raise TimeoutError("Kaggle kernel không hoàn thành trong 1 giờ")

# ── Pull output ───────────────────────────────────────────────────────────────
print("Pulling output từ Kaggle…")
OUT_DIR = Path("_kaggle_output")
OUT_DIR.mkdir(exist_ok=True)
api.kernel_output(USERNAME, KERNEL_SLUG, path=str(OUT_DIR))

# Copy artifacts về đúng chỗ
for fname in ["metrics.txt", "confusion_matrix.png"]:
    src = OUT_DIR / fname
    if src.exists():
        shutil.copy(src, fname)
        print(f"  Copied {fname}")
    else:
        print(f"  ⚠ {fname} không tìm thấy trong output")

print("Done!")
