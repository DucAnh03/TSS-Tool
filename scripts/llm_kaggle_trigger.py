#!/usr/bin/env python3
"""
Trigger Kaggle kernel cho Qwen2.5-1.5B QLoRA training từ GitHub Actions.
Clone pattern từ scripts/kaggle_trigger.py.
"""

import json
import os
import shutil
import subprocess
import time
import yaml
from pathlib import Path

USERNAME    = os.environ["KAGGLE_USERNAME"]
HF_TOKEN    = os.environ.get("HF_TOKEN", "")
KERNEL_SLUG = "qwen-vi-training"
KERNEL_ID   = f"{USERNAME}/{KERNEL_SLUG}"
PUSH_DIR    = Path("_kaggle_push_llm")


def kaggle(*args):
    """Chạy kaggle CLI, in output, raise nếu lỗi."""
    cmd = ["kaggle"] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.returncode != 0:
        print(result.stderr.strip())
        raise RuntimeError(f"kaggle {' '.join(args)} → exit {result.returncode}")
    return result.stdout


def get_status():
    """Trả về status string: running / queued / complete / error / unknown."""
    r = subprocess.run(
        ["kaggle", "kernels", "status", KERNEL_ID],
        capture_output=True, text=True,
    )
    out = (r.stdout + r.stderr).lower()
    print(f"    [raw] {(r.stdout + r.stderr).strip()[:120]}")
    for st in ["running", "queued", "complete", "error", "cancelacknowledged", "cancel"]:
        if st in out:
            return st
    return "unknown"


# ── Chuẩn bị thư mục push ─────────────────────────────────────────────────────
if PUSH_DIR.exists():
    shutil.rmtree(PUSH_DIR)
PUSH_DIR.mkdir()

# ── Inject config + HF_TOKEN trực tiếp vào train.py trước khi push ──────────
cfg = yaml.safe_load(Path("config/qwen_params.yaml").read_text())

train_code = Path("llm/train.py").read_text()

# Thay block đọc yaml bằng dict inline — không cần file path trên Kaggle
train_code = train_code.replace(
    'cfg_path = Path("config/qwen_params.yaml")\n'
    'if not cfg_path.exists():\n'
    '    cfg_path = Path("/kaggle/working/config/qwen_params.yaml")\n'
    'cfg = yaml.safe_load(cfg_path.read_text())',
    f'cfg = {repr(cfg)}'
)

# Inject HF_TOKEN qua placeholder
train_code = train_code.replace(
    '"INJECT_HF_TOKEN"',
    f'"{HF_TOKEN}"' if HF_TOKEN else '""',
)

(PUSH_DIR / "train.py").write_text(train_code)

meta = {
    "id":              KERNEL_ID,
    "title":           "Qwen VI Training",
    "code_file":       "train.py",
    "language":        "python",
    "kernel_type":     "script",
    "is_private":      True,
    "enable_gpu":      True,
    "enable_internet": True,
    "dataset_sources":     [],
    "competition_sources": [],
    "kernel_sources":      [],
}
(PUSH_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

# ── Push kernel ───────────────────────────────────────────────────────────────
print(f"Pushing kernel → {KERNEL_ID}")
kaggle("kernels", "push", "-p", str(PUSH_DIR))
print("Chờ Kaggle queue run mới (60s)…")
time.sleep(60)

# ── Poll trạng thái ───────────────────────────────────────────────────────────
MAX_WAIT     = 90 * 60   # 90 phút (QLoRA lâu hơn PhoBERT)
waited       = 0
interval     = 30
new_run_seen = False

while waited < MAX_WAIT:
    st = get_status()
    print(f"  [{waited//60:02d}m] status = {st}")

    if st in ("running", "queued"):
        new_run_seen = True

    if st == "complete":
        if new_run_seen:
            print("✅ Kernel hoàn thành!")
            break
        else:
            print("  (status cũ — chờ run mới…)")
    elif st in ("error", "cancelacknowledged", "cancel"):
        if new_run_seen:
            raise RuntimeError(f"Kaggle kernel thất bại: {st}")
        else:
            print(f"  (status cũ {st} — chờ run mới…)")

    time.sleep(interval)
    waited += interval
else:
    raise TimeoutError("Kaggle kernel không hoàn thành trong 90 phút")

# ── Pull output ───────────────────────────────────────────────────────────────
print("Pulling output từ Kaggle…")
OUT_DIR = Path("_kaggle_output_llm")
OUT_DIR.mkdir(exist_ok=True)
kaggle("kernels", "output", KERNEL_ID, "-p", str(OUT_DIR))

for fname in ["metrics.txt", "loss_curve.png"]:
    src = OUT_DIR / fname
    if src.exists():
        shutil.copy(src, fname)
        print(f"  Copied {fname}")
    else:
        print(f"  ⚠ {fname} không tìm thấy trong output")

print("Done!")
