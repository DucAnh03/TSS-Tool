#!/usr/bin/env python3
"""
Qwen3 Ollama evaluation — chạy trên GitHub Actions hoặc laptop.
Đọc prompts từ config/qwen_params.yaml, gọi Ollama, lưu metrics + chart.
"""

import json
import os
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

# ── Config ─────────────────────────────────────────────────────────────────────
cfg_path = Path("config/qwen_params.yaml")
cfg = yaml.safe_load(cfg_path.read_text())

OLLAMA_URL  = os.getenv("OLLAMA_URL",   "http://localhost:11434")
# CI dùng model nhỏ hơn, local dùng 4b
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL",
    cfg["ollama"].get("model_ci", "qwen3:1.7b"))
MAX_TOKENS  = cfg["eval"]["max_tokens"]
PROMPTS     = cfg["eval"]["prompts"]

SYSTEM_PROMPT = (
    "/no_think "
    "Bạn là Jarvis, trợ lý AI. Trả lời ngắn gọn bằng tiếng Việt. Tối đa 2-3 câu."
)

print(f"Model : {OLLAMA_MODEL}")
print(f"Ollama: {OLLAMA_URL}")
print(f"Prompts: {len(PROMPTS)}\n")


def call_ollama(prompt: str) -> tuple[str, float]:
    """Gọi Ollama chat API, trả về (response, elapsed_seconds)."""
    payload = json.dumps({
        "model":  OLLAMA_MODEL,
        "stream": False,
        "options": {"num_predict": MAX_TOKENS, "temperature": 0.7},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    }).encode()
    req = Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())
    elapsed = time.time() - t0
    response = body["message"]["content"].strip()
    if "<think>" in response:
        end = response.find("</think>")
        response = response[end + 8:].strip() if end != -1 else response
    return response, elapsed


# ── Chạy eval ─────────────────────────────────────────────────────────────────
results = []
for i, prompt in enumerate(PROMPTS, 1):
    print(f"[{i}/{len(PROMPTS)}] {prompt[:50]}…")
    try:
        response, elapsed = call_ollama(prompt)
        char_len = len(response)
        print(f"  → {elapsed:.1f}s | {char_len} chars | {response[:80]}…\n")
        results.append({
            "prompt":   prompt,
            "response": response,
            "time_s":   round(elapsed, 2),
            "chars":    char_len,
            "ok":       True,
        })
    except URLError as e:
        print(f"  ✗ Ollama không chạy: {e}\n")
        results.append({"prompt": prompt, "response": "", "time_s": 0,
                         "chars": 0, "ok": False})
    except Exception as e:
        print(f"  ✗ Lỗi: {e}\n")
        results.append({"prompt": prompt, "response": str(e), "time_s": 0,
                         "chars": 0, "ok": False})

ok_results = [r for r in results if r["ok"]]
if not ok_results:
    print("❌ Không có kết quả nào — Ollama có đang chạy không?")
    sys.exit(1)

avg_time  = sum(r["time_s"] for r in ok_results) / len(ok_results)
avg_chars = sum(r["chars"]  for r in ok_results) / len(ok_results)
print(f"\n✅ {len(ok_results)}/{len(results)} prompts OK")
print(f"   Avg response time : {avg_time:.2f}s")
print(f"   Avg response length: {avg_chars:.0f} chars")


# ── metrics.txt ───────────────────────────────────────────────────────────────
lines = [
    f"## Qwen3 Ollama Eval — {OLLAMA_MODEL}\n",
    f"| Metric              | Value     |",
    f"|---------------------|-----------|",
    f"| Model               | {OLLAMA_MODEL} |",
    f"| Prompts chạy        | {len(ok_results)}/{len(results)} |",
    f"| Avg response time   | {avg_time:.2f}s |",
    f"| Avg response length | {avg_chars:.0f} chars |",
    "",
    "### Sample Responses",
    "",
]
for r in results:
    status = "✅" if r["ok"] else "❌"
    lines.append(f"**{status} Q:** {r['prompt']}")
    lines.append(f"**A:** {r['response'][:200] if r['response'] else '(lỗi)'}")
    lines.append(f"*({r['time_s']}s, {r['chars']} chars)*")
    lines.append("")

Path("metrics.txt").write_text("\n".join(lines), encoding="utf-8")
print("Saved metrics.txt")


# ── eval_chart.png ────────────────────────────────────────────────────────────
labels = [f"P{i+1}" for i in range(len(results))]
times  = [r["time_s"] for r in results]
chars  = [r["chars"]  for r in results]
colors = ["#4f8ef7" if r["ok"] else "#f87171" for r in results]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.bar(labels, times, color=colors)
ax1.set_title("Response Time (s)")
ax1.set_ylabel("seconds")
ax1.axhline(avg_time, color="#fbbf24", linestyle="--", label=f"avg {avg_time:.1f}s")
ax1.legend(); ax1.grid(axis="y", alpha=0.3)

ax2.bar(labels, chars, color=colors)
ax2.set_title("Response Length (chars)")
ax2.set_ylabel("characters")
ax2.axhline(avg_chars, color="#fbbf24", linestyle="--", label=f"avg {avg_chars:.0f}")
ax2.legend(); ax2.grid(axis="y", alpha=0.3)

fig.suptitle(f"Qwen3 Ollama Eval — {OLLAMA_MODEL}", fontsize=12)
fig.tight_layout()
fig.savefig("eval_chart.png", dpi=120)
print("Saved eval_chart.png")
print("\n✅ Eval hoàn thành!")
