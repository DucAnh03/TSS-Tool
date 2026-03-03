#!/usr/bin/env python3
"""
Jarvis LLM inference.
Primary  : HF Inference API (model fine-tuned trên HF Hub)
Fallback : Ollama local (qwen3:4b) nếu HF lỗi / chưa có model
"""

import json
import os
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

HF_REPO      = os.getenv("LLM_ADAPTER_REPO", "DucAnh03/qwen-vi-jarvis")
HF_TOKEN     = os.getenv("HF_TOKEN", "")
OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")

HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_REPO}"

SYSTEM_PROMPT = (
    "Bạn là Jarvis, trợ lý AI thông minh của chủ nhân. "
    "Trả lời ngắn gọn, rõ ràng bằng tiếng Việt. Tối đa 2-3 câu."
)


# ── HF Inference API ──────────────────────────────────────────────────────────

def _hf_generate(text: str, max_tokens: int) -> str:
    """Gọi HF Serverless Inference API."""
    # Dùng chat completions endpoint (hỗ trợ instruction models)
    payload = json.dumps({
        "model": HF_REPO,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": text.strip()},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }).encode()

    req = Request(
        "https://api-inference.huggingface.co/v1/chat/completions",
        data=payload,
        headers={
            "Authorization":  f"Bearer {HF_TOKEN}",
            "Content-Type":   "application/json",
        },
        method="POST",
    )
    with urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read())
    return body["choices"][0]["message"]["content"].strip()


# ── Ollama fallback ────────────────────────────────────────────────────────────

def _ollama_generate(text: str, max_tokens: int) -> str:
    """Gọi Ollama local (qwen3:4b), think mode tắt."""
    payload = json.dumps({
        "model":  OLLAMA_MODEL,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0.7},
        "messages": [
            {"role": "system", "content": "/no_think " + SYSTEM_PROMPT},
            {"role": "user",   "content": text.strip()},
        ],
    }).encode()
    req = Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read())
    response = body["message"]["content"].strip()
    # Xóa <think>...</think> nếu còn sót
    if "<think>" in response:
        end = response.find("</think>")
        response = response[end + 8:].strip() if end != -1 else response
    return response


# ── Public API ────────────────────────────────────────────────────────────────

def generate(text: str, max_tokens: int = 150) -> str:
    """
    Sinh response cho câu hỏi/lệnh từ user.
    Thử HF Inference API trước, fallback Ollama nếu lỗi.
    """
    if HF_TOKEN:
        try:
            response = _hf_generate(text, max_tokens)
            print("[LLM] source=HF_API")
            return response
        except HTTPError as e:
            if e.code == 503:
                print(f"[LLM] HF model đang load ({e}) → fallback Ollama")
            else:
                print(f"[LLM] HF API lỗi {e.code} → fallback Ollama")
        except Exception as e:
            print(f"[LLM] HF API lỗi: {e} → fallback Ollama")
    else:
        print("[LLM] HF_TOKEN chưa đặt → dùng Ollama")

    try:
        response = _ollama_generate(text, max_tokens)
        print("[LLM] source=Ollama")
        return response
    except Exception as e:
        return f"[Lỗi LLM: {e}]"


# ── CLI test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    prompt = " ".join(sys.argv[1:]) or "Xin chào, bạn là ai?"
    print(f"Q: {prompt}")
    print(f"A: {generate(prompt)}")
