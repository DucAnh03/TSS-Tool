source .venv/bin/activate

uvicorn api.server:app --host 0.0.0.0 --port 8000

==============================Run=============================
python voice/client.py

=====================flow=========================
git push (sửa config/params.yaml)
↓
GitHub Actions (ubuntu-latest)
↓ Kaggle API
Kaggle T4 GPU — fine-tune PhoBERT
↓ push model
HuggingFace Hub (lưu model)
↓ CML
PR comment: metrics + confusion matrix

─────────────────────────────────
Runtime flow (sau khi có model):

Laptop mic → "Hey Jarvis"
↓ WebSocket
VPS :8000 — faster-whisper → text
↓ PhoBERT (load từ HF Hub)
intent / sentiment
↓
Trả về laptop: {text, intent, confidence}
