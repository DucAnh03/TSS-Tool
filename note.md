source .venv/bin/activate

uvicorn api.server:app --host 0.0.0.0 --port 8000

==============================Run=============================
python voice/client.py

=====================flow=========================
git push config/qwen_params.yaml
↓
GitHub Actions (ubuntu-latest)
kaggle_trigger.py → Kaggle T4 (~10-15 phút)
↓
llm/train.py (Kaggle):
QLoRA fine-tune Qwen2.5-1.5B
→ merge_and_unload() ← full model, không phải adapter
→ push_to_hub DucAnh03/qwen-vi-jarvis
→ metrics.txt + loss_curve.png
↓
CML comment vào PR:
loss_curve.png + train_loss + eval_loss + perplexity + 3 sample generations
↓
Laptop (Jarvis runtime):
llm/inference.py
→ HF Inference API (fine-tuned model) ← primary
→ Ollama qwen3:4b ← fallback

Secrets cần thêm vào GitHub repo:

- KAGGLE_USERNAME, KAGGLE_KEY
- HF_TOKEN
- HF_ADAPTER_REPO = DucAnh03/qwen-vi-jarvis
- REPO_TOKEN (GitHub PAT, CML dùng để comment PR)

# ─────────────────────────────────

# echo "" >> config/params.yaml

=================================

# 1. Download model mới từ HF (overwrite)

huggingface-cli download Ducanh1123312/qwen-vi-jarvis --local-dir D:\local_bot\qwen-vi-jarvis-clean --force-download

# 2. Convert lại GGUF (overwrite)

python D:\llama.cpp\convert_hf_to_gguf.py D:\local_bot\qwen-vi-jarvis-clean --outfile D:\local_bot\qwen-vi-jarvis.gguf --outtype q8_0

# 3. Update Ollama (overwrite model cũ)

ollama create jarvis-vi -f D:\local_bot\Modelfile
