source .venv/bin/activate

uvicorn api.server:app --host 0.0.0.0 --port 8000

==============================Run=============================
python voice/client.py
