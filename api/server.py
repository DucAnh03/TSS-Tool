#!/usr/bin/env python3
"""
Jarvis STT Server
-----------------
POST /stt          → HTTP upload (backward compat)
WS   /ws/stt       → real-time PCM stream + VAD → transcript

Audio format expected from client:
  PCM 16-bit signed, mono, 16 000 Hz, 480-sample chunks (30 ms)
"""

import asyncio
import io
import os
import tempfile
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
from dotenv import load_dotenv
from fastapi import (
    FastAPI, File, Header, HTTPException,
    Query, UploadFile, WebSocket, WebSocketDisconnect,
)
from faster_whisper import WhisperModel

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY     = os.getenv("API_SECRET_KEY", "")
MODEL_NAME  = os.getenv("WHISPER_MODEL", "base")
DEVICE      = os.getenv("WHISPER_DEVICE", "cpu")
COMPUTE     = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
LANG        = os.getenv("WHISPER_LANGUAGE", "vi")
SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
HOST        = os.getenv("VPS_HOST", "0.0.0.0")
PORT        = int(os.getenv("VPS_PORT", "8000"))

# VAD tuning — adjust RMS_THRESHOLD if mic is too quiet / too sensitive
CHUNK_SAMPLES  = 480    # 30 ms @16 kHz — must match client STT_CHUNK
RMS_THRESHOLD  = 400    # RMS energy above this → voice detected
SILENCE_SEC    = 0.8    # seconds of silence after speech → trigger transcription
MIN_SPEECH_SEC = 0.2    # ignore noise bursts shorter than this

# ── App ───────────────────────────────────────────────────────────────────────
app      = FastAPI(title="Jarvis STT API")
_pool    = ThreadPoolExecutor(max_workers=2)

print("Loading Whisper model…")
_model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE)
print("Whisper ready!")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _rms(pcm: bytes) -> float:
    arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(arr ** 2))) if len(arr) else 0.0


def _frames_to_wav(frames: List[bytes]) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)          # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))
    return buf.getvalue()


def _transcribe(wav_bytes: bytes) -> dict:
    """Blocking CPU call — always run inside thread executor."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        path = f.name
    try:
        segs, info = _model.transcribe(path, beam_size=5, language=LANG)
        text = " ".join(s.text for s in segs).strip()
        return {"text": text, "language": info.language}
    finally:
        os.remove(path)


# ── HTTP /stt (backward compat) ───────────────────────────────────────────────

@app.post("/stt")
async def stt_http(
    audio_file: UploadFile = File(...),
    x_api_key:  str        = Header(...),
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    wav_bytes = await audio_file.read()
    loop   = asyncio.get_running_loop()
    result = await loop.run_in_executor(_pool, _transcribe, wav_bytes)
    return {"status": "success", **result}


# ── WebSocket /ws/stt ─────────────────────────────────────────────────────────

@app.websocket("/ws/stt")
async def stt_ws(ws: WebSocket, api_key: str = Query(...)):
    if api_key != API_KEY:
        await ws.close(code=4001, reason="Unauthorized")
        return

    await ws.accept()
    await ws.send_json({
        "type": "ready",
        "chunk_samples": CHUNK_SAMPLES,
        "sample_rate":   SAMPLE_RATE,
    })

    chunk_sec       = CHUNK_SAMPLES / SAMPLE_RATE       # 0.030 s
    silence_chunks  = int(SILENCE_SEC    / chunk_sec)   # frames of silence needed
    min_spch_chunks = int(MIN_SPEECH_SEC / chunk_sec)   # minimum speech frames

    speech:    List[bytes] = []
    n_speech   = 0
    n_silence  = 0
    talking    = False
    loop       = asyncio.get_running_loop()

    try:
        while True:
            chunk = await ws.receive_bytes()
            is_voice = _rms(chunk) > RMS_THRESHOLD

            if is_voice:
                speech.append(chunk)
                n_speech  += 1
                n_silence  = 0
                if not talking:
                    talking = True
                    await ws.send_json({"type": "status", "status": "detecting"})

            else:
                if talking:
                    speech.append(chunk)        # buffer trailing silence too
                    n_silence += 1

                    if n_silence >= silence_chunks:
                        if n_speech >= min_spch_chunks:
                            wav    = _frames_to_wav(speech)
                            result = await loop.run_in_executor(
                                _pool, _transcribe, wav
                            )
                            await ws.send_json({"type": "transcript", **result})

                        # reset → ready for next utterance
                        speech.clear()
                        n_speech  = 0
                        n_silence = 0
                        talking   = False
                        await ws.send_json({"type": "status", "status": "waiting"})

    except WebSocketDisconnect:
        pass


# ── Dev entry ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=HOST, port=PORT, reload=False)
