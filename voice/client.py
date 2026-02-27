#!/usr/bin/env python3
"""
Jarvis Voice Client (Laptop)
----------------------------
1. Liên tục nghe mic với openWakeWord.
2. Khi phát hiện "Hey Jarvis" → mở WebSocket tới VPS.
3. Stream PCM realtime qua WebSocket.
4. VPS dùng VAD + faster-whisper → trả text về.
5. In text → đóng WebSocket → quay lại bước 1.

Chạy: python voice/client.py
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np
import pyaudio
import websockets
from dotenv import load_dotenv
from openwakeword.model import Model

# Load .env từ thư mục gốc project
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── Config ────────────────────────────────────────────────────────────────────
VPS_IP      = os.getenv("VPS_IP", "")
VPS_PORT    = os.getenv("VPS_PORT", "8000")
API_KEY     = os.getenv("CLIENT_API_KEY", "")
WAKE_MODEL  = os.getenv("WAKE_WORD_MODEL", "hey_jarvis")
THRESHOLD   = float(os.getenv("WAKE_WORD_THRESHOLD", "0.5"))
SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))

# Chunk sizes
WW_CHUNK  = 1280   # openWakeWord cần frame 80ms (1280 samples @ 16kHz)
STT_CHUNK = 480    # VAD server cần frame 30ms  (480  samples @ 16kHz)

# Build WebSocket URL — IPv6 cần dấu ngoặc vuông []
_ip    = f"[{VPS_IP}]" if ":" in VPS_IP else VPS_IP
WS_URL = f"ws://{_ip}:{VPS_PORT}/ws/stt?api_key={API_KEY}"

FORMAT   = pyaudio.paInt16
CHANNELS = 1

# ── PyAudio instance (dùng chung) ─────────────────────────────────────────────
_pa = pyaudio.PyAudio()


# ── WebSocket streaming ───────────────────────────────────────────────────────

async def _send_audio(ws, stop: asyncio.Event):
    """Đọc mic theo chunk 480 mẫu và gửi lên WebSocket cho đến khi stop."""
    loop   = asyncio.get_running_loop()
    stream = _pa.open(
        format=FORMAT, channels=CHANNELS,
        rate=SAMPLE_RATE, input=True,
        frames_per_buffer=STT_CHUNK,
    )
    try:
        while not stop.is_set():
            data = await loop.run_in_executor(
                None, lambda: stream.read(STT_CHUNK, exception_on_overflow=False)
            )
            await ws.send(data)
    finally:
        stream.stop_stream()
        stream.close()


async def _recv_text(ws, stop: asyncio.Event):
    """Nhận message từ server và in transcript."""
    async for raw in ws:
        msg = json.loads(raw)

        if msg["type"] == "transcript":
            text = msg.get("text", "").strip()
            if text:
                print(f"\n\033[92m[Jarvis]\033[0m {text}\n")
            stop.set()          # nhận được transcript → dừng gửi audio

        elif msg["type"] == "status":
            status = msg.get("status", "")
            if status == "detecting":
                print("  \033[93m●\033[0m Đang nghe...", end="\r", flush=True)
            elif status == "waiting":
                print("  \033[94m◌\033[0m Đang xử lý...", end="\r", flush=True)


async def stream_to_vps():
    """Kết nối WebSocket, stream audio và nhận kết quả STT."""
    stop = asyncio.Event()
    try:
        async with websockets.connect(WS_URL, ping_interval=None) as ws:
            # Handshake
            hello = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
            if hello.get("type") != "ready":
                print("[!] Handshake thất bại, thử lại.")
                return

            print("  \033[91m●\033[0m Đang ghi âm — nói câu lệnh của bạn...")

            send_task = asyncio.create_task(_send_audio(ws, stop))
            recv_task = asyncio.create_task(_recv_text(ws, stop))

            done, pending = await asyncio.wait(
                {send_task, recv_task},
                return_when=asyncio.FIRST_COMPLETED,
                timeout=20.0,
            )
            for t in pending:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass

    except asyncio.TimeoutError:
        print("[!] Timeout — VPS không phản hồi.")
    except Exception as exc:
        print(f"[!] WebSocket lỗi: {exc}")


# ── Wake word loop ────────────────────────────────────────────────────────────

def main():
    if not VPS_IP:
        print("[!] VPS_IP chưa được đặt trong .env")
        sys.exit(1)
    if not API_KEY:
        print("[!] CLIENT_API_KEY chưa được đặt trong .env")
        sys.exit(1)

    print("Đang tải openWakeWord…")
    oww = Model(wakeword_models=[WAKE_MODEL], inference_framework="onnx")
    print(f"\n\033[96m[Sẵn sàng]\033[0m Đang nghe wake word: '\033[1m{WAKE_MODEL}\033[0m' (ngưỡng={THRESHOLD})")
    print("Nói 'Hey Jarvis' để bắt đầu. Ctrl+C để thoát.\n")

    ww_stream = _pa.open(
        format=FORMAT, channels=CHANNELS,
        rate=SAMPLE_RATE, input=True,
        frames_per_buffer=WW_CHUNK,
    )

    try:
        while True:
            raw  = ww_stream.read(WW_CHUNK, exception_on_overflow=False)
            arr  = np.frombuffer(raw, dtype=np.int16)
            pred = oww.predict(arr)
            score = pred.get(WAKE_MODEL, 0.0)

            if score >= THRESHOLD:
                print(f"\033[93m[Wake Word]\033[0m score={score:.2f} → kết nối VPS…")
                ww_stream.stop_stream()

                asyncio.run(stream_to_vps())

                print(f"\n\033[96m[Sẵn sàng]\033[0m Đang nghe lại: '{WAKE_MODEL}'…\n")
                ww_stream.start_stream()

    except KeyboardInterrupt:
        print("\nTắt Jarvis.")
    finally:
        ww_stream.stop_stream()
        ww_stream.close()
        _pa.terminate()


if __name__ == "__main__":
    main()
