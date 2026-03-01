#!/usr/bin/env python3
"""
Jarvis Voice Client + Real-time Debug Dashboard
------------------------------------------------
Architecture:
  Thread-1 (main)  : synchronous wake-word loop (PyAudio stays in one thread)
  Thread-2 (daemon): asyncio event loop for debug WebSocket server → browser
  asyncio.run()    : called from main thread when wake word fires → STT stream
"""

import asyncio
import json
import os
import queue
import sys
import threading
from pathlib import Path
from typing import Set

import numpy as np
import pyaudio
import websockets
from dotenv import load_dotenv
from openwakeword.model import Model

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── Config ────────────────────────────────────────────────────────────────────
VPS_IP      = os.getenv("VPS_IP", "")
VPS_PORT    = os.getenv("VPS_PORT", "8000")
API_KEY     = os.getenv("CLIENT_API_KEY", "")
WAKE_MODEL  = os.getenv("WAKE_WORD_MODEL", "hey_jarvis")
THRESHOLD   = float(os.getenv("WAKE_WORD_THRESHOLD", "0.5"))
SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
DEBUG_PORT  = int(os.getenv("DEBUG_PORT", "8765"))

WW_CHUNK  = 1280   # 80 ms @ 16 kHz
STT_CHUNK = 480    # 30 ms @ 16 kHz

_ip    = f"[{VPS_IP}]" if ":" in VPS_IP else VPS_IP
WS_URL = f"ws://{_ip}:{VPS_PORT}/ws/stt?api_key={API_KEY}"

FORMAT   = pyaudio.paInt16
CHANNELS = 1

# ── Debug broadcast (thread-safe queue → async WS server) ────────────────────
_emit_q: queue.Queue = queue.Queue(maxsize=500)
_browser_clients: Set = set()
_debug_loop: asyncio.AbstractEventLoop = None


def emit(event: dict):
    """Call from ANY thread — puts JSON into thread-safe queue."""
    try:
        _emit_q.put_nowait(json.dumps(event))
    except queue.Full:
        pass


async def _broadcast_loop():
    """Drain _emit_q and fan-out to all connected browser clients."""
    while True:
        try:
            msg = _emit_q.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.03)
            continue
        dead = set()
        for ws in list(_browser_clients):
            try:
                await ws.send(msg)
            except Exception:
                dead.add(ws)
        for ws in dead:
            _browser_clients.discard(ws)   # gọi method, không reassign → không bị local scope


async def _handle_browser(ws):
    _browser_clients.add(ws)
    emit({"type": "log", "level": "info", "msg": "Dashboard connected"})
    try:
        async for _ in ws:   # discard incoming, just keep connection alive
            pass
    finally:
        _browser_clients.discard(ws)


def _start_debug_server():
    """Run in daemon thread — owns its own event loop."""
    global _debug_loop
    _debug_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_debug_loop)

    async def _serve():
        # websockets 14+ requires async-with, not await
        async with websockets.serve(_handle_browser, "localhost", DEBUG_PORT):
            print(f"[Debug WS]  ws://localhost:{DEBUG_PORT}")
            await _broadcast_loop()   # runs forever

    _debug_loop.run_until_complete(_serve())


# ── STT WebSocket stream (called via asyncio.run from main thread) ────────────

async def stream_to_vps():
    pa  = pyaudio.PyAudio()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()

    async def _send(ws):
        stream = pa.open(
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

    async def _recv(ws):
        async for raw in ws:
            msg = json.loads(raw)
            if msg["type"] == "transcript":
                text = msg.get("text", "").strip()
                emit({"type": "transcript", "text": text})
                if text:
                    print(f"\n\033[92m[Jarvis]\033[0m {text}\n")
                stop.set()
            elif msg["type"] == "status":
                s = msg.get("status", "")
                emit({"type": "status",
                      "status": "recording" if s == "detecting" else "processing"})

    emit({"type": "log", "level": "info", "msg": f"Connecting → {WS_URL}"})
    try:
        async with websockets.connect(WS_URL, ping_interval=None) as ws:
            emit({"type": "vps_status", "connected": True})
            hello = json.loads(await asyncio.wait_for(ws.recv(), timeout=5.0))
            if hello.get("type") != "ready":
                emit({"type": "log", "level": "error", "msg": "VPS handshake failed"})
                return

            emit({"type": "status", "status": "recording"})
            st = asyncio.create_task(_send(ws))
            rt = asyncio.create_task(_recv(ws))
            done, pending = await asyncio.wait(
                {st, rt}, return_when=asyncio.FIRST_COMPLETED, timeout=20.0
            )
            for t in pending:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
    except asyncio.TimeoutError:
        emit({"type": "log", "level": "error", "msg": "VPS timeout"})
    except Exception as e:
        emit({"type": "log", "level": "error", "msg": f"VPS error: {e}"})
    finally:
        emit({"type": "vps_status", "connected": False})
        emit({"type": "status",     "status":    "idle"})
        pa.terminate()


# ── Wake word loop (runs fully synchronous in main thread) ───────────────────

def main():
    if not VPS_IP:
        print("[!] VPS_IP chưa đặt trong .env"); sys.exit(1)
    if not API_KEY:
        print("[!] CLIENT_API_KEY chưa đặt trong .env"); sys.exit(1)

    # Start debug WS server in background thread
    t = threading.Thread(target=_start_debug_server, daemon=True)
    t.start()

    emit({"type": "log", "level": "info", "msg": "Loading openWakeWord…"})
    print("Đang tải openWakeWord…")
    oww = Model(wakeword_models=[WAKE_MODEL], inference_framework="onnx")

    emit({"type": "log", "level": "info",
          "msg": f"Listening for '{WAKE_MODEL}' (threshold={THRESHOLD})"})
    emit({"type": "status", "status": "idle"})

    pa = pyaudio.PyAudio()
    ww_stream = pa.open(
        format=FORMAT, channels=CHANNELS,
        rate=SAMPLE_RATE, input=True,
        frames_per_buffer=WW_CHUNK,
    )

    print(f"\n\033[96m[Sẵn sàng]\033[0m Nghe: '\033[1m{WAKE_MODEL}\033[0m'")
    print(f"[Dashboard] http://localhost:3000\n")

    frame_n = 0
    try:
        while True:
            # ── đọc mic hoàn toàn sync, cùng thread ──
            raw   = ww_stream.read(WW_CHUNK, exception_on_overflow=False)
            arr   = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            rms   = float(np.sqrt(np.mean(arr ** 2))) / 32768.0
            pred  = oww.predict(np.frombuffer(raw, dtype=np.int16))
            score = float(pred.get(WAKE_MODEL, 0.0))

            frame_n += 1
            if frame_n % 4 == 0:
                emit({"type": "rms",   "value": round(rms,   3)})
                emit({"type": "score", "value": round(score, 3)})

            if score >= THRESHOLD:
                print(f"\033[93m[Wake Word]\033[0m score={score:.2f}")
                emit({"type": "log", "level": "info",
                      "msg": f"Wake word! score={score:.2f}"})
                emit({"type": "status", "status": "detected"})
                ww_stream.stop_stream()

                asyncio.run(stream_to_vps())   # WebSocket STT

                ww_stream.start_stream()
                emit({"type": "status", "status": "idle"})

    except KeyboardInterrupt:
        print("\nTắt Jarvis.")
    finally:
        ww_stream.stop_stream()
        ww_stream.close()
        pa.terminate()


if __name__ == "__main__":
    main()
