#!/usr/bin/env python3
"""
Liệt kê tất cả thiết bị âm thanh và test RMS từng mic.
Chạy: python scripts/list_mics.py
"""
import pyaudio
import numpy as np

pa = pyaudio.PyAudio()
print(f"\n{'─'*60}")
print(f"{'IDX':>4}  {'TÊN THIẾT BỊ':<38}  {'INPUT CH':>8}")
print(f"{'─'*60}")

input_devices = []
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        print(f"  {i:>2}  {info['name'][:38]:<38}  {int(info['maxInputChannels']):>8}")
        input_devices.append(i)

print(f"{'─'*60}")
print(f"\nTìm thấy {len(input_devices)} thiết bị input: {input_devices}\n")

# Test nhanh từng mic (0.5 giây mỗi cái)
CHUNK  = 1024
RATE   = 16000
FRAMES = int(RATE / CHUNK * 0.5)

print("Đang test RMS từng mic (giữ yên lặng)…\n")
for idx in input_devices:
    info = pa.get_device_info_by_index(idx)
    try:
        stream = pa.open(
            format=pyaudio.paInt16, channels=1,
            rate=RATE, input=True,
            input_device_index=idx,
            frames_per_buffer=CHUNK,
        )
        rms_vals = []
        for _ in range(FRAMES):
            raw = stream.read(CHUNK, exception_on_overflow=False)
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            rms_vals.append(np.sqrt(np.mean(arr**2)) / 32768.0)
        stream.stop_stream()
        stream.close()
        avg_rms = np.mean(rms_vals)
        bar = '█' * int(avg_rms * 200) + '░' * (20 - int(avg_rms * 200))
        bar = bar[:20]
        status = "← có tín hiệu ✓" if avg_rms > 0.005 else "(im lặng)"
        print(f"  [{idx:>2}] {bar}  rms={avg_rms:.4f}  {status}")
        print(f"       {info['name'][:50]}")
    except Exception as e:
        print(f"  [{idx:>2}] Lỗi: {e}")
    print()

pa.terminate()
print("\n→ Ghi lại IDX của mic bạn muốn dùng, đặt vào .env: MIC_DEVICE_INDEX=<idx>")
