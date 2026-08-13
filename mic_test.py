"""
Minimal mic test, no Whisper, no server, just: record 5 seconds,
report the maximum volume level detected. If this stays near 0.0
throughout, the microphone isn't actually being captured (permission
issue or wrong input device), not a bug in the rest of the pipeline.
"""

import sounddevice as sd
import numpy as np

print("Available audio devices:")
print(sd.query_devices())
print(f"\nDefault input device: {sd.default.device}")

print("\nRecording 5 seconds... talk normally now.")
duration = 5
sample_rate = 16000
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
sd.wait()

max_level = float(np.max(np.abs(audio)))
rms_level = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))

print(f"\nMax amplitude: {max_level:.4f}")
print(f"RMS level:     {rms_level:.4f}")

if max_level < 0.005:
    print("\n>>> Essentially silent. Likely a mic permission or wrong-device issue.")
else:
    print("\n>>> Real audio detected, capture itself is working.")