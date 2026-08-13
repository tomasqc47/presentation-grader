"""
Live audio capture with silence-based chunking + faster-whisper
transcription.

Explicitly pins device=0 (MacBook Pro Microphone), confirmed working
via mic_test.py, rather than relying on the system's "default" input
device, which silently changed mid-session in earlier testing.

Also fixes a real pause-tracking bug: chunk end_time was previously
set to chunk_start_offset + full buffer_duration, which INCLUDES the
trailing silence that triggered the cut. That made every chunk's
end_time exactly equal the next chunk's start_time, with zero gap,
regardless of how long the real pause was, structurally hiding all
pauses from metrics.py's gap-based calculation. Now the trailing
silence is trimmed out of the reported spoken duration, so real
pauses show up as real, non-zero gaps between chunks.
"""

import time
import queue
import threading
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION_S = 0.6
MIN_CHUNK_DURATION_S = 2.0
MAX_CHUNK_DURATION_S = 25.0
INPUT_DEVICE = 0  # MacBook Pro Microphone -- confirmed working via mic_test.py

FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)


class LiveTranscriber:
    def __init__(self, model_size: str = "base"):
        print(f"Loading faster-whisper ({model_size})...")
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print("Model loaded.")

        self.transcript_chunks: list[dict] = []
        self._audio_queue: queue.Queue = queue.Queue()
        self._running = False
        self._start_time = None

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[DEBUG] audio callback status: {status}")
        self._audio_queue.put(indata.copy())

    def _rms(self, frame: np.ndarray) -> float:
        return float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))

    def _transcribe_chunk(self, audio: np.ndarray, chunk_start_offset: float, spoken_duration: float):
        if len(audio) < SAMPLE_RATE * 0.3:
            print(f"[DEBUG] chunk too short to transcribe ({len(audio)/SAMPLE_RATE:.2f}s), skipping")
            return
        segments, _ = self.model.transcribe(audio, language="en", vad_filter=True)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        if not text:
            print(f"[DEBUG] Whisper returned EMPTY text for a {len(audio)/SAMPLE_RATE:.2f}s chunk "
                  f"(max amplitude in chunk: {np.max(np.abs(audio)):.4f})")
            return
        chunk_end_offset = chunk_start_offset + spoken_duration
        entry = {
            "text": text,
            "start_time": round(chunk_start_offset, 1),
            "end_time": round(chunk_end_offset, 1),
        }
        self.transcript_chunks.append(entry)
        print(f"[{entry['start_time']:.1f}s-{entry['end_time']:.1f}s] {text}")

    def start(self):
        self._running = True
        self._start_time = time.time()
        buffer = np.zeros((0,), dtype=np.float32)
        silence_run_s = 0.0
        chunk_start_offset = 0.0
        last_debug_print = time.time()
        frames_received = 0

        try:
            devices = sd.query_devices()
            print(f"[DEBUG] Forcing input device index {INPUT_DEVICE}: {devices[INPUT_DEVICE]['name']}")
        except Exception as e:
            print(f"[DEBUG] Could not query device {INPUT_DEVICE}: {e}")

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=FRAME_SAMPLES, callback=self._audio_callback,
            device=INPUT_DEVICE,
        )
        stream.start()
        print("Listening... (Ctrl+C to stop)")

        try:
            while self._running:
                try:
                    frame = self._audio_queue.get(timeout=1.0)
                except queue.Empty:
                    print("[DEBUG] no audio frames received in the last 1s -- "
                          "stream may not be delivering data at all")
                    continue

                frames_received += 1
                frame = frame.flatten()
                buffer = np.concatenate([buffer, frame])
                current_rms = self._rms(frame)
                is_quiet = current_rms < SILENCE_THRESHOLD
                silence_run_s = silence_run_s + FRAME_DURATION_MS / 1000 if is_quiet else 0.0

                buffer_duration = len(buffer) / SAMPLE_RATE

                if time.time() - last_debug_print >= 1.0:
                    print(f"[DEBUG] frames={frames_received} rms={current_rms:.5f} "
                          f"quiet={is_quiet} buffer_duration={buffer_duration:.2f}s "
                          f"silence_run={silence_run_s:.2f}s")
                    last_debug_print = time.time()

                should_cut_on_silence = (
                    silence_run_s >= SILENCE_DURATION_S and buffer_duration >= MIN_CHUNK_DURATION_S
                )
                should_force_cut = buffer_duration >= MAX_CHUNK_DURATION_S

                if should_cut_on_silence or should_force_cut:
                    reason = "silence" if should_cut_on_silence else "max_length"
                    print(f"[DEBUG] cutting chunk: duration={buffer_duration:.2f}s reason={reason}")
                    spoken_duration = (
                        max(buffer_duration - silence_run_s, 0.1) if should_cut_on_silence else buffer_duration
                    )
                    threading.Thread(
                        target=self._transcribe_chunk,
                        args=(buffer.copy(), chunk_start_offset, spoken_duration),
                        daemon=True,
                    ).start()
                    chunk_start_offset += buffer_duration
                    buffer = np.zeros((0,), dtype=np.float32)
                    silence_run_s = 0.0
        except KeyboardInterrupt:
            pass
        finally:
            stream.stop()
            stream.close()
            print(f"[DEBUG] stream closed. Final buffer duration: {len(buffer)/SAMPLE_RATE:.2f}s, "
                  f"total frames received: {frames_received}")
            if len(buffer) / SAMPLE_RATE >= 0.5:
                final_duration = len(buffer) / SAMPLE_RATE
                self._transcribe_chunk(buffer.copy(), chunk_start_offset, final_duration)
            else:
                print(f"[DEBUG] final buffer too short ({len(buffer)/SAMPLE_RATE:.2f}s < 0.5s) to transcribe")
            print("\nStopped.")

    def stop(self):
        self._running = False

    def full_transcript(self) -> str:
        return " ".join(c["text"] for c in self.transcript_chunks)


if __name__ == "__main__":
    transcriber = LiveTranscriber(model_size="base")
    transcriber.start()
    print("\n" + "=" * 60)
    print("FULL TRANSCRIPT")
    print("=" * 60)
    print(transcriber.full_transcript())