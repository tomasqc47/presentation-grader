"""
Presentation Grader -- backend for the web UI. Wraps the pipeline
behind /start and /stop endpoints. Models preload at server startup
so Start/Stop are fast during the actual demo.
"""

import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from capture import LiveTranscriber
from metrics import compute_metrics
from pii_redactor import redact_transcript, get_detector, rehydrate
from grader import grade_presentation
from judge import judge_grader_output

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

transcriber: LiveTranscriber | None = None
capture_thread: threading.Thread | None = None
state = {"recording": False}


@app.on_event("startup")
def preload_models():
    global transcriber
    print("Preloading models (this happens once, at server startup)...")
    transcriber = LiveTranscriber(model_size="base")
    get_detector()
    print("All models ready. Server is demo-ready.")


@app.post("/start")
def start_recording():
    global capture_thread
    if state["recording"]:
        return {"status": "already recording"}

    transcriber.transcript_chunks.clear()
    state["recording"] = True

    capture_thread = threading.Thread(target=transcriber.start, daemon=True)
    capture_thread.start()
    return {"status": "recording started"}


@app.post("/stop")
def stop_recording():
    if not state["recording"]:
        return {"status": "not recording"}

    transcriber.stop()
    capture_thread.join(timeout=10)
    state["recording"] = False

    chunks = transcriber.transcript_chunks
    if not chunks:
        return {"status": "no speech captured", "chunks": []}

    metrics = compute_metrics(chunks)
    pii_result = redact_transcript(chunks)
    grader_result = grade_presentation(pii_result["redacted_transcript"], metrics.__dict__)
    judge_result = judge_grader_output(pii_result["redacted_transcript"], grader_result)

    # For the light-screen "chatbot reply", rehydrate the summary so it
    # reads naturally (real name/details, not placeholder tokens). The
    # dark reveal screen shows the REDACTED version + the mapping itself,
    # so the mechanics stay visible there.
    if grader_result.get("summary"):
        chat_reply = rehydrate(grader_result["summary"], pii_result["mapping"])
    else:
        chat_reply = "I had trouble putting together a full evaluation, but you can see everything I did behind the scenes."

    return {
        "status": "done",
        "chat_reply": chat_reply,
        "metrics": metrics.__dict__,
        "pii": pii_result,
        "grader": grader_result,
        "judge": judge_result,
    }


@app.get("/status")
def get_status():
    return {"recording": state["recording"]}


@app.get("/")
def root():
    return {"status": "Presentation grader backend running locally"}