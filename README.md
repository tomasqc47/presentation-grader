# Presentation Grader

A live tool that listens to a spoken presentation, transcribes it, redacts any personal
information before it touches an AI model, and produces a structured evaluation with a second AI
model checking the first one's work — all running locally.

## Context

Built as an applied demo combining two things from the same take-home technical exercise: PII
protection for AI-facing text, and evaluating whether an LLM's output is actually trustworthy. A
presentation coach was chosen as a concrete, testable case for both.

## How it works

```
Live mic
   │
   ▼
capture.py     silence-based chunked transcription (faster-whisper)
   │
   ▼
metrics.py     deterministic speech metrics — WPM, filler words, pauses (no AI)
   │
   ▼
pii_redactor   detect + redact PII before anything reaches an LLM (see the PII Redaction Module)
   │
   ▼
grader.py      AI call #1 — evaluates the redacted transcript: clarity, structure, engagement
   │
   ▼
judge.py       AI call #2 — checks whether the grader's evaluation is actually accurate
   │
   ▼
Reveal         full pipeline shown step by step, including the redaction mapping
```

Nothing needs to happen on-screen during the talk itself — recording runs silently, and every
step above runs automatically once it stops.

## Two ways to run it

**Web UI** (`app.py` + `index.html`) — a two-screen demo experience: a listening orb while
recording, a natural chat-style reply once done, and a "reveal" screen showing every pipeline
step underneath, including exactly what was redacted and what the judge said about the grader.

**CLI** (`main.py`) — the same pipeline, no server or browser needed: start it, talk, `Ctrl+C` to
stop, and the full evaluation prints straight to the terminal.

Both call the same underlying pipeline; the web UI just wraps it behind `/start` and `/stop`
endpoints and preloads models at server startup so the actual demo is fast.

## Why two separate AI calls

The grader alone is just one model's opinion. The judge's job is narrower and more useful:
checking whether the grader's *summary* actually reflects what was said, and whether its *scores*
are justified by the transcript — not re-grading the talk itself.

Getting the judge right took real iteration. Three different approaches using DeepEval's GEval
wrapper (free-text criteria, explicit evaluation steps, rubric score-bands) all failed the same
way on this task: the judge would claim the grader's suggestions weren't specific to the
transcript while quoting the exact transcript words in its own reasoning. The fix was moving off
GEval entirely for this task — a plain, direct LLM call with a fixed system prompt and manual
JSON parsing, checking exactly two things (is the summary accurate, are the scores plausible)
instead of a general-purpose rubric.

A `technical_accuracy` score was tried and removed: it defaulted to a placeholder "7" for
non-technical talks, which read as a real evaluation rather than the fallback it actually was —
three dimensions that are always meaningful beat four where one is sometimes fake.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/download) installed
- A working microphone
- ~2GB free disk space for the judge model, plus space for the grader model, Whisper, and the
  PII detection model

```bash
pip install fastapi uvicorn ollama sounddevice numpy faster-whisper \
            presidio-analyzer presidio-anonymizer spacy
python -m spacy download en_core_web_trf
```

## Setup

**1. Pull the required Ollama models:**

```bash
ollama pull llama3.2:3b   # grader
ollama pull mistral:7b    # judge
```

**2. Test your microphone first** — this catches permission/device issues before you're mid-demo:

```bash
python mic_test.py
```

Records 5 seconds and reports the volume detected. If it stays near zero, it's a mic permission
or wrong-input-device issue, not a bug in the rest of the pipeline — worth ruling out early.

## Running it

**Ollama's server must be running throughout:**

```bash
ollama serve
```

**Web UI:**

```bash
uvicorn app:app --reload --port 8002
```

Models preload at startup (watch the terminal for "All models ready"), then open `index.html`.
Click **Start Listening**, give your talk, click **Stop & Send**. You'll get a short natural
reply first, then a **Reveal** button showing the full pipeline underneath — speech metrics, what
was detected and redacted, the grader's evaluation, and the judge's verdict on it.

**CLI:**

```bash
python main.py
```

Talk, then `Ctrl+C` to stop. The full evaluation prints to the terminal automatically.

## Troubleshooting

- **Silence detected, nothing transcribes** — run `mic_test.py` first to isolate whether it's a
  capture problem or a transcription problem.
- **"Error: could not reach backend"** (web UI) — confirm `uvicorn` is running on port 8002 and
  that `index.html`'s `API_URL` matches.
- **Startup is slow** — expected. Whisper, the PII detector, and both Ollama models all need to
  load; the web UI preloads them once at startup specifically so the actual recording/grading
  loop is fast afterward.
- **Judge flags something that looks fine** — read the actual redacted transcript and grader
  output yourself before trusting a "FLAGGED" verdict. Even the more carefully validated judge
  setup here has produced verdicts that didn't match its own stated reasoning in testing.
