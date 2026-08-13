"""
AI call #1: the presentation grader itself, the "chatbot" imagined in
the original brief, applied here to grading presentations instead of
answering support questions.

Takes the REDACTED transcript plus deterministic metrics, and produces
a structured evaluation. Uses the same fail-closed JSON parsing
pattern proven throughout the report.

technical_accuracy was removed after live testing: it defaulted to a
placeholder "7" for non-technical talks, which the judge (reasonably)
flagged as an unjustified score, since it looked like a real
evaluation rather than a fallback. Three dimensions, all always
meaningful, beat four where one is sometimes fake.

Local Ollama only.
"""

import json
import re
import ollama

GRADER_MODEL = "llama3.2:3b"

GRADER_SYSTEM_PROMPT = """You are an expert presentation coach evaluating a live talk.
You are NOT a conversational assistant. You will be given a transcript of the talk
(some personal details have been redacted with placeholder tokens like [PERSON_1],
treat these as normal parts of the sentence, do not comment on them) and some speech
metrics (words per minute, filler word count, pauses).

Evaluate the presentation on three dimensions, each from 1 (very poor) to 10 (excellent):
- clarity: was the content easy to follow and understand?
- structure: did the talk have a clear beginning, middle, and end, with logical flow?
- engagement: was the delivery likely to hold an audience's attention?

Also provide:
- summary: a 2-3 sentence summary of what the talk covered
- suggestions: a list of 2-4 concrete, actionable suggestions for improvement

Important: base your suggestions on what the transcript and metrics actually show, not
generic presentation advice. Specifically:
- Only suggest reducing filler words if the filler word count is greater than 0. If it is
  0, do not mention filler words at all.
- Only suggest addressing long pauses if longest_pause_seconds is notably long (over ~3s)
  or num_pauses is unusually high for the talk's length. Do not mention pacing/pauses if
  the numbers are unremarkable.
- Ground every suggestion in something specific from the transcript content itself
  (a topic that was underexplained, a claim without support, a missing example) rather
  than a suggestion that could apply to any presentation.

Respond with ONLY a JSON object, nothing else. No explanation, no markdown fences.
All string values MUST be wrapped in double quotes.
Format exactly:
{"clarity": <1-10>, "structure": <1-10>, "engagement": <1-10>,
 "summary": "<2-3 sentences>", "suggestions": ["<suggestion 1>", "<suggestion 2>", ...]}
"""


def _call_grader(redacted_transcript: str, metrics: dict) -> str:
    metrics_summary = (
        f"Duration: {metrics['duration_seconds']}s | "
        f"Words per minute: {metrics['words_per_minute']} | "
        f"Filler words used: {metrics['filler_count']} ({metrics['filler_breakdown']}) | "
        f"Longest pause: {metrics['longest_pause_seconds']}s | "
        f"Number of pauses: {metrics['num_pauses']}"
    )
    content = f"TRANSCRIPT:\n{redacted_transcript}\n\nSPEECH METRICS:\n{metrics_summary}"

    response = ollama.chat(
        model=GRADER_MODEL,
        messages=[
            {"role": "system", "content": GRADER_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        format="json",
        options={"temperature": 0.3},
    )
    return response["message"]["content"].strip()


def grade_presentation(redacted_transcript: str, metrics: dict) -> dict:
    raw = _call_grader(redacted_transcript, metrics)
    result = _parse(raw)

    if result.get("parse_failure"):
        print("[DEBUG] Grader JSON parse failed, retrying once...")
        raw = _call_grader(redacted_transcript, metrics)
        result = _parse(raw)

    return result


def _parse(raw: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    required_scores = {"clarity", "structure", "engagement"}
    try:
        result = json.loads(cleaned)
        if not required_scores.issubset(result.keys()):
            raise ValueError("missing required score fields")
        for key in required_scores:
            if not isinstance(result[key], (int, float)) or not (1 <= result[key] <= 10):
                raise ValueError(f"invalid score for {key}: {result[key]}")
        if "summary" not in result or "suggestions" not in result:
            raise ValueError("missing summary or suggestions")
        return result
    except (json.JSONDecodeError, ValueError) as e:
        return {
            "clarity": None, "structure": None, "engagement": None,
            "summary": None, "suggestions": [],
            "parse_failure": f"{e} | raw: {raw[:200]}",
        }


if __name__ == "__main__":
    sample_transcript = (
        "Hi everyone, my name is [PERSON_1] and today I want to talk about how "
        "we protect user privacy when building AI chatbots. First, I'll cover "
        "the problem of sending personal information to language models. Then "
        "I'll walk through the solution we built, using a tool called Presidio "
        "combined with a transformer-based language model to detect names, "
        "phone numbers, and other sensitive data before it ever reaches the AI. "
        "Finally, I'll show you a live demo of the system working in real time. "
        "Let's get started."
    )
    sample_metrics = {
        "duration_seconds": 22.0,
        "words_per_minute": 145.0,
        "filler_count": 0,
        "filler_breakdown": {},
        "longest_pause_seconds": 0.8,
        "num_pauses": 3,
    }

    print("Grading sample presentation (this calls the local LLM, may take a few seconds)...\n")
    result = grade_presentation(sample_transcript, sample_metrics)

    if result.get("parse_failure"):
        print("PARSE FAILURE:", result["parse_failure"])
    else:
        print(f"Clarity: {result['clarity']}/10")
        print(f"Structure: {result['structure']}/10")
        print(f"Engagement: {result['engagement']}/10")
        print(f"\nSummary: {result['summary']}")
        print(f"\nSuggestions:")
        for s in result['suggestions']:
            print(f"  - {s}")