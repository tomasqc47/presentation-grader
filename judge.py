"""
AI call #2: the judge. Evaluates the GRADER's response, not the
presentation.

Moved OFF DeepEval's GEval wrapper entirely. Three GEval mechanisms
(free-text criteria, evaluation_steps, rubric score-bands) all failed
the same way on this task, the judge repeatedly claimed the grader's
suggestions weren't specific to the transcript while quoting the exact
transcript words in its own reasoning.

This version uses a plain, direct LLM call instead: a system prompt +
format="json" + manual parsing, the same pattern used successfully in
grader.py, and the same pattern that held up well in the original
report's hand_built_judge.py.

Trimmed to two conditions (summary accuracy, score plausibility).
Calibrated to distinguish reasonable paraphrase/interpretation from
actual fabrication after an early run flagged a fair paraphrase
("nice because my wife loves it" -> "to impress their wife") as
inaccurate.

Now returns and prints summary_accurate/scores_plausible individually,
not just a blended reason string, so we can actually see which of the
two checks is driving a FLAGGED verdict.

technical_accuracy removed from the text shown to the judge, matching
its removal from grader.py.

Local Ollama only (mistral:7b).
"""

import json
import re
import ollama

JUDGE_MODEL = "mistral:7b"

JUDGE_SYSTEM_PROMPT = """You are a quality reviewer. You are NOT grading the
presentation itself, you are checking whether ANOTHER AI's evaluation of that
presentation was done well.

You will be given the original (redacted) transcript, and the grader AI's scores
and summary. Check exactly two things:

1. summary_accurate: does the grader's summary honestly reflect what the transcript
   actually covered? Be lenient here: reasonable paraphrasing, and reasonable
   interpretation of something clearly implied by the transcript (even if not
   stated in those exact words), should PASS. For example, if the speaker says
   something is "nice because my partner loves it," describing that as "done to
   please their partner" is a fair paraphrase, not fabrication. Only mark this
   false if the summary invents a detail, claim, or topic that has NO reasonable
   basis in the transcript at all, or directly contradicts what was said.
2. scores_plausible: are the numeric scores (clarity, structure, engagement)
   reasonable given the actual transcript content, not arbitrarily high or low
   without any justification?

Respond with ONLY a JSON object, nothing else. No explanation outside the JSON,
no markdown fences. All string values must be wrapped in double quotes.
Format exactly:
{"summary_accurate": true or false, "scores_plausible": true or false,
 "reason": "<one or two sentences explaining your judgment, quoting specific words
 from the transcript or the grader's output as evidence>"}
"""


def judge_grader_output(redacted_transcript: str, grader_result: dict) -> dict:
    if grader_result.get("parse_failure"):
        return {"score": 0.0, "verdict": "N/A", "reason": "Grader output failed to parse, cannot judge.",
                "summary_accurate": None, "scores_plausible": None}

    grader_output_text = (
        f"Clarity: {grader_result['clarity']}/10\n"
        f"Structure: {grader_result['structure']}/10\n"
        f"Engagement: {grader_result['engagement']}/10\n"
        f"Summary: {grader_result['summary']}"
    )

    content = (
        f"ORIGINAL TRANSCRIPT (redacted):\n{redacted_transcript}\n\n"
        f"GRADER AI'S OUTPUT:\n{grader_output_text}"
    )

    response = ollama.chat(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        format="json",
        options={"temperature": 0},
    )
    raw = response["message"]["content"].strip()
    return _parse(raw)


def _parse(raw: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        result = json.loads(cleaned)
        if "summary_accurate" not in result or "scores_plausible" not in result:
            raise ValueError("missing required fields")
        if not isinstance(result["summary_accurate"], bool) or not isinstance(result["scores_plausible"], bool):
            raise ValueError("summary_accurate/scores_plausible must be true/false")

        passed = result["summary_accurate"] and result["scores_plausible"]
        return {
            "score": 1.0 if passed else 0.0,
            "verdict": "PASS" if passed else "FLAGGED for review",
            "summary_accurate": result["summary_accurate"],
            "scores_plausible": result["scores_plausible"],
            "reason": result.get("reason", ""),
        }
    except (json.JSONDecodeError, ValueError) as e:
        return {"score": 0.0, "verdict": "N/A", "reason": f"Judge output failed to parse: {e} | raw: {raw[:200]}",
                "summary_accurate": None, "scores_plausible": None}


if __name__ == "__main__":
    from grader import grade_presentation

    sample_transcript = (
        "Hello, my name is [PERSON_1] I smell very good. I'm a human, and I "
        "smell very, very, very good. Almost always Because there are a lot "
        "of times that I have to go to bed and I don't want to take a shower "
        "So I just gave myself in bed all dirty, but it's nice because my "
        "wife loves it. Out."
    )
    sample_metrics = {
        "duration_seconds": 57.9, "words_per_minute": 108.8, "filler_count": 0,
        "filler_breakdown": {}, "longest_pause_seconds": 0.0, "num_pauses": 0,
    }

    print("Step 1: grading (llama3.2:3b)...")
    grader_result = grade_presentation(sample_transcript, sample_metrics)
    print("Grader output:", grader_result)

    print("\nStep 2: judging the grader's output (hand-built, mistral:7b)...")
    judge_result = judge_grader_output(sample_transcript, grader_result)
    print("\nJudge score:", judge_result["score"], f"({judge_result['verdict']})")
    print("  summary_accurate:", judge_result.get("summary_accurate"))
    print("  scores_plausible:", judge_result.get("scores_plausible"))
    print("Judge reasoning:", judge_result["reason"])