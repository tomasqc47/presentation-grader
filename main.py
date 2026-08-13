"""
Orchestrator: wires the full pipeline together.

    Live mic
       -> capture.py       (silence-based chunked transcription)
       -> metrics.py        (deterministic WPM, fillers, pauses)
       -> pii_redactor.py   (Presidio + en_core_web_trf redaction)
       -> grader.py         (AI call #1: the presentation grader)
       -> judge.py          (AI call #2: evaluates the grader's output)

Recording stops on Ctrl+C, everything after that runs automatically,
that's the "reveal" moment: nothing needs to happen live/on-screen
during the talk itself, only silent background capture.

No Langfuse, no database: kept in-memory for this single live run,
same simplicity principle as the monitoring prototype.
"""

from capture import LiveTranscriber
from metrics import compute_metrics
from pii_redactor import redact_transcript, rehydrate
from grader import grade_presentation
from judge import judge_grader_output


def run():
    # --- Phase 1: live capture (silent, runs until Ctrl+C) ---
    transcriber = LiveTranscriber(model_size="base")
    transcriber.start()

    if not transcriber.transcript_chunks:
        print("No speech captured. Nothing to grade.")
        return

    # --- Phase 2: deterministic metrics (instant, no AI) ---
    metrics = compute_metrics(transcriber.transcript_chunks)

    # --- Phase 3: PII redaction (Presidio, no AI call) ---
    pii_result = redact_transcript(transcriber.transcript_chunks)

    # --- Phase 4: AI call #1, the grader ---
    print("\nGrading presentation...")
    grader_result = grade_presentation(pii_result["redacted_transcript"], metrics.__dict__)

    # --- Phase 5: AI call #2, the judge (evaluates the grader, not the talk) ---
    print("Checking grader's evaluation quality...")
    judge_result = judge_grader_output(pii_result["redacted_transcript"], grader_result)

    # --- Reveal ---
    print_reveal(metrics, pii_result, grader_result, judge_result)

    return {
        "metrics": metrics,
        "pii_result": pii_result,
        "grader_result": grader_result,
        "judge_result": judge_result,
    }


def print_reveal(metrics, pii_result, grader_result, judge_result):
    print("\n" + "=" * 70)
    print("PRESENTATION EVALUATION")
    print("=" * 70)

    print(f"\nDuration: {metrics.duration_seconds}s | Words: {metrics.word_count} | "
          f"WPM: {metrics.words_per_minute}")
    print(f"Filler words: {metrics.filler_count} {metrics.filler_breakdown}")
    print(f"Longest pause: {metrics.longest_pause_seconds}s | Pauses: {metrics.num_pauses}")

    print(f"\nPII detected and redacted before reaching any AI ({len(pii_result['entities_found'])} entities):")
    for e in pii_result["entities_found"]:
        print(f"  {e['type']}: \"{e['text']}\" (score={e['score']})")

    print(f"\nRedacted transcript sent to the grading AI:")
    print(f"  {pii_result['redacted_transcript']}")

    if grader_result.get("parse_failure"):
        print(f"\nGRADER FAILED: {grader_result['parse_failure']}")
        return

    print(f"\n--- Grader's Evaluation ---")
    print(f"Clarity: {grader_result['clarity']}/10")
    print(f"Structure: {grader_result['structure']}/10")
    print(f"Engagement: {grader_result['engagement']}/10")
    print(f"Technical accuracy: {grader_result['technical_accuracy']}/10")
    print(f"\nSummary: {grader_result['summary']}")
    print(f"\nSuggestions:")
    for s in grader_result["suggestions"]:
        print(f"  - {s}")

    print(f"\n--- Judge's Verdict on the Grader's Evaluation ---")
    print(f"Score: {judge_result['score']} ({judge_result['verdict']})")
    print(f"Reason: {judge_result['reason']}")


if __name__ == "__main__":
    run()