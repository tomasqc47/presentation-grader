"""
Deterministic presentation metrics, computed directly from transcript
chunks, no AI involved. Same principle as Problem 2's proxy metrics
layer: don't pay an LLM for things plain code can measure exactly.
"""

import re
from dataclasses import dataclass

FILLER_WORDS = [
    "um", "uh", "uhh", "umm", "like", "basically", "actually",
    "you know", "sort of", "kind of", "i mean", "so yeah",
]


@dataclass
class PresentationMetrics:
    duration_seconds: float
    word_count: int
    words_per_minute: float
    filler_count: int
    filler_breakdown: dict
    longest_pause_seconds: float
    num_pauses: int


def _count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _count_fillers(full_text: str) -> tuple[int, dict]:
    text = full_text.lower()
    breakdown = {}
    total = 0
    for filler in FILLER_WORDS:
        pattern = r"\b" + re.escape(filler) + r"\b"
        count = len(re.findall(pattern, text))
        if count > 0:
            breakdown[filler] = count
            total += count
    return total, breakdown


def _pause_analysis(chunks: list[dict]) -> tuple[float, int]:
    """
    Gaps between the end of one chunk and the start of the next are
    real pauses (that's literally what triggered the chunk cut).
    """
    if len(chunks) < 2:
        return 0.0, 0
    pauses = []
    for i in range(1, len(chunks)):
        gap = chunks[i]["start_time"] - chunks[i - 1]["end_time"]
        if gap > 0.1:  # ignore near-zero rounding noise
            pauses.append(gap)
    longest = max(pauses) if pauses else 0.0
    return round(longest, 1), len(pauses)


def compute_metrics(chunks: list[dict]) -> PresentationMetrics:
    if not chunks:
        return PresentationMetrics(0, 0, 0, 0, {}, 0, 0)

    full_text = " ".join(c["text"] for c in chunks)
    duration = chunks[-1]["end_time"] - chunks[0]["start_time"]
    word_count = _count_words(full_text)
    wpm = round((word_count / duration) * 60, 1) if duration > 0 else 0.0
    filler_total, filler_breakdown = _count_fillers(full_text)
    longest_pause, num_pauses = _pause_analysis(chunks)

    return PresentationMetrics(
        duration_seconds=round(duration, 1),
        word_count=word_count,
        words_per_minute=wpm,
        filler_count=filler_total,
        filler_breakdown=filler_breakdown,
        longest_pause_seconds=longest_pause,
        num_pauses=num_pauses,
    )


if __name__ == "__main__":
    sample_chunks = [
        {"text": "So, um, today I want to talk about, like, our new project.", "start_time": 0.0, "end_time": 5.0},
        {"text": "It basically covers three main areas of the business.", "start_time": 6.5, "end_time": 10.0},
        {"text": "You know, the first one is actually pretty important.", "start_time": 10.2, "end_time": 14.0},
    ]

    m = compute_metrics(sample_chunks)
    print("Duration:", m.duration_seconds, "seconds")
    print("Word count:", m.word_count)
    print("Words per minute:", m.words_per_minute)
    print("Filler word count:", m.filler_count)
    print("Filler breakdown:", m.filler_breakdown)
    print("Longest pause:", m.longest_pause_seconds, "seconds")
    print("Number of pauses:", m.num_pauses)