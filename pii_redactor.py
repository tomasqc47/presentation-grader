"""
PII redaction for the presentation transcript. Exact reuse of Problem 1's
recommended, tested configuration: Presidio + en_core_web_trf, with
custom phone/NIF recognizers, and the same numbered-placeholder
mask/rehydrate pattern used throughout the PII prototype.

The only change from the PII prototype's app.py: this operates on a
full transcript (or a single chunk of one) rather than a single chat
message, and exposes redact_transcript() as the main entry point used
by the rest of this pipeline.
"""

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


class PresidioDetectorTRF:
    """
    Same as PresidioDetector, but backed by en_core_web_trf
    (transformer-based spaCy model) instead of en_core_web_lg.
    Unchanged from the PII prototype / Problem 1 report.
    """

    def __init__(self, language: str = "en"):
        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_trf"}],
        })
        nlp_engine = provider.create_engine()
        self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
        self.anonymizer = AnonymizerEngine()
        self.language = language
        self._add_custom_recognizers()

    def _add_custom_recognizers(self):
        nif_pattern = Pattern(name="nif_pattern", regex=r"\b[1-3]\d{8}\b", score=0.75)
        nif_recognizer = PatternRecognizer(
            supported_entity="NIF",
            patterns=[nif_pattern],
            context=["nif", "tax id", "contribuinte"],
        )
        self.analyzer.registry.add_recognizer(nif_recognizer)

        phone_pattern = Pattern(
            name="pt_phone_pattern",
            regex=r"\b(?:\+351\s?)?9\d{2}[\s.-]?\d{3}[\s.-]?\d{3}\b",
            score=0.85,
        )
        phone_recognizer = PatternRecognizer(
            supported_entity="PHONE_NUMBER",
            patterns=[phone_pattern],
            context=["phone", "call", "mobile", "telemóvel", "contacto"],
        )
        self.analyzer.registry.add_recognizer(phone_recognizer)

    def detect(self, text: str, score_threshold: float = 0.5):
        results = self.analyzer.analyze(text=text, language=self.language)
        return [r for r in results if r.score >= score_threshold]

    def anonymize(self, text: str, results, mode: str = "placeholder") -> str:
        if mode == "placeholder":
            operators = {"DEFAULT": OperatorConfig("replace", {"new_value": None})}
        else:
            operators = {}
        return self.anonymizer.anonymize(text=text, analyzer_results=results, operators=operators).text


def mask_pii(text: str, results, mode: str = "placeholder"):
    """Unchanged from the PII prototype: numbered placeholder + mapping."""
    mapping = {}
    counters = {}
    result_parts = []
    last_idx = 0
    sorted_results = sorted(results, key=lambda r: r.start)

    for r in sorted_results:
        result_parts.append(text[last_idx:r.start])
        entity_type = r.entity_type
        counters[entity_type] = counters.get(entity_type, 0) + 1
        if mode == "placeholder":
            token = f"[{entity_type}_{counters[entity_type]}]"
            mapping[token] = text[r.start:r.end]
            result_parts.append(token)
        last_idx = r.end

    result_parts.append(text[last_idx:])
    return "".join(result_parts), mapping


def rehydrate(text: str, mapping: dict) -> str:
    for token, original in mapping.items():
        text = text.replace(token, original)
    return text


_detector = None
def get_detector() -> PresidioDetectorTRF:
    """Lazy singleton -- loading en_core_web_trf is slow, do it once."""
    global _detector
    if _detector is None:
        print("Loading Presidio + en_core_web_trf...")
        _detector = PresidioDetectorTRF()
        print("PII detector ready.")
    return _detector


def redact_transcript(chunks: list[dict]) -> dict:
    """
    Takes the list of transcript chunks from capture.py, redacts PII
    across the FULL transcript (not per-chunk), so an entity split
    across a chunk boundary is still caught correctly, and returns:
      - the redacted full transcript (safe to send to any AI call)
      - the entities that were found (for displaying at the reveal)
      - the mapping (kept local, used only to rehydrate for display)
    """
    detector = get_detector()
    full_text = " ".join(c["text"] for c in chunks)

    results = detector.detect(full_text)
    redacted_text, mapping = mask_pii(full_text, results)

    entities_found = [
        {"type": r.entity_type, "text": full_text[r.start:r.end], "score": round(r.score, 2)}
        for r in results
    ]

    return {
        "original_transcript": full_text,
        "redacted_transcript": redacted_text,
        "entities_found": entities_found,
        "mapping": mapping,
    }


if __name__ == "__main__":
    sample_chunks = [
        {"text": "Hi everyone, my name is Tomas Silva and I work at Euronext.", "start_time": 0.0, "end_time": 5.0},
        {"text": "You can reach me at 912 345 678 if you have questions.", "start_time": 5.5, "end_time": 10.0},
    ]

    result = redact_transcript(sample_chunks)
    print("\nOriginal:\n", result["original_transcript"])
    print("\nRedacted (safe to send to AI):\n", result["redacted_transcript"])
    print("\nEntities found:")
    for e in result["entities_found"]:
        print(f"  {e['type']}: '{e['text']}' (score={e['score']})")
    print("\nMapping (local only):", result["mapping"])