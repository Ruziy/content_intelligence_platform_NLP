import time

from .extraction.ner_spacy import extract_entities_spacy
from .formatting.json_formatter import build_document
from .preprocessing.cleaning import run_cleaning
from .preprocessing.language_detection import detect_language
from .preprocessing.tokenization import spacy_tokenize


def process_text_with_trace(text, cleaning_method="full", ner_extractor=None):
    """
    Прогоняет полный pipeline и возвращает trace с промежуточными данными и таймингами.
    """
    ner_extractor = ner_extractor or extract_entities_spacy
    stage_timings = {}

    start = time.perf_counter()
    clean_text = run_cleaning(text, method=cleaning_method)
    stage_timings["cleaning_ms"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    tokens = spacy_tokenize(clean_text)
    stage_timings["tokenization_ms"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    language = detect_language(clean_text)
    stage_timings["language_ms"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    entities = ner_extractor(clean_text)
    stage_timings["ner_ms"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    document = build_document(
        text=clean_text,
        tokens=tokens,
        entities=entities,
        language=language,
    )
    stage_timings["formatting_ms"] = (time.perf_counter() - start) * 1000.0
    stage_timings["total_ms"] = sum(stage_timings.values())

    return {
        "clean_text": clean_text,
        "tokens": tokens,
        "language": language,
        "entities": entities,
        "document": document,
        "stage_timings": stage_timings,
    }


def process_text(text, cleaning_method="full", ner_extractor=None):
    """Упрощенный API: возвращает только итоговый document без trace."""
    trace = process_text_with_trace(
        text=text,
        cleaning_method=cleaning_method,
        ner_extractor=ner_extractor,
    )
    return trace["document"]