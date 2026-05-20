from source.v2.rus_text_platform.text_processing.evaluation.ner_evaluator import (
    evaluate_full_pipeline,
)


def _make_trace(text, clean_text, tokens, language, entities):
    """Формирует trace-объект с фиктивными таймингами для smoke-теста."""
    return {
        "clean_text": clean_text,
        "tokens": tokens,
        "language": language,
        "entities": entities,
        "document": {
            "text": clean_text,
            "tokens": tokens,
            "entities": entities,
            "language": language,
        },
        "stage_timings": {
            "cleaning_ms": 1.2,
            "tokenization_ms": 0.8,
            "language_ms": 0.4,
            "ner_ms": 1.0,
            "formatting_ms": 0.2,
            "total_ms": 3.6,
        },
    }


def _good_pipeline(text):
    """Пайплайн-заглушка: имитирует корректную работу этапов и NER."""
    if "Макрон" in text:
        entities = [
            {"text": "Эммануэль Макрон", "label": "PER", "start": 18, "end": 34},
            {"text": "Москву", "label": "LOC", "start": 44, "end": 50},
        ]
    else:
        entities = []
    return _make_trace(
        text=text,
        clean_text=text,
        tokens=text.split(),
        language="ru",
        entities=entities,
    )


def _bad_pipeline(text):
    """Пайплайн-заглушка: имитирует деградацию для проверки penalties."""
    return _make_trace(
        text=text,
        clean_text="x",
        tokens=[],
        language="en",
        entities=[],
    )


def run_full_pipeline_smoke():
    """Сравнивает good/bad pipeline и возвращает оба результата оценки."""
    texts = [
        "Президент Франции Эммануэль Макрон прибыл в Москву",
        "Сегодня хорошая погода",
    ]
    gold_entities = [
        [
            {"text": "Эммануэль Макрон", "label": "PER", "start": 18, "end": 34},
            {"text": "Москву", "label": "LOC", "start": 44, "end": 50},
        ],
        [],
    ]

    config = {
        "matching_mode": "strict",
        "w_f1": 0.8,
        "w_latency": 0.2,
        "reference_latency_ms": 100.0,
        "expected_language": "ru",
        "min_clean_char_ratio": 0.35,
        "min_token_count": 3,
        "max_empty_entity_ratio": 0.8,
        "penalty_weights": {
            "cleaning_overdelete": 0.2,
            "token_count": 0.15,
            "lang_mismatch": 0.25,
            "entity_empty": 0.25,
        },
    }

    good_result = evaluate_full_pipeline(
        texts=texts,
        gold_entities=gold_entities,
        pipeline_fn=_good_pipeline,
        config=config,
    )
    bad_result = evaluate_full_pipeline(
        texts=texts,
        gold_entities=gold_entities,
        pipeline_fn=_bad_pipeline,
        config=config,
    )
    return good_result, bad_result


if __name__ == "__main__":
    good_result, bad_result = run_full_pipeline_smoke()
    print("GOOD:", good_result)
    print("BAD:", bad_result)
    assert "runtime_stages" in good_result["metrics"]
    assert bad_result["score"] < good_result["score"]
