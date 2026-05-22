"""Генетический алгоритм подбора гиперпараметров для text_processing pipeline.

Гены:
    cleaning_method   ∈ {basic, full, remove_urls, remove_stopwords, lemmatize}
    ner_extractor     ∈ {spacy, transformers:rubert, transformers:bert}
    w_f1              ∈ [0.5, 0.9]      — вес качества в fitness
    min_token_count   ∈ {2, 3, 4, 5, 6}  — порог штрафа за мало токенов

Fitness: evaluate_full_pipeline(...).score — F1 - w_latency*latency - penalties.
Gold-датасет: GOLD_SAMPLES (5 предложений с разметкой PER/LOC/ORG).
"""

from __future__ import annotations

import json
import os
import sys
from functools import partial
from typing import Any, Dict, List

_HERE = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ga_common import (  # noqa: E402
    Categorical,
    FloatRange,
    GAConfig,
    GAResult,
    run_ga,
)
from text_processing.evaluation.ner_evaluator import evaluate_full_pipeline  # noqa: E402
from text_processing.extraction.ner_spacy import extract_entities_spacy  # noqa: E402
from text_processing.orchestrator import process_text_with_trace  # noqa: E402


BEST_PARAMS_PATH = os.path.join(_HERE, "best_params.json")


# Минимальный gold-датасет: RU-предложения с разметкой основных сущностей.
GOLD_SAMPLES: List[Dict[str, Any]] = [
    {
        "text": "Президент Франции Эммануэль Макрон прибыл в Москву.",
        "entities": [
            {"text": "Эммануэль Макрон", "label": "PER"},
            {"text": "Москву", "label": "LOC"},
        ],
    },
    {
        "text": "Компания Газпром объявила о новом контракте с Германией.",
        "entities": [
            {"text": "Газпром", "label": "ORG"},
            {"text": "Германией", "label": "LOC"},
        ],
    },
    {
        "text": "Илон Маск посетил завод Tesla в Берлине.",
        "entities": [
            {"text": "Илон Маск", "label": "PER"},
            {"text": "Tesla", "label": "ORG"},
            {"text": "Берлине", "label": "LOC"},
        ],
    },
    {
        "text": "Министр обороны России Сергей Шойгу встретился с делегацией ООН.",
        "entities": [
            {"text": "России", "label": "LOC"},
            {"text": "Сергей Шойгу", "label": "PER"},
            {"text": "ООН", "label": "ORG"},
        ],
    },
    {
        "text": "Сбербанк открыл новое представительство в Санкт-Петербурге.",
        "entities": [
            {"text": "Сбербанк", "label": "ORG"},
            {"text": "Санкт-Петербурге", "label": "LOC"},
        ],
    },
]


SEARCH_SPACE = [
    Categorical(
        "cleaning_method",
        ["basic", "full", "remove_urls", "remove_stopwords", "lemmatize"],
    ),
    Categorical(
        "ner_extractor",
        ["spacy", "transformers:rubert", "transformers:bert"],
    ),
    FloatRange("w_f1", 0.5, 0.9),
    Categorical("min_token_count", [2, 3, 4, 5, 6]),
]


# Ленивая инициализация трансформер-экстракторов — модели тяжёлые, не грузим без нужды.
_transformer_extractors: Dict[str, Any] = {}


def _get_extractor(name: str):
    if name == "spacy":
        return extract_entities_spacy
    if name in _transformer_extractors:
        return _transformer_extractors[name]
    # Лениво импортируем и грузим модель
    from text_processing.extraction.ner_transformers import extract_entities_transformer

    model_key = name.split(":", 1)[1]  # "transformers:rubert" -> "rubert"
    extractor = partial(extract_entities_transformer, model=model_key)
    _transformer_extractors[name] = extractor
    return extractor


def fitness(params: Dict[str, Any]) -> float:
    extractor = _get_extractor(params["ner_extractor"])
    cleaning_method = params["cleaning_method"]

    def pipeline_fn(text: str):
        return process_text_with_trace(
            text=text,
            cleaning_method=cleaning_method,
            ner_extractor=extractor,
        )

    texts = [sample["text"] for sample in GOLD_SAMPLES]
    gold = [sample["entities"] for sample in GOLD_SAMPLES]

    config = {
        "matching_mode": "relaxed",  # span у gold-разметки не указан
        "w_f1": float(params["w_f1"]),
        "w_latency": 1.0 - float(params["w_f1"]),
        "expected_language": "ru",
        "min_token_count": int(params["min_token_count"]),
    }

    result = evaluate_full_pipeline(
        texts=texts,
        gold_entities=gold,
        pipeline_fn=pipeline_fn,
        config=config,
    )
    return float(result["score"])


def optimize(
    population_size: int = 8,
    generations: int = 4,
    seed: int = 42,
    on_progress=None,
) -> GAResult:
    config = GAConfig(
        population_size=population_size,
        generations=generations,
        seed=seed,
    )
    result = run_ga(SEARCH_SPACE, fitness, config, on_progress=on_progress)
    _save_best(result)
    return result


def _save_best(result: GAResult) -> None:
    payload = {
        "best_params": result.best_params,
        "best_fitness": result.best_fitness,
        "evaluations": result.evaluations,
        "runtime_s": result.runtime_s,
        "ga_config": result.ga_config,
        "history": result.history,
    }
    with open(BEST_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_best() -> Dict[str, Any] | None:
    if not os.path.exists(BEST_PARAMS_PATH):
        return None
    with open(BEST_PARAMS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    def _print(entry):
        print(
            f"gen={entry['generation']:>2}  "
            f"best={entry['best_fitness']:.4f}  "
            f"avg={entry['avg_fitness']:.4f}  "
            f"params={entry['best_params']}"
        )

    result = optimize(population_size=8, generations=4, on_progress=_print)
    print("=" * 80)
    print(f"BEST fitness={result.best_fitness:.4f}")
    print(f"BEST params={result.best_params}")
    print(f"runtime={result.runtime_s:.1f}s, evals={result.evaluations}")
