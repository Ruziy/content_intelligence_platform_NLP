"""Генетический алгоритм подбора гиперпараметров для sentiment-модуля.

Гены:
    model_key          ∈ {rubert_tiny, rubert_base}
    max_length         ∈ {64, 128, 256, 512}
    padding            ∈ {"max_length", "longest"}
    neutral_threshold  ∈ [0.30, 0.95]    — если max-score < threshold, лейбл → NEUTRAL

Fitness: accuracy на TEST_CASES из validate.py.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict

# Импортируем общие модули по абсолютному пути проекта
_HERE = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ga_common import Categorical, FloatRange, GAConfig, GAResult, run_ga  # noqa: E402
from sentiment_Razuvaev_module.validate import (  # noqa: E402
    TEST_CASES,
    models,
    normalize_label,
)


SEARCH_SPACE = [
    Categorical("model_key", ["rubert_tiny", "rubert_base"]),
    Categorical("max_length", [64, 128, 256, 512]),
    Categorical("padding", ["max_length", "longest"]),
    FloatRange("neutral_threshold", 0.30, 0.95),
]


BEST_PARAMS_PATH = os.path.join(_HERE, "best_params.json")


def _predict_label(text: str, params: Dict[str, Any]) -> str:
    """Инференс с порогом уверенности для NEUTRAL."""
    pipe = models[params["model_key"]]["pipeline"]
    raw = pipe(
        text,
        max_length=int(params["max_length"]),
        truncation=True,
        padding=params["padding"],
    )

    # Формат может быть либо [{label, score}] (return_all_scores=False),
    # либо [[{label, score}, ...]] (return_all_scores=True).
    item = raw[0]
    if isinstance(item, list):
        # Берём топ-вероятность
        best = max(item, key=lambda x: x["score"])
        label = best["label"]
        score = best["score"]
    else:
        label = item["label"]
        score = item["score"]

    label = normalize_label(label)
    if score < float(params["neutral_threshold"]) and label != "NEUTRAL":
        label = "NEUTRAL"
    return label


def fitness(params: Dict[str, Any]) -> float:
    correct = 0
    for case in TEST_CASES:
        predicted = _predict_label(case["text"], params)
        expected = normalize_label(case["expected"])
        if predicted == expected:
            correct += 1
    return correct / len(TEST_CASES)


def optimize(
    population_size: int = 10,
    generations: int = 5,
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
