"""Генетический алгоритм подбора гиперпараметров для модуля поиска.

Гены:
    ngram_max     ∈ {1, 2, 3}        — TF-IDF ngram_range = (1, ngram_max)
    min_df        ∈ {1, 2, 3}
    max_df        ∈ {0.80, 0.90, 0.95, 1.00}
    sublinear_tf  ∈ {True, False}
    tfidf_weight  ∈ [0.0, 1.0]       — semantic_weight = 1 - tfidf_weight
    mode          ∈ {TFIDF, BERT, HYBRID}

Fitness: accuracy top-1 на TEST_CASES из validate.py.
Семантическая модель и embeddings документов рассчитываются один раз и переиспользуются.
"""

from __future__ import annotations

import json
import os
import sys
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
from text_search_module_Pyataev.search_module import (  # noqa: E402
    SearchDocument,
    SearchMode,
    TextSearchEngine,
)
from text_search_module_Pyataev.validate import TEST_CASES  # noqa: E402


DOCS_PATH = os.path.join(_HERE, "db", "docs.json")
BEST_PARAMS_PATH = os.path.join(_HERE, "best_params.json")


SEARCH_SPACE = [
    Categorical("ngram_max", [1, 2, 3]),
    Categorical("min_df", [1, 2, 3]),
    Categorical("max_df", [0.80, 0.90, 0.95, 1.00]),
    Categorical("sublinear_tf", [True, False]),
    FloatRange("tfidf_weight", 0.0, 1.0),
    Categorical("mode", ["tfidf", "bert", "hybrid"]),
]


def _load_documents() -> List[SearchDocument]:
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [SearchDocument(**item) for item in data]


# Документы и движок инициализируются один раз — embeddings считаются единожды.
_documents: List[SearchDocument] = []
_engine: TextSearchEngine | None = None


def _get_engine() -> TextSearchEngine:
    global _engine, _documents
    if _engine is None:
        _documents = _load_documents()
        _engine = TextSearchEngine(enable_semantic=True)
        _engine.index_documents(_documents)
    return _engine


def fitness(params: Dict[str, Any]) -> float:
    engine = _get_engine()

    # Переиндексация только TF-IDF — embeddings переиспользуются
    engine.index_documents(
        _documents,
        tfidf_params={
            "ngram_range": (1, int(params["ngram_max"])),
            "min_df": int(params["min_df"]),
            "max_df": float(params["max_df"]),
            "sublinear_tf": bool(params["sublinear_tf"]),
            "norm": "l2",
        },
        reuse_embeddings=True,
    )

    tfidf_w = float(params["tfidf_weight"])
    semantic_w = 1.0 - tfidf_w
    mode = SearchMode(params["mode"])

    correct = 0
    for case in TEST_CASES:
        results = engine.search(
            query=case["query"],
            mode=mode,
            top_k=1,
            min_score=0.0,
            tfidf_weight=tfidf_w,
            semantic_weight=semantic_w,
        )
        predicted_id = results[0].id if results else None
        if predicted_id == case["expected_id"]:
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

    result = optimize(population_size=10, generations=5, on_progress=_print)
    print("=" * 80)
    print(f"BEST fitness={result.best_fitness:.4f}")
    print(f"BEST params={result.best_params}")
    print(f"runtime={result.runtime_s:.1f}s, evals={result.evaluations}")
