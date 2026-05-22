import json
import os

from search_module import TextSearchEngine, SearchDocument, SearchMode


DOCS_PATH = os.path.join(os.path.dirname(__file__), "db", "docs.json")


TEST_CASES = [
    # RU
    {"query": "отпуск сотрудника", "expected_id": "1"},
    {"query": "как получить доступ в систему", "expected_id": "2"},
    {"query": "забыли пароль восстановление", "expected_id": "3"},
    {"query": "ежемесячные отчёты сотрудников", "expected_id": "6"},
    {"query": "удалённая работа правила", "expected_id": "8"},

    # EN
    {"query": "employee onboarding tasks", "expected_id": "4"},
    {"query": "security guidelines system access", "expected_id": "5"},
    {"query": "email corporate access", "expected_id": "7"},

    # смешанные (проверка семантики)
    {"query": "how to reset password", "expected_id": "3"},
    {"query": "company policy remote work", "expected_id": "8"},
]


def load_documents(path: str) -> list[SearchDocument]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return [SearchDocument(**item) for item in data]


def evaluate_mode(
    engine: TextSearchEngine,
    mode: SearchMode,
    test_cases: list[dict],
) -> dict:
    correct = 0
    total = len(test_cases)
    details = []

    for case in test_cases:
        query = case["query"]
        expected_id = case["expected_id"]

        results = engine.search(
            query=query,
            mode=mode,
            top_k=1,
            min_score=0.0
        )

        predicted_id = results[0].id if results else None
        is_correct = predicted_id == expected_id

        if is_correct:
            correct += 1

        details.append({
            "query": query,
            "expected_id": expected_id,
            "predicted_id": predicted_id,
            "correct": is_correct,
            "score": results[0].score if results else None,
        })

    accuracy = correct / total if total else 0.0

    return {
        "mode": mode.value,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "details": details,
    }


def main():
    docs = load_documents(DOCS_PATH)

    engine = TextSearchEngine(enable_semantic=True)
    engine.index_documents(docs)

    modes = [
        SearchMode.TFIDF,
        SearchMode.BERT,
        SearchMode.HYBRID,
    ]

    for mode in modes:
        report = evaluate_mode(engine, mode, TEST_CASES)

        print("=" * 70)
        print(f"Метод: {report['mode']}")
        print(f"Точность: {report['accuracy']:.2%} ({report['correct']}/{report['total']})")

        for item in report["details"]:
            status = "OK" if item["correct"] else "FAIL"
            print(
                f"[{status}] query='{item['query']}' | "
                f"expected={item['expected_id']} | "
                f"predicted={item['predicted_id']} | "
                f"score={item['score']}"
            )


if __name__ == "__main__":
    main()