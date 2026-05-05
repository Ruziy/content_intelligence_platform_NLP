from transformers import pipeline


models = {
    "rubert_tiny": {
        "pipeline": pipeline(
            "sentiment-analysis",
            model="cointegrated/rubert-tiny-sentiment-balanced"
        ),
        "hyperparams": {
            "max_length": 128,
            "batch_size": 16,
            "truncation": True,
            "padding": "max_length",
            "return_all_scores": False,
        }
    },
    "rubert_base": {
        "pipeline": pipeline(
            "sentiment-analysis",
            model="blanchefort/rubert-base-cased-sentiment"
        ),
        "hyperparams": {
            "max_length": 256,
            "batch_size": 16,
            "truncation": True,
            "padding": "max_length",
            "return_all_scores": False,
        }
    }
}


TEST_CASES = [
    {
        "text": "Мне очень понравился этот сервис, всё работает быстро и удобно.",
        "expected": "POSITIVE"
    },
    {
        "text": "Отличный результат, я полностью доволен качеством работы.",
        "expected": "POSITIVE"
    },
    {
        "text": "Программа постоянно зависает и вызывает раздражение.",
        "expected": "NEGATIVE"
    },
    {
        "text": "Ужасный интерфейс, ничего не понятно и всё работает плохо.",
        "expected": "NEGATIVE"
    },
    {
        "text": "Документ был загружен в систему и обработан модулем.",
        "expected": "NEUTRAL"
    },
    {
        "text": "Пользователь отправил запрос на обработку текста.",
        "expected": "NEUTRAL"
    },
    {
        "text": "Качество нормальное, но результат мог быть лучше.",
        "expected": "NEUTRAL"
    },
    {
        "text": "Сервис хороший, но иногда работает слишком медленно.",
        "expected": "NEUTRAL"
    }
]


def normalize_label(label: str) -> str:
    label = label.upper()

    mapping = {
        "POS": "POSITIVE",
        "POSITIVE": "POSITIVE",
        "LABEL_2": "POSITIVE",

        "NEG": "NEGATIVE",
        "NEGATIVE": "NEGATIVE",
        "LABEL_0": "NEGATIVE",

        "NEU": "NEUTRAL",
        "NEUTRAL": "NEUTRAL",
        "LABEL_1": "NEUTRAL",
    }

    return mapping.get(label, label)


def analyze_sentiment(text: str, model_key: str):
    pipe = models[model_key]["pipeline"]
    params = models[model_key]["hyperparams"]

    return pipe(
        text,
        max_length=params["max_length"],
        truncation=params["truncation"],
        padding=params["padding"]
    )


def evaluate_model(model_key: str, test_cases: list[dict]) -> dict:
    correct = 0
    total = len(test_cases)
    details = []

    for case in test_cases:
        text = case["text"]
        expected = case["expected"]

        raw_result = analyze_sentiment(text, model_key=model_key)

        predicted_label = normalize_label(raw_result[0]["label"])
        expected_label = normalize_label(expected)

        is_correct = predicted_label == expected_label

        if is_correct:
            correct += 1

        details.append({
            "text": text,
            "expected": expected_label,
            "predicted": predicted_label,
            "score": raw_result[0]["score"],
            "correct": is_correct
        })

    accuracy = correct / total if total else 0.0

    return {
        "model": model_key,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "details": details
    }


def main():
    for model_key in models:
        report = evaluate_model(model_key, TEST_CASES)

        print("=" * 80)
        print(f"Модель: {report['model']}")
        print(f"Точность: {report['accuracy']:.2%} ({report['correct']}/{report['total']})")

        for item in report["details"]:
            status = "OK" if item["correct"] else "FAIL"

            print(
                f"[{status}] expected={item['expected']} | "
                f"predicted={item['predicted']} | "
                f"score={item['score']:.4f}"
            )
            print(f"Текст: {item['text']}")
            print("-" * 80)


if __name__ == "__main__":
    main()