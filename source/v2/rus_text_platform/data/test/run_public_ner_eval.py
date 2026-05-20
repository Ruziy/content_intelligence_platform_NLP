from source.v2.rus_text_platform.text_processing.evaluation.dataset_adapter import (
    load_public_ru_ner_dataset,
)
from source.v2.rus_text_platform.text_processing.evaluation.ner_evaluator import (
    evaluate_pipeline,
)
from source.v2.rus_text_platform.text_processing.extraction.ner_transformers import (
    extract_entities_transformer,
)


def extractor(text):
    return extract_entities_transformer(text, model="rubert")


def main():
    texts, gold_entities = load_public_ru_ner_dataset(
        dataset_name="RCC-MSU/russian-ner",
        split="test",
        limit=100,
    )
    result = evaluate_pipeline(
        texts=texts,
        gold_entities=gold_entities,
        extractor=extractor,
        config={
            "matching_mode": "strict",
            "w_f1": 0.8,
            "w_latency": 0.2,
            "reference_latency_ms": 100.0,
        },
    )
    print(result)


if __name__ == "__main__":
    main()
