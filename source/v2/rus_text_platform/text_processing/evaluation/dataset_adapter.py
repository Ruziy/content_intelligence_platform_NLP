import json
from pathlib import Path


def _normalize_hf_entity(entity):
    """Нормализует сущность из формата HuggingFace dataset."""
    return {
        "text": entity.get("text", ""),
        "label": entity.get("label", "UNK"),
        "start": entity.get("start"),
        "end": entity.get("end"),
    }


def load_public_ru_ner_dataset(
    dataset_name="RCC-MSU/russian-ner",
    split="test",
    text_key="text",
    entities_key="entities",
    limit=None,
):
    """
    Загружает публичный русскоязычный NER датасет и приводит к внутреннему контракту.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Для загрузки публичного NER-датасета установите пакет 'datasets': pip install datasets"
        ) from exc

    dataset = load_dataset(dataset_name, split=split)
    texts = []
    gold_entities = []

    for row in dataset:
        text = str(row[text_key])
        entities = [_normalize_hf_entity(entity) for entity in row[entities_key]]
        texts.append(text)
        gold_entities.append(entities)
        if limit is not None and len(texts) >= limit:
            break

    return texts, gold_entities


def save_dataset_snapshot(texts, gold_entities, output_path):
    """Сохраняет тексты и gold-сущности в JSON snapshot для воспроизводимости."""
    payload = []
    for text, entities in zip(texts, gold_entities):
        payload.append(
            {
                "text": text,
                "entities": entities,
            }
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

