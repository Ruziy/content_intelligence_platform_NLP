# Text Processing Module

Модуль отвечает за полный конвейер обработки текста:

1. очистка текста;
2. токенизация;
3. детекция языка;
4. извлечение сущностей (NER);
5. сборка итогового документа.

## Быстрый старт

### Что передавать на вход

В базовом режиме используйте функцию `process_text(text, ...)` из `orchestrator.py`.

- `text` (`str`) - исходный неструктурированный текст.
- `cleaning_method` (`str`, optional) - метод очистки, по умолчанию `"full"`.
- `ner_extractor` (`callable`, optional) - функция извлечения сущностей; если не передана, используется spaCy extractor.

### Что получаем на выходе

`process_text(...)` возвращает словарь `document`:

- `text` - очищенный текст;
- `tokens` - список токенов;
- `entities` - список сущностей (`text`, `label`, при наличии `start/end`);
- `language` - определенный язык.

Пример структуры:

```python
{
    "text": "президент франции эммануэль макрон прибыл москву",
    "tokens": ["президент", "франции", "эммануэль", "макрон", "прибыл", "москву"],
    "entities": [
        {"text": "Эммануэль Макрон", "label": "PER", "start": 18, "end": 34},
        {"text": "Москву", "label": "LOC", "start": 44, "end": 50}
    ],
    "language": "ru"
}
```

## Пример использования в другом модуле (sentiment)

Сентимент-модуль ожидает строку, поэтому передаем в него `document["text"]`:

```python
from source.v2.rus_text_platform.text_processing.orchestrator import process_text
from source.v2.rus_text_platform.sentiment.models import analyze_sentiment

raw_text = "Президент Франции Эммануэль Макрон прибыл в Москву"
doc = process_text(raw_text)
sentiment = analyze_sentiment(doc["text"], model_key="rubert_tiny")

print(doc)
print(sentiment)
```

## Расширенный режим для оценки и отладки

Если нужен доступ к промежуточным данным и времени этапов, используйте `process_text_with_trace(...)`.

На выходе:

- `clean_text`, `tokens`, `language`, `entities`;
- `document` (итог как в `process_text`);
- `stage_timings` (`cleaning_ms`, `tokenization_ms`, `language_ms`, `ner_ms`, `formatting_ms`, `total_ms`).

## Пример для генетического алгоритма (ГА)

Идея: каждый индивид ГА задает конфиг pipeline, а fitness вычисляется через `evaluate_full_pipeline(...)`.

```python
from source.v2.rus_text_platform.text_processing.orchestrator import process_text_with_trace
from source.v2.rus_text_platform.text_processing.evaluation.ner_evaluator import evaluate_full_pipeline


def build_pipeline_fn(individual):
    def pipeline_fn(text):
        return process_text_with_trace(
            text=text,
            cleaning_method=individual["cleaning_method"],
            ner_extractor=individual["ner_extractor"],
        )
    return pipeline_fn


def fitness(individual, texts, gold_entities, eval_config):
    pipeline_fn = build_pipeline_fn(individual)
    result = evaluate_full_pipeline(
        texts=texts,
        gold_entities=gold_entities,
        pipeline_fn=pipeline_fn,
        config=eval_config,
    )
    return result["score"], result
```

`score` используйте как fitness, а `result["penalties"]` и `result["metrics"]` - для диагностики и отбора конфигураций.

## Зависимости и установка

Установите зависимости из `source/v2/rus_text_platform/requirements.txt`.

Для spaCy дополнительно установите языковую модель:

```bash
python -m spacy download ru_core_news_sm
```
