## Модуль поиска текстовой информации

## Описание

Модуль поиска текстовых данных с использованием NLP.

Поддерживает:

* TF-IDF поиск
* Семантический поиск (BERT / Sentence Transformers)
* Гибридное ранжирование
* Фильтрацию по метаданным

---

## Установка

```bash
pip install -r requirements.txt
```

---

## Применение

```python
from text_search_module import TextSearchEngine, SearchDocument

engine = TextSearchEngine(enable_semantic=True)

docs = [
    SearchDocument(
        id="1",
        text="Пример документа",
        language="ru"
    )
]

engine.index_documents(docs)

results = engine.search("пример")
```

---

## API для интеграции

### Индексация документов

```python
engine.index_documents(documents: List[SearchDocument])
```

### Поиск

```python
engine.search(
    query: str,
    mode: str = "hybrid",
    top_k: int = 10,
    filters: dict = {}
)
```

---

## Примечания

* Если не установлен `sentence-transformers`, используется только TF-IDF
* Все данные хранятся in-memory (без БД)

