"""Модуль поиска текстовой информации.

Свойства:
- NLP-препроцессинг
- индексация документов
- поиск по TF-IDF
- семантический поиск с применением Sentence-Transformers / BERT-эмбеддингов
- оценка релевантности и фильтрация
- интеграция с FastAPI

Зависимости:
    pip install numpy scikit-learn pydantic
    # Опционально, для семантического поиска:
    pip install sentence-transformers
    # Опционально, для API:
    pip install fastapi uvicorn
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - опциональная зависимость
    SentenceTransformer = None

try:
    from fastapi import APIRouter, HTTPException
except Exception:  # pragma: no cover - опциональная зависимость
    APIRouter = None
    HTTPException = None


# -----------------------------
# МОДЕЛИ ДАННЫХ
# -----------------------------


class SearchMode(str, Enum):
    TFIDF = "tfidf"
    BERT = "bert"
    HYBRID = "hybrid"


class SearchDocument(BaseModel):
    """Представление документа, используемого в индексе."""

    id: str = Field(..., description="Уникальный идентификатор документа")
    text: str = Field(..., description="Текст документа")
    title: Optional[str] = None
    source: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    language: Optional[str] = None
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    query: str
    mode: SearchMode = SearchMode.HYBRID
    top_k: int = Field(default=10, ge=1, le=100)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    filters: Dict[str, Any] = Field(default_factory=dict)
    # Пример фильтров: {"source": "kb", "language": "ru", "tags_any": ["policy"], "tags_all": ["hr", "2026"]}


class SearchHit(BaseModel):
    id: str
    title: Optional[str] = None
    source: Optional[str] = None
    score: float
    mode_score: Dict[str, float] = Field(default_factory=dict)
    snippet: str
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IndexStats(BaseModel):
    documents_count: int
    tfidf_vocab_size: int
    has_semantic_index: bool
    last_indexed_at: Optional[datetime] = None


# -----------------------------
# NLP ПРЕДОБРАБОТКА
# -----------------------------


class TextPreprocessor:
    """Простой NLP-препроцессинг без тяжёлых зависимостей."""

    _token_pattern = re.compile(r"\w+", re.UNICODE)

    def __init__(self, extra_stopwords: Optional[Iterable[str]] = None) -> None:
        self.stopwords = self._build_stopwords(extra_stopwords)

    @staticmethod
    def _build_stopwords(extra_stopwords: Optional[Iterable[str]]) -> set[str]:
        # В sklearn стоп-слова только для английского. Добавляем распространённые слова.
        base = set(TfidfVectorizer(stop_words="english").get_stop_words() or [])
        base |= {
            "http", "https", "www", "com", "ru", "org", "net",
            "это", "как", "для", "или", "что", "так", "при", "без",
            "with", "from", "your", "you", "are", "was", "were",
        }
        if extra_stopwords:
            base |= {w.strip().lower() for w in extra_stopwords if w}
        return base

    @staticmethod
    def _strip_accents(text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text)
        return "".join(ch for ch in normalized if not unicodedata.combining(ch))

    def normalize(self, text: str) -> str:
        text = text or ""
        text = self._strip_accents(text)
        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        text = self.normalize(text)
        tokens = self._token_pattern.findall(text)
        tokens = [t for t in tokens if len(t) > 1 and t not in self.stopwords]
        return tokens

    def preprocess_for_tfidf(self, text: str) -> str:
        return " ".join(self.tokenize(text))

    def build_snippet(self, text: str, query: str, max_len: int = 240) -> str:
        """Формирование читаемого фрагмента текста вокруг запроса."""
        if not text:
            return ""

        clean_text = re.sub(r"\s+", " ", text).strip()
        q_tokens = [t for t in self.tokenize(query) if len(t) >= 2]
        if not q_tokens:
            return clean_text[:max_len].rstrip()

        lowered = clean_text.lower()
        positions = [lowered.find(tok) for tok in q_tokens if lowered.find(tok) >= 0]
        if not positions:
            return clean_text[:max_len].rstrip()

        start = max(0, min(positions) - max_len // 3)
        end = min(len(clean_text), start + max_len)
        snippet = clean_text[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(clean_text):
            snippet = snippet + "..."
        return snippet


# -----------------------------
# ОСНОВНОЙ ДВИЖОК
# -----------------------------


@dataclass
class IndexedCorpus:
    documents: List[SearchDocument] = field(default_factory=list)
    processed_texts: List[str] = field(default_factory=list)
    tfidf_matrix: Any = None
    tfidf_vectorizer: Optional[TfidfVectorizer] = None
    embeddings: Optional[np.ndarray] = None
    semantic_model_name: Optional[str] = None
    last_indexed_at: Optional[datetime] = None


class TextSearchEngine:
    """Гибридный поисковый движок.

    Порядок работы:
    1. применение фильтров
    2. расчёт TF-IDF
    3. расчёт семантического (BERT) сходства
    4. объединение результатов
    """

    def __init__(
        self,
        semantic_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        enable_semantic: bool = True,
        extra_stopwords: Optional[Iterable[str]] = None,
    ) -> None:
        self.preprocessor = TextPreprocessor(extra_stopwords=extra_stopwords)
        self.corpus = IndexedCorpus()
        self.enable_semantic = enable_semantic and SentenceTransformer is not None
        self.semantic_model_name = semantic_model_name
        self.semantic_model = None
        if self.enable_semantic:
            self.semantic_model = SentenceTransformer(semantic_model_name)
            self.corpus.semantic_model_name = semantic_model_name

    # -------- индексация --------

    def index_documents(self, documents: Sequence[SearchDocument]) -> IndexStats:
        docs = list(documents)

        # Предобработка текста документов для TF-IDF
        processed = [self.preprocessor.preprocess_for_tfidf(d.text) for d in docs]

        # Создание TF-IDF векторизатора
        vectorizer = TfidfVectorizer(
            min_df=1,              # минимальная частота слова
            max_df=0.95,           # игнорировать слишком частые слова
            ngram_range=(1, 2),    # учитывать униграммы и биграммы
            sublinear_tf=True,     # логарифмическое масштабирование TF
            norm="l2",             # нормализация
        )

        # Построение TF-IDF матрицы
        tfidf_matrix = vectorizer.fit_transform(processed) if processed else None

        embeddings = None

        # Построение эмбеддингов (если включён семантический поиск)
        if self.enable_semantic and docs:
            texts_for_semantic = [f"{d.title or ''} {d.text}".strip() for d in docs]

            embeddings = self.semantic_model.encode(
                texts_for_semantic,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            embeddings = np.asarray(embeddings, dtype=np.float32)

        # Сохранение всех данных в корпусе
        self.corpus = IndexedCorpus(
            documents=docs,
            processed_texts=processed,
            tfidf_matrix=tfidf_matrix,
            tfidf_vectorizer=vectorizer,
            embeddings=embeddings,
            semantic_model_name=self.semantic_model_name if self.enable_semantic else None,
            last_indexed_at=datetime.utcnow(),
        )

        # Возвращаем статистику индекса
        return IndexStats(
            documents_count=len(docs),
            tfidf_vocab_size=len(vectorizer.vocabulary_) if vectorizer else 0,
            has_semantic_index=embeddings is not None,
            last_indexed_at=self.corpus.last_indexed_at,
        )

    # -------- фильтрация --------

    def _passes_filters(self, doc: SearchDocument, filters: Dict[str, Any]) -> bool:
        # Если фильтров нет — документ проходит
        if not filters:
            return True

        for key, value in filters.items():
            if value is None:
                continue

            # Фильтр по источнику
            if key == "source" and doc.source != value:
                return False

            # Фильтр по языку
            if key == "language" and doc.language != value:
                return False

            # Фильтр по ID
            if key == "id" and doc.id != value:
                return False

            # Проверка вхождения строки в заголовок
            if key == "title_contains":
                title = (doc.title or "").lower()
                if str(value).lower() not in title:
                    return False

            # Проверка вхождения строки в текст
            if key == "text_contains":
                if str(value).lower() not in doc.text.lower():
                    return False

            # Проверка: есть ли хотя бы один тег
            if key == "tags_any":
                wanted = {str(x).lower() for x in value}
                have = {str(x).lower() for x in doc.tags}
                if wanted.isdisjoint(have):
                    return False

            # Проверка: есть ли все теги
            if key == "tags_all":
                wanted = {str(x).lower() for x in value}
                have = {str(x).lower() for x in doc.tags}
                if not wanted.issubset(have):
                    return False

            # Фильтр по metadata (точное совпадение)
            if key == "metadata":
                if not isinstance(value, dict):
                    continue
                for mk, mv in value.items():
                    if doc.metadata.get(mk) != mv:
                        return False

        return True

    def _filtered_indices(self, filters: Dict[str, Any]) -> List[int]:
        # Возвращает индексы документов, прошедших фильтры
        return [i for i, d in enumerate(self.corpus.documents) if self._passes_filters(d, filters)]

    # -------- вычисление релевантности --------

    @staticmethod
    def _minmax_normalize(scores: np.ndarray) -> np.ndarray:
        # Нормализация значений в диапазон [0, 1]
        if scores.size == 0:
            return scores

        mn = float(np.min(scores))
        mx = float(np.max(scores))

        # Если все значения одинаковые — возвращаем единицы
        if math.isclose(mx, mn):
            return np.ones_like(scores, dtype=np.float32)

        return ((scores - mn) / (mx - mn)).astype(np.float32)

    def _tfidf_scores(self, query: str, indices: List[int]) -> np.ndarray:
        # Расчёт релевантности через TF-IDF
        if not indices or self.corpus.tfidf_matrix is None or self.corpus.tfidf_vectorizer is None:
            return np.array([], dtype=np.float32)

        q_processed = self.preprocessor.preprocess_for_tfidf(query)

        # Вектор запроса
        q_vec = self.corpus.tfidf_vectorizer.transform([q_processed])

        # Матрица документов
        mat = self.corpus.tfidf_matrix[indices]

        # Косинусное сходство
        scores = cosine_similarity(q_vec, mat).ravel().astype(np.float32)

        return scores

    def _semantic_scores(self, query: str, indices: List[int]) -> np.ndarray:
        # Расчёт релевантности через BERT
        if not indices or not self.enable_semantic or self.corpus.embeddings is None:
            return np.array([], dtype=np.float32)

        query_text = query.strip()

        # Эмбеддинг запроса
        q_emb = self.semantic_model.encode(
            [query_text], normalize_embeddings=True, show_progress_bar=False
        )

        q_emb = np.asarray(q_emb, dtype=np.float32)

        # Эмбеддинги документов
        doc_emb = self.corpus.embeddings[indices]

        # Косинусное сходство
        scores = cosine_similarity(q_emb, doc_emb).ravel().astype(np.float32)

        return scores

    def search(
        self,
        query: str,
        mode: SearchMode = SearchMode.HYBRID,
        top_k: int = 10,
        min_score: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
        tfidf_weight: float = 0.45,
        semantic_weight: float = 0.55,
    ) -> List[SearchHit]:

        # Если документов нет — возвращаем пустой список
        if not self.corpus.documents:
            return []

        filters = filters or {}

        # Получаем индексы подходящих документов
        indices = self._filtered_indices(filters)
        if not indices:
            return []

        # Считаем оба типа релевантности
        tfidf_scores = self._tfidf_scores(query, indices)
        semantic_scores = self._semantic_scores(query, indices)

        # Нормализация оценок
        if tfidf_scores.size > 0:
            tfidf_scores = self._minmax_normalize(tfidf_scores)

        if semantic_scores.size > 0:
            semantic_scores = self._minmax_normalize(semantic_scores)

        # Выбор режима поиска
        if mode == SearchMode.TFIDF or semantic_scores.size == 0:
            final_scores = tfidf_scores
            mode_scores_name = "tfidf"

        elif mode == SearchMode.BERT:
            final_scores = semantic_scores
            mode_scores_name = "bert"

        else:
            # Гибридный режим
            if tfidf_scores.size == 0:
                final_scores = semantic_scores
            elif semantic_scores.size == 0:
                final_scores = tfidf_scores
            else:
                final_scores = tfidf_weight * tfidf_scores + semantic_weight * semantic_scores

            mode_scores_name = "hybrid"

        # Сортировка по убыванию релевантности
        ranked = np.argsort(-final_scores)

        hits: List[SearchHit] = []

        for rank_pos in ranked[:top_k]:
            score = float(final_scores[rank_pos])

            # Отсечение по минимальному порогу
            if score < min_score:
                continue

            doc = self.corpus.documents[indices[rank_pos]]

            # Формируем сниппет
            snippet = self.preprocessor.build_snippet(doc.text, query)

            hit = SearchHit(
                id=doc.id,
                title=doc.title,
                source=doc.source,
                score=score,
                mode_score={
                    mode_scores_name: score,
                    "tfidf": float(tfidf_scores[rank_pos]) if tfidf_scores.size else 0.0,
                    "bert": float(semantic_scores[rank_pos]) if semantic_scores.size else 0.0,
                },
                snippet=snippet,
                tags=doc.tags,
                metadata=doc.metadata,
            )

            hits.append(hit)

        return hits

    def get_stats(self) -> IndexStats:
        # Получение статистики по индексу
        vocab_size = 0
        if self.corpus.tfidf_vectorizer is not None:
            vocab_size = len(self.corpus.tfidf_vectorizer.vocabulary_)

        return IndexStats(
            documents_count=len(self.corpus.documents),
            tfidf_vocab_size=vocab_size,
            has_semantic_index=self.corpus.embeddings is not None,
            last_indexed_at=self.corpus.last_indexed_at,
        )

    # -------- вспомогательные методы --------

    def upsert_documents(self, documents: Sequence[SearchDocument]) -> IndexStats:
        # Обновление или добавление документов
        by_id: Dict[str, SearchDocument] = {d.id: d for d in self.corpus.documents}

        for doc in documents:
            by_id[doc.id] = doc

        return self.index_documents(list(by_id.values()))

    def delete_document(self, doc_id: str) -> IndexStats:
        # Удаление документа по ID
        docs = [d for d in self.corpus.documents if d.id != doc_id]
        return self.index_documents(docs)

    def get_document(self, doc_id: str) -> Optional[SearchDocument]:
        # Получение документа по ID
        for d in self.corpus.documents:
            if d.id == doc_id:
                return d
        return None


# -----------------------------
# FastAPI router
# -----------------------------


def create_search_router(engine: TextSearchEngine):
    """Создание FastAPI-роутера для поискового движка."""

    if APIRouter is None:
        raise RuntimeError("fastapi не установлен. Установите fastapi для использования API.")

    router = APIRouter(prefix="/search", tags=["search"])

    class IndexPayload(BaseModel):
        documents: List[SearchDocument]

    class UpsertPayload(BaseModel):
        documents: List[SearchDocument]

    class DeletePayload(BaseModel):
        doc_id: str

    @router.post("/index", response_model=IndexStats)
    def index_documents(payload: IndexPayload):
        return engine.index_documents(payload.documents)

    @router.post("/upsert", response_model=IndexStats)
    def upsert_documents(payload: UpsertPayload):
        return engine.upsert_documents(payload.documents)

    @router.post("/delete", response_model=IndexStats)
    def delete_document(payload: DeletePayload):
        return engine.delete_document(payload.doc_id)

    @router.post("/query", response_model=List[SearchHit])
    def query(payload: SearchRequest):
        if not engine.corpus.documents:
            if HTTPException is None:
                raise RuntimeError("Нет проиндексированных документов")

            raise HTTPException(status_code=400, detail="Документы не загружены")

        return engine.search(
            query=payload.query,
            mode=payload.mode,
            top_k=payload.top_k,
            min_score=payload.min_score,
            filters=payload.filters,
        )

    @router.get("/stats", response_model=IndexStats)
    def stats():
        return engine.get_stats()

    return router


# -----------------------------
# Пример использования
# -----------------------------


if __name__ == "__main__":
    docs = [
        SearchDocument(
            id="1",
            title="Политика отпусков",
            text="Сотрудник имеет право на ежегодный оплачиваемый отпуск согласно трудовому кодексу.",
            source="hr",
            tags=["policy", "hr"],
            language="ru",
            metadata={"department": "hr"},
        ),
        SearchDocument(
            id="2",
            title="Инструкция по доступу",
            text="Доступ в систему предоставляется по заявке после согласования с руководителем.",
            source="it",
            tags=["security", "it"],
            language="ru",
            metadata={"department": "it"},
        ),
        SearchDocument(
            id="3",
            title="Onboarding checklist",
            text="New employees should complete onboarding tasks within the first week.",
            source="kb",
            tags=["onboarding", "employees"],
            language="en",
            metadata={"department": "hr"},
        ),
    ]

    engine = TextSearchEngine(enable_semantic=False)  # становится True если sentence-transformers установлены
    print(engine.index_documents(docs))
    results = engine.search(
        query="как получить доступ в систему",
        mode=SearchMode.HYBRID,
        top_k=5,
        filters={"language": "ru"},
    )
    for r in results:
        print(r.model_dump())
