# Генетический алгоритм для подбора гиперпараметров

Этот документ описывает, как устроена и работает подсистема оптимизации
гиперпараметров на основе **генетического алгоритма (ГА)** в нашей платформе
обработки русскоязычных текстов. Сначала — теория, потом — конкретная реализация
для каждого из трёх модулей (sentiment, search, text_processing) и интеграция с
FastAPI.

---

## 1. Теоретическая часть

### 1.1. Что такое генетический алгоритм

**Генетический алгоритм** — это эволюционный метод оптимизации, вдохновлённый
естественным отбором. Алгоритм работает с **популяцией** кандидатных решений и
итеративно её улучшает: лучшие особи отбираются, скрещиваются и мутируют,
постепенно сдвигая распределение в сторону более приспособленных индивидов.

ГА особенно полезен там, где:

- пространство поиска **дискретное или смешанное** (категориальные + численные параметры);
- **функция качества (fitness) дорогая** — каждое вычисление требует прогона
  целого пайплайна (например, инференс модели на тестовом наборе);
- **градиент недоступен** — нельзя применить градиентный спуск напрямую;
- нужна **разведка пространства** (exploration), а не локальный минимум.

В нашем случае все три условия выполняются: гиперпараметры моделей — это
например `model_key="rubert_tiny"` или `mode="hybrid"`, fitness — это accuracy
на тестовом наборе после полной инициализации модели, а градиент по выбору
бэкенда не определён.

### 1.2. Базовая терминология

| Термин | Что это в общем виде | Что это у нас |
|---|---|---|
| **Ген** | Один параметр решения | Например, `model_key` или `tfidf_weight` |
| **Индивид (хромосома)** | Полное решение из набора генов | Список из 4–6 значений |
| **Популяция** | Набор индивидов | `population_size=10` особей |
| **Поколение (generation)** | Одна итерация цикла | Шаг эволюции |
| **Fitness (функция приспособленности)** | Качество индивида (число, которое максимизируется) | Accuracy на TEST_CASES |
| **Селекция** | Выбор родителей | Турнирная (берём лучшего из 3 случайных) |
| **Кроссовер** | Скрещивание двух родителей | Равномерный (uniform crossover) |
| **Мутация** | Случайное изменение гена | Категориальный ресэмпл / гауссовский шаг |
| **Элитизм** | Перенос лучшего индивида без изменений | Сохраняем 1 элиту между поколениями |

### 1.3. Псевдокод

```
P ← инициализировать_популяцию(N)               # случайная стартовая популяция
оценить(P)                                       # вычислить fitness каждой особи
for g = 1 ... G:                                 # G поколений
    elites ← лучшие(P, k)                        # элитизм
    parents ← селекция(P, N - k)                 # турнирная селекция
    offspring ← скрещивание(parents, cxpb)       # кроссовер с вероятностью cxpb
    offspring ← мутация(offspring, mutpb)        # мутация с вероятностью mutpb
    оценить(offspring)                           # пересчитать fitness
    P ← elites ∪ offspring                       # новое поколение
вернуть лучшего(P)
```

### 1.4. Почему именно ГА, а не grid/random search

| Подход | Стоимость на N точек | Хорошо находит | Плохо в |
|---|---|---|---|
| Grid search | экспоненциально по числу осей | гарантированно перебирает сетку | смешанных и неравномерных пространствах |
| Random search | O(N), любые оси | равномерно покрывает шары высокой плотности | не "учится" на прошлых пробах |
| Bayesian opt. | O(N), любые оси | очень эффективен при ≤10 параметров | сложно реализовать для категориальных |
| **Генетический алгоритм** | O(N), любые оси | **умеет смешанные пространства**, копит "хорошие схемы" | требует выбрать операторы и популяцию |

В нашем проекте каждый прогон fitness — это полный инференс модели (секунды),
поэтому простой grid search невозможен (комбинаций сотни). ГА хорошо подходит,
потому что:

1. Большинство наших генов — категориальные (выбор модели, режим поиска, метод очистки).
2. Мы можем себе позволить только ~50–80 вычислений fitness.
3. У ГА есть полезное свойство **building blocks** — он быстро собирает "хорошие"
   подкомбинации генов в один индивид.

---

## 2. Архитектура реализации

### 2.1. Общая структура

```
source/v2/rus_text_platform/
├── ga_common.py                         # ядро ГА: типы генов, run_ga, конфиг
├── ga_jobs.py                           # in-memory реестр фоновых задач
├── main.py                              # FastAPI endpoints /optimize/...
├── sentiment_Razuvaev_module/
│   ├── ga.py                            # SEARCH_SPACE + fitness для sentiment
│   └── best_params.json                 # сохранённый результат
├── text_search_module_Pyataev/
│   ├── ga.py                            # SEARCH_SPACE + fitness для поиска
│   └── best_params.json
└── text_processing/
    ├── ga.py                            # SEARCH_SPACE + fitness для NER-пайплайна
    └── best_params.json
```

**Идея разделения:** ядро ГА (`ga_common.py`) ничего не знает про NLP, оно
универсальное. Каждый модуль декларирует своё пространство поиска и свою
fitness-функцию — это всё, что нужно для запуска.

### 2.2. Ядро [ga_common.py](source/v2/rus_text_platform/ga_common.py)

Реализовано поверх библиотеки **DEAP** (Distributed Evolutionary Algorithms in Python).

#### Типы генов

```python
@dataclass
class Categorical:                  # выбор из списка вариантов
    name: str
    choices: List[Any]              # ["tfidf", "bert", "hybrid"]

@dataclass
class IntRange:                     # целое в [low, high]
    name: str
    low: int
    high: int

@dataclass
class FloatRange:                   # вещественное в [low, high]
    name: str
    low: float
    high: float
```

Внутренне индивид хранится как `list[float]` фиксированной длины; декодирование
во "внешний" словарь параметров происходит перед каждым вызовом fitness через
`decode_individual()`. Это позволяет DEAP-овским операторам (cxUniform, etc.)
работать одинаково над всеми типами.

#### Операторы

| Оператор | Реализация | Зачем |
|---|---|---|
| **Init** | каждый ген сэмплируется независимо по своему `sample()` | равномерное стартовое покрытие |
| **Crossover** | `tools.cxUniform(indpb=0.5)` — каждый ген с p=0.5 обменивается | хорошо работает на смешанных хромосомах |
| **Mutation** | категориальный: ресэмпл; float: гауссовский шаг σ=0.1·(high-low); int: ±1 с p=0.5 или ресэмпл | разные типы — разная "локальность" |
| **Selection** | `tools.selTournament(tournsize=3)` — берём лучшего из 3 случайных | мягкое селективное давление |
| **Elitism** | `tools.selBest(population, 1)` копируется в следующее поколение | гарантирует, что лучший не теряется |

#### Конфиг по умолчанию

```python
@dataclass
class GAConfig:
    population_size: int = 10      # размер популяции
    generations: int = 5           # количество поколений
    cxpb: float = 0.6              # вероятность кроссовера для пары
    mutpb: float = 0.3             # вероятность мутации для особи
    tournament_size: int = 3       # размер турнира
    elitism: int = 1               # сколько элит сохранять
    seed: Optional[int] = 42       # для воспроизводимости
```

При `population_size=10` и `generations=5` получаем `10 + 5·9 ≈ 55` вычислений
fitness (элита переоценкой не пересчитывается).

#### Главный цикл — `run_ga()`

```python
def run_ga(search_space, fitness_fn, config, on_progress=None) -> GAResult:
    population = init(N)
    оценить(population)
    _report(0)                                  # колбэк on_progress

    for gen in 1..G:
        elites = selBest(population, k)
        offspring = selTournament(population, N - k)
        offspring = cxUniform(offspring, cxpb)
        offspring = mutate(offspring, mutpb)
        оценить(offspring)
        population = elites + offspring
        _report(gen)
    return GAResult(best_params, best_fitness, history, ...)
```

Колбэк `on_progress(entry)` вызывается после каждого поколения и содержит:

```python
{
    "generation": int,
    "best_fitness": float,
    "avg_fitness": float,
    "best_params": dict,
    "evaluations": int,
}
```

Этот же словарь складывается в `history` итогового `GAResult`. Именно через
этот колбэк FastAPI стримит прогресс в браузер (см. §4).

---

## 3. Реализация для конкретных модулей

Каждый модуль декларирует **только** свой `SEARCH_SPACE` и `fitness()`. Ниже —
расшифровка генов и метрики качества для каждого из них.

### 3.1. Sentiment — анализ тональности

Файл: [sentiment_Razuvaev_module/ga.py](source/v2/rus_text_platform/sentiment_Razuvaev_module/ga.py)

#### Пространство поиска

| Ген | Тип | Значения | Что регулирует |
|---|---|---|---|
| `model_key` | Categorical | `rubert_tiny`, `rubert_base` | Какая модель из HuggingFace используется |
| `max_length` | Categorical | `64, 128, 256, 512` | Максимальная длина токенов на входе |
| `padding` | Categorical | `max_length`, `longest` | Стратегия паддинга батча |
| `neutral_threshold` | FloatRange | `[0.30, 0.95]` | Порог уверенности: ниже → лейбл переводим в `NEUTRAL` |

Размер пространства: `2 × 4 × 2 × ∞ = 16 × континуум`. Для перебора грубой
сеткой по `neutral_threshold` (шаг 0.05) → 14 × 16 = 224 точек, что заметно
больше нашего бюджета в ~55 запусков. ГА справляется лучше.

#### Fitness

Accuracy на тестовом наборе из `validate.py` (~50 предложений с известным
лейблом). На каждом шаге:

1. Берём пайплайн для выбранной модели (`models[model_key]["pipeline"]`).
2. Прогоняем его на каждом тестовом тексте с заданными `max_length`/`padding`.
3. Если `score < neutral_threshold` → переписываем лейбл в `NEUTRAL` (логика
   "не уверен — значит нейтрально").
4. Возвращаем `correct / len(TEST_CASES)`.

#### Зачем нужен `neutral_threshold`

Модели `rubert_tiny` и `rubert_base` обучены на 2-3 классах с разной
калибровкой. ГА сам подбирает порог, при котором "размытые" предсказания одной
модели начинают совпадать с разметкой `validate.py` через NEUTRAL. Это даёт
ощутимый прирост accuracy без дообучения.

### 3.2. Search — поиск по корпусу документов

Файл: [text_search_module_Pyataev/ga.py](source/v2/rus_text_platform/text_search_module_Pyataev/ga.py)

#### Пространство поиска

| Ген | Тип | Значения | Что регулирует |
|---|---|---|---|
| `ngram_max` | Categorical | `1, 2, 3` | TF-IDF `ngram_range=(1, ngram_max)` |
| `min_df` | Categorical | `1, 2, 3` | Минимальная частотность токена |
| `max_df` | Categorical | `0.80, 0.90, 0.95, 1.00` | Доля документов с термом, выше которой он игнорируется |
| `sublinear_tf` | Categorical | `True, False` | Сабленейный TF (1+log) |
| `tfidf_weight` | FloatRange | `[0.0, 1.0]` | Вес TF-IDF в гибридной формуле (BERT-вес = 1 − tfidf_weight) |
| `mode` | Categorical | `tfidf, bert, hybrid` | Какой бэкенд использовать |

#### Fitness

Accuracy top-1 на `TEST_CASES` из `validate.py` (вопрос → ожидаемый
`expected_id` документа). Для каждого запроса берём `top_k=1` и проверяем
совпадение `id`.

#### Хитрость с переиспользованием эмбеддингов

BERT-эмбеддинги документов считаются **один раз** при инициализации движка.
Дальше `index_documents(..., reuse_embeddings=True)` пересоздаёт только TF-IDF
(быстро), а семантические эмбеддинги переиспользует. Это **критично** для
скорости — иначе каждая особь грузила бы `sentence-transformers` модель и
считала эмбеддинги 18 документов заново, что превратило бы 55 запусков в 10+
минут.

### 3.3. Text Processing — пайплайн NER

Файл: [text_processing/ga.py](source/v2/rus_text_platform/text_processing/ga.py)

#### Пространство поиска

| Ген | Тип | Значения | Что регулирует |
|---|---|---|---|
| `cleaning_method` | Categorical | `basic, full, remove_urls, remove_stopwords, lemmatize` | Способ предобработки текста |
| `ner_extractor` | Categorical | `spacy, transformers:rubert, transformers:bert` | Какой экстрактор сущностей |
| `w_f1` | FloatRange | `[0.5, 0.9]` | Вес F1 в итоговой метрике; `w_latency = 1 − w_f1` |
| `min_token_count` | Categorical | `2, 3, 4, 5, 6` | Порог штрафа за слишком короткие тексты |

#### Fitness

Используется `evaluate_full_pipeline()` из `text_processing/evaluation/ner_evaluator.py`.
Итоговая метрика:

```
score = w_f1 · F1 − w_latency · normalized_latency − penalties
```

где:

- **F1** — стандартный F1 по совпадениям предсказанных сущностей с
  `GOLD_SAMPLES` (5 предложений с разметкой PER/LOC/ORG);
- **normalized_latency** — относительная задержка пайплайна;
- **penalties** — штрафы за нарушение `min_token_count` и других проверок.

Используется `matching_mode="relaxed"` — мы сравниваем сущности по тексту, а не
по точным offset'ам, потому что в gold-разметке span'ы не указаны.

#### Ленивая инициализация трансформеров

Модели `rubert` и `bert` для NER — тяжёлые (>400 MB). Чтобы не грузить их без
необходимости, в `_get_extractor()` стоит кэш `_transformer_extractors`,
который инициализирует модель только при первом обращении.

---

## 4. Интеграция с FastAPI

### 4.1. Endpoints

Реализованы в [main.py](source/v2/rus_text_platform/main.py):

| Метод | URL | Что делает |
|---|---|---|
| `POST` | `/optimize/{module}` | Запускает ГА в фоне, возвращает `job_id` |
| `GET` | `/optimize/{module}/status/{job_id}` | Текущий статус, прогресс, история |
| `GET` | `/optimize/{module}/best` | Сохранённые лучшие параметры (`best_params.json`) |

Где `{module}` ∈ {`sentiment`, `search`, `text_processing`}.

### 4.2. Фоновое выполнение

ГА — операция **долгая** (десятки секунд — минуты) и **CPU/GPU-bound**.
Поэтому она вынесена в `BackgroundTasks`:

```python
@app.post("/optimize/{module}")
async def start_optimization(module, background_tasks, population, generations, seed):
    job = ga_registry.create(module=module, total_generations=generations)
    background_tasks.add_task(_run_ga_job, module, job.job_id, population, generations, seed)
    return {"job_id": job.job_id, "status": "queued", "module": module}
```

Функция `_run_ga_job()`:

1. Динамически импортирует `ga.py` нужного модуля через `_resolve_ga_module()`.
2. Регистрирует колбэк `on_progress`, который после каждого поколения
   пушит в `ga_registry` запись: `generation`, `progress`, `best_fitness`,
   `best_params`, `history`.
3. По завершении — переводит job в статус `done` и сохраняет полный результат;
   при ошибке — `error` с `type(exc).__name__: exc`.

### 4.3. Реестр задач — [ga_jobs.py](source/v2/rus_text_platform/ga_jobs.py)

In-memory словарь `{job_id: JobState}` под `threading.Lock`. JobState содержит:

```python
status: queued | running | done | error
progress: float                # 0.0 .. 1.0
generation: int
best_fitness: float
best_params: dict
history: list                  # последовательность on_progress entries
result: dict                   # финальный GAResult, когда status == done
error: str                     # сообщение, когда status == error
```

Реестр хранится **только в памяти процесса** — после рестарта сервера активные
задачи теряются, но `best_params.json` каждого модуля сохраняется на диск.

### 4.4. UI — лайв-прогресс

В [web/templates/index.html](source/v2/rus_text_platform/web/templates/index.html)
добавлена секция оптимизации. JS-код:

1. На клик "Запустить ГА" → `POST /optimize/{module}` → получает `job_id`.
2. Запускает `setInterval(...)` с шагом 1 сек, на каждом тике делает
   `GET /optimize/{module}/status/{job_id}`.
3. Рендерит прогресс-бар (`generation / total_generations`), текущий лучший
   fitness и параметры.
4. На статусе `done` или `error` — останавливает поллинг и показывает финал.

---

## 5. Как пользоваться

### 5.1. Через UI

1. Открыть `http://127.0.0.1:8000/`.
2. Спуститься к секции "Оптимизация гиперпараметров (ГА)".
3. Выбрать модуль, задать `population` и `generations` (по умолчанию 10 / 5).
4. Нажать "Запустить ГА" — увидеть прогресс в реальном времени.
5. После завершения — нажать "Показать лучшие", чтобы увидеть `best_params.json`.

### 5.2. Через curl

```bash
# Запуск
curl -X POST http://127.0.0.1:8000/optimize/sentiment \
     -F "population=10" -F "generations=5" -F "seed=42"
# → {"job_id": "...", "status": "queued", "module": "sentiment"}

# Опрос статуса
curl http://127.0.0.1:8000/optimize/sentiment/status/<job_id>

# Лучшие параметры
curl http://127.0.0.1:8000/optimize/sentiment/best
```

### 5.3. Локально, без сервера

Каждый `ga.py` запускается как самостоятельный скрипт и логирует прогресс
в stdout:

```bash
python -m sentiment_Razuvaev_module.ga
python -m text_search_module_Pyataev.ga
python -m text_processing.ga
```

---

## 6. Известные ограничения и пути расширения

| Что | Как сейчас | Куда можно развить |
|---|---|---|
| **Параллелизм** | Особи оцениваются последовательно | DEAP поддерживает `toolbox.register("map", multiprocessing.Pool.map)` |
| **Хранение задач** | In-memory, теряется при рестарте | Redis / SQLite |
| **Метрика поиска** | Только top-1 accuracy | MRR, NDCG на разметке релевантности |
| **Gold-датасет NER** | 5 предложений, hardcoded | Внешний файл, расширение до 100+ примеров |
| **Sentiment модели** | Только 2 модели в выборке | Добавить `xlm-roberta`, `dostoevsky` |
| **Early stopping** | Не реализован | Останавливать при стагнации best_fitness на K поколений |

---

## 7. Источники

- DEAP documentation: https://deap.readthedocs.io/
- Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*.
- Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*.
- FastAPI BackgroundTasks: https://fastapi.tiangolo.com/tutorial/background-tasks/
