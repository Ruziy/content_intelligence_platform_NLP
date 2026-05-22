import os
import shutil
import json
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, Form, Request, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from sentiment_Razuvaev_module.models import analyze_sentiment
from processing_of_text_documents_Chizhov_module.source.parser_text import extract_text
from text_search_module_Pyataev.search_module import TextSearchEngine, SearchDocument, SearchMode

from ga_jobs import registry as ga_registry


app = FastAPI()

BASE_DIR = os.path.dirname(__file__)
WEB_DIR = os.path.join(BASE_DIR, "web")
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
SEARCH_DB_PATH = os.path.join(
    BASE_DIR, "text_search_module_Pyataev", "db", "docs.json"
)

os.makedirs(UPLOAD_DIR, exist_ok=True)

templates = Jinja2Templates(directory=os.path.join(WEB_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(WEB_DIR, "static")), name="static")

search_engine = TextSearchEngine(enable_semantic=True)

with open(SEARCH_DB_PATH, "r", encoding="utf-8") as f:
    search_data = json.load(f)

search_docs = [SearchDocument(**item) for item in search_data]
search_engine.index_documents(search_docs)

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/process")
async def process_ajax(
    module_type: str = Form(...),
    text: str = Form(""),
    query: str = Form(""),
    file: UploadFile | None = File(None)
):
    try:
        result = {}

        if module_type == "Анализ настроений по текстам":
            if not text.strip():
                return JSONResponse(
                    {"error": "Для анализа настроений нужно ввести текст."},
                    status_code=400
                )

            result = analyze_sentiment(text)

        elif module_type == "Поиск текстовых данных в системе":
            if not query.strip():
                return JSONResponse(
                    {"error": "Для поиска нужно ввести поисковый запрос."},
                    status_code=400
                )

            search_results = search_engine.search(
                query=query,
                mode=SearchMode.HYBRID,
                top_k=1,
                min_score=0.0
            )

            if not search_results:
                result = {
                    "query": query,
                    "found": False,
                    "message": "Подходящий документ не найден."
                }
            else:
                best = search_results[0]
                doc = search_engine.get_document(best.id)

                result = {
                    "query": query,
                    "found": True,
                    "document": {
                        "id": doc.id,
                        "title": doc.title,
                        "text": doc.text,
                    }
                }

        elif module_type == "Обработка текстовых документов":
            if file is None:
                return JSONResponse(
                    {"error": "Для обработки документов нужно загрузить файл."},
                    status_code=400
                )

            original_name = file.filename or "uploaded_file"
            ext = os.path.splitext(original_name)[1].lower()

            allowed_exts = {".txt", ".docx", ".pdf", ".jpg", ".jpeg", ".png"}
            if ext not in allowed_exts:
                return JSONResponse(
                    {
                        "error": (
                            f"Неподдерживаемый тип файла: {ext}. "
                            f"Поддерживаются: {', '.join(sorted(allowed_exts))}"
                        )
                    },
                    status_code=400
                )

            safe_name = f"{uuid4().hex}{ext}"
            saved_path = os.path.join(UPLOAD_DIR, safe_name)

            with open(saved_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            parsed_text = extract_text(saved_path)

            result = {
                "parsed_text": parsed_text
            }

        elif module_type == "Обработка неструктурированных текстовых данных":
            if not text.strip():
                return JSONResponse(
                    {"error": "Для этого модуля нужно ввести текст."},
                    status_code=400
                )

            result = {
                "module": module_type,
                "message": "Здесь будет логика обработки неструктурированных текстовых данных.",
                "source_text": text
            }

        else:
            result = {
                "message": f"Модуль '{module_type}' пока не реализован"
            }

        return JSONResponse(result)

    except Exception as e:
        return JSONResponse(
            {"error": f"Ошибка обработки: {str(e)}"},
            status_code=500
        )


# -----------------------------
# Оптимизация гиперпараметров (ГА)
# -----------------------------


def _resolve_ga_module(module: str):
    """Возвращает функцию optimize и load_best для запрошенного модуля."""
    if module == "sentiment":
        from sentiment_Razuvaev_module import ga as mod
    elif module == "search":
        from text_search_module_Pyataev import ga as mod
    elif module == "text_processing":
        from text_processing import ga as mod
    else:
        return None
    return mod


def _run_ga_job(module: str, job_id: str, population: int, generations: int, seed: int):
    """Запускается в фоне. Прогресс пушится в registry."""
    mod = _resolve_ga_module(module)
    if mod is None:
        ga_registry.update(job_id, status="error", error=f"Unknown module: {module}")
        return

    ga_registry.update(job_id, status="running")

    def on_progress(entry):
        ga_registry.update(
            job_id,
            generation=entry["generation"],
            progress=entry["generation"] / max(1, generations),
            best_fitness=entry["best_fitness"],
            best_params=entry["best_params"],
            history=ga_registry.get(job_id).history + [entry],
        )

    try:
        result = mod.optimize(
            population_size=population,
            generations=generations,
            seed=seed,
            on_progress=on_progress,
        )
        ga_registry.update(
            job_id,
            status="done",
            progress=1.0,
            best_fitness=result.best_fitness,
            best_params=result.best_params,
            finished_at=__import__("time").time(),
            result={
                "best_params": result.best_params,
                "best_fitness": result.best_fitness,
                "evaluations": result.evaluations,
                "runtime_s": result.runtime_s,
                "ga_config": result.ga_config,
                "history": result.history,
            },
        )
    except Exception as exc:  # noqa: BLE001
        ga_registry.update(
            job_id,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            finished_at=__import__("time").time(),
        )


@app.post("/optimize/{module}")
async def start_optimization(
    module: str,
    background_tasks: BackgroundTasks,
    population: int = Form(10),
    generations: int = Form(5),
    seed: int = Form(42),
):
    if _resolve_ga_module(module) is None:
        return JSONResponse(
            {"error": f"Неизвестный модуль: {module}. Доступны: sentiment, search, text_processing."},
            status_code=400,
        )

    job = ga_registry.create(module=module, total_generations=generations)
    background_tasks.add_task(_run_ga_job, module, job.job_id, population, generations, seed)
    return {"job_id": job.job_id, "status": "queued", "module": module}


@app.get("/optimize/{module}/status/{job_id}")
async def optimization_status(module: str, job_id: str):
    job = ga_registry.get(job_id)
    if not job or job.module != module:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return job.to_dict()


@app.get("/optimize/{module}/best")
async def optimization_best(module: str):
    mod = _resolve_ga_module(module)
    if mod is None:
        return JSONResponse({"error": f"Неизвестный модуль: {module}"}, status_code=400)

    best = mod.load_best()
    if best is None:
        return JSONResponse(
            {"error": "Лучшие параметры ещё не рассчитаны. Запустите оптимизацию."},
            status_code=404,
        )
    return best