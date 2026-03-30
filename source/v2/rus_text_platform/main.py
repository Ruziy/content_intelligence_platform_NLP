import os
import shutil
from uuid import uuid4

from fastapi import FastAPI, Form, Request, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from sentiment_Razuvaev_module.models import analyze_sentiment
from processing_of_text_documents_Chizhov_module.source.parser_text import extract_text

app = FastAPI()

BASE_DIR = os.path.dirname(__file__)
WEB_DIR = os.path.join(BASE_DIR, "web")
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)

templates = Jinja2Templates(directory=os.path.join(WEB_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(WEB_DIR, "static")), name="static")


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
            result = {
                "query": query,
                "found": "Здесь будет результат поиска (пустая оболочка)"
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
            result = {"message": f"Модуль '{module_type}' пока не реализован"}

        return JSONResponse(result)

    except Exception as e:
        return JSONResponse(
            {"error": f"Ошибка обработки: {str(e)}"},
            status_code=500
        )