from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sentiment.models import analyze_sentiment
import os

app = FastAPI()

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

@app.get("/")
async def index():
    # просто возвращаем шаблон, результат пустой — JSON будет через AJAX
    return templates.TemplateResponse("index.html", {"request": {}})

@app.post("/process")
async def process_ajax(
    module_type: str = Form(...),
    text: str = Form(...),
    query: str = Form(None)
):
    result = {}
    if module_type == "Анализ настроений по текстам":
        result = analyze_sentiment(text)
    elif module_type == "Поиск текстовых данных в системе":
        result = {"query": query, "found": "Здесь будет результат поиска (пустая оболочка)"}
    else:
        result = {"message": f"Модуль '{module_type}' пока не реализован"}
    return JSONResponse(result)