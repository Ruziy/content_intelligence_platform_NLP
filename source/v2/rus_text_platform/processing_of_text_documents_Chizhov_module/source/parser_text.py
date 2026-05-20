import re
from pathlib import Path

from docx import Document
import pdfplumber
from paddleocr import PaddleOCR


_OCR_MODEL = None


def get_ocr():
    global _OCR_MODEL
    if _OCR_MODEL is None:
        _OCR_MODEL = PaddleOCR(
            lang="ru",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )
    return _OCR_MODEL


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"([,.;:!?])(\S)", r"\1 \2", text)

    return text


def read_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return clean_text(f.read())


def read_docx(file_path: str) -> str:
    doc = Document(file_path)
    text = " ".join(p.text for p in doc.paragraphs if p.text.strip())
    return clean_text(text)


def read_pdf(file_path: str) -> str:
    pages_text = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
    return clean_text(" ".join(pages_text))


def read_image_ocr(file_path: str, ocr_model=None) -> str:
    if ocr_model is None:
        ocr_model = get_ocr()

    results = ocr_model.predict(file_path)

    all_text = []
    for page in results:
        if isinstance(page, dict):
            rec_texts = page.get("rec_texts", [])
            if rec_texts:
                all_text.extend(rec_texts)

    return clean_text(" ".join(all_text))


def extract_text(file_path: str, ocr_model=None) -> str:
    ext = Path(file_path).suffix.lower()

    if ext == ".txt":
        return read_txt(file_path)
    elif ext == ".docx":
        return read_docx(file_path)
    elif ext == ".pdf":
        return read_pdf(file_path)
    elif ext in [".jpg", ".jpeg", ".png"]:
        return read_image_ocr(file_path, ocr_model=ocr_model)
    else:
        raise ValueError(
            f"Неподдерживаемый тип файла: {ext}. "
            f"Поддерживаются: .txt, .docx, .pdf, .jpg, .jpeg, .png"
        )