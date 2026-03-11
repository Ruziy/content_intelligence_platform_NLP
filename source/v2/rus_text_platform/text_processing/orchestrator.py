from preprocessing.cleaning import run_cleaning
from preprocessing.tokenization import spacy_tokenize
from preprocessing.language_detection import detect_language

from extraction.ner_spacy import extract_entities_spacy
# from extraction.ner_transformers import extract_entities_transformer
from formatting.json_formatter import build_document


def process_text(text):

    clean_text = run_cleaning(text, method="full")

    tokens = spacy_tokenize(clean_text)

    language = detect_language(clean_text)

    entities = extract_entities_spacy(clean_text)
    # entities = extract_entities_transformer(clean_text)

    doc = build_document(
        text=clean_text,
        tokens=tokens,
        entities=entities,
        language=language
    )

    return doc


text = "Президент Франции Эммануэль Макрон прибыл в Москву"
res = process_text(text)
print(res)