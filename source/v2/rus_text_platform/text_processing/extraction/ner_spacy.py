import spacy

nlp = spacy.load("ru_core_news_sm")

def extract_entities_spacy(text):

    doc = nlp(text)

    entities = []

    for ent in doc.ents:

        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        })

    return entities

