import spacy

nlp = spacy.load("ru_core_news_sm")

def spacy_tokenize(text):

    doc = nlp(text)

    return [token.text for token in doc]