
import re
def basic_clean(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)

    return text

from bs4 import BeautifulSoup
def remove_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ")

def remove_urls(text):
    return re.sub(r"http\S+|www\S+", "", text)

def remove_numbers(text):

    return re.sub(r"\d+", "", text)

def normalize_spaces(text):

    text = re.sub(r"\s+", " ", text)

    return text.strip()

import string

def remove_punctuation(text):

    translator = str.maketrans("", "", string.punctuation)

    return text.translate(translator)

import spacy
nlp = spacy.load("ru_core_news_sm")
def remove_stopwords(text):
    doc = nlp(text)
    tokens = [
        token.text
        for token in doc
        if not token.is_stop
    ]
    return " ".join(tokens)

def lemmatize_text(text):
    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
    ]
    return " ".join(tokens)

def remove_short_words(text, min_len=3):

    tokens = text.split()

    tokens = [
        word
        for word in tokens
        if len(word) >= min_len
    ]

    return " ".join(tokens)


def full_clean(text):

    text = remove_html(text)

    text = remove_urls(text)

    text = basic_clean(text)

    text = remove_stopwords(text)

    text = remove_punctuation(text)

    text = normalize_spaces(text)

    return text


CLEANING_METHODS = {

    "basic": basic_clean,

    "full": full_clean,

    "remove_urls": remove_urls,

    "remove_numbers": remove_numbers,

    "remove_stopwords": remove_stopwords,

    "lemmatize": lemmatize_text
}

def run_cleaning(text, method="basic"):
    cleaner = CLEANING_METHODS[method]
    return cleaner(text)