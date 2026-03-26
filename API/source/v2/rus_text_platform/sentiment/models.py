from transformers import pipeline

models = {
    "rubert_tiny": {
        "pipeline": pipeline(
            "sentiment-analysis",
            model="cointegrated/rubert-tiny-sentiment-balanced"
        ),
        "hyperparams": {
            "max_length": 128,
            "batch_size": 16,
            "truncation": True,
            "padding": "max_length",
            "return_all_scores": True,
        }
    },
    "rubert_base": {
        "pipeline": pipeline(
            "sentiment-analysis",
            model="blanchefort/rubert-base-cased-sentiment"
        ),
        "hyperparams": {
            "max_length": 256,
            "batch_size": 16,
            "truncation": True,
            "padding": "max_length",
            "return_all_scores": True,
        }
    }
}

def analyze_sentiment(text, model_key="rubert_tiny"):
    """
    Возвращает список с результатом:
    [{'label': 'POSITIVE'/'NEGATIVE'/'NEUTRAL', 'score': 0.95}, ...]
    """
    pipe = models[model_key]["pipeline"]
    return pipe(text)