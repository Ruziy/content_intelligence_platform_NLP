from transformers import pipeline

ner_models = {

    "rubert": pipeline(
        "ner",
        model="DeepPavlov/rubert-base-cased-conversational",
        aggregation_strategy="simple"
    ),

    "bert": pipeline(
        "ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple"
    )
}


def extract_entities_transformer(text, model="rubert"):

    pipe = ner_models[model]

    return pipe(text)