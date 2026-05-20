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
    raw_entities = pipe(text)
    normalized_entities = []
    for entity in raw_entities:
        normalized_entities.append(
            {
                "text": entity.get("word", ""),
                "label": entity.get("entity_group", entity.get("entity", "UNK")),
                "start": entity.get("start"),
                "end": entity.get("end"),
            }
        )
    return normalized_entities