from transformers import pipeline

'''
Если позже подключить генетический алгоритм, он может подбирать:
model
aggregation_strategy
max_length
batch_size
stride
confidence_threshold
'''

ner_models = {

    "rubert_ner": {
        "pipeline": pipeline(
            "ner",
            model="DeepPavlov/rubert-base-cased-conversational",
            aggregation_strategy="simple"
        ),

        "hyperparams": {
            "max_length": 256,
            "batch_size": 16,
            "truncation": True,
            "stride": 32
        }
    },

    "bert_ner": {
        "pipeline": pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple"
        ),

        "hyperparams": {
            "max_length": 512,
            "batch_size": 8,
            "truncation": True
        }
    }
}

def extract_entities(text, model_key):

    model = ner_models[model_key]

    pipe = model["pipeline"]
    params = model["hyperparams"]

    return pipe(text)