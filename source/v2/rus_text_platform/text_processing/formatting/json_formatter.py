def build_document(text, tokens, entities, language):

    return {
        "text": text,
        "tokens": tokens,
        "entities": entities,
        "language": language
    }