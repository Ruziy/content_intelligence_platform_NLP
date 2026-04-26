import math
import time
from statistics import mean


def _normalize_label(label):
    """Приводит label к единому формату (UPPER, без BIO-префикса)."""
    if label is None:
        return "UNK"
    normalized = str(label).strip().upper()
    if "-" in normalized:
        normalized = normalized.split("-")[-1]
    return normalized


def _normalize_entity(entity):
    """Нормализует одну сущность к контракту text/label/start/end."""
    text = str(entity.get("text", "")).strip()
    label = _normalize_label(entity.get("label"))
    start = entity.get("start")
    end = entity.get("end")
    return {
        "text": text,
        "label": label,
        "start": start,
        "end": end,
    }


def _normalize_entities(entities):
    """Нормализует список сущностей от любого extractor."""
    return [_normalize_entity(entity) for entity in entities]


def _strict_key(entity):
    return (entity.get("start"), entity.get("end"), entity.get("label"))


def _relaxed_key(entity):
    return (entity.get("text", "").casefold(), entity.get("label"))


def _build_key(entity, matching_mode):
    """Строит ключ сравнения: strict (span+label) или relaxed (text+label)."""
    if matching_mode == "strict":
        start = entity.get("start")
        end = entity.get("end")
        if start is not None and end is not None:
            return _strict_key(entity)
    return _relaxed_key(entity)


def _compute_quality_metrics(gold_entities_by_text, pred_entities_by_text, matching_mode):
    """Считает TP/FP/FN и micro precision/recall/F1 по всему датасету."""
    tp = 0
    fp = 0
    fn = 0

    for gold_entities, pred_entities in zip(gold_entities_by_text, pred_entities_by_text):
        gold_keys = [_build_key(entity, matching_mode) for entity in gold_entities]
        pred_keys = [_build_key(entity, matching_mode) for entity in pred_entities]

        unmatched_gold = list(gold_keys)
        local_tp = 0
        for pred_key in pred_keys:
            if pred_key in unmatched_gold:
                unmatched_gold.remove(pred_key)
                local_tp += 1

        local_fp = max(0, len(pred_keys) - local_tp)
        local_fn = max(0, len(unmatched_gold))

        tp += local_tp
        fp += local_fp
        fn += local_fn

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _percentile(values, percentile):
    """Вычисляет percentile для списка значений в миллисекундах."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100.0
    floor = math.floor(index)
    ceil = math.ceil(index)
    if floor == ceil:
        return float(ordered[int(index)])
    lower = ordered[floor] * (ceil - index)
    upper = ordered[ceil] * (index - floor)
    return float(lower + upper)


def _normalize_latency(avg_latency_ms, reference_latency_ms):
    """Нормирует latency в диапазон [0, 1] относительно reference."""
    if reference_latency_ms <= 0:
        return 0.0
    return min(1.0, avg_latency_ms / reference_latency_ms)


def _summarize_latency(values_ms):
    """Собирает агрегаты latency: avg, p95, throughput, count."""
    total_time_s = sum(values_ms) / 1000.0 if values_ms else 0.0
    throughput_tps = (len(values_ms) / total_time_s) if total_time_s > 0 else 0.0
    return {
        "avg_ms": mean(values_ms) if values_ms else 0.0,
        "p95_ms": _percentile(values_ms, 95),
        "throughput_tps": throughput_tps,
        "count": len(values_ms),
    }


def _extract_stage_timing(stage_timings, stage_name):
    """Безопасно извлекает числовой тайминг этапа из trace."""
    value = stage_timings.get(stage_name, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def evaluate_pipeline(
    texts,
    gold_entities,
    extractor,
    config=None,
):
    """
    Оценивает только NER extractor (без preprocessing этапов).

    Возвращает словарь с quality/runtime метриками и итоговым score.
    """
    config = config or {}
    matching_mode = config.get("matching_mode", "strict")
    w_f1 = float(config.get("w_f1", 0.8))
    w_latency = float(config.get("w_latency", 0.2))
    reference_latency_ms = float(config.get("reference_latency_ms", 100.0))

    gold_entities_by_text = [_normalize_entities(entities) for entities in gold_entities]
    pred_entities_by_text = []
    latencies_ms = []

    for text in texts:
        start = time.perf_counter()
        pred_entities = extractor(text)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)
        pred_entities_by_text.append(_normalize_entities(pred_entities))

    quality_metrics = _compute_quality_metrics(
        gold_entities_by_text=gold_entities_by_text,
        pred_entities_by_text=pred_entities_by_text,
        matching_mode=matching_mode,
    )

    total_time_s = sum(latencies_ms) / 1000.0
    throughput_tps = (len(texts) / total_time_s) if total_time_s > 0 else 0.0
    avg_latency_ms = mean(latencies_ms) if latencies_ms else 0.0
    p95_latency_ms = _percentile(latencies_ms, 95)
    normalized_latency = _normalize_latency(avg_latency_ms, reference_latency_ms)

    score = (w_f1 * quality_metrics["f1"]) - (w_latency * normalized_latency)

    metrics = {
        "matching_mode": matching_mode,
        "precision": quality_metrics["precision"],
        "recall": quality_metrics["recall"],
        "f1": quality_metrics["f1"],
        "tp": quality_metrics["tp"],
        "fp": quality_metrics["fp"],
        "fn": quality_metrics["fn"],
        "avg_latency_ms": avg_latency_ms,
        "p95_latency_ms": p95_latency_ms,
        "throughput_tps": throughput_tps,
    }

    return {
        "metrics": metrics,
        "score": score,
    }


def evaluate_full_pipeline(
    texts,
    gold_entities,
    pipeline_fn,
    config=None,
):
    """
    Оценивает полный pipeline по trace-контракту.

    В score учитываются NER качество, общая latency и penalties за hard constraints.
    """
    config = config or {}
    matching_mode = config.get("matching_mode", "strict")
    w_f1 = float(config.get("w_f1", 0.8))
    w_latency = float(config.get("w_latency", 0.2))
    reference_latency_ms = float(config.get("reference_latency_ms", 100.0))

    expected_language = config.get("expected_language", "ru")
    min_clean_char_ratio = float(config.get("min_clean_char_ratio", 0.35))
    min_token_count = int(config.get("min_token_count", 3))
    max_empty_entity_ratio = float(config.get("max_empty_entity_ratio", 0.8))

    penalty_weights = config.get("penalty_weights", {})
    w_cleaning_penalty = float(penalty_weights.get("cleaning_overdelete", 0.2))
    w_token_penalty = float(penalty_weights.get("token_count", 0.15))
    w_lang_penalty = float(penalty_weights.get("lang_mismatch", 0.25))
    w_empty_penalty = float(penalty_weights.get("entity_empty", 0.25))

    gold_entities_by_text = [_normalize_entities(entities) for entities in gold_entities]
    pred_entities_by_text = []
    traces = []

    stage_latencies = {
        "cleaning_ms": [],
        "tokenization_ms": [],
        "language_ms": [],
        "ner_ms": [],
        "formatting_ms": [],
        "total_ms": [],
    }

    cleaning_penalty_sum = 0.0
    token_penalty_sum = 0.0
    lang_penalty_sum = 0.0
    empty_entity_count = 0

    for text in texts:
        total_start = time.perf_counter()
        trace = pipeline_fn(text)
        total_ms = (time.perf_counter() - total_start) * 1000.0

        clean_text = str(trace.get("clean_text", ""))
        tokens = trace.get("tokens", [])
        language = trace.get("language")
        entities = _normalize_entities(trace.get("entities", []))
        pred_entities_by_text.append(entities)
        traces.append(trace)

        if not entities:
            empty_entity_count += 1

        raw_stage_timings = trace.get("stage_timings", {})
        stage_latencies["cleaning_ms"].append(_extract_stage_timing(raw_stage_timings, "cleaning_ms"))
        stage_latencies["tokenization_ms"].append(_extract_stage_timing(raw_stage_timings, "tokenization_ms"))
        stage_latencies["language_ms"].append(_extract_stage_timing(raw_stage_timings, "language_ms"))
        stage_latencies["ner_ms"].append(_extract_stage_timing(raw_stage_timings, "ner_ms"))
        stage_latencies["formatting_ms"].append(_extract_stage_timing(raw_stage_timings, "formatting_ms"))
        stage_latencies["total_ms"].append(_extract_stage_timing(raw_stage_timings, "total_ms") or total_ms)

        original_len = max(1, len(text))
        clean_len = len(clean_text)
        clean_ratio = clean_len / original_len
        if clean_ratio < min_clean_char_ratio:
            cleaning_penalty_sum += (min_clean_char_ratio - clean_ratio) / max(min_clean_char_ratio, 1e-9)

        token_count = len(tokens)
        if token_count < min_token_count:
            token_penalty_sum += (min_token_count - token_count) / max(min_token_count, 1)

        if expected_language and language != expected_language:
            lang_penalty_sum += 1.0

    quality_metrics = _compute_quality_metrics(
        gold_entities_by_text=gold_entities_by_text,
        pred_entities_by_text=pred_entities_by_text,
        matching_mode=matching_mode,
    )

    runtime_stages = {
        "cleaning": _summarize_latency(stage_latencies["cleaning_ms"]),
        "tokenization": _summarize_latency(stage_latencies["tokenization_ms"]),
        "language": _summarize_latency(stage_latencies["language_ms"]),
        "ner": _summarize_latency(stage_latencies["ner_ms"]),
        "formatting": _summarize_latency(stage_latencies["formatting_ms"]),
        "total": _summarize_latency(stage_latencies["total_ms"]),
    }

    normalized_latency = _normalize_latency(runtime_stages["total"]["avg_ms"], reference_latency_ms)
    base_score = (w_f1 * quality_metrics["f1"]) - (w_latency * normalized_latency)

    sample_count = max(1, len(texts))
    cleaning_penalty = w_cleaning_penalty * (cleaning_penalty_sum / sample_count)
    token_penalty = w_token_penalty * (token_penalty_sum / sample_count)
    lang_penalty = w_lang_penalty * (lang_penalty_sum / sample_count)

    empty_entity_ratio = empty_entity_count / sample_count
    empty_ratio_excess = max(0.0, empty_entity_ratio - max_empty_entity_ratio)
    entity_empty_penalty = w_empty_penalty * empty_ratio_excess

    total_penalty = cleaning_penalty + token_penalty + lang_penalty + entity_empty_penalty
    final_score = base_score - total_penalty

    penalties = {
        "cleaning_overdelete_penalty": cleaning_penalty,
        "token_count_penalty": token_penalty,
        "lang_mismatch_penalty": lang_penalty,
        "entity_empty_penalty": entity_empty_penalty,
        "total_penalty": total_penalty,
        "empty_entity_ratio": empty_entity_ratio,
    }

    metrics = {
        "matching_mode": matching_mode,
        "quality": {
            "precision": quality_metrics["precision"],
            "recall": quality_metrics["recall"],
            "f1": quality_metrics["f1"],
            "tp": quality_metrics["tp"],
            "fp": quality_metrics["fp"],
            "fn": quality_metrics["fn"],
        },
        "runtime_stages": runtime_stages,
    }

    response = {
        "metrics": metrics,
        "penalties": penalties,
        "score": final_score,
        "base_score": base_score,
    }
    if config.get("return_traces", False):
        response["traces"] = traces
    return response
