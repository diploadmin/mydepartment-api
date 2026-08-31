"""
Dynamic label weight boosting via prototype embeddings.

At startup, loads prototype questions from config/prototype_questions.txt,
embeds them via TEI, and averages into one prototype vector per post_type.

At query time, computes cosine similarity between the query vector and each
prototype. Adds an additive bonus on top of static weights:
    bonus = min((similarity - min_similarity) * alpha, 0.5)
    final_weight = static_weight + bonus
The bonus is capped at 0.5 to prevent over-boosting high-similarity matches.
"""

import math
import os
from pathlib import Path
from typing import Optional

from app.ai.ai_services.label_weights import (
    POST_TYPE_TO_LABEL,
    DEFAULT_LABEL_WEIGHT,
    get_label_weights,
)

_prototypes: dict[str, list[float]] = {}
_initialized = False

#
def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _parse_prototype_file(path: str) -> dict[str, list[str]]:
    """Parse config/prototype_questions.txt into {post_type: [questions]}."""
    result: dict[str, list[str]] = {}
    current_section = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                current_section = line[1:-1].strip().lower()
                if current_section not in result:
                    result[current_section] = []
            elif current_section is not None:
                result[current_section].append(line)

    return result


def initialize_prototypes(embeddings) -> None:
    """
    Load prototype questions and compute averaged prototype vectors.
    Call once at startup after embeddings are available.
    """
    global _prototypes, _initialized

    config_path = Path(__file__).resolve().parents[3] / "config" / "prototype_questions.txt"
    if not config_path.exists():
        print(f"WARNING: {config_path} not found, dynamic label weights disabled", flush=True)
        _initialized = True
        return

    questions_by_type = _parse_prototype_file(str(config_path))
    if not questions_by_type:
        print("WARNING: No prototype questions parsed, dynamic label weights disabled", flush=True)
        _initialized = True
        return

    all_questions = []
    question_map = []
    for post_type, questions in questions_by_type.items():
        for q in questions:
            all_questions.append(q)
            question_map.append(post_type)

    print(f"INTENT: Embedding {len(all_questions)} prototype questions "
          f"across {len(questions_by_type)} post_types...", flush=True)

    try:
        vectors = []
        batch_size = 30
        for i in range(0, len(all_questions), batch_size):
            batch = all_questions[i : i + batch_size]
            vectors.extend(embeddings.embed_documents(batch))
    except Exception as e:
        print(f"ERROR: Failed to embed prototype questions: {e}", flush=True)
        _initialized = True
        return

    type_vectors: dict[str, list[list[float]]] = {}
    for i, vec in enumerate(vectors):
        pt = question_map[i]
        if pt not in type_vectors:
            type_vectors[pt] = []
        type_vectors[pt].append(vec)

    for pt, vecs in type_vectors.items():
        dim = len(vecs[0])
        avg = [0.0] * dim
        for v in vecs:
            for j in range(dim):
                avg[j] += v[j]
        n = len(vecs)
        avg = [x / n for x in avg]
        norm = math.sqrt(sum(x * x for x in avg))
        if norm > 0:
            avg = [x / norm for x in avg]
        _prototypes[pt] = avg

    _initialized = True
    print(f"INTENT: Loaded {len(_prototypes)} prototype vectors: "
          f"{', '.join(sorted(_prototypes.keys()))}", flush=True)


def compute_similarities(query_vector: list[float]) -> dict[str, float]:
    """Compute cosine similarity between query and each prototype."""
    if not _prototypes:
        return {}
    return {
        pt: _cosine_similarity(query_vector, proto)
        for pt, proto in _prototypes.items()
    }


def get_dynamic_label_weights(
    query_vector: list[float],
    user_type: str,
    alpha: float = 2.2,
    min_similarity: float = 0.3,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Compute label weights by adding an intent-based bonus on top of static weights.

    For each post type whose similarity exceeds min_similarity, an additive bonus
    is computed: bonus = min((sim - min_similarity) * alpha, 0.5).
    The cap of 0.5 prevents over-boosting high-similarity matches.

    Args:
        query_vector: Already-computed query embedding.
        user_type: User profile for static weight baseline.
        alpha: Boost scale factor applied to (similarity - min_similarity).
               With alpha=2.2 and sim=0.5, bonus = (0.5-0.3)*2.2 = 0.44.
        min_similarity: Below this threshold, no bonus is applied.

    Returns:
        Tuple of (adjusted_weights, similarities) where:
        - adjusted_weights: {label: static + bonus} dict ready for scoring
        - similarities: {post_type: similarity} dict for debug/logging
    """
    static_weights = get_label_weights(user_type)

    if not _initialized or not _prototypes:
        return dict(static_weights), {}

    similarities = compute_similarities(query_vector)

    adjusted = dict(static_weights)
    for post_type, sim in similarities.items():
        if sim < min_similarity:
            continue
        label = POST_TYPE_TO_LABEL.get(post_type)
        if not label:
            continue
        static_val = adjusted.get(label, DEFAULT_LABEL_WEIGHT)
        bonus = min((sim - min_similarity) * alpha, 0.5)
        adjusted[label] = static_val + bonus

    return adjusted, similarities
