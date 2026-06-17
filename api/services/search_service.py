import math
import re
from typing import List, Tuple, Dict, Any, Optional

def _vector_norm(vec: List[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))

def _normalize_vector(vec: List[float]) -> List[float]:
    norm = _vector_norm(vec)
    if norm <= 0.0:
        return [0.0 for _ in vec]
    return [v / norm for v in vec]

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b: return 0.0
    n = min(len(a), len(b))
    if n == 0: return 0.0
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(a[i] * a[i] for i in range(n)))
    nb = math.sqrt(sum(b[i] * b[i] for i in range(n)))
    if na <= 0 or nb <= 0: return 0.0
    return dot / (na * nb)

def _average_vectors(vectors: List[List[float]]) -> List[float]:
    if not vectors: return []
    dim = max(len(v) for v in vectors)
    if dim <= 0: return []
    acc = [0.0] * dim
    used = 0
    for vec in vectors:
        if not vec: continue
        for i, val in enumerate(vec):
            acc[i] += val
        used += 1
    if used > 0:
        return [v / used for v in acc]
    return [0.0] * dim

def _normalize_weight_triplet(a: float, b: float, c: float) -> Tuple[float, float, float]:
    total = max(0.001, float(a or 0) + float(b or 0) + float(c or 0))
    return (max(0.0, float(a or 0))/total, max(0.0, float(b or 0))/total, max(0.0, float(c or 0))/total)

def _analyze_query_profile(query: str) -> Dict[str, Any]:
    q = str(query or "").lower().strip()
    profile = {
        "is_question": any(q.startswith(w) for w in ["quien", "que", "como", "donde", "cuando", "por qué"]),
        "is_time_sensitive": any(w in q for w in ["después", "antes", "mientras", "luego"]),
        "weights": (0.33, 0.33, 0.34), # Semantic, Keyword, Context
        "min_score": 0.25,
        "padding_sec": 2.0
    }
    # Heuristics to adjust weights based on query content
    if len(q.split()) > 6:
        profile["weights"] = (0.6, 0.2, 0.2)
    elif len(q.split()) < 3:
        profile["weights"] = (0.2, 0.6, 0.2)
    return profile

def _relax_query_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    new_profile = dict(profile)
    new_profile["min_score"] = max(0.1, profile.get("min_score", 0.2) * 0.7)
    new_profile["padding_sec"] = profile.get("padding_sec", 2.0) + 1.0
    return new_profile
