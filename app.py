import os
import sys
import uuid
import subprocess
import threading
import json
import shutil
import glob
import time
import asyncio
import re
import csv
import io
import zipfile
import math
import zlib
from dotenv import load_dotenv
from typing import Dict, Optional, List, Any, Tuple
from collections import Counter
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from s3_uploader import upload_job_artifacts

load_dotenv()

# Constants
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration
# Default to 1 if not set, but user can set higher for powerful servers
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "5"))
MAX_FILE_SIZE_MB = 500  # 500 MB limit
JOB_RETENTION_SECONDS = 3600  # 1 hour retention
MAX_AUTO_RETRIES_DEFAULT = int(os.environ.get("MAX_AUTO_RETRIES", "1"))
JOB_RETRY_DELAY_SECONDS_DEFAULT = int(os.environ.get("JOB_RETRY_DELAY_SECONDS", "10"))

# Application State
job_queue = asyncio.Queue()
jobs: Dict[str, Dict] = {}
# Semester to limit concurrency to MAX_CONCURRENT_JOBS
concurrency_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
SEARCH_INDEX_CACHE: Dict[str, Dict[str, Any]] = {}
LOCAL_EMBED_DIM = 256
SEMANTIC_EMBED_MODEL = os.environ.get("SEMANTIC_EMBED_MODEL", "text-embedding-004")
ALLOWED_ASPECT_RATIOS = {"9:16", "16:9"}
ALLOWED_CLIP_LENGTH_TARGETS = {"short", "balanced", "long"}


def normalize_aspect_ratio(raw_value: Optional[str], default: Optional[str] = None) -> Optional[str]:
    if raw_value is None:
        return default
    value = str(raw_value).strip().replace("/", ":")
    if not value:
        return default
    if value not in ALLOWED_ASPECT_RATIOS:
        raise HTTPException(status_code=400, detail="Invalid aspect_ratio. Allowed values: 9:16, 16:9")
    return value

def _relocate_root_job_artifacts(job_id: str, job_output_dir: str) -> bool:
    """
    Backward-compat rescue:
    If main.py accidentally wrote metadata/clips into OUTPUT_DIR root (e.g. output/<jobid>_...),
    move them into output/<job_id>/ so the API can find and serve them.
    """
    try:
        os.makedirs(job_output_dir, exist_ok=True)
        root = OUTPUT_DIR
        pattern = os.path.join(root, f"{job_id}_*_metadata.json")
        meta_candidates = sorted(glob.glob(pattern), key=lambda p: os.path.getmtime(p), reverse=True)
        if not meta_candidates:
            return False

        # Move the newest metadata and its associated clips.
        metadata_path = meta_candidates[0]
        base_name = os.path.basename(metadata_path).replace("_metadata.json", "")

        # Move metadata
        dest_metadata = os.path.join(job_output_dir, os.path.basename(metadata_path))
        if os.path.abspath(metadata_path) != os.path.abspath(dest_metadata):
            shutil.move(metadata_path, dest_metadata)

        # Move any clips that match the same base_name into the job folder
        clip_pattern = os.path.join(root, f"{base_name}_clip_*.mp4")
        for clip_path in glob.glob(clip_pattern):
            dest_clip = os.path.join(job_output_dir, os.path.basename(clip_path))
            if os.path.abspath(clip_path) != os.path.abspath(dest_clip):
                shutil.move(clip_path, dest_clip)

        # Also move any temp_ clips that might remain
        temp_clip_pattern = os.path.join(root, f"temp_{base_name}_clip_*.mp4")
        for clip_path in glob.glob(temp_clip_pattern):
            dest_clip = os.path.join(job_output_dir, os.path.basename(clip_path))
            if os.path.abspath(clip_path) != os.path.abspath(dest_clip):
                shutil.move(clip_path, dest_clip)

        return True
    except Exception:
        return False

def _default_score_by_rank(rank: int) -> int:
    return max(55, 92 - (rank * 6))

def _score_band(score: int) -> str:
    if score >= 80:
        return "top"
    if score >= 65:
        return "medium"
    return "low"

def _normalize_confidence(raw_confidence, score: int) -> float:
    try:
        conf = float(raw_confidence)
    except (TypeError, ValueError):
        conf = score / 100.0
    return round(max(0.0, min(1.0, conf)), 2)

def _normalize_topic_tags(raw_tags) -> List[str]:
    if isinstance(raw_tags, str):
        raw_tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
    if not isinstance(raw_tags, list):
        return []

    out: List[str] = []
    seen = set()
    for tag in raw_tags:
        if not isinstance(tag, str):
            continue
        clean = tag.strip().lstrip("#").lower()[:24]
        if not clean or clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
        if len(out) >= 5:
            break
    return out

def _default_topic_tags(clip: Dict) -> List[str]:
    text = " ".join([
        str(clip.get("video_title_for_youtube_short", "")),
        str(clip.get("video_description_for_tiktok", "")),
        str(clip.get("video_description_for_instagram", "")),
    ]).lower()
    words = re.findall(r"[a-zA-ZÀ-ÿ0-9]{4,}", text)
    stop = {
        "this", "that", "with", "para", "como", "este", "esta", "from",
        "about", "your", "have", "will", "they", "porque", "cuando",
        "donde", "video", "viral", "short", "shorts", "follow", "comment"
    }
    tags: List[str] = []
    seen = set()
    for w in words:
        if w in stop or w in seen:
            continue
        seen.add(w)
        tags.append(w[:24])
        if len(tags) >= 3:
            break
    return tags

def _tokenize_query(text: str) -> List[str]:
    if not text:
        return []
    words = re.findall(r"[a-zA-ZÀ-ÿ0-9]{3,}", text.lower())
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "como", "para",
        "cuando", "donde", "sobre", "porque", "video", "clip", "short", "shorts"
    }
    out: List[str] = []
    seen = set()
    for w in words:
        if w in stop or w in seen:
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= 8:
            break
    return out

def _extract_query_phrases(query: str, keywords: List[str]) -> List[str]:
    phrases: List[str] = []
    seen = set()
    for raw in re.findall(r'"([^"]{3,80})"', str(query or "")):
        p = _normalize_space(raw).lower()
        if len(p) < 3:
            continue
        if p in seen:
            continue
        seen.add(p)
        phrases.append(p)

    # If user didn't quote a phrase, recover a couple of n-grams for exact-ish matching.
    if not phrases and len(keywords) >= 2:
        for n in (3, 2):
            if len(keywords) < n:
                continue
            for i in range(len(keywords) - n + 1):
                p = " ".join(keywords[i:i + n]).strip()
                if len(p) < 6 or p in seen:
                    continue
                seen.add(p)
                phrases.append(p)
                if len(phrases) >= 2:
                    break
            if len(phrases) >= 2:
                break
    return phrases[:3]

def _normalize_weight_triplet(a: float, b: float, c: float) -> Tuple[float, float, float]:
    vals = [max(0.0, float(a)), max(0.0, float(b)), max(0.0, float(c))]
    total = sum(vals)
    if total <= 0:
        return (0.62, 0.23, 0.15)
    return (vals[0] / total, vals[1] / total, vals[2] / total)

def _analyze_query_profile(query: str, keywords: List[str], phrases: List[str]) -> Dict[str, Any]:
    q = str(query or "")
    q_l = q.lower()
    token_count = len(keywords)

    is_question = q.strip().endswith("?") or bool(re.search(r"\b(how|why|what|when|where|who|como|por que|porque|que|qué|cuando|cuándo|donde|dónde|cual|cuál)\b", q_l))
    is_time_seek = bool(re.search(r"\b(\d{1,2}:\d{2}|\d{1,4}\s*(s|sec|secs|seg|segundo|min|minute|minuto)s?)\b", q_l)) or any(
        marker in q_l for marker in ("timestamp", "timecode", "minuto", "segundo", "at ", "desde ", "between ")
    )
    is_exact_phrase = len(phrases) > 0 or bool(re.search(r'"[^"]{3,80}"', q))
    is_broad_topic = token_count >= 5 and not is_time_seek

    mode = "topic"
    if is_exact_phrase:
        mode = "exact_phrase"
    elif is_time_seek:
        mode = "time_seek"
    elif is_question:
        mode = "question"
    elif is_broad_topic:
        mode = "broad_topic"
    elif token_count <= 2:
        mode = "entity_focus"

    presets: Dict[str, Dict[str, Any]] = {
        "exact_phrase": {
            "weights": _normalize_weight_triplet(0.48, 0.40, 0.12),
            "min_hybrid_score": 0.18,
            "min_semantic_score": 0.07,
            "min_keyword_score": 0.10,
            "pad_before": 1.8,
            "pad_after": 8.0,
            "shortlist_weights": _normalize_weight_triplet(0.70, 0.20, 0.10),
            "shortlist_min_score": 0.16,
        },
        "time_seek": {
            "weights": _normalize_weight_triplet(0.50, 0.36, 0.14),
            "min_hybrid_score": 0.14,
            "min_semantic_score": 0.06,
            "min_keyword_score": 0.08,
            "pad_before": 1.2,
            "pad_after": 6.0,
            "shortlist_weights": _normalize_weight_triplet(0.62, 0.24, 0.14),
            "shortlist_min_score": 0.12,
        },
        "question": {
            "weights": _normalize_weight_triplet(0.66, 0.20, 0.14),
            "min_hybrid_score": 0.13,
            "min_semantic_score": 0.10,
            "min_keyword_score": 0.05,
            "pad_before": 2.2,
            "pad_after": 10.0,
            "shortlist_weights": _normalize_weight_triplet(0.72, 0.14, 0.14),
            "shortlist_min_score": 0.12,
        },
        "broad_topic": {
            "weights": _normalize_weight_triplet(0.56, 0.18, 0.26),
            "min_hybrid_score": 0.10,
            "min_semantic_score": 0.06,
            "min_keyword_score": 0.03,
            "pad_before": 3.2,
            "pad_after": 13.0,
            "shortlist_weights": _normalize_weight_triplet(0.56, 0.14, 0.30),
            "shortlist_min_score": 0.10,
        },
        "entity_focus": {
            "weights": _normalize_weight_triplet(0.54, 0.31, 0.15),
            "min_hybrid_score": 0.12,
            "min_semantic_score": 0.06,
            "min_keyword_score": 0.08,
            "pad_before": 2.0,
            "pad_after": 9.0,
            "shortlist_weights": _normalize_weight_triplet(0.60, 0.23, 0.17),
            "shortlist_min_score": 0.12,
        },
        "topic": {
            "weights": _normalize_weight_triplet(0.62, 0.23, 0.15),
            "min_hybrid_score": 0.11,
            "min_semantic_score": 0.07,
            "min_keyword_score": 0.04,
            "pad_before": 2.8,
            "pad_after": 11.0,
            "shortlist_weights": _normalize_weight_triplet(0.64, 0.16, 0.20),
            "shortlist_min_score": 0.11,
        },
    }

    selected = dict(presets.get(mode, presets["topic"]))
    selected["mode"] = mode
    selected["token_count"] = token_count
    selected["phrase_count"] = len(phrases)
    selected["is_question"] = is_question
    selected["is_time_seek"] = is_time_seek
    selected["is_exact_phrase"] = is_exact_phrase
    selected["relaxed"] = False
    return selected

def _relax_query_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    relaxed = dict(profile or {})
    relaxed["min_hybrid_score"] = max(0.06, _safe_float(profile.get("min_hybrid_score", 0.1), 0.1) * 0.72)
    relaxed["min_semantic_score"] = max(0.03, _safe_float(profile.get("min_semantic_score", 0.06), 0.06) * 0.68)
    relaxed["min_keyword_score"] = max(0.02, _safe_float(profile.get("min_keyword_score", 0.04), 0.04) * 0.7)
    relaxed["pad_before"] = _safe_float(profile.get("pad_before", 2.5), 2.5) + 0.8
    relaxed["pad_after"] = _safe_float(profile.get("pad_after", 10.0), 10.0) + 2.2
    relaxed["relaxed"] = True
    return relaxed

def _normalize_search_mode(raw_mode: Optional[str]) -> str:
    value = str(raw_mode or "balanced").strip().lower()
    aliases = {
        "precision": "exact",
        "precise": "exact",
        "strict": "exact",
        "exact": "exact",
        "balanced": "balanced",
        "balance": "balanced",
        "default": "balanced",
        "recall": "broad",
        "wide": "broad",
        "broad": "broad",
    }
    return aliases.get(value, "balanced")

def _apply_search_mode_override(profile: Dict[str, Any], search_mode: str) -> Dict[str, Any]:
    out = dict(profile or {})
    mode = _normalize_search_mode(search_mode)
    out["search_mode"] = mode

    if mode == "exact":
        out["weights"] = _normalize_weight_triplet(0.56, 0.34, 0.10)
        out["shortlist_weights"] = _normalize_weight_triplet(0.73, 0.19, 0.08)
        out["min_hybrid_score"] = max(_safe_float(out.get("min_hybrid_score", 0.1), 0.1), 0.16)
        out["min_semantic_score"] = max(_safe_float(out.get("min_semantic_score", 0.08), 0.08), 0.09)
        out["min_keyword_score"] = max(_safe_float(out.get("min_keyword_score", 0.04), 0.04), 0.08)
        out["shortlist_min_score"] = max(_safe_float(out.get("shortlist_min_score", 0.11), 0.11), 0.15)
        out["pad_before"] = min(_safe_float(out.get("pad_before", 2.5), 2.5), 2.0)
        out["pad_after"] = min(_safe_float(out.get("pad_after", 9.0), 9.0), 8.5)
        return out

    if mode == "broad":
        out["weights"] = _normalize_weight_triplet(0.55, 0.16, 0.29)
        out["shortlist_weights"] = _normalize_weight_triplet(0.54, 0.12, 0.34)
        out["min_hybrid_score"] = min(_safe_float(out.get("min_hybrid_score", 0.1), 0.1), 0.09)
        out["min_semantic_score"] = min(_safe_float(out.get("min_semantic_score", 0.07), 0.07), 0.05)
        out["min_keyword_score"] = min(_safe_float(out.get("min_keyword_score", 0.04), 0.04), 0.03)
        out["shortlist_min_score"] = min(_safe_float(out.get("shortlist_min_score", 0.11), 0.11), 0.09)
        out["pad_before"] = max(_safe_float(out.get("pad_before", 2.5), 2.5), 3.5)
        out["pad_after"] = max(_safe_float(out.get("pad_after", 10.0), 10.0), 13.5)
        return out

    return out

def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()

def _vector_norm(vec: List[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))

def _normalize_vector(vec: List[float]) -> List[float]:
    norm = _vector_norm(vec)
    if norm <= 0.0:
        return [0.0 for _ in vec]
    return [v / norm for v in vec]

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(a[i] * a[i] for i in range(n)))
    nb = math.sqrt(sum(b[i] * b[i] for i in range(n)))
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)

def _average_vectors(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    dim = max(len(v) for v in vectors)
    if dim <= 0:
        return []
    acc = [0.0] * dim
    used = 0
    for vec in vectors:
        if not vec:
            continue
        used += 1
        for i in range(min(dim, len(vec))):
            acc[i] += vec[i]
    if used <= 0:
        return []
    return _normalize_vector([v / used for v in acc])

def _blend_vector(base: List[float], new_vec: List[float], base_count: int) -> List[float]:
    if not base:
        return new_vec[:]
    if not new_vec:
        return base[:]
    dim = max(len(base), len(new_vec))
    out = [0.0] * dim
    for i in range(dim):
        b = base[i] if i < len(base) else 0.0
        n = new_vec[i] if i < len(new_vec) else 0.0
        out[i] = ((b * base_count) + n) / max(1, base_count + 1)
    return _normalize_vector(out)

def _local_semantic_embedding(text: str, dim: int = LOCAL_EMBED_DIM) -> List[float]:
    words = re.findall(r"[a-zA-ZÀ-ÿ0-9]{2,}", str(text or "").lower())
    vec = [0.0] * dim
    if not words:
        return vec

    for w in words:
        weight = 1.0 + min(1.5, len(w) / 10.0)
        token_idx = zlib.crc32(w.encode("utf-8")) % dim
        vec[token_idx] += 0.8 * weight

        for n in (3, 4):
            if len(w) < n:
                continue
            for i in range(len(w) - n + 1):
                gram = w[i:i+n]
                idx = zlib.crc32(gram.encode("utf-8")) % dim
                vec[idx] += weight

    for i in range(len(words) - 1):
        bigram = f"{words[i]}_{words[i+1]}"
        idx = zlib.crc32(bigram.encode("utf-8")) % dim
        vec[idx] += 0.7

    return _normalize_vector(vec)

def _local_embed_texts(texts: List[str]) -> List[List[float]]:
    return [_local_semantic_embedding(t) for t in texts]

def _extract_embedding_values(raw_embedding) -> List[float]:
    if raw_embedding is None:
        return []

    values = None
    if isinstance(raw_embedding, dict):
        values = raw_embedding.get("values")
    elif isinstance(raw_embedding, (list, tuple)):
        values = raw_embedding
    else:
        values = getattr(raw_embedding, "values", None)
        if values is None:
            maybe = getattr(raw_embedding, "embedding", None)
            if maybe is not None:
                values = getattr(maybe, "values", maybe)

    if values is None:
        return []

    out: List[float] = []
    for v in values:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(0.0)
    return _normalize_vector(out)

def _embed_texts_with_gemini(texts: List[str], api_key: Optional[str]) -> Optional[List[List[float]]]:
    if not api_key or not texts:
        return None
    try:
        from google import genai
    except Exception:
        return None

    try:
        client = genai.Client(api_key=api_key)
        vectors: List[List[float]] = []
        batch_size = 48
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.models.embed_content(
                model=SEMANTIC_EMBED_MODEL,
                contents=batch
            )
            raw_embeddings = None
            if isinstance(response, dict):
                raw_embeddings = response.get("embeddings") or response.get("embedding")
            else:
                raw_embeddings = getattr(response, "embeddings", None)
                if raw_embeddings is None:
                    single = getattr(response, "embedding", None)
                    if single is not None:
                        raw_embeddings = [single]

            if raw_embeddings is None:
                return None
            if not isinstance(raw_embeddings, list):
                raw_embeddings = [raw_embeddings]

            if len(raw_embeddings) != len(batch):
                return None

            for emb in raw_embeddings:
                values = _extract_embedding_values(emb)
                if not values:
                    return None
                vectors.append(values)

        if len(vectors) != len(texts):
            return None
        return vectors
    except Exception:
        return None

def _build_search_units(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    max_unit_seconds = 26.0
    max_unit_chars = 420

    for idx, seg in enumerate(segments):
        seg_text = _normalize_space(seg.get("text", ""))
        if not seg_text:
            continue
        speaker = _normalize_space(seg.get("speaker", ""))
        start = max(0.0, _safe_float(seg.get("start", 0.0), 0.0))
        end = _safe_float(seg.get("end", start), start)
        if end <= start:
            end = start + min(4.0, max(1.0, len(seg_text.split()) * 0.35))

        if current is None:
            current = {
                "start": start,
                "end": end,
                "text": seg_text,
                "segment_indices": [idx],
                "speakers": [speaker] if speaker else []
            }
            continue

        projected_duration = end - current["start"]
        projected_chars = len(current["text"]) + 1 + len(seg_text)
        gap = start - current["end"]
        should_flush = (
            projected_duration > max_unit_seconds
            or projected_chars > max_unit_chars
            or gap > 8.0
        )
        if should_flush:
            units.append(current)
            current = {
                "start": start,
                "end": end,
                "text": seg_text,
                "segment_indices": [idx],
                "speakers": [speaker] if speaker else []
            }
            continue

        current["end"] = max(current["end"], end)
        current["text"] = f"{current['text']} {seg_text}".strip()
        current["segment_indices"].append(idx)
        if speaker and speaker not in current["speakers"]:
            current["speakers"].append(speaker)

    if current is not None:
        units.append(current)
    return units

def _find_overlapping_unit_indices(units: List[Dict[str, Any]], start: float, end: float) -> List[int]:
    overlap_indices: List[int] = []
    for i, unit in enumerate(units):
        us = _safe_float(unit.get("start", 0.0), 0.0)
        ue = _safe_float(unit.get("end", us), us)
        if ue <= start or us >= end:
            continue
        overlap_indices.append(i)
    return overlap_indices

def _unit_overlaps_range(unit: Dict[str, Any], scope_start: float, scope_end: float) -> bool:
    us = _safe_float(unit.get("start", 0.0), 0.0)
    ue = _safe_float(unit.get("end", us), us)
    if ue <= us:
        return False
    return not (ue <= scope_start or us >= scope_end)

def _clip_overlaps_range(clip: Dict[str, Any], scope_start: float, scope_end: float) -> bool:
    cs = _safe_float(clip.get("start", 0.0), 0.0)
    ce = _safe_float(clip.get("end", cs), cs)
    if ce <= cs:
        return False
    return not (ce <= scope_start or cs >= scope_end)

def _parse_scope_inputs(
    duration: float,
    chapters: List[Dict[str, Any]],
    chapter_index: Optional[int],
    start_time: Optional[float],
    end_time: Optional[float]
) -> Optional[Tuple[float, float, Dict[str, Any]]]:
    scope_start = None
    scope_end = None
    chapter_payload = None

    if chapter_index is not None:
        target = None
        for chapter in chapters or []:
            if int(chapter.get("chapter_index", -1)) == int(chapter_index):
                target = chapter
                break
        if target is None:
            raise HTTPException(status_code=400, detail="Invalid chapter_index for this transcript")
        chapter_payload = {
            "chapter_index": int(target.get("chapter_index", chapter_index)),
            "title": _normalize_space(target.get("title", "")) or f"Chapter {int(chapter_index) + 1}"
        }
        scope_start = max(0.0, _safe_float(target.get("start", 0.0), 0.0))
        scope_end = _safe_float(target.get("end", scope_start), scope_start)

    has_start = start_time is not None
    has_end = end_time is not None
    if has_start or has_end:
        user_start = max(0.0, _safe_float(start_time if has_start else scope_start, 0.0))
        user_end_default = duration if duration > 0 else max(user_start + 1.0, user_start)
        user_end = _safe_float(end_time if has_end else scope_end, user_end_default)
        if scope_start is not None and scope_end is not None:
            user_start = max(user_start, scope_start)
            user_end = min(user_end, scope_end)
        scope_start = user_start
        scope_end = user_end

    if scope_start is None or scope_end is None:
        return None

    scope_start = max(0.0, float(scope_start))
    scope_end = float(scope_end)
    if duration > 0:
        scope_end = min(scope_end, duration)
    if scope_end <= scope_start:
        raise HTTPException(status_code=400, detail="Invalid search scope: end_time must be greater than start_time")

    meta = {
        "start": round(scope_start, 3),
        "end": round(scope_end, 3),
        "duration": round(max(0.0, scope_end - scope_start), 3),
        "chapter": chapter_payload
    }
    return scope_start, scope_end, meta

def _filter_units_embeddings_and_clips(
    units: List[Dict[str, Any]],
    unit_embeddings: List[List[float]],
    clips: List[Dict[str, Any]],
    chapters: List[Dict[str, Any]],
    scope_range: Optional[Tuple[float, float, Dict[str, Any]]],
    speaker: Optional[str]
) -> Tuple[List[Dict[str, Any]], List[List[float]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    scope_meta: Dict[str, Any] = {
        "applied": False,
        "speaker": None,
        "start": None,
        "end": None,
        "duration": None,
        "chapter": None
    }
    speaker_norm = _normalize_space(speaker or "")
    speaker_l = speaker_norm.lower()

    selected_units: List[Dict[str, Any]] = []
    selected_embeddings: List[List[float]] = []
    for i, unit in enumerate(units):
        include = True
        if scope_range is not None:
            scope_start, scope_end, _ = scope_range
            include = include and _unit_overlaps_range(unit, scope_start, scope_end)
        if include and speaker_l:
            unit_speakers = [
                _normalize_space(s).lower()
                for s in (unit.get("speakers") or [])
                if _normalize_space(s)
            ]
            include = speaker_l in unit_speakers
        if not include:
            continue
        selected_units.append(unit)
        selected_embeddings.append(unit_embeddings[i] if i < len(unit_embeddings) else [])

    if not selected_units:
        raise HTTPException(status_code=404, detail="No transcript units found for selected scope/speaker")

    selected_chapters = chapters
    selected_clips = clips
    if scope_range is not None:
        scope_start, scope_end, scope_data = scope_range
        selected_chapters = [
            chapter for chapter in (chapters or [])
            if _clip_overlaps_range(chapter, scope_start, scope_end)
        ]
        selected_clips = [
            clip for clip in (clips or [])
            if _clip_overlaps_range(clip, scope_start, scope_end)
        ]
        scope_meta.update(scope_data)
        scope_meta["chapter"] = scope_data.get("chapter")
        scope_meta["applied"] = True

    if speaker_l:
        scope_meta["speaker"] = speaker_norm
        scope_meta["applied"] = True

    return selected_units, selected_embeddings, selected_clips, selected_chapters, scope_meta

def _bounded_clip_range(start: float, end: float, duration: float) -> (float, float):
    clip_start = max(0.0, float(start))
    clip_end = max(clip_start, float(end))
    if clip_end - clip_start < 15.0:
        clip_end = clip_start + 15.0
    if clip_end - clip_start > 60.0:
        clip_end = clip_start + 60.0

    if duration > 0:
        if clip_start >= duration:
            clip_start = max(0.0, duration - 15.0)
        clip_end = min(clip_end, duration)
        if clip_end <= clip_start:
            clip_end = min(duration, clip_start + 15.0)
        if clip_end <= clip_start:
            clip_start = max(0.0, clip_end - 1.0)
    return clip_start, clip_end

def _extract_topic_keywords(text: str, limit: int = 4) -> List[str]:
    stop = {
        "about", "after", "algo", "algun", "algunas", "algunos", "antes", "aqui", "also", "aunque",
        "because", "between", "cada", "como", "con", "cuando", "donde", "este", "esta", "estos",
        "estas", "from", "haber", "hasta", "into", "para", "pero", "porque", "sobre", "still",
        "that", "their", "them", "there", "they", "this", "those", "todos", "todas", "very", "video",
        "while", "with", "would", "you", "your", "clip", "short", "shorts", "momento", "momento"
    }
    tokens = re.findall(r"[a-zA-ZÀ-ÿ0-9]{4,}", str(text or "").lower())
    counts = Counter(t for t in tokens if t not in stop)
    return [w for (w, _) in counts.most_common(limit)]

def _build_chapter_payload(chapter_idx: int, units_slice: List[Dict[str, Any]]) -> Dict[str, Any]:
    start = _safe_float(units_slice[0].get("start", 0.0), 0.0)
    end = _safe_float(units_slice[-1].get("end", start), start)
    text = _normalize_space(" ".join(str(u.get("text", "")) for u in units_slice))
    keywords = _extract_topic_keywords(text, limit=4)
    if keywords:
        title = " / ".join(keywords[:2])
    else:
        title = text[:52] + ("..." if len(text) > 52 else "")
        if not title:
            title = f"Chapter {chapter_idx + 1}"

    return {
        "chapter_index": chapter_idx,
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(max(0.0, end - start), 3),
        "title": title,
        "keywords": keywords,
        "snippet": text[:240]
    }

def _generate_auto_chapters(units: List[Dict[str, Any]], unit_embeddings: List[List[float]], total_duration: float) -> List[Dict[str, Any]]:
    if not units or total_duration < 480:
        return []

    if total_duration >= 3600:
        min_duration = 150.0
        max_duration = 420.0
    elif total_duration >= 1800:
        min_duration = 120.0
        max_duration = 300.0
    else:
        min_duration = 75.0
        max_duration = 210.0

    split_threshold = 0.52
    chapters: List[Dict[str, Any]] = []
    start_idx = 0
    centroid = unit_embeddings[0] if unit_embeddings else []
    centroid_count = 1

    for i in range(1, len(units)):
        prev_unit = units[i - 1]
        cur_unit = units[i]
        chapter_start = _safe_float(units[start_idx].get("start", 0.0), 0.0)
        chapter_end = _safe_float(cur_unit.get("end", chapter_start), chapter_start)
        chapter_duration = chapter_end - chapter_start

        similarity = _cosine_similarity(centroid, unit_embeddings[i]) if unit_embeddings else 1.0
        gap = _safe_float(cur_unit.get("start", 0.0), 0.0) - _safe_float(prev_unit.get("end", 0.0), 0.0)
        should_split = False
        if chapter_duration >= max_duration:
            should_split = True
        elif chapter_duration >= min_duration and similarity < split_threshold:
            should_split = True
        elif chapter_duration >= (min_duration * 0.7) and gap > 12.0:
            should_split = True

        if should_split:
            slice_units = units[start_idx:i]
            if slice_units:
                chapters.append(_build_chapter_payload(len(chapters), slice_units))
            start_idx = i
            centroid = unit_embeddings[i] if unit_embeddings else []
            centroid_count = 1
        else:
            centroid = _blend_vector(centroid, unit_embeddings[i] if unit_embeddings else [], centroid_count)
            centroid_count += 1

    tail = units[start_idx:]
    if tail:
        chapters.append(_build_chapter_payload(len(chapters), tail))
    return chapters

def _virality_overlap_score(start: float, end: float, clips: List[Dict[str, Any]]) -> (float, Optional[Dict[str, Any]]):
    best_score = 0.0
    best_clip = None
    candidate_duration = max(0.001, end - start)
    center = (start + end) / 2.0

    for clip in clips:
        cs = _safe_float(clip.get("start", 0.0), 0.0)
        ce = _safe_float(clip.get("end", cs), cs)
        if ce <= cs:
            continue
        virality = max(0.0, min(1.0, _safe_float(clip.get("virality_score", 0), 0.0) / 100.0))
        intersection = max(0.0, min(end, ce) - max(start, cs))
        if intersection > 0:
            overlap_ratio = intersection / candidate_duration
            score = min(1.0, overlap_ratio * 1.4) * virality
        else:
            clip_center = (cs + ce) / 2.0
            dist = abs(clip_center - center)
            if dist > 20.0:
                continue
            proximity = (20.0 - dist) / 20.0
            score = 0.15 * proximity * virality

        if score > best_score:
            best_score = score
            best_clip = clip

    return best_score, best_clip

def _ensure_search_index(
    job_id: str,
    metadata_path: str,
    transcript: Dict[str, Any],
    clips: List[Dict[str, Any]],
    semantic_api_key: Optional[str]
) -> Dict[str, Any]:
    mtime = os.path.getmtime(metadata_path) if os.path.exists(metadata_path) else 0.0
    cached = SEARCH_INDEX_CACHE.get(job_id)
    stale = (
        not cached
        or cached.get("metadata_path") != metadata_path
        or _safe_float(cached.get("metadata_mtime", 0.0), 0.0) != mtime
    )

    segments = transcript.get("segments") if isinstance(transcript, dict) else []
    if not isinstance(segments, list):
        segments = []
    transcript_text = _normalize_space(transcript.get("text", "") if isinstance(transcript, dict) else "")
    normalized_clips = [
        _normalize_clip_payload(
            dict(clip) if isinstance(clip, dict) else {},
            idx,
            transcript=transcript
        )
        for idx, clip in enumerate(clips or [])
    ]

    if stale:
        units = _build_search_units(segments)
        unit_texts = [u["text"] for u in units]
        local_embeddings = _local_embed_texts(unit_texts)
        duration = 0.0
        speakers_seen = set()
        for unit in units:
            duration = max(duration, _safe_float(unit.get("end", 0.0), 0.0))
            for speaker in (unit.get("speakers") or []):
                speaker_norm = _normalize_space(speaker)
                if speaker_norm:
                    speakers_seen.add(speaker_norm)

        cached = {
            "metadata_path": metadata_path,
            "metadata_mtime": mtime,
            "units": units,
            "unit_texts": unit_texts,
            "unit_embeddings_local": local_embeddings,
            "unit_embeddings_semantic": None,
            "duration": duration,
            "transcript_text": transcript_text,
            "chapters_by_provider": {},
            "clips": normalized_clips,
            "speakers": sorted(speakers_seen)
        }
        SEARCH_INDEX_CACHE[job_id] = cached
    else:
        cached["clips"] = normalized_clips
        cached["transcript_text"] = transcript_text
        if not isinstance(cached.get("speakers"), list):
            speakers_seen = set()
            for unit in (cached.get("units") or []):
                for speaker in (unit.get("speakers") or []):
                    speaker_norm = _normalize_space(speaker)
                    if speaker_norm:
                        speakers_seen.add(speaker_norm)
            cached["speakers"] = sorted(speakers_seen)

    if semantic_api_key and cached.get("unit_texts") and not cached.get("unit_embeddings_semantic"):
        semantic_vectors = _embed_texts_with_gemini(cached["unit_texts"], semantic_api_key)
        if semantic_vectors and len(semantic_vectors) == len(cached["unit_texts"]):
            cached["unit_embeddings_semantic"] = semantic_vectors
            cached["chapters_by_provider"] = {}

    provider = "gemini" if (semantic_api_key and cached.get("unit_embeddings_semantic")) else "local"
    unit_embeddings = cached.get("unit_embeddings_semantic") if provider == "gemini" else cached.get("unit_embeddings_local")
    unit_embeddings = unit_embeddings or []

    chapters_by_provider = cached.get("chapters_by_provider") or {}
    if provider not in chapters_by_provider:
        chapters_by_provider[provider] = _generate_auto_chapters(
            cached.get("units") or [],
            unit_embeddings,
            _safe_float(cached.get("duration", 0.0), 0.0)
        )
        cached["chapters_by_provider"] = chapters_by_provider

    return {
        "units": cached.get("units") or [],
        "unit_embeddings": unit_embeddings,
        "provider": provider,
        "chapters": chapters_by_provider.get(provider) or [],
        "duration": _safe_float(cached.get("duration", 0.0), 0.0),
        "transcript_text": cached.get("transcript_text", ""),
        "clips": cached.get("clips") or [],
        "speakers": cached.get("speakers") or []
    }

def _build_semantic_matches(
    query: str,
    keywords: List[str],
    phrases: List[str],
    query_profile: Dict[str, Any],
    units: List[Dict[str, Any]],
    unit_embeddings: List[List[float]],
    query_embedding: List[float],
    transcript_text: str,
    clips: List[Dict[str, Any]],
    duration: float,
    limit: int
) -> List[Dict[str, Any]]:
    query_l = query.lower()
    transcript_l = transcript_text.lower()
    weights = query_profile.get("weights", (0.62, 0.23, 0.15))
    w_sem, w_kw, w_vir = _normalize_weight_triplet(weights[0], weights[1], weights[2])
    min_hybrid = _safe_float(query_profile.get("min_hybrid_score", 0.08), 0.08)
    min_semantic = _safe_float(query_profile.get("min_semantic_score", 0.08), 0.08)
    min_keyword = _safe_float(query_profile.get("min_keyword_score", 0.0), 0.0)
    pad_before = _safe_float(query_profile.get("pad_before", 3.0), 3.0)
    pad_after = _safe_float(query_profile.get("pad_after", 12.0), 12.0)
    mode = str(query_profile.get("mode", "topic"))
    ranked: List[Dict[str, Any]] = []

    for idx, unit in enumerate(units):
        unit_text = _normalize_space(unit.get("text", ""))
        if not unit_text:
            continue
        unit_l = unit_text.lower()
        matched_keywords: List[str] = []
        keyword_score = 0.0
        for kw in keywords:
            if kw in unit_l:
                matched_keywords.append(kw)
                keyword_score += 1.0

        phrase_hits: List[str] = []
        phrase_score = 0.0
        for phrase in phrases:
            if phrase in unit_l:
                phrase_hits.append(phrase)
                phrase_score += 1.6
            elif phrase in transcript_l:
                phrase_score += 0.2

        if query_l in unit_l:
            keyword_score += 1.5
        elif query_l and query_l in transcript_l:
            keyword_score += 0.25

        semantic_score = 0.0
        if idx < len(unit_embeddings):
            semantic_score = max(0.0, _cosine_similarity(query_embedding, unit_embeddings[idx]))

        kw_norm = min(1.0, keyword_score / max(1.0, len(keywords) + 1.5))
        phrase_norm = min(1.0, phrase_score / max(1.0, (len(phrases) * 1.6) or 1.0))
        keyword_channel = min(1.0, (kw_norm * 0.72) + (phrase_norm * 0.28))
        if phrase_hits:
            keyword_channel = min(1.0, keyword_channel + 0.08)

        base_start = _safe_float(unit.get("start", 0.0), 0.0)
        base_end = _safe_float(unit.get("end", base_start), base_start)
        virality_boost, overlap_clip = _virality_overlap_score(base_start, base_end, clips)

        hybrid_score = (w_sem * semantic_score) + (w_kw * keyword_channel) + (w_vir * virality_boost)
        if mode == "exact_phrase" and phrases and phrase_hits:
            hybrid_score = min(1.0, hybrid_score + 0.04)

        passes_threshold = (
            hybrid_score >= min_hybrid
            and (
                semantic_score >= min_semantic
                or keyword_channel >= min_keyword
                or virality_boost >= 0.42
            )
        )
        if mode == "exact_phrase" and phrases and not phrase_hits and semantic_score < (min_semantic + 0.04):
            passes_threshold = False

        if not passes_threshold:
            continue

        clip_start, clip_end = _bounded_clip_range(base_start - pad_before, base_end + pad_after, duration)
        ranked.append({
            "unit_index": idx,
            "start": clip_start,
            "end": clip_end,
            "duration": max(0.0, clip_end - clip_start),
            "match_score": hybrid_score,
            "semantic_score": semantic_score,
            "keyword_score": keyword_channel,
            "phrase_score": phrase_norm,
            "virality_boost": virality_boost,
            "keywords": matched_keywords,
            "phrases": phrase_hits,
            "snippet": unit_text[:240],
            "speakers": unit.get("speakers") or [],
            "source_clip_index": overlap_clip.get("clip_index") if overlap_clip else None,
            "source_clip_virality": overlap_clip.get("virality_score") if overlap_clip else None
        })

    ranked.sort(
        key=lambda x: (
            x["match_score"],
            x["semantic_score"],
            x["keyword_score"],
            x["phrase_score"],
            x["virality_boost"],
            -x["unit_index"]
        ),
        reverse=True
    )

    results: List[Dict[str, Any]] = []
    used_ranges: List[tuple] = []
    for candidate in ranked:
        overlap = False
        for s0, e0 in used_ranges:
            if not (candidate["end"] <= s0 or candidate["start"] >= e0):
                overlap = True
                break
        if overlap:
            continue
        used_ranges.append((candidate["start"], candidate["end"]))
        results.append({
            "start": round(candidate["start"], 3),
            "end": round(candidate["end"], 3),
            "duration": round(candidate["duration"], 3),
            "match_score": round(candidate["match_score"], 4),
            "semantic_score": round(candidate["semantic_score"], 4),
            "keyword_score": round(candidate["keyword_score"], 4),
            "phrase_score": round(candidate["phrase_score"], 4),
            "virality_boost": round(candidate["virality_boost"], 4),
            "keywords": candidate["keywords"],
            "phrases": candidate["phrases"],
            "snippet": candidate["snippet"],
            "speakers": candidate["speakers"][:3] if isinstance(candidate["speakers"], list) else [],
            "source_clip_index": candidate["source_clip_index"],
            "source_clip_virality": candidate["source_clip_virality"]
        })
        if len(results) >= limit:
            break
    return results

def _build_hybrid_shortlist(
    clips: List[Dict[str, Any]],
    units: List[Dict[str, Any]],
    unit_embeddings: List[List[float]],
    query_embedding: List[float],
    keywords: List[str],
    phrases: List[str],
    query_profile: Dict[str, Any],
    limit: int
) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    if not clips or not units:
        return ranked

    shortlist_weights = query_profile.get("shortlist_weights", (0.64, 0.16, 0.2))
    w_sem, w_lex, w_vir = _normalize_weight_triplet(shortlist_weights[0], shortlist_weights[1], shortlist_weights[2])
    min_shortlist_score = _safe_float(query_profile.get("shortlist_min_score", 0.11), 0.11)

    for clip in clips:
        start = _safe_float(clip.get("start", 0.0), 0.0)
        end = _safe_float(clip.get("end", start), start)
        if end <= start:
            continue

        overlap_indices = _find_overlapping_unit_indices(units, start, end)
        if not overlap_indices:
            continue
        vectors = [
            unit_embeddings[i]
            for i in overlap_indices
            if i < len(unit_embeddings) and unit_embeddings[i]
        ]
        clip_vec = _average_vectors(vectors)
        semantic_score = max(0.0, _cosine_similarity(query_embedding, clip_vec)) if clip_vec else 0.0
        virality_score = max(0.0, min(100.0, _safe_float(clip.get("virality_score", 0.0), 0.0)))
        virality_norm = virality_score / 100.0

        lexical_text = _normalize_space(" ".join([
            str(clip.get("video_title_for_youtube_short", "")),
            str(clip.get("video_description_for_tiktok", "")),
            str(clip.get("video_description_for_instagram", "")),
            " ".join(clip.get("topic_tags", []) or [])
        ])).lower()
        lexical_hits = 0.0
        for kw in keywords:
            if kw in lexical_text:
                lexical_hits += 1.0
        for phrase in phrases:
            if phrase in lexical_text:
                lexical_hits += 1.6
        lexical_norm = min(1.0, lexical_hits / max(1.0, len(keywords) + (len(phrases) * 1.6)))
        hybrid_score = (w_sem * semantic_score) + (w_lex * lexical_norm) + (w_vir * virality_norm)
        if hybrid_score < min_shortlist_score:
            continue

        ranked.append({
            "clip_index": int(clip.get("clip_index", 0)),
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(max(0.0, end - start), 3),
            "title": _normalize_space(clip.get("video_title_for_youtube_short", ""))[:140] or f"Clip {int(clip.get('clip_index', 0)) + 1}",
            "virality_score": int(round(virality_score)),
            "semantic_score": round(semantic_score, 4),
            "lexical_score": round(lexical_norm, 4),
            "hybrid_score": round(hybrid_score, 4)
        })

    ranked.sort(key=lambda x: (x["hybrid_score"], x["semantic_score"], x["lexical_score"], x["virality_score"]), reverse=True)
    return ranked[:max(1, min(12, limit))]

def _clip_transcript_excerpt(
    clip: Dict[str, Any],
    transcript: Optional[Dict[str, Any]],
    max_chars: int = 420,
    max_segments: int = 8
) -> str:
    if not isinstance(clip, dict) or not isinstance(transcript, dict):
        return ""

    segments = transcript.get("segments") if isinstance(transcript.get("segments"), list) else []
    if not segments:
        return ""

    start = max(0.0, _safe_float(clip.get("start", 0.0), 0.0))
    end = max(start, _safe_float(clip.get("end", start), start))
    if end <= start:
        return ""

    chunks: List[str] = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        text = _normalize_space(seg.get("text", ""))
        if not text:
            continue
        ss = max(0.0, _safe_float(seg.get("start", 0.0), 0.0))
        se = max(ss, _safe_float(seg.get("end", ss), ss))
        if se <= start or ss >= end:
            continue
        chunks.append(text)
        if len(chunks) >= max_segments:
            break

    if not chunks:
        return ""

    excerpt = _normalize_space(" ".join(chunks))
    if len(excerpt) > max_chars:
        excerpt = f"{excerpt[:max_chars - 3].rstrip()}..."
    return excerpt

def _normalize_clip_payload(clip: Dict, rank: int, transcript: Optional[Dict[str, Any]] = None) -> Dict:
    """
    Ensure clip carries stable metadata used by the dashboard:
    - clip_index
    - virality_score
    - score_reason
    """
    if not isinstance(clip, dict):
        clip = {}

    clip['clip_index'] = int(clip.get('clip_index', rank))

    raw_score = clip.get('virality_score')
    try:
        score = int(round(float(raw_score)))
    except (TypeError, ValueError):
        score = _default_score_by_rank(rank)
    clip['virality_score'] = max(0, min(100, score))
    clip['score_band'] = _score_band(clip['virality_score'])
    clip['selection_confidence'] = _normalize_confidence(clip.get('selection_confidence'), clip['virality_score'])

    reason = clip.get('score_reason')
    if not reason:
        reason = f"Ranking IA #{rank+1}: buen gancho inicial y alto potencial de retención."
    clip['score_reason'] = str(reason).strip()[:220]

    tags = _normalize_topic_tags(clip.get('topic_tags'))
    if not tags:
        tags = _default_topic_tags(clip)
    clip['topic_tags'] = tags
    raw_aspect = str(clip.get('aspect_ratio', '9:16')).strip().replace("/", ":")
    clip['aspect_ratio'] = raw_aspect if raw_aspect in ALLOWED_ASPECT_RATIOS else "9:16"

    if not _normalize_space(clip.get("transcript_excerpt", "")):
        excerpt = _clip_transcript_excerpt(clip, transcript)
        if excerpt:
            clip["transcript_excerpt"] = excerpt
    if not _normalize_space(clip.get("transcript_text", "")) and _normalize_space(clip.get("transcript_excerpt", "")):
        clip["transcript_text"] = clip["transcript_excerpt"]

    return clip

async def cleanup_jobs():
    """Background task to remove old jobs and files."""
    import time
    print("🧹 Cleanup task started.")
    while True:
        try:
            await asyncio.sleep(300) # Check every 5 minutes
            now = time.time()
            
            # Simple directory cleanup based on modification time
            # Check OUTPUT_DIR
            for job_id in os.listdir(OUTPUT_DIR):
                job_path = os.path.join(OUTPUT_DIR, job_id)
                if os.path.isdir(job_path):
                    if now - os.path.getmtime(job_path) > JOB_RETENTION_SECONDS:
                        print(f"🧹 Purging old job: {job_id}")
                        shutil.rmtree(job_path, ignore_errors=True)
                        if job_id in jobs:
                            del jobs[job_id]
                        SEARCH_INDEX_CACHE.pop(job_id, None)

            # Cleanup Uploads
            for filename in os.listdir(UPLOAD_DIR):
                file_path = os.path.join(UPLOAD_DIR, filename)
                try:
                    if now - os.path.getmtime(file_path) > JOB_RETENTION_SECONDS:
                         os.remove(file_path)
                except Exception: pass

        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")

async def process_queue():
    """Background worker to process jobs from the queue with concurrency limit."""
    print(f"🚀 Job Queue Worker started with {MAX_CONCURRENT_JOBS} concurrent slots.")
    while True:
        try:
            # Wait for a job
            job_id = await job_queue.get()
            
            # Acquire semaphore slot (waits if max jobs are running)
            await concurrency_semaphore.acquire()
            print(f"🔄 Acquired slot for job: {job_id}")

            # Process in background task to not block the loop (allowing other slots to fill)
            asyncio.create_task(run_job_wrapper(job_id))
            
        except Exception as e:
            print(f"❌ Queue dispatch error: {e}")
            await asyncio.sleep(1)

async def run_job_wrapper(job_id):
    """Wrapper to run job and release semaphore"""
    try:
        job = jobs.get(job_id)
        if job:
            await run_job(job_id, job)
    except Exception as e:
         print(f"❌ Job wrapper error {job_id}: {e}")
    finally:
        # Always release semaphore and mark queue task done
        concurrency_semaphore.release()
        job_queue.task_done()
        print(f"✅ Released slot for job: {job_id}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start worker and cleanup
    worker_task = asyncio.create_task(process_queue())
    cleanup_task = asyncio.create_task(cleanup_jobs())
    yield
    # Cleanup (optional: cancel worker)

app = FastAPI(lifespan=lifespan)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving videos
app.mount("/videos", StaticFiles(directory=OUTPUT_DIR), name="videos")

class ProcessRequest(BaseModel):
    url: str
    aspect_ratio: Optional[str] = "9:16"

def enqueue_output(out, job_id):
    """Reads output from a subprocess and appends it to jobs logs."""
    try:
        for line in iter(out.readline, b''):
            decoded_line = line.decode('utf-8').strip()
            if decoded_line:
                print(f"📝 [Job Output] {decoded_line}")
                if job_id in jobs:
                    jobs[job_id]['logs'].append(decoded_line)
    except Exception as e:
        print(f"Error reading output for job {job_id}: {e}")
    finally:
        out.close()

def _reset_job_output_for_retry(output_dir: Optional[str]):
    if not output_dir:
        return
    try:
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
        os.makedirs(output_dir, exist_ok=True)
    except Exception:
        pass

async def _delayed_retry_enqueue(job_id: str, delay_seconds: int, trigger: str):
    await asyncio.sleep(max(0, int(delay_seconds)))
    job = jobs.get(job_id)
    if not job:
        return
    if job.get('status') in ('completed', 'processing'):
        return
    job['status'] = 'queued'
    job['logs'].append(f"Retry enqueued ({trigger}).")
    await job_queue.put(job_id)

async def _queue_job_retry(job_id: str, reason: str, trigger: str = "auto", delay_seconds: Optional[int] = None) -> Tuple[bool, str]:
    job = jobs.get(job_id)
    if not job:
        return False, "Job not found"

    status = str(job.get('status', ''))
    if trigger == "manual" and status in ("queued", "processing", "retrying"):
        return False, f"Job is already {status}"

    delay = max(0, int(delay_seconds if delay_seconds is not None else job.get('retry_delay_seconds', JOB_RETRY_DELAY_SECONDS_DEFAULT)))
    msg = ""

    if trigger == "auto":
        max_auto = max(0, int(job.get('max_auto_retries', MAX_AUTO_RETRIES_DEFAULT)))
        used = max(0, int(job.get('auto_retry_count', 0)))
        if used >= max_auto:
            return False, f"Auto retry limit reached ({used}/{max_auto})"
        job['auto_retry_count'] = used + 1
        retry_num = int(job['auto_retry_count'])
        delay = min(180, delay * retry_num if delay > 0 else 0)
        msg = f"Auto-retry {retry_num}/{max_auto} scheduled in {delay}s. Reason: {reason}"
    else:
        job['manual_retry_count'] = max(0, int(job.get('manual_retry_count', 0))) + 1
        retry_num = int(job['manual_retry_count'])
        msg = f"Manual retry #{retry_num} queued. Reason: {reason}"

    # Drop stale artifacts/results before retrying.
    _reset_job_output_for_retry(job.get('output_dir'))
    SEARCH_INDEX_CACHE.pop(job_id, None)
    job.pop('result', None)
    job['last_error'] = reason
    job['logs'].append(msg)

    if delay > 0:
        job['status'] = 'retrying'
        asyncio.create_task(_delayed_retry_enqueue(job_id, delay, trigger))
    else:
        job['status'] = 'queued'
        job['logs'].append(f"Retry enqueued ({trigger}).")
        await job_queue.put(job_id)

    return True, msg

async def run_job(job_id, job_data):
    """Executes the subprocess for a specific job."""
    
    cmd = job_data['cmd']
    env = job_data['env']
    output_dir = job_data['output_dir']

    jobs[job_id]['attempt_count'] = max(0, int(jobs[job_id].get('attempt_count', 0))) + 1
    attempt_count = int(jobs[job_id]['attempt_count'])
    jobs[job_id]['status'] = 'processing'
    jobs[job_id]['logs'].append(f"Job attempt #{attempt_count} started by worker.")
    print(f"🎬 [run_job] Executing command for {job_id}: {' '.join(cmd)}")

    async def _fail_or_retry(reason: str):
        jobs[job_id]['last_error'] = reason
        retry_ok, _ = await _queue_job_retry(job_id, reason, trigger="auto")
        if retry_ok:
            return
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['logs'].append(reason)
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Merge stderr to stdout
            env=env,
            cwd=os.getcwd()
        )
        
        # We need to capture logs in a thread because Popen isn't async
        t_log = threading.Thread(target=enqueue_output, args=(process.stdout, job_id))
        t_log.daemon = True
        t_log.start()
        
        # Async wait for process with incremental updates
        start_wait = time.time()
        while process.poll() is None:
            await asyncio.sleep(2)
            
            # Check for partial results every 2 seconds
            # Look for metadata file
            try:
                json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
                if json_files:
                    target_json = json_files[0]
                    # Read metadata (it might be being written to, so simple try/except or just read)
                    # Use a lock or just robust read? json.load might fail if file is partial.
                    # Usually main.py writes it once at start (based on my review).
                    if os.path.getsize(target_json) > 0:
                        with open(target_json, 'r') as f:
                            data = json.load(f)
                            
                        base_name = os.path.basename(target_json).replace('_metadata.json', '')
                        clips = data.get('shorts', [])
                        cost_analysis = data.get('cost_analysis')
                        
                        # Check which clips actually exist on disk
                        ready_clips = []
                        transcript_data = data.get('transcript') if isinstance(data, dict) else None
                        for i, clip in enumerate(clips):
                             clip = _normalize_clip_payload(clip, i, transcript=transcript_data)
                             clip_filename = f"{base_name}_clip_{i+1}.mp4"
                             clip_path = os.path.join(output_dir, clip_filename)
                             if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
                                 # Checking if file is growing? For now assume if it exists and main.py moves it there, it's done.
                                 # main.py writes to temp_... then moves to final name. So presence means ready!
                                 clip['video_url'] = f"/videos/{job_id}/{clip_filename}"
                                 ready_clips.append(clip)
                        
                        if ready_clips:
                             jobs[job_id]['result'] = {'clips': ready_clips, 'cost_analysis': cost_analysis}
            except Exception as e:
                # Ignore read errors during processing
                pass

        returncode = process.returncode
        
        if returncode == 0:
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['logs'].append("Process finished successfully.")
            
            # Start S3 upload in background (silent, non-blocking)
            loop = asyncio.get_event_loop()
            loop.run_in_executor(None, upload_job_artifacts, output_dir, job_id)
            
            # Find result JSON
            json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
            if not json_files:
                # Backward-compat rescue if outputs were written to OUTPUT_DIR root
                if _relocate_root_job_artifacts(job_id, output_dir):
                    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
            if json_files:
                target_json = json_files[0] 
                with open(target_json, 'r') as f:
                    data = json.load(f)
                
                # Enhance result with video URLs
                base_name = os.path.basename(target_json).replace('_metadata.json', '')
                clips = data.get('shorts', [])
                cost_analysis = data.get('cost_analysis')

                transcript_data = data.get('transcript') if isinstance(data, dict) else None
                for i, clip in enumerate(clips):
                     clip = _normalize_clip_payload(clip, i, transcript=transcript_data)
                     clip_filename = f"{base_name}_clip_{i+1}.mp4"
                     clip['video_url'] = f"/videos/{job_id}/{clip_filename}"
                
                jobs[job_id]['result'] = {'clips': clips, 'cost_analysis': cost_analysis}
            else:
                 await _fail_or_retry("No metadata file generated.")
        else:
            await _fail_or_retry(f"Process failed with exit code {returncode}")
            
    except Exception as e:
        await _fail_or_retry(f"Execution error: {str(e)}")

@app.post("/api/process")
async def process_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    max_clips: Optional[int] = Form(None),
    whisper_backend: Optional[str] = Form(None),
    whisper_model: Optional[str] = Form(None),
    word_timestamps: Optional[str] = Form(None),
    ffmpeg_preset: Optional[str] = Form(None),
    ffmpeg_crf: Optional[int] = Form(None),
    aspect_ratio: Optional[str] = Form(None),
    clip_length_target: Optional[str] = Form(None),
    style_template: Optional[str] = Form(None),
    content_profile: Optional[str] = Form(None),
    max_auto_retries: Optional[int] = Form(None),
    retry_delay_seconds: Optional[int] = Form(None)
):
    api_key = request.headers.get("X-Gemini-Key")
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing X-Gemini-Key header")
    
    # Handle JSON body manually for URL payload
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        url = body.get("url")
        language = body.get("language") or language
        max_clips = body.get("max_clips") or max_clips
        whisper_backend = body.get("whisper_backend") or whisper_backend
        whisper_model = body.get("whisper_model") or whisper_model
        word_timestamps = body.get("word_timestamps") or word_timestamps
        ffmpeg_preset = body.get("ffmpeg_preset") or ffmpeg_preset
        ffmpeg_crf = body.get("ffmpeg_crf") or ffmpeg_crf
        aspect_ratio = body.get("aspect_ratio") or aspect_ratio
        clip_length_target = body.get("clip_length_target") or clip_length_target
        style_template = body.get("style_template") or style_template
        content_profile = body.get("content_profile") or content_profile
        if body.get("max_auto_retries") is not None:
            max_auto_retries = body.get("max_auto_retries")
        if body.get("retry_delay_seconds") is not None:
            retry_delay_seconds = body.get("retry_delay_seconds")
    
    if not url and not file:
        raise HTTPException(status_code=400, detail="Must provide URL or File")
    aspect_ratio = normalize_aspect_ratio(aspect_ratio)
    if clip_length_target is not None:
        clip_length_target = str(clip_length_target).strip().lower()
        if clip_length_target and clip_length_target not in ALLOWED_CLIP_LENGTH_TARGETS:
            raise HTTPException(status_code=400, detail="Invalid clip_length_target. Allowed values: short, balanced, long")
    style_template = _normalize_space(style_template or "") or None
    if style_template:
        style_template = re.sub(r"[^a-zA-Z0-9_-]", "", style_template)[:40] or None
    content_profile = _normalize_space(content_profile or "") or None
    if content_profile:
        content_profile = re.sub(r"[^a-zA-Z0-9_-]", "", content_profile)[:40] or None
    max_auto_retries = max(0, min(5, int(max_auto_retries if max_auto_retries is not None else MAX_AUTO_RETRIES_DEFAULT)))
    retry_delay_seconds = max(0, min(300, int(retry_delay_seconds if retry_delay_seconds is not None else JOB_RETRY_DELAY_SECONDS_DEFAULT)))

    job_id = str(uuid.uuid4())
    job_output_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_output_dir, exist_ok=True)
    
    # Prepare Command
    python_bin = sys.executable or shutil.which("python3") or "python3"
    cmd = [python_bin, "-u", "main.py"] # -u for unbuffered
    env = os.environ.copy()
    env["GEMINI_API_KEY"] = api_key # Override with key from request
    
    if url:
        cmd.extend(["-u", url])
    else:
        # Save uploaded file with size limit check
        input_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
        
        # Read file in chunks to check size
        size = 0
        limit_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
        
        with open(input_path, "wb") as buffer:
            while content := await file.read(1024 * 1024): # Read 1MB chunks
                size += len(content)
                if size > limit_bytes:
                    os.remove(input_path)
                    shutil.rmtree(job_output_dir)
                    raise HTTPException(status_code=413, detail=f"File too large. Max size {MAX_FILE_SIZE_MB}MB")
                buffer.write(content)
                
        cmd.extend(["-i", input_path])

    if language:
        cmd.extend(["--language", str(language)])
    if max_clips:
        cmd.extend(["--max-clips", str(max_clips)])
    if whisper_backend:
        cmd.extend(["--whisper-backend", str(whisper_backend)])
    if whisper_model:
        cmd.extend(["--whisper-model", str(whisper_model)])
    if word_timestamps is not None:
        cmd.extend(["--word-timestamps", str(word_timestamps)])
    if ffmpeg_preset:
        cmd.extend(["--ffmpeg-preset", str(ffmpeg_preset)])
    if ffmpeg_crf:
        cmd.extend(["--ffmpeg-crf", str(ffmpeg_crf)])
    if aspect_ratio:
        cmd.extend(["--aspect-ratio", aspect_ratio])
    if clip_length_target:
        cmd.extend(["--clip-length-target", clip_length_target])
    if style_template:
        cmd.extend(["--style-template", style_template])
    if content_profile:
        cmd.extend(["--content-profile", content_profile])

    cmd.extend(["-o", job_output_dir])

    # Enqueue Job
    jobs[job_id] = {
        'status': 'queued',
        'logs': [f"Job {job_id} queued."],
        'cmd': cmd,
        'env': env,
        'output_dir': job_output_dir,
        'input_path': input_path if not url else None,
        'attempt_count': 0,
        'auto_retry_count': 0,
        'manual_retry_count': 0,
        'max_auto_retries': max_auto_retries,
        'retry_delay_seconds': retry_delay_seconds,
        'last_error': None
    }
    
    await job_queue.put(job_id)
    
    return {"job_id": job_id, "status": "queued"}

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return {
        "status": job['status'],
        "logs": job['logs'],
        "result": job.get('result'),
        "retry": {
            "attempt_count": int(job.get('attempt_count', 0)),
            "auto_retry_count": int(job.get('auto_retry_count', 0)),
            "manual_retry_count": int(job.get('manual_retry_count', 0)),
            "max_auto_retries": int(job.get('max_auto_retries', MAX_AUTO_RETRIES_DEFAULT)),
            "retry_delay_seconds": int(job.get('retry_delay_seconds', JOB_RETRY_DELAY_SECONDS_DEFAULT)),
            "last_error": job.get('last_error')
        }
    }

@app.post("/api/retry/{job_id}")
async def retry_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job.get('status') != 'failed':
        raise HTTPException(status_code=409, detail=f"Job is {job.get('status')}. Manual retry is only allowed for failed jobs.")
    ok, msg = await _queue_job_retry(job_id, reason="Manual retry requested", trigger="manual", delay_seconds=0)
    if not ok:
        raise HTTPException(status_code=409, detail=msg)

    return {
        "job_id": job_id,
        "status": job.get('status', 'queued'),
        "message": msg,
        "retry": {
            "attempt_count": int(job.get('attempt_count', 0)),
            "auto_retry_count": int(job.get('auto_retry_count', 0)),
            "manual_retry_count": int(job.get('manual_retry_count', 0)),
            "max_auto_retries": int(job.get('max_auto_retries', MAX_AUTO_RETRIES_DEFAULT))
        }
    }

from editor import VideoEditor
from subtitles import generate_srt, burn_subtitles

class EditRequest(BaseModel):
    job_id: str
    clip_index: int
    api_key: Optional[str] = None
    input_filename: Optional[str] = None

@app.post("/api/edit")
async def edit_clip(
    req: EditRequest,
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key")
):
    # Determine API Key
    final_api_key = req.api_key or x_gemini_key or os.environ.get("GEMINI_API_KEY")
    
    if not final_api_key:
        raise HTTPException(status_code=400, detail="Missing Gemini API Key (Header or Body)")

    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[req.job_id]
    if 'result' not in job or 'clips' not in job['result']:
        raise HTTPException(status_code=400, detail="Job result not available")
        
    try:
        # Resolve Input Path: Prefer explict input_filename from frontend (chaining edits)
        if req.input_filename:
            # Security: Ensure just a filename, no paths
            safe_name = os.path.basename(req.input_filename)
            input_path = os.path.join(OUTPUT_DIR, req.job_id, safe_name)
            filename = safe_name
        else:
            # Fallback to original clip
            clip = job['result']['clips'][req.clip_index]
            filename = clip['video_url'].split('/')[-1]
            input_path = os.path.join(OUTPUT_DIR, req.job_id, filename)
        
        if not os.path.exists(input_path):
             raise HTTPException(status_code=404, detail=f"Video file not found: {input_path}")

        # Define output path for edited video
        base_name = os.path.splitext(filename)[0]
        edited_filename = f"edited_{base_name}.mp4"
        output_path = os.path.join(OUTPUT_DIR, req.job_id, edited_filename)
        
        # Run editing in a thread to avoid blocking main loop
        # Since VideoEditor uses blocking calls (subprocess, API wait)
        def run_edit():
            editor = VideoEditor(api_key=final_api_key)
            
            # SAFE FILE RENAMING STRATEGY (Avoid UnicodeEncodeError in Docker)
            # Create a safe ASCII filename in the same directory
            safe_filename = f"temp_input_{req.job_id}.mp4"
            safe_input_path = os.path.join(OUTPUT_DIR, req.job_id, safe_filename)
            
            # Copy original file to safe path
            # (Copy is safer than rename if something crashes, we keep original)
            shutil.copy(input_path, safe_input_path)
            
            try:
                # 1. Upload (using safe path)
                vid_file = editor.upload_video(safe_input_path)
                
                # 2. Get duration
                import cv2
                cap = cv2.VideoCapture(safe_input_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = frame_count / fps if fps else 0
                cap.release()
                
                # Load transcript from metadata
                transcript = None
                try:
                    meta_files = glob.glob(os.path.join(OUTPUT_DIR, req.job_id, "*_metadata.json"))
                    if meta_files:
                        with open(meta_files[0], 'r') as f:
                            data = json.load(f)
                            transcript = data.get('transcript')
                except Exception as e:
                    print(f"⚠️ Could not load transcript for editing context: {e}")

                # 3. Get Plan (Filter String)
                filter_data = editor.get_ffmpeg_filter(vid_file, duration, fps=fps, width=width, height=height, transcript=transcript)
                
                # 4. Apply
                # Use safe output name first
                safe_output_path = os.path.join(OUTPUT_DIR, req.job_id, f"temp_output_{req.job_id}.mp4")
                editor.apply_edits(safe_input_path, safe_output_path, filter_data)
                
                # Move result to final destination (rename works even if dest name has unicode if filesystem supports it, 
                # but python might still struggle if locale is broken? No, os.rename usually handles it better than subprocess args)
                # Actually, output_path is defined above: f"edited_{filename}"
                # If filename has unicode, output_path has unicode.
                # Let's hope shutil.move / os.rename works.
                if os.path.exists(safe_output_path):
                    shutil.move(safe_output_path, output_path)
                
                return filter_data
            finally:
                # Cleanup temp safe input
                if os.path.exists(safe_input_path):
                    os.remove(safe_input_path)

        # Run in thread pool
        loop = asyncio.get_event_loop()
        plan = await loop.run_in_executor(None, run_edit)
        
        # Update clip URL in the job result? 
        # Or return new URL and let frontend handle it?
        # Updating job result allows persistence if page refreshes.
        
        new_video_url = f"/videos/{req.job_id}/{edited_filename}"
        
        # Start a new "edited" clip entry or just update the current one?
        # Let's update the current one's video_url but keep backup?
        # Or return the new URL to the frontend to display.
        
        return {
            "success": True, 
            "new_video_url": new_video_url,
            "edit_plan": plan
        }

    except Exception as e:
        print(f"❌ Edit Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class SubtitleRequest(BaseModel):
    job_id: str
    clip_index: int
    position: str = "bottom" # top, middle, bottom
    font_size: int = 16
    font_family: str = "Verdana"
    font_color: str = "#FFFFFF"
    stroke_color: str = "#000000"
    stroke_width: int = 2
    bold: bool = True
    box_color: str = "#000000"
    box_opacity: int = 60
    srt_content: Optional[str] = None
    input_filename: Optional[str] = None

@app.post("/api/subtitle")
async def add_subtitles(req: SubtitleRequest):
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Reload job data from disk just in case metadata was updated
    job = jobs[req.job_id]
    
    # We need to access metadata.json to get the transcript
    output_dir = os.path.join(OUTPUT_DIR, req.job_id)
    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
    
    if not json_files:
        raise HTTPException(status_code=404, detail="Metadata not found")
        
    with open(json_files[0], 'r') as f:
        data = json.load(f)
        
    transcript = data.get('transcript')
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript not found in metadata. Please process a new video.")
        
    clips = data.get('shorts', [])
    if req.clip_index >= len(clips):
        raise HTTPException(status_code=404, detail="Clip not found")
        
    clip_data = clips[req.clip_index]
    
    # Video Path
    if req.input_filename:
        # Use chained file
        filename = os.path.basename(req.input_filename)
    else:
        # Fallback to standard naming
        filename = clip_data.get('video_url', '').split('/')[-1]
        if not filename:
             base_name = os.path.basename(json_files[0]).replace('_metadata.json', '')
             filename = f"{base_name}_clip_{req.clip_index+1}.mp4"
         
    input_path = os.path.join(output_dir, filename)
    if not os.path.exists(input_path):
        # Try looking for edited version if url implied it?
        # Just fail if not found.
        raise HTTPException(status_code=404, detail=f"Video file not found: {input_path}")
        
    # Define outputs
    srt_filename = f"subs_{req.clip_index}_{int(time.time())}.srt"
    srt_path = os.path.join(output_dir, srt_filename)
    
    # Output video
    # We create a new file "subtitled_..."
    base_name = os.path.splitext(filename)[0]
    output_filename = f"subtitled_{base_name}.mp4"
    output_path = os.path.join(output_dir, output_filename)
    
    try:
        # 1. Generate or use provided SRT
        if req.srt_content:
             with open(srt_path, 'w', encoding='utf-8') as f:
                  f.write(req.srt_content)
        else:
             success = generate_srt(transcript, clip_data['start'], clip_data['end'], srt_path)
             if not success:
                  raise HTTPException(status_code=400, detail="No words found for this clip range.")
             
        # 2. Burn Subtitles
        # Run in thread pool
        def run_burn():
             burn_subtitles(
                 input_path,
                 srt_path,
                 output_path,
                 alignment=req.position,
                 fontsize=req.font_size,
                 font_name=req.font_family,
                 font_color=req.font_color,
                 stroke_color=req.stroke_color,
                 stroke_width=req.stroke_width,
                 bold=req.bold,
                 box_color=req.box_color,
                 box_opacity=req.box_opacity
             )
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, run_burn)
        
        # 3. Update Result?
        # We return the new URL.
        new_video_url = f"/videos/{req.job_id}/{output_filename}"
        
        return {
            "success": True,
            "new_video_url": new_video_url
        }
        
    except Exception as e:
        print(f"❌ Subtitle Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class SubtitlePreviewRequest(BaseModel):
    job_id: str
    clip_index: int

@app.post("/api/subtitle/preview")
async def preview_subtitles(req: SubtitlePreviewRequest):
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    output_dir = os.path.join(OUTPUT_DIR, req.job_id)
    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
    if not json_files:
        raise HTTPException(status_code=404, detail="Metadata not found")

    with open(json_files[0], 'r') as f:
        data = json.load(f)

    transcript = data.get('transcript')
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript not found in metadata.")

    clips = data.get('shorts', [])
    if req.clip_index >= len(clips):
        raise HTTPException(status_code=404, detail="Clip not found")

    clip_data = clips[req.clip_index]
    temp_name = f"subs_preview_{req.clip_index}_{int(time.time())}.srt"
    temp_path = os.path.join(output_dir, temp_name)
    success = generate_srt(transcript, clip_data['start'], clip_data['end'], temp_path)
    if not success:
        raise HTTPException(status_code=400, detail="No words found for this clip range.")

    with open(temp_path, 'r', encoding='utf-8') as f:
        content = f.read()
    try:
        os.remove(temp_path)
    except Exception:
        pass

    return {"srt": content}

class RecutRequest(BaseModel):
    job_id: str
    clip_index: int
    start: float
    end: float
    aspect_ratio: Optional[str] = None

def _parse_form_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raw = str(value).strip().lower()
    if raw in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default

@app.post("/api/recut")
async def recut_clip(req: RecutRequest):
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[req.job_id]
    input_path = job.get('input_path')
    if not input_path or not os.path.exists(input_path):
        raise HTTPException(status_code=400, detail="Original source video not available for recut.")

    if req.start < 0 or req.end <= req.start:
        raise HTTPException(status_code=400, detail="Invalid start/end times.")
    aspect_ratio = normalize_aspect_ratio(req.aspect_ratio, default="9:16")

    output_dir = os.path.join(OUTPUT_DIR, req.job_id)
    os.makedirs(output_dir, exist_ok=True)

    # Cut from original source
    temp_cut = os.path.join(output_dir, f"temp_recut_{req.clip_index}_{int(time.time())}.mp4")
    out_name = f"reclip_{req.clip_index+1}_{int(time.time())}.mp4"
    out_path = os.path.join(output_dir, out_name)

    cut_cmd = [
        'ffmpeg', '-y',
        '-ss', str(req.start),
        '-to', str(req.end),
        '-i', input_path,
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', '-preset', 'fast',
        '-c:a', 'aac',
        '-movflags', '+faststart',
        temp_cut
    ]
    res = subprocess.run(cut_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if res.returncode != 0:
        raise HTTPException(status_code=500, detail=res.stderr.decode())

    # Re-process with target aspect ratio
    try:
        from main import process_video_to_vertical
        success = process_video_to_vertical(temp_cut, out_path, "fast", 23, aspect_ratio)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to process recut video.")
    finally:
        if os.path.exists(temp_cut):
            os.remove(temp_cut)

    # Update metadata
    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
    if json_files:
        try:
            with open(json_files[0], 'r') as f:
                data = json.load(f)
            clips = data.get('shorts', [])
            if req.clip_index < len(clips):
                clips[req.clip_index]['start'] = req.start
                clips[req.clip_index]['end'] = req.end
                clips[req.clip_index]['aspect_ratio'] = aspect_ratio
                with open(json_files[0], 'w') as f:
                    json.dump(data, f, indent=2)
        except Exception:
            pass

    # Update job result
    if 'result' in job and 'clips' in job['result'] and req.clip_index < len(job['result']['clips']):
        job['result']['clips'][req.clip_index]['video_url'] = f"/videos/{req.job_id}/{out_name}"
        job['result']['clips'][req.clip_index]['start'] = req.start
        job['result']['clips'][req.clip_index]['end'] = req.end
        job['result']['clips'][req.clip_index]['aspect_ratio'] = aspect_ratio

    return {"success": True, "new_video_url": f"/videos/{req.job_id}/{out_name}"}

@app.post("/api/music")
async def add_background_music(
    job_id: str = Form(...),
    clip_index: int = Form(...),
    file: UploadFile = File(...),
    input_filename: Optional[str] = Form(None),
    music_volume: float = Form(0.18),
    duck_voice: Optional[str] = Form("true")
):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    output_dir = os.path.join(OUTPUT_DIR, job_id)
    if not os.path.isdir(output_dir):
        raise HTTPException(status_code=404, detail="Job output directory not found")

    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
    if not json_files:
        raise HTTPException(status_code=404, detail="Metadata not found")

    with open(json_files[0], "r", encoding="utf-8") as f:
        data = json.load(f)

    clips = data.get("shorts", [])
    if clip_index < 0 or clip_index >= len(clips):
        raise HTTPException(status_code=404, detail="Clip not found")

    clip_data = clips[clip_index]
    if input_filename:
        source_name = os.path.basename(input_filename)
    else:
        source_name = os.path.basename(str(clip_data.get("video_url", "")))
        if not source_name:
            base_name = os.path.basename(json_files[0]).replace("_metadata.json", "")
            source_name = f"{base_name}_clip_{clip_index + 1}.mp4"

    input_path = os.path.join(output_dir, source_name)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail=f"Video file not found: {input_path}")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Music file is required")

    ducking_enabled = _parse_form_bool(duck_voice, default=True)
    safe_music_volume = max(0.0, min(1.2, _safe_float(music_volume, 0.18)))

    music_ext = os.path.splitext(file.filename)[1] or ".mp3"
    temp_music_name = f"music_upload_{clip_index}_{int(time.time())}_{uuid.uuid4().hex[:8]}{music_ext}"
    temp_music_path = os.path.join(output_dir, temp_music_name)

    with open(temp_music_path, "wb") as out_music:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            out_music.write(chunk)

    if not os.path.exists(temp_music_path) or os.path.getsize(temp_music_path) == 0:
        try:
            os.remove(temp_music_path)
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="Uploaded music file is empty")

    base_video_name = os.path.splitext(source_name)[0]
    output_name = f"music_{base_video_name}_{int(time.time())}.mp4"
    output_path = os.path.join(output_dir, output_name)

    has_input_audio = False
    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        input_path
    ]
    try:
        probe = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        has_input_audio = probe.returncode == 0 and bool(str(probe.stdout).strip())
    except Exception:
        has_input_audio = False

    volume_expr = f"{safe_music_volume:.4f}"

    try:
        if has_input_audio:
            if ducking_enabled:
                filter_complex = (
                    f"[0:a]volume={volume_expr}[bg];"
                    f"[bg][1:a]sidechaincompress=threshold=0.03:ratio=8:attack=20:release=300[ducked];"
                    f"[ducked][1:a]amix=inputs=2:weights='1 1':duration=first:dropout_transition=2[aout]"
                )
            else:
                filter_complex = (
                    f"[0:a]volume={volume_expr}[bg];"
                    f"[1:a][bg]amix=inputs=2:weights='1 1':duration=first:dropout_transition=2[aout]"
                )

            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-stream_loop", "-1", "-i", temp_music_path,
                "-i", input_path,
                "-filter_complex", filter_complex,
                "-map", "1:v:0",
                "-map", "[aout]",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                "-movflags", "+faststart",
                output_path
            ]
        else:
            filter_complex = f"[0:a]volume={volume_expr}[aout]"
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-stream_loop", "-1",
                "-i", temp_music_path,
                "-i", input_path,
                "-filter_complex", filter_complex,
                "-map", "1:v:0",
                "-map", "[aout]",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                "-movflags", "+faststart",
                output_path
            ]

        ffmpeg_res = subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if ffmpeg_res.returncode != 0 or not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail=ffmpeg_res.stderr.decode("utf-8", errors="ignore"))

        new_video_url = f"/videos/{job_id}/{output_name}"

        # Persist updated clip reference in metadata.
        try:
            clips[clip_index]["video_url"] = new_video_url
            with open(json_files[0], "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        # Keep in-memory job result in sync.
        job = jobs.get(job_id, {})
        if isinstance(job.get("result"), dict):
            job_clips = job["result"].get("clips", [])
            if isinstance(job_clips, list) and clip_index < len(job_clips):
                job_clips[clip_index]["video_url"] = new_video_url

        return {"success": True, "new_video_url": new_video_url}
    finally:
        try:
            if os.path.exists(temp_music_path):
                os.remove(temp_music_path)
        except Exception:
            pass

def _clip_file_path_from_payload(output_dir: str, clip: Dict[str, Any], clip_number: int) -> Optional[str]:
    video_url = str(clip.get("video_url", ""))
    if video_url:
        name = os.path.basename(video_url)
        if name:
            candidate = os.path.join(output_dir, name)
            if os.path.exists(candidate):
                return candidate

    fallback = sorted(glob.glob(os.path.join(output_dir, f"*_clip_{clip_number}.mp4")))
    if fallback:
        return fallback[0]
    return None

def _safe_slug(value: str, fallback: str = "asset") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value or "")).strip("_")
    return cleaned or fallback

def _generate_clip_thumbnail(
    clip_path: str,
    output_dir: str,
    clip_number: int,
    start: float,
    end: float,
    thumb_format: str = "jpg",
    max_width: int = 1080
) -> Optional[str]:
    if not clip_path or not os.path.exists(clip_path):
        return None
    fmt = (thumb_format or "jpg").lower().strip()
    if fmt not in {"jpg", "jpeg", "png", "webp"}:
        fmt = "jpg"

    base = _safe_slug(os.path.splitext(os.path.basename(clip_path))[0], fallback=f"clip_{clip_number}")
    thumb_name = f"{base}_thumb.{fmt}"
    thumb_path = os.path.join(output_dir, thumb_name)

    clip_duration = max(0.0, _safe_float(end, 0.0) - _safe_float(start, 0.0))
    capture_second = 1.0 if clip_duration <= 0 else min(max(0.35, clip_duration * 0.22), max(0.7, clip_duration - 0.2))

    vf_filters = [f"scale='min({int(max_width)},iw)':-2"]
    if fmt == "jpg" or fmt == "jpeg":
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-ss", f"{capture_second:.3f}",
            "-i", clip_path,
            "-vframes", "1",
            "-q:v", "2",
            "-vf", ",".join(vf_filters),
            thumb_path
        ]
    else:
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-ss", f"{capture_second:.3f}",
            "-i", clip_path,
            "-vframes", "1",
            "-vf", ",".join(vf_filters),
            thumb_path
        ]

    try:
        result = subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if result.returncode != 0 or not os.path.exists(thumb_path) or os.path.getsize(thumb_path) <= 0:
            return None
        return thumb_path
    except Exception:
        return None

def _clip_hashtag_string(clip: Dict[str, Any]) -> str:
    tags = clip.get("topic_tags", []) if isinstance(clip, dict) else []
    if not isinstance(tags, list):
        tags = []
    formatted = []
    seen = set()
    for t in tags:
        clean = _safe_slug(str(t), fallback="")
        if not clean:
            continue
        if clean.lower() in seen:
            continue
        seen.add(clean.lower())
        formatted.append(f"#{clean.lower()}")
        if len(formatted) >= 5:
            break
    base = ["#shorts", "#viral"]
    for b in base:
        if b not in formatted:
            formatted.insert(0, b)
            if len(formatted) > 6:
                formatted = formatted[:6]
    return " ".join(formatted)

def _build_platform_variants_rows(normalized_clips: List[Dict[str, Any]]) -> List[List[Any]]:
    rows: List[List[Any]] = []
    for i, clip in enumerate(normalized_clips):
        clip_no = i + 1
        clip_index = clip.get("clip_index", i)
        score = clip.get("virality_score", "")
        start = clip.get("start", "")
        end = clip.get("end", "")
        tags_joined = ",".join(clip.get("topic_tags", []) or [])
        hashtags = _clip_hashtag_string(clip)

        title_yt = str(clip.get("video_title_for_youtube_short", "")).strip()
        desc_tk = str(clip.get("video_description_for_tiktok", "")).strip()
        desc_ig = str(clip.get("video_description_for_instagram", "")).strip()
        desc_yt = desc_ig or desc_tk

        rows.append([clip_no, clip_index, "youtube", title_yt, desc_yt, hashtags, tags_joined, score, start, end])
        rows.append([clip_no, clip_index, "instagram", title_yt, desc_ig or desc_tk, hashtags, tags_joined, score, start, end])
        rows.append([clip_no, clip_index, "tiktok", title_yt, desc_tk or desc_ig, hashtags, tags_joined, score, start, end])
    return rows

class ExportPackRequest(BaseModel):
    job_id: str
    include_video_files: bool = True
    include_srt_files: bool = True
    include_thumbnails: bool = True
    include_platform_variants: bool = True
    thumbnail_format: str = "jpg"
    thumbnail_width: int = 1080

@app.post("/api/export/pack")
async def export_pack(req: ExportPackRequest):
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    output_dir = os.path.join(OUTPUT_DIR, req.job_id)
    if not os.path.isdir(output_dir):
        raise HTTPException(status_code=404, detail="Job output directory not found")

    metadata_files = sorted(glob.glob(os.path.join(output_dir, "*_metadata.json")))
    metadata_path = metadata_files[0] if metadata_files else None
    metadata = {}
    if metadata_path:
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}

    job = jobs[req.job_id]
    clips = []
    if isinstance(job.get("result"), dict):
        clips = job["result"].get("clips", []) or []
    if not clips:
        clips = metadata.get("shorts", []) if isinstance(metadata, dict) else []

    normalized_clips = []
    transcript_data = metadata.get("transcript") if isinstance(metadata, dict) else None
    for i, clip in enumerate(clips):
        normalized = _normalize_clip_payload(
            dict(clip) if isinstance(clip, dict) else {},
            i,
            transcript=transcript_data
        )
        if not normalized.get("video_url"):
            clip_filename = f"{req.job_id}_clip_{i+1}.mp4"
            # Best effort fallback if current naming is unknown.
            candidates = sorted(glob.glob(os.path.join(output_dir, f"*_clip_{i+1}.mp4")))
            if candidates:
                clip_filename = os.path.basename(candidates[0])
            normalized["video_url"] = f"/videos/{req.job_id}/{clip_filename}"
        normalized_clips.append(normalized)

    thumb_format = str(req.thumbnail_format or "jpg").strip().lower()
    if thumb_format not in {"jpg", "jpeg", "png", "webp"}:
        thumb_format = "jpg"
    thumb_width = max(360, min(2160, int(req.thumbnail_width or 1080)))

    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf)
    writer.writerow([
        "clip_number",
        "clip_index",
        "virality_score",
        "score_band",
        "selection_confidence",
        "start_seconds",
        "end_seconds",
        "youtube_title",
        "caption_tiktok",
        "caption_instagram",
        "topic_tags"
    ])
    for i, clip in enumerate(normalized_clips):
        writer.writerow([
            i + 1,
            clip.get("clip_index", i),
            clip.get("virality_score", ""),
            clip.get("score_band", ""),
            clip.get("selection_confidence", ""),
            clip.get("start", ""),
            clip.get("end", ""),
            clip.get("video_title_for_youtube_short", ""),
            clip.get("video_description_for_tiktok", ""),
            clip.get("video_description_for_instagram", ""),
            ",".join(clip.get("topic_tags", []) or [])
        ])

    platform_rows = _build_platform_variants_rows(normalized_clips)
    variants_buf = io.StringIO()
    variants_writer = csv.writer(variants_buf)
    variants_writer.writerow([
        "clip_number",
        "clip_index",
        "platform",
        "title",
        "description",
        "hashtags",
        "topic_tags",
        "virality_score",
        "start_seconds",
        "end_seconds"
    ])
    for row in platform_rows:
        variants_writer.writerow(row)

    thumbnails_generated = []
    for i, clip in enumerate(normalized_clips):
        clip_file = _clip_file_path_from_payload(output_dir, clip, i + 1)
        if not clip_file:
            continue
        clip["video_filename"] = os.path.basename(clip_file)
        if not req.include_thumbnails:
            continue
        thumb_path = _generate_clip_thumbnail(
            clip_path=clip_file,
            output_dir=output_dir,
            clip_number=i + 1,
            start=_safe_float(clip.get("start", 0.0), 0.0),
            end=_safe_float(clip.get("end", 0.0), 0.0),
            thumb_format=thumb_format,
            max_width=thumb_width
        )
        if thumb_path:
            clip["thumbnail_file"] = os.path.basename(thumb_path)
            clip["thumbnail_url"] = f"/videos/{req.job_id}/{os.path.basename(thumb_path)}"
            thumbnails_generated.append(thumb_path)

    manifest = {
        "job_id": req.job_id,
        "generated_at": int(time.time()),
        "export_version": "v2",
        "options": {
            "include_video_files": bool(req.include_video_files),
            "include_srt_files": bool(req.include_srt_files),
            "include_thumbnails": bool(req.include_thumbnails),
            "include_platform_variants": bool(req.include_platform_variants),
            "thumbnail_format": thumb_format,
            "thumbnail_width": thumb_width
        },
        "clips": normalized_clips
    }

    zip_name = f"agency_pack_{req.job_id}_{int(time.time())}.zip"
    zip_path = os.path.join(output_dir, zip_name)
    clip_files_added = 0
    srt_files_added = 0
    thumbnail_files_added = 0

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        zf.writestr("copies.csv", csv_buf.getvalue())
        if req.include_platform_variants:
            zf.writestr("copies_by_platform.csv", variants_buf.getvalue())

        if metadata_path and os.path.exists(metadata_path):
            zf.write(metadata_path, arcname=f"metadata/{os.path.basename(metadata_path)}")

        if req.include_video_files:
            seen = set()
            for clip in normalized_clips:
                video_url = clip.get("video_url", "")
                filename = os.path.basename(video_url)
                if not filename:
                    continue
                file_path = os.path.join(output_dir, filename)
                if file_path in seen or not os.path.exists(file_path):
                    continue
                seen.add(file_path)
                zf.write(file_path, arcname=f"clips/{filename}")
                clip_files_added += 1

        if req.include_srt_files:
            srt_paths = sorted(glob.glob(os.path.join(output_dir, "*.srt")))
            for srt_path in srt_paths:
                zf.write(srt_path, arcname=f"srt/{os.path.basename(srt_path)}")
                srt_files_added += 1

        if req.include_thumbnails:
            seen_thumbs = set()
            for thumb_path in thumbnails_generated:
                if thumb_path in seen_thumbs or not os.path.exists(thumb_path):
                    continue
                seen_thumbs.add(thumb_path)
                zf.write(thumb_path, arcname=f"thumbnails/{os.path.basename(thumb_path)}")
                thumbnail_files_added += 1

    return {
        "success": True,
        "pack_url": f"/videos/{req.job_id}/{zip_name}",
        "clips_in_manifest": len(normalized_clips),
        "video_files_added": clip_files_added,
        "srt_files_added": srt_files_added,
        "thumbnail_files_added": thumbnail_files_added,
        "platform_variant_rows": len(platform_rows) if req.include_platform_variants else 0
    }

@app.get("/api/transcript/{job_id}")
async def get_transcript_segments(
    job_id: str,
    q: Optional[str] = None,
    limit: int = 800,
    offset: int = 0
):
    output_dir = os.path.join(OUTPUT_DIR, job_id)
    if not os.path.isdir(output_dir):
        raise HTTPException(status_code=404, detail="Job output not found")

    json_files = sorted(glob.glob(os.path.join(output_dir, "*_metadata.json")))
    if not json_files:
        raise HTTPException(status_code=404, detail="Metadata not found")

    try:
        with open(json_files[0], "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read metadata: {e}")

    transcript = data.get("transcript") or {}
    segments = transcript.get("segments") if isinstance(transcript, dict) else []
    if not isinstance(segments, list):
        raise HTTPException(status_code=400, detail="Transcript segments not available")

    query = _normalize_space(q or "").lower()
    normalized: List[Dict[str, Any]] = []
    has_speaker_labels = False

    for idx, seg in enumerate(segments):
        if not isinstance(seg, dict):
            continue
        text = _normalize_space(seg.get("text", ""))
        if not text:
            continue
        start = max(0.0, _safe_float(seg.get("start", 0.0), 0.0))
        end = _safe_float(seg.get("end", start), start)
        if end < start:
            end = start
        speaker = _normalize_space(seg.get("speaker", ""))
        if speaker:
            has_speaker_labels = True

        if query:
            haystack = f"{text} {speaker}".lower()
            if query not in haystack:
                continue

        words = seg.get("words") if isinstance(seg.get("words"), list) else []
        normalized.append({
            "segment_index": idx,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(max(0.0, end - start), 3),
            "speaker": speaker or None,
            "word_count": len(words),
            "text": text
        })

    total = len(normalized)
    safe_offset = max(0, int(offset))
    safe_limit = max(1, min(2000, int(limit)))
    paged = normalized[safe_offset:safe_offset + safe_limit]

    return {
        "job_id": job_id,
        "total": total,
        "offset": safe_offset,
        "limit": safe_limit,
        "returned": len(paged),
        "query": query or None,
        "has_speaker_labels": has_speaker_labels,
        "segments": paged
    }

class ClipSearchRequest(BaseModel):
    job_id: str
    query: str
    limit: int = 5
    shortlist_limit: int = 5
    search_mode: str = "balanced"
    chapter_index: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    speaker: Optional[str] = None

class ClipSearchEvalCase(BaseModel):
    query: str
    expected_start: Optional[float] = None
    expected_end: Optional[float] = None
    chapter_index: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    speaker: Optional[str] = None
    min_match_score: Optional[float] = None

class ClipSearchEvalRequest(BaseModel):
    job_id: str
    cases: List[ClipSearchEvalCase]
    search_mode: str = "balanced"
    limit: int = 6
    shortlist_limit: int = 6
    expected_overlap_threshold: float = 0.35

def _temporal_overlap_ratio(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    a_len = max(0.001, a_end - a_start)
    b_len = max(0.001, b_end - b_start)
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    if inter <= 0:
        return 0.0
    return inter / min(a_len, b_len)

@app.post("/api/search/clips")
async def search_clips(
    req: ClipSearchRequest,
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key")
):
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    output_dir = os.path.join(OUTPUT_DIR, req.job_id)
    json_files = sorted(glob.glob(os.path.join(output_dir, "*_metadata.json")))
    if not json_files:
        raise HTTPException(status_code=404, detail="Metadata not found")

    try:
        with open(json_files[0], "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read metadata: {e}")

    transcript = (data.get("transcript") or {})
    segments = transcript.get("segments") or []
    if not isinstance(segments, list) or not segments:
        raise HTTPException(status_code=400, detail="Transcript segments not available")

    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    search_mode = _normalize_search_mode(req.search_mode)
    keywords = _tokenize_query(query)
    query_phrases = _extract_query_phrases(query, keywords)
    if not keywords and not query_phrases:
        raise HTTPException(status_code=400, detail="Query too short or unsupported")
    query_profile = _analyze_query_profile(query, keywords, query_phrases)
    query_profile = _apply_search_mode_override(query_profile, search_mode)

    limit = max(1, min(20, int(req.limit)))
    shortlist_limit = max(1, min(12, int(req.shortlist_limit)))
    semantic_api_key = x_gemini_key or os.environ.get("GEMINI_API_KEY")

    clips = []
    if isinstance(data.get("shorts"), list):
        clips = data.get("shorts") or []
    if not clips:
        job_result = jobs.get(req.job_id, {}).get("result", {})
        if isinstance(job_result, dict):
            clips = job_result.get("clips", []) or []

    index = _ensure_search_index(
        job_id=req.job_id,
        metadata_path=json_files[0],
        transcript=transcript,
        clips=clips,
        semantic_api_key=semantic_api_key
    )

    active_index = index
    query_embedding = _local_semantic_embedding(query)
    if index["provider"] == "gemini":
        remote_query_vec = _embed_texts_with_gemini([query], semantic_api_key)
        if remote_query_vec and len(remote_query_vec) == 1 and remote_query_vec[0]:
            query_embedding = remote_query_vec[0]
        else:
            active_index = _ensure_search_index(
                job_id=req.job_id,
                metadata_path=json_files[0],
                transcript=transcript,
                clips=clips,
                semantic_api_key=None
            )

    scope_range = _parse_scope_inputs(
        duration=_safe_float(active_index.get("duration", 0.0), 0.0),
        chapters=active_index.get("chapters") or [],
        chapter_index=req.chapter_index,
        start_time=req.start_time,
        end_time=req.end_time
    )
    scoped_units, scoped_embeddings, scoped_clips, scoped_chapters, search_scope = _filter_units_embeddings_and_clips(
        units=active_index.get("units") or [],
        unit_embeddings=active_index.get("unit_embeddings") or [],
        clips=active_index.get("clips") or [],
        chapters=active_index.get("chapters") or [],
        scope_range=scope_range,
        speaker=req.speaker
    )

    matches = _build_semantic_matches(
        query=query,
        keywords=keywords,
        phrases=query_phrases,
        query_profile=query_profile,
        units=scoped_units,
        unit_embeddings=scoped_embeddings,
        query_embedding=query_embedding,
        transcript_text=active_index["transcript_text"],
        clips=scoped_clips,
        duration=active_index["duration"],
        limit=limit
    )
    used_relaxed_profile = False
    # If intent is strict (e.g. phrase/entity) and recall is poor, run a softer pass.
    if search_mode != "exact" and len(matches) < max(2, int(math.ceil(limit * 0.4))):
        relaxed_profile = _relax_query_profile(query_profile)
        relaxed_matches = _build_semantic_matches(
            query=query,
            keywords=keywords,
            phrases=query_phrases,
            query_profile=relaxed_profile,
            units=scoped_units,
            unit_embeddings=scoped_embeddings,
            query_embedding=query_embedding,
            transcript_text=active_index["transcript_text"],
            clips=scoped_clips,
            duration=active_index["duration"],
            limit=limit
        )
        if len(relaxed_matches) > len(matches):
            matches = relaxed_matches
            query_profile = relaxed_profile
            used_relaxed_profile = True

    if scope_range is not None:
        scope_start, scope_end, _ = scope_range
        scoped_matches = []
        for m in matches:
            ms = max(scope_start, _safe_float(m.get("start", scope_start), scope_start))
            me = min(scope_end, _safe_float(m.get("end", scope_end), scope_end))
            if me <= ms:
                continue
            item = dict(m)
            item["start"] = round(ms, 3)
            item["end"] = round(me, 3)
            item["duration"] = round(max(0.0, me - ms), 3)
            scoped_matches.append(item)
        matches = scoped_matches[:limit]

    hybrid_shortlist = _build_hybrid_shortlist(
        clips=scoped_clips,
        units=scoped_units,
        unit_embeddings=scoped_embeddings,
        query_embedding=query_embedding,
        keywords=keywords,
        phrases=query_phrases,
        query_profile=query_profile,
        limit=shortlist_limit
    )

    return {
        "matches": matches,
        "keywords": keywords,
        "phrases": query_phrases,
        "chapters": scoped_chapters,
        "hybrid_shortlist": hybrid_shortlist,
        "semantic_provider": active_index["provider"],
        "semantic_enabled": bool(active_index["provider"] == "gemini"),
        "speakers": active_index.get("speakers") or [],
        "search_scope": search_scope,
        "query_profile": {
            "mode": query_profile.get("mode", "topic"),
            "search_mode": search_mode,
            "relaxed": bool(query_profile.get("relaxed", False)),
            "weights": query_profile.get("weights", (0.62, 0.23, 0.15)),
            "thresholds": {
                "min_hybrid_score": query_profile.get("min_hybrid_score", 0.08),
                "min_semantic_score": query_profile.get("min_semantic_score", 0.08),
                "min_keyword_score": query_profile.get("min_keyword_score", 0.0),
            }
        },
        "used_relaxed_profile": used_relaxed_profile
    }

@app.post("/api/search/clips/eval")
async def evaluate_clip_search(
    req: ClipSearchEvalRequest,
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key")
):
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    if not req.cases:
        raise HTTPException(status_code=400, detail="cases cannot be empty")

    safe_limit = max(1, min(20, int(req.limit)))
    safe_shortlist = max(1, min(12, int(req.shortlist_limit)))
    overlap_threshold = max(0.05, min(0.95, _safe_float(req.expected_overlap_threshold, 0.35)))
    safe_cases = req.cases[:100]

    details: List[Dict[str, Any]] = []
    passed_cases = 0
    expected_cases = 0
    top_score_sum = 0.0
    top_overlap_sum = 0.0
    top_overlap_count = 0
    reciprocal_rank_sum = 0.0

    for idx, case in enumerate(safe_cases):
        query = _normalize_space(case.query)
        if not query:
            details.append({
                "case_index": idx,
                "query": "",
                "passed": False,
                "error": "Empty query"
            })
            continue

        request_payload = ClipSearchRequest(
            job_id=req.job_id,
            query=query,
            limit=safe_limit,
            shortlist_limit=safe_shortlist,
            search_mode=req.search_mode,
            chapter_index=case.chapter_index,
            start_time=case.start_time,
            end_time=case.end_time,
            speaker=case.speaker
        )

        try:
            search_result = await search_clips(request_payload, x_gemini_key)
        except HTTPException as e:
            details.append({
                "case_index": idx,
                "query": query,
                "passed": False,
                "error": str(e.detail),
                "status_code": e.status_code
            })
            continue
        except Exception as e:
            details.append({
                "case_index": idx,
                "query": query,
                "passed": False,
                "error": str(e)
            })
            continue

        matches = search_result.get("matches") or []
        top_match = matches[0] if matches else None
        top_score = _safe_float(top_match.get("match_score"), 0.0) if isinstance(top_match, dict) else 0.0
        top_score_sum += top_score

        has_expected_range = (
            case.expected_start is not None
            and case.expected_end is not None
            and _safe_float(case.expected_end, 0.0) > _safe_float(case.expected_start, 0.0)
        )

        first_hit_rank = None
        top_overlap = 0.0
        best_overlap = 0.0
        passed = False

        if has_expected_range:
            expected_cases += 1
            exp_start = max(0.0, _safe_float(case.expected_start, 0.0))
            exp_end = max(exp_start, _safe_float(case.expected_end, exp_start))

            for match_idx, m in enumerate(matches):
                if not isinstance(m, dict):
                    continue
                ms = max(0.0, _safe_float(m.get("start", 0.0), 0.0))
                me = max(ms, _safe_float(m.get("end", ms), ms))
                overlap = _temporal_overlap_ratio(ms, me, exp_start, exp_end)
                if match_idx == 0:
                    top_overlap = overlap
                if overlap > best_overlap:
                    best_overlap = overlap
                if first_hit_rank is None and overlap >= overlap_threshold:
                    first_hit_rank = match_idx + 1

            top_overlap_sum += top_overlap
            top_overlap_count += 1
            if first_hit_rank is not None:
                reciprocal_rank_sum += 1.0 / first_hit_rank
                passed = True
        else:
            min_match_score = max(0.0, min(1.0, _safe_float(case.min_match_score, 0.12)))
            passed = bool(top_match and top_score >= min_match_score)

        if passed:
            passed_cases += 1

        details.append({
            "case_index": idx,
            "query": query,
            "passed": passed,
            "matches_found": len(matches),
            "top_match_score": round(top_score, 4),
            "top_match_start": round(_safe_float(top_match.get("start", 0.0), 0.0), 3) if isinstance(top_match, dict) else None,
            "top_match_end": round(_safe_float(top_match.get("end", 0.0), 0.0), 3) if isinstance(top_match, dict) else None,
            "top_overlap": round(top_overlap, 4) if has_expected_range else None,
            "best_overlap": round(best_overlap, 4) if has_expected_range else None,
            "first_hit_rank": first_hit_rank,
            "expected_start": round(_safe_float(case.expected_start, 0.0), 3) if case.expected_start is not None else None,
            "expected_end": round(_safe_float(case.expected_end, 0.0), 3) if case.expected_end is not None else None,
            "search_scope": search_result.get("search_scope")
        })

    total_cases = len(safe_cases)
    pass_rate = (passed_cases / total_cases) if total_cases else 0.0
    mean_top_score = (top_score_sum / total_cases) if total_cases else 0.0
    mean_top_overlap = (top_overlap_sum / top_overlap_count) if top_overlap_count else None
    mrr = (reciprocal_rank_sum / expected_cases) if expected_cases else None

    return {
        "job_id": req.job_id,
        "total_cases": total_cases,
        "evaluated_cases": len(details),
        "passed_cases": passed_cases,
        "pass_rate": round(pass_rate, 4),
        "expected_range_cases": expected_cases,
        "mean_top_match_score": round(mean_top_score, 4),
        "mean_top_overlap": round(mean_top_overlap, 4) if mean_top_overlap is not None else None,
        "mrr": round(mrr, 4) if mrr is not None else None,
        "overlap_threshold": round(overlap_threshold, 3),
        "search_mode": req.search_mode,
        "details": details
    }

class SocialPostRequest(BaseModel):
    job_id: str
    clip_index: int
    api_key: str
    user_id: str
    platforms: List[str] # ["tiktok", "instagram", "youtube"]
    # Optional overrides if frontend wants to edit them
    title: Optional[str] = None
    description: Optional[str] = None
    scheduled_date: Optional[str] = None # ISO-8601 string
    timezone: Optional[str] = "UTC"

import httpx

@app.post("/api/social/post")
async def post_to_socials(req: SocialPostRequest):
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[req.job_id]
    if 'result' not in job or 'clips' not in job['result']:
        raise HTTPException(status_code=400, detail="Job result not available")
        
    try:
        clip = job['result']['clips'][req.clip_index]
        # Video URL is relative /videos/..., we need absolute file path
        # clip['video_url'] is like "/videos/{job_id}/{filename}"
        # We constructed it as: f"/videos/{job_id}/{clip_filename}"
        # And file is at f"{OUTPUT_DIR}/{job_id}/{clip_filename}"
        
        filename = clip['video_url'].split('/')[-1]
        file_path = os.path.join(OUTPUT_DIR, req.job_id, filename)
        
        if not os.path.exists(file_path):
             raise HTTPException(status_code=404, detail=f"Video file not found: {file_path}")

        # Construct parameters for Upload-Post API
        # Fallbacks
        final_title = req.title or clip.get('title', 'Viral Short')
        final_description = req.description or clip.get('video_description_for_instagram') or clip.get('video_description_for_tiktok') or "Check this out!"
        
        # Prepare form data
        url = "https://api.upload-post.com/api/upload"
        headers = {
            "Authorization": f"Apikey {req.api_key}"
        }
        
        # Prepare data as dict (httpx handles lists for multiple values)
        data_payload = {
            "user": req.user_id,
            "title": final_title,
            "platform[]": req.platforms # Pass list directly
        }

        # Add scheduling if present
        if req.scheduled_date:
            data_payload["scheduled_date"] = req.scheduled_date
            if req.timezone:
                data_payload["timezone"] = req.timezone
        
        # Add Platform specifics
        if "tiktok" in req.platforms:
             data_payload["tiktok_title"] = final_description
             
        if "instagram" in req.platforms:
             data_payload["instagram_title"] = final_description
             data_payload["media_type"] = "REELS"

        if "youtube" in req.platforms:
             yt_title = req.title or clip.get('video_title_for_youtube_short', final_title)
             data_payload["youtube_title"] = yt_title
             data_payload["youtube_description"] = final_description
             data_payload["privacyStatus"] = "public"

        # Send File
        # httpx AsyncClient requires async file reading or bytes. 
        # Since we have MAX_FILE_SIZE_MB, reading into memory is safe-ish.
        with open(file_path, "rb") as f:
            file_content = f.read()
            
        files = {
            "video": (filename, file_content, "video/mp4")
        }

        # Switch to synchronous Client to avoid "sync request with AsyncClient" error with multipart/files
        with httpx.Client(timeout=120.0) as client:
            print(f"📡 Sending to Upload-Post for platforms: {req.platforms}")
            response = client.post(url, headers=headers, data=data_payload, files=files)
            
        if response.status_code not in [200, 201, 202]: # Added 201
             print(f"❌ Upload-Post Error: {response.text}")
             raise HTTPException(status_code=response.status_code, detail=f"Vendor API Error: {response.text}")

        return response.json()

    except Exception as e:
        print(f"❌ Social Post Exception: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/social/user")
async def get_social_user(api_key: str = Header(..., alias="X-Upload-Post-Key")):
    """Proxy to fetch user ID from Upload-Post"""
    if not api_key:
         raise HTTPException(status_code=400, detail="Missing X-Upload-Post-Key header")
         
    url = "https://api.upload-post.com/api/uploadposts/users"
    print(f"🔍 Fetching User ID from: {url}")
    headers = {"Authorization": f"Apikey {api_key}"}
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                print(f"❌ Upload-Post User Fetch Error: {resp.text}")
                raise HTTPException(status_code=resp.status_code, detail=f"Failed to fetch user: {resp.text}")
            
            data = resp.json()
            print(f"🔍 Upload-Post User Response: {data}")
            
            user_id = None
            # The structure is {'success': True, 'profiles': [{'username': '...'}, ...]}
            profiles_list = []
            if isinstance(data, dict):
                 raw_profiles = data.get('profiles', [])
                 if isinstance(raw_profiles, list):
                     for p in raw_profiles:
                         username = p.get('username')
                         if username:
                             # Determine connected platforms
                             socials = p.get('social_accounts', {})
                             connected = []
                             # Check typical platforms
                             for platform in ['tiktok', 'instagram', 'youtube']:
                                 account_info = socials.get(platform)
                                 # If it's a dict and typically has data, or just not empty string
                                 if isinstance(account_info, dict):
                                     connected.append(platform)
                             
                             profiles_list.append({
                                 "username": username,
                                 "connected": connected
                             })
            
            if not profiles_list:
                # Fallback if no profiles found
                return {"profiles": [], "error": "No profiles found"}
                
            return {"profiles": profiles_list}
            
        except Exception as e:
             raise HTTPException(status_code=500, detail=str(e))
