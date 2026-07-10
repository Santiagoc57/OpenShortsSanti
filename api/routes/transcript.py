import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

from ..core.config import OUTPUT_DIR
from ..core.database import _ensure_job_context
from ..services.job_history import build_job_result
from s3_uploader import download_job_artifacts


router = APIRouter(prefix="/api")


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return fallback


def _normalize_word(word: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(word, dict):
        return None
    token = str(word.get("word") or word.get("text") or "").strip()
    if not token:
        return None
    return {
        "word": token,
        "start": _safe_float(word.get("start"), 0.0),
        "end": _safe_float(word.get("end"), _safe_float(word.get("start"), 0.0)),
        "probability": _safe_float(word.get("probability", word.get("score", 0.0)), 0.0),
        "speaker": word.get("speaker"),
    }


def _safe_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(fallback)


def _normalize_segment(segment: Any, index: int, include_words: bool) -> Optional[Dict[str, Any]]:
    if not isinstance(segment, dict):
        return None

    start = _safe_float(segment.get("start"), 0.0)
    end = _safe_float(segment.get("end"), start)
    words: List[Dict[str, Any]] = []
    if include_words and isinstance(segment.get("words"), list):
        words = [item for item in (_normalize_word(word) for word in segment["words"]) if item]

    text = str(segment.get("text") or "").strip()
    if not text and words:
        text = " ".join(str(word.get("word") or "").strip() for word in words).strip()
    if not text:
        return None

    out = {
        "segment_index": _safe_int(segment.get("segment_index"), index),
        "start": round(start, 3),
        "end": round(max(start, end), 3),
        "duration": round(max(0.0, end - start), 3),
        "speaker": segment.get("speaker"),
        "text": text,
        "word_count": len(segment.get("words") or []) if isinstance(segment.get("words"), list) else 0,
    }

    scene_description = str(
        segment.get("scene_description")
        or segment.get("sceneDescription")
        or segment.get("visual_description")
        or segment.get("description")
        or ""
    ).strip()
    if scene_description:
        out["scene_description"] = scene_description
    if include_words:
        out["words"] = words
    return out


def _load_result_for_job(job_id: str) -> Optional[Dict[str, Any]]:
    safe_job_id = os.path.basename(str(job_id or "").strip())
    if not safe_job_id or safe_job_id != str(job_id or "").strip():
        return None

    job = _ensure_job_context(safe_job_id)
    if isinstance(job, dict) and isinstance(job.get("result"), dict):
        result = job["result"]
        if isinstance(result.get("transcript"), dict):
            return result

    output_dir = os.path.join(OUTPUT_DIR, safe_job_id)
    result = build_job_result(output_dir, safe_job_id)
    if not result:
        restored_count = download_job_artifacts(safe_job_id, output_dir)
        if restored_count:
            result = build_job_result(output_dir, safe_job_id)

    if isinstance(job, dict) and result:
        job["result"] = result
    return result


@router.get("/transcript/{job_id}")
async def get_transcript(job_id: str, limit: int = 2000, offset: int = 0, include_words: bool = False):
    result = _load_result_for_job(job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Transcript not found")

    transcript = result.get("transcript") if isinstance(result, dict) else None
    if not isinstance(transcript, dict):
        raise HTTPException(status_code=404, detail="Transcript not found")

    raw_segments = transcript.get("segments")
    if not isinstance(raw_segments, list):
        raw_segments = []

    normalized = [
        item
        for item in (
            _normalize_segment(segment, index, include_words=include_words)
            for index, segment in enumerate(raw_segments)
        )
        if item
    ]

    safe_offset = max(0, int(offset or 0))
    safe_limit = max(1, min(10000, int(limit or 2000)))
    page = normalized[safe_offset:safe_offset + safe_limit]

    return {
        "job_id": job_id,
        "language": transcript.get("language") or "unknown",
        "text": str(transcript.get("text") or "").strip(),
        "segments": page,
        "total": len(normalized),
        "offset": safe_offset,
        "limit": safe_limit,
        "include_words": bool(include_words),
    }
