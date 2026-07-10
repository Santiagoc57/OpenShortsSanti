import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..core.config import OUTPUT_DIR
from ..core.database import _ensure_job_context
from ..services.job_history import build_job_result
from s3_uploader import download_job_artifacts


router = APIRouter(prefix="/api")


class SubtitlePreviewRequest(BaseModel):
    job_id: str
    clip_index: int


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return fallback


def _format_srt_time(seconds: float) -> str:
    safe = max(0.0, float(seconds or 0.0))
    hours = int(safe // 3600)
    minutes = int((safe % 3600) // 60)
    whole_seconds = int(safe % 60)
    millis = int(round((safe - int(safe)) * 1000))
    if millis >= 1000:
        millis = 0
        whole_seconds += 1
        if whole_seconds >= 60:
            whole_seconds = 0
            minutes += 1
            if minutes >= 60:
                minutes = 0
                hours += 1
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{millis:03d}"


def _load_result_for_job(job_id: str) -> Optional[Dict[str, Any]]:
    safe_job_id = os.path.basename(str(job_id or "").strip())
    if not safe_job_id or safe_job_id != str(job_id or "").strip():
        return None

    job = _ensure_job_context(safe_job_id)
    if isinstance(job, dict) and isinstance(job.get("result"), dict):
        result = job["result"]
        if isinstance(result.get("transcript"), dict) or isinstance(result.get("clips"), list):
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


def _clip_at_index(clips: List[Dict[str, Any]], clip_index: int) -> Optional[Dict[str, Any]]:
    for index, clip in enumerate(clips):
        if not isinstance(clip, dict):
            continue
        candidates = [
            clip.get("clip_index"),
            clip.get("index"),
            index,
        ]
        if any(str(candidate) == str(clip_index) for candidate in candidates):
            return clip
    return None


def _segment_text(segment: Dict[str, Any]) -> str:
    text = str(segment.get("text") or "").strip()
    if text:
        return text
    words = segment.get("words")
    if isinstance(words, list):
        return " ".join(str(word.get("word") or word.get("text") or "").strip() for word in words if isinstance(word, dict)).strip()
    return ""


def _segments_to_srt(segments: List[Dict[str, Any]], clip_start: float, clip_end: float) -> str:
    blocks: List[str] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        text = _segment_text(segment)
        if not text:
            continue

        seg_start = _safe_float(segment.get("start"), clip_start)
        seg_end = _safe_float(segment.get("end"), seg_start)
        if seg_end <= clip_start or seg_start >= clip_end:
            continue

        rel_start = max(0.0, seg_start - clip_start)
        rel_end = max(rel_start + 0.05, min(seg_end, clip_end) - clip_start)
        blocks.append(
            f"{len(blocks) + 1}\n"
            f"{_format_srt_time(rel_start)} --> {_format_srt_time(rel_end)}\n"
            f"{text}"
        )
    return "\n\n".join(blocks)


def _clip_transcript_segments(clip: Dict[str, Any], clip_start: float, clip_end: float) -> Optional[str]:
    segments = clip.get("transcript_segments")
    if not isinstance(segments, list) or not segments:
        return None

    timebase = str(clip.get("transcript_timebase") or "").strip().lower()
    normalized: List[Dict[str, Any]] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        item = dict(segment)
        if timebase == "clip":
            item["start"] = _safe_float(item.get("start"), 0.0) + clip_start
            item["end"] = _safe_float(item.get("end"), _safe_float(item.get("start"), 0.0)) + clip_start
        normalized.append(item)
    return _segments_to_srt(normalized, clip_start, clip_end)


@router.post("/subtitle/preview")
async def subtitle_preview(req: SubtitlePreviewRequest):
    result = _load_result_for_job(req.job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Project not found")

    clips = result.get("clips") if isinstance(result, dict) else None
    if not isinstance(clips, list):
        clips = []
    clip = _clip_at_index(clips, req.clip_index)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")

    clip_start = _safe_float(clip.get("start"), 0.0)
    clip_end = _safe_float(clip.get("end"), clip_start)
    if clip_end <= clip_start:
        duration = _safe_float(clip.get("duration"), 0.0)
        clip_end = clip_start + max(0.0, duration)
    if clip_end <= clip_start:
        raise HTTPException(status_code=404, detail="Clip has no subtitle range")

    srt = _clip_transcript_segments(clip, clip_start, clip_end)
    if not srt:
        transcript = result.get("transcript") if isinstance(result, dict) else None
        segments = transcript.get("segments") if isinstance(transcript, dict) else None
        srt = _segments_to_srt(segments if isinstance(segments, list) else [], clip_start, clip_end)

    return {
        "job_id": req.job_id,
        "clip_index": req.clip_index,
        "srt": srt or "",
        "has_subtitles": bool(srt),
    }
