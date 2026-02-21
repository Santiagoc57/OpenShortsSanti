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
import struct
import zlib
import sqlite3
import unicodedata
from urllib.parse import unquote
from dotenv import load_dotenv
from typing import Dict, Optional, List, Any, Tuple, Set
from collections import Counter
from contextlib import asynccontextmanager, suppress
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
JOBS_DB_PATH = os.environ.get("JOBS_DB_PATH", os.path.join(OUTPUT_DIR, "jobs_state.sqlite3"))

# Application State
job_queue = asyncio.Queue()
jobs: Dict[str, Dict] = {}
# Semester to limit concurrency to MAX_CONCURRENT_JOBS
concurrency_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
_JOBS_DB_LOCK = threading.Lock()
SEARCH_INDEX_CACHE: Dict[str, Dict[str, Any]] = {}
LOCAL_EMBED_DIM = 256
SEMANTIC_EMBED_MODEL = os.environ.get("SEMANTIC_EMBED_MODEL", "text-embedding-004")
DEFAULT_TITLE_REWRITE_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
]

def _parse_model_candidates(raw_value: Optional[str], fallback_models: List[str]) -> List[str]:
    raw = str(raw_value or "").strip()
    if not raw:
        return list(fallback_models)
    parts = [p.strip() for p in raw.split(",") if str(p or "").strip()]
    if not parts:
        return list(fallback_models)
    out = []
    seen = set()
    for model in parts:
        key = model.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(model)
    return out or list(fallback_models)

def _is_gemini_model_unavailable_error(err: Exception) -> bool:
    msg = str(err or "").lower()
    if not msg:
        return False
    if "models/" in msg and "not found" in msg:
        return True
    if "model" in msg and "not found" in msg:
        return True
    if "model" in msg and "not supported" in msg:
        return True
    if "for api version" in msg and "model" in msg:
        return True
    if "unknown model" in msg:
        return True
    return False

TITLE_REWRITE_MODELS = _parse_model_candidates(
    os.environ.get("TITLE_REWRITE_MODELS") or os.environ.get("TITLE_REWRITE_MODEL"),
    DEFAULT_TITLE_REWRITE_MODELS
)
SOCIAL_REWRITE_MODELS = _parse_model_candidates(
    os.environ.get("SOCIAL_REWRITE_MODELS") or os.environ.get("SOCIAL_REWRITE_MODEL"),
    TITLE_REWRITE_MODELS
)
TITLE_VARIANTS_PER_CLIP = max(2, min(8, int(os.environ.get("TITLE_VARIANTS_PER_CLIP", "5"))))
TITLE_VARIANTS_TOPUP_COUNT = max(1, min(6, int(os.environ.get("TITLE_VARIANTS_TOPUP_COUNT", "3"))))
SOCIAL_VARIANTS_PER_CLIP = max(2, min(8, int(os.environ.get("SOCIAL_VARIANTS_PER_CLIP", "5"))))
ALLOWED_ASPECT_RATIOS = {"9:16", "16:9"}
ALLOWED_CLIP_LENGTH_TARGETS = {"short", "balanced", "long"}
_SMART_REF_YOLO_MODEL = None
_SMART_REF_YOLO_UNAVAILABLE = False
_SMART_REF_FACE_CASCADE = None


def normalize_aspect_ratio(raw_value: Optional[str], default: Optional[str] = None) -> Optional[str]:
    if raw_value is None:
        return default
    value = str(raw_value).strip().replace("/", ":")
    if not value:
        return default
    if value not in ALLOWED_ASPECT_RATIOS:
        raise HTTPException(status_code=400, detail="Invalid aspect_ratio. Allowed values: 9:16, 16:9")
    return value

def _safe_input_filename(value: Optional[str]) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    # Handles full URLs and encoded names from frontend (e.g. %20).
    filename = os.path.basename(raw)
    filename = unquote(filename)
    # In case it arrives double-encoded from chained transformations.
    filename = unquote(filename)
    return os.path.basename(filename)

def _resolve_subtitle_source_filename(output_dir: str, filename: str) -> str:
    """
    If the incoming file is already a generated `subtitled_*` artifact,
    peel the prefix while the underlying source exists, so re-applying
    subtitles does not stack previous burned captions.
    """
    current = _safe_input_filename(filename)
    if not current:
        return ""

    depth = 0
    while current.startswith("subtitled_") and depth < 8:
        candidate = current[len("subtitled_"):]
        if not candidate or candidate == current:
            break
        if not os.path.exists(os.path.join(output_dir, candidate)):
            break
        current = candidate
        depth += 1
    return current

def _looks_like_trailer_clip(clip: Dict[str, Any]) -> bool:
    if not isinstance(clip, dict):
        return False
    if bool(clip.get("is_trailer")):
        return True
    name = _safe_input_filename(clip.get("video_url", ""))
    if name and "_trailer" in name.lower():
        return True
    title = str(clip.get("video_title_for_youtube_short") or clip.get("title") or "").lower()
    return "super trailer" in title

def _repair_trailer_clip_range(clip: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    Heal stale trailer clips that were persisted with end=3.0 due missing ffprobe.
    """
    if not isinstance(clip, dict) or not _looks_like_trailer_clip(clip):
        return clip
    trailer_name = _safe_input_filename(clip.get("video_url", ""))
    if not trailer_name:
        return clip
    trailer_path = os.path.join(output_dir, trailer_name)
    if not os.path.exists(trailer_path):
        return clip
    real_duration = _probe_media_duration_seconds(trailer_path)
    if real_duration < 3.5:
        return clip

    start = max(0.0, _safe_float(clip.get("start", 0.0), 0.0))
    end = max(start, _safe_float(clip.get("end", start), start))
    current_duration = max(0.0, end - start)
    if current_duration < max(3.2, real_duration * 0.7):
        clip["start"] = 0.0 if start <= 1.0 else round(start, 3)
        clip["end"] = round(float(clip["start"]) + real_duration, 3)
    clip["duration"] = round(real_duration, 3)
    if clip.get("transcript_segments") and not str(clip.get("transcript_timebase", "")).strip():
        clip["transcript_timebase"] = "clip"
    return clip

def _resolve_clip_scoped_transcript_for_srt(
    clip_data: Dict[str, Any],
    fallback_transcript: Dict[str, Any]
) -> Tuple[Dict[str, Any], float, float]:
    """
    Returns (transcript, clip_start, clip_end) for SRT generation.
    Supports clip-local transcript segments (timebase=clip).
    """
    if not isinstance(clip_data, dict):
        return fallback_transcript, 0.0, 0.0

    clip_start = max(0.0, _safe_float(clip_data.get("start", 0.0), 0.0))
    clip_end = max(clip_start, _safe_float(clip_data.get("end", clip_start), clip_start))
    raw_segments = clip_data.get("transcript_segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        return fallback_transcript, clip_start, clip_end

    normalized_segments: List[Dict[str, Any]] = []
    max_seg_end = 0.0
    for idx, seg in enumerate(raw_segments):
        if not isinstance(seg, dict):
            continue
        seg_start = max(0.0, _safe_float(seg.get("start", 0.0), 0.0))
        seg_end = max(seg_start, _safe_float(seg.get("end", seg_start), seg_start))
        text = _normalize_space(seg.get("text", ""))

        words_payload: List[Dict[str, Any]] = []
        raw_words = seg.get("words") if isinstance(seg.get("words"), list) else []
        for w in raw_words:
            if not isinstance(w, dict):
                continue
            ws = max(seg_start, _safe_float(w.get("start", seg_start), seg_start))
            we = max(ws, _safe_float(w.get("end", ws), ws))
            wt = _normalize_space(w.get("word", "")) or _normalize_space(w.get("text", ""))
            if not wt:
                continue
            words_payload.append({
                "word": wt,
                "start": round(ws, 3),
                "end": round(we, 3)
            })

        if not text and words_payload:
            text = _normalize_space(" ".join(item["word"] for item in words_payload))
        if not text:
            continue

        normalized_segments.append({
            "segment_index": int(_safe_float(seg.get("segment_index", idx), idx)),
            "start": round(seg_start, 3),
            "end": round(seg_end, 3),
            "text": text,
            "speaker": _normalize_space(seg.get("speaker", "")) or None,
            "words": words_payload
        })
        max_seg_end = max(max_seg_end, seg_end)

    if not normalized_segments:
        return fallback_transcript, clip_start, clip_end

    transcript_text = _normalize_space(clip_data.get("transcript_text", "")) or _normalize_space(
        " ".join(seg.get("text", "") for seg in normalized_segments)
    )
    scoped_transcript = {
        "text": transcript_text,
        "segments": normalized_segments
    }

    timebase = str(clip_data.get("transcript_timebase", "")).strip().lower()
    if timebase == "clip":
        clip_duration = max(0.0, clip_end - clip_start)
        local_end = max(0.1, max_seg_end, clip_duration)
        return scoped_transcript, 0.0, local_end
    return scoped_transcript, clip_start, clip_end

def _srt_timestamp_to_seconds(value: str) -> float:
    raw = str(value or "").strip().replace(",", ".")
    parts = raw.split(":")
    if len(parts) != 3:
        return 0.0
    try:
        hh = float(parts[0])
        mm = float(parts[1])
        ss = float(parts[2])
        return max(0.0, hh * 3600 + mm * 60 + ss)
    except Exception:
        return 0.0

def _seconds_to_srt_timestamp(seconds: float) -> str:
    total = max(0.0, float(seconds or 0.0))
    hh = int(total // 3600)
    mm = int((total % 3600) // 60)
    ss = int(total % 60)
    ms = int(round((total - int(total)) * 1000))
    if ms >= 1000:
        ms = 0
        ss += 1
        if ss >= 60:
            ss = 0
            mm += 1
            if mm >= 60:
                mm = 0
                hh += 1
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

def _parse_srt_blocks(raw_srt: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    blocks = re.split(r"\n\s*\n", str(raw_srt or "").strip())
    for block in blocks:
        lines = [ln.rstrip() for ln in block.splitlines() if str(ln).strip()]
        if len(lines) < 2:
            continue
        if re.match(r"^\d+$", lines[0].strip()):
            timeline = lines[1].strip() if len(lines) > 1 else ""
            text_lines = lines[2:]
        else:
            timeline = lines[0].strip()
            text_lines = lines[1:]
        if "-->" not in timeline:
            continue
        start_raw, end_raw = [part.strip() for part in timeline.split("-->", 1)]
        start_s = _srt_timestamp_to_seconds(start_raw)
        end_s = _srt_timestamp_to_seconds(end_raw)
        text = "\n".join(str(line or "") for line in text_lines).strip()
        if not text or end_s <= start_s:
            continue
        entries.append({
            "start": start_s,
            "end": end_s,
            "text": text
        })
    return entries

def _shift_and_trim_srt(raw_srt: str, shift_seconds: float, window_duration: float) -> str:
    entries = _parse_srt_blocks(raw_srt)
    if not entries:
        return ""
    safe_window = max(0.1, _safe_float(window_duration, 0.0))
    out_chunks: List[str] = []
    out_idx = 1
    for entry in entries:
        start = _safe_float(entry.get("start"), 0.0) + _safe_float(shift_seconds, 0.0)
        end = _safe_float(entry.get("end"), 0.0) + _safe_float(shift_seconds, 0.0)
        if end <= 0.0 or start >= safe_window:
            continue
        clipped_start = max(0.0, start)
        clipped_end = min(safe_window, end)
        if (clipped_end - clipped_start) < 0.04:
            continue
        text = str(entry.get("text", "")).strip()
        if not text:
            continue
        out_chunks.append(
            f"{out_idx}\n"
            f"{_seconds_to_srt_timestamp(clipped_start)} --> {_seconds_to_srt_timestamp(clipped_end)}\n"
            f"{text}"
        )
        out_idx += 1
    return "\n\n".join(out_chunks)

def _build_trailer_timeline_from_fragments(
    fragments: Any,
    fade_duration: float = 0.5,
    duration_cap: Optional[float] = None
) -> Dict[str, Any]:
    safe_fade = max(0.0, _safe_float(fade_duration, 0.5))
    if not isinstance(fragments, list) or not fragments:
        return {
            "transition_points": [],
            "fragment_ranges": [],
            "timeline_duration": 0.0,
            "fade_duration": safe_fade,
        }

    valid: List[Tuple[float, float]] = []
    for frag in fragments:
        if not isinstance(frag, dict):
            continue
        fs = max(0.0, _safe_float(frag.get("start", 0.0), 0.0))
        fe = max(fs, _safe_float(frag.get("end", fs), fs))
        if (fe - fs) < 0.08:
            continue
        valid.append((fs, fe))

    if not valid:
        return {
            "transition_points": [],
            "fragment_ranges": [],
            "timeline_duration": 0.0,
            "fade_duration": safe_fade,
        }

    markers: List[float] = []
    ranges: List[Dict[str, Any]] = []
    cursor = 0.0
    for idx, (fs, fe) in enumerate(valid):
        dur = max(0.0, fe - fs)
        out_start = cursor
        out_end = out_start + dur
        ranges.append({
            "fragment_index": idx,
            "start": round(out_start, 3),
            "end": round(out_end, 3),
            "source_start": round(fs, 3),
            "source_end": round(fe, 3),
        })
        if idx < len(valid) - 1:
            t_start = max(0.0, out_end - safe_fade)
            markers.append(round(t_start, 3))
            cursor = t_start
        else:
            cursor = out_end

    total = max(0.0, cursor)
    if duration_cap is not None:
        total = min(total, max(0.0, float(duration_cap)))
        markers = [round(max(0.0, min(total, float(p))), 3) for p in markers]
        ranges = [
            {
                **r,
                "start": round(max(0.0, min(total, _safe_float(r.get("start", 0.0), 0.0))), 3),
                "end": round(max(0.0, min(total, _safe_float(r.get("end", 0.0), 0.0))), 3),
            }
            for r in ranges
        ]
    return {
        "transition_points": markers,
        "fragment_ranges": ranges,
        "timeline_duration": round(total, 3),
        "fade_duration": safe_fade,
    }

def _probe_media_duration_seconds(media_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        media_path
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode == 0:
            return max(0.0, float((proc.stdout or "").strip() or 0.0))
    except (FileNotFoundError, Exception):
        # Fallback to ffmpeg -i if ffprobe is missing
        try:
            cmd_fallback = ["ffmpeg", "-i", media_path]
            proc_fb = subprocess.run(cmd_fallback, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            # Duration is in stderr for ffmpeg -i
            err_out = proc_fb.stderr or ""
            # Look for "Duration: 00:00:10.50"
            import re
            match = re.search(r"Duration:\s+(\d+):(\d+):(\d+\.\d+)", err_out)
            if match:
                h, m, s = match.groups()
                return float(h)*3600 + float(m)*60 + float(s)
        except Exception:
            pass
    return 0.0

def _probe_video_dimensions(media_path: str) -> Tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x",
        media_path
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode == 0:
            raw = str(proc.stdout or "").strip()
            if "x" in raw:
                w_raw, h_raw = raw.split("x", 1)
                return int(float(w_raw)), int(float(h_raw))
    except (FileNotFoundError, Exception):
        # Fallback to ffmpeg -i if ffprobe is missing
        try:
            cmd_fallback = ["ffmpeg", "-i", media_path]
            proc_fb = subprocess.run(cmd_fallback, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            err_out = proc_fb.stderr or ""
            # Look for "Video: ..., 1280x720 ..."
            import re
            match = re.search(r"Video:.*?\s(\d+)x(\d+)", err_out)
            if match:
                return int(match.group(1)), int(match.group(2))
        except Exception:
            pass
    return 0, 0

def _even_int(value: float) -> int:
    iv = int(round(float(value)))
    if iv < 2:
        iv = 2
    if iv % 2 != 0:
        iv -= 1
    return max(2, iv)

# Keep manual pan intensity aligned with editor preview (ClipStudioModal.jsx).
# 1.0 lets users reach the full pan range allowed by current crop/pad math.
_LAYOUT_OFFSET_FACTOR = 1.0

def _normalize_layout_fit_mode(fit_mode: Optional[str]) -> str:
    fit = str(fit_mode or "cover").strip().lower()
    if fit not in {"cover", "contain", "blur"}:
        fit = "cover"
    return fit

def _coerce_layout_zoom(value: Any, default: float = 1.0, fit_mode: str = "cover") -> float:
    fit = _normalize_layout_fit_mode(fit_mode)
    min_zoom = 1.0 if fit == "cover" else 0.5
    fallback = max(min_zoom, _safe_float(default, 1.0))
    return max(min_zoom, min(2.5, _safe_float(value, fallback)))

def _coerce_layout_offset(value: Any, default: float = 0.0) -> float:
    return max(-100.0, min(100.0, _safe_float(value, default)))

def _build_manual_layout_ops_for_target(
    in_w: int,
    in_h: int,
    out_w: int,
    out_h: int,
    fit_mode: str = "cover",
    zoom: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    input_label: Optional[str] = None
) -> str:
    fit = _normalize_layout_fit_mode(fit_mode)
    z = _coerce_layout_zoom(zoom, 1.0, fit)
    ox_raw = _coerce_layout_offset(offset_x, 0.0) / 100.0
    oy_raw = _coerce_layout_offset(offset_y, 0.0) / 100.0
    
    if fit == "cover" and z <= 1.0001 and (abs(ox_raw) > 1e-6 or abs(oy_raw) > 1e-6):
        z = 1.06
    ox = ox_raw * _LAYOUT_OFFSET_FACTOR
    oy = oy_raw * _LAYOUT_OFFSET_FACTOR

    out_w = _even_int(out_w)
    out_h = _even_int(out_h)

    # 1. Standard Linear modes (cover, contain)
    if fit in {"cover", "contain"}:
        if fit == "cover":
            base_scale = max(out_w / max(1, in_w), out_h / max(1, in_h))
        else:
            base_scale = min(out_w / max(1, in_w), out_h / max(1, in_h))

        # How much of the original frame do we want to capture?
        # A zoom of 1.0 means we capture the full frame (or as much as fits the aspect ratio).
        # A zoom of 2.0 means we capture half the frame.
        # A zoom of 0.5 means we capture twice the frame (impossible, so we pad later).
        scale_factor = max(0.1, base_scale * z)
        
        # Calculate the size of the "capture box" on the original source image
        capture_w = int(out_w / scale_factor)
        capture_h = int(out_h / scale_factor)
        
        filters = []
        stage_w, stage_h = in_w, in_h

        # 1. Crop: If the capture box is smaller than the input, we crop the input.
        # This is where pan (offset) applies.
        if capture_w < stage_w or capture_h < stage_h:
            actual_crop_w = min(stage_w, capture_w)
            actual_crop_h = min(stage_h, capture_h)
            
            max_crop_x = stage_w - actual_crop_w
            max_crop_y = stage_h - actual_crop_h
            
            crop_x = int(round((max_crop_x / 2.0) + (ox * (max_crop_x / 2.0))))
            crop_y = int(round((max_crop_y / 2.0) + (oy * (max_crop_y / 2.0))))
            
            # Ensure crop doesn't go out of bounds
            crop_x = max(0, min(max_crop_x, crop_x))
            crop_y = max(0, min(max_crop_y, crop_y))
            
            filters.append(f"crop={actual_crop_w}:{actual_crop_h}:{crop_x}:{crop_y}")
            stage_w, stage_h = actual_crop_w, actual_crop_h

        # 2. Scale: Now scale the cropped portion (or the full image if capture box was larger)
        scaled_w = _even_int(stage_w * scale_factor)
        scaled_h = _even_int(stage_h * scale_factor)
        filters.append(f"scale={scaled_w}:{scaled_h}")
        stage_w, stage_h = scaled_w, scaled_h

        # 3. Pad: If the scaled result is smaller than the output box, we pad with black.
        # This happens in Contain mode, or if zoomed out (z < 1.0).
        if stage_w < out_w or stage_h < out_h:
            max_pad_x = max(0, out_w - stage_w)
            max_pad_y = max(0, out_h - stage_h)
            
            # If we crop AND pad (e.g. weird aspect ratios), offset shouldn't double dip, 
            # but usually it's one or the other per axis.
            # When padding, panning moves the image within the black box.
            pad_x = int(round((max_pad_x / 2.0) + (ox * (max_pad_x / 2.0))))
            pad_y = int(round((max_pad_y / 2.0) + (oy * (max_pad_y / 2.0))))
            
            pad_x = max(0, min(max_pad_x, pad_x))
            pad_y = max(0, min(max_pad_y, pad_y))
            
            filters.append(f"pad={out_w}:{out_h}:{pad_x}:{pad_y}:black")

        return ",".join(filters)


    # 2. Blur Fill Mode
    # background cover + blur, then overlay contain on top
    # Note: We use unique labels to avoid collision if used multiple times in same filter complex
    lbl = f"blur_{uuid.uuid4().hex[:4]}"
    bg_scale = max(out_w / max(1, in_w), out_h / max(1, in_h))
    bg_scaled_w = _even_int(in_w * bg_scale)
    bg_scaled_h = _even_int(in_h * bg_scale)
    
    fg_scale = min(out_w / max(1, in_w), out_h / max(1, in_h))
    fg_scaled_w = _even_int(in_w * fg_scale * z)
    fg_scaled_h = _even_int(in_h * fg_scale * z)
    
    # Background branch
    bg_filters = [
        f"scale={bg_scaled_w}:{bg_scaled_h}",
        f"crop={out_w}:{out_h}",
        "boxblur=40:10"
    ]
    # Foreground branch
    fg_filters = [
        f"scale={fg_scaled_w}:{fg_scaled_h}"
    ]
    # Center or offset FG
    fg_x = f"(W-w)/2+{ox*out_w/2}"
    fg_y = f"(H-h)/2+{oy*out_h/2}"
    
    # Combine
    in_prefix = f"[{input_label}]" if input_label else ""
    return (
        f"{in_prefix}split[v{lbl}_bg][v{lbl}_fg];"
        f"[v{lbl}_bg]{','.join(bg_filters)}[v{lbl}_out_bg];"
        f"[v{lbl}_fg]{','.join(fg_filters)}[v{lbl}_out_fg];"
        f"[v{lbl}_out_bg][v{lbl}_out_fg]overlay={fg_x}:{fg_y}"
    )

def _build_manual_layout_filter(
    in_w: int,
    in_h: int,
    aspect_ratio: str,
    fit_mode: str = "cover",
    zoom: float = 1.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0
) -> Tuple[str, int, int]:
    ratio_map = {"9:16": 9.0 / 16.0, "16:9": 16.0 / 9.0}
    target_ratio = ratio_map.get(aspect_ratio, 9.0 / 16.0)
    source_ratio = (in_w / in_h) if in_w > 0 and in_h > 0 else target_ratio

    if source_ratio >= target_ratio:
        out_h = _even_int(in_h)
        out_w = _even_int(out_h * target_ratio)
    else:
        out_w = _even_int(in_w)
        out_h = _even_int(out_w / target_ratio)

    filters = _build_manual_layout_ops_for_target(
        in_w=in_w,
        in_h=in_h,
        out_w=out_w,
        out_h=out_h,
        fit_mode=fit_mode,
        zoom=zoom,
        offset_x=offset_x,
        offset_y=offset_y,
        input_label="0:v"
    )

    return f"{filters}[out_v]", out_w, out_h

def _build_split_layout_filter_complex(
    in_w: int,
    in_h: int,
    aspect_ratio: str,
    fit_mode: str = "cover",
    zoom_a: float = 1.0,
    offset_a_x: float = 0.0,
    offset_a_y: float = 0.0,
    zoom_b: float = 1.0,
    offset_b_x: float = 0.0,
    offset_b_y: float = 0.0
) -> Tuple[str, int, int, str]:
    target_ratio = _aspect_ratio_to_float(aspect_ratio)
    out_w, out_h = _derive_output_dimensions(in_w, in_h, target_ratio)

    split_stacked = str(aspect_ratio or "9:16") == "9:16"
    if split_stacked:
        pane_w = out_w
        pane_h = _even_int(out_h / 2.0)
        out_h = pane_h * 2
        pane_a_ops = _build_manual_layout_ops_for_target(
            in_w=in_w,
            in_h=in_h,
            out_w=pane_w,
            out_h=pane_h,
            fit_mode=fit_mode,
            zoom=zoom_a,
            offset_x=offset_a_x,
            offset_y=offset_a_y,
            input_label="vsplit_a"
        )
        pane_b_ops = _build_manual_layout_ops_for_target(
            in_w=in_w,
            in_h=in_h,
            out_w=pane_w,
            out_h=pane_h,
            fit_mode=fit_mode,
            zoom=zoom_b,
            offset_x=offset_b_x,
            offset_y=offset_b_y,
            input_label="vsplit_b"
        )
        out_label = "vsplit_out"
        filter_complex = (
            f"[0:v]split=2[vsplit_a][vsplit_b];"
            f"{pane_a_ops}[vsplit_top];"
            f"{pane_b_ops}[vsplit_bottom];"
            f"[vsplit_top][vsplit_bottom]vstack=inputs=2[{out_label}]"
        )
        return filter_complex, out_w, out_h, out_label

    pane_w = _even_int(out_w / 2.0)
    pane_h = out_h
    out_w = pane_w * 2
    pane_a_ops = _build_manual_layout_ops_for_target(
        in_w=in_w,
        in_h=in_h,
        out_w=pane_w,
        out_h=pane_h,
        fit_mode=fit_mode,
        zoom=zoom_a,
        offset_x=offset_a_x,
        offset_y=offset_a_y,
        input_label="hsplit_a"
    )
    pane_b_ops = _build_manual_layout_ops_for_target(
        in_w=in_w,
        in_h=in_h,
        out_w=pane_w,
        out_h=pane_h,
        fit_mode=fit_mode,
        zoom=zoom_b,
        offset_x=offset_b_x,
        offset_y=offset_b_y,
        input_label="hsplit_b"
    )
    out_label = "hsplit_out"
    filter_complex = (
        f"[0:v]split=2[hsplit_a][hsplit_b];"
        f"{pane_a_ops}[hsplit_left];"
        f"{pane_b_ops}[hsplit_right];"
        f"[hsplit_left][hsplit_right]hstack=inputs=2[{out_label}]"
    )
    return filter_complex, out_w, out_h, out_label

def _aspect_ratio_to_float(aspect_ratio: str) -> float:
    ratio_map = {"9:16": 9.0 / 16.0, "16:9": 16.0 / 9.0}
    return ratio_map.get(str(aspect_ratio or "9:16"), 9.0 / 16.0)

def _derive_output_dimensions(in_w: int, in_h: int, target_ratio: float) -> Tuple[int, int]:
    source_ratio = (in_w / in_h) if in_w > 0 and in_h > 0 else target_ratio
    if source_ratio >= target_ratio:
        out_h = _even_int(in_h)
        out_w = _even_int(out_h * target_ratio)
    else:
        out_w = _even_int(in_w)
        out_h = _even_int(out_w / max(1e-6, target_ratio))
    return out_w, out_h

def _smart_ref_get_yolo_model():
    global _SMART_REF_YOLO_MODEL, _SMART_REF_YOLO_UNAVAILABLE
    if _SMART_REF_YOLO_UNAVAILABLE:
        return None
    if _SMART_REF_YOLO_MODEL is None:
        try:
            from ultralytics import YOLO
            _SMART_REF_YOLO_MODEL = YOLO("yolov8n.pt")
        except Exception as e:
            _SMART_REF_YOLO_UNAVAILABLE = True
            _SMART_REF_YOLO_MODEL = None
            print(f"⚠️ SmartReframe: YOLO unavailable ({e})")
    return _SMART_REF_YOLO_MODEL

def _smart_ref_get_face_cascade():
    global _SMART_REF_FACE_CASCADE
    if _SMART_REF_FACE_CASCADE is not None:
        return _SMART_REF_FACE_CASCADE
    try:
        import cv2
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if cascade.empty():
            _SMART_REF_FACE_CASCADE = None
        else:
            _SMART_REF_FACE_CASCADE = cascade
    except Exception:
        _SMART_REF_FACE_CASCADE = None
    return _SMART_REF_FACE_CASCADE

def _smart_ref_enclosing_box(boxes: List[List[int]]) -> Optional[List[int]]:
    if not boxes:
        return None
    min_x = min(int(b[0]) for b in boxes)
    min_y = min(int(b[1]) for b in boxes)
    max_x = max(int(b[2]) for b in boxes)
    max_y = max(int(b[3]) for b in boxes)
    return [min_x, min_y, max_x, max_y]

def _smart_ref_detect_people(frame) -> List[Dict[str, Any]]:
    detections: List[Dict[str, Any]] = []
    try:
        import cv2
    except Exception:
        return detections

    frame_h, frame_w = frame.shape[:2]
    face_cascade = _smart_ref_get_face_cascade()
    model = _smart_ref_get_yolo_model()

    if model is not None:
        try:
            results = model([frame], verbose=False)
            for result in results:
                boxes = getattr(result, "boxes", None)
                if boxes is None:
                    continue
                for box in boxes:
                    try:
                        cls_id = int(float(box.cls[0]))
                    except Exception:
                        cls_id = -1
                    if cls_id != 0:
                        continue

                    conf = 0.0
                    try:
                        conf = float(box.conf[0])
                    except Exception:
                        conf = 0.0
                    if conf < 0.20:
                        continue

                    try:
                        x1f, y1f, x2f, y2f = [float(v) for v in box.xyxy[0].tolist()]
                    except Exception:
                        continue

                    x1 = max(0, min(frame_w - 1, int(round(x1f))))
                    y1 = max(0, min(frame_h - 1, int(round(y1f))))
                    x2 = max(x1 + 1, min(frame_w, int(round(x2f))))
                    y2 = max(y1 + 1, min(frame_h, int(round(y2f))))
                    person_box = [x1, y1, x2, y2]
                    face_box = None

                    if face_cascade is not None:
                        try:
                            roi = frame[y1:y2, x1:x2]
                            if roi.size > 0:
                                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                                faces = face_cascade.detectMultiScale(
                                    gray,
                                    scaleFactor=1.1,
                                    minNeighbors=5,
                                    minSize=(24, 24)
                                )
                                if len(faces) > 0:
                                    largest = sorted(faces, key=lambda f: int(f[2]) * int(f[3]), reverse=True)[0]
                                    fx, fy, fw, fh = [int(v) for v in largest]
                                    face_box = [x1 + fx, y1 + fy, x1 + fx + fw, y1 + fy + fh]
                        except Exception:
                            face_box = None

                    detections.append({"person_box": person_box, "face_box": face_box})
        except Exception as e:
            print(f"⚠️ SmartReframe: YOLO inference failed ({e})")

    if detections:
        return detections

    # Fallback: detect faces on full frame and synthesize person boxes.
    if face_cascade is not None:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
            for fx, fy, fw, fh in faces[:4]:
                fx, fy, fw, fh = int(fx), int(fy), int(fw), int(fh)
                pad_x = int(round(fw * 0.70))
                pad_y = int(round(fh * 1.30))
                px1 = max(0, fx - pad_x)
                py1 = max(0, fy - pad_y)
                px2 = min(frame_w, fx + fw + pad_x)
                py2 = min(frame_h, fy + fh + pad_y)
                detections.append({
                    "person_box": [px1, py1, px2, py2],
                    "face_box": [fx, fy, fx + fw, fy + fh]
                })
        except Exception:
            return []

    return detections

def _smart_ref_decide_strategy(
    scene_analysis: List[Dict[str, Any]],
    frame_w: int,
    frame_h: int,
    target_ratio: float
) -> Tuple[str, Optional[List[int]]]:
    count = len(scene_analysis)
    if count == 0:
        return "LETTERBOX", None
    if count == 1:
        chosen = scene_analysis[0].get("face_box") or scene_analysis[0].get("person_box")
        return "TRACK", chosen

    person_boxes = [obj.get("person_box") for obj in scene_analysis if isinstance(obj.get("person_box"), list)]
    group_box = _smart_ref_enclosing_box(person_boxes)
    if not group_box:
        return "LETTERBOX", None

    source_ratio = (frame_w / max(1, frame_h))
    if source_ratio >= target_ratio:
        max_crop_width = frame_h * target_ratio
        group_width = max(1.0, float(group_box[2] - group_box[0]))
        if group_width <= (max_crop_width * 1.02):
            return "TRACK", group_box
        return "LETTERBOX", None

    max_crop_height = frame_w / max(1e-6, target_ratio)
    group_height = max(1.0, float(group_box[3] - group_box[1]))
    if group_height <= (max_crop_height * 1.02):
        return "TRACK", group_box
    return "LETTERBOX", None

def _smart_ref_crop_box(
    target_box: List[int],
    frame_w: int,
    frame_h: int,
    target_ratio: float
) -> Tuple[int, int, int, int]:
    source_ratio = frame_w / max(1, frame_h)
    if source_ratio >= target_ratio:
        crop_h = frame_h
        crop_w = _even_int(crop_h * target_ratio)
    else:
        crop_w = frame_w
        crop_h = _even_int(crop_w / max(1e-6, target_ratio))

    crop_w = max(2, min(frame_w, crop_w))
    crop_h = max(2, min(frame_h, crop_h))

    center_x = (float(target_box[0]) + float(target_box[2])) / 2.0
    center_y = (float(target_box[1]) + float(target_box[3])) / 2.0

    x1 = int(round(center_x - (crop_w / 2.0)))
    y1 = int(round(center_y - (crop_h / 2.0)))

    x1 = max(0, min(max(0, frame_w - crop_w), x1))
    y1 = max(0, min(max(0, frame_h - crop_h), y1))
    x2 = x1 + crop_w
    y2 = y1 + crop_h
    return x1, y1, x2, y2

def _smart_ref_letterbox_frame(frame, out_w: int, out_h: int, np_mod):
    import cv2
    in_h, in_w = frame.shape[:2]
    if in_w <= 0 or in_h <= 0:
        return np_mod.zeros((out_h, out_w, 3), dtype=np_mod.uint8)

    scale = min(out_w / max(1, in_w), out_h / max(1, in_h))
    scaled_w = max(2, _even_int(in_w * scale))
    scaled_h = max(2, _even_int(in_h * scale))
    resized = cv2.resize(frame, (scaled_w, scaled_h))

    canvas = np_mod.zeros((out_h, out_w, 3), dtype=np_mod.uint8)
    pad_x = max(0, (out_w - scaled_w) // 2)
    pad_y = max(0, (out_h - scaled_h) // 2)
    canvas[pad_y:pad_y + scaled_h, pad_x:pad_x + scaled_w] = resized
    return canvas

def _smart_ref_detect_scene_ranges(
    video_path: str,
    total_frames: int,
    frame_skip: int = 1,
    downscale: int = 0
) -> List[Tuple[int, int]]:
    safe_total = max(1, int(total_frames or 1))
    try:
        try:
            from scenedetect import detect_scenes as sd_detect_scenes
        except ImportError:
            from scenedetect import detect as sd_detect_scenes
        try:
            from scenedetect import ContentDetector
        except ImportError:
            from scenedetect.detectors import ContentDetector
        scene_list = sd_detect_scenes(video_path, ContentDetector(), show_progress=False)
    except Exception as e:
        print(f"⚠️ SmartReframe: scene detection fallback ({e})")
        return [(0, safe_total)]

    ranges: List[Tuple[int, int]] = []
    for start_tc, end_tc in scene_list:
        try:
            start_f = max(0, int(start_tc.get_frames()))
            end_f = max(start_f + 1, int(end_tc.get_frames()))
            start_f = min(start_f, safe_total - 1)
            end_f = min(safe_total, max(start_f + 1, end_f))
            ranges.append((start_f, end_f))
        except Exception:
            continue

    if not ranges:
        return [(0, safe_total)]

    ranges.sort(key=lambda item: item[0])
    normalized: List[Tuple[int, int]] = []
    cursor = 0
    for start_f, end_f in ranges:
        start = max(cursor, int(start_f))
        end = min(safe_total, int(end_f))
        if end <= start:
            continue
        normalized.append((start, end))
        cursor = end

    if not normalized:
        return [(0, safe_total)]
    if normalized[-1][1] < safe_total:
        normalized.append((normalized[-1][1], safe_total))
    return normalized

def _smart_ref_analyze_scene(video_path: str, start_frame: int, end_frame: int) -> List[Dict[str, Any]]:
    try:
        import cv2
    except Exception:
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    middle = max(0, int(start_frame + max(0, end_frame - start_frame) // 2))
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return []
    return _smart_ref_detect_people(frame)

def _render_smart_reframe_video(
    input_video_path: str,
    output_path: str,
    aspect_ratio: str,
    scene_frame_skip: int = 1,
    scene_downscale: int = 0
) -> Dict[str, Any]:
    try:
        import cv2
        import numpy as np
    except Exception as e:
        raise RuntimeError(f"OpenCV/Numpy unavailable for smart reframe: {e}") from e

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open clipped video for smart reframe.")

    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if in_w <= 0 or in_h <= 0:
        cap.release()
        raise RuntimeError("Invalid input dimensions for smart reframe.")
    if fps <= 0:
        fps = 30.0
    if total_frames <= 0:
        total_frames = max(1, int(round(_probe_media_duration_seconds(input_video_path) * fps)))

    target_ratio = _aspect_ratio_to_float(aspect_ratio)
    out_w, out_h = _derive_output_dimensions(in_w, in_h, target_ratio)
    scene_ranges = _smart_ref_detect_scene_ranges(
        input_video_path,
        total_frames=total_frames,
        frame_skip=max(0, int(scene_frame_skip or 0)),
        downscale=max(0, int(scene_downscale or 0))
    )

    scene_plan: List[Dict[str, Any]] = []
    for start_f, end_f in scene_ranges:
        analysis = _smart_ref_analyze_scene(input_video_path, start_f, end_f)
        strategy, target_box = _smart_ref_decide_strategy(
            analysis,
            frame_w=in_w,
            frame_h=in_h,
            target_ratio=target_ratio
        )
        scene_plan.append({
            "start_frame": int(start_f),
            "end_frame": int(end_f),
            "strategy": strategy,
            "target_box": target_box
        })

    if not scene_plan:
        scene_plan = [{
            "start_frame": 0,
            "end_frame": max(1, total_frames),
            "strategy": "LETTERBOX",
            "target_box": None
        }]

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{out_w}x{out_h}",
        "-r", f"{fps:.6f}",
        "-i", "-",
        "-i", input_video_path,
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        "-preset", "fast",
        "-c:a", "aac",
        "-movflags", "+faststart",
        "-shortest",
        output_path
    ]
    ffmpeg_process = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE
    )

    frame_index = 0
    scene_index = 0
    dropped_frames = 0
    last_output_frame = None
    broken_pipe = False

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        while scene_index < len(scene_plan) - 1 and frame_index >= scene_plan[scene_index + 1]["start_frame"]:
            scene_index += 1

        scene = scene_plan[scene_index]
        try:
            if scene["strategy"] == "TRACK" and scene["target_box"]:
                x1, y1, x2, y2 = _smart_ref_crop_box(scene["target_box"], in_w, in_h, target_ratio)
                cropped = frame[y1:y2, x1:x2]
                if cropped.size == 0:
                    raise RuntimeError("Empty crop region")
                output_frame = cv2.resize(cropped, (out_w, out_h))
            else:
                output_frame = _smart_ref_letterbox_frame(frame, out_w, out_h, np)
            last_output_frame = output_frame
        except Exception:
            dropped_frames += 1
            if last_output_frame is not None:
                output_frame = last_output_frame
            else:
                output_frame = np.zeros((out_h, out_w, 3), dtype=np.uint8)

        try:
            if ffmpeg_process.stdin is None:
                broken_pipe = True
                break
            ffmpeg_process.stdin.write(output_frame.tobytes())
        except Exception:
            broken_pipe = True
            break
        frame_index += 1

    cap.release()
    try:
        if ffmpeg_process.stdin:
            ffmpeg_process.stdin.close()
    except Exception:
        pass
    stderr_text = ffmpeg_process.stderr.read().decode(errors="ignore") if ffmpeg_process.stderr else ""
    ffmpeg_process.wait()
    if ffmpeg_process.returncode != 0 or broken_pipe:
        raise RuntimeError(stderr_text or "Smart reframe encoding failed.")

    track_scenes = sum(1 for s in scene_plan if s["strategy"] == "TRACK")
    letterbox_scenes = sum(1 for s in scene_plan if s["strategy"] == "LETTERBOX")
    return {
        "scene_count": len(scene_plan),
        "track_scenes": track_scenes,
        "letterbox_scenes": letterbox_scenes,
        "frame_skip": max(0, int(scene_frame_skip or 0)),
        "downscale": max(0, int(scene_downscale or 0)),
        "dropped_frames": int(dropped_frames)
    }

def _extract_waveform_peaks(media_path: str, buckets: int = 240, sample_rate: int = 11025) -> List[float]:
    safe_buckets = max(32, min(2000, int(buckets)))
    safe_rate = max(2000, min(48000, int(sample_rate)))

    cmd = [
        "ffmpeg",
        "-v", "error",
        "-i", media_path,
        "-ac", "1",
        "-ar", str(safe_rate),
        "-vn",
        "-f", "f32le",
        "pipe:1"
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="ignore") or "ffmpeg waveform extraction failed")

    raw = proc.stdout or b""
    sample_count = len(raw) // 4
    if sample_count <= 0:
        return [0.0] * safe_buckets

    floats = struct.unpack(f"<{sample_count}f", raw[:sample_count * 4])
    chunk_size = max(1, int(math.ceil(sample_count / safe_buckets)))
    peaks: List[float] = []

    for i in range(safe_buckets):
        start = i * chunk_size
        if start >= sample_count:
            peaks.append(0.0)
            continue
        end = min(sample_count, start + chunk_size)
        peak = 0.0
        for value in floats[start:end]:
            amp = abs(float(value))
            if amp > peak:
                peak = amp
        peaks.append(peak)

    max_peak = max(peaks) if peaks else 0.0
    if max_peak <= 0:
        return [0.0] * len(peaks)
    return [round(min(1.0, p / max_peak), 4) for p in peaks]

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

def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()

def _extract_generated_text(response: Any) -> str:
    if response is None:
        return ""

    if isinstance(response, dict):
        direct = _normalize_space(response.get("text", ""))
        candidates = response.get("candidates") or []
    else:
        direct = _normalize_space(getattr(response, "text", ""))
        candidates = getattr(response, "candidates", None) or []
    if direct:
        return direct

    for candidate in candidates:
        if isinstance(candidate, dict):
            content = candidate.get("content")
        else:
            content = getattr(candidate, "content", None)
        if content is None:
            continue
        if isinstance(content, dict):
            parts = content.get("parts") or []
        else:
            parts = getattr(content, "parts", None) or []
        chunks: List[str] = []
        for part in parts:
            if isinstance(part, dict):
                text = part.get("text")
            else:
                text = getattr(part, "text", None)
            text_norm = _normalize_space(text)
            if text_norm:
                chunks.append(text_norm)
        merged = _normalize_space(" ".join(chunks))
        if merged:
            return merged
    return ""

def _sanitize_short_title(raw_title: str, max_chars: int = 95) -> str:
    text = _normalize_space(raw_title)
    text = text.strip(" \"'`")
    text = re.sub(r"^[\-\–\—:;,.!?¡¿\s]+", "", text).strip()
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = text.replace("#", "").replace("@", "")
    text = _normalize_space(text)
    if len(text) > max_chars:
        sliced = text[:max_chars]
        if " " in sliced:
            sliced = sliced.rsplit(" ", 1)[0]
        text = sliced.strip()
    return text

def _title_fingerprint(raw_title: str) -> str:
    clean = _sanitize_short_title(raw_title).lower()
    clean = unicodedata.normalize("NFD", clean)
    clean = "".join(ch for ch in clean if unicodedata.category(ch) != "Mn")
    return re.sub(r"[^a-z0-9]+", "", clean)

def _dedupe_title_candidates(candidates: List[str], blocked: Optional[List[str]] = None) -> List[str]:
    blocked_keys = {
        _title_fingerprint(item)
        for item in (blocked or [])
        if _title_fingerprint(item)
    }
    seen = set()
    out: List[str] = []
    for raw in (candidates or []):
        clean = _sanitize_short_title(raw)
        if not clean:
            continue
        key = _title_fingerprint(clean)
        if not key or key in seen or key in blocked_keys:
            continue
        seen.add(key)
        out.append(clean)
    return out

def _parse_title_variants_payload(raw_text: str) -> List[str]:
    payload = str(raw_text or "").strip()
    if not payload:
        return []

    block = payload
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", payload, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        block = fenced.group(1).strip()

    parsed_candidates: List[str] = []
    try:
        parsed = json.loads(block)
        if isinstance(parsed, list):
            parsed_candidates = [str(item) for item in parsed]
        elif isinstance(parsed, dict):
            for key in ("titles", "variants", "options"):
                value = parsed.get(key)
                if isinstance(value, list):
                    parsed_candidates = [str(item) for item in value]
                    break
                if isinstance(value, str):
                    parsed_candidates = [value]
                    break
    except Exception:
        parsed_candidates = []

    if parsed_candidates:
        return _dedupe_title_candidates(parsed_candidates)

    rough_lines = re.split(r"[\r\n]+|(?<!\d)\.\s+", block)
    fallback_candidates: List[str] = []
    for line in rough_lines:
        cleaned = re.sub(r"^\s*[-*•\d\)\.\:]+\s*", "", str(line or "")).strip(" \"'`")
        cleaned = _sanitize_short_title(cleaned)
        if cleaned:
            fallback_candidates.append(cleaned)
    return _dedupe_title_candidates(fallback_candidates)

def _social_fingerprint(raw_text: str) -> str:
    clean = _sanitize_social_copy(raw_text, max_chars=360).lower()
    clean = unicodedata.normalize("NFD", clean)
    clean = "".join(ch for ch in clean if unicodedata.category(ch) != "Mn")
    return re.sub(r"[^a-z0-9]+", "", clean)

def _dedupe_social_candidates(candidates: List[str], blocked: Optional[List[str]] = None) -> List[str]:
    blocked_keys = {
        _social_fingerprint(item)
        for item in (blocked or [])
        if _social_fingerprint(item)
    }
    seen = set()
    out: List[str] = []
    for raw in (candidates or []):
        clean = _sanitize_social_copy(raw, max_chars=320)
        if not clean:
            continue
        key = _social_fingerprint(clean)
        if not key or key in seen or key in blocked_keys:
            continue
        seen.add(key)
        out.append(clean)
    return out

def _parse_social_variants_payload(raw_text: str) -> List[str]:
    payload = str(raw_text or "").strip()
    if not payload:
        return []

    block = payload
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", payload, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        block = fenced.group(1).strip()

    parsed_candidates: List[str] = []
    try:
        parsed = json.loads(block)
        if isinstance(parsed, list):
            parsed_candidates = [str(item) for item in parsed]
        elif isinstance(parsed, dict):
            for key in ("socials", "copies", "variants", "options", "captions"):
                value = parsed.get(key)
                if isinstance(value, list):
                    parsed_candidates = [str(item) for item in value]
                    break
                if isinstance(value, str):
                    parsed_candidates = [value]
                    break
    except Exception:
        parsed_candidates = []

    if parsed_candidates:
        return _dedupe_social_candidates(parsed_candidates)

    rough_lines = re.split(r"[\r\n]+|(?<!\d)\.\s+", block)
    fallback_candidates: List[str] = []
    for line in rough_lines:
        cleaned = re.sub(r"^\s*[-*•\d\)\.\:]+\s*", "", str(line or "")).strip(" \"'`")
        cleaned = _sanitize_social_copy(cleaned, max_chars=320)
        if cleaned:
            fallback_candidates.append(cleaned)
    return _dedupe_social_candidates(fallback_candidates)

def _build_fallback_title(
    current_title: str,
    transcript_excerpt: str,
    topic_tags: List[str],
    avoid_title: str
) -> str:
    clean_current = _sanitize_short_title(current_title)
    clean_avoid = _sanitize_short_title(avoid_title).lower()

    keyword = ""
    for tag in topic_tags or []:
        t = _sanitize_short_title(tag, max_chars=24).lower()
        if len(t) >= 4:
            keyword = t
            break
    if not keyword:
        words = re.findall(r"[a-zA-ZÀ-ÿ0-9]{4,}", str(transcript_excerpt or "").lower())
        stop = {"esto", "esta", "este", "para", "como", "cuando", "donde", "sobre", "porque", "video", "clip"}
        for word in words:
            if word in stop:
                continue
            keyword = word
            break

    hooks = [
        "cambia el debate",
        "deja una alerta clara",
        "explica el punto clave",
        "abre una discusión fuerte",
        "resume lo más importante"
    ]
    lead = [
        "Lo que no te contaron",
        "La parte más fuerte",
        "El momento que explica todo",
        "Esta frase lo resume",
        "Así lo dijo sin filtro"
    ]
    seed_raw = f"{clean_current}|{transcript_excerpt}|{clean_avoid}|{int(time.time())}"
    seed = zlib.crc32(seed_raw.encode("utf-8"))
    lead_text = lead[seed % len(lead)]
    hook_text = hooks[(seed // max(1, len(lead))) % len(hooks)]

    candidates = []
    if keyword:
        candidates.append(f"{lead_text}: {keyword} y por qué {hook_text}")
        candidates.append(f"{keyword}: {hook_text} en este corte")
    if clean_current:
        candidates.append(f"{clean_current} | {hook_text}")
    candidates.append(f"{lead_text} y por qué {hook_text}")

    for candidate in candidates:
        clean = _sanitize_short_title(candidate)
        if not clean:
            continue
        if clean.lower() == clean_avoid:
            continue
        return clean
    return _sanitize_short_title(clean_current or "Momento clave del video")

def _generate_rewritten_title(
    current_title: str,
    transcript_excerpt: str,
    social_excerpt: str,
    topic_tags: List[str],
    avoid_title: str,
    api_key: Optional[str]
) -> str:
    clean_current = _sanitize_short_title(current_title or avoid_title or "")
    clean_avoid = _sanitize_short_title(avoid_title).lower()
    clean_social = _normalize_space(social_excerpt)[:300]
    clean_transcript = _normalize_space(transcript_excerpt)[:420]
    safe_tags = [str(tag).strip().lower()[:24] for tag in (topic_tags or []) if str(tag).strip()]
    tag_line = ", ".join(safe_tags[:6])

    if api_key:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            prompt = (
                "Reescribe SOLO el titulo para un clip corto vertical.\n"
                "Devuelve una sola linea sin comillas.\n"
                "Reglas: español neutro, 55-95 caracteres, gancho claro, sin emojis, sin hashtags, sin clickbait engañoso.\n"
                f"Evita repetir literalmente este titulo: {clean_avoid or clean_current or 'n/a'}.\n"
                f"Titulo actual: {clean_current or 'n/a'}\n"
                f"Contexto social: {clean_social or 'n/a'}\n"
                f"Contexto transcript: {clean_transcript or 'n/a'}\n"
                f"Etiquetas: {tag_line or 'n/a'}"
            )
            for model_name in TITLE_REWRITE_MODELS:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt
                    )
                except Exception as model_err:
                    if _is_gemini_model_unavailable_error(model_err):
                        continue
                    raise
                generated = _sanitize_short_title(_extract_generated_text(response))
                if generated and generated.lower() != clean_avoid:
                    return generated
        except Exception:
            pass

    return _build_fallback_title(
        current_title=clean_current,
        transcript_excerpt=clean_transcript,
        topic_tags=safe_tags,
        avoid_title=clean_avoid
    )

def _generate_rewritten_title_variants(
    current_title: str,
    transcript_excerpt: str,
    social_excerpt: str,
    topic_tags: List[str],
    avoid_titles: Optional[List[str]],
    target_count: int,
    api_key: Optional[str]
) -> List[str]:
    safe_target = max(1, min(8, int(target_count or 1)))
    clean_current = _sanitize_short_title(current_title or "Momento clave del video")
    clean_social = _normalize_space(social_excerpt)[:320]
    clean_transcript = _normalize_space(transcript_excerpt)[:460]
    safe_tags = [str(tag).strip().lower()[:24] for tag in (topic_tags or []) if str(tag).strip()]
    blocked = _dedupe_title_candidates(list(avoid_titles or []) + [clean_current])
    results: List[str] = []

    if api_key:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            blocked_line = "; ".join(blocked[:8]) if blocked else "n/a"
            tag_line = ", ".join(safe_tags[:6]) if safe_tags else "n/a"
            prompt = (
                f"Genera exactamente {safe_target} títulos distintos para un clip vertical.\n"
                "Responde SOLO como JSON array de strings y nada más.\n"
                "Reglas: español neutro, 55-95 caracteres, puedes usar emojis relevantes, sin hashtags, sin comillas extra.\n"
                "Estilo: Usa 'Sentence case' (ej: 'Así me recibe la gente en Argentina'), evita mayúsculas en cada palabra.\n"
                f"Evita repetir literalmente estos títulos: {blocked_line}\n"
                f"Título base: {clean_current or 'n/a'}\n"
                f"Contexto social: {clean_social or 'n/a'}\n"
                f"Contexto transcript: {clean_transcript or 'n/a'}\n"
                f"Etiquetas: {tag_line}"
            )
            for model_name in TITLE_REWRITE_MODELS:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt
                    )
                except Exception as model_err:
                    if _is_gemini_model_unavailable_error(model_err):
                        continue
                    raise

                raw = _extract_generated_text(response)
                parsed = _parse_title_variants_payload(raw)
                parsed = _dedupe_title_candidates(parsed, blocked=blocked + results)
                if parsed:
                    results.extend(parsed)
                    results = _dedupe_title_candidates(results, blocked=blocked)
                if len(results) >= safe_target:
                    break
        except Exception:
            pass

    attempts = max(8, safe_target * 4)
    while len(results) < safe_target and attempts > 0:
        avoid_line = " | ".join((blocked + results)[-10:] or [clean_current])
        candidate = _generate_rewritten_title(
            current_title=clean_current,
            transcript_excerpt=clean_transcript,
            social_excerpt=clean_social,
            topic_tags=safe_tags,
            avoid_title=avoid_line,
            api_key=api_key
        )
        deduped = _dedupe_title_candidates([candidate], blocked=blocked + results)
        if deduped:
            results.extend(deduped)
        attempts -= 1

    filler_guard = 0
    while len(results) < safe_target and filler_guard < (safe_target * 4):
        salt = f"{int(time.time() * 1000)}-{len(results)}-{filler_guard}"
        candidate = _build_fallback_title(
            current_title=clean_current,
            transcript_excerpt=clean_transcript,
            topic_tags=safe_tags,
            avoid_title=" | ".join((blocked + results + [salt])[-12:])
        )
        deduped = _dedupe_title_candidates([candidate], blocked=blocked + results)
        if deduped:
            results.extend(deduped)
        filler_guard += 1

    if len(results) < safe_target:
        keyword = ""
        for tag in safe_tags:
            if len(tag) >= 4:
                keyword = tag
                break
        if not keyword:
            words = re.findall(r"[a-zA-ZÀ-ÿ0-9]{4,}", clean_transcript.lower())
            for word in words:
                if word not in {"esto", "esta", "este", "para", "como", "cuando", "donde", "sobre", "porque", "video", "clip"}:
                    keyword = word
                    break

        lead_opts = [
            "Lo que no te contaron",
            "La parte más fuerte",
            "El momento que explica todo",
            "Esta frase lo resume",
            "Así lo dijo sin filtro",
            "El dato que cambia la lectura"
        ]
        hook_opts = [
            "abre una discusión fuerte",
            "cambia el debate",
            "deja una alerta clara",
            "explica el punto clave",
            "resumen el punto central",
            "marca la conversación"
        ]
        synth_idx = 0
        while len(results) < safe_target and synth_idx < (len(lead_opts) * len(hook_opts) + 8):
            lead = lead_opts[synth_idx % len(lead_opts)]
            hook = hook_opts[(synth_idx // max(1, len(lead_opts))) % len(hook_opts)]
            if keyword:
                candidate = f"{lead}: {keyword} y por qué {hook}"
            else:
                candidate = f"{lead}: por qué {hook}"
            deduped = _dedupe_title_candidates([candidate], blocked=blocked + results)
            if deduped:
                results.extend(deduped)
            synth_idx += 1

    return _dedupe_title_candidates(results, blocked=blocked)[:safe_target]

def _sanitize_social_copy(text: str, max_chars: int = 280) -> str:
    raw = _normalize_space(text)
    if not raw:
        return ""
    raw = raw.replace("\n", " ").strip()
    raw = re.sub(r"\s+", " ", raw)
    if len(raw) > max_chars:
        cut = raw[:max_chars].rsplit(" ", 1)[0].strip()
        raw = cut or raw[:max_chars].strip()
    return raw

def _ensure_social_cta(text: str, token_hint: str = "CLIP") -> str:
    clean = _sanitize_social_copy(text)
    if not clean:
        return f'Sígueme y comenta "{token_hint}" y te envío más análisis.'
    if re.search(r"sig[uú]eme\s+y\s+comenta", clean, flags=re.IGNORECASE):
        return clean
    suffix = f'Sígueme y comenta "{token_hint}" y te envío más análisis.'
    return _sanitize_social_copy(f"{clean} {suffix}")

def _build_fallback_social_copy(
    current_social: str,
    current_title: str,
    transcript_excerpt: str,
    score_reason: str,
    topic_tags: List[str]
) -> str:
    seed_raw = "|".join([
        str(current_social or ""),
        str(current_title or ""),
        str(transcript_excerpt or ""),
        str(score_reason or ""),
        ",".join(topic_tags or [])
    ])
    seed = zlib.crc32(seed_raw.encode("utf-8"))

    base = _sanitize_social_copy(current_social)
    if not base:
        title = _sanitize_short_title(current_title or "Este clip")
        score = _normalize_space(score_reason or "")
        score = score[:120]
        openers = [
            f"{title}: este momento resume el punto más fuerte del debate.",
            f"{title}: aquí se explica por qué este tema está generando tanta conversación.",
            f"{title}: un corte clave para entender el contexto completo."
        ]
        base = openers[seed % len(openers)]
        if score:
            base = f"{base} {score}"

    token = "CLIP"
    for tag in topic_tags or []:
        normalized = _sanitize_short_title(tag, max_chars=24)
        if normalized and len(normalized) >= 4:
            token = normalized.upper()
            break
    if token == "CLIP":
        words = re.findall(r"[a-zA-ZÀ-ÿ0-9]{4,}", str(transcript_excerpt or ""))
        for word in words[:20]:
            token = word.upper()
            break

    return _ensure_social_cta(base, token_hint=token[:18] if token else "CLIP")

def _generate_rewritten_social_copy(
    current_social: str,
    current_title: str,
    transcript_excerpt: str,
    score_reason: str,
    topic_tags: List[str],
    api_key: Optional[str]
) -> str:
    clean_current = _sanitize_social_copy(current_social, max_chars=340)
    clean_title = _sanitize_short_title(current_title or "")
    clean_transcript = _normalize_space(transcript_excerpt)[:420]
    clean_reason = _normalize_space(score_reason)[:220]
    safe_tags = [str(tag).strip().lower()[:24] for tag in (topic_tags or []) if str(tag).strip()]
    tag_line = ", ".join(safe_tags[:6]) or "n/a"

    if api_key:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            prompt = (
                "Reescribe SOLO el texto social para un clip vertical.\n"
                "Devuelve una sola línea sin comillas.\n"
                "Reglas: español neutro, 150-280 caracteres, claro y natural, sin inventar datos, máximo 2 hashtags.\n"
                "Incluye CTA al final con formato: Sígueme y comenta \"PALABRA\" y te envío más análisis.\n"
                f"Texto actual: {clean_current or 'n/a'}\n"
                f"Título del clip: {clean_title or 'n/a'}\n"
                f"Transcript: {clean_transcript or 'n/a'}\n"
                f"Razón de viralidad: {clean_reason or 'n/a'}\n"
                f"Etiquetas: {tag_line}"
            )
            for model_name in SOCIAL_REWRITE_MODELS:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt
                    )
                except Exception as model_err:
                    if _is_gemini_model_unavailable_error(model_err):
                        continue
                    raise
                generated = _sanitize_social_copy(_extract_generated_text(response))
                generated = _ensure_social_cta(generated, token_hint="CLIP")
                if generated:
                    return generated
        except Exception:
            pass

    return _build_fallback_social_copy(
        current_social=clean_current,
        current_title=clean_title,
        transcript_excerpt=clean_transcript,
        score_reason=clean_reason,
        topic_tags=safe_tags
    )

def _generate_rewritten_social_variants(
    current_social: str,
    current_title: str,
    transcript_excerpt: str,
    score_reason: str,
    topic_tags: List[str],
    avoid_socials: Optional[List[str]],
    target_count: int,
    api_key: Optional[str]
) -> List[str]:
    safe_target = max(1, min(8, int(target_count or 1)))
    clean_current = _sanitize_social_copy(current_social, max_chars=320)
    clean_title = _sanitize_short_title(current_title or "")
    clean_transcript = _normalize_space(transcript_excerpt)[:460]
    clean_reason = _normalize_space(score_reason)[:240]
    safe_tags = [str(tag).strip().lower()[:24] for tag in (topic_tags or []) if str(tag).strip()]
    blocked = _dedupe_social_candidates(list(avoid_socials or []) + ([clean_current] if clean_current else []))
    results: List[str] = []

    if api_key:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            blocked_line = " || ".join(blocked[:8]) if blocked else "n/a"
            tag_line = ", ".join(safe_tags[:6]) if safe_tags else "n/a"
            prompt = (
                f"Genera exactamente {safe_target} copies sociales distintas para un clip vertical.\n"
                "Responde SOLO como JSON array de strings y nada más.\n"
                "Reglas: español neutro, 150-280 caracteres, tono natural con emojis relevantes, sin inventar datos, máximo 2 hashtags.\n"
                "Cada copy debe terminar con CTA en formato: Sígueme y comenta \"PALABRA\" y te envío más análisis.\n"
                f"Evita repetir literalmente estas copies: {blocked_line}\n"
                f"Título del clip: {clean_title or 'n/a'}\n"
                f"Copy base: {clean_current or 'n/a'}\n"
                f"Transcript: {clean_transcript or 'n/a'}\n"
                f"Razón de viralidad: {clean_reason or 'n/a'}\n"
                f"Etiquetas: {tag_line}"
            )
            for model_name in SOCIAL_REWRITE_MODELS:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt
                    )
                except Exception as model_err:
                    if _is_gemini_model_unavailable_error(model_err):
                        continue
                    raise
                raw = _extract_generated_text(response)
                parsed = _parse_social_variants_payload(raw)
                parsed = [_ensure_social_cta(_sanitize_social_copy(item, max_chars=320), token_hint="CLIP") for item in parsed]
                parsed = _dedupe_social_candidates(parsed, blocked=blocked + results)
                if parsed:
                    results.extend(parsed)
                    results = _dedupe_social_candidates(results, blocked=blocked)
                if len(results) >= safe_target:
                    break
        except Exception:
            pass

    filler_guard = 0
    while len(results) < safe_target and filler_guard < (safe_target * 6):
        salt = f"{int(time.time() * 1000)}-{len(results)}-{filler_guard}"
        candidate = _build_fallback_social_copy(
            current_social=clean_current,
            current_title=clean_title,
            transcript_excerpt=clean_transcript,
            score_reason=f"{clean_reason} {salt}".strip(),
            topic_tags=safe_tags
        )
        candidate = _sanitize_social_copy(candidate, max_chars=320)
        deduped = _dedupe_social_candidates([candidate], blocked=blocked + results)
        if deduped:
            results.extend(deduped)
        filler_guard += 1

    if len(results) < safe_target:
        token_hint = "CLIP"
        for tag in safe_tags:
            if len(tag) >= 4:
                token_hint = tag.upper()
                break
        if token_hint == "CLIP":
            words = re.findall(r"[a-zA-ZÀ-ÿ0-9]{4,}", clean_transcript)
            if words:
                token_hint = words[0].upper()
        short_tag = safe_tags[0] if safe_tags else ""
        hashtag = f" #{short_tag}" if short_tag else ""
        base_title = clean_title or "Este clip"
        openers = [
            "resume el punto más fuerte",
            "abre el debate en segundos",
            "explica una parte clave del tema",
            "muestra por qué este tema está en tendencia",
            "deja una lectura clara del contexto",
            "te da el momento exacto del debate"
        ]
        questions = [
            "¿Qué opinas tú?",
            "¿Te pasó algo parecido?",
            "¿Estás de acuerdo con este punto?",
            "¿Lo compartirías con tu equipo?",
            "¿Qué agregarías a este análisis?"
        ]
        synth_idx = 0
        while len(results) < safe_target and synth_idx < (len(openers) * len(questions) + 8):
            opener = openers[synth_idx % len(openers)]
            question = questions[(synth_idx // max(1, len(openers))) % len(questions)]
            body = _sanitize_social_copy(
                f"{base_title}: {opener}. {question}{hashtag}",
                max_chars=260
            )
            candidate = _ensure_social_cta(body, token_hint=token_hint[:18] if token_hint else "CLIP")
            candidate = _sanitize_social_copy(candidate, max_chars=320)
            deduped = _dedupe_social_candidates([candidate], blocked=blocked + results)
            if deduped:
                results.extend(deduped)
            synth_idx += 1

    return _dedupe_social_candidates(results, blocked=blocked)[:safe_target]

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

def _ensure_clip_title_variant_pool(
    clip: Dict[str, Any],
    transcript: Optional[Dict[str, Any]],
    api_key: Optional[str],
    target_size: int = TITLE_VARIANTS_PER_CLIP
) -> Tuple[List[str], int, bool]:
    if not isinstance(clip, dict):
        return [], 0, False

    safe_target = max(2, min(8, int(target_size or TITLE_VARIANTS_PER_CLIP)))
    current_title = _sanitize_short_title(
        clip.get("video_title_for_youtube_short")
        or clip.get("title")
        or "Momento clave del video"
    ) or "Momento clave del video"
    current_fp = _title_fingerprint(current_title)

    raw_variants = clip.get("title_variants") if isinstance(clip.get("title_variants"), list) else []
    variants = _dedupe_title_candidates(raw_variants)
    variant_keys = {_title_fingerprint(v) for v in variants if _title_fingerprint(v)}
    if current_fp and current_fp not in variant_keys:
        variants = _dedupe_title_candidates([current_title] + variants)
    elif not variants:
        variants = [current_title]

    changed = False
    if len(variants) < safe_target:
        social_excerpt = _normalize_space(" ".join([
            str(clip.get("video_description_for_tiktok", "")),
            str(clip.get("video_description_for_instagram", "")),
            str(clip.get("score_reason", ""))
        ]))
        transcript_excerpt = _normalize_space(
            clip.get("transcript_excerpt")
            or _clip_transcript_excerpt(clip, transcript, max_chars=420, max_segments=10)
        )
        tags = _normalize_topic_tags(clip.get("topic_tags"))
        missing = safe_target - len(variants)
        generated = _generate_rewritten_title_variants(
            current_title=current_title,
            transcript_excerpt=transcript_excerpt,
            social_excerpt=social_excerpt,
            topic_tags=tags,
            avoid_titles=variants,
            target_count=missing,
            api_key=api_key
        )
        if generated:
            variants = _dedupe_title_candidates(variants + generated)
            changed = True

    raw_idx = clip.get("title_variant_index")
    idx: Optional[int] = None
    if raw_idx is not None:
        try:
            idx = int(raw_idx)
        except Exception:
            idx = None
    if idx is None:
        idx = 0
        if current_fp:
            for i, candidate in enumerate(variants):
                if _title_fingerprint(candidate) == current_fp:
                    idx = i
                    break
    idx = max(0, min(len(variants) - 1, idx)) if variants else 0

    active_title = variants[idx] if variants else current_title
    if clip.get("video_title_for_youtube_short") != active_title:
        clip["video_title_for_youtube_short"] = active_title
        changed = True
    if clip.get("title") != active_title:
        clip["title"] = active_title
        changed = True
    if clip.get("title_variants") != variants:
        clip["title_variants"] = variants
        changed = True
    prev_idx_raw = clip.get("title_variant_index")
    try:
        prev_idx = int(prev_idx_raw)
    except Exception:
        prev_idx = None
    if prev_idx != idx:
        clip["title_variant_index"] = idx
        changed = True

    return variants, idx, changed

def _ensure_clip_social_variant_pool(
    clip: Dict[str, Any],
    transcript: Optional[Dict[str, Any]],
    api_key: Optional[str],
    target_size: int = SOCIAL_VARIANTS_PER_CLIP
) -> Tuple[List[str], int, bool]:
    if not isinstance(clip, dict):
        return [], 0, False

    safe_target = max(2, min(8, int(target_size or SOCIAL_VARIANTS_PER_CLIP)))
    current_social = _sanitize_social_copy(
        clip.get("video_description_for_tiktok")
        or clip.get("video_description_for_instagram")
        or "",
        max_chars=320
    )
    current_title = _sanitize_short_title(
        clip.get("video_title_for_youtube_short")
        or clip.get("title")
        or "Momento clave del video"
    ) or "Momento clave del video"
    transcript_excerpt = _normalize_space(
        clip.get("transcript_excerpt")
        or _clip_transcript_excerpt(clip, transcript, max_chars=420, max_segments=10)
    )
    score_reason = _normalize_space(clip.get("score_reason") or "")
    tags = _normalize_topic_tags(clip.get("topic_tags"))

    if not current_social:
        current_social = _build_fallback_social_copy(
            current_social="",
            current_title=current_title,
            transcript_excerpt=transcript_excerpt,
            score_reason=score_reason,
            topic_tags=tags
        )

    current_fp = _social_fingerprint(current_social)
    raw_variants = clip.get("social_variants") if isinstance(clip.get("social_variants"), list) else []
    variants = _dedupe_social_candidates(raw_variants)
    variant_keys = {_social_fingerprint(v) for v in variants if _social_fingerprint(v)}
    if current_fp and current_fp not in variant_keys:
        variants = _dedupe_social_candidates([current_social] + variants)
    elif not variants:
        variants = [current_social]

    changed = False
    if len(variants) < safe_target:
        missing = safe_target - len(variants)
        generated = _generate_rewritten_social_variants(
            current_social=current_social,
            current_title=current_title,
            transcript_excerpt=transcript_excerpt,
            score_reason=score_reason,
            topic_tags=tags,
            avoid_socials=variants,
            target_count=missing,
            api_key=api_key
        )
        if generated:
            variants = _dedupe_social_candidates(variants + generated)
            changed = True

    raw_idx = clip.get("social_variant_index")
    idx: Optional[int] = None
    if raw_idx is not None:
        try:
            idx = int(raw_idx)
        except Exception:
            idx = None
    if idx is None:
        idx = 0
        if current_fp:
            for i, candidate in enumerate(variants):
                if _social_fingerprint(candidate) == current_fp:
                    idx = i
                    break
    idx = max(0, min(len(variants) - 1, idx)) if variants else 0

    active_social = variants[idx] if variants else current_social
    if clip.get("video_description_for_tiktok") != active_social:
        clip["video_description_for_tiktok"] = active_social
        changed = True
    if clip.get("video_description_for_instagram") != active_social:
        clip["video_description_for_instagram"] = active_social
        changed = True
    if clip.get("social_variants") != variants:
        clip["social_variants"] = variants
        changed = True
    prev_idx_raw = clip.get("social_variant_index")
    try:
        prev_idx = int(prev_idx_raw)
    except Exception:
        prev_idx = None
    if prev_idx != idx:
        clip["social_variant_index"] = idx
        changed = True

    return variants, idx, changed

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

def _jobs_db_connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(JOBS_DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(JOBS_DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def _init_jobs_store():
    with _JOBS_DB_LOCK:
        conn = _jobs_db_connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS job_state (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                """
            )
            conn.commit()
        finally:
            conn.close()

def _safe_job_snapshot(job: Dict[str, Any]) -> Dict[str, Any]:
    snapshot = dict(job or {})

    logs = snapshot.get("logs")
    if isinstance(logs, list):
        snapshot["logs"] = [str(item) for item in logs][-1600:]
    else:
        snapshot["logs"] = []

    social_posts = snapshot.get("social_posts")
    if isinstance(social_posts, list):
        snapshot["social_posts"] = social_posts[-300:]
    else:
        snapshot["social_posts"] = []

    env_payload = snapshot.get("env")
    if isinstance(env_payload, dict):
        snapshot["env"] = {str(k): str(v) for k, v in env_payload.items()}

    return snapshot

def _persist_job_state(job_id: str):
    job = jobs.get(job_id)
    if not isinstance(job, dict):
        return
    now = time.time()
    job["updated_at"] = now
    snapshot = _safe_job_snapshot(job)
    status = str(snapshot.get("status", "unknown"))
    payload = json.dumps(snapshot, ensure_ascii=False, default=str)

    with _JOBS_DB_LOCK:
        conn = _jobs_db_connect()
        try:
            conn.execute(
                """
                INSERT INTO job_state (job_id, status, payload_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status = excluded.status,
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (job_id, status, payload, now),
            )
            conn.commit()
        finally:
            conn.close()

def _delete_persisted_job(job_id: str):
    with _JOBS_DB_LOCK:
        conn = _jobs_db_connect()
        try:
            conn.execute("DELETE FROM job_state WHERE job_id = ?", (job_id,))
            conn.commit()
        finally:
            conn.close()

def _is_uuid_like_job_id(value: str) -> bool:
    candidate = str(value or "").strip()
    if not candidate:
        return False
    try:
        uuid.UUID(candidate)
        return True
    except Exception:
        return False

def _discover_job_ids_on_disk() -> Set[str]:
    discovered: Set[str] = set()

    try:
        for entry in os.listdir(OUTPUT_DIR):
            candidate = str(entry).strip()
            if _is_uuid_like_job_id(candidate):
                discovered.add(candidate)
    except Exception:
        pass

    try:
        for path in glob.glob(os.path.join(UPLOAD_DIR, "*")):
            base = os.path.basename(path)
            if not base:
                continue
            prefix = base.split("_", 1)[0].strip()
            if _is_uuid_like_job_id(prefix):
                discovered.add(prefix)
    except Exception:
        pass

    return discovered

def _delete_path_if_exists(path: str) -> bool:
    try:
        if not os.path.exists(path):
            return False
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            os.remove(path)
        return True
    except Exception:
        return False

def _purge_job_artifacts(job_id: str, include_artifacts: bool = True, include_uploads: bool = True) -> Dict[str, int]:
    removed_artifact_paths = 0
    removed_upload_paths = 0

    if include_artifacts:
        job_output_dir = os.path.join(OUTPUT_DIR, job_id)
        if _delete_path_if_exists(job_output_dir):
            removed_artifact_paths += 1
        for pattern in (
            os.path.join(OUTPUT_DIR, f"{job_id}_*"),
            os.path.join(OUTPUT_DIR, f"temp_{job_id}_*"),
        ):
            for path in glob.glob(pattern):
                if _delete_path_if_exists(path):
                    removed_artifact_paths += 1

    if include_uploads:
        for path in glob.glob(os.path.join(UPLOAD_DIR, f"{job_id}_*")):
            if _delete_path_if_exists(path):
                removed_upload_paths += 1

    return {
        "removed_artifact_paths": removed_artifact_paths,
        "removed_upload_paths": removed_upload_paths,
    }

def _load_persisted_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _JOBS_DB_LOCK:
        conn = _jobs_db_connect()
        try:
            row = conn.execute(
                "SELECT payload_json FROM job_state WHERE job_id = ? LIMIT 1",
                (job_id,)
            ).fetchone()
        finally:
            conn.close()
    if not row:
        return None
    try:
        payload = json.loads(row[0])
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None

def _load_all_persisted_jobs(limit: int = 400) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    safe_limit = max(10, min(4000, int(limit)))
    with _JOBS_DB_LOCK:
        conn = _jobs_db_connect()
        try:
            rows = conn.execute(
                "SELECT job_id, payload_json FROM job_state ORDER BY updated_at DESC LIMIT ?",
                (safe_limit,)
            ).fetchall()
        finally:
            conn.close()
    for row in rows:
        try:
            job_id = str(row[0])
            payload = json.loads(row[1])
            if isinstance(payload, dict):
                out[job_id] = payload
        except Exception:
            continue
    return out

def _metadata_path_for_job(job_id: str) -> Optional[str]:
    output_dir = os.path.join(OUTPUT_DIR, job_id)
    if not os.path.isdir(output_dir):
        return None
    candidates = sorted(glob.glob(os.path.join(output_dir, "*_metadata.json")))
    if not candidates:
        return None
    return candidates[0]

def _materialize_result_from_metadata(job_id: str, metadata: Dict[str, Any], metadata_path: str) -> Dict[str, Any]:
    output_dir = os.path.join(OUTPUT_DIR, job_id)
    transcript_data = metadata.get("transcript") if isinstance(metadata, dict) else None
    shorts = metadata.get("shorts", []) if isinstance(metadata, dict) else []
    if not isinstance(shorts, list):
        shorts = []

    base_name = os.path.basename(metadata_path).replace("_metadata.json", "")
    clips: List[Dict[str, Any]] = []
    trailer_url = str(metadata.get("latest_trailer_url", "") or "").strip() if isinstance(metadata, dict) else ""
    for i, raw_clip in enumerate(shorts):
        clip = _normalize_clip_payload(
            dict(raw_clip) if isinstance(raw_clip, dict) else {},
            i,
            transcript=transcript_data
        )
        video_url = str(clip.get("video_url", "") or "").strip()
        if video_url:
            clip_name = _safe_input_filename(video_url)
            if clip_name:
                clip["video_url"] = f"/videos/{job_id}/{clip_name}"
        else:
            clip_name = f"{base_name}_clip_{i+1}.mp4"
            if os.path.exists(os.path.join(output_dir, clip_name)):
                clip["video_url"] = f"/videos/{job_id}/{clip_name}"
        clip = _repair_trailer_clip_range(clip, output_dir)
        if _looks_like_trailer_clip(clip):
            raw_markers = clip.get("transition_points")
            if not isinstance(raw_markers, list) or not raw_markers:
                clip_duration = max(
                    0.0,
                    _safe_float(clip.get("end", 0.0), 0.0) - _safe_float(clip.get("start", 0.0), 0.0)
                )
                timeline_meta = _build_trailer_timeline_from_fragments(
                    metadata.get("trailer_fragments", []) if isinstance(metadata, dict) else [],
                    fade_duration=_safe_float(clip.get("transition_duration", 0.5), 0.5),
                    duration_cap=clip_duration if clip_duration > 0 else None
                )
                if timeline_meta.get("transition_points"):
                    clip["transition_points"] = timeline_meta["transition_points"]
                    clip["fragment_ranges"] = timeline_meta.get("fragment_ranges", [])
                    clip["transition_duration"] = timeline_meta.get("fade_duration", 0.5)
        clips.append(clip)

    if not clips and trailer_url:
        trailer_name = _safe_input_filename(trailer_url)
        trailer_path = os.path.join(output_dir, trailer_name) if trailer_name else ""
        trailer_duration = _probe_media_duration_seconds(trailer_path) if trailer_path and os.path.exists(trailer_path) else 0.0
        synthetic_clip = _normalize_clip_payload(
            {
                "clip_index": 0,
                "start": 0.0,
                "end": max(3.0, float(trailer_duration or 0.0)),
                "video_title_for_youtube_short": "Super Trailer",
                "video_description_for_tiktok": "Resumen rápido con los mejores momentos.",
                "video_description_for_instagram": "Resumen rápido con los mejores momentos.",
                "virality_score": 90,
                "selection_confidence": 0.9,
                "is_trailer": True,
            },
            0,
            transcript=transcript_data
        )
        if trailer_name:
            synthetic_clip["video_url"] = f"/videos/{job_id}/{trailer_name}"
        timeline_meta = _build_trailer_timeline_from_fragments(
            metadata.get("trailer_fragments", []) if isinstance(metadata, dict) else [],
            fade_duration=0.5,
            duration_cap=max(0.0, _safe_float(synthetic_clip.get("end", 0.0), 0.0))
        )
        if timeline_meta.get("transition_points"):
            synthetic_clip["transition_points"] = timeline_meta["transition_points"]
            synthetic_clip["fragment_ranges"] = timeline_meta.get("fragment_ranges", [])
            synthetic_clip["transition_duration"] = timeline_meta.get("fade_duration", 0.5)
        clips = [synthetic_clip]

    result_payload: Dict[str, Any] = {
        "clips": clips,
        "cost_analysis": metadata.get("cost_analysis") if isinstance(metadata, dict) else None
    }
    if trailer_url:
        result_payload["latest_trailer_url"] = trailer_url
    if isinstance(metadata.get("highlight_reels"), list):
        result_payload["highlight_reels"] = metadata.get("highlight_reels", [])
        if result_payload["highlight_reels"]:
            result_payload["latest_highlight_reel"] = result_payload["highlight_reels"][-1]
    return result_payload

def _normalize_recovered_job(job_id: str, payload: Dict[str, Any], source: str) -> Dict[str, Any]:
    job = dict(payload or {})
    job["status"] = str(job.get("status", "completed"))
    job["output_dir"] = str(job.get("output_dir") or os.path.join(OUTPUT_DIR, job_id))
    raw_input = job.get("input_path")
    job["input_path"] = str(raw_input) if raw_input else None
    job["logs"] = [str(item) for item in (job.get("logs") if isinstance(job.get("logs"), list) else [])][-1600:]
    if not job["logs"]:
        job["logs"] = [f"Recovered from {source}."]
    elif not any("Recovered from" in entry for entry in job["logs"][-3:]):
        job["logs"].append(f"Recovered from {source}.")

    for key, default in (
        ("attempt_count", 0),
        ("auto_retry_count", 0),
        ("manual_retry_count", 0),
        ("max_auto_retries", MAX_AUTO_RETRIES_DEFAULT),
        ("retry_delay_seconds", JOB_RETRY_DELAY_SECONDS_DEFAULT),
    ):
        try:
            job[key] = int(job.get(key, default))
        except Exception:
            job[key] = int(default)

    job["last_error"] = job.get("last_error")
    social_posts = job.get("social_posts")
    job["social_posts"] = social_posts[-300:] if isinstance(social_posts, list) else []
    return job

def _recover_job_from_artifacts(job_id: str) -> Optional[Dict[str, Any]]:
    output_dir = os.path.join(OUTPUT_DIR, job_id)
    if not os.path.isdir(output_dir):
        return None

    metadata_path = _metadata_path_for_job(job_id)
    metadata = {}
    result_payload: Optional[Dict[str, Any]] = None
    status = "processing"

    if metadata_path:
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            if isinstance(metadata, dict):
                result_payload = _materialize_result_from_metadata(job_id, metadata, metadata_path)
                clips = result_payload.get("clips", []) if isinstance(result_payload, dict) else []
                if isinstance(clips, list) and len(clips) > 0:
                    status = "completed"
        except Exception:
            metadata = {}

    upload_candidates = sorted(glob.glob(os.path.join(UPLOAD_DIR, f"{job_id}_*")), key=lambda p: os.path.getmtime(p), reverse=True)
    input_path = upload_candidates[0] if upload_candidates else None

    job: Dict[str, Any] = {
        "status": status,
        "logs": [f"Recovered from artifacts ({job_id})."],
        "output_dir": output_dir,
        "input_path": input_path,
        "attempt_count": 0,
        "auto_retry_count": 0,
        "manual_retry_count": 0,
        "max_auto_retries": MAX_AUTO_RETRIES_DEFAULT,
        "retry_delay_seconds": JOB_RETRY_DELAY_SECONDS_DEFAULT,
        "last_error": None,
        "social_posts": metadata.get("social_posts", []) if isinstance(metadata.get("social_posts"), list) else []
    }
    if isinstance(result_payload, dict):
        job["result"] = result_payload
    return _normalize_recovered_job(job_id, job, source="artifacts")

def _ensure_job_context(job_id: str) -> Optional[Dict[str, Any]]:
    existing = jobs.get(job_id)
    if isinstance(existing, dict):
        return existing

    from_store = _load_persisted_job(job_id)
    if isinstance(from_store, dict):
        recovered = _normalize_recovered_job(job_id, from_store, source="sqlite")
        if "result" not in recovered:
            artifact = _recover_job_from_artifacts(job_id)
            if isinstance(artifact, dict) and isinstance(artifact.get("result"), dict):
                recovered["result"] = artifact["result"]
                recovered["status"] = artifact.get("status", recovered.get("status", "completed"))
        jobs[job_id] = recovered
        _persist_job_state(job_id)
        return recovered

    artifact = _recover_job_from_artifacts(job_id)
    if isinstance(artifact, dict):
        jobs[job_id] = artifact
        _persist_job_state(job_id)
        return artifact
    return None

async def _restore_runtime_jobs_from_store():
    persisted = _load_all_persisted_jobs(limit=600)

    for job_id, payload in persisted.items():
        if not isinstance(payload, dict):
            continue
        if job_id in jobs:
            continue
        jobs[job_id] = _normalize_recovered_job(job_id, payload, source="sqlite (startup)")

    # Recover completed jobs that may exist only as artifacts.
    try:
        for entry in os.listdir(OUTPUT_DIR):
            candidate = str(entry).strip()
            if not candidate or candidate in jobs:
                continue
            recovered = _recover_job_from_artifacts(candidate)
            if isinstance(recovered, dict):
                jobs[candidate] = recovered
    except Exception:
        pass

    for job_id in list(jobs.keys()):
        job = jobs.get(job_id)
        if not isinstance(job, dict):
            continue
        status = str(job.get("status", ""))
        if status in {"queued", "retrying", "processing"}:
            cmd = job.get("cmd")
            env = job.get("env")
            if isinstance(cmd, list) and len(cmd) > 0 and isinstance(env, dict):
                if status != "queued":
                    job["status"] = "queued"
                    job.setdefault("logs", []).append("Recovered after restart; job re-queued.")
                await job_queue.put(job_id)
            else:
                job["status"] = "failed"
                job.setdefault("logs", []).append(
                    "Recovered after restart, but command context was incomplete."
                )
                if not job.get("last_error"):
                    job["last_error"] = "Lost active process after restart"
        _persist_job_state(job_id)

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
                        _delete_persisted_job(job_id)
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
        job = _ensure_job_context(job_id)
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
    _init_jobs_store()
    await _restore_runtime_jobs_from_store()
    # Start worker and cleanup
    worker_task = asyncio.create_task(process_queue())
    cleanup_task = asyncio.create_task(cleanup_jobs())
    try:
        yield
    finally:
        worker_task.cancel()
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await worker_task
        with suppress(asyncio.CancelledError):
            await cleanup_task

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
    persist_tick = 0
    try:
        for line in iter(out.readline, b''):
            decoded_line = line.decode('utf-8').strip()
            if decoded_line:
                print(f"📝 [Job Output] {decoded_line}")
                if job_id in jobs:
                    jobs[job_id]['logs'].append(decoded_line)
                    persist_tick += 1
                    if persist_tick % 12 == 0:
                        _persist_job_state(job_id)
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
    _persist_job_state(job_id)
    await job_queue.put(job_id)

async def _queue_job_retry(job_id: str, reason: str, trigger: str = "auto", delay_seconds: Optional[int] = None) -> Tuple[bool, str]:
    job = _ensure_job_context(job_id)
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
        _persist_job_state(job_id)
        asyncio.create_task(_delayed_retry_enqueue(job_id, delay, trigger))
    else:
        job['status'] = 'queued'
        job['logs'].append(f"Retry enqueued ({trigger}).")
        _persist_job_state(job_id)
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
    _persist_job_state(job_id)
    print(f"🎬 [run_job] Executing command for {job_id}: {' '.join(cmd)}")

    async def _fail_or_retry(reason: str):
        jobs[job_id]['last_error'] = reason
        retry_ok, _ = await _queue_job_retry(job_id, reason, trigger="auto")
        if retry_ok:
            _persist_job_state(job_id)
            return
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['logs'].append(reason)
        _persist_job_state(job_id)
    
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
        last_partial_clip_count = -1
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
                             if len(ready_clips) != last_partial_clip_count:
                                 last_partial_clip_count = len(ready_clips)
                                 _persist_job_state(job_id)
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
                trailer_url = str(data.get("latest_trailer_url", "") or "").strip() if isinstance(data, dict) else ""
                trailer_name = _safe_input_filename(trailer_url) if trailer_url else ""
                trailer_path = os.path.join(output_dir, trailer_name) if trailer_name else ""

                transcript_data = data.get('transcript') if isinstance(data, dict) else None
                title_variants_changed = False
                social_variants_changed = False
                title_variant_api_key = _normalize_space(
                    (jobs.get(job_id, {}).get("env", {}) or {}).get("GEMINI_API_KEY", "")
                ) or _normalize_space(os.environ.get("GEMINI_API_KEY", ""))
                for i, clip in enumerate(clips):
                     clip = _normalize_clip_payload(clip, i, transcript=transcript_data)
                     _, _, clip_changed = _ensure_clip_title_variant_pool(
                         clip,
                         transcript=transcript_data,
                         api_key=title_variant_api_key,
                         target_size=TITLE_VARIANTS_PER_CLIP
                     )
                     title_variants_changed = bool(title_variants_changed or clip_changed)
                     _, _, social_changed = _ensure_clip_social_variant_pool(
                         clip,
                         transcript=transcript_data,
                         api_key=title_variant_api_key,
                         target_size=SOCIAL_VARIANTS_PER_CLIP
                     )
                     social_variants_changed = bool(social_variants_changed or social_changed)
                     clip_filename = f"{base_name}_clip_{i+1}.mp4"
                     clip_path = os.path.join(output_dir, clip_filename)
                     existing_video_url = str(clip.get("video_url", "") or "").strip()
                     selected_video_url = ""
                     if existing_video_url:
                         existing_name = _safe_input_filename(existing_video_url)
                         if existing_name and os.path.exists(os.path.join(output_dir, existing_name)):
                             selected_video_url = f"/videos/{job_id}/{existing_name}"
                     if not selected_video_url and os.path.exists(clip_path):
                         selected_video_url = f"/videos/{job_id}/{clip_filename}"
                     if not selected_video_url and trailer_name and os.path.exists(trailer_path):
                         selected_video_url = f"/videos/{job_id}/{trailer_name}"
                     if selected_video_url:
                         clip['video_url'] = selected_video_url
                     clip = _repair_trailer_clip_range(clip, output_dir)

                if (title_variants_changed or social_variants_changed) and isinstance(data, dict):
                    try:
                        data["shorts"] = clips
                        with open(target_json, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                    except Exception:
                        pass

                if (not isinstance(clips, list) or len(clips) == 0) and trailer_name and os.path.exists(trailer_path):
                    trailer_duration = _probe_media_duration_seconds(trailer_path)
                    synthetic_clip = _normalize_clip_payload(
                        {
                            "clip_index": 0,
                            "start": 0.0,
                            "end": max(3.0, float(trailer_duration or 0.0)),
                            "video_title_for_youtube_short": "Super Trailer",
                            "video_description_for_tiktok": "Resumen rápido con los mejores momentos.",
                            "video_description_for_instagram": "Resumen rápido con los mejores momentos.",
                            "virality_score": 90,
                            "selection_confidence": 0.9,
                            "video_url": f"/videos/{job_id}/{trailer_name}",
                            "is_trailer": True,
                        },
                        0,
                        transcript=transcript_data
                    )
                    synthetic_clip = _repair_trailer_clip_range(synthetic_clip, output_dir)
                    clips = [synthetic_clip]

                result_payload = {'clips': clips, 'cost_analysis': cost_analysis}
                if trailer_url:
                    result_payload['latest_trailer_url'] = trailer_url
                jobs[job_id]['result'] = result_payload
                _persist_job_state(job_id)
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
    llm_model: Optional[str] = Form(None),
    llm_provider: Optional[str] = Form(None),
    generation_mode: Optional[str] = Form(None),
    build_trailer: Optional[str] = Form(None),
    trailer_fragments_target: Optional[int] = Form(None),
    groq_api_key: Optional[str] = Form(None),
    max_auto_retries: Optional[int] = Form(None),
    retry_delay_seconds: Optional[int] = Form(None)
):
    x_gemini_key = request.headers.get("X-Gemini-Key")
    x_groq_key = request.headers.get("X-Groq-Key")
    
    # Handle JSON body manually for URL payload
    content_type = request.headers.get("content-type", "")
    body: Dict[str, Any] = {}
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
        llm_model = body.get("llm_model") or llm_model
        llm_provider = body.get("llm_provider") or llm_provider
        generation_mode = body.get("generation_mode") or generation_mode
        build_trailer = body.get("build_trailer") if body.get("build_trailer") is not None else build_trailer
        if body.get("trailer_fragments_target") is not None:
            trailer_fragments_target = body.get("trailer_fragments_target")
        groq_api_key = body.get("groq_api_key") or groq_api_key
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
    generation_mode = str(generation_mode or "clips").strip().lower()
    generation_mode = "trailer" if generation_mode in {"trailer", "super_trailer", "super-trailer"} else "clips"
    build_trailer_flag = _parse_form_bool(build_trailer, default=False) or generation_mode == "trailer"
    if trailer_fragments_target is not None:
        try:
            trailer_fragments_target = max(2, min(12, int(trailer_fragments_target)))
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="Invalid trailer_fragments_target. Must be an integer between 2 and 12")
    max_auto_retries = max(0, min(5, int(max_auto_retries if max_auto_retries is not None else MAX_AUTO_RETRIES_DEFAULT)))
    retry_delay_seconds = max(0, min(300, int(retry_delay_seconds if retry_delay_seconds is not None else JOB_RETRY_DELAY_SECONDS_DEFAULT)))

    job_id = str(uuid.uuid4())
    job_output_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_output_dir, exist_ok=True)
    
    # Defaults if not provided in form or json body
    llm_provider = llm_provider or "gemini"
    llm_model = llm_model or "gemini-2.5-flash-lite"

    
    # Prepare Command
    python_bin = sys.executable or shutil.which("python3") or "python3"
    cmd = [python_bin, "-u", "main.py"] 
    env = os.environ.copy()
    
    if llm_provider == 'gemini':
        env["GEMINI_API_KEY"] = x_gemini_key
        if not x_gemini_key:
            raise HTTPException(status_code=400, detail="Missing Gemini API Key (X-Gemini-Key header)")
    elif llm_provider == 'groq':
        final_groq_key = x_groq_key or (body.get("groq_api_key") if "application/json" in content_type else None)
        env["GROQ_API_KEY"] = final_groq_key
        # Allow main.py to fallback to Gemini automatically when Groq hits hard limits.
        if x_gemini_key:
            env["GEMINI_API_KEY"] = x_gemini_key
        if not final_groq_key:
            raise HTTPException(status_code=400, detail="Missing Groq API Key (X-Groq-Key header or payload)")
        cmd.extend(["--llm-provider", "groq", "--groq-api-key", final_groq_key])
    
    if llm_model:
        cmd.extend(["--llm-model", str(llm_model)])
    if build_trailer_flag:
        cmd.append("--build-trailer")
    if generation_mode == "trailer":
        cmd.append("--trailer-only")
    if trailer_fragments_target is not None:
        cmd.extend(["--trailer-fragments-target", str(trailer_fragments_target)])
    
    if url:
        cmd.extend(["-u", url])
        cmd.append("--keep-original")
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
    _persist_job_state(job_id)
    
    await job_queue.put(job_id)
    
    return {"job_id": job_id, "status": "queued"}

@app.get("/api/status/__healthcheck__")
async def status_healthcheck():
    return {
        "ok": True,
        "timestamp": int(time.time()),
        "jobs_in_memory": len(jobs)
    }

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    job = _ensure_job_context(job_id)
    if not isinstance(job, dict):
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "status": job['status'],
        "logs": job['logs'],
        "result": job.get('result'),
        "updated_at": int(_safe_float(job.get('updated_at', 0), 0)),
        "retry": {
            "attempt_count": int(job.get('attempt_count', 0)),
            "auto_retry_count": int(job.get('auto_retry_count', 0)),
            "manual_retry_count": int(job.get('manual_retry_count', 0)),
            "max_auto_retries": int(job.get('max_auto_retries', MAX_AUTO_RETRIES_DEFAULT)),
            "retry_delay_seconds": int(job.get('retry_delay_seconds', JOB_RETRY_DELAY_SECONDS_DEFAULT)),
            "last_error": job.get('last_error')
        }
    }

@app.get("/api/jobs/recent")
async def list_recent_jobs(limit: int = 20):
    safe_limit = max(1, min(100, int(limit or 20)))
    items: List[Dict[str, Any]] = []
    for job_id, job in list(jobs.items()):
        if not isinstance(job, dict):
            continue
        result = job.get("result", {}) if isinstance(job.get("result"), dict) else {}
        clips = result.get("clips", []) if isinstance(result, dict) else []
        highlight_reels = result.get("highlight_reels", []) if isinstance(result, dict) else []
        items.append({
            "job_id": job_id,
            "status": str(job.get("status", "unknown")),
            "clips_count": len(clips) if isinstance(clips, list) else 0,
            "has_highlight_reel": bool(isinstance(highlight_reels, list) and len(highlight_reels) > 0),
            "last_error": job.get("last_error"),
            "updated_at": int(_safe_float(job.get("updated_at", 0), 0))
        })
    items = sorted(items, key=lambda row: row.get("updated_at", 0), reverse=True)
    return {"items": items[:safe_limit]}

@app.delete("/api/jobs/all")
async def delete_all_jobs(
    include_artifacts: bool = True,
    include_uploads: bool = True,
):
    persisted = _load_all_persisted_jobs(limit=4000)
    candidate_ids: Set[str] = set(jobs.keys()) | set(persisted.keys()) | _discover_job_ids_on_disk()

    removed_ids: List[str] = []
    skipped_active: List[str] = []
    removed_artifact_paths_total = 0
    removed_upload_paths_total = 0

    for job_id in sorted(candidate_ids):
        job = jobs.get(job_id)
        status = str(job.get("status", "")).lower() if isinstance(job, dict) else ""
        is_active = status in {"queued", "retrying", "processing"}
        if is_active:
            skipped_active.append(job_id)
            continue

        if job_id in jobs:
            del jobs[job_id]
        _delete_persisted_job(job_id)
        SEARCH_INDEX_CACHE.pop(job_id, None)

        cleanup = _purge_job_artifacts(
            job_id=job_id,
            include_artifacts=include_artifacts,
            include_uploads=include_uploads,
        )
        removed_artifact_paths_total += int(cleanup.get("removed_artifact_paths", 0))
        removed_upload_paths_total += int(cleanup.get("removed_upload_paths", 0))
        removed_ids.append(job_id)

    return {
        "success": True,
        "removed_jobs": len(removed_ids),
        "removed_ids": removed_ids,
        "skipped_active": len(skipped_active),
        "skipped_active_ids": skipped_active,
        "removed_artifact_paths": removed_artifact_paths_total,
        "removed_upload_paths": removed_upload_paths_total,
    }

@app.post("/api/retry/{job_id}")
async def retry_job(job_id: str):
    job = _ensure_job_context(job_id)
    if not isinstance(job, dict):
        raise HTTPException(status_code=404, detail="Job not found")
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
from subtitles import generate_srt, burn_subtitles, generate_karaoke_ass_from_srt, generate_styled_ass_from_srt, _sanitize_font_name

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
    final_api_key = req.api_key or x_gemini_key
    if not final_api_key:
        raise HTTPException(status_code=400, detail="Gemini API Key is required. Please set it in the configuration.")
    
    if not final_api_key:
        raise HTTPException(status_code=400, detail="Missing Gemini API Key (Header or Body)")

    job = _ensure_job_context(req.job_id)
    if not isinstance(job, dict):
        raise HTTPException(status_code=404, detail="Job not found")
    if 'result' not in job or 'clips' not in job['result']:
        raise HTTPException(status_code=400, detail="Job result not available")
        
    try:
        # Resolve Input Path: Prefer explict input_filename from frontend (chaining edits)
        if req.input_filename:
            # Security: Ensure just a filename, no paths
            safe_name = _safe_input_filename(req.input_filename)
            input_path = os.path.join(OUTPUT_DIR, req.job_id, safe_name)
            filename = safe_name
        else:
            # Fallback to original clip
            clip = job['result']['clips'][req.clip_index]
            filename = _safe_input_filename(clip.get('video_url', ''))
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
    font_size: int = 40
    font_family: str = "Anton"
    font_color: str = "#FFFFFF"
    stroke_color: str = "#000000"
    stroke_width: int = 5
    bold: bool = True
    box_color: str = "#000000"
    box_opacity: int = 0
    karaoke_mode: bool = False
    caption_offset_x: float = 0.0  # -100..100
    caption_offset_y: float = 0.0  # -100..100
    srt_content: Optional[str] = None
    input_filename: Optional[str] = None

@app.post("/api/subtitle")
async def add_subtitles(req: SubtitleRequest):
    job = _ensure_job_context(req.job_id)
    if not isinstance(job, dict):
        raise HTTPException(status_code=404, detail="Job not found")

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
        
    raw_clip_data = clips[req.clip_index]
    clip_data = _repair_trailer_clip_range(
        dict(raw_clip_data) if isinstance(raw_clip_data, dict) else {},
        output_dir
    )
    if isinstance(raw_clip_data, dict):
        for key in ("start", "end", "duration", "transcript_timebase"):
            if key in clip_data:
                raw_clip_data[key] = clip_data[key]
    effective_transcript, srt_clip_start, srt_clip_end = _resolve_clip_scoped_transcript_for_srt(
        clip_data=clip_data,
        fallback_transcript=transcript
    )
    
    # Video Path
    if req.input_filename:
        # Use chained file
        filename = _safe_input_filename(req.input_filename)
        filename = _resolve_subtitle_source_filename(output_dir, filename)
    else:
        # Fallback to standard naming
        filename = _safe_input_filename(clip_data.get('video_url', ''))
        if filename:
            filename = _resolve_subtitle_source_filename(output_dir, filename)
        if not filename:
             base_name = os.path.basename(json_files[0]).replace('_metadata.json', '')
             filename = f"{base_name}_clip_{req.clip_index+1}.mp4"
         
    input_path = os.path.join(output_dir, filename)
    if not os.path.exists(input_path):
        # Try looking for edited version if url implied it?
        # Just fail if not found.
        raise HTTPException(status_code=404, detail=f"Video file not found: {input_path}")
        
    caption_offset_x = max(-100.0, min(100.0, _safe_float(req.caption_offset_x, 0.0)))
    caption_offset_y = max(-100.0, min(100.0, _safe_float(req.caption_offset_y, 0.0)))
    resolved_font_family = _sanitize_font_name(req.font_family)

    # Define outputs
    subtitle_ext = "ass"
    subtitle_filename = f"subs_{req.clip_index}_{int(time.time())}.{subtitle_ext}"
    subtitle_path = os.path.join(output_dir, subtitle_filename)
    
    # Output video
    # We create a new file "subtitled_..."
    base_name = os.path.splitext(filename)[0]
    output_nonce = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
    output_filename = f"subtitled_{base_name}_{output_nonce}.mp4"
    output_path = os.path.join(output_dir, output_filename)
    
    try:
        # 1. Generate or use provided SRT
        if req.srt_content:
            raw_srt_content = req.srt_content
        else:
            temp_srt_name = f"subs_tmp_{req.clip_index}_{int(time.time())}.srt"
            temp_srt_path = os.path.join(output_dir, temp_srt_name)
            try:
                success = generate_srt(effective_transcript, srt_clip_start, srt_clip_end, temp_srt_path)
                if (not success) and (effective_transcript is not transcript):
                    success = generate_srt(
                        transcript,
                        max(0.0, _safe_float(clip_data.get('start', 0.0), 0.0)),
                        max(
                            max(0.0, _safe_float(clip_data.get('start', 0.0), 0.0)),
                            _safe_float(clip_data.get('end', 0.0), 0.0)
                        ),
                        temp_srt_path
                    )
                if not success:
                    raise HTTPException(status_code=400, detail="No words found for this clip range.")
                with open(temp_srt_path, "r", encoding="utf-8") as f:
                    raw_srt_content = f.read()
            finally:
                if os.path.exists(temp_srt_path):
                    os.remove(temp_srt_path)

        if req.karaoke_mode:
            success = generate_karaoke_ass_from_srt(
                raw_srt_content,
                subtitle_path,
                alignment=req.position,
                font_size=req.font_size,
                font_name=resolved_font_family,
                font_color=req.font_color,
                active_word_color="auto",
                stroke_color=req.stroke_color,
                stroke_width=req.stroke_width,
                bold=req.bold,
                box_color=req.box_color,
                box_opacity=req.box_opacity,
                pop_scale=118,
                offset_x=caption_offset_x,
                offset_y=caption_offset_y
            )
            if not success:
                raise HTTPException(status_code=400, detail="No se pudo generar karaoke para este clip.")
        else:
            success = generate_styled_ass_from_srt(
                raw_srt_content,
                subtitle_path,
                alignment=req.position,
                font_size=req.font_size,
                font_name=resolved_font_family,
                font_color=req.font_color,
                stroke_color=req.stroke_color,
                stroke_width=req.stroke_width,
                bold=req.bold,
                box_color=req.box_color,
                box_opacity=req.box_opacity,
                offset_x=caption_offset_x,
                offset_y=caption_offset_y
            )
            if not success:
                raise HTTPException(status_code=400, detail="No se pudo generar subtítulos para este clip.")
             
        # 2. Burn Subtitles
        # Run in thread pool
        def run_burn():
             burn_subtitles(
                 input_path,
                 subtitle_path,
                 output_path,
                 alignment=req.position,
                 fontsize=req.font_size,
                 font_name=resolved_font_family,
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
        
        try:
            clips[req.clip_index]['video_url'] = new_video_url
            clips[req.clip_index]['caption_position'] = req.position
            clips[req.clip_index]['caption_offset_x'] = round(caption_offset_x, 3)
            clips[req.clip_index]['caption_offset_y'] = round(caption_offset_y, 3)
            clips[req.clip_index]['caption_font_size'] = int(req.font_size)
            clips[req.clip_index]['caption_font_family'] = str(resolved_font_family or "Anton")
            clips[req.clip_index]['caption_font_color'] = str(req.font_color or "#FFFFFF")
            clips[req.clip_index]['caption_stroke_color'] = str(req.stroke_color or "#000000")
            clips[req.clip_index]['caption_stroke_width'] = int(req.stroke_width)
            clips[req.clip_index]['caption_bold'] = bool(req.bold)
            clips[req.clip_index]['caption_box_color'] = str(req.box_color or "#000000")
            clips[req.clip_index]['caption_box_opacity'] = int(req.box_opacity)
            clips[req.clip_index]['caption_karaoke_mode'] = bool(req.karaoke_mode)
            with open(json_files[0], 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        if 'result' in job and 'clips' in job['result'] and req.clip_index < len(job['result']['clips']):
            job['result']['clips'][req.clip_index]['video_url'] = new_video_url
            job['result']['clips'][req.clip_index]['caption_position'] = req.position
            job['result']['clips'][req.clip_index]['caption_offset_x'] = round(caption_offset_x, 3)
            job['result']['clips'][req.clip_index]['caption_offset_y'] = round(caption_offset_y, 3)
            job['result']['clips'][req.clip_index]['caption_font_size'] = int(req.font_size)
            job['result']['clips'][req.clip_index]['caption_font_family'] = str(resolved_font_family or "Anton")
            job['result']['clips'][req.clip_index]['caption_font_color'] = str(req.font_color or "#FFFFFF")
            job['result']['clips'][req.clip_index]['caption_stroke_color'] = str(req.stroke_color or "#000000")
            job['result']['clips'][req.clip_index]['caption_stroke_width'] = int(req.stroke_width)
            job['result']['clips'][req.clip_index]['caption_bold'] = bool(req.bold)
            job['result']['clips'][req.clip_index]['caption_box_color'] = str(req.box_color or "#000000")
            job['result']['clips'][req.clip_index]['caption_box_opacity'] = int(req.box_opacity)
            job['result']['clips'][req.clip_index]['caption_karaoke_mode'] = bool(req.karaoke_mode)
            _persist_job_state(req.job_id)

        return {
            "success": True,
            "new_video_url": new_video_url,
            "caption_position": req.position,
            "caption_offset_x": round(caption_offset_x, 3),
            "caption_offset_y": round(caption_offset_y, 3),
            "caption_font_size": int(req.font_size),
            "caption_font_family": str(resolved_font_family or "Anton"),
            "caption_font_color": str(req.font_color or "#FFFFFF"),
            "caption_stroke_color": str(req.stroke_color or "#000000"),
            "caption_stroke_width": int(req.stroke_width),
            "caption_bold": bool(req.bold),
            "caption_box_color": str(req.box_color or "#000000"),
            "caption_box_opacity": int(req.box_opacity),
            "caption_karaoke_mode": bool(req.karaoke_mode)
        }
        
    except Exception as e:
        print(f"❌ Subtitle Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class SubtitlePreviewRequest(BaseModel):
    job_id: str
    clip_index: int

@app.post("/api/subtitle/preview")
async def preview_subtitles(req: SubtitlePreviewRequest):
    job = _ensure_job_context(req.job_id)
    if not isinstance(job, dict):
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

    raw_clip_data = clips[req.clip_index]
    clip_data = _repair_trailer_clip_range(
        dict(raw_clip_data) if isinstance(raw_clip_data, dict) else {},
        output_dir
    )
    if isinstance(raw_clip_data, dict):
        for key in ("start", "end", "duration", "transcript_timebase"):
            if key in clip_data:
                raw_clip_data[key] = clip_data[key]
    effective_transcript, srt_clip_start, srt_clip_end = _resolve_clip_scoped_transcript_for_srt(
        clip_data=clip_data,
        fallback_transcript=transcript
    )
    temp_name = f"subs_preview_{req.clip_index}_{int(time.time())}.srt"
    temp_path = os.path.join(output_dir, temp_name)
    success = generate_srt(effective_transcript, srt_clip_start, srt_clip_end, temp_path)
    if (not success) and (effective_transcript is not transcript):
        success = generate_srt(
            transcript,
            max(0.0, _safe_float(clip_data.get('start', 0.0), 0.0)),
            max(
                max(0.0, _safe_float(clip_data.get('start', 0.0), 0.0)),
                _safe_float(clip_data.get('end', 0.0), 0.0)
            ),
            temp_path
        )
    if not success:
        raise HTTPException(status_code=400, detail="No words found for this clip range.")

    with open(temp_path, 'r', encoding='utf-8') as f:
        content = f.read()
    try:
        os.remove(temp_path)
    except Exception:
        pass

    return {"srt": content}

class FastPreviewRequest(BaseModel):
    job_id: str
    clip_index: int
    input_filename: Optional[str] = None
    start: Optional[float] = None
    duration: float = 3.0
    aspect_ratio: Optional[str] = None
    fit_mode: Optional[str] = None
    zoom: Optional[float] = None
    offset_x: Optional[float] = None
    offset_y: Optional[float] = None
    captions_on: bool = False
    caption_position: Optional[str] = None
    caption_font_size: Optional[int] = None
    caption_font_family: Optional[str] = None
    caption_font_color: Optional[str] = None
    caption_stroke_color: Optional[str] = None
    caption_stroke_width: Optional[int] = None
    caption_bold: Optional[bool] = None
    caption_box_color: Optional[str] = None
    caption_box_opacity: Optional[int] = None
    caption_karaoke_mode: Optional[bool] = None
    caption_offset_x: Optional[float] = None
    caption_offset_y: Optional[float] = None
    srt_content: Optional[str] = None

@app.post("/api/clip/fast-preview")
async def generate_fast_preview(req: FastPreviewRequest):
    job = _ensure_job_context(req.job_id)
    if not isinstance(job, dict):
        raise HTTPException(status_code=404, detail="Job not found")

    output_dir = os.path.join(OUTPUT_DIR, req.job_id)
    if not os.path.isdir(output_dir):
        raise HTTPException(status_code=404, detail="Job output directory not found")

    metadata_path = _metadata_path_for_job(req.job_id)
    if not metadata_path:
        raise HTTPException(status_code=404, detail="Metadata not found")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    clips = metadata.get("shorts", []) if isinstance(metadata.get("shorts"), list) else []
    transcript_data = metadata.get("transcript") if isinstance(metadata, dict) else None
    if req.clip_index < 0 or req.clip_index >= len(clips):
        raise HTTPException(status_code=404, detail="Clip not found")
    clip_data = clips[req.clip_index] if isinstance(clips[req.clip_index], dict) else {}

    source_candidates: List[Tuple[str, str]] = []
    seen_sources: Set[str] = set()

    def _add_source_candidate(path_value: str, reason: str):
        p = str(path_value or "").strip()
        if not p:
            return
        try:
            rp = os.path.realpath(p)
        except Exception:
            rp = p
        if rp in seen_sources:
            return
        if not os.path.exists(rp):
            return
        seen_sources.add(rp)
        source_candidates.append((rp, reason))

    if req.input_filename:
        source_name = _safe_input_filename(req.input_filename)
        if source_name:
            _add_source_candidate(os.path.join(output_dir, source_name), "input_filename")

    job_input_path = str(job.get("input_path") or "").strip()
    if job_input_path:
        _add_source_candidate(job_input_path, "job_input_path")

    source_name = _safe_input_filename(clip_data.get("video_url", ""))
    if not source_name:
        base_name = os.path.basename(metadata_path).replace("_metadata.json", "")
        source_name = f"{base_name}_clip_{req.clip_index + 1}.mp4"
    
    # Prioritize uncut version for layout flexibility
    uncut_name = source_name.replace(".mp4", "_uncut.mp4")
    _add_source_candidate(os.path.join(output_dir, uncut_name), "clip_uncut")
    _add_source_candidate(os.path.join(output_dir, source_name), "clip_video_url")

    if not source_candidates:
        raise HTTPException(status_code=404, detail="Video source not found for fast preview")

    clip_start = max(0.0, _safe_float(clip_data.get("start", 0.0), 0.0))
    requested_start_raw = max(0.0, _safe_float(req.start, clip_start))
    duration_target = max(0.6, min(8.0, _safe_float(req.duration, 3.0)))
    aspect_ratio = normalize_aspect_ratio(req.aspect_ratio, default=str(clip_data.get("aspect_ratio", "9:16"))) or "9:16"
    fit_mode = _normalize_layout_fit_mode(req.fit_mode or clip_data.get("layout_fit_mode", "cover"))
    zoom = _coerce_layout_zoom(req.zoom if req.zoom is not None else clip_data.get("layout_zoom"), 1.0, fit_mode)
    offset_x = _coerce_layout_offset(req.offset_x if req.offset_x is not None else clip_data.get("layout_offset_x"), 0.0)
    offset_y = _coerce_layout_offset(req.offset_y if req.offset_y is not None else clip_data.get("layout_offset_y"), 0.0)

    last_error = "Failed to render fast preview"

    for source_path, source_reason in source_candidates:
        in_w, in_h = _probe_video_dimensions(source_path)
        if in_w <= 0 or in_h <= 0:
            continue
            
        vf_filter, target_w, target_h = _build_manual_layout_filter(
            in_w=in_w,
            in_h=in_h,
            aspect_ratio=aspect_ratio,
            fit_mode=fit_mode,
            zoom=zoom,
            offset_x=offset_x,
            offset_y=offset_y
        )
        source_duration = _probe_media_duration_seconds(source_path)
        requested_start = requested_start_raw
        if source_duration > 0:
            max_start = max(0.0, source_duration - 0.25)
            if requested_start > max_start:
                # If candidate is already a clipped file, restart from 0.
                if source_reason in {"clip_video_url", "input_filename"}:
                    requested_start = 0.0
                else:
                    requested_start = max_start
            remaining = max(0.0, source_duration - requested_start)
            if remaining > 0:
                duration = max(0.6, min(duration_target, remaining))
            else:
                duration = duration_target
        else:
            duration = duration_target

        out_name = f"fast_preview_clip_{req.clip_index+1}_{int(time.time())}_{uuid.uuid4().hex[:6]}.mp4"
        out_path = os.path.join(output_dir, out_name)
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{requested_start:.3f}",
            "-t", f"{duration:.3f}",
            "-i", source_path,
            "-filter_complex", vf_filter,
            "-map", "[out_v]",
            "-map", "0:a?",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "24",
            "-preset", "veryfast",
            "-c:a", "aac",
            "-movflags", "+faststart",
            out_path
        ]
        run = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if run.returncode != 0 or not os.path.exists(out_path):
            last_error = run.stderr.decode("utf-8", errors="ignore") or f"Failed rendering from {source_reason}"
            try:
                if os.path.exists(out_path):
                    os.remove(out_path)
            except Exception:
                pass
            continue

        out_w, out_h = _probe_video_dimensions(out_path)
        out_duration = _probe_media_duration_seconds(out_path)
        if out_w <= 0 or out_h <= 0 or out_duration < 0.2:
            last_error = f"Invalid fast preview output from {source_reason} ({out_w}x{out_h}, {out_duration:.3f}s)"
            try:
                os.remove(out_path)
            except Exception:
                pass
            continue

        final_out_path = out_path
        final_out_name = out_name
        captions_burned = False

        if bool(req.captions_on):
            raw_srt = str(req.srt_content or "").strip()
            if not raw_srt:
                effective_transcript, srt_clip_start, srt_clip_end = _resolve_clip_scoped_transcript_for_srt(
                    clip_data=clip_data,
                    fallback_transcript=transcript_data or {}
                )
                temp_srt_name = f"fast_preview_srt_{req.clip_index}_{int(time.time())}_{uuid.uuid4().hex[:5]}.srt"
                temp_srt_path = os.path.join(output_dir, temp_srt_name)
                try:
                    success = generate_srt(effective_transcript, srt_clip_start, srt_clip_end, temp_srt_path)
                    if (not success) and (effective_transcript is not transcript_data) and isinstance(transcript_data, dict):
                        success = generate_srt(
                            transcript_data,
                            max(0.0, _safe_float(clip_data.get("start", 0.0), 0.0)),
                            max(
                                max(0.0, _safe_float(clip_data.get("start", 0.0), 0.0)),
                                _safe_float(clip_data.get("end", 0.0), 0.0)
                            ),
                            temp_srt_path
                        )
                    if not success:
                        raise HTTPException(status_code=400, detail="No words found for this preview range.")
                    with open(temp_srt_path, "r", encoding="utf-8") as f:
                        raw_srt = f.read()
                finally:
                    if os.path.exists(temp_srt_path):
                        try:
                            os.remove(temp_srt_path)
                        except Exception:
                            pass

            shift_candidates = [-requested_start]
            if source_reason == "job_input_path":
                shift_candidates.insert(0, clip_start - requested_start)
            elif source_reason == "clip_uncut":
                shift_candidates.append(clip_start - requested_start)

            shifted_srt = ""
            seen_shift: Set[float] = set()
            for shift in shift_candidates:
                rounded_shift = round(float(shift), 6)
                if rounded_shift in seen_shift:
                    continue
                seen_shift.add(rounded_shift)
                shifted_srt = _shift_and_trim_srt(raw_srt, shift_seconds=rounded_shift, window_duration=out_duration)
                if shifted_srt:
                    break

            if not shifted_srt:
                raise HTTPException(status_code=400, detail="No words found for this preview range.")

            subtitle_filename = f"fast_preview_subs_{req.clip_index}_{int(time.time())}_{uuid.uuid4().hex[:6]}.ass"
            subtitle_path = os.path.join(output_dir, subtitle_filename)
            subtitled_name = f"fast_preview_subtitled_clip_{req.clip_index+1}_{int(time.time())}_{uuid.uuid4().hex[:6]}.mp4"
            subtitled_path = os.path.join(output_dir, subtitled_name)

            caption_position = str(
                req.caption_position
                or clip_data.get("caption_position")
                or "bottom"
            ).strip().lower()
            if caption_position not in {"top", "middle", "bottom"}:
                caption_position = "bottom"

            caption_font_size = int(max(12, min(84, _safe_float(
                req.caption_font_size if req.caption_font_size is not None else clip_data.get("caption_font_size"),
                40.0
            ))))
            caption_font_family = _sanitize_font_name(
                req.caption_font_family if req.caption_font_family is not None else clip_data.get("caption_font_family")
            )
            caption_font_color = str(req.caption_font_color or clip_data.get("caption_font_color") or "#FFFFFF")
            caption_stroke_color = str(req.caption_stroke_color or clip_data.get("caption_stroke_color") or "#000000")
            caption_stroke_width = int(max(0, min(8, _safe_float(
                req.caption_stroke_width if req.caption_stroke_width is not None else clip_data.get("caption_stroke_width"),
                3.0
            ))))
            caption_bold = bool(req.caption_bold if req.caption_bold is not None else clip_data.get("caption_bold", True))
            caption_box_color = str(req.caption_box_color or clip_data.get("caption_box_color") or "#000000")
            caption_box_opacity = int(max(0, min(100, _safe_float(
                req.caption_box_opacity if req.caption_box_opacity is not None else clip_data.get("caption_box_opacity"),
                0.0
            ))))
            caption_karaoke_mode = bool(
                req.caption_karaoke_mode if req.caption_karaoke_mode is not None else clip_data.get("caption_karaoke_mode", False)
            )
            caption_offset_x = max(-100.0, min(100.0, _safe_float(
                req.caption_offset_x if req.caption_offset_x is not None else clip_data.get("caption_offset_x"),
                0.0
            )))
            caption_offset_y = max(-100.0, min(100.0, _safe_float(
                req.caption_offset_y if req.caption_offset_y is not None else clip_data.get("caption_offset_y"),
                0.0
            )))

            try:
                if caption_karaoke_mode:
                    ok_ass = generate_karaoke_ass_from_srt(
                        shifted_srt,
                        subtitle_path,
                        alignment=caption_position,
                        font_size=caption_font_size,
                        font_name=caption_font_family,
                        font_color=caption_font_color,
                        active_word_color="auto",
                        stroke_color=caption_stroke_color,
                        stroke_width=caption_stroke_width,
                        bold=caption_bold,
                        box_color=caption_box_color,
                        box_opacity=caption_box_opacity,
                        pop_scale=118,
                        offset_x=caption_offset_x,
                        offset_y=caption_offset_y
                    )
                else:
                    ok_ass = generate_styled_ass_from_srt(
                        shifted_srt,
                        subtitle_path,
                        alignment=caption_position,
                        font_size=caption_font_size,
                        font_name=caption_font_family,
                        font_color=caption_font_color,
                        stroke_color=caption_stroke_color,
                        stroke_width=caption_stroke_width,
                        bold=caption_bold,
                        box_color=caption_box_color,
                        box_opacity=caption_box_opacity,
                        offset_x=caption_offset_x,
                        offset_y=caption_offset_y
                    )
                if not ok_ass:
                    raise HTTPException(status_code=400, detail="No se pudo generar subtítulos para el preview rápido.")

                burn_subtitles(
                    out_path,
                    subtitle_path,
                    subtitled_path,
                    alignment=caption_position,
                    fontsize=caption_font_size,
                    font_name=caption_font_family,
                    font_color=caption_font_color,
                    stroke_color=caption_stroke_color,
                    stroke_width=caption_stroke_width,
                    bold=caption_bold,
                    box_color=caption_box_color,
                    box_opacity=caption_box_opacity
                )

                if not os.path.exists(subtitled_path):
                    raise HTTPException(status_code=500, detail="No se pudo renderizar subtítulos en preview rápido.")

                try:
                    os.remove(out_path)
                except Exception:
                    pass

                final_out_path = subtitled_path
                final_out_name = subtitled_name
                captions_burned = True
            finally:
                if os.path.exists(subtitle_path):
                    try:
                        os.remove(subtitle_path)
                    except Exception:
                        pass

        final_w, final_h = _probe_video_dimensions(final_out_path)
        final_duration = _probe_media_duration_seconds(final_out_path)
        if final_w <= 0 or final_h <= 0 or final_duration < 0.2:
            last_error = f"Invalid fast preview output from {source_reason} ({final_w}x{final_h}, {final_duration:.3f}s)"
            try:
                if os.path.exists(final_out_path):
                    os.remove(final_out_path)
            except Exception:
                pass
            continue

        return {
            "success": True,
            "clip_index": req.clip_index,
            "aspect_ratio": aspect_ratio,
            "start": round(requested_start, 3),
            "duration": round(final_duration, 3),
            "source_reason": source_reason,
            "preview_video_url": f"/videos/{req.job_id}/{final_out_name}",
            "captions_burned": bool(captions_burned)
        }

    raise HTTPException(status_code=500, detail=last_error or "Failed to render fast preview")

class RetitleRequest(BaseModel):
    job_id: str
    clip_index: int
    current_title: Optional[str] = None

@app.post("/api/clip/retitle")
async def retitle_clip(
    req: RetitleRequest,
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key")
):
    job = _ensure_job_context(req.job_id)
    job_env = job.get("env", {}) if isinstance(job, dict) else {}
    final_api_key = _normalize_space(
        x_gemini_key
        or job_env.get("GEMINI_API_KEY", "")
        or os.environ.get("GEMINI_API_KEY", "")
    ) or None

    output_dir = os.path.join(OUTPUT_DIR, req.job_id)
    metadata_path: Optional[str] = None
    metadata_data: Dict[str, Any] = {}
    metadata_clips: List[Dict[str, Any]] = []
    clip_payload: Dict[str, Any] = {}
    transcript_data: Optional[Dict[str, Any]] = None
    context_source = "request_only"

    if os.path.isdir(output_dir):
        json_files = sorted(glob.glob(os.path.join(output_dir, "*_metadata.json")))
        if json_files:
            metadata_path = json_files[0]
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata_data = json.load(f)
                if isinstance(metadata_data.get("shorts"), list):
                    metadata_clips = metadata_data.get("shorts") or []
                transcript_data = metadata_data.get("transcript") if isinstance(metadata_data, dict) else None
            except Exception:
                metadata_path = None
                metadata_data = {}
                metadata_clips = []
                transcript_data = None

    if req.clip_index >= 0 and req.clip_index < len(metadata_clips):
        maybe_clip = metadata_clips[req.clip_index]
        if isinstance(maybe_clip, dict):
            clip_payload = maybe_clip
            context_source = "metadata"

    if not clip_payload and isinstance(job, dict) and isinstance(job.get("result"), dict):
        result_clips = job["result"].get("clips", [])
        if isinstance(result_clips, list) and req.clip_index >= 0 and req.clip_index < len(result_clips):
            maybe_clip = result_clips[req.clip_index]
            if isinstance(maybe_clip, dict):
                clip_payload = maybe_clip
                context_source = "runtime_job"

    current_title = _sanitize_short_title(
        req.current_title
        or clip_payload.get("video_title_for_youtube_short")
        or clip_payload.get("title")
        or "Momento clave del video"
    ) or "Momento clave del video"

    variants, active_index, _ = _ensure_clip_title_variant_pool(
        clip_payload,
        transcript=transcript_data,
        api_key=final_api_key,
        target_size=TITLE_VARIANTS_PER_CLIP
    )
    if not variants:
        variants = [current_title]
        active_index = 0

    current_fp = _title_fingerprint(current_title)
    if current_fp:
        current_match = next((i for i, title in enumerate(variants) if _title_fingerprint(title) == current_fp), None)
        if current_match is not None:
            active_index = current_match
        else:
            variants = _dedupe_title_candidates([current_title] + variants)
            active_index = 0

    next_index = (active_index + 1) % max(1, len(variants))

    new_title = _sanitize_short_title(variants[next_index] if variants else current_title) or current_title
    if not new_title:
        raise HTTPException(status_code=500, detail="Could not generate a new title.")

    clip_patch = {
        "video_title_for_youtube_short": new_title,
        "title": new_title,
        "title_variants": variants,
        "title_variant_index": int(next_index)
    }

    if metadata_path and metadata_clips and req.clip_index >= 0 and req.clip_index < len(metadata_clips):
        try:
            metadata_clips[req.clip_index].update(clip_patch)
            if isinstance(metadata_data, dict):
                metadata_data["shorts"] = metadata_clips
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata_data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    if isinstance(job, dict) and isinstance(job.get("result"), dict):
        result_clips = job["result"].get("clips", [])
        if isinstance(result_clips, list) and req.clip_index < len(result_clips) and isinstance(result_clips[req.clip_index], dict):
            result_clips[req.clip_index].update(clip_patch)
        _persist_job_state(req.job_id)

    return {
        "success": True,
        "clip_index": req.clip_index,
        "context_source": context_source,
        "previous_title": current_title,
        "new_title": new_title,
        "title_variants": variants,
        "title_variant_index": int(next_index),
        "title_variant_total": len(variants),
        "clip_patch": clip_patch
    }

class ResocialRequest(BaseModel):
    job_id: str
    clip_index: int
    current_social: Optional[str] = None

@app.post("/api/clip/resocial")
async def rewrite_clip_social(
    req: ResocialRequest,
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key")
):
    job = _ensure_job_context(req.job_id)
    output_dir = os.path.join(OUTPUT_DIR, req.job_id)
    if not os.path.isdir(output_dir):
        raise HTTPException(status_code=404, detail="Job output directory not found")

    json_files = sorted(glob.glob(os.path.join(output_dir, "*_metadata.json")))
    if not json_files:
        raise HTTPException(status_code=404, detail="Metadata not found")

    metadata_path = json_files[0]
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read metadata: {e}")

    metadata_clips = data.get("shorts") if isinstance(data.get("shorts"), list) else []
    if req.clip_index < 0 or req.clip_index >= len(metadata_clips):
        raise HTTPException(status_code=404, detail="Clip not found")

    clip_payload = metadata_clips[req.clip_index] if isinstance(metadata_clips[req.clip_index], dict) else {}
    transcript_data = data.get("transcript") if isinstance(data, dict) else None

    current_social = _sanitize_social_copy(
        req.current_social
        or clip_payload.get("video_description_for_tiktok")
        or clip_payload.get("video_description_for_instagram")
        or "",
        max_chars=320
    )
    job_env = job.get("env", {}) if isinstance(job, dict) else {}
    final_api_key = _normalize_space(
        x_gemini_key
        or job_env.get("GEMINI_API_KEY", "")
        or os.environ.get("GEMINI_API_KEY", "")
    ) or None

    variants, active_index, _ = _ensure_clip_social_variant_pool(
        clip_payload,
        transcript=transcript_data,
        api_key=final_api_key,
        target_size=SOCIAL_VARIANTS_PER_CLIP
    )
    if not variants:
        variants = [current_social] if current_social else [_build_fallback_social_copy("", "", "", "", [])]
        active_index = 0

    current_fp = _social_fingerprint(current_social)
    if current_fp:
        current_match = next((i for i, item in enumerate(variants) if _social_fingerprint(item) == current_fp), None)
        if current_match is not None:
            active_index = current_match
        else:
            variants = _dedupe_social_candidates([current_social] + variants)
            active_index = 0

    next_index = (active_index + 1) % max(1, len(variants))
    new_social = _sanitize_social_copy(variants[next_index] if variants else current_social, max_chars=320)
    if not new_social:
        new_social = _build_fallback_social_copy(
            current_social=current_social,
            current_title=clip_payload.get("video_title_for_youtube_short") or clip_payload.get("title") or "",
            transcript_excerpt=clip_payload.get("transcript_excerpt") or "",
            score_reason=clip_payload.get("score_reason") or "",
            topic_tags=_normalize_topic_tags(clip_payload.get("topic_tags"))
        )
        next_index = 0

    clip_patch = {
        "video_description_for_tiktok": new_social,
        "video_description_for_instagram": new_social,
        "social_variants": variants,
        "social_variant_index": int(next_index)
    }

    try:
        metadata_clips[req.clip_index].update(clip_patch)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    if isinstance(job, dict) and isinstance(job.get("result"), dict):
        result_clips = job["result"].get("clips", [])
        if isinstance(result_clips, list) and req.clip_index < len(result_clips) and isinstance(result_clips[req.clip_index], dict):
            result_clips[req.clip_index].update(clip_patch)
    _persist_job_state(req.job_id)

    return {
        "success": True,
        "clip_index": req.clip_index,
        "previous_social": current_social,
        "new_social": new_social,
        "social_variants": variants,
        "social_variant_index": int(next_index),
        "social_variant_total": len(variants),
        "clip_patch": clip_patch
    }

class RecutRequest(BaseModel):
    job_id: str
    clip_index: int
    start: float
    end: float
    aspect_ratio: Optional[str] = None
    layout_mode: Optional[str] = "single"  # single | split
    fit_mode: Optional[str] = "cover"  # cover | contain
    zoom: Optional[float] = 1.0        # contain: 0.5..2.5 | cover: 1.0..2.5
    offset_x: Optional[float] = 0.0    # -100 .. 100
    offset_y: Optional[float] = 0.0    # -100 .. 100
    split_zoom_a: Optional[float] = 1.0
    split_offset_a_x: Optional[float] = 0.0
    split_offset_a_y: Optional[float] = 0.0
    split_zoom_b: Optional[float] = 1.0
    split_offset_b_x: Optional[float] = 0.0
    split_offset_b_y: Optional[float] = 0.0
    auto_smart_reframe: Optional[bool] = False
    smart_scene_frame_skip: Optional[int] = 1
    smart_scene_downscale: Optional[int] = 0

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
    job = _ensure_job_context(req.job_id)
    if not isinstance(job, dict):
        raise HTTPException(status_code=404, detail="Job not found")

    input_path = job.get('input_path')
    if not input_path or not os.path.exists(input_path):
        raise HTTPException(status_code=400, detail="Original source video not available for recut.")

    if req.start < 0 or req.end <= req.start:
        raise HTTPException(status_code=400, detail="Invalid start/end times.")
    aspect_ratio = normalize_aspect_ratio(req.aspect_ratio, default="9:16")
    layout_mode = str(req.layout_mode or "single").strip().lower()
    if layout_mode not in {"single", "split"}:
        layout_mode = "single"
    fit_mode = _normalize_layout_fit_mode(req.fit_mode)
    zoom = _coerce_layout_zoom(req.zoom, 1.0, fit_mode)
    offset_x = _coerce_layout_offset(req.offset_x, 0.0)
    offset_y = _coerce_layout_offset(req.offset_y, 0.0)
    split_zoom_a = _coerce_layout_zoom(req.split_zoom_a, zoom, fit_mode)
    split_offset_a_x = _coerce_layout_offset(req.split_offset_a_x, offset_x)
    split_offset_a_y = _coerce_layout_offset(req.split_offset_a_y, offset_y)
    split_zoom_b = _coerce_layout_zoom(req.split_zoom_b, zoom, fit_mode)
    split_offset_b_x = _coerce_layout_offset(req.split_offset_b_x, -offset_x)
    split_offset_b_y = _coerce_layout_offset(req.split_offset_b_y, offset_y)
    auto_smart_requested = bool(req.auto_smart_reframe)
    smart_scene_frame_skip = max(0, min(12, int(req.smart_scene_frame_skip if req.smart_scene_frame_skip is not None else 1)))
    smart_scene_downscale = max(0, min(8, int(req.smart_scene_downscale if req.smart_scene_downscale is not None else 0)))
    auto_smart_applied = False
    smart_summary: Optional[Dict[str, Any]] = None
    recut_warnings: List[str] = []
    if layout_mode == "split" and auto_smart_requested:
        recut_warnings.append("Auto smart reframe no se aplica en layout Split; se usó split manual.")

    output_dir = os.path.join(OUTPUT_DIR, req.job_id)
    os.makedirs(output_dir, exist_ok=True)

    # Try to find the uncut version of the clip first
    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
    source_to_cut = input_path
    if json_files:
        base_name = os.path.basename(json_files[0]).replace("_metadata.json", "")
        uncut_name = f"{base_name}_clip_{req.clip_index + 1}_uncut.mp4"
        uncut_path = os.path.join(output_dir, uncut_name)
        if os.path.exists(uncut_path):
            source_to_cut = uncut_path
            # Since uncut is already cut to the correct segment, we just use it directly
            # Or we can cut from it again just to be safe if start/end tweaked
            print(f"Recutting from uncut variant: {uncut_path}")

    # Cut from source
    temp_cut = os.path.join(output_dir, f"temp_recut_{req.clip_index}_{int(time.time())}.mp4")
    out_name = f"reclip_{req.clip_index+1}_{int(time.time())}.mp4"
    out_path = os.path.join(output_dir, out_name)

    # Note: If source_to_cut is the uncut clip, it's already bounded to the clip's original start.
    # However, 'req.start' is absolute relative to the *full* video.
    # We must adjust the cut command depending on what source we use.
    if source_to_cut == uncut_path:
        # The uncut clip starts at the clip's original start time.
        # We need to find what that original start time was.
        try:
            with open(json_files[0], 'r') as f:
                data = json.load(f)
            clips = data.get('shorts', [])
            original_start = 0.0
            if req.clip_index < len(clips):
                original_start = clips[req.clip_index].get('start', 0.0)
            
            # The requested start might be slightly tweaked by the user
            relative_start = max(0.0, req.start - original_start)
            duration = max(0.6, req.end - req.start)
            
            cut_cmd = [
                'ffmpeg', '-y',
                '-ss', str(relative_start),
                '-t', str(duration),
                '-i', source_to_cut,
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', '-preset', 'fast',
                '-c:a', 'aac',
                '-movflags', '+faststart',
                temp_cut
            ]
        except Exception as e:
            print(f"Failed to use uncut clip: {e}, falling back to full input")
            source_to_cut = input_path
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
    else:
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

    # Re-process with target aspect ratio + optional smart reframe/manual layout transform
    try:
        in_w, in_h = _probe_video_dimensions(temp_cut)
        if in_w <= 0 or in_h <= 0:
            raise HTTPException(status_code=500, detail="Could not inspect recut dimensions.")

        if auto_smart_requested and layout_mode == "single":
            try:
                smart_summary = _render_smart_reframe_video(
                    input_video_path=temp_cut,
                    output_path=out_path,
                    aspect_ratio=aspect_ratio,
                    scene_frame_skip=smart_scene_frame_skip,
                    scene_downscale=smart_scene_downscale
                )
                auto_smart_applied = True
            except Exception as smart_error:
                recut_warnings.append(
                    "Auto smart reframe no estuvo disponible para este clip; se aplicó layout manual."
                )
                print(f"⚠️ SmartReframe fallback to manual: {smart_error}")

        if not auto_smart_applied:
            if layout_mode == "split":
                filter_complex, out_w, out_h, out_label = _build_split_layout_filter_complex(
                    in_w=in_w,
                    in_h=in_h,
                    aspect_ratio=aspect_ratio,
                    fit_mode=fit_mode,
                    zoom_a=split_zoom_a,
                    offset_a_x=split_offset_a_x,
                    offset_a_y=split_offset_a_y,
                    zoom_b=split_zoom_b,
                    offset_b_x=split_offset_b_x,
                    offset_b_y=split_offset_b_y
                )
                recut_cmd = [
                    "ffmpeg", "-y",
                    "-i", temp_cut,
                    "-filter_complex", filter_complex,
                    "-map", f"[{out_label}]",
                    "-map", "0:a?",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "fast",
                    "-c:a", "aac",
                    "-movflags", "+faststart",
                    out_path
                ]
            else:
                vf_filter, out_w, out_h = _build_manual_layout_filter(
                    in_w=in_w,
                    in_h=in_h,
                    aspect_ratio=aspect_ratio,
                    fit_mode=fit_mode,
                    zoom=zoom,
                    offset_x=offset_x,
                    offset_y=offset_y
                )
                recut_cmd = [
                    "ffmpeg", "-y",
                    "-i", temp_cut,
                    "-filter_complex", vf_filter,
                    "-map", "[out_v]",
                    "-map", "0:a?",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "fast",
                    "-c:a", "aac",
                    "-movflags", "+faststart",
                    out_path
                ]
            recut_run = subprocess.run(recut_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if recut_run.returncode != 0:
                raise HTTPException(status_code=500, detail=recut_run.stderr.decode() or "Failed to apply manual layout recut.")
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
                clips[req.clip_index]['layout_mode'] = layout_mode
                clips[req.clip_index]['layout_fit_mode'] = fit_mode
                clips[req.clip_index]['layout_zoom'] = round(zoom, 3)
                clips[req.clip_index]['layout_offset_x'] = round(offset_x, 3)
                clips[req.clip_index]['layout_offset_y'] = round(offset_y, 3)
                clips[req.clip_index]['layout_split_zoom_a'] = round(split_zoom_a, 3)
                clips[req.clip_index]['layout_split_offset_a_x'] = round(split_offset_a_x, 3)
                clips[req.clip_index]['layout_split_offset_a_y'] = round(split_offset_a_y, 3)
                clips[req.clip_index]['layout_split_zoom_b'] = round(split_zoom_b, 3)
                clips[req.clip_index]['layout_split_offset_b_x'] = round(split_offset_b_x, 3)
                clips[req.clip_index]['layout_split_offset_b_y'] = round(split_offset_b_y, 3)
                clips[req.clip_index]['layout_auto_smart'] = bool(auto_smart_applied)
                if smart_summary:
                    clips[req.clip_index]['layout_smart_summary'] = smart_summary
                elif 'layout_smart_summary' in clips[req.clip_index]:
                    del clips[req.clip_index]['layout_smart_summary']
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
        job['result']['clips'][req.clip_index]['layout_mode'] = layout_mode
        job['result']['clips'][req.clip_index]['layout_fit_mode'] = fit_mode
        job['result']['clips'][req.clip_index]['layout_zoom'] = round(zoom, 3)
        job['result']['clips'][req.clip_index]['layout_offset_x'] = round(offset_x, 3)
        job['result']['clips'][req.clip_index]['layout_offset_y'] = round(offset_y, 3)
        job['result']['clips'][req.clip_index]['layout_split_zoom_a'] = round(split_zoom_a, 3)
        job['result']['clips'][req.clip_index]['layout_split_offset_a_x'] = round(split_offset_a_x, 3)
        job['result']['clips'][req.clip_index]['layout_split_offset_a_y'] = round(split_offset_a_y, 3)
        job['result']['clips'][req.clip_index]['layout_split_zoom_b'] = round(split_zoom_b, 3)
        job['result']['clips'][req.clip_index]['layout_split_offset_b_x'] = round(split_offset_b_x, 3)
        job['result']['clips'][req.clip_index]['layout_split_offset_b_y'] = round(split_offset_b_y, 3)
        job['result']['clips'][req.clip_index]['layout_auto_smart'] = bool(auto_smart_applied)
        if smart_summary:
            job['result']['clips'][req.clip_index]['layout_smart_summary'] = smart_summary
        elif 'layout_smart_summary' in job['result']['clips'][req.clip_index]:
            del job['result']['clips'][req.clip_index]['layout_smart_summary']
        _persist_job_state(req.job_id)

    return {
        "success": True,
        "new_video_url": f"/videos/{req.job_id}/{out_name}",
        "start": req.start,
        "end": req.end,
        "layout_mode": layout_mode,
        "split_layout_applied": layout_mode == "split" and not auto_smart_applied,
        "auto_smart_reframe_requested": bool(auto_smart_requested),
        "auto_smart_reframe_applied": bool(auto_smart_applied),
        "smart_reframe_summary": smart_summary,
        "warnings": recut_warnings
    }

@app.post("/api/music")
async def add_background_music(
    job_id: str = Form(...),
    clip_index: int = Form(...),
    file: UploadFile = File(...),
    input_filename: Optional[str] = Form(None),
    music_volume: float = Form(0.18),
    duck_voice: Optional[str] = Form("true")
):
    job = _ensure_job_context(job_id)
    if not isinstance(job, dict):
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
        source_name = _safe_input_filename(input_filename)
    else:
        source_name = _safe_input_filename(clip_data.get("video_url", ""))
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
        if isinstance(job.get("result"), dict):
            job_clips = job["result"].get("clips", [])
            if isinstance(job_clips, list) and clip_index < len(job_clips):
                job_clips[clip_index]["video_url"] = new_video_url
            _persist_job_state(job_id)

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

def _ffmpeg_concat_escape(path: str) -> str:
    return str(path or "").replace("\\", "\\\\").replace(" ", "\\ ").replace("'", "\\'")

def _normalize_highlight_source_mode(raw_mode: Optional[str]) -> str:
    value = str(raw_mode or "semantic").strip().lower()
    aliases = {
        "semantic": "semantic",
        "semantico": "semantic",
        "semántico": "semantic",
        "auto": "semantic",
        "default": "semantic",
        "clips": "clips",
        "clip": "clips",
        "legacy": "clips",
    }
    return aliases.get(value, "semantic")

def _is_generated_video_artifact_name(filename: str) -> bool:
    name = str(filename or "").strip().lower()
    if not name:
        return True
    if "_clip_" in name:
        return True
    generated_prefixes = (
        "temp_",
        "reclip_",
        "subtitled_",
        "highlight_reel_",
        "social_",
        "preview_",
        "platform_",
    )
    if name.startswith(generated_prefixes):
        return True
    return False

def _resolve_highlight_source_video_path(
    job: Dict[str, Any],
    output_dir: str,
    metadata: Dict[str, Any]
) -> Tuple[Optional[str], str]:
    candidates: List[Tuple[int, float, str, str]] = []
    seen_paths = set()
    allowed_ext = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".mpg", ".mpeg"}

    def add_candidate(raw_path: Optional[str], priority: int, reason: str):
        path = str(raw_path or "").strip()
        if not path:
            return
        path = os.path.abspath(path)
        if path in seen_paths:
            return
        seen_paths.add(path)
        if not os.path.isfile(path):
            return
        ext = os.path.splitext(path)[1].lower()
        if ext and ext not in allowed_ext:
            return
        try:
            size = float(os.path.getsize(path))
        except Exception:
            size = 0.0
        candidates.append((priority, size, path, reason))

    add_candidate(job.get("input_path"), 0, "job_input_path")

    if isinstance(metadata, dict):
        for key in (
            "source_video_path",
            "original_video_path",
            "input_video_path",
            "input_path",
            "source_path",
            "source_file_path",
        ):
            add_candidate(metadata.get(key), 1, f"metadata:{key}")

    for path in sorted(glob.glob(os.path.join(output_dir, "*"))):
        if not os.path.isfile(path):
            continue
        name = os.path.basename(path)
        ext = os.path.splitext(name)[1].lower()
        if ext not in allowed_ext:
            continue
        if _is_generated_video_artifact_name(name):
            continue
        name_l = name.lower()
        priority = 2 if name_l.endswith("_vertical.mp4") else 3
        add_candidate(path, priority, "output_scan")

    if not candidates:
        return None, "not_found"

    candidates.sort(key=lambda item: (item[0], -item[1]))
    _, _, best_path, reason = candidates[0]
    return best_path, reason

def _build_semantic_highlight_queries(
    transcript_text: str,
    chapters: List[Dict[str, Any]],
    clips: List[Dict[str, Any]],
    max_queries: int = 10
) -> List[str]:
    queries: List[str] = []
    seen = set()

    def add_query(raw_value: str):
        text = _normalize_space(raw_value)
        if len(text) < 4:
            return
        key = text.lower()
        if key in seen:
            return
        seen.add(key)
        queries.append(text[:180])

    for chapter in (chapters or [])[:6]:
        add_query(chapter.get("title", ""))

    keywords = _extract_topic_keywords(transcript_text or "", limit=10)
    for i in range(0, max(0, len(keywords) - 1), 2):
        add_query(f"{keywords[i]} {keywords[i + 1]}")
    for kw in keywords[:4]:
        add_query(f"momento clave sobre {kw}")

    if isinstance(clips, list) and clips:
        ranked = sorted(
            clips,
            key=lambda c: _safe_float((c or {}).get("virality_score", 0.0), 0.0),
            reverse=True
        )
        for clip in ranked[:4]:
            add_query(str((clip or {}).get("video_title_for_youtube_short", "")))

    if not queries:
        for fallback in (
            "momento clave del video",
            "idea principal del debate",
            "punto mas fuerte del episodio",
            "frase mas importante",
        ):
            add_query(fallback)

    return queries[:max(1, min(120, int(max_queries)))]

def _select_highlight_reel_segments_semantic(
    job_id: str,
    metadata_path: str,
    transcript: Dict[str, Any],
    clips: List[Dict[str, Any]],
    max_segments: int,
    target_duration: float,
    min_segment_seconds: float,
    max_segment_seconds: float,
    min_gap_seconds: float
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    safe_max_segments = max(1, min(240, int(max_segments)))
    safe_target_duration = max(8.0, min(10800.0, _safe_float(target_duration, 50.0)))
    safe_min_segment = max(1.0, min(20.0, _safe_float(min_segment_seconds, 4.5)))
    safe_max_segment = max(safe_min_segment, min(35.0, _safe_float(max_segment_seconds, 11.0)))
    safe_min_gap = max(0.0, min(80.0, _safe_float(min_gap_seconds, 7.0)))

    index = _ensure_search_index(
        job_id=job_id,
        metadata_path=metadata_path,
        transcript=transcript if isinstance(transcript, dict) else {},
        clips=clips or [],
        semantic_api_key=None
    )

    units = index.get("units") or []
    unit_embeddings = index.get("unit_embeddings") or []
    duration = _safe_float(index.get("duration", 0.0), 0.0)
    transcript_text = _normalize_space(index.get("transcript_text", ""))
    index_clips = index.get("clips") or []
    chapters = index.get("chapters") or []
    if duration <= 0.8 and units:
        duration = max(_safe_float(u.get("end", 0.0), 0.0) for u in units)

    if not units or duration <= 0.8:
        return [], {
            "provider": "local",
            "queries": [],
            "candidates": 0,
            "selected": 0,
            "reason": "missing_transcript_units"
        }

    queries = _build_semantic_highlight_queries(
        transcript_text=transcript_text,
        chapters=chapters,
        clips=index_clips,
        max_queries=max(4, safe_max_segments + 3)
    )

    by_unit: Dict[int, Dict[str, Any]] = {}
    max_matches_per_query = max(2, min(10, max(4, (safe_max_segments // 5) + 2)))
    for q_index, query in enumerate(queries):
        keywords = _tokenize_query(query)
        phrases = _extract_query_phrases(query, keywords)
        if not keywords and not phrases:
            continue
        profile = _apply_search_mode_override(_analyze_query_profile(query, keywords, phrases), "broad")
        query_embedding = _local_semantic_embedding(query)

        matches = _build_semantic_matches(
            query=query,
            keywords=keywords,
            phrases=phrases,
            query_profile=profile,
            units=units,
            unit_embeddings=unit_embeddings,
            query_embedding=query_embedding,
            transcript_text=transcript_text,
            clips=index_clips,
            duration=duration,
            limit=max_matches_per_query
        )

        if len(matches) < max(2, math.ceil(max_matches_per_query * 0.5)):
            relaxed = _relax_query_profile(profile)
            relaxed_matches = _build_semantic_matches(
                query=query,
                keywords=keywords,
                phrases=phrases,
                query_profile=relaxed,
                units=units,
                unit_embeddings=unit_embeddings,
                query_embedding=query_embedding,
                transcript_text=transcript_text,
                clips=index_clips,
                duration=duration,
                limit=max_matches_per_query
            )
            if len(relaxed_matches) > len(matches):
                matches = relaxed_matches

        for m_index, match in enumerate(matches[:max_matches_per_query]):
            unit_index = int(_safe_float(match.get("unit_index", -1), -1))
            if unit_index < 0 or unit_index >= len(units):
                continue
            unit = units[unit_index]

            window_start = max(0.0, _safe_float(match.get("start", unit.get("start", 0.0)), 0.0))
            window_end = max(window_start + 0.8, _safe_float(match.get("end", unit.get("end", window_start + 0.8)), window_start + 0.8))
            if duration > 0:
                window_end = min(window_end, duration)
            if window_end - window_start < 0.8:
                continue

            unit_start = max(window_start, _safe_float(unit.get("start", window_start), window_start))
            unit_end = min(window_end, max(unit_start + 0.4, _safe_float(unit.get("end", unit_start + 0.4), unit_start + 0.4)))
            if unit_end <= unit_start:
                unit_start = window_start
                unit_end = min(window_end, unit_start + 1.2)

            match_score = max(0.0, min(1.0, _safe_float(match.get("match_score", 0.0), 0.0)))
            semantic_score = max(0.0, min(1.0, _safe_float(match.get("semantic_score", 0.0), 0.0)))
            keyword_score = max(0.0, min(1.0, _safe_float(match.get("keyword_score", 0.0), 0.0)))
            phrase_score = max(0.0, min(1.0, _safe_float(match.get("phrase_score", 0.0), 0.0)))
            virality_boost = max(0.0, min(1.0, _safe_float(match.get("virality_boost", 0.0), 0.0)))
            confidence = max(
                0.0,
                min(
                    1.0,
                    (match_score * 0.68)
                    + (semantic_score * 0.12)
                    + (keyword_score * 0.10)
                    + (phrase_score * 0.05)
                    + (virality_boost * 0.05)
                )
            )

            cap = min(safe_max_segment, max(0.8, window_end - window_start))
            floor = min(cap, safe_min_segment)
            desired = min(
                cap,
                max(
                    floor,
                    safe_min_segment
                    + ((safe_max_segment - safe_min_segment) * (0.36 + (confidence * 0.64)))
                )
            )

            center = (unit_start + unit_end) / 2.0
            seg_start = max(window_start, min(center - (desired * 0.48), window_end - desired))
            seg_end = min(window_end, seg_start + desired)
            if seg_end - seg_start < 0.8:
                seg_start = window_start
                seg_end = min(window_end, seg_start + max(0.8, floor))
            if seg_end - seg_start < 0.8:
                continue

            candidate = {
                "clip_index": int(_safe_float(match.get("source_clip_index", unit_index), unit_index)),
                "unit_index": unit_index,
                "score": round(confidence * 100.0, 4),
                "confidence": round(confidence, 6),
                "timeline_start": round(window_start, 4),
                "timeline_end": round(window_end, 4),
                "timeline_center": round((window_start + window_end) / 2.0, 4),
                "segment_floor": round(floor, 4),
                "segment_cap": round(cap, 4),
                "desired": round(min(desired, max(0.8, seg_end - seg_start)), 4),
                "segment_local_start": round(seg_start, 4),
                "segment_local_end_cap": round(seg_end, 4),
                "title": _normalize_space(match.get("snippet", ""))[:120] or f"Moment {unit_index + 1}",
                "transcript_excerpt": _normalize_space(match.get("snippet", ""))[:280],
                "seed_query": query,
                "query_rank": q_index,
                "match_rank": m_index,
                "source_clip_index": match.get("source_clip_index")
            }

            prev = by_unit.get(unit_index)
            if not prev or _safe_float(candidate.get("score", 0.0), 0.0) > _safe_float(prev.get("score", 0.0), 0.0):
                by_unit[unit_index] = candidate

    candidates = list(by_unit.values())
    if not candidates:
        # Fallback semantic ranking without explicit query (still transcript-level).
        for unit_index, unit in enumerate(units):
            unit_text = _normalize_space(unit.get("text", ""))
            if len(unit_text) < 4:
                continue
            start = max(0.0, _safe_float(unit.get("start", 0.0), 0.0))
            end = max(start + 0.8, _safe_float(unit.get("end", start + 0.8), start + 0.8))
            if duration > 0:
                end = min(end, duration)
            if end - start < 0.8:
                continue
            virality_boost, overlap_clip = _virality_overlap_score(start, end, index_clips)
            emphasis_bonus = 0.10 if ("?" in unit_text or "!" in unit_text) else 0.0
            length_bonus = min(0.22, len(unit_text) / 260.0)
            confidence = min(0.88, 0.34 + (virality_boost * 0.28) + emphasis_bonus + length_bonus)

            cap = min(safe_max_segment, max(0.8, end - start))
            floor = min(cap, safe_min_segment)
            desired = min(
                cap,
                max(floor, safe_min_segment + ((safe_max_segment - safe_min_segment) * (0.3 + (confidence * 0.7))))
            )
            seg_start = start
            seg_end = min(end, seg_start + desired)
            if seg_end - seg_start < 0.8:
                continue

            candidates.append({
                "clip_index": int(_safe_float((overlap_clip or {}).get("clip_index", unit_index), unit_index)),
                "unit_index": unit_index,
                "score": round(confidence * 100.0, 4),
                "confidence": round(confidence, 6),
                "timeline_start": round(start, 4),
                "timeline_end": round(end, 4),
                "timeline_center": round((start + end) / 2.0, 4),
                "segment_floor": round(floor, 4),
                "segment_cap": round(cap, 4),
                "desired": round(min(desired, max(0.8, seg_end - seg_start)), 4),
                "segment_local_start": round(seg_start, 4),
                "segment_local_end_cap": round(seg_end, 4),
                "title": unit_text[:120] or f"Moment {unit_index + 1}",
                "transcript_excerpt": unit_text[:280],
                "seed_query": "fallback",
                "query_rank": 999,
                "match_rank": unit_index,
                "source_clip_index": (overlap_clip or {}).get("clip_index")
            })

    if not candidates:
        return [], {
            "provider": "local",
            "queries": queries,
            "candidates": 0,
            "selected": 0,
            "reason": "no_semantic_candidates"
        }

    candidates.sort(
        key=lambda item: (
            _safe_float(item.get("score", 0.0), 0.0),
            _safe_float(item.get("confidence", 0.0), 0.0),
            -_safe_float(item.get("query_rank", 999.0), 999.0),
            -_safe_float(item.get("timeline_start", 0.0), 0.0)
        ),
        reverse=True
    )

    selected: List[Dict[str, Any]] = []
    selected_total = 0.0
    if safe_max_segments >= 24:
        max_per_query = max(2, min(6, safe_max_segments // 8))
    else:
        max_per_query = max(1, min(2, safe_max_segments // 2 if safe_max_segments > 2 else 1))
    query_counts: Dict[str, int] = {}
    for candidate in candidates:
        if len(selected) >= safe_max_segments:
            break
        q_key = str(candidate.get("seed_query", "")).lower()
        if q_key and query_counts.get(q_key, 0) >= max_per_query:
            continue

        is_conflicting = False
        for prev in selected:
            overlap = _temporal_overlap_ratio(
                _safe_float(candidate.get("timeline_start", 0.0), 0.0),
                _safe_float(candidate.get("timeline_end", 0.0), 0.0),
                _safe_float(prev.get("timeline_start", 0.0), 0.0),
                _safe_float(prev.get("timeline_end", 0.0), 0.0),
            )
            centers_too_close = abs(
                _safe_float(candidate.get("timeline_center", 0.0), 0.0)
                - _safe_float(prev.get("timeline_center", 0.0), 0.0)
            ) < safe_min_gap
            if overlap >= 0.42 or centers_too_close:
                is_conflicting = True
                break
        if is_conflicting:
            continue

        selected.append(dict(candidate))
        selected_total += _safe_float(candidate.get("desired", 0.0), 0.0)
        if q_key:
            query_counts[q_key] = query_counts.get(q_key, 0) + 1
        if selected_total >= safe_target_duration:
            break

    if not selected:
        selected = [dict(candidates[0])]
        selected_total = _safe_float(selected[0].get("desired", 0.0), 0.0)

    if len(selected) < safe_max_segments and selected_total < (safe_target_duration * 0.82):
        used = {
            int(_safe_float(item.get("unit_index", -1), -1))
            for item in selected
        }
        for candidate in candidates:
            if len(selected) >= safe_max_segments:
                break
            marker = int(_safe_float(candidate.get("unit_index", -1), -1))
            if marker in used:
                continue
            selected.append(dict(candidate))
            used.add(marker)
            selected_total += _safe_float(candidate.get("desired", 0.0), 0.0)
            if selected_total >= safe_target_duration:
                break

    selected.sort(key=lambda item: _safe_float(item.get("timeline_start", 0.0), 0.0))

    planned: List[Dict[str, Any]] = []
    remaining = safe_target_duration
    for i, candidate in enumerate(selected):
        if remaining <= 0.35:
            break
        reserve_for_rest = 0.0
        for future in selected[i + 1:]:
            reserve_for_rest += min(
                _safe_float(future.get("segment_floor", 0.8), 0.8),
                _safe_float(future.get("segment_cap", safe_max_segment), safe_max_segment)
            )

        seg_cap = _safe_float(candidate.get("segment_cap", safe_max_segment), safe_max_segment)
        seg_floor = min(seg_cap, _safe_float(candidate.get("segment_floor", safe_min_segment), safe_min_segment))
        desired = _safe_float(candidate.get("desired", seg_floor), seg_floor)
        max_for_this = min(seg_cap, max(0.8, remaining - reserve_for_rest))
        seg_duration = min(desired, max_for_this)
        if seg_duration < seg_floor:
            if planned and remaining - reserve_for_rest <= 0.8:
                continue
            seg_duration = min(seg_cap, max(0.8, remaining - reserve_for_rest))

        window_start = max(0.0, _safe_float(candidate.get("timeline_start", 0.0), 0.0))
        window_end = max(window_start + 0.8, _safe_float(candidate.get("timeline_end", window_start + 0.8), window_start + 0.8))
        if duration > 0:
            window_end = min(window_end, duration)
        if window_end - window_start < 0.8:
            continue

        center = _safe_float(candidate.get("timeline_center", (window_start + window_end) / 2.0), (window_start + window_end) / 2.0)
        seg_start = max(window_start, min(center - (seg_duration * 0.5), window_end - seg_duration))
        seg_end = min(window_end, seg_start + seg_duration)
        actual_duration = seg_end - seg_start
        if actual_duration < 0.8:
            continue

        item = dict(candidate)
        item["segment_local_start"] = round(seg_start, 3)
        item["segment_local_end"] = round(seg_end, 3)
        item["segment_duration"] = round(actual_duration, 3)
        planned.append(item)
        remaining -= actual_duration

    return planned, {
        "provider": "local",
        "queries": queries,
        "candidates": len(candidates),
        "selected": len(planned),
    }

def _select_highlight_reel_segments(
    normalized_clips: List[Dict[str, Any]],
    output_dir: str,
    max_segments: int,
    target_duration: float,
    min_segment_seconds: float,
    max_segment_seconds: float,
    min_gap_seconds: float
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    safe_max_segments = max(1, min(240, int(max_segments)))
    safe_target_duration = max(8.0, min(10800.0, _safe_float(target_duration, 50.0)))
    safe_min_segment = max(1.0, min(20.0, _safe_float(min_segment_seconds, 4.5)))
    safe_max_segment = max(safe_min_segment, min(35.0, _safe_float(max_segment_seconds, 11.0)))
    safe_min_gap = max(0.0, min(80.0, _safe_float(min_gap_seconds, 7.0)))

    for i, clip in enumerate(normalized_clips):
        clip_index = int(_safe_float(clip.get("clip_index", i), i))
        clip_path = _clip_file_path_from_payload(output_dir, clip, i + 1)
        if not clip_path:
            continue

        timeline_start = max(0.0, _safe_float(clip.get("start", 0.0), 0.0))
        timeline_end = max(timeline_start, _safe_float(clip.get("end", timeline_start), timeline_start))
        timeline_duration = max(0.0, timeline_end - timeline_start)
        source_duration = _probe_media_duration_seconds(clip_path)
        if source_duration <= 0:
            source_duration = timeline_duration
        if source_duration <= 0.8:
            continue

        score = max(0.0, min(100.0, _safe_float(clip.get("virality_score", 0.0), 0.0)))
        confidence = max(0.0, min(1.0, _safe_float(clip.get("selection_confidence", score / 100.0), score / 100.0)))
        cap = min(safe_max_segment, source_duration)
        if cap <= 0.8:
            continue
        floor = min(cap, safe_min_segment)
        score_norm = score / 100.0
        desired = min(
            cap,
            max(floor, safe_min_segment + ((safe_max_segment - safe_min_segment) * (0.45 + (score_norm * 0.55))))
        )

        local_start = 0.0
        if source_duration - desired > 0.35:
            # Avoid first-frame hard cuts but keep the hook very close to the start.
            local_start = min(max(0.12, source_duration * 0.04), source_duration - desired)
        local_end_cap = min(source_duration, local_start + cap)
        desired = min(desired, max(0.8, local_end_cap - local_start))

        candidates.append({
            "clip_index": clip_index,
            "score": score,
            "confidence": confidence,
            "timeline_start": timeline_start,
            "timeline_end": timeline_end if timeline_end > timeline_start else timeline_start + source_duration,
            "timeline_center": (timeline_start + (timeline_end if timeline_end > timeline_start else timeline_start + source_duration)) / 2.0,
            "source_file_path": clip_path,
            "source_duration": source_duration,
            "segment_local_start": local_start,
            "segment_local_end_cap": local_end_cap,
            "segment_cap": cap,
            "segment_floor": floor,
            "desired": desired,
            "title": _normalize_space(clip.get("video_title_for_youtube_short", "")) or f"Clip {clip_index + 1}",
            "transcript_excerpt": _normalize_space(clip.get("transcript_excerpt", ""))[:260]
        })

    if not candidates:
        return []

    candidates.sort(
        key=lambda c: (
            c["score"],
            c["confidence"],
            -c["timeline_start"]
        ),
        reverse=True
    )

    selected: List[Dict[str, Any]] = []
    selected_total = 0.0
    for candidate in candidates:
        if len(selected) >= safe_max_segments:
            break

        is_conflicting = False
        for prev in selected:
            overlap = _temporal_overlap_ratio(
                candidate["timeline_start"],
                candidate["timeline_end"],
                prev["timeline_start"],
                prev["timeline_end"]
            )
            centers_too_close = abs(candidate["timeline_center"] - prev["timeline_center"]) < safe_min_gap
            if overlap >= 0.38 or centers_too_close:
                is_conflicting = True
                break
        if is_conflicting:
            continue

        selected.append(dict(candidate))
        selected_total += candidate["desired"]
        if selected_total >= safe_target_duration:
            break

    if not selected:
        selected = [dict(candidates[0])]
        selected_total = selected[0]["desired"]

    if len(selected) < safe_max_segments and selected_total < (safe_target_duration * 0.82):
        selected_indexes = {int(item["clip_index"]) for item in selected}
        for candidate in candidates:
            if len(selected) >= safe_max_segments:
                break
            if int(candidate["clip_index"]) in selected_indexes:
                continue
            selected.append(dict(candidate))
            selected_indexes.add(int(candidate["clip_index"]))
            selected_total += candidate["desired"]
            if selected_total >= safe_target_duration:
                break

    selected.sort(key=lambda item: item["timeline_start"])

    planned: List[Dict[str, Any]] = []
    remaining = safe_target_duration
    for i, candidate in enumerate(selected):
        if remaining <= 0.35:
            break
        reserve_for_rest = 0.0
        for future in selected[i + 1:]:
            reserve_for_rest += min(future["segment_floor"], future["segment_cap"])

        max_for_this = min(candidate["segment_cap"], max(0.8, remaining - reserve_for_rest))
        seg_duration = min(candidate["desired"], max_for_this)
        floor = min(candidate["segment_floor"], candidate["segment_cap"])
        if seg_duration < floor:
            if planned:
                if remaining - reserve_for_rest <= 0.8:
                    continue
                seg_duration = min(candidate["segment_cap"], max(0.8, remaining - reserve_for_rest))
            else:
                seg_duration = min(candidate["segment_cap"], max(0.8, remaining))

        seg_duration = max(0.8, min(seg_duration, candidate["segment_cap"]))
        local_start = max(0.0, min(candidate["segment_local_start"], candidate["source_duration"] - 0.8))
        local_end = min(candidate["segment_local_end_cap"], local_start + seg_duration)
        actual_duration = local_end - local_start
        if actual_duration < 0.8:
            continue

        item = dict(candidate)
        item["segment_local_start"] = round(local_start, 3)
        item["segment_local_end"] = round(local_end, 3)
        item["segment_duration"] = round(actual_duration, 3)
        planned.append(item)
        remaining -= actual_duration

    return planned

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

def _platform_variant_target(platform: str) -> Tuple[str, int, int]:
    key = str(platform or "").strip().lower()
    if key == "youtube":
        return "16:9", 1920, 1080
    return "9:16", 1080, 1920

def _render_platform_variant_video(
    source_path: str,
    output_path: str,
    platform: str
) -> bool:
    if not source_path or not os.path.exists(source_path):
        return False
    ratio, target_w, target_h = _platform_variant_target(platform)

    in_w, in_h = _probe_video_dimensions(source_path)
    if in_w <= 0 or in_h <= 0:
        return False
    vf_filter, _, _ = _build_manual_layout_filter(
        in_w=in_w,
        in_h=in_h,
        aspect_ratio=ratio,
        fit_mode="cover",
        zoom=1.0,
        offset_x=0.0,
        offset_y=0.0
    )
    # Normalize final output dimensions per platform for delivery consistency.
    vf_final = f"{vf_filter},scale={target_w}:{target_h}"
    cmd = [
        "ffmpeg", "-y",
        "-i", source_path,
        "-vf", vf_final,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        "-preset", "fast",
        "-c:a", "aac",
        "-movflags", "+faststart",
        output_path
    ]
    run = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return run.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0

class ExportPackRequest(BaseModel):
    job_id: str
    include_video_files: bool = True
    include_srt_files: bool = True
    include_thumbnails: bool = True
    include_platform_variants: bool = True
    include_platform_video_variants: bool = False
    thumbnail_format: str = "jpg"
    thumbnail_width: int = 1080

@app.post("/api/export/pack")
async def export_pack(req: ExportPackRequest):
    job = _ensure_job_context(req.job_id)
    if not isinstance(job, dict):
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
    platform_video_variants: List[Dict[str, Any]] = []
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

        if req.include_platform_video_variants:
            for platform in ("youtube", "instagram", "tiktok"):
                variant_name = f"variant_clip_{i+1}_{platform}_{int(time.time())}.mp4"
                variant_path = os.path.join(output_dir, variant_name)
                ok = _render_platform_variant_video(
                    source_path=clip_file,
                    output_path=variant_path,
                    platform=platform
                )
                if not ok:
                    continue
                ratio, _, _ = _platform_variant_target(platform)
                variant_entry = {
                    "clip_number": i + 1,
                    "clip_index": clip.get("clip_index", i),
                    "platform": platform,
                    "aspect_ratio": ratio,
                    "filename": variant_name,
                    "video_url": f"/videos/{req.job_id}/{variant_name}",
                }
                platform_video_variants.append(variant_entry)
                clip.setdefault("platform_video_variants", []).append(variant_entry)

    manifest = {
        "job_id": req.job_id,
        "generated_at": int(time.time()),
        "export_version": "v3",
        "options": {
            "include_video_files": bool(req.include_video_files),
            "include_srt_files": bool(req.include_srt_files),
            "include_thumbnails": bool(req.include_thumbnails),
            "include_platform_variants": bool(req.include_platform_variants),
            "include_platform_video_variants": bool(req.include_platform_video_variants),
            "thumbnail_format": thumb_format,
            "thumbnail_width": thumb_width
        },
        "clips": normalized_clips,
        "platform_video_variants": platform_video_variants
    }

    zip_name = f"agency_pack_{req.job_id}_{int(time.time())}.zip"
    zip_path = os.path.join(output_dir, zip_name)
    clip_files_added = 0
    srt_files_added = 0
    thumbnail_files_added = 0
    platform_video_variant_files_added = 0

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

        if req.include_platform_video_variants and platform_video_variants:
            seen_variants = set()
            for item in platform_video_variants:
                filename = _safe_input_filename(item.get("filename", ""))
                if not filename:
                    continue
                variant_path = os.path.join(output_dir, filename)
                if variant_path in seen_variants or not os.path.exists(variant_path):
                    continue
                seen_variants.add(variant_path)
                zf.write(variant_path, arcname=f"platform_variants/{filename}")
                platform_video_variant_files_added += 1

    return {
        "success": True,
        "pack_url": f"/videos/{req.job_id}/{zip_name}",
        "clips_in_manifest": len(normalized_clips),
        "video_files_added": clip_files_added,
        "srt_files_added": srt_files_added,
        "thumbnail_files_added": thumbnail_files_added,
        "platform_variant_rows": len(platform_rows) if req.include_platform_variants else 0,
        "platform_video_variant_files_added": platform_video_variant_files_added
    }

class TrailerRequest(BaseModel):
    job_id: str
    aspect_ratio: Optional[str] = None
    fragments: Optional[List[Dict[str, Any]]] = None

@app.post("/api/clip/trailer")
async def generate_trailer(req: TrailerRequest):
    output_dir = os.path.join(OUTPUT_DIR, req.job_id)
    if not os.path.isdir(output_dir):
        raise HTTPException(status_code=404, detail="Job output not found")

    metadata_path = None
    json_files = sorted(glob.glob(os.path.join(output_dir, "*_metadata.json")))
    if json_files:
        metadata_path = json_files[0]

    metadata = {}
    if metadata_path and os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            pass

    fragments = req.fragments or metadata.get("trailer_fragments")
    if not fragments or len(fragments) < 2:
        raise HTTPException(status_code=400, detail="No se encontraron fragmentos para generar el trailer.")

    target_aspect_ratio = req.aspect_ratio or metadata.get("aspect_ratio") or "9:16"
    
    # Try to resolve source video
    source_video_path = None
    # 1. Check job input path if preserved
    job = jobs.get(req.job_id)
    if job and job.get("input_path") and os.path.exists(job["input_path"]):
        source_video_path = job["input_path"]
    
    # 2. Look in output dir for files not like *_clip_* or subtitled_* or highlight_*
    if not source_video_path:
        candidates = []
        for f in glob.glob(os.path.join(output_dir, "*.mp4")):
            bn = os.path.basename(f)
            if "_clip_" in bn or "subtitled_" in bn or "highlight_" in bn or "trailer" in bn or "temp_" in bn:
                continue
            candidates.append(f)
        if candidates:
            # Prefer largest or first
            source_video_path = sorted(candidates, key=lambda x: os.path.getsize(x), reverse=True)[0]

    if not source_video_path or not os.path.exists(source_video_path):
        raise HTTPException(status_code=400, detail="No se encontró el video original para generar el trailer.")

    # We call main.py with --build-trailer
    # but since fragments might be customized, we pass them if possible.
    # For now, build_super_trailer is in main.py, so we trigger a minimal run if needed,
    # or just use the logic in main.py if we import it.
    
    # Actually, importing main is heavy. Let's use a temporary metadata override and call main.py.
    # Or just call the functions directly. Let's try direct call as it's faster.
    import main
    video_title = os.path.splitext(os.path.basename(source_video_path))[0]
    trailer_uncut_name = f"{video_title}_trailer_uncut.mp4"
    trailer_final_name = f"{video_title}_trailer.mp4"
    trailer_uncut_path = os.path.join(output_dir, trailer_uncut_name)
    trailer_final_path = os.path.join(output_dir, trailer_final_name)
    
    try:
        ok = main.build_super_trailer(source_video_path, fragments, trailer_uncut_path)
        if not ok:
            raise HTTPException(status_code=500, detail="Error al construir los fragmentos del trailer.")
            
        ok_v = main.process_video_to_vertical(trailer_uncut_path, trailer_final_path, aspect_ratio=target_aspect_ratio)
        if not ok_v:
            raise HTTPException(status_code=500, detail="Error al procesar el trailer verticalmente.")
            
        trailer_url = f"/videos/{req.job_id}/{trailer_final_name}"
        metadata['latest_trailer_url'] = trailer_url
        metadata['trailer_fragments'] = fragments
        
        # Save metadata
        if metadata_path:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
                
        if job and isinstance(job.get("result"), dict):
            job["result"]["latest_trailer_url"] = trailer_url
            _persist_job_state(req.job_id)

        return {
            "success": True,
            "trailer_url": trailer_url,
            "fragments_used": len(fragments)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trailer generation failed: {str(e)}")

class HighlightReelRequest(BaseModel):
    job_id: str
    max_segments: int = 6
    target_duration: float = 50.0
    min_segment_seconds: float = 4.5
    max_segment_seconds: float = 11.0
    min_gap_seconds: float = 7.0
    aspect_ratio: Optional[str] = None
    source_mode: Optional[str] = "semantic"

@app.post("/api/highlight/reel")
@app.post("/api/highlight-reel")
async def generate_highlight_reel(req: HighlightReelRequest):
    job = _ensure_job_context(req.job_id)
    if not isinstance(job, dict):
        raise HTTPException(status_code=404, detail="Job not found")

    output_dir = os.path.join(OUTPUT_DIR, req.job_id)
    if not os.path.isdir(output_dir):
        raise HTTPException(status_code=404, detail="Job output directory not found")

    metadata_files = sorted(glob.glob(os.path.join(output_dir, "*_metadata.json")))
    if not metadata_files:
        raise HTTPException(status_code=404, detail="Metadata not found")
    metadata_path = metadata_files[0]

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read metadata: {e}")

    clips = []
    if isinstance(job, dict) and isinstance(job.get("result"), dict):
        clips = job["result"].get("clips", []) or []
    if not clips and isinstance(metadata, dict):
        clips = metadata.get("shorts", []) or []
    if not isinstance(clips, list):
        clips = []

    transcript_data = metadata.get("transcript") if isinstance(metadata, dict) else None
    normalized_clips = [
        _normalize_clip_payload(
            dict(clip) if isinstance(clip, dict) else {},
            i,
            transcript=transcript_data
        )
        for i, clip in enumerate(clips)
    ]
    inferred_aspect_ratio = "9:16"
    for clip in normalized_clips:
        raw_ar = str(clip.get("aspect_ratio", "")).strip().replace("/", ":")
        if raw_ar in ALLOWED_ASPECT_RATIOS:
            inferred_aspect_ratio = raw_ar
            break
    target_aspect_ratio = normalize_aspect_ratio(req.aspect_ratio, default=inferred_aspect_ratio) or inferred_aspect_ratio
    target_w, target_h = (1080, 1920) if target_aspect_ratio == "9:16" else (1920, 1080)
    segment_vf = f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h}"

    safe_min_segment = max(1.0, min(20.0, _safe_float(req.min_segment_seconds, 4.5)))
    safe_max_segment = max(safe_min_segment, min(35.0, _safe_float(req.max_segment_seconds, 11.0)))
    safe_min_gap = max(0.0, min(80.0, _safe_float(req.min_gap_seconds, 7.0)))

    requested_source_mode = _normalize_highlight_source_mode(req.source_mode)
    selection_mode = "clips"
    warnings: List[str] = []
    semantic_context: Dict[str, Any] = {
        "provider": "local",
        "queries": [],
        "candidates": 0,
        "selected": 0
    }

    source_video_path, source_path_reason = _resolve_highlight_source_video_path(
        job=job,
        output_dir=output_dir,
        metadata=metadata if isinstance(metadata, dict) else {}
    )
    source_video_duration = _probe_media_duration_seconds(source_video_path) if source_video_path else 0.0
    timeline_duration = 0.0
    if normalized_clips:
        min_start = min(max(0.0, _safe_float(c.get("start", 0.0), 0.0)) for c in normalized_clips)
        max_end = max(
            max(max(0.0, _safe_float(c.get("start", 0.0), 0.0)), _safe_float(c.get("end", 0.0), 0.0))
            for c in normalized_clips
        )
        timeline_duration = max(0.0, max_end - min_start)

    raw_max_segments = _safe_int(req.max_segments, 0)
    if raw_max_segments <= 0:
        # Auto mode: estimate segments by source/timeline length.
        basis = source_video_duration if source_video_duration > 0 else timeline_duration
        if basis > 0:
            safe_max_segments = max(6, min(240, int(math.ceil(basis / max(2.5, safe_min_segment * 0.9)))))
        elif normalized_clips:
            safe_max_segments = max(6, min(240, len(normalized_clips) * 2))
        else:
            safe_max_segments = 80
    else:
        safe_max_segments = max(1, min(240, raw_max_segments))

    raw_target_duration = _safe_float(req.target_duration, 0.0)
    if raw_target_duration <= 0:
        basis = source_video_duration if source_video_duration > 0 else timeline_duration
        if basis > 0:
            safe_target_duration = max(12.0, min(10800.0, basis))
        elif normalized_clips:
            clips_total = sum(
                max(0.0, _safe_float(c.get("end", 0.0), 0.0) - _safe_float(c.get("start", 0.0), 0.0))
                for c in normalized_clips
            )
            safe_target_duration = max(12.0, min(10800.0, clips_total or 1200.0))
        else:
            safe_target_duration = 1200.0
    else:
        safe_target_duration = max(12.0, min(10800.0, raw_target_duration))

    planned_segments: List[Dict[str, Any]] = []
    transcript_segments = transcript_data.get("segments") if isinstance(transcript_data, dict) else []
    has_transcript_segments = isinstance(transcript_segments, list) and len(transcript_segments) > 0

    if requested_source_mode == "semantic":
        if not source_video_path:
            warnings.append("No se encontró video fuente completo para highlight semántico; se usará modo clips.")
        elif not has_transcript_segments:
            warnings.append("No hay transcript segmentado para highlight semántico; se usará modo clips.")
        else:
            planned_segments, semantic_context = _select_highlight_reel_segments_semantic(
                job_id=req.job_id,
                metadata_path=metadata_path,
                transcript=transcript_data if isinstance(transcript_data, dict) else {},
                clips=normalized_clips,
                max_segments=safe_max_segments,
                target_duration=safe_target_duration,
                min_segment_seconds=safe_min_segment,
                max_segment_seconds=safe_max_segment,
                min_gap_seconds=safe_min_gap
            )
            if planned_segments:
                selection_mode = "semantic"
                for segment in planned_segments:
                    segment["source_file_path"] = source_video_path
                    segment["segment_source_start"] = _safe_float(segment.get("segment_local_start", 0.0), 0.0)
                    segment["segment_source_end"] = _safe_float(
                        segment.get("segment_local_end", segment.get("segment_local_end_cap", 0.0)),
                        0.0
                    )
            else:
                warnings.append("No hubo suficientes momentos semánticos; se usará modo clips.")

    if selection_mode != "semantic":
        if not normalized_clips:
            raise HTTPException(
                status_code=400,
                detail="No hay clips disponibles para modo legacy y tampoco fue posible usar modo semántico."
            )
        planned_segments = _select_highlight_reel_segments(
            normalized_clips=normalized_clips,
            output_dir=output_dir,
            max_segments=safe_max_segments,
            target_duration=safe_target_duration,
            min_segment_seconds=safe_min_segment,
            max_segment_seconds=safe_max_segment,
            min_gap_seconds=safe_min_gap
        )
        if not planned_segments:
            raise HTTPException(status_code=400, detail="No se encontraron momentos suficientes para componer el highlight reel")

    ts = int(time.time())
    reel_id = f"hl_{ts}_{uuid.uuid4().hex[:8]}"
    out_name = f"highlight_reel_{ts}.mp4"
    out_path = os.path.join(output_dir, out_name)
    concat_file = os.path.join(output_dir, f"temp_highlight_concat_{ts}_{uuid.uuid4().hex[:6]}.txt")
    temp_segments: List[str] = []
    stitched_segments: List[Dict[str, Any]] = []

    try:
        for i, item in enumerate(planned_segments):
            seg_path = os.path.join(output_dir, f"temp_highlight_seg_{ts}_{i+1}.mp4")
            segment_source_path = str(item.get("source_file_path") or "").strip()
            if not segment_source_path and selection_mode == "semantic":
                segment_source_path = str(source_video_path or "").strip()
            if not segment_source_path or not os.path.exists(segment_source_path):
                warnings.append(f"Segmento {i + 1} no tiene fuente de video válida y se omitió.")
                continue

            start_local = max(
                0.0,
                _safe_float(
                    item.get("segment_source_start", item.get("segment_local_start", 0.0)),
                    0.0
                )
            )
            end_local = max(
                start_local + 0.08,
                _safe_float(
                    item.get("segment_source_end", item.get("segment_local_end", start_local + 0.08)),
                    start_local + 0.08
                )
            )
            cut_cmd = [
                "ffmpeg", "-y",
                "-ss", f"{start_local:.3f}",
                "-to", f"{end_local:.3f}",
                "-i", segment_source_path,
                "-vf", segment_vf,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "fast",
                "-c:a", "aac",
                "-movflags", "+faststart",
                seg_path
            ]
            cut_run = subprocess.run(cut_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if cut_run.returncode != 0 or not os.path.exists(seg_path) or os.path.getsize(seg_path) <= 0:
                warnings.append(f"Segmento {i + 1} no pudo renderizarse y se omitio.")
                continue

            dur = _probe_media_duration_seconds(seg_path)
            if dur <= 0.4:
                warnings.append(f"Segmento {i + 1} quedo demasiado corto y se omitio.")
                try:
                    os.remove(seg_path)
                except Exception:
                    pass
                continue

            temp_segments.append(seg_path)
            stitched_segments.append({
                "order": len(stitched_segments) + 1,
                "clip_index": int(item["clip_index"]),
                "virality_score": int(round(_safe_float(item.get("score", 0.0), 0.0))),
                "selection_mode": selection_mode,
                "clip_start": round(_safe_float(item.get("timeline_start", 0.0), 0.0), 3),
                "clip_end": round(_safe_float(item.get("timeline_end", 0.0), 0.0), 3),
                "segment_start_in_clip": round(start_local, 3),
                "segment_end_in_clip": round(end_local, 3),
                "segment_start_in_source": round(start_local, 3),
                "segment_end_in_source": round(end_local, 3),
                "duration": round(dur, 3),
                "title": item.get("title", ""),
                "transcript_excerpt": item.get("transcript_excerpt", ""),
                "seed_query": item.get("seed_query", None)
            })

        if len(temp_segments) < 1:
            raise HTTPException(status_code=400, detail="No se pudieron renderizar segmentos para el highlight reel")

        with open(concat_file, "w", encoding="utf-8") as f:
            for path in temp_segments:
                f.write(f"file {_ffmpeg_concat_escape(path)}\n")

        concat_cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-preset", "fast",
            "-c:a", "aac",
            "-movflags", "+faststart",
            out_path
        ]
        concat_run = subprocess.run(concat_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if concat_run.returncode != 0 or not os.path.exists(out_path):
            raise HTTPException(status_code=500, detail=concat_run.stderr.decode("utf-8", errors="ignore") or "Failed to build highlight reel")
    finally:
        if os.path.exists(concat_file):
            try:
                os.remove(concat_file)
            except Exception:
                pass
        for path in temp_segments:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

    final_duration = _probe_media_duration_seconds(out_path)
    highlight_entry = {
        "reel_id": reel_id,
        "created_at": int(time.time()),
        "video_url": f"/videos/{req.job_id}/{out_name}",
        "duration": round(final_duration, 3),
        "segments_count": len(stitched_segments),
        "segments": stitched_segments,
        "settings": {
            "aspect_ratio": target_aspect_ratio,
            "max_segments": safe_max_segments,
            "target_duration": round(safe_target_duration, 3),
            "min_segment_seconds": round(safe_min_segment, 3),
            "max_segment_seconds": round(safe_max_segment, 3),
            "min_gap_seconds": round(safe_min_gap, 3),
            "source_mode_requested": requested_source_mode,
            "selection_mode": selection_mode,
            "source_resolution": source_path_reason
        },
        "warnings": warnings
    }
    if selection_mode == "semantic":
        highlight_entry["semantic_context"] = {
            "provider": semantic_context.get("provider", "local"),
            "queries": semantic_context.get("queries", []),
            "candidates": int(_safe_float(semantic_context.get("candidates", 0), 0)),
            "selected": int(_safe_float(semantic_context.get("selected", 0), 0)),
        }
    if len(stitched_segments) == 1:
        highlight_entry["warnings"].append("Solo se detectó 1 momento utilizable; se generó highlight reel de 1 segmento.")

    if not isinstance(metadata.get("highlight_reels"), list):
        metadata["highlight_reels"] = []
    metadata["highlight_reels"].append(highlight_entry)
    metadata["highlight_reels"] = metadata["highlight_reels"][-20:]
    try:
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    if isinstance(job, dict):
        if not isinstance(job.get("result"), dict):
            job["result"] = {"clips": normalized_clips}
        result_payload = job["result"]
        if not isinstance(result_payload.get("highlight_reels"), list):
            result_payload["highlight_reels"] = []
        result_payload["highlight_reels"].append(highlight_entry)
        result_payload["highlight_reels"] = result_payload["highlight_reels"][-20:]
        result_payload["latest_highlight_reel"] = highlight_entry
        _persist_job_state(req.job_id)

    return {
        "success": True,
        "reel": highlight_entry,
        "reel_url": highlight_entry["video_url"],
        "aspect_ratio": target_aspect_ratio,
        "warnings": warnings
    }

@app.get("/api/transcript/{job_id}")
async def get_transcript_segments(
    job_id: str,
    q: Optional[str] = None,
    limit: int = 800,
    offset: int = 0,
    include_words: bool = False
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
        row = {
            "segment_index": idx,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(max(0.0, end - start), 3),
            "speaker": speaker or None,
            "word_count": len(words),
            "text": text
        }
        if include_words:
            safe_words: List[Dict[str, Any]] = []
            for word_item in words:
                if not isinstance(word_item, dict):
                    continue
                ws = max(0.0, _safe_float(word_item.get("start", start), start))
                we = _safe_float(word_item.get("end", ws), ws)
                if we < ws:
                    we = ws
                wt = _normalize_space(word_item.get("word", "")) or _normalize_space(word_item.get("text", ""))
                safe_words.append({
                    "start": round(ws, 3),
                    "end": round(we, 3),
                    "word": wt
                })
            row["words"] = safe_words
        normalized.append(row)

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
        "include_words": bool(include_words),
        "has_speaker_labels": has_speaker_labels,
        "segments": paged
    }

@app.get("/api/waveform/{job_id}/{clip_index}")
async def get_clip_waveform(
    job_id: str,
    clip_index: int,
    input_filename: Optional[str] = None,
    buckets: int = 240
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

    clips = data.get("shorts", [])
    if not isinstance(clips, list) or clip_index < 0 or clip_index >= len(clips):
        raise HTTPException(status_code=404, detail="Clip not found")

    clip_data = clips[clip_index] or {}
    if input_filename:
        filename = _safe_input_filename(input_filename)
    else:
        filename = _safe_input_filename(clip_data.get("video_url", ""))
        if not filename:
            base_name = os.path.basename(json_files[0]).replace("_metadata.json", "")
            filename = f"{base_name}_clip_{clip_index + 1}.mp4"

    media_path = os.path.join(output_dir, filename)
    if not os.path.exists(media_path):
        raise HTTPException(status_code=404, detail=f"Video file not found: {media_path}")

    safe_buckets = max(60, min(1200, int(buckets)))
    try:
        peaks = _extract_waveform_peaks(media_path, buckets=safe_buckets, sample_rate=11025)
        duration = _probe_media_duration_seconds(media_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Waveform extraction failed: {e}")

    return {
        "job_id": job_id,
        "clip_index": clip_index,
        "filename": filename,
        "duration": round(duration, 3),
        "buckets": len(peaks),
        "peaks": peaks
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
    job = _ensure_job_context(req.job_id)
    if not isinstance(job, dict):
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
    semantic_api_key = x_gemini_key
    if not semantic_api_key:
        # For semantic search we have a silent failure/bypass if key is missing to not block whole app, 
        # but the user asked for strictness, so let's at least log it or raise if it's a critical path.
        # But for /api/search usually we want an error if they expect AI search.
        raise HTTPException(status_code=400, detail="Gemini API Key is required for semantic search.")

    clips = []
    if isinstance(data.get("shorts"), list):
        clips = data.get("shorts") or []
    if not clips:
        job_result = job.get("result", {}) if isinstance(job, dict) else {}
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
    job = _ensure_job_context(req.job_id)
    if not isinstance(job, dict):
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

def _append_social_post_event(
    job_id: str,
    clip_index: int,
    platforms: List[str],
    status: str,
    status_code: int,
    detail: str = "",
    vendor_payload: Optional[Dict[str, Any]] = None
):
    job = _ensure_job_context(job_id)
    if not isinstance(job, dict):
        return

    event = {
        "timestamp": int(time.time()),
        "clip_index": int(max(0, clip_index)),
        "platforms": [str(p) for p in (platforms or [])],
        "status": str(status or "unknown"),
        "status_code": int(status_code or 0),
        "detail": str(detail or "")[:400],
        "vendor_payload": vendor_payload if isinstance(vendor_payload, dict) else None,
    }
    posts = job.setdefault("social_posts", [])
    if not isinstance(posts, list):
        posts = []
        job["social_posts"] = posts
    posts.append(event)
    job["social_posts"] = posts[-300:]
    _persist_job_state(job_id)

    metadata_path = _metadata_path_for_job(job_id)
    if metadata_path:
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            events = metadata.get("social_posts", [])
            if not isinstance(events, list):
                events = []
            events.append(event)
            metadata["social_posts"] = events[-300:]
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

def _compute_social_metrics(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    safe_events = [e for e in events if isinstance(e, dict)]
    total = len(safe_events)
    success = sum(1 for e in safe_events if str(e.get("status", "")).lower() == "success")
    failed = total - success

    by_platform: Dict[str, Dict[str, int]] = {
        "tiktok": {"attempted": 0, "success": 0, "failed": 0},
        "instagram": {"attempted": 0, "success": 0, "failed": 0},
        "youtube": {"attempted": 0, "success": 0, "failed": 0},
    }
    for event in safe_events:
        platforms = event.get("platforms", [])
        if not isinstance(platforms, list):
            continue
        is_ok = str(event.get("status", "")).lower() == "success"
        for platform in platforms:
            key = str(platform or "").strip().lower()
            if key not in by_platform:
                by_platform[key] = {"attempted": 0, "success": 0, "failed": 0}
            by_platform[key]["attempted"] += 1
            if is_ok:
                by_platform[key]["success"] += 1
            else:
                by_platform[key]["failed"] += 1

    return {
        "total_attempts": total,
        "successful_attempts": success,
        "failed_attempts": failed,
        "success_rate": round((success / total), 4) if total > 0 else 0.0,
        "by_platform": by_platform,
    }

@app.post("/api/social/post")
async def post_to_socials(req: SocialPostRequest):
    job = _ensure_job_context(req.job_id)
    if not isinstance(job, dict):
        raise HTTPException(status_code=404, detail="Job not found")

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

        try:
            vendor_payload = response.json()
        except Exception:
            vendor_payload = {"raw_response": response.text[:1500]}
        _append_social_post_event(
            job_id=req.job_id,
            clip_index=req.clip_index,
            platforms=req.platforms,
            status="success",
            status_code=response.status_code,
            detail="Published/scheduled successfully",
            vendor_payload=vendor_payload if isinstance(vendor_payload, dict) else None
        )
        return vendor_payload

    except HTTPException as e:
        print(f"❌ Social Post HTTPException: {e.detail}")
        _append_social_post_event(
            job_id=req.job_id,
            clip_index=req.clip_index,
            platforms=req.platforms,
            status="failed",
            status_code=e.status_code,
            detail=str(e.detail)
        )
        raise

    except Exception as e:
        print(f"❌ Social Post Exception: {e}")
        _append_social_post_event(
            job_id=req.job_id,
            clip_index=req.clip_index,
            platforms=req.platforms,
            status="failed",
            status_code=500,
            detail=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/social/metrics/{job_id}")
async def get_social_metrics(job_id: str):
    job = _ensure_job_context(job_id)
    if not isinstance(job, dict):
        raise HTTPException(status_code=404, detail="Job not found")

    events: List[Dict[str, Any]] = []
    if isinstance(job.get("social_posts"), list):
        events.extend([e for e in job.get("social_posts", []) if isinstance(e, dict)])

    if not events:
        metadata_path = _metadata_path_for_job(job_id)
        if metadata_path:
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                metadata_events = metadata.get("social_posts", [])
                if isinstance(metadata_events, list):
                    events.extend([e for e in metadata_events if isinstance(e, dict)])
            except Exception:
                pass

    events = sorted(events, key=lambda e: int(_safe_float(e.get("timestamp", 0), 0)), reverse=True)
    metrics = _compute_social_metrics(events)
    return {
        "job_id": job_id,
        **metrics,
        "recent_events": events[:30]
    }

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
