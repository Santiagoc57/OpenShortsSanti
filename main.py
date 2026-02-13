import time
import cv2
import scenedetect
import subprocess
import argparse
import re
import sys
import math
import zlib
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from ultralytics import YOLO
import torch
import os
import numpy as np
from tqdm import tqdm
import yt_dlp
import mediapipe as mp
# import whisper (replaced by faster_whisper inside function)
from google import genai
from dotenv import load_dotenv
import json
from typing import List, Dict, Any, Optional, Tuple

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')

# Load environment variables
load_dotenv()

# --- Constants ---
ASPECT_RATIO_PRESETS = {
    "9:16": 9 / 16,
    "16:9": 16 / 9,
}
DEFAULT_ASPECT_RATIO = "9:16"


def normalize_aspect_ratio(raw_aspect_ratio):
    value = str(raw_aspect_ratio or DEFAULT_ASPECT_RATIO).strip().replace("/", ":")
    if value not in ASPECT_RATIO_PRESETS:
        raise ValueError(f"Invalid aspect ratio '{raw_aspect_ratio}'. Allowed values: 9:16, 16:9")
    return value, ASPECT_RATIO_PRESETS[value]


def _make_even(value):
    rounded = int(round(value))
    if rounded < 2:
        return 2
    return rounded if rounded % 2 == 0 else rounded - 1


def compute_output_dimensions(input_width, input_height, target_ratio):
    source_ratio = input_width / input_height
    if source_ratio >= target_ratio:
        out_height = input_height
        out_width = out_height * target_ratio
    else:
        out_width = input_width
        out_height = out_width / target_ratio
    return _make_even(out_width), _make_even(out_height)

GEMINI_PROMPT_TEMPLATE = """
You are a senior short-form video editor. Read the ENTIRE transcript and word-level timestamps to choose the 3–15 MOST VIRAL moments for TikTok/IG Reels/YouTube Shorts. Each clip must be between 15 and 60 seconds long.
{max_clips_rule}
{clip_length_rule}

⚠️ FFMPEG TIME CONTRACT — STRICT REQUIREMENTS:
- Return timestamps in ABSOLUTE SECONDS from the start of the video (usable in: ffmpeg -ss <start> -to <end> -i <input> ...).
- Only NUMBERS with decimal point, up to 3 decimals (examples: 0, 1.250, 17.350).
- Ensure 0 ≤ start < end ≤ VIDEO_DURATION_SECONDS.
- Each clip between 15 and 60 s (inclusive).
- Prefer starting 0.2–0.4 s BEFORE the hook and ending 0.2–0.4 s AFTER the payoff.
- Use silence moments for natural cuts; never cut in the middle of a word or phrase.
- STRICTLY FORBIDDEN to use time formats other than absolute seconds.

VIDEO_DURATION_SECONDS: {video_duration}

TRANSCRIPT_TEXT (raw):
{transcript_text}

WORDS_JSON (array of {{w, s, e}} where s/e are seconds):
{words_json}

STRICT EXCLUSIONS:
- No generic intros/outros or purely sponsorship segments unless they contain the hook.
- No clips < 15 s or > 60 s.

OUTPUT — RETURN ONLY VALID JSON (no markdown, no comments). Order clips by predicted performance (best to worst).
LANGUAGE RULE (STRICT): all textual fields MUST be in Spanish (español neutro): score_reason, topic_tags, video_description_for_tiktok, video_description_for_instagram, video_title_for_youtube_short.
In the descriptions, ALWAYS include a CTA in Spanish like "Sígueme y comenta X y te envío el workflow" (especially if discussing an n8n workflow):
{{
  "shorts": [
    {{
      "start": <number in seconds, e.g., 12.340>,
      "end": <number in seconds, e.g., 37.900>,
      "virality_score": <integer 0-100, where 100 is highest predicted performance>,
      "selection_confidence": <number between 0 and 1 indicating confidence in this selection>,
      "score_reason": "<razón corta en español de por qué este clip puede rendir>",
      "topic_tags": ["<hasta 5 etiquetas cortas en español, sin #, ej: politica, debate, economia>"],
      "video_description_for_tiktok": "<descripción en español orientada a views para TikTok>",
      "video_description_for_instagram": "<descripción en español orientada a views para Instagram>",
      "video_title_for_youtube_short": "<título en español para YouTube Short orientado a views, máximo 100 caracteres>"
    }}
  ]
}}
"""

def clip_length_guidance(target):
    t = str(target or "").strip().lower()
    if t == "short":
        return "CLIP LENGTH PRIORITY: Prefer short, punchy clips in the 18-30s range whenever possible."
    if t == "long":
        return "CLIP LENGTH PRIORITY: Prefer contextual clips in the 40-60s range whenever possible."
    if t == "balanced":
        return "CLIP LENGTH PRIORITY: Prefer balanced clips in the 25-45s range."
    return ""

def _default_score_by_rank(rank):
    """Fallback score when model does not provide virality_score."""
    return max(55, 92 - (rank * 6))

def _normalize_clip_score(raw_score, rank):
    default = _default_score_by_rank(rank)
    try:
        score = int(round(float(raw_score)))
    except (TypeError, ValueError):
        return default
    return max(0, min(100, score))

def _score_band(score):
    if score >= 80:
        return "top"
    if score >= 65:
        return "medium"
    return "low"

def _normalize_confidence(raw_confidence, score):
    try:
        conf = float(raw_confidence)
    except (TypeError, ValueError):
        conf = score / 100.0
    return round(max(0.0, min(1.0, conf)), 2)

def _normalize_topic_tags(raw_tags):
    if isinstance(raw_tags, str):
        raw_tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
    if not isinstance(raw_tags, list):
        return []

    out = []
    seen = set()
    for tag in raw_tags:
        if not isinstance(tag, str):
            continue
        clean = tag.strip().lstrip("#").lower()
        if not clean:
            continue
        # Keep tags short and UI-friendly.
        clean = clean[:24]
        if clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
        if len(out) >= 5:
            break
    return out

def _default_topic_tags(clip):
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
    tags = []
    seen = set()
    for w in words:
        if w in stop or w in seen:
            continue
        seen.add(w)
        tags.append(w[:24])
        if len(tags) >= 3:
            break
    return tags

def normalize_shorts_payload(result_json):
    """
    Ensures each clip has stable scoring metadata for UI sorting:
    - virality_score: int [0,100]
    - score_reason: short string
    """
    if not isinstance(result_json, dict):
        return result_json

    shorts = result_json.get('shorts')
    if not isinstance(shorts, list):
        return result_json

    normalized = []
    for i, clip in enumerate(shorts):
        if not isinstance(clip, dict):
            continue
        clip['virality_score'] = _normalize_clip_score(clip.get('virality_score'), i)
        clip['score_band'] = _score_band(clip['virality_score'])
        clip['selection_confidence'] = _normalize_confidence(clip.get('selection_confidence'), clip['virality_score'])
        reason = clip.get('score_reason')
        if not reason:
            reason = f"Ranking IA #{i+1}: buen gancho inicial y alto potencial de retención."
        clip['score_reason'] = str(reason).strip()[:220]
        tags = _normalize_topic_tags(clip.get('topic_tags'))
        if not tags:
            tags = _default_topic_tags(clip)
        clip['topic_tags'] = tags
        normalized.append(clip)

    result_json['shorts'] = normalized
    return result_json


# --- Clip Post-Processing (Smart Boundaries + Semantic De-duplication) ---
DEFAULT_MIN_CLIP_SECONDS = 15.0
DEFAULT_MAX_CLIP_SECONDS = 60.0
SMART_START_PAD = 0.25
SMART_END_PAD = 0.30
SMART_LOOKBACK_SECONDS = 2.0
SMART_LOOKAHEAD_SECONDS = 2.0
SMART_PAUSE_GAP_SECONDS = 0.22
LOCAL_EMBED_DIM = 192
SEMANTIC_DEDUPE_SIM_THRESHOLD = 0.93
SEMANTIC_DEDUPE_OVERLAP_THRESHOLD = 0.35
SEMANTIC_DEDUPE_CENTER_WINDOW_SECONDS = 18.0


def _length_bounds_from_target(target: Optional[str]) -> Tuple[float, float]:
    t = str(target or "").strip().lower()
    if t == "short":
        return 18.0, 32.0
    if t == "long":
        return 38.0, 60.0
    if t == "balanced":
        return 24.0, 46.0
    return DEFAULT_MIN_CLIP_SECONDS, DEFAULT_MAX_CLIP_SECONDS


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_space(text):
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
    if n <= 0:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(a[i] * a[i] for i in range(n)))
    nb = math.sqrt(sum(b[i] * b[i] for i in range(n)))
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)


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
                gram = w[i:i + n]
                idx = zlib.crc32(gram.encode("utf-8")) % dim
                vec[idx] += weight

    for i in range(len(words) - 1):
        bigram = f"{words[i]}_{words[i+1]}"
        idx = zlib.crc32(bigram.encode("utf-8")) % dim
        vec[idx] += 0.7

    return _normalize_vector(vec)


def _extract_transcript_words(transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
    words: List[Dict[str, Any]] = []
    segments = transcript.get("segments", []) if isinstance(transcript, dict) else []
    if not isinstance(segments, list):
        return words

    for seg_idx, segment in enumerate(segments):
        if not isinstance(segment, dict):
            continue
        seg_start = max(0.0, _safe_float(segment.get("start", 0.0), 0.0))
        seg_end = max(seg_start, _safe_float(segment.get("end", seg_start), seg_start))
        seg_text = _normalize_space(segment.get("text", ""))
        seg_words = segment.get("words", []) if isinstance(segment.get("words"), list) else []

        if seg_words:
            for w in seg_words:
                if not isinstance(w, dict):
                    continue
                token = _normalize_space(w.get("word", ""))
                if not token:
                    continue
                ws = max(0.0, _safe_float(w.get("start", seg_start), seg_start))
                we = max(ws, _safe_float(w.get("end", ws), ws))
                words.append({
                    "word": token,
                    "start": ws,
                    "end": we,
                    "segment_index": seg_idx
                })
            continue

        # Fallback if no word timestamps were produced.
        tokens = re.findall(r"\S+", seg_text)
        if not tokens:
            continue
        seg_duration = max(0.001, seg_end - seg_start)
        slot = seg_duration / max(1, len(tokens))
        for token_idx, token in enumerate(tokens):
            ws = seg_start + (token_idx * slot)
            we = min(seg_end, ws + slot)
            words.append({
                "word": token,
                "start": ws,
                "end": we,
                "segment_index": seg_idx
            })

    words.sort(key=lambda x: (x["start"], x["end"]))
    return words


def _build_boundary_points(words: List[Dict[str, Any]], duration: float) -> List[float]:
    points = {0.0}
    if duration > 0:
        points.add(float(duration))

    for i, w in enumerate(words):
        ws = max(0.0, _safe_float(w.get("start", 0.0), 0.0))
        we = max(ws, _safe_float(w.get("end", ws), ws))
        points.add(ws)
        points.add(we)
        token = str(w.get("word", ""))
        if re.search(r"[.!?;:]\s*$", token):
            points.add(we)

        if i < len(words) - 1:
            nw = words[i + 1]
            ns = max(0.0, _safe_float(nw.get("start", we), we))
            gap = ns - we
            if gap >= SMART_PAUSE_GAP_SECONDS:
                points.add(we + (gap / 2.0))

    out = sorted(max(0.0, min(duration, p)) if duration > 0 else max(0.0, p) for p in points)
    return [round(p, 3) for p in out]


def _closest_boundary(
    anchor: float,
    points: List[float],
    lower: float,
    upper: float,
    prefer: str
) -> float:
    lower = min(lower, upper)
    upper = max(lower, upper)
    candidates = [p for p in points if lower <= p <= upper]
    if not candidates:
        return max(lower, min(upper, anchor))

    best = candidates[0]
    best_score = float("inf")
    for p in candidates:
        score = abs(p - anchor)
        if prefer == "earlier" and p > anchor:
            score += 0.18
        elif prefer == "later" and p < anchor:
            score += 0.18
        if score < best_score:
            best = p
            best_score = score
    return best


def _enforce_clip_duration(
    start: float,
    end: float,
    duration: float,
    points: List[float],
    min_clip_seconds: float,
    max_clip_seconds: float
) -> Tuple[float, float]:
    start = max(0.0, start)
    end = max(start + 0.01, end)
    if duration > 0:
        end = min(duration, end)

    current = end - start
    if current < min_clip_seconds:
        target_end = start + min_clip_seconds
        if duration > 0:
            target_end = min(duration, target_end)
        end = _closest_boundary(
            anchor=target_end,
            points=points,
            lower=max(start + min_clip_seconds, target_end - 0.4),
            upper=min(duration if duration > 0 else target_end + 2.0, target_end + 2.0),
            prefer="later"
        )
        if end - start < min_clip_seconds:
            end = max(start + min_clip_seconds, end)
            if duration > 0:
                end = min(duration, end)

    current = end - start
    if current > max_clip_seconds:
        target_end = start + max_clip_seconds
        end = _closest_boundary(
            anchor=target_end,
            points=points,
            lower=max(start + min_clip_seconds, target_end - 2.0),
            upper=min(duration if duration > 0 else target_end, target_end),
            prefer="earlier"
        )
        if end - start > max_clip_seconds:
            end = start + max_clip_seconds
            if duration > 0:
                end = min(duration, end)

    if duration > 0 and end > duration:
        end = duration
    if end - start < min_clip_seconds:
        # Last-resort correction near tail of video.
        if duration > 0:
            start = max(0.0, duration - min_clip_seconds)
            end = duration
        else:
            end = start + min_clip_seconds

    return start, end


def _smart_refine_clip_range(
    start: float,
    end: float,
    duration: float,
    points: List[float],
    min_clip_seconds: float,
    max_clip_seconds: float
) -> Tuple[float, float]:
    start = max(0.0, _safe_float(start, 0.0))
    end = max(start + 0.01, _safe_float(end, start + min_clip_seconds))
    if duration > 0:
        end = min(duration, end)

    anchor_start = max(0.0, start - SMART_START_PAD)
    anchor_end = end + SMART_END_PAD
    if duration > 0:
        anchor_end = min(duration, anchor_end)

    refined_start = _closest_boundary(
        anchor=anchor_start,
        points=points,
        lower=max(0.0, anchor_start - SMART_LOOKBACK_SECONDS),
        upper=min(anchor_start + 1.0, duration if duration > 0 else anchor_start + 1.0),
        prefer="earlier"
    )
    refined_end = _closest_boundary(
        anchor=anchor_end,
        points=points,
        lower=max(refined_start + 0.5, anchor_end - 1.0),
        upper=min(anchor_end + SMART_LOOKAHEAD_SECONDS, duration if duration > 0 else anchor_end + SMART_LOOKAHEAD_SECONDS),
        prefer="later"
    )

    refined_start, refined_end = _enforce_clip_duration(
        refined_start,
        refined_end,
        duration,
        points,
        min_clip_seconds=min_clip_seconds,
        max_clip_seconds=max_clip_seconds
    )
    return round(refined_start, 3), round(refined_end, 3)


def _clip_overlap_ratio(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    a_len = max(0.001, a_end - a_start)
    b_len = max(0.001, b_end - b_start)
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    if inter <= 0:
        return 0.0
    return inter / min(a_len, b_len)


def _extract_clip_text_for_embedding(clip: Dict[str, Any], transcript: Dict[str, Any]) -> str:
    start = _safe_float(clip.get("start", 0.0), 0.0)
    end = _safe_float(clip.get("end", start), start)
    pieces: List[str] = []

    segments = transcript.get("segments", []) if isinstance(transcript, dict) else []
    if isinstance(segments, list):
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            ss = _safe_float(seg.get("start", 0.0), 0.0)
            se = _safe_float(seg.get("end", ss), ss)
            if se <= start or ss >= end:
                continue
            text = _normalize_space(seg.get("text", ""))
            if text:
                pieces.append(text)

    if not pieces:
        pieces = [
            _normalize_space(clip.get("video_title_for_youtube_short", "")),
            _normalize_space(clip.get("video_description_for_tiktok", "")),
            _normalize_space(clip.get("video_description_for_instagram", "")),
            " ".join(clip.get("topic_tags", []) or [])
        ]
    return _normalize_space(" ".join(pieces))


def _semantic_deduplicate_shorts(shorts: List[Dict[str, Any]], transcript: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not shorts:
        return [], {"removed": 0, "kept": 0}

    ranked = list(shorts)
    ranked.sort(
        key=lambda c: (
            int(_safe_float(c.get("virality_score", 0.0), 0.0)),
            _safe_float(c.get("selection_confidence", 0.0), 0.0),
            -_safe_float(c.get("_original_rank", 0.0), 0.0)
        ),
        reverse=True
    )

    kept: List[Dict[str, Any]] = []
    removed_items: List[Dict[str, Any]] = []

    for clip in ranked:
        text = _extract_clip_text_for_embedding(clip, transcript)
        vec = _local_semantic_embedding(text)

        c_start = _safe_float(clip.get("start", 0.0), 0.0)
        c_end = _safe_float(clip.get("end", c_start), c_start)
        c_center = (c_start + c_end) / 2.0

        is_duplicate = False
        duplicate_of = None
        for kept_clip in kept:
            k_start = _safe_float(kept_clip.get("start", 0.0), 0.0)
            k_end = _safe_float(kept_clip.get("end", k_start), k_start)
            k_center = (k_start + k_end) / 2.0
            overlap = _clip_overlap_ratio(c_start, c_end, k_start, k_end)
            center_dist = abs(c_center - k_center)
            sim = _cosine_similarity(vec, kept_clip.get("_semantic_vec", []))

            if sim >= SEMANTIC_DEDUPE_SIM_THRESHOLD and (
                overlap >= SEMANTIC_DEDUPE_OVERLAP_THRESHOLD or center_dist <= SEMANTIC_DEDUPE_CENTER_WINDOW_SECONDS
            ):
                is_duplicate = True
                duplicate_of = int(kept_clip.get("_original_rank", 0)) + 1
                break

        clip["_semantic_vec"] = vec
        if is_duplicate:
            removed_items.append({
                "clip_rank": int(clip.get("_original_rank", 0)) + 1,
                "duplicate_of": duplicate_of
            })
            continue

        kept.append(clip)

    for clip in kept:
        clip.pop("_semantic_vec", None)
    return kept, {
        "removed": len(removed_items),
        "kept": len(kept),
        "removed_items": removed_items[:20]
    }


def postprocess_shorts_with_transcript(
    clips_data: Dict[str, Any],
    transcript: Dict[str, Any],
    duration: float,
    max_clips: Optional[int] = None,
    clip_length_target: Optional[str] = None
) -> Dict[str, Any]:
    if not isinstance(clips_data, dict):
        return clips_data
    shorts = clips_data.get("shorts", [])
    if not isinstance(shorts, list) or not shorts:
        return clips_data

    words = _extract_transcript_words(transcript or {})
    points = _build_boundary_points(words, duration)
    if not points:
        points = [0.0, round(max(0.0, duration), 3)]
    min_clip_seconds, max_clip_seconds = _length_bounds_from_target(clip_length_target)

    prepared: List[Dict[str, Any]] = []
    refined_count = 0
    for i, clip in enumerate(shorts):
        if not isinstance(clip, dict):
            continue
        item = dict(clip)
        item["_original_rank"] = i

        raw_start = max(0.0, _safe_float(item.get("start", 0.0), 0.0))
        raw_end = _safe_float(item.get("end", raw_start + min_clip_seconds), raw_start + min_clip_seconds)
        if duration > 0:
            raw_end = min(duration, raw_end)
        if raw_end <= raw_start:
            raw_end = raw_start + min_clip_seconds
            if duration > 0:
                raw_end = min(duration, raw_end)

        refined_start, refined_end = _smart_refine_clip_range(
            raw_start,
            raw_end,
            duration,
            points,
            min_clip_seconds=min_clip_seconds,
            max_clip_seconds=max_clip_seconds
        )
        if abs(refined_start - raw_start) >= 0.05 or abs(refined_end - raw_end) >= 0.05:
            refined_count += 1

        item["start"] = refined_start
        item["end"] = refined_end
        prepared.append(item)

    deduped, dedupe_report = _semantic_deduplicate_shorts(prepared, transcript or {})
    deduped.sort(
        key=lambda c: (
            int(_safe_float(c.get("virality_score", 0.0), 0.0)),
            _safe_float(c.get("selection_confidence", 0.0), 0.0),
            -_safe_float(c.get("_original_rank", 0.0), 0.0)
        ),
        reverse=True
    )

    if max_clips:
        deduped = deduped[:max(1, int(max_clips))]

    for item in deduped:
        item.pop("_original_rank", None)

    out = dict(clips_data)
    out["shorts"] = deduped
    out["postprocess"] = {
        "smart_boundaries": {
            "enabled": True,
            "clips_refined": refined_count,
            "boundary_points": len(points),
            "word_timestamps": len(words),
            "target_profile": clip_length_target or "default",
            "target_min_seconds": round(min_clip_seconds, 2),
            "target_max_seconds": round(max_clip_seconds, 2)
        },
        "semantic_dedupe": {
            "enabled": True,
            "removed_duplicates": int(dedupe_report.get("removed", 0)),
            "kept_clips": int(dedupe_report.get("kept", len(deduped))),
            "similarity_threshold": SEMANTIC_DEDUPE_SIM_THRESHOLD
        }
    }
    return out

# Load the YOLO model once (Keep for backup or scene analysis if needed)
model = YOLO('yolov8n.pt')

# --- MediaPipe Setup ---
# Use standard Face Detection (BlazeFace) for speed
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

class SmoothedCameraman:
    """
    Handles smooth camera movement.
    Simplified Logic: "Heavy Tripod"
    Only moves if the subject leaves the center safe zone.
    Moves slowly and linearly.
    """
    def __init__(self, output_width, output_height, video_width, video_height, aspect_ratio=ASPECT_RATIO_PRESETS[DEFAULT_ASPECT_RATIO]):
        self.output_width = output_width
        self.output_height = output_height
        self.video_width = video_width
        self.video_height = video_height
        self.aspect_ratio = aspect_ratio
        
        # Initial State
        self.current_center_x = video_width / 2
        self.target_center_x = video_width / 2
        
        # Calculate crop dimensions once
        self.crop_height = video_height
        self.crop_width = int(round(self.crop_height * self.aspect_ratio))
        if self.crop_width > video_width:
             self.crop_width = video_width
             self.crop_height = int(round(self.crop_width / self.aspect_ratio))
             
        # Safe Zone: 20% of the video width
        # As long as the target is within this zone relative to current center, DO NOT MOVE.
        self.safe_zone_radius = self.crop_width * 0.25

    def update_target(self, face_box):
        """
        Updates the target center based on detected face/person.
        """
        if face_box:
            x, y, w, h = face_box
            self.target_center_x = x + w / 2
    
    def get_crop_box(self, force_snap=False):
        """
        Returns the (x1, y1, x2, y2) for the current frame.
        """
        if force_snap:
            self.current_center_x = self.target_center_x
        else:
            diff = self.target_center_x - self.current_center_x
            
            # SIMPLIFIED LOGIC:
            # 1. Is the target outside the safe zone?
            if abs(diff) > self.safe_zone_radius:
                # 2. If yes, move towards it slowly (Linear Speed)
                # Determine direction
                direction = 1 if diff > 0 else -1
                
                # Speed: 2 pixels per frame (Slow pan)
                # If the distance is HUGE (scene change or fast movement), speed up slightly
                if abs(diff) > self.crop_width * 0.5:
                    speed = 15.0 # Fast re-frame
                else:
                    speed = 3.0  # Slow, steady pan
                
                self.current_center_x += direction * speed
                
                # Check if we overshot (prevent oscillation)
                new_diff = self.target_center_x - self.current_center_x
                if (direction == 1 and new_diff < 0) or (direction == -1 and new_diff > 0):
                    self.current_center_x = self.target_center_x
            
            # If inside safe zone, DO NOTHING (Stationary Camera)
                
        # Clamp center
        half_crop = self.crop_width / 2
        
        if self.current_center_x - half_crop < 0:
            self.current_center_x = half_crop
        if self.current_center_x + half_crop > self.video_width:
            self.current_center_x = self.video_width - half_crop
            
        x1 = int(self.current_center_x - half_crop)
        x2 = int(self.current_center_x + half_crop)
        
        x1 = max(0, x1)
        x2 = min(self.video_width, x2)
        
        y1 = int((self.video_height - self.crop_height) / 2)
        y2 = y1 + self.crop_height

        return x1, y1, x2, y2

class SpeakerTracker:
    """
    Tracks speakers over time to prevent rapid switching and handle temporary obstructions.
    """
    def __init__(self, stabilization_frames=15, cooldown_frames=30):
        self.active_speaker_id = None
        self.speaker_scores = {}  # {id: score}
        self.last_seen = {}       # {id: frame_number}
        self.locked_counter = 0   # How long we've been locked on current speaker
        
        # Hyperparameters
        self.stabilization_threshold = stabilization_frames # Frames needed to confirm a new speaker
        self.switch_cooldown = cooldown_frames              # Minimum frames before switching again
        self.last_switch_frame = -1000
        
        # ID tracking
        self.next_id = 0
        self.known_faces = [] # [{'id': 0, 'center': x, 'last_frame': 123}]

    def get_target(self, face_candidates, frame_number, width):
        """
        Decides which face to focus on.
        face_candidates: list of {'box': [x,y,w,h], 'score': float}
        """
        current_candidates = []
        
        # 1. Match faces to known IDs (simple distance tracking)
        for face in face_candidates:
            x, y, w, h = face['box']
            center_x = x + w / 2
            
            best_match_id = -1
            min_dist = width * 0.15 # Reduced matching radius to avoid jumping in groups
            
            # Try to match with known faces seen recently
            for kf in self.known_faces:
                if frame_number - kf['last_frame'] > 30: # Forgot faces older than 1s (was 2s)
                    continue
                    
                dist = abs(center_x - kf['center'])
                if dist < min_dist:
                    min_dist = dist
                    best_match_id = kf['id']
            
            # If no match, assign new ID
            if best_match_id == -1:
                best_match_id = self.next_id
                self.next_id += 1
            
            # Update known face
            self.known_faces = [kf for kf in self.known_faces if kf['id'] != best_match_id]
            self.known_faces.append({'id': best_match_id, 'center': center_x, 'last_frame': frame_number})
            
            current_candidates.append({
                'id': best_match_id,
                'box': face['box'],
                'score': face['score']
            })

        # 2. Update Scores with decay
        for pid in list(self.speaker_scores.keys()):
             self.speaker_scores[pid] *= 0.85 # Faster decay (was 0.9)
             if self.speaker_scores[pid] < 0.1:
                 del self.speaker_scores[pid]

        # Add new scores
        for cand in current_candidates:
            pid = cand['id']
            # Score is purely based on size (proximity) now that we don't have mouth
            raw_score = cand['score'] / (width * width * 0.05)
            self.speaker_scores[pid] = self.speaker_scores.get(pid, 0) + raw_score

        # 3. Determine Best Speaker
        if not current_candidates:
            # If no one found, maintain last active speaker if cooldown allows
            # to avoid black screen or jump to 0,0
            return None 
            
        best_candidate = None
        max_score = -1
        
        for cand in current_candidates:
            pid = cand['id']
            total_score = self.speaker_scores.get(pid, 0)
            
            # Hysteresis: HUGE Bonus for current active speaker
            if pid == self.active_speaker_id:
                total_score *= 3.0 # Sticky factor
                
            if total_score > max_score:
                max_score = total_score
                best_candidate = cand

        # 4. Decide Switch
        if best_candidate:
            target_id = best_candidate['id']
            
            if target_id == self.active_speaker_id:
                self.locked_counter += 1
                return best_candidate['box']
            
            # New person
            if frame_number - self.last_switch_frame < self.switch_cooldown:
                old_cand = next((c for c in current_candidates if c['id'] == self.active_speaker_id), None)
                if old_cand:
                    return old_cand['box']
            
            self.active_speaker_id = target_id
            self.last_switch_frame = frame_number
            self.locked_counter = 0
            return best_candidate['box']
            
        return None

def detect_face_candidates(frame):
    """
    Returns list of all detected faces using lightweight FaceDetection.
    """
    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    
    candidates = []
    
    if not results.detections:
        return []
        
    for detection in results.detections:
        bboxC = detection.location_data.relative_bounding_box
        x = int(bboxC.xmin * width)
        y = int(bboxC.ymin * height)
        w = int(bboxC.width * width)
        h = int(bboxC.height * height)
        
        candidates.append({
            'box': [x, y, w, h],
            'score': w * h # Area as score
        })
            
    return candidates

def detect_person_yolo(frame):
    """
    Fallback: Detect largest person using YOLO when face detection fails.
    Returns [x, y, w, h] of the person's 'upper body' approximation.
    """
    # Use the globally loaded model
    results = model(frame, verbose=False, classes=[0]) # class 0 is person
    
    if not results:
        return None
        
    best_box = None
    max_area = 0
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
            w = x2 - x1
            h = y2 - y1
            area = w * h
            
            if area > max_area:
                max_area = area
                # Focus on the top 40% of the person (head/chest) for framing
                # This approximates where the face is if we can't detect it directly
                face_h = int(h * 0.4)
                best_box = [x1, y1, w, face_h]
                
    return best_box

def create_general_frame(frame, output_width, output_height):
    """
    Creates a 'General Shot' frame: 
    - Background: Blurred zoom of original
    - Foreground: Original video scaled to fit width, centered vertically.
    """
    orig_h, orig_w = frame.shape[:2]
    
    # 1. Background (Cover target canvas and crop center)
    bg_scale = max(output_width / orig_w, output_height / orig_h)
    bg_w = int(orig_w * bg_scale)
    bg_h = int(orig_h * bg_scale)
    bg_resized = cv2.resize(frame, (bg_w, bg_h))

    # Crop center of background
    start_x = (bg_w - output_width) // 2
    start_y = (bg_h - output_height) // 2
    if start_x < 0: start_x = 0
    if start_y < 0: start_y = 0
    background = bg_resized[start_y:start_y+output_height, start_x:start_x+output_width]
    if background.shape[0] != output_height or background.shape[1] != output_width:
        background = cv2.resize(background, (output_width, output_height))
        
    # Blur background
    background = cv2.GaussianBlur(background, (51, 51), 0)
    
    # 2. Foreground (Contain)
    scale = min(output_width / orig_w, output_height / orig_h)
    fg_w = int(orig_w * scale)
    fg_h = int(orig_h * scale)
    foreground = cv2.resize(frame, (fg_w, fg_h))
    
    # 3. Overlay
    x_offset = (output_width - fg_w) // 2
    y_offset = (output_height - fg_h) // 2

    # Clone background to avoid modifying it
    final_frame = background.copy()
    final_frame[y_offset:y_offset+fg_h, x_offset:x_offset+fg_w] = foreground

    return final_frame

def analyze_scenes_strategy(video_path, scenes):
    """
    Analyzes each scene to determine if it should be TRACK (Single person) or GENERAL (Group/Wide).
    Returns list of strategies corresponding to scenes.
    """
    cap = cv2.VideoCapture(video_path)
    strategies = []
    
    if not cap.isOpened():
        return ['TRACK'] * len(scenes)
        
    for start, end in tqdm(scenes, desc="   Analyzing Scenes"):
        # Sample 3 frames (start, middle, end)
        frames_to_check = [
            start.get_frames() + 5,
            int((start.get_frames() + end.get_frames()) / 2),
            end.get_frames() - 5
        ]
        
        face_counts = []
        for f_idx in frames_to_check:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret: continue
            
            # Detect faces
            candidates = detect_face_candidates(frame)
            face_counts.append(len(candidates))
            
        # Decision Logic
        if not face_counts:
            avg_faces = 0
        else:
            avg_faces = sum(face_counts) / len(face_counts)
            
        # Strategy:
        # 0 faces -> GENERAL (Landscape/B-roll)
        # 1 face -> TRACK
        # > 1.2 faces -> GENERAL (Group)
        
        if avg_faces > 1.2 or avg_faces < 0.5:
            strategies.append('GENERAL')
        else:
            strategies.append('TRACK')
            
    cap.release()
    return strategies

def detect_scenes(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    fps = video_manager.get_framerate()
    video_manager.release()
    return scene_list, fps

def get_video_resolution(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video file {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


def sanitize_filename(filename):
    """Remove invalid characters from filename."""
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    filename = filename.replace(' ', '_')
    return filename[:100]

def is_audio_input(path):
    audio_exts = {'.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg', '.opus', '.wma'}
    return os.path.splitext(path.lower())[1] in audio_exts

def build_audio_canvas_video(input_audio, output_video, ffmpeg_preset="fast", ffmpeg_crf=23, aspect_ratio=DEFAULT_ASPECT_RATIO):
    """
    Creates a vertical visual canvas from an audio file using ffmpeg waveform rendering.
    This allows reusing the same clipping + vertical pipeline for audio-only podcasts.
    """
    aspect_label, _ = normalize_aspect_ratio(aspect_ratio)
    if aspect_label == "16:9":
        canvas_w, canvas_h = 1920, 1080
    else:
        canvas_w, canvas_h = 1080, 1920

    wave_w = _make_even(max(360, int(canvas_w * 0.9)))
    wave_h = _make_even(max(180, int(canvas_h * 0.27)))
    filter_complex = (
        f"color=c=0x0f1117:s={canvas_w}x{canvas_h}[bg];"
        f"[0:a]showwaves=s={wave_w}x{wave_h}:mode=line:colors=0x3b82f6,format=rgba[sw];"
        "[bg][sw]overlay=(W-w)/2:(H-h)/2,format=yuv420p[v]"
    )

    command = [
        'ffmpeg', '-y',
        '-i', input_audio,
        '-filter_complex', filter_complex,
        '-map', '[v]',
        '-map', '0:a',
        '-c:v', 'libx264',
        '-preset', str(ffmpeg_preset),
        '-crf', str(ffmpeg_crf),
        '-c:a', 'aac',
        '-shortest',
        '-movflags', '+faststart',
        output_video
    ]
    res = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if res.returncode != 0:
        print("❌ Failed to generate audio canvas video.")
        try:
            print(res.stderr.decode())
        except Exception:
            pass
        return False
    return True


def download_youtube_video(url, output_dir="."):
    """
    Downloads a YouTube video using yt-dlp.
    Returns the path to the downloaded video and the video title.
    """
    print(f"🔍 Debug: yt-dlp version: {yt_dlp.version.__version__}")
    print("📥 Downloading video from YouTube...")
    step_start_time = time.time()

    cookies_path = None
    cookiesfrombrowser = None
    cookies_env = os.environ.get("YOUTUBE_COOKIES")
    cookies_file_env = os.environ.get("YOUTUBE_COOKIES_FILE")
    cookies_from_browser = os.environ.get("YOUTUBE_COOKIES_FROM_BROWSER")

    if cookies_from_browser:
        parts = cookies_from_browser.split(":")
        browser = parts[0].strip()
        profile = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
        cookiesfrombrowser = (browser, profile) if profile else (browser,)
        print(f"🍪 Using cookies from browser: {cookies_from_browser}")
    elif cookies_file_env and os.path.exists(cookies_file_env):
        cookies_path = cookies_file_env
        print(f"🍪 Using cookies file from YOUTUBE_COOKIES_FILE: {cookies_path}")
    elif cookies_env:
        print("🍪 Found YOUTUBE_COOKIES env var, creating cookies file...")
        try:
            os.makedirs(output_dir, exist_ok=True)
            cookies_path = os.path.join(output_dir, ".yt_cookies.txt")
            with open(cookies_path, 'w') as f:
                f.write(cookies_env)
            if os.path.exists(cookies_path):
                print(f"   Debug: Cookies file created. Size: {os.path.getsize(cookies_path)} bytes")
        except Exception as e:
            print(f"⚠️ Failed to write cookies file: {e}")
            cookies_path = None
    else:
        print("⚠️ No YouTube cookies provided (env vars not set).")
    
    ydl_opts_info = {
        'quiet': False,
        'verbose': True,
        'no_warnings': False,
        'cookiefile': cookies_path if cookies_path else None,
        'cookiesfrombrowser': cookiesfrombrowser,
        'sleep_interval_requests': 5,
        'sleep_interval': 10,
        'max_sleep_interval': 30,
        'socket_timeout': 30,
        'retries': 10,
        'nocheckcertificate': True,
        'force_ipv4': True,
        'cachedir': False,
        'extractor_args': {'youtube': {'player_client': ['android', 'web']}},
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            video_title = info.get('title', 'youtube_video')
            sanitized_title = sanitize_filename(video_title)
        except Exception as e:
            # Force print to stderr/stdout immediately so it's captured before crash
            import sys
            import traceback
            
            # Print minimal error first to ensure something gets out
            print("🚨 YOUTUBE DOWNLOAD ERROR 🚨", file=sys.stderr)
            
            error_msg = f"""
            
❌ ================================================================= ❌
❌ FATAL ERROR: YOUTUBE DOWNLOAD FAILED
❌ ================================================================= ❌
            
REASON: YouTube has blocked the download request (Error 429/Unavailable).
        This is likely a temporary IP ban on this server.

👇 SOLUTION FOR USER 👇
---------------------------------------------------------------------
1. Download the video manually to your computer.
2. Use the 'Upload Video' tab in this app to process it.
---------------------------------------------------------------------

Technical Details: {str(e)}
            """
            # Print to both streams to ensure capture
            print(error_msg, file=sys.stdout)
            print(error_msg, file=sys.stderr)
            
            # Force flush
            sys.stdout.flush()
            sys.stderr.flush()
            
            # Wait a split second to allow buffer to drain before raising
            time.sleep(0.5)
            
            raise e
    
    output_template = os.path.join(output_dir, f'{sanitized_title}.%(ext)s')
    expected_file = os.path.join(output_dir, f'{sanitized_title}.mp4')
    if os.path.exists(expected_file):
        os.remove(expected_file)
        print(f"🗑️  Removed existing file to re-download with H.264 codec")
    
    ydl_opts = {
        'format': 'bestvideo[vcodec^=avc1][ext=mp4]+bestaudio[ext=m4a]/bestvideo[vcodec^=avc1]+bestaudio/best[ext=mp4]/best',
        'outtmpl': output_template,
        'merge_output_format': 'mp4',
        'quiet': False,
        'verbose': True,
        'no_warnings': False,
        'overwrites': True,
        'cookiefile': cookies_path if cookies_path else None,
        'cookiesfrombrowser': cookiesfrombrowser
    }

    def _download_with_opts(opts, label):
        print(f"⬇️  Download attempt: {label}")
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])

    try:
        _download_with_opts(ydl_opts, "default")
    except Exception as e:
        print(f"⚠️ Default download failed: {e}")
        print("🔁 Retrying with fallback client/format...")
        ydl_opts_fallback = {
            'format': '18/22/best[ext=mp4]/best',
            'outtmpl': output_template,
            'merge_output_format': 'mp4',
            'quiet': False,
            'verbose': True,
            'no_warnings': False,
            'overwrites': True,
            'cookiefile': cookies_path if cookies_path else None,
            'cookiesfrombrowser': cookiesfrombrowser,
            'extractor_args': {'youtube': {'player_client': ['web_safari', 'web']}},
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15'
            }
        }
        _download_with_opts(ydl_opts_fallback, "fallback")
    
    downloaded_file = os.path.join(output_dir, f'{sanitized_title}.mp4')
    
    if not os.path.exists(downloaded_file):
        for f in os.listdir(output_dir):
            if f.startswith(sanitized_title) and f.endswith('.mp4'):
                downloaded_file = os.path.join(output_dir, f)
                break
    
    step_end_time = time.time()
    print(f"✅ Video downloaded in {step_end_time - step_start_time:.2f}s: {downloaded_file}")
    
    return downloaded_file, sanitized_title

def process_video_to_vertical(input_video, final_output_video, ffmpeg_preset="fast", ffmpeg_crf=23, aspect_ratio=DEFAULT_ASPECT_RATIO):
    """
    Core logic to process video using scene detection and Active Speaker Tracking (MediaPipe)
    targeting a configurable aspect ratio.
    """
    script_start_time = time.time()
    
    # Define temporary file paths based on the output name
    base_name = os.path.splitext(final_output_video)[0]
    temp_video_output = f"{base_name}_temp_video.mp4"
    temp_audio_output = f"{base_name}_temp_audio.aac"
    
    # Clean up previous temp files if they exist
    if os.path.exists(temp_video_output): os.remove(temp_video_output)
    if os.path.exists(temp_audio_output): os.remove(temp_audio_output)
    if os.path.exists(final_output_video): os.remove(final_output_video)

    print(f"🎬 Processing clip: {input_video}")
    print("   Step 1: Detecting scenes...")
    scenes, fps = detect_scenes(input_video)
    
    if not scenes:
        print("   ❌ No scenes were detected. Using full video as one scene.")
        # If scene detection fails or finds nothing, treat whole video as one scene
        cap = cv2.VideoCapture(input_video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        from scenedetect import FrameTimecode
        scenes = [(FrameTimecode(0, fps), FrameTimecode(total_frames, fps))]

    print(f"   ✅ Found {len(scenes)} scenes.")

    print("\n   🧠 Step 2: Preparing Active Tracking...")
    original_width, original_height = get_video_resolution(input_video)
    aspect_label, target_ratio = normalize_aspect_ratio(aspect_ratio)
    OUTPUT_WIDTH, OUTPUT_HEIGHT = compute_output_dimensions(original_width, original_height, target_ratio)
    print(f"   Target aspect ratio: {aspect_label} ({OUTPUT_WIDTH}x{OUTPUT_HEIGHT})")

    # Initialize Cameraman
    cameraman = SmoothedCameraman(OUTPUT_WIDTH, OUTPUT_HEIGHT, original_width, original_height, aspect_ratio=target_ratio)
    
    # --- New Strategy: Per-Scene Analysis ---
    print("\n   🤖 Step 3: Analyzing Scenes for Strategy (Single vs Group)...")
    scene_strategies = analyze_scenes_strategy(input_video, scenes)
    # scene_strategies is a list of 'TRACK' or 'General' corresponding to scenes
    
    print("\n   ✂️ Step 4: Processing video frames...")
    
    command = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}', '-pix_fmt', 'bgr24',
        '-r', str(fps), '-i', '-', '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', str(ffmpeg_preset), '-crf', str(ffmpeg_crf), '-an',
        '-movflags', '+faststart', temp_video_output
    ]

    ffmpeg_process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    cap = cv2.VideoCapture(input_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_number = 0
    current_scene_index = 0
    
    # Pre-calculate scene boundaries
    scene_boundaries = []
    for s_start, s_end in scenes:
        scene_boundaries.append((s_start.get_frames(), s_end.get_frames()))

    # Global tracker for single-person shots
    speaker_tracker = SpeakerTracker(cooldown_frames=30)

    with tqdm(total=total_frames, desc="   Processing", file=sys.stdout) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Update Scene Index
            if current_scene_index < len(scene_boundaries):
                start_f, end_f = scene_boundaries[current_scene_index]
                if frame_number >= end_f and current_scene_index < len(scene_boundaries) - 1:
                    current_scene_index += 1
            
            # Determine Strategy for current frame based on scene
            current_strategy = scene_strategies[current_scene_index] if current_scene_index < len(scene_strategies) else 'TRACK'
            
            # Apply Strategy
            if current_strategy == 'GENERAL':
                # "Plano General" -> Blur Background + Fit Width
                output_frame = create_general_frame(frame, OUTPUT_WIDTH, OUTPUT_HEIGHT)
                
                # Reset cameraman/tracker so they don't drift while inactive
                cameraman.current_center_x = original_width / 2
                cameraman.target_center_x = original_width / 2
                
            else:
                # "Single Speaker" -> Track & Crop
                
                # Detect every 2nd frame for performance
                if frame_number % 2 == 0:
                    candidates = detect_face_candidates(frame)
                    target_box = speaker_tracker.get_target(candidates, frame_number, original_width)
                    if target_box:
                        cameraman.update_target(target_box)
                    else:
                        person_box = detect_person_yolo(frame)
                        if person_box:
                            cameraman.update_target(person_box)

                # Snap camera on scene change to avoid panning from previous scene position
                is_scene_start = (frame_number == scene_boundaries[current_scene_index][0])
                
                x1, y1, x2, y2 = cameraman.get_crop_box(force_snap=is_scene_start)
                
                # Crop
                if y2 > y1 and x2 > x1:
                    cropped = frame[y1:y2, x1:x2]
                    output_frame = cv2.resize(cropped, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
                else:
                    output_frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

            ffmpeg_process.stdin.write(output_frame.tobytes())
            frame_number += 1
            pbar.update(1)
    
    ffmpeg_process.stdin.close()
    stderr_output = ffmpeg_process.stderr.read().decode()
    ffmpeg_process.wait()
    cap.release()

    if ffmpeg_process.returncode != 0:
        print("\n   ❌ FFmpeg frame processing failed.")
        print("   Stderr:", stderr_output)
        return False

    print("\n   🔊 Step 5: Extracting audio...")
    audio_extract_command = [
        'ffmpeg', '-y', '-i', input_video, '-vn', '-acodec', 'copy', temp_audio_output
    ]
    try:
        subprocess.run(audio_extract_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        print("\n   ❌ Audio extraction failed (maybe no audio?). Proceeding without audio.")
        pass

    print("\n   ✨ Step 6: Merging...")
    if os.path.exists(temp_audio_output):
        merge_command = [
            'ffmpeg', '-y', '-i', temp_video_output, '-i', temp_audio_output,
            '-c:v', 'copy', '-c:a', 'copy', '-movflags', '+faststart', final_output_video
        ]
    else:
         merge_command = [
            'ffmpeg', '-y', '-i', temp_video_output,
            '-c:v', 'copy', '-movflags', '+faststart', final_output_video
        ]
        
    try:
        subprocess.run(merge_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(f"   ✅ Clip saved to {final_output_video}")
    except subprocess.CalledProcessError as e:
        print("\n   ❌ Final merge failed.")
        print("   Stderr:", e.stderr.decode())
        return False

    # Clean up temp files
    if os.path.exists(temp_video_output): os.remove(temp_video_output)
    if os.path.exists(temp_audio_output): os.remove(temp_audio_output)
    
    return True

def transcribe_video(video_path, language=None, backend=None, model_name=None, word_timestamps=True, compute_type=None, cpu_threads=0, num_workers=1):
    backend = (backend or os.getenv("WHISPER_BACKEND", "faster")).lower()
    model_name = model_name or os.getenv("WHISPER_MODEL", "large-v3")

    if backend == "openai":
        print(f"🎙️  Transcribing video with OpenAI Whisper (model={model_name})...")
        import whisper
        device = os.getenv("WHISPER_DEVICE", "cpu")
        model = whisper.load_model(model_name, device=device)
        result = model.transcribe(
            video_path,
            word_timestamps=word_timestamps,
            verbose=False,
            language=language,
            task="transcribe"
        )

        transcript_segments = []
        full_text = result.get("text", "").strip()
        for segment in result.get("segments", []):
            seg_dict = {
                'text': segment.get("text", ""),
                'start': segment.get("start", 0.0),
                'end': segment.get("end", 0.0),
                'words': []
            }
            for word in segment.get("words", []) or []:
                seg_dict['words'].append({
                    'word': word.get("word", ""),
                    'start': word.get("start", 0.0),
                    'end': word.get("end", 0.0),
                    'probability': word.get("probability", 0.0)
                })
            transcript_segments.append(seg_dict)

        return {
            'text': full_text,
            'segments': transcript_segments,
            'language': result.get("language", "unknown")
        }

    requested_device = (os.getenv("WHISPER_DEVICE", "auto") or "auto").lower().strip()
    if requested_device in ("auto", ""):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested_device in ("cuda", "gpu") and not torch.cuda.is_available():
        print("⚠️ WHISPER_DEVICE=cuda solicitado pero CUDA no está disponible. Fallback a CPU.")
        device = "cpu"
    elif requested_device in ("mps", "metal"):
        # faster-whisper usa principalmente cpu/cuda. En Mac usamos CPU.
        device = "cpu"
    else:
        device = "cpu" if requested_device not in ("cuda", "cpu") else requested_device

    if not compute_type:
        # Valores seguros por dispositivo
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "float16" if device == "cuda" else "int8")

    print(f"🎙️  Transcribing video with Faster-Whisper (model={model_name}, device={device}, compute={compute_type})...")
    from faster_whisper import WhisperModel

    cpu_threads_env = os.getenv("WHISPER_CPU_THREADS", "").strip()
    num_workers_env = os.getenv("WHISPER_NUM_WORKERS", "").strip()
    if cpu_threads_env and not cpu_threads:
        cpu_threads = int(cpu_threads_env)
    if num_workers_env and not num_workers:
        num_workers = int(num_workers_env)

    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        num_workers=num_workers
    )

    segments, info = model.transcribe(video_path, word_timestamps=word_timestamps, language=language, task="transcribe")
    print(f"   Detected language '{info.language}' with probability {info.language_probability:.2f}")

    transcript_segments = []
    full_text = ""

    for segment in segments:
        print(f"   [{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

        seg_dict = {
            'text': segment.text,
            'start': segment.start,
            'end': segment.end,
            'words': []
        }

        if segment.words:
            for word in segment.words:
                seg_dict['words'].append({
                    'word': word.word,
                    'start': word.start,
                    'end': word.end,
                    'probability': word.probability
                })

        transcript_segments.append(seg_dict)
        full_text += segment.text + " "

    return {
        'text': full_text.strip(),
        'segments': transcript_segments,
        'language': info.language
    }

def get_viral_clips(transcript_result, video_duration, max_clips=None, clip_length_target=None):
    print("🤖  Analyzing with Gemini...")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not found in environment variables.")
        return None


    client = genai.Client(api_key=api_key)
    
    # We use gemini-2.5-flash as requested.
    model_name = 'gemini-2.5-flash' 
    
    print(f"🤖  Initializing Gemini with model: {model_name}")

    # Extract words
    words = []
    for segment in transcript_result['segments']:
        for word in segment.get('words', []):
            words.append({
                'w': word['word'],
                's': word['start'],
                'e': word['end']
            })

    max_clips_rule = ""
    if max_clips:
        max_clips_rule = f"IMPORTANT: Return at most {max_clips} clips."
    length_rule = clip_length_guidance(clip_length_target)

    prompt = GEMINI_PROMPT_TEMPLATE.format(
        video_duration=video_duration,
        transcript_text=json.dumps(transcript_result['text']),
        words_json=json.dumps(words),
        max_clips_rule=max_clips_rule,
        clip_length_rule=length_rule
    )

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        
        # --- Cost Calculation ---
        try:
            usage = response.usage_metadata
            if usage:
                # Gemini 2.5 Flash Pricing (Dec 2025)
                # Input: $0.10 per 1M tokens
                # Output: $0.40 per 1M tokens
                
                input_price_per_million = 0.10
                output_price_per_million = 0.40
                
                prompt_tokens = usage.prompt_token_count
                output_tokens = usage.candidates_token_count
                
                input_cost = (prompt_tokens / 1_000_000) * input_price_per_million
                output_cost = (output_tokens / 1_000_000) * output_price_per_million
                total_cost = input_cost + output_cost
                
                cost_analysis = {
                    "input_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "input_cost": input_cost,
                    "output_cost": output_cost,
                    "total_cost": total_cost,
                    "model": model_name
                }

                print(f"💰 Token Usage ({model_name}):")
                print(f"   - Input Tokens: {prompt_tokens} (${input_cost:.6f})")
                print(f"   - Output Tokens: {output_tokens} (${output_cost:.6f})")
                print(f"   - Total Estimated Cost: ${total_cost:.6f}")
                
        except Exception as e:
            print(f"⚠️ Could not calculate cost: {e}")
            cost_analysis = None
        # ------------------------

        # Clean response if it contains markdown code blocks
        text = response.text
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        result_json = json.loads(text)
        if cost_analysis:
            result_json['cost_analysis'] = cost_analysis

        if max_clips and isinstance(result_json.get('shorts'), list):
            result_json['shorts'] = result_json['shorts'][:max_clips]

        result_json = normalize_shorts_payload(result_json)
            
        return result_json
    except Exception as e:
        print(f"❌ Gemini Error: {e}")
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AutoCrop with Viral Clip Detection.")
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-i', '--input', type=str, help="Path to the input video/audio file.")
    input_group.add_argument('-u', '--url', type=str, help="YouTube URL to download and process.")
    
    parser.add_argument('-o', '--output', type=str, help="Output directory or file (if processing whole video).")
    parser.add_argument('--keep-original', action='store_true', help="Keep the downloaded YouTube video.")
    parser.add_argument('--skip-analysis', action='store_true', help="Skip AI analysis and convert the whole video.")
    parser.add_argument('--language', type=str, default=None, help="Force transcription language (e.g., 'es', 'en').")
    parser.add_argument('--max-clips', type=int, default=None, help="Max number of clips to generate (1-15).")
    parser.add_argument('--whisper-backend', type=str, default=None, help="Whisper backend: openai|faster.")
    parser.add_argument('--whisper-model', type=str, default=None, help="Whisper model: tiny|base|small|medium|large|large-v2|large-v3.")
    parser.add_argument('--word-timestamps', type=str, default="true", help="true/false for word-level timestamps.")
    parser.add_argument('--ffmpeg-preset', type=str, default="fast", help="FFmpeg preset: ultrafast|fast|medium.")
    parser.add_argument('--ffmpeg-crf', type=int, default=23, help="FFmpeg CRF quality (lower=better).")
    parser.add_argument('--aspect-ratio', type=str, default=DEFAULT_ASPECT_RATIO, help="Output aspect ratio: 9:16 or 16:9.")
    parser.add_argument('--clip-length-target', type=str, default=None, help="Preferred clip length profile: short|balanced|long.")
    parser.add_argument('--style-template', type=str, default=None, help="UI template id used for this generation (metadata only).")
    parser.add_argument('--content-profile', type=str, default=None, help="Content profile selected in UI (metadata only).")
    
    args = parser.parse_args()

    script_start_time = time.time()

    if args.max_clips:
        args.max_clips = max(1, min(15, args.max_clips))
    args.word_timestamps = str(args.word_timestamps).lower() in ("1", "true", "yes", "y")
    if args.clip_length_target:
        args.clip_length_target = str(args.clip_length_target).strip().lower()
        if args.clip_length_target not in ("short", "balanced", "long"):
            print("⚠️ Invalid --clip-length-target. Using default behavior.")
            args.clip_length_target = None
    try:
        args.aspect_ratio, _ = normalize_aspect_ratio(args.aspect_ratio)
    except ValueError as e:
        print(f"❌ {e}")
        exit(1)
    
    def _ensure_dir(path: str) -> str:
        """Create directory if missing and return the same path."""
        if path:
            os.makedirs(path, exist_ok=True)
        return path
    
    # 1. Get Input Video
    if args.url:
        # For multi-clip runs, treat --output as an OUTPUT DIRECTORY (create it if needed).
        # For whole-video runs (--skip-analysis), --output can be a file path.
        if args.output and not args.skip_analysis:
            output_dir = _ensure_dir(args.output)
        else:
            # If output is a directory, use it; if it's a filename, use its directory; else default "."
            if args.output and os.path.isdir(args.output):
                output_dir = args.output
            elif args.output and not os.path.isdir(args.output):
                output_dir = os.path.dirname(args.output) or "."
            else:
                output_dir = "."
        
        input_video, video_title = download_youtube_video(args.url, output_dir)
    else:
        input_video = args.input
        video_title = os.path.splitext(os.path.basename(input_video))[0]
        
        if args.output and not args.skip_analysis:
            # For multi-clip runs, treat --output as an OUTPUT DIRECTORY (create it if needed).
            output_dir = _ensure_dir(args.output)
        else:
            # If output is a directory, use it; if it's a filename, use its directory; else default to input dir.
            if args.output and os.path.isdir(args.output):
                output_dir = args.output
            elif args.output and not os.path.isdir(args.output):
                output_dir = os.path.dirname(args.output) or os.path.dirname(input_video)
            else:
                output_dir = os.path.dirname(input_video)

    if not os.path.exists(input_video):
        print(f"❌ Input file not found: {input_video}")
        exit(1)

    generated_audio_canvas = False
    if is_audio_input(input_video):
        print("🎧 Audio-only input detected. Generating visual canvas...")
        audio_base = sanitize_filename(os.path.splitext(os.path.basename(input_video))[0])
        canvas_video = os.path.join(output_dir, f"{audio_base}_audio_canvas.mp4")
        ok = build_audio_canvas_video(
            input_video,
            canvas_video,
            ffmpeg_preset=args.ffmpeg_preset,
            ffmpeg_crf=args.ffmpeg_crf,
            aspect_ratio=args.aspect_ratio
        )
        if not ok:
            exit(1)
        input_video = canvas_video
        video_title = audio_base
        generated_audio_canvas = True

    # 2. Decision: Analyze clips or process whole?
    if args.skip_analysis:
        print("⏩ Skipping analysis, processing entire video...")
        output_file = args.output if args.output else os.path.join(output_dir, f"{video_title}_vertical.mp4")
        process_video_to_vertical(input_video, output_file, args.ffmpeg_preset, args.ffmpeg_crf, args.aspect_ratio)
    else:
        # 3. Transcribe
        transcript = transcribe_video(
            input_video,
            language=args.language,
            backend=args.whisper_backend,
            model_name=args.whisper_model,
            word_timestamps=args.word_timestamps
        )
        
        # Get duration
        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        cap.release()

        # 4. Gemini Analysis
        clips_data = get_viral_clips(
            transcript,
            duration,
            max_clips=args.max_clips,
            clip_length_target=args.clip_length_target
        )
        
        if not clips_data or 'shorts' not in clips_data:
            print("❌ Failed to identify clips. Converting whole video as fallback.")
            output_file = os.path.join(output_dir, f"{video_title}_vertical.mp4")
            process_video_to_vertical(input_video, output_file, args.ffmpeg_preset, args.ffmpeg_crf, args.aspect_ratio)
        else:
            clips_data = postprocess_shorts_with_transcript(
                clips_data=clips_data,
                transcript=transcript,
                duration=duration,
                max_clips=args.max_clips,
                clip_length_target=args.clip_length_target
            )
            clips_data = normalize_shorts_payload(clips_data)
            clips_data["generation_profile"] = {
                "clip_length_target": args.clip_length_target or "default",
                "style_template": (str(args.style_template).strip() if args.style_template else None),
                "content_profile": (str(args.content_profile).strip() if args.content_profile else None)
            }
            post = clips_data.get("postprocess", {}) if isinstance(clips_data, dict) else {}
            smart_meta = post.get("smart_boundaries", {}) if isinstance(post, dict) else {}
            dedupe_meta = post.get("semantic_dedupe", {}) if isinstance(post, dict) else {}
            if smart_meta:
                print(
                    "✂️ Smart boundaries:",
                    f"{smart_meta.get('clips_refined', 0)} clips refined,"
                    f" {smart_meta.get('boundary_points', 0)} boundary points"
                )
            if dedupe_meta:
                print(
                    "🧠 Semantic dedupe:",
                    f"{dedupe_meta.get('removed_duplicates', 0)} duplicates removed,"
                    f" {dedupe_meta.get('kept_clips', len(clips_data.get('shorts', [])))} kept"
                )

            print(f"🔥 Found {len(clips_data['shorts'])} viral clips!")
            for clip in clips_data['shorts']:
                if isinstance(clip, dict):
                    clip['aspect_ratio'] = args.aspect_ratio
            
            # Save metadata
            clips_data['transcript'] = transcript # Save full transcript for subtitles
            metadata_file = os.path.join(output_dir, f"{video_title}_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump(clips_data, f, indent=2)
            print(f"   Saved metadata to {metadata_file}")

            # 5. Process each clip
            for i, clip in enumerate(clips_data['shorts']):
                start = clip['start']
                end = clip['end']
                print(f"\n🎬 Processing Clip {i+1}: {start}s - {end}s")
                print(f"   Title: {clip.get('video_title_for_youtube_short', 'No Title')}")
                
                # Cut clip
                clip_filename = f"{video_title}_clip_{i+1}.mp4"
                clip_temp_path = os.path.join(output_dir, f"temp_{clip_filename}")
                clip_final_path = os.path.join(output_dir, clip_filename)
                
                # ffmpeg cut
                # Using re-encoding for precision as requested by strict seconds
                cut_command = [
                    'ffmpeg', '-y', 
                    '-ss', str(start), 
                    '-to', str(end), 
                    '-i', input_video,
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', str(args.ffmpeg_crf), '-preset', str(args.ffmpeg_preset),
                    '-c:a', 'aac',
                    '-movflags', '+faststart',
                    clip_temp_path
                ]
                subprocess.run(cut_command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                
                # Process vertical
                success = process_video_to_vertical(clip_temp_path, clip_final_path, args.ffmpeg_preset, args.ffmpeg_crf, args.aspect_ratio)
                
                if success:
                    print(f"   ✅ Clip {i+1} ready: {clip_final_path}")
                
                # Clean up temp cut
                if os.path.exists(clip_temp_path):
                    os.remove(clip_temp_path)

    # Clean up original if requested
    if args.url and not args.keep_original and os.path.exists(input_video):
        os.remove(input_video)
        print(f"🗑️  Cleaned up downloaded video.")
    elif generated_audio_canvas and os.path.exists(input_video):
        os.remove(input_video)
        print("🗑️  Cleaned up temporary audio canvas video.")

    total_time = time.time() - script_start_time
    print(f"\n⏱️  Total execution time: {total_time:.2f}s")
