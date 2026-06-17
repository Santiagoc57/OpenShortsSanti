import time
import cv2
import subprocess
import argparse
import re
import sys
import math
import zlib
import shutil
import glob
import threading
import queue
import torch
import os
import numpy as np
from tqdm import tqdm
from autocrop import (
    detect_scenes,
    SmoothedCameraman,
    SpeakerTracker,
    detect_face_candidates,
    detect_person_yolo,
    analyze_scenes_strategy,
    create_general_frame,
    is_variable_frame_rate,
    normalize_to_cfr,
    _DummyTime
)
import yt_dlp
# import whisper (replaced by faster_whisper inside function)
from google import genai
from groq import Groq
from dotenv import load_dotenv
import json
from typing import List, Dict, Any, Optional, Tuple
from tight_edit import build_tight_edit_plan, normalize_tight_edit_preset, render_keep_segments
from runtime_limits import ffmpeg_thread_args, subprocess_priority_kwargs
from whisper_runtime import transcribe_with_runtime

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
DEFAULT_FFMPEG_PRESET = os.environ.get("OPENSHORTS_FFMPEG_PRESET", "medium").strip() or "medium"
DEFAULT_FFMPEG_CRF = int(os.environ.get("OPENSHORTS_FFMPEG_CRF", "18"))
FORCE_STANDARD_VERTICAL_OUTPUT = os.environ.get("OPENSHORTS_FORCE_STANDARD_VERTICAL_OUTPUT", "true").strip().lower() not in {"0", "false", "no"}
DEFAULT_VIDEO_ENHANCE_FILTER = os.environ.get(
    "OPENSHORTS_VIDEO_ENHANCE_FILTER",
    "unsharp=5:5:1.5,eq=brightness=0.06:contrast=1.1:saturation=1.15",
).strip()
DEFAULT_TIGHT_EDIT_PRESET = normalize_tight_edit_preset(os.environ.get("TIGHT_EDIT_PRESET", "off"), "off")
VIDEO_ENCODER = (os.environ.get("OPENSHORTS_VIDEO_ENCODER") or "libx264").strip() or "libx264"


def video_encoder_args(ffmpeg_preset, ffmpeg_crf):
    """
    Encoder args for the final H.264 encodes. OPENSHORTS_VIDEO_ENCODER selects
    hardware encoders (h264_videotoolbox on macOS, h264_nvenc/hevc_nvenc on
    NVIDIA); they ignore preset and map CRF to their own quality scale.
    """
    if VIDEO_ENCODER == "h264_videotoolbox":
        quality = max(1, min(100, 100 - int(ffmpeg_crf) * 2))
        return ['-c:v', 'h264_videotoolbox', '-q:v', str(quality), '-allow_sw', '1']
    if VIDEO_ENCODER in ("h264_nvenc", "hevc_nvenc"):
        return ['-c:v', VIDEO_ENCODER, '-preset', 'p5', '-cq', str(ffmpeg_crf)]
    return ['-c:v', 'libx264', '-preset', str(ffmpeg_preset), '-crf', str(ffmpeg_crf)]


MP4_SAFE_AUDIO_CODECS = {"aac", "mp3", "ac3", "eac3", "alac"}


def _source_audio_codec(media_path):
    """Audio codec of the first audio stream, or '' if none/undetectable."""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
             '-show_entries', 'stream=codec_name', '-of', 'csv=p=0', media_path],
            capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0].lower()
    except FileNotFoundError:
        pass
    # Fallback when ffprobe is unavailable: parse `ffmpeg -i` stream info.
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-i', media_path],
            capture_output=True, text=True
        )
        match = re.search(r"Audio:\s*([A-Za-z0-9_]+)", result.stderr or "")
        if match:
            return match.group(1).lower()
    except Exception:
        pass
    return ""


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
    if FORCE_STANDARD_VERTICAL_OUTPUT and math.isclose(target_ratio, ASPECT_RATIO_PRESETS["9:16"], rel_tol=0.001):
        return 1080, 1920

    source_ratio = input_width / input_height
    if source_ratio >= target_ratio:
        out_height = input_height
        out_width = out_height * target_ratio
    else:
        out_width = input_width
        out_height = out_width / target_ratio
    return _make_even(out_width), _make_even(out_height)

GEMINI_PROMPT_TEMPLATE = """
You are a senior short-form video editor. Read the ENTIRE transcript and its timestamped word timeline to choose the 3–15 MOST VIRAL moments for TikTok/IG Reels/YouTube Shorts. Each clip must be between 15 and 60 seconds long.
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

WORD_TIMELINE (each line: [start_seconds-end_seconds] words spoken in that span; use these to anchor exact cut points):
{words_json}

STRICT EXCLUSIONS:
- No generic intros/outros or purely sponsorship segments unless they contain the hook.
- No clips < 15 s or > 60 s.

OUTPUT — RETURN ONLY VALID JSON (no markdown, no comments). Order clips by predicted performance (best to worst).
LANGUAGE RULE (STRICT): all textual fields MUST be in Spanish (español neutro): score_reason, hook_explanation, topic_tags, title_variants, social_variants.
STYLE RULES:
- Titles: Use "Sentence case" (e.g., "Así me recibe la gente en Argentina"), only the first letter and proper names in uppercase. NO excessive capitalization.
- Emojis: You MAY use relevant emojis in titles and social variants to increase engagement.
- CTA: In the social descriptions, ALWAYS include a CTA in Spanish like "Sígueme y comenta X y te envío el workflow".

{trailer_fragments_rule}

{{
  "shorts": [
    {{
      "start": <number in seconds, e.g., 12.340>,
      "end": <number in seconds, e.g., 37.900>,
      "virality_score": <integer 0-100, where 100 is highest predicted performance>,
      "selection_confidence": <number between 0 and 1 indicating confidence in this selection>,
      "score_reason": "<razón corta en español de por qué este clip puede rendir>",
      "hook_explanation": "<explicación del gancho inicial de los primeros 3 segundos y por qué atrapará al usuario>",
      "topic_tags": ["<hasta 5 etiquetas cortas en español, sin #, ej: politica, debate, economia>"],
      "title_variants": ["<array de 5 títulos distintos en español, máximo 100 caracteres cada uno, sentence case, emojis permitidos>"],
      "social_variants": ["<array de 5 descripciones sociales distintas en español, incluyendo CTA, emojis permitidos, orientas a views para TikTok/IGReels>"]
    }}
  ],
  "trailer_fragments": [
    {{
      "start": <number in seconds>,
      "end": <number in seconds>,
      "reason": "<breve razón en español de por qué este fragmento es bueno para el trailer>"
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

        hook_exp = clip.get('hook_explanation')
        if not hook_exp:
            hook_exp = "Gancho visual o auditivo fuerte en los primeros segundos."
        clip['hook_explanation'] = str(hook_exp).strip()[:200]
        
        # Variants logic
        t_vars = clip.get('title_variants') or clip.get('video_title_variants')
        if not isinstance(t_vars, list) or not t_vars:
            # Check old single field
            fallback = clip.get('video_title_for_youtube_short') or clip.get('title') or f"Clip viral #{i+1}"
            t_vars = [fallback]
        
        s_vars = clip.get('social_variants') or clip.get('video_social_variants')
        if not isinstance(s_vars, list) or not s_vars:
            # Check old single fields
            fallback = clip.get('video_description_for_tiktok') or clip.get('video_description_for_instagram') or ""
            s_vars = [fallback] if fallback else []

        clip['title_variants'] = [str(v).strip() for v in t_vars if str(v).strip()][:8]
        clip['social_variants'] = [str(v).strip() for v in s_vars if str(v).strip()][:8]

        # Primary fields for backward compatibility
        if clip['title_variants']:
            primary_title = clip['title_variants'][0]
            clip['video_title_for_youtube_short'] = primary_title
            clip['title'] = primary_title
            clip['title_variant_index'] = 0
        
        if clip['social_variants']:
            primary_social = clip['social_variants'][0]
            clip['video_description_for_tiktok'] = primary_social
            clip['video_description_for_instagram'] = primary_social
            clip['social_variant_index'] = 0

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

def build_audio_canvas_video(input_audio, output_video, ffmpeg_preset=DEFAULT_FFMPEG_PRESET, ffmpeg_crf=DEFAULT_FFMPEG_CRF, aspect_ratio=DEFAULT_ASPECT_RATIO):
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
        'extractor_args': {'youtube': {'player_client': ['ios', 'android', 'mweb', 'web']}},
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
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best',
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

def process_video_to_vertical(input_video, final_output_video, ffmpeg_preset=DEFAULT_FFMPEG_PRESET, ffmpeg_crf=DEFAULT_FFMPEG_CRF, aspect_ratio=DEFAULT_ASPECT_RATIO):
    """
    Core logic to convert horizontal video to vertical using scene detection and Active Speaker Tracking (MediaPipe).
    """
    script_start_time = time.time()
    
    # Define temporary file paths based on the output name
    base_name = os.path.splitext(final_output_video)[0]
    temp_video_output = f"{base_name}_temp_video.mp4"
    temp_cfr_input = f"{base_name}_temp_cfr_input.mp4"

    # Clean up previous temp files if they exist
    for f in [temp_video_output, final_output_video, temp_cfr_input]:
        if os.path.exists(f): 
            try: os.remove(f)
            except: pass

    print(f"🎬 Processing clip: {input_video}")
    
    # Pre-processing: normalize VFR to CFR if needed
    if is_variable_frame_rate(input_video):
        print("   ⚠️  Variable frame rate detected — normalizing to constant frame rate first...")
        if normalize_to_cfr(input_video, temp_cfr_input):
            input_video = temp_cfr_input
            print("   ✅ VFR normalization complete.")
        else:
            print("   ⚠️  Proceeding with original VFR file (audio sync may be affected).")

    print("   🧠 Step 1: Detecting scenes...")
    scenes, fps = detect_scenes(input_video)
    
    if not scenes:
        print("   ❌ No scenes were detected. Using full video as one scene.")
        cap = cv2.VideoCapture(input_video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        scenes = [(_DummyTime(0, fps), _DummyTime(total_frames, fps))]

    print(f"   ✅ Found {len(scenes)} scenes.")

    print("\n   🧠 Step 2: Preparing Active Tracking...")
    cap = cv2.VideoCapture(input_video)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    aspect_label, target_ratio = normalize_aspect_ratio(aspect_ratio)
    OUTPUT_WIDTH, OUTPUT_HEIGHT = compute_output_dimensions(original_width, original_height, target_ratio)
    print(f"   Target aspect ratio: {aspect_label} ({OUTPUT_WIDTH}x{OUTPUT_HEIGHT})")

    cameraman = SmoothedCameraman(OUTPUT_WIDTH, OUTPUT_HEIGHT, original_width, original_height)
    cameraman.crop_width = int(cameraman.crop_height * target_ratio)
    if cameraman.crop_width > original_width:
        cameraman.crop_width = original_width
        cameraman.crop_height = int(cameraman.crop_width / max(1e-6, target_ratio))
    cameraman.safe_zone_radius = cameraman.crop_width * 0.25

    print("\n   🤖 Step 3: Analyzing Scenes for Strategy (Single vs Group)...")
    scene_strategies = analyze_scenes_strategy(input_video, scenes)
    
    print("\n   ✂️ Step 4: Processing video frames...")
    
    command = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}', '-pix_fmt', 'bgr24',
        '-r', str(fps), '-i', '-',
        *ffmpeg_thread_args(include_filter_threads=True),
        '-pix_fmt', 'yuv420p',
        '-r', str(fps), '-vsync', 'cfr',
    ]
    if DEFAULT_VIDEO_ENHANCE_FILTER:
        command.extend(['-vf', DEFAULT_VIDEO_ENHANCE_FILTER])
    command.extend([
        *video_encoder_args(ffmpeg_preset, ffmpeg_crf), '-an',
        '-movflags', '+faststart', temp_video_output
    ])

    ffmpeg_process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        **subprocess_priority_kwargs()
    )

    cap = cv2.VideoCapture(input_video)
    frame_number = 0
    current_scene_index = 0
    dropped_frames = 0
    last_output_frame = None
    
    scene_boundaries = []
    for s_start, s_end in scenes:
        scene_boundaries.append((s_start.get_frames(), s_end.get_frames()))

    speaker_tracker = SpeakerTracker(cooldown_frames=30)

    # Decode (reader) and stdin writes to the encoder (writer) run on their own
    # threads so video decoding, detection/cropping and x264 encoding overlap
    # instead of executing strictly in series on one thread.
    frame_queue = queue.Queue(maxsize=48)
    write_queue = queue.Queue(maxsize=48)
    stop_reading = threading.Event()
    writer_errors = []

    def _read_frames():
        try:
            while not stop_reading.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                while not stop_reading.is_set():
                    try:
                        frame_queue.put(frame, timeout=0.5)
                        break
                    except queue.Full:
                        continue
        finally:
            while True:
                try:
                    frame_queue.put(None, timeout=0.5)
                    break
                except queue.Full:
                    if stop_reading.is_set():
                        break

    def _write_frames():
        failed = False
        while True:
            data = write_queue.get()
            if data is None:
                break
            if failed:
                continue
            try:
                ffmpeg_process.stdin.write(data)
            except Exception as exc:
                writer_errors.append(exc)
                failed = True

    reader_thread = threading.Thread(target=_read_frames, daemon=True)
    writer_thread = threading.Thread(target=_write_frames, daemon=True)
    reader_thread.start()
    writer_thread.start()

    with tqdm(total=total_frames, desc="   Processing", file=sys.stdout) as pbar:
        while True:
            if writer_errors:
                break
            frame = frame_queue.get()
            if frame is None:
                break

            if current_scene_index < len(scene_boundaries):
                start_f, end_f = scene_boundaries[current_scene_index]
                if frame_number >= end_f and current_scene_index < len(scene_boundaries) - 1:
                    current_scene_index += 1
            
            current_strategy = scene_strategies[current_scene_index] if current_scene_index < len(scene_strategies) else 'TRACK'
            
            try:
                if current_strategy == 'GENERAL':
                    output_frame = create_general_frame(frame, OUTPUT_WIDTH, OUTPUT_HEIGHT)
                    cameraman.current_center_x = original_width / 2
                    cameraman.target_center_x = original_width / 2
                else:
                    if frame_number % 2 == 0:
                        candidates = detect_face_candidates(frame)
                        target_box = speaker_tracker.get_target(candidates, frame_number, original_width)
                        if target_box:
                            cameraman.update_target(target_box)
                        else:
                            person_box = detect_person_yolo(frame)
                            if person_box:
                                cameraman.update_target(person_box)

                    is_scene_start = (frame_number == scene_boundaries[current_scene_index][0])
                    x1, y1, x2, y2 = cameraman.get_crop_box(force_snap=is_scene_start)
                    
                    if y2 > y1 and x2 > x1:
                        cropped = frame[y1:y2, x1:x2]
                        output_frame = cv2.resize(cropped, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_CUBIC)
                    else:
                        output_frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_CUBIC)
                
                last_output_frame = output_frame
            except Exception as e:
                dropped_frames += 1
                if last_output_frame is not None:
                    output_frame = last_output_frame
                else:
                    output_frame = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)

            write_queue.put(output_frame.tobytes())
            frame_number += 1
            pbar.update(1)

    stop_reading.set()
    try:
        while frame_queue.get_nowait() is not None:
            pass
    except queue.Empty:
        pass
    write_queue.put(None)
    writer_thread.join(timeout=60)
    reader_thread.join(timeout=10)

    if dropped_frames > 0:
        print(f"  ⚠️  {dropped_frames} frame(s) could not be processed and were duplicated.")

    try:
        ffmpeg_process.stdin.close()
    except Exception:
        pass
    stderr_output = ffmpeg_process.stderr.read().decode()
    ffmpeg_process.wait()
    cap.release()

    if ffmpeg_process.returncode != 0 or writer_errors:
        print("\n   ❌ FFmpeg frame processing failed.")
        if writer_errors:
            print("   Pipe error:", writer_errors[0])
        print("   Stderr:", stderr_output)
        return False

    print("\n   ✨ Step 5: Merging with source audio...")

    def _merge_with_audio(audio_args):
        merge_command = [
            'ffmpeg', '-y', '-i', temp_video_output, '-i', input_video,
            '-map', '0:v:0', '-map', '1:a:0?',
            '-c:v', 'copy', *audio_args,
            '-movflags', '+faststart', '-shortest',
            final_output_video
        ]
        return subprocess.run(merge_command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    # Copy the source audio only when its codec is broadly supported inside
    # MP4 (newer ffmpeg happily muxes Opus into MP4, but TikTok/IG and many
    # players reject it). Anything else — e.g. Opus from YouTube WebM, which
    # used to produce MUTE clips — is re-encoded to AAC.
    source_audio_codec = _source_audio_codec(input_video)
    if source_audio_codec in MP4_SAFE_AUDIO_CODECS:
        merge_result = _merge_with_audio(['-c:a', 'copy'])
        if merge_result.returncode != 0:
            merge_result = _merge_with_audio(['-c:a', 'aac', '-b:a', '192k'])
    else:
        if source_audio_codec:
            print(f"   ℹ️ Source audio is '{source_audio_codec}' (not MP4-safe), re-encoding to AAC...")
        merge_result = _merge_with_audio(['-c:a', 'aac', '-b:a', '192k'])
    if merge_result.returncode != 0:
        print("\n   ❌ Final merge failed.")
        print("   Stderr:", merge_result.stderr.decode(errors='ignore')[-800:])
        return False
    print(f"   ✅ Clip saved to {final_output_video}")

    # Clean up temp files
    for f in [temp_video_output, temp_cfr_input]:
        if os.path.exists(f): 
            try: os.remove(f)
            except: pass
    
    return True
def build_super_trailer(input_video, fragments, output_path, ffmpeg_preset=DEFAULT_FFMPEG_PRESET, ffmpeg_crf=DEFAULT_FFMPEG_CRF):
    """
    Creates a fast-paced summary (Super Trailer) with crossfade transitions.
    fragments: List[dict] with 'start', 'end' in seconds.
    """
    if not fragments:
        return False
        
    print(f"🎬 Building Super Trailer with {len(fragments)} fragments...")
    
    # 1. Extract each fragment as a temp file
    temp_dir = os.path.dirname(output_path)
    base_name = os.path.basename(output_path).replace(".mp4", "")
    temp_files = []
    
    try:
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except Exception:
                pass

        for i, frag in enumerate(fragments):
            start = frag['start']
            end = frag['end']
            temp_frag_path = os.path.join(temp_dir, f"temp_trailer_frag_{i}_{base_name}.mp4")
            if os.path.exists(temp_frag_path):
                try:
                    os.remove(temp_frag_path)
                except Exception:
                    pass
            
            # Use same format for consistency
            cut_cmd = [
                'ffmpeg', '-y', 
                '-ss', f"{start:.3f}", 
                '-to', f"{end:.3f}", 
                '-i', input_video,
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', str(ffmpeg_crf), '-preset', ffmpeg_preset,
                '-c:a', 'aac', '-ar', '44100', '-ac', '2',
                '-movflags', '+faststart',
                temp_frag_path
            ]
            res = subprocess.run(cut_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if res.returncode == 0 and os.path.exists(temp_frag_path) and os.path.getsize(temp_frag_path) > 2048:
                temp_files.append(temp_frag_path)
            else:
                stderr_output = res.stderr.decode() if res.stderr else "No stderr"
                print(f"❌ Error extracting fragment {i}: {stderr_output}")
        
        if len(temp_files) < 2:
            print(f"❌ Not enough fragments extracted ({len(temp_files)}/2).")
            # Just copy the first one if we can't find others
            if temp_files:
                shutil.copyfile(temp_files[0], output_path)
                return True
            return False

        # 2. Build the complex filter for crossfades
        fade_dur = 0.5
        filter_complex = ""
        inputs = ""
        
        for i, f_path in enumerate(temp_files):
            inputs += f'-i "{f_path}" '
        
        offsets = []
        current_total_time = 0
        for i in range(len(temp_files)):
            dur = _probe_media_duration_seconds(temp_files[i])
            if dur is None or dur <= 0.05:
                try:
                    fallback_dur = float(fragments[i].get('end', 0)) - float(fragments[i].get('start', 0))
                except Exception:
                    fallback_dur = 0.0
                dur = max(0.3, fallback_dur)
            offsets.append(current_total_time + dur - fade_dur)
            current_total_time += (dur - fade_dur)

        # Build video filter chain
        for i in range(1, len(temp_files)):
            prev_v = f"v{i-1}" if i > 1 else "0:v"
            next_v = f"{i}:v"
            out_v = f"v{i}"
            offset = offsets[i-1]
            filter_complex += f"[{prev_v}][{next_v}]xfade=transition=fade:duration={fade_dur}:offset={offset}[{out_v}]; "

        # Build audio filter chain
        for i in range(1, len(temp_files)):
            prev_a = f"a{i-1}" if i > 1 else "0:a"
            next_a = f"{i}:a"
            out_a = f"a{i}"
            filter_complex += f"[{prev_a}][{next_a}]acrossfade=d={fade_dur}:c1=tri:c2=tri[{out_a}]; "

        final_v = f"[v{len(temp_files)-1}]"
        final_a = f"[a{len(temp_files)-1}]"
        
        # Assemble command
        full_cmd = ['ffmpeg', '-y']
        for f_path in temp_files:
            full_cmd.extend(['-i', f_path])
        full_cmd.extend([
            '-filter_complex', filter_complex.strip("; "),
            '-map', final_v,
            '-map', final_a,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', str(ffmpeg_crf),
            '-preset', str(ffmpeg_preset),
            '-r', '30',
            '-vsync', 'cfr',
            '-shortest',
            '-movflags', '+faststart',
            output_path
        ])

        proc = subprocess.run(full_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if proc.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 4096:
            return True

        print("❌ FFmpeg crossfade command failed. Trying hard-cut fallback...")
        if proc.stderr:
            print(proc.stderr.decode())

        concat_list_path = os.path.join(temp_dir, f"temp_trailer_concat_{base_name}.txt")
        try:
            with open(concat_list_path, "w", encoding="utf-8") as f:
                for seg_path in temp_files:
                    safe_path = seg_path.replace("'", "'\\''")
                    f.write(f"file '{safe_path}'\n")
            fallback_cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_list_path,
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', str(ffmpeg_crf),
                '-preset', str(ffmpeg_preset),
                '-c:a', 'aac',
                '-ar', '44100',
                '-ac', '2',
                '-movflags', '+faststart',
                output_path
            ]
            fb = subprocess.run(fallback_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if fb.returncode != 0 and fb.stderr:
                print(fb.stderr.decode())
            return fb.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 4096
        finally:
            if os.path.exists(concat_list_path):
                try:
                    os.remove(concat_list_path)
                except Exception:
                    pass
        
    finally:
        # Cleanup temp fragments
        for f in temp_files:
            if os.path.exists(f):
                try: os.remove(f)
                except: pass

def build_fallback_trailer_fragments(shorts, video_duration, max_fragments=6):
    """Create fallback trailer fragments when LLM did not return trailer_fragments."""
    safe_duration = max(0.0, float(video_duration or 0.0))
    out = []

    for clip in (shorts or []):
        try:
            start = max(0.0, float(clip.get("start", 0.0)))
            end = max(start, float(clip.get("end", start)))
        except Exception:
            continue
        if end - start < 2.2:
            continue
        seg_start = max(0.0, start + min(1.2, (end - start) * 0.15))
        seg_end = min(end, seg_start + min(6.0, max(3.0, (end - start) * 0.35)))
        if seg_end - seg_start < 2.2:
            seg_end = min(end, seg_start + 3.0)
        if seg_end - seg_start >= 2.2:
            out.append({
                "start": round(seg_start, 3),
                "end": round(seg_end, 3),
                "reason": "fallback-from-short"
            })
        if len(out) >= max(2, int(max_fragments or 6)):
            break

    # Absolute fallback: pick two moments in the timeline.
    if len(out) < 2 and safe_duration >= 8.0:
        anchors = [safe_duration * 0.20, safe_duration * 0.62]
        for anchor in anchors:
            seg_start = max(0.0, anchor - 1.8)
            seg_end = min(safe_duration, seg_start + 4.2)
            if seg_end - seg_start >= 2.2:
                out.append({
                    "start": round(seg_start, 3),
                    "end": round(seg_end, 3),
                    "reason": "timeline-fallback"
                })
            if len(out) >= 2:
                break

    return out


def normalize_trailer_fragments(fragments, video_duration, max_fragments=6):
    """Normalize and clamp trailer fragments to valid ranges and desired count."""
    safe_duration = max(0.0, float(video_duration or 0.0))
    safe_limit = max(2, int(max_fragments or 6))
    out = []
    seen = set()

    for frag in (fragments or []):
        if not isinstance(frag, dict):
            continue

        start = max(0.0, _safe_float(frag.get("start", 0.0), 0.0))
        end = max(start, _safe_float(frag.get("end", start), start))
        if safe_duration > 0:
            start = min(start, safe_duration)
            end = min(end, safe_duration)

        seg_len = max(0.0, end - start)
        if seg_len < 2.2:
            continue
        if seg_len > 6.2:
            end = start + 6.0
            if safe_duration > 0:
                end = min(end, safe_duration)
            seg_len = max(0.0, end - start)
            if seg_len < 2.2:
                continue

        key = (round(start, 2), round(end, 2))
        if key in seen:
            continue
        seen.add(key)

        out.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "reason": _normalize_space(frag.get("reason", "")) or "llm-selected"
        })
        if len(out) >= safe_limit:
            break

    return out


def merge_trailer_fragments(primary, fallback, target_count=6):
    """Merge primary+fallback fragments without duplicates up to target count."""
    safe_limit = max(2, int(target_count or 6))
    out = []
    seen = set()

    for frag in list(primary or []) + list(fallback or []):
        if not isinstance(frag, dict):
            continue
        start = _safe_float(frag.get("start", 0.0), 0.0)
        end = _safe_float(frag.get("end", start), start)
        key = (round(start, 2), round(end, 2))
        if key in seen:
            continue
        seen.add(key)
        out.append(frag)
        if len(out) >= safe_limit:
            break

    return out

def build_trailer_transcript_from_fragments(
    transcript: Dict[str, Any],
    fragments: List[Dict[str, Any]],
    fade_duration: float = 0.5
) -> Dict[str, Any]:
    """
    Build a synthetic transcript aligned to the final trailer timeline.
    Output timestamps are relative to trailer start (timebase=clip).
    """
    if not isinstance(transcript, dict) or not isinstance(fragments, list) or not fragments:
        return {"text": "", "segments": []}

    words = _extract_transcript_words(transcript)
    if not words:
        return {"text": "", "segments": []}

    source_segments = transcript.get("segments", []) if isinstance(transcript.get("segments"), list) else []
    mapped_words: List[Dict[str, Any]] = []
    timeline_cursor = 0.0

    for frag_idx, frag in enumerate(fragments):
        frag_start = max(0.0, _safe_float((frag or {}).get("start", 0.0), 0.0))
        frag_end = max(frag_start, _safe_float((frag or {}).get("end", frag_start), frag_start))
        frag_duration = max(0.0, frag_end - frag_start)
        if frag_duration < 0.08:
            continue

        for w in words:
            ws_abs = max(frag_start, _safe_float(w.get("start", frag_start), frag_start))
            we_abs = min(frag_end, _safe_float(w.get("end", ws_abs), ws_abs))
            if we_abs <= frag_start or ws_abs >= frag_end:
                continue
            token = _normalize_space(w.get("word", ""))
            if not token:
                continue
            local_start = max(0.0, ws_abs - frag_start)
            local_end = max(local_start, we_abs - frag_start)
            mapped_words.append({
                "word": token,
                "start": round(timeline_cursor + local_start, 3),
                "end": round(timeline_cursor + local_end, 3),
                "segment_index": int(_safe_float(w.get("segment_index", 0), 0)),
                "fragment_index": frag_idx
            })

        if frag_idx < len(fragments) - 1:
            timeline_cursor += max(0.2, frag_duration - max(0.0, float(fade_duration or 0.0)))
        else:
            timeline_cursor += frag_duration

    if not mapped_words:
        return {"text": "", "segments": []}

    mapped_words.sort(key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))))

    speaker_by_seg_idx: Dict[int, str] = {}
    for idx, seg in enumerate(source_segments):
        if not isinstance(seg, dict):
            continue
        speaker = _normalize_space(seg.get("speaker", ""))
        if speaker:
            speaker_by_seg_idx[idx] = speaker

    segments_out: List[Dict[str, Any]] = []
    current_words: List[Dict[str, Any]] = []

    def flush_segment():
        nonlocal current_words
        if not current_words:
            return
        start = max(0.0, _safe_float(current_words[0].get("start", 0.0), 0.0))
        end = max(start, _safe_float(current_words[-1].get("end", start), start))
        text = _normalize_space(" ".join(str(w.get("word", "")).strip() for w in current_words))
        if not text:
            current_words = []
            return
        speaker_counts: Dict[str, int] = {}
        for w in current_words:
            seg_idx = int(_safe_float(w.get("segment_index", -1), -1))
            speaker = speaker_by_seg_idx.get(seg_idx, "")
            if speaker:
                speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        speaker = max(speaker_counts.items(), key=lambda x: x[1])[0] if speaker_counts else None
        words_payload = [
            {
                "word": str(w.get("word", "")).strip(),
                "start": round(max(0.0, _safe_float(w.get("start", 0.0), 0.0)), 3),
                "end": round(max(0.0, _safe_float(w.get("end", 0.0), 0.0)), 3),
            }
            for w in current_words
            if str(w.get("word", "")).strip()
        ]
        segments_out.append({
            "segment_index": len(segments_out),
            "start": round(start, 3),
            "end": round(end, 3),
            "text": text,
            "speaker": speaker,
            "words": words_payload
        })
        current_words = []

    max_chars = 28
    max_duration = 2.25
    max_gap = 0.45

    for w in mapped_words:
        ws = max(0.0, _safe_float(w.get("start", 0.0), 0.0))
        we = max(ws, _safe_float(w.get("end", ws), ws))
        token = _normalize_space(w.get("word", ""))
        if not token:
            continue

        if not current_words:
            current_words = [dict(w, start=ws, end=we, word=token)]
            continue

        cur_start = max(0.0, _safe_float(current_words[0].get("start", 0.0), 0.0))
        cur_chars = sum(len(str(item.get("word", "")).strip()) + 1 for item in current_words)
        last_end = max(cur_start, _safe_float(current_words[-1].get("end", cur_start), cur_start))
        gap = ws - last_end
        next_duration = we - cur_start

        should_split = (
            gap > max_gap
            or next_duration > max_duration
            or (cur_chars + len(token)) > max_chars
        )
        if should_split:
            flush_segment()
        current_words.append(dict(w, start=ws, end=we, word=token))

    flush_segment()

    transcript_text = _normalize_space(" ".join(seg.get("text", "") for seg in segments_out if isinstance(seg, dict)))
    return {
        "text": transcript_text,
        "segments": segments_out
    }

def build_trailer_timeline_markers(
    fragments: List[Dict[str, Any]],
    fade_duration: float = 0.5
) -> Dict[str, Any]:
    """
    Build transition markers and fragment ranges in trailer timeline timebase (0-based).
    """
    if not isinstance(fragments, list) or not fragments:
        return {
            "transition_points": [],
            "fragment_ranges": [],
            "timeline_duration": 0.0,
            "fade_duration": max(0.0, float(fade_duration or 0.0))
        }

    safe_fade = max(0.0, float(fade_duration or 0.0))
    valid_fragments: List[Tuple[float, float]] = []
    for frag in fragments:
        if not isinstance(frag, dict):
            continue
        fs = max(0.0, _safe_float(frag.get("start", 0.0), 0.0))
        fe = max(fs, _safe_float(frag.get("end", fs), fs))
        if (fe - fs) < 0.08:
            continue
        valid_fragments.append((fs, fe))

    if not valid_fragments:
        return {
            "transition_points": [],
            "fragment_ranges": [],
            "timeline_duration": 0.0,
            "fade_duration": safe_fade
        }

    transition_points: List[float] = []
    fragment_ranges: List[Dict[str, Any]] = []
    cursor = 0.0

    for idx, (fs, fe) in enumerate(valid_fragments):
        dur = max(0.0, fe - fs)
        out_start = cursor
        out_end = out_start + dur
        fragment_ranges.append({
            "fragment_index": idx,
            "start": round(out_start, 3),
            "end": round(out_end, 3),
            "source_start": round(fs, 3),
            "source_end": round(fe, 3)
        })
        if idx < len(valid_fragments) - 1:
            transition_start = max(0.0, out_end - safe_fade)
            transition_points.append(round(transition_start, 3))
            cursor = transition_start
        else:
            cursor = out_end

    return {
        "transition_points": transition_points,
        "fragment_ranges": fragment_ranges,
        "timeline_duration": round(max(0.0, cursor), 3),
        "fade_duration": safe_fade
    }

def _probe_media_duration_seconds(file_path):
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', file_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return max(0.0, float((result.stdout or "").strip() or 0.0))
    except Exception:
        pass

    # Fallback for environments where ffprobe is unavailable (e.g. some local mac setups).
    try:
        fb = subprocess.run(['ffmpeg', '-i', file_path], capture_output=True, text=True)
        raw = str(fb.stderr or "")
        match = re.search(r"Duration:\s+(\d+):(\d+):(\d+(?:\.\d+)?)", raw)
        if match:
            hh, mm, ss = match.groups()
            return (float(hh) * 3600.0) + (float(mm) * 60.0) + float(ss)
    except Exception:
        pass
    return 0.0


def transcribe_video(video_path, language=None, backend=None, model_name=None, word_timestamps=True, compute_type=None, cpu_threads=0, num_workers=1, hf_token=None, enable_diarization=False):
    """
    Transcribe and Diarize using the selected backend (openai, faster, whisperx).
    """
    backend = str(backend or os.getenv("WHISPER_BACKEND", "faster")).lower().strip()
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
        device = "cpu"
    else:
        device = "cpu" if requested_device not in ("cuda", "cpu") else requested_device

    if not compute_type:
        compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "float16" if device == "cuda" else "int8")

    if backend == "whisperx":
        print(f"🎙️  Transcribing video with WhisperX (model={model_name}, device={device}, compute={compute_type})...")
        import whisperx
        import gc
        
        # 1. Transcribe
        batch_size = 16 if device == "cuda" else 4
        model = whisperx.load_model(model_name, device, compute_type=compute_type)
        
        audio = whisperx.load_audio(video_path)
        result = model.transcribe(audio, batch_size=batch_size, language=language)
        detected_language = result["language"]
        
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            
        # 2. Align (For Perfect Karaoke)
        print("🎯 Aligning words with audio for exact timestamps...")
        model_a, metadata = whisperx.load_align_model(language_code=detected_language, device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        
        del model_a
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            
        # 3. Diarize (Detect Speakers) if Token Provided AND explicitly enabled
        if hf_token and enable_diarization:
            print("👥 Detecting speakers (Diarization)...")
            try:
                diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
                diarize_segments = diarize_model(audio, min_speakers=1, max_speakers=8)
                result = whisperx.assign_word_speakers(diarize_segments, result)
            except Exception as e:
                print(f"⚠️ Diarization failed: {e}. Falling back to default speakers.")
        else:
            if not enable_diarization:
                print("ℹ️ Diarization disabled. Skipping pyannote (fast mode).")
            else:
                print("⚠️ No HuggingFace token provided. Skipping Speaker Diarization.")
            
        # 4. Format Output match existing schema
        full_text = ""
        transcript_segments = []
        
        for segment in result["segments"]:
            seg_dict = {
                'text': segment.get("text", "").strip(),
                'start': segment.get("start", 0.0),
                'end': segment.get("end", 0.0),
                'speaker': segment.get("speaker", "SPEAKER_00"),
                'words': []
            }
            
            if "words" in segment:
                for word in segment["words"]:
                    if "start" not in word or "end" not in word:
                        continue
                    seg_dict['words'].append({
                        'word': word.get("word", ""),
                        'start': word["start"],
                        'end': word["end"],
                        'probability': word.get("score", 0.0),
                        'speaker': word.get("speaker", segment.get("speaker", "SPEAKER_00"))
                    })
            
            transcript_segments.append(seg_dict)
            full_text += seg_dict['text'] + " "

        return {
            'text': full_text.strip(),
            'segments': transcript_segments,
            'language': detected_language
        }

    # Fallback to faster-whisper with runtime cache, GPU/CPU fallback, and safer defaults.
    if model_name:
        os.environ["WHISPER_MODEL"] = str(model_name)
    if compute_type:
        os.environ["WHISPER_COMPUTE_TYPE"] = str(compute_type)
    if cpu_threads:
        os.environ["WHISPER_CPU_THREADS"] = str(cpu_threads)
    if num_workers:
        os.environ["WHISPER_NUM_WORKERS"] = str(num_workers)

    print("🎙️  Transcribing video with Faster-Whisper runtime manager...")
    segments, info, runtime_meta = transcribe_with_runtime(
        video_path,
        word_timestamps=word_timestamps,
        language=language
    )
    print(
        "   Runtime: "
        f"model={runtime_meta.get('model')}, device={runtime_meta.get('device')}, "
        f"compute={runtime_meta.get('compute_type')}, beam={runtime_meta.get('beam_size')}, "
        f"vad={runtime_meta.get('vad_filter')}"
    )
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

def _is_groq_rate_limit_error(err):
    msg = str(err or "").lower()
    if not msg:
        return False
    return (
        "rate_limit_exceeded" in msg
        or "rate limit reached" in msg
        or "error code: 429" in msg
        or "too many requests" in msg
    )

def _extract_retry_after_seconds(err):
    msg = str(err or "")
    match = re.search(r"try again in\s*([0-9]+(?:\.[0-9]+)?)s", msg, re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None

def get_viral_clips(
    transcript_result,
    video_duration,
    max_clips=None,
    clip_length_target=None,
    trailer_fragments_target=6,
    model_name='gemini-2.5-flash-lite',
    llm_provider='gemini',
    groq_api_key=None
):
    print(f"🤖  Analyzing with {llm_provider.capitalize()}...")
    
    if llm_provider == 'gemini':
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ Error: GEMINI_API_KEY not found in environment variables.")
            return None
        client = genai.Client(api_key=api_key)
    elif llm_provider == 'groq':
        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            print("❌ Error: GROQ_API_KEY not found in environment variables.")
            return None
        client = Groq(api_key=api_key)
    else:
        print(f"❌ Error: Unsupported LLM provider: {llm_provider}")
        return None
    
    print(f"🤖  Initializing {llm_provider.capitalize()} with model: {model_name}")

    # Extract words
    words = []
    for segment in transcript_result['segments']:
        for word in segment.get('words', []):
            words.append({
                'w': word['word'],
                's': word['start'],
                'e': word['end']
            })

    # Compact timeline (one line per ~10 words) instead of per-word JSON:
    # ~4x fewer tokens on long videos with no loss of cut precision, since
    # clip boundaries are refined locally afterwards by
    # postprocess_shorts_with_transcript().
    def _compact_word_timeline(word_list, group_size=10):
        timeline_lines = []
        for idx in range(0, len(word_list), group_size):
            group = word_list[idx:idx + group_size]
            text = " ".join(str(w['w']).strip() for w in group)
            timeline_lines.append(f"[{group[0]['s']:.2f}-{group[-1]['e']:.2f}] {text}")
        return "\n".join(timeline_lines)

    max_clips_rule = ""
    if max_clips:
        max_clips_rule = f"IMPORTANT: Return at most {max_clips} clips."
    length_rule = clip_length_guidance(clip_length_target)
    trailer_target = max(2, min(12, int(trailer_fragments_target or 6)))
    trailer_min = max(2, trailer_target - 1)
    trailer_max = min(12, trailer_target + 1)
    trailer_rule = (
        "TRAILER RULE: Identify additional \"explosive\" or \"hooky\" fragments (3-6 seconds each) "
        f"to create a Super Trailer summary. Prefer {trailer_target} fragments (allowed range: {trailer_min}-{trailer_max})."
    )

    prompt = GEMINI_PROMPT_TEMPLATE.format(
        video_duration=video_duration,
        transcript_text=json.dumps(transcript_result['text']),
        words_json=_compact_word_timeline(words),
        max_clips_rule=max_clips_rule,
        clip_length_rule=length_rule,
        trailer_fragments_rule=trailer_rule
    )

    max_attempts = 2 if llm_provider == 'groq' else 1
    for attempt in range(1, max_attempts + 1):
        try:
            if llm_provider == 'gemini':
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={'response_mime_type': 'application/json'}
                )
                result_json = json.loads(response.text)
            elif llm_provider == 'groq':
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=model_name,
                    response_format={"type": "json_object"},
                )
                result_json = json.loads(chat_completion.choices[0].message.content)
            else:
                return None

            if max_clips and isinstance(result_json.get('shorts'), list):
                result_json['shorts'] = result_json['shorts'][:max_clips]

            return result_json
        except Exception as e:
            if llm_provider == 'groq' and _is_groq_rate_limit_error(e) and attempt < max_attempts:
                retry_after = _extract_retry_after_seconds(e)
                wait_seconds = retry_after if retry_after is not None else 30.0
                wait_seconds = max(2.0, min(wait_seconds + 1.0, 90.0))
                print(
                    f"⚠️ Groq rate limit alcanzado. Esperando {wait_seconds:.1f}s y reintentando "
                    f"({attempt}/{max_attempts})..."
                )
                time.sleep(wait_seconds)
                continue

            if llm_provider == 'groq':
                gemini_key = os.getenv("GEMINI_API_KEY")
                fallback_model = os.getenv("GROQ_FALLBACK_GEMINI_MODEL", "gemini-2.5-flash-lite")
                if gemini_key:
                    print(
                        f"⚠️ Groq no disponible ({e}). Fallback automatico a Gemini ({fallback_model})."
                    )
                    return get_viral_clips(
                        transcript_result=transcript_result,
                        video_duration=video_duration,
                        max_clips=max_clips,
                        clip_length_target=clip_length_target,
                        model_name=fallback_model,
                        llm_provider='gemini',
                        groq_api_key=None
                    )

            print(f"❌ {llm_provider.capitalize()} Error: {e}")
            return None

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
    parser.add_argument('--whisper-backend', type=str, default=None, help="Whisper backend: openai|faster|whisperx.")
    parser.add_argument('--enable-diarization', action='store_true', default=False, help="Run pyannote speaker diarization (slow on CPU, requires HF token).")
    parser.add_argument('--whisper-model', type=str, default=None, help="Whisper model: tiny|base|small|medium|large|large-v2|large-v3.")
    parser.add_argument('--word-timestamps', type=str, default="true", help="true/false for word-level timestamps.")
    parser.add_argument('--ffmpeg-preset', type=str, default=DEFAULT_FFMPEG_PRESET, help="FFmpeg preset: ultrafast|fast|medium.")
    parser.add_argument('--ffmpeg-crf', type=int, default=DEFAULT_FFMPEG_CRF, help="FFmpeg CRF quality (lower=better).")
    parser.add_argument('--aspect-ratio', type=str, default=DEFAULT_ASPECT_RATIO, help="Output aspect ratio: 9:16 or 16:9.")
    parser.add_argument('--clip-length-target', type=str, default=None, help="Preferred clip length profile: short|balanced|long.")
    parser.add_argument('--style-template', type=str, default=None, help="UI template id used for this generation (metadata only).")
    parser.add_argument('--content-profile', type=str, default=None, help="Content profile selected in UI (metadata only).")
    parser.add_argument('--llm-model', type=str, default='gemini-2.5-flash-lite', help="Gemini model name.")
    parser.add_argument('--llm-provider', type=str, default='gemini', help="LLM provider: gemini or groq.")
    parser.add_argument('--groq-api-key', type=str, default=None, help="Groq API Key.")
    parser.add_argument('--hf-token', type=str, default=None, help="HuggingFace token for WhisperX Diarization.")
    parser.add_argument('--build-trailer', action='store_true', help="If true, generates a Super Trailer from identified fragments.")
    parser.add_argument('--trailer-only', action='store_true', help="If true, skips clip rendering and generates only the Super Trailer.")
    parser.add_argument('--trailer-fragments-target', type=int, default=6, help="Desired number of highlighted segments for Super Trailer (2-12).")
    parser.add_argument('--tight-edit-preset', type=str, default=DEFAULT_TIGHT_EDIT_PRESET, choices=['off', 'balanced', 'aggressive', 'very_aggressive'], help="Remove pauses/filler words before vertical rendering.")
    
    args = parser.parse_args()
    args.tight_edit_preset = normalize_tight_edit_preset(args.tight_edit_preset, DEFAULT_TIGHT_EDIT_PRESET)

    script_start_time = time.time()

    if args.max_clips:
        args.max_clips = max(1, min(15, args.max_clips))
    args.trailer_fragments_target = max(2, min(12, int(args.trailer_fragments_target or 6)))
    if args.trailer_only:
        args.build_trailer = True
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
            word_timestamps=args.word_timestamps,
            hf_token=args.hf_token,
            enable_diarization=args.enable_diarization
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
            clip_length_target=args.clip_length_target,
            trailer_fragments_target=args.trailer_fragments_target,
            model_name=args.llm_model,
            llm_provider=args.llm_provider,
            groq_api_key=args.groq_api_key
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
                "content_profile": (str(args.content_profile).strip() if args.content_profile else None),
                "trailer_fragments_target": args.trailer_fragments_target
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

            shorts_for_trailer = list(clips_data.get('shorts', []))

            # 5. Process each clip (skip in trailer-only mode)
            if not args.trailer_only:
                for i, clip in enumerate(clips_data['shorts']):
                    start = clip['start']
                    end = clip['end']
                    print(f"\n🎬 Processing Clip {i+1}: {start}s - {end}s")
                    print(f"   Title: {clip.get('video_title_for_youtube_short', 'No Title')}")
                    
                    # Cut clip
                    clip_filename = f"{video_title}_clip_{i+1}.mp4"
                    clip_uncut_filename = f"{video_title}_clip_{i+1}_uncut.mp4"
                    clip_final_path = os.path.join(output_dir, clip_filename)
                    clip_uncut_path = os.path.join(output_dir, clip_uncut_filename)
                    
                    # ffmpeg cut
                    # Using re-encoding for precision as requested by strict seconds
                    # Save directly to the uncut path first so we preserve the original frame.
                    tight_edit_plan = build_tight_edit_plan(transcript, start, end, args.tight_edit_preset)
                    keep_segments = tight_edit_plan.get("keep_segments") or [(start, end)]
                    if tight_edit_plan.get("compacted"):
                        print(
                            "   ✂️ Tight edit:",
                            f"{len(keep_segments)} keep segment(s),",
                            f"preset={tight_edit_plan.get('preset')},",
                            f"new duration ≈ {tight_edit_plan.get('output_duration', end - start):.2f}s"
                        )
                        render_keep_segments(
                            input_video,
                            keep_segments,
                            clip_uncut_path,
                            ffmpeg_preset=str(args.ffmpeg_preset),
                            crf=str(args.ffmpeg_crf),
                            thread_args=ffmpeg_thread_args(include_filter_threads=True),
                            subprocess_kwargs=subprocess_priority_kwargs(),
                        )
                        clip["display_duration"] = tight_edit_plan.get("output_duration", round(end - start, 3))
                        clip["tight_edit_preset"] = tight_edit_plan.get("preset")
                        clip["tight_edit_removed_ranges"] = [
                            {"start": range_start, "end": range_end}
                            for range_start, range_end in tight_edit_plan.get("remove_ranges", [])
                        ]
                        clip["tight_edit_keep_segments"] = [
                            {"start": segment_start, "end": segment_end}
                            for segment_start, segment_end in keep_segments
                        ]
                    else:
                        # The uncut file is an intermediate that gets re-encoded by the
                        # vertical render, so cap its CRF at 18 to limit generational loss.
                        intermediate_crf = min(int(args.ffmpeg_crf), 18)
                        cut_command = [
                            'ffmpeg', '-y',
                            '-ss', str(start),
                            '-to', str(end),
                            '-i', input_video,
                            *ffmpeg_thread_args(include_filter_threads=False),
                            *video_encoder_args(args.ffmpeg_preset, intermediate_crf),
                            '-pix_fmt', 'yuv420p',
                            '-c:a', 'aac',
                            '-movflags', '+faststart',
                            clip_uncut_path
                        ]
                        cut_result = subprocess.run(cut_command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, **subprocess_priority_kwargs())
                        if cut_result.returncode != 0:
                            print(f"   ❌ Failed to cut clip {i+1}:", cut_result.stderr.decode(errors='ignore')[-500:])
                            continue
                        clip.pop("display_duration", None)
                        clip.pop("tight_edit_preset", None)
                        clip.pop("tight_edit_removed_ranges", None)
                        clip.pop("tight_edit_keep_segments", None)
                    
                    # Process vertical from the uncut source instead of input_video to save processing time
                    # but input_video would also work if uncut_path is deleted. Let's use uncut_path.
                    success = process_video_to_vertical(clip_uncut_path, clip_final_path, args.ffmpeg_preset, args.ffmpeg_crf, args.aspect_ratio)
                    
                    if success:
                        print(f"   ✅ Clip {i+1} ready: {clip_final_path}")
                        print(f"   ✅ Uncut Clip {i+1} saved: {clip_uncut_path}")
                with open(metadata_file, 'w') as f:
                    json.dump(clips_data, f, indent=2)
            else:
                print("🎯 Trailer-only mode: omitiendo render de clips individuales.")

            # 6. Optional Super Trailer
            target_trailer_fragments = max(2, int(args.trailer_fragments_target or 6))
            trailer_fragments = normalize_trailer_fragments(
                clips_data.get("trailer_fragments", []),
                duration,
                max_fragments=target_trailer_fragments
            )
            if len(trailer_fragments) < target_trailer_fragments:
                fallback_fragments = normalize_trailer_fragments(
                    build_fallback_trailer_fragments(
                        shorts_for_trailer,
                        duration,
                        max_fragments=target_trailer_fragments
                    ),
                    duration,
                    max_fragments=target_trailer_fragments
                )
                trailer_fragments = merge_trailer_fragments(
                    trailer_fragments,
                    fallback_fragments,
                    target_count=target_trailer_fragments
                )
            if trailer_fragments:
                clips_data["trailer_fragments"] = trailer_fragments

            if args.build_trailer:
                print(
                    f"🎞️ Super Trailer configurado para {target_trailer_fragments} segmentos "
                    f"(disponibles: {len(trailer_fragments)})."
                )

            if args.build_trailer and len(trailer_fragments) >= 2:
                print("\n⚡ Generating Super Trailer...")
                trailer_uncut_path = os.path.join(output_dir, f"{video_title}_trailer_uncut.mp4")
                trailer_final_path = os.path.join(output_dir, f"{video_title}_trailer.mp4")
                
                # Build the horizontal trailer first
                ok = build_super_trailer(input_video, trailer_fragments, trailer_uncut_path, args.ffmpeg_preset, args.ffmpeg_crf)
                if ok:
                    print(f"   ✅ Uncut Trailer ready: {trailer_uncut_path}")
                    # Apply AutoCrop to the trailer
                    ok_v = process_video_to_vertical(trailer_uncut_path, trailer_final_path, args.ffmpeg_preset, args.ffmpeg_crf, args.aspect_ratio)
                    if ok_v:
                        print(f"   ✅ Super Trailer ready: {trailer_final_path}")
                        clips_data['latest_trailer_url'] = f"/videos/{os.path.basename(output_dir)}/{os.path.basename(trailer_final_path)}"
                        # Update metadata one last time
                        with open(metadata_file, 'w') as f:
                            json.dump(clips_data, f, indent=2)
                else:
                    print("   ❌ Failed to build Super Trailer fragments.")

            if args.trailer_only:
                trailer_url = str(clips_data.get("latest_trailer_url", "") or "").strip()
                trailer_duration = 0.0
                if trailer_url:
                    trailer_name = trailer_url.split("/")[-1]
                    trailer_path = os.path.join(output_dir, trailer_name)
                    if os.path.exists(trailer_path):
                        trailer_duration = _probe_media_duration_seconds(trailer_path)
                trailer_timeline_meta = build_trailer_timeline_markers(
                    fragments=trailer_fragments,
                    fade_duration=0.5
                )
                trailer_transcript = build_trailer_transcript_from_fragments(
                    transcript=transcript,
                    fragments=trailer_fragments,
                    fade_duration=0.5
                )
                if trailer_duration <= 0.0:
                    try:
                        transcript_end = max(
                            (_safe_float(seg.get("end", 0.0), 0.0) for seg in trailer_transcript.get("segments", [])),
                            default=0.0
                        )
                        trailer_duration = max(
                            transcript_end,
                            _safe_float(trailer_timeline_meta.get("timeline_duration", 0.0), 0.0)
                        )
                    except Exception:
                        trailer_duration = 0.0

                base_clip = shorts_for_trailer[0] if shorts_for_trailer else {}
                trailer_title = (
                    (base_clip.get("video_title_for_youtube_short") if isinstance(base_clip, dict) else None)
                    or "Super Trailer"
                )
                trailer_desc = (
                    (base_clip.get("video_description_for_tiktok") if isinstance(base_clip, dict) else None)
                    or "Resumen rápido con los mejores momentos."
                )

                synthetic_clip = {
                    "clip_index": 0,
                    "start": 0.0,
                    "end": round(max(3.0, trailer_duration or 0.0), 3),
                    "aspect_ratio": args.aspect_ratio,
                    "virality_score": int((base_clip.get("virality_score", 90) if isinstance(base_clip, dict) else 90) or 90),
                    "selection_confidence": float((base_clip.get("selection_confidence", 0.9) if isinstance(base_clip, dict) else 0.9) or 0.9),
                    "score_reason": "Montaje resumen de momentos clave.",
                    "video_title_for_youtube_short": str(trailer_title),
                    "video_description_for_tiktok": str(trailer_desc),
                    "video_description_for_instagram": str(trailer_desc),
                    "video_url": trailer_url or None,
                    "title_variants": (base_clip.get("title_variants", []) if isinstance(base_clip, dict) else []),
                    "social_variants": (base_clip.get("social_variants", []) if isinstance(base_clip, dict) else []),
                    "is_trailer": True,
                    "transition_points": trailer_timeline_meta.get("transition_points", []),
                    "fragment_ranges": trailer_timeline_meta.get("fragment_ranges", []),
                    "transition_duration": _safe_float(trailer_timeline_meta.get("fade_duration", 0.5), 0.5),
                }
                if trailer_transcript.get("segments"):
                    trailer_text = _normalize_space(trailer_transcript.get("text", ""))
                    synthetic_clip["transcript_segments"] = trailer_transcript.get("segments", [])
                    synthetic_clip["transcript_text"] = trailer_text
                    synthetic_clip["transcript_excerpt"] = trailer_text[:420]
                    synthetic_clip["transcript_timebase"] = "clip"
                clips_data["shorts"] = [synthetic_clip]
                with open(metadata_file, 'w') as f:
                    json.dump(clips_data, f, indent=2)

    # Clean up original if requested
    if args.url and not args.keep_original and os.path.exists(input_video):
        os.remove(input_video)
        print(f"🗑️  Cleaned up downloaded video.")
    elif generated_audio_canvas and os.path.exists(input_video):
        os.remove(input_video)
        print("🗑️  Cleaned up temporary audio canvas video.")

    total_time = time.time() - script_start_time
    print(f"\n⏱️  Total execution time: {total_time:.2f}s")
