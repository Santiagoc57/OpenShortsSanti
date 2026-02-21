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
try:
    from scenedetect import detect_scenes as sd_detect_scenes
except ImportError:
    try:
        from scenedetect import detect as sd_detect_scenes
    except ImportError:
        sd_detect_scenes = None

try:
    from scenedetect import ContentDetector
except ImportError:
    from scenedetect.detectors import ContentDetector
from ultralytics import YOLO
import torch
import os
import numpy as np
from tqdm import tqdm
from autocrop import analyze_video_for_autocrop, is_variable_frame_rate, normalize_to_cfr
import numpy as np
from tqdm import tqdm
import yt_dlp
import mediapipe as mp
# import whisper (replaced by faster_whisper inside function)
from google import genai
from groq import Groq
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
LANGUAGE RULE (STRICT): all textual fields MUST be in Spanish (español neutro): score_reason, topic_tags, title_variants, social_variants.
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
    """Detect scene boundaries using PySceneDetect (Modern API)."""
    try:
        if sd_detect_scenes is None:
            raise ImportError("PySceneDetect detect API not available")
        scene_list = sd_detect_scenes(video_path, ContentDetector(), show_progress=False)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return scene_list, fps
    except Exception as e:
        print(f"❌ Error in PySceneDetect: {str(e)}")
        return [], 0.0

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
    Core logic to process video using AutoCrop scene detection and YOLOv8 tracking
    targeting a configurable aspect ratio.
    """
    script_start_time = time.time()
    
    # Define temporary file paths based on the output name
    base_name = os.path.splitext(final_output_video)[0]
    temp_video_output = f"{base_name}_temp_video.mp4"
    temp_audio_output = f"{base_name}_temp_audio.aac"
    temp_cfr_input = f"{base_name}_temp_cfr_input.mp4"
    
    # Clean up previous temp files if they exist
    for f in [temp_video_output, temp_audio_output, final_output_video, temp_cfr_input]:
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

    print("   🧠 Step 1: Analyzing Scenes and Tracking Targets...")
    # This does scene detection and middle-frame YOLO analysis
    scenes_plan = analyze_video_for_autocrop(input_video)
    
    if not scenes_plan:
        print("   ❌ Failed to analyze scenes. Returning original video unchanged.")
        try:
            shutil.copyfile(input_video, final_output_video)
            return os.path.exists(final_output_video)
        except Exception as copy_err:
            print(f"   ❌ Fallback copy failed: {copy_err}")
            return False
        
    print(f"   ✅ Target Plan Generated for {len(scenes_plan)} scenes.")

    print("\n   ✂️ Step 2: Processing video frames...")
    
    cap = cv2.VideoCapture(input_video)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    aspect_label, target_ratio = normalize_aspect_ratio(aspect_ratio)
    OUTPUT_WIDTH, OUTPUT_HEIGHT = compute_output_dimensions(original_width, original_height, target_ratio)
    print(f"   Target aspect ratio: {aspect_label} ({OUTPUT_WIDTH}x{OUTPUT_HEIGHT})")
    
    command = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}', '-pix_fmt', 'bgr24',
        '-r', str(fps), '-i', '-', '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-r', str(fps), '-vsync', 'cfr',
        '-preset', str(ffmpeg_preset), '-crf', str(ffmpeg_crf), '-an',
        '-movflags', '+faststart', temp_video_output
    ]

    ffmpeg_process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    frame_number = 0
    current_scene_index = 0
    num_scenes = len(scenes_plan)
    last_output_frame = None
    dropped_frames = 0
    
    with tqdm(total=total_frames, desc=f"   Processing [scene 1/{num_scenes}]", file=sys.stdout) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_time_sec = frame_number / fps
            
            # Update Scene Index based on time
            if current_scene_index < len(scenes_plan) - 1:
                if frame_time_sec >= scenes_plan[current_scene_index + 1]['start_sec']:
                    current_scene_index += 1
                    pbar.set_description(f"   Processing [scene {current_scene_index + 1}/{num_scenes}]")

            scene_data = scenes_plan[current_scene_index]
            strategy = scene_data['strategy']
            crop_coords = scene_data['crop_coords']

            try:
                if strategy == 'TRACK' and crop_coords:
                    x, y, w, h = crop_coords
                    cropped = frame[y:y+h, x:x+w]
                    # Resize to exact output size
                    output_frame = cv2.resize(cropped, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
                else:  
                    # LETTERBOX
                    scale_factor = OUTPUT_WIDTH / original_width
                    scaled_height = int(original_height * scale_factor)
                    scaled_frame = cv2.resize(frame, (OUTPUT_WIDTH, scaled_height))

                    output_frame = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)
                    y_offset = (OUTPUT_HEIGHT - scaled_height) // 2
                    output_frame[y_offset:y_offset + scaled_height, :] = scaled_frame
                    
                last_output_frame = output_frame
            except Exception as e:
                dropped_frames += 1
                if last_output_frame is not None:
                    output_frame = last_output_frame
                else:
                    output_frame = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)

            ffmpeg_process.stdin.write(output_frame.tobytes())
            frame_number += 1
            pbar.update(1)
            
    if dropped_frames > 0:
        print(f"  ⚠️  {dropped_frames} frame(s) could not be processed and were duplicated.")
        
    ffmpeg_process.stdin.close()
    stderr_output = ffmpeg_process.stderr.read().decode()
    ffmpeg_process.wait()
    cap.release()

    if ffmpeg_process.returncode != 0:
        print("\n   ❌ FFmpeg frame processing failed.")
        print("   Stderr:", stderr_output)
        return False

    print("\n   🔊 Step 3: Extracting audio...")
    audio_extract_command = [
        'ffmpeg', '-y', '-i', input_video, '-vn', '-acodec', 'copy', temp_audio_output
    ]
    try:
        subprocess.run(audio_extract_command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        print("\n   ❌ Audio extraction failed (maybe no audio?). Proceeding without audio.")
        pass

    print("\n   ✨ Step 4: Merging...")
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
    for f in [temp_video_output, temp_audio_output, temp_cfr_input]:
        if os.path.exists(f): 
            try: os.remove(f)
            except: pass
    
    return True
def build_super_trailer(input_video, fragments, output_path, ffmpeg_preset="fast", ffmpeg_crf=23):
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
        words_json=json.dumps(words),
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
    parser.add_argument('--whisper-backend', type=str, default=None, help="Whisper backend: openai|faster.")
    parser.add_argument('--whisper-model', type=str, default=None, help="Whisper model: tiny|base|small|medium|large|large-v2|large-v3.")
    parser.add_argument('--word-timestamps', type=str, default="true", help="true/false for word-level timestamps.")
    parser.add_argument('--ffmpeg-preset', type=str, default="fast", help="FFmpeg preset: ultrafast|fast|medium.")
    parser.add_argument('--ffmpeg-crf', type=int, default=23, help="FFmpeg CRF quality (lower=better).")
    parser.add_argument('--aspect-ratio', type=str, default=DEFAULT_ASPECT_RATIO, help="Output aspect ratio: 9:16 or 16:9.")
    parser.add_argument('--clip-length-target', type=str, default=None, help="Preferred clip length profile: short|balanced|long.")
    parser.add_argument('--style-template', type=str, default=None, help="UI template id used for this generation (metadata only).")
    parser.add_argument('--content-profile', type=str, default=None, help="Content profile selected in UI (metadata only).")
    parser.add_argument('--llm-model', type=str, default='gemini-2.5-flash-lite', help="Gemini model name.")
    parser.add_argument('--llm-provider', type=str, default='gemini', help="LLM provider: gemini or groq.")
    parser.add_argument('--groq-api-key', type=str, default=None, help="Groq API Key.")
    parser.add_argument('--build-trailer', action='store_true', help="If true, generates a Super Trailer from identified fragments.")
    parser.add_argument('--trailer-only', action='store_true', help="If true, skips clip rendering and generates only the Super Trailer.")
    parser.add_argument('--trailer-fragments-target', type=int, default=6, help="Desired number of highlighted segments for Super Trailer (2-12).")
    
    args = parser.parse_args()

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
                    cut_command = [
                        'ffmpeg', '-y', 
                        '-ss', str(start), 
                        '-to', str(end), 
                        '-i', input_video,
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', str(args.ffmpeg_crf), '-preset', str(args.ffmpeg_preset),
                        '-c:a', 'aac',
                        '-movflags', '+faststart',
                        clip_uncut_path
                    ]
                    subprocess.run(cut_command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    
                    # Process vertical from the uncut source instead of input_video to save processing time
                    # but input_video would also work if uncut_path is deleted. Let's use uncut_path.
                    success = process_video_to_vertical(clip_uncut_path, clip_final_path, args.ffmpeg_preset, args.ffmpeg_crf, args.aspect_ratio)
                    
                    if success:
                        print(f"   ✅ Clip {i+1} ready: {clip_final_path}")
                        print(f"   ✅ Uncut Clip {i+1} saved: {clip_uncut_path}")
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
