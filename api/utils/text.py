import re
import json
import unicodedata
from typing import List, Optional, Any

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def _safe_int(value: Any, default: int = 0) -> int:
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
