import os
import json
import time
import zlib
import math
import re
from typing import List, Optional, Any, Dict, Tuple
import unicodedata

from ..core.config import TITLE_REWRITE_MODELS, SOCIAL_REWRITE_MODELS, VIRAL_TITLE_CRITERIA
from ..utils.text import (
    _normalize_space, _extract_generated_text, _sanitize_short_title,
    _title_fingerprint, _dedupe_title_candidates, _parse_title_variants_payload,
    _social_fingerprint, _dedupe_social_candidates, _parse_social_variants_payload,
    _sanitize_social_copy
)

def _is_gemini_model_unavailable_error(err: Exception) -> bool:
    msg = str(err or "").lower()
    if not msg:
        return False
    unavailable_keywords = ["models/", "not found", "not supported", "for api version", "unknown model"]
    return any(keyword in msg for keyword in unavailable_keywords)

def _extract_viral_title_keyword(topic_tags: List[str], transcript_excerpt: str) -> str:
    if isinstance(topic_tags, list) and topic_tags:
        for tag in topic_tags:
            clean = _sanitize_short_title(tag)
            if clean and len(clean) >= 4:
                return clean
    words = re.findall(r"[a-zA-ZÀ-ÿ0-9]{5,}", str(transcript_excerpt or ""))
    for word in words[:15]:
        return word
    return "esto"

def _build_viral_title_from_criterion(criterion: str, outcome: str, keyword: str, number_hint: int = 7) -> str:
    safe_keyword = _sanitize_short_title(keyword or "esto", max_chars=30).lower() or "esto"
    safe_outcome = _sanitize_short_title(outcome or safe_keyword, max_chars=70) or safe_keyword
    safe_number = max(2, min(30, int(number_hint or 7)))

    templates = {
        "Nunca vas a creer esto...": f"Nunca vas a creer esto: {safe_outcome}",
        "No pierdas tu tiempo...": f"No pierdas tu tiempo: {safe_keyword} no funciona como te dijeron",
        "Esto fue un verdadero shock...": f"Esto fue un verdadero shock: {safe_outcome}",
        "Cómo lograr [resultado específico] en [#] tiempo": f"Cómo lograr {safe_outcome} en {safe_number} minutos",
        "Evita [cosa] a toda costa": f"Evita {safe_keyword} a toda costa",
        "Por esto eres tan malo en...": f"Por esto eres tan malo en {safe_keyword}",
        "Opinión impopular sobre X": f"Opinión impopular sobre {safe_keyword}",
        "Esto me volvió loco": f"Esto me volvió loco: {safe_outcome}",
        "La estrategia secreta para...": f"La estrategia secreta para {safe_outcome}",
        "Si una persona más me dice...": f"Si una persona más me dice '{safe_keyword}', exploto",
        "Esto no tiene ningún sentido": f"Esto no tiene ningún sentido: {safe_outcome}",
        "¿Por qué esto es tan difícil?": f"¿Por qué {safe_keyword} es tan difícil?",
        "Simplemente no puedo hacerlo": f"Simplemente no puedo hacerlo: {safe_keyword}",
        "Mi mayor arrepentimiento es...": f"Mi mayor arrepentimiento es no entender {safe_keyword}",
        "Esto me tuvo estancado": f"Esto me tuvo estancado: {safe_keyword}",
        "¿Cómo es que esto no funciona?": f"¿Cómo es que {safe_keyword} no funciona?",
        "Se me cayó la mandíbula cuando...": f"Se me cayó la mandíbula cuando {safe_outcome.lower()}",
        "La increíble razón por la que...": f"La increíble razón por la que {safe_keyword} falla",
    }
    return templates.get(criterion, _sanitize_short_title(safe_outcome or "Momento clave del video"))

def _build_viral_title_candidates(current_title: str, transcript_excerpt: str, topic_tags: List[str], seed: int) -> List[str]:
    keyword = _extract_viral_title_keyword(topic_tags, transcript_excerpt) or "esto"
    outcome = _sanitize_short_title(current_title, max_chars=72)
    if not outcome:
        transcript_hint = _normalize_space(str(transcript_excerpt or "")).split(".")[0].strip()
        outcome = _sanitize_short_title(transcript_hint, max_chars=72)
    if not outcome:
        outcome = keyword

    criteria = list(VIRAL_TITLE_CRITERIA)
    if criteria:
        offset = abs(int(seed or 0)) % len(criteria)
        criteria = criteria[offset:] + criteria[:offset]

    number_hint = 3 + (abs(int(seed or 0)) % 10)
    raw_candidates = [
        _build_viral_title_from_criterion(criterion, outcome, keyword, number_hint)
        for criterion in criteria
    ]
    return _dedupe_title_candidates(raw_candidates)

def _build_fallback_title(current_title: str, transcript_excerpt: str, topic_tags: List[str], avoid_title: str) -> str:
    clean_current = _sanitize_short_title(current_title)
    clean_avoid = _sanitize_short_title(avoid_title)
    avoid_fp = _title_fingerprint(clean_avoid)
    seed_raw = f"{clean_current}|{transcript_excerpt}|{clean_avoid}|{int(time.time())}"
    seed = zlib.crc32(seed_raw.encode("utf-8"))
    candidates = _build_viral_title_candidates(clean_current, transcript_excerpt, topic_tags, seed)
    for candidate in candidates:
        if avoid_fp and _title_fingerprint(candidate) == avoid_fp:
            continue
        return candidate
    return _sanitize_short_title(clean_current or "Momento clave del video")

def _generate_rewritten_title(current_title: str, transcript_excerpt: str, social_excerpt: str, topic_tags: List[str], avoid_title: str, api_key: Optional[str]) -> str:
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
            criteria_line = "\n".join(f"- {item}" for item in VIRAL_TITLE_CRITERIA)
            prompt = (
                "Eres un Director Creativo experto en videos cortos virales (Shorts/Reels). Tu objetivo es reescribir el título o 'gancho de apertura' para que sea magnético e imposible de ignorar.\n"
                "Hagamos esto paso a paso:\n"
                "1. Lee el título actual, el contexto de la transcripción y el contexto social para entender sobre qué trata el video.\n"
                "2. Identifica lo más nuevo, sorprendente o contraintuitivo.\n"
                "3. Escribe un título directo, en español neutro, de 45 a 95 caracteres. Cero introducciones o explicaciones. Máximo 12 palabras para que el lector lo capte al instante.\n"
                "4. Usa palabras simples y emocionalmente cargadas.\n"
                "Devuelve SOLO el título final, en una sola línea, sin comillas.\n"
                "Reglas críticas: sin hashtags, emoji opcional (máximo 1), NADA de clickbait engañoso.\n"
                "Debe seguir exactamente uno de estos criterios de viralidad:\n"
                f"{criteria_line}\n"
                f"Evita repetir literalmente este titulo: {clean_avoid or clean_current or 'n/a'}.\n"
                f"Titulo actual: {clean_current or 'n/a'}\n"
                f"Contexto social: {clean_social or 'n/a'}\n"
                f"Contexto transcript: {clean_transcript or 'n/a'}\n"
                f"Etiquetas: {tag_line or 'n/a'}"
            )
            for model_name in TITLE_REWRITE_MODELS:
                try:
                    response = client.models.generate_content(model=model_name, contents=prompt)
                except Exception as model_err:
                    if _is_gemini_model_unavailable_error(model_err): continue
                    raise
                generated = _sanitize_short_title(_extract_generated_text(response))
                if generated and generated.lower() != clean_avoid:
                    return generated
        except Exception: pass

    return _build_fallback_title(clean_current, clean_transcript, safe_tags, clean_avoid)

def _generate_rewritten_title_variants(current_title: str, transcript_excerpt: str, social_excerpt: str, topic_tags: List[str], avoid_titles: Optional[List[str]], target_count: int, api_key: Optional[str]) -> List[str]:
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
            criteria_line = "\n".join(f"- {item}" for item in VIRAL_TITLE_CRITERIA)
            prompt = (
                f"Eres un Periodista y Director de Contenido especializado en videos cortos. Genera exactamente {safe_target} títulos/ganchos distintos para este clip vertical.\n"
                "Hagamos esto paso a paso:\n"
                "1. Lee el título actual y el contexto para entender los puntos clave que atraparán al espectador.\n"
                "2. Asegúrate de que cada título ofrezca una nueva perspectiva respetando el tono del contenido.\n"
                "3. Responde SOLO con un array JSON válido de strings (`[\"titulo 1\", \"titulo 2\", ...]`) y nada más. Ni Markdown, ni texto adicional.\n"
                "Reglas formales: español neutro, 45-95 caracteres por título, palabras simples, sin hashtags, emoji opcional (máx 1), 'Sentence case' (minúsculas apropiadas).\n"
                "Cada título debe seguir un criterio de viralidad distinto de la siguiente lista:\n"
                f"{criteria_line}\n"
                f"Evita repetir literalmente estos títulos: {blocked_line}\n"
                f"Título base: {clean_current or 'n/a'}\n"
                f"Contexto social: {clean_social or 'n/a'}\n"
                f"Contexto transcript: {clean_transcript or 'n/a'}\n"
                f"Etiquetas: {tag_line}"
            )
            for model_name in TITLE_REWRITE_MODELS:
                try:
                    response = client.models.generate_content(model=model_name, contents=prompt)
                except Exception as model_err:
                    if _is_gemini_model_unavailable_error(model_err): continue
                    raise
                raw = _extract_generated_text(response)
                parsed = _parse_title_variants_payload(raw)
                parsed = _dedupe_title_candidates(parsed, blocked=blocked + results)
                if parsed:
                    results.extend(parsed)
                    results = _dedupe_title_candidates(results, blocked=blocked)
                if len(results) >= safe_target: break
        except Exception: pass

    attempts = max(8, safe_target * 4)
    while len(results) < safe_target and attempts > 0:
        candidate = _generate_rewritten_title(clean_current, clean_transcript, clean_social, safe_tags, " | ".join((blocked + results)[-10:]), api_key)
        deduped = _dedupe_title_candidates([candidate], blocked=blocked + results)
        if deduped: results.extend(deduped)
        attempts -= 1

    return _dedupe_title_candidates(results, blocked=blocked)[:safe_target]

def _embed_texts_with_gemini(texts: List[str], api_key: Optional[str]) -> Optional[List[List[float]]]:
    if not api_key or not texts: return None
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        out = []
        for i in range(0, len(texts), 90):
            batch = texts[i:i+90]
            resp = client.models.embed_content(model="text-embedding-004", contents=batch)
            if hasattr(resp, "embeddings"):
                out.extend([e.values for e in resp.embeddings])
        return out if out else None
    except Exception: return None
