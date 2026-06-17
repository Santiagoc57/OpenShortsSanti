import os
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _detect_cuda_available() -> bool:
    forced = str(os.environ.get("OPENSHORTS_FORCE_CUDA", "")).strip().lower()
    if forced in {"1", "true", "yes", "y"}:
        return True
    if forced in {"0", "false", "no", "n"}:
        return False
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False

CUDA_AVAILABLE = _detect_cuda_available()

def _parse_int_or_default(value: Optional[str], default: int) -> int:
    try:
        return int(str(value).strip())
    except Exception:
        return int(default)

def _resolve_default_max_concurrent_jobs() -> int:
    override = os.environ.get("MAX_CONCURRENT_JOBS")
    if override is not None and str(override).strip() != "":
        return max(1, min(16, _parse_int_or_default(override, 1)))

    cpu_count = max(1, int(os.cpu_count() or 1))
    if CUDA_AVAILABLE:
        return max(2, min(4, cpu_count // 4 or 2))
    return 2 if cpu_count >= 12 else 1

# Configuration Constants
MAX_CONCURRENT_JOBS = _resolve_default_max_concurrent_jobs()
MAX_FILE_SIZE_MB = 500
JOB_RETENTION_SECONDS = 3600
MAX_AUTO_RETRIES_DEFAULT = int(os.environ.get("MAX_AUTO_RETRIES", "1"))
JOB_RETRY_DELAY_SECONDS_DEFAULT = int(os.environ.get("JOB_RETRY_DELAY_SECONDS", "10"))
JOBS_DB_PATH = os.environ.get("JOBS_DB_PATH", os.path.join(OUTPUT_DIR, "jobs_state.sqlite3"))

# AI Models
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
SEMANTIC_EMBED_MODEL = os.environ.get("SEMANTIC_EMBED_MODEL", "text-embedding-004")
DEFAULT_TITLE_REWRITE_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
]
VIRAL_TITLE_CRITERIA = [
    "Nunca vas a creer esto...",
    "No pierdas tu tiempo...",
    "Esto fue un verdadero shock...",
    "Cómo lograr [resultado específico] en [#] tiempo",
    "Evita [cosa] a toda costa",
    "Por esto eres tan malo en...",
    "Opinión impopular sobre X",
    "Esto me volvió loco",
    "La estrategia secreta para...",
    "Si una persona más me dice...",
    "Esto no tiene ningún sentido",
    "¿Por qué esto es tan difícil?",
    "Simplemente no puedo hacerlo",
    "Mi mayor arrepentimiento es...",
    "Esto me tuvo estancado",
    "¿Cómo es que esto no funciona?",
    "Se me cayó la mandíbula cuando...",
    "La increíble razón por la que...",
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
LOCAL_EMBED_DIM = 256
DISABLE_YOUTUBE_URL = os.environ.get("DISABLE_YOUTUBE_URL", "false").strip().lower() in {"1", "true", "yes", "on"}
RENDER_SERVICE_URL = os.environ.get("RENDER_SERVICE_URL", "http://localhost:3100").rstrip("/")
