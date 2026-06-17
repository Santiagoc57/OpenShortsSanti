import json
import os
import time
from typing import Any, Dict, List, Optional


SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
ENV_ALLOW_PREFIXES = ("OPENSHORTS_", "WHISPER_", "FFMPEG_", "PYTHON")
ENV_ALLOW_KEYS = {
    "CUDA_VISIBLE_DEVICES",
    "DISABLE_YOUTUBE_URL",
    "HF_HOME",
    "MAX_CONCURRENT_JOBS",
    "RENDER_SERVICE_URL",
    "TRANSFORMERS_CACHE",
}


def _manifest_path(output_dir: str) -> str:
    return os.path.join(output_dir, "job_manifest.json")


def _safe_json_load(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _is_secret_key(key: str) -> bool:
    upper = str(key or "").upper()
    return any(marker in upper for marker in SECRET_MARKERS)


def _sanitize_env(env: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in sorted((env or {}).items()):
        name = str(key)
        if _is_secret_key(name):
            if value:
                out[name] = "***"
            continue
        if name in ENV_ALLOW_KEYS or any(name.startswith(prefix) for prefix in ENV_ALLOW_PREFIXES):
            out[name] = str(value)
    return out


def _list_artifacts(output_dir: str) -> List[Dict[str, Any]]:
    artifacts: List[Dict[str, Any]] = []
    if not os.path.isdir(output_dir):
        return artifacts

    for root, _dirs, files in os.walk(output_dir):
        for filename in files:
            if filename == "job_manifest.json":
                continue
            path = os.path.join(root, filename)
            try:
                stat = os.stat(path)
            except OSError:
                continue
            rel_path = os.path.relpath(path, output_dir).replace(os.sep, "/")
            artifacts.append({
                "path": rel_path,
                "size_bytes": stat.st_size,
                "updated_at": stat.st_mtime,
            })

    artifacts.sort(key=lambda item: item.get("path") or "")
    return artifacts


def write_job_manifest(
    job_id: str,
    job: Dict[str, Any],
    event: str,
    returncode: Optional[int] = None,
    error: Optional[str] = None,
) -> Optional[str]:
    output_dir = str((job or {}).get("output_dir") or "").strip()
    if not output_dir:
        return None

    os.makedirs(output_dir, exist_ok=True)
    path = _manifest_path(output_dir)
    now = time.time()
    existing = _safe_json_load(path)

    events = existing.get("events")
    if not isinstance(events, list):
        events = []
    events.append({
        "event": str(event or "update"),
        "timestamp": now,
        "status": (job or {}).get("status"),
        "returncode": returncode,
        "error": str(error) if error else None,
    })
    events = events[-200:]

    manifest = {
        "schema_version": 1,
        "job_id": job_id,
        "status": (job or {}).get("status"),
        "created_at": existing.get("created_at") or (job or {}).get("created_at") or now,
        "updated_at": now,
        "source": {
            "kind": (job or {}).get("source_kind"),
            "label": (job or {}).get("source_label"),
            "url": (job or {}).get("source_url") or "",
        },
        "generation": {
            "aspect_ratio": (job or {}).get("aspect_ratio"),
            "clip_count_target": (job or {}).get("clip_count_target"),
            "video_type": (job or {}).get("video_type"),
            "ownership_attested": bool((job or {}).get("ownership_attested")),
        },
        "command": [str(item) for item in ((job or {}).get("cmd") or [])],
        "runtime_env": _sanitize_env((job or {}).get("env") or {}),
        "returncode": returncode if returncode is not None else existing.get("returncode"),
        "last_error": str(error) if error else (job or {}).get("last_error") or existing.get("last_error"),
        "events": events,
        "artifacts": _list_artifacts(output_dir),
    }

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    return path


def read_job_manifest(output_dir: str) -> Optional[Dict[str, Any]]:
    path = _manifest_path(output_dir)
    if not os.path.exists(path):
        return None
    data = _safe_json_load(path)
    return data if data else None
