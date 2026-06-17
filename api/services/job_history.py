import glob
import json
import os
from typing import Any, Dict, List, Optional

from ..core.config import OUTPUT_DIR
from ..core.database import _discover_job_ids_on_disk, _load_all_persisted_jobs
from s3_uploader import list_remote_job_ids


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return fallback


def _metadata_path_for_output_dir(output_dir: str) -> Optional[str]:
    candidates = sorted(glob.glob(os.path.join(output_dir, "*_metadata.json")))
    return candidates[0] if candidates else None


def _load_metadata(output_dir: str) -> Dict[str, Any]:
    metadata_path = _metadata_path_for_output_dir(output_dir)
    if not metadata_path:
        return {}
    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _existing_video_url(output_dir: str, job_id: str, filename: str) -> str:
    safe_name = os.path.basename(str(filename or ""))
    if not safe_name:
        return ""
    path = os.path.join(output_dir, safe_name)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return f"/videos/{job_id}/{safe_name}"
    return ""


def _clip_filename_candidates(base_name: str, index: int, clip: Dict[str, Any]) -> List[str]:
    candidates = []
    for key in ("video_filename", "filename", "source_video_filename", "preview_video_filename"):
        value = os.path.basename(str(clip.get(key) or ""))
        if value:
            candidates.append(value)
    for key in ("video_url", "url", "preview_video_url"):
        value = str(clip.get(key) or "")
        if value:
            candidates.append(os.path.basename(value.split("?", 1)[0].split("#", 1)[0]))
    if base_name:
        candidates.append(f"{base_name}_clip_{index + 1}.mp4")
        candidates.append(f"{base_name}_clip_{index + 1}_uncut.mp4")

    out = []
    seen = set()
    for item in candidates:
        safe = os.path.basename(str(item or ""))
        if safe and safe not in seen:
            seen.add(safe)
            out.append(safe)
    return out


def build_job_result(output_dir: str, job_id: str) -> Optional[Dict[str, Any]]:
    if not os.path.isdir(output_dir):
        return None

    metadata = _load_metadata(output_dir)
    clips = metadata.get("shorts")
    if not isinstance(clips, list):
        clips = []

    metadata_path = _metadata_path_for_output_dir(output_dir)
    base_name = ""
    if metadata_path:
        base_name = os.path.basename(metadata_path).replace("_metadata.json", "")

    ready_clips: List[Dict[str, Any]] = []
    for index, raw_clip in enumerate(clips):
        if not isinstance(raw_clip, dict):
            continue
        clip = dict(raw_clip)
        filename = ""
        video_url = ""
        for candidate in _clip_filename_candidates(base_name, index, clip):
            maybe_url = _existing_video_url(output_dir, job_id, candidate)
            if maybe_url:
                filename = candidate
                video_url = maybe_url
                break

        if video_url:
            clip["video_url"] = video_url
            clip["url"] = video_url
            clip["video_filename"] = filename
        elif clip.get("video_url") or clip.get("url"):
            clip["video_url"] = clip.get("video_url") or clip.get("url")
            clip["url"] = clip["video_url"]
        else:
            clip["status"] = clip.get("status") or "draft"

        clip["clip_index"] = index
        clip["index"] = index
        if "duration" not in clip:
            clip["duration"] = max(0.0, _safe_float(clip.get("end")) - _safe_float(clip.get("start")))
        ready_clips.append(clip)

    latest_trailer_url = str(metadata.get("latest_trailer_url") or "")
    if latest_trailer_url and not any(item.get("is_trailer") for item in ready_clips):
        trailer_name = os.path.basename(latest_trailer_url)
        trailer_url = _existing_video_url(output_dir, job_id, trailer_name) or latest_trailer_url
        ready_clips.insert(0, {
            "clip_index": 0,
            "index": 0,
            "is_trailer": True,
            "title": "Super trailer",
            "video_title_for_youtube_short": "Super trailer",
            "video_url": trailer_url,
            "url": trailer_url,
            "video_filename": trailer_name,
            "start": 0,
            "end": _safe_float(metadata.get("latest_trailer_duration"), 0),
        })

    return {
        "clips": ready_clips,
        "cost_analysis": metadata.get("cost_analysis") or {},
        "transcript": metadata.get("transcript") or {},
        "trailer_fragments": metadata.get("trailer_fragments") or [],
        "metadata": {
            "path": metadata_path,
            "base_name": base_name,
        },
    }


def get_job_summary(job_id: str, persisted_job: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    output_dir = os.path.join(OUTPUT_DIR, job_id)
    if not os.path.isdir(output_dir) and not persisted_job:
        return None

    result = build_job_result(output_dir, job_id)
    clips = result.get("clips", []) if result else []
    metadata = _load_metadata(output_dir)
    stat_target = output_dir if os.path.isdir(output_dir) else None
    updated_at = os.path.getmtime(stat_target) if stat_target else (persisted_job or {}).get("updated_at")

    first_clip = clips[0] if clips else {}
    status = (persisted_job or {}).get("status") or ("completed" if clips else "unknown")
    ui_status = "complete" if status == "completed" else ("error" if status == "failed" else "processing")
    title = (
        (persisted_job or {}).get("title")
        or first_clip.get("video_title_for_youtube_short")
        or first_clip.get("title")
        or metadata.get("title")
        or job_id
    )

    return {
        "job_id": job_id,
        "title": title,
        "status": ui_status,
        "backend_status": status,
        "created_at": (persisted_job or {}).get("created_at") or updated_at,
        "updated_at": updated_at,
        "expires_at": None,
        "clip_count": len(clips),
        "clip_count_actual": len(clips),
        "clip_count_target": (persisted_job or {}).get("clip_count_target"),
        "ratio": (persisted_job or {}).get("aspect_ratio") or first_clip.get("aspect_ratio") or "9:16",
        "source_kind": (persisted_job or {}).get("source_kind") or ("youtube" if (persisted_job or {}).get("source_url") else "file"),
        "source_label": (persisted_job or {}).get("source_label") or ("YouTube" if (persisted_job or {}).get("source_url") else "Archivo local"),
        "video_type": (persisted_job or {}).get("video_type") or "Topic-clips",
        "preview_video_url": first_clip.get("video_url") or first_clip.get("url") or "",
        "thumbnail_url": first_clip.get("thumbnail_url") or first_clip.get("preview_image_url") or "",
        "clips": clips[:3],
    }


def list_project_history(limit: int = 48) -> List[Dict[str, Any]]:
    persisted = _load_all_persisted_jobs(limit=max(400, limit))
    disk_ids = _discover_job_ids_on_disk(limit=max(400, limit))
    remote_ids = list_remote_job_ids(limit=max(400, limit))
    ids = list(dict.fromkeys([*persisted.keys(), *disk_ids, *remote_ids]))
    local_ids = set(persisted.keys()) | set(disk_ids)
    projects = []
    for job_id in ids:
        summary = get_job_summary(job_id, persisted.get(job_id))
        if summary:
            summary["storage"] = "local"
            projects.append(summary)
        elif job_id in remote_ids and job_id not in local_ids:
            projects.append({
                "job_id": job_id,
                "title": job_id,
                "status": "archived",
                "backend_status": "archived",
                "created_at": None,
                "updated_at": None,
                "expires_at": None,
                "clip_count": 0,
                "clip_count_actual": 0,
                "clip_count_target": None,
                "ratio": "9:16",
                "source_kind": "remote",
                "source_label": "S3/MinIO",
                "video_type": "Archivado",
                "preview_video_url": "",
                "thumbnail_url": "",
                "clips": [],
                "storage": "s3",
            })
    projects.sort(key=lambda item: item.get("updated_at") or item.get("created_at") or 0, reverse=True)
    return projects[: max(1, int(limit or 48))]
