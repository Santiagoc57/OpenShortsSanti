import os
import sys
import uuid
import shutil
import subprocess
import time
import re
import json
from typing import Optional, Dict, Any, List, Set
from fastapi import APIRouter, Request, UploadFile, File, Form, HTTPException, Header

from ..core.config import (
    UPLOAD_DIR, OUTPUT_DIR, MAX_FILE_SIZE_MB, 
    MAX_AUTO_RETRIES_DEFAULT, JOB_RETRY_DELAY_SECONDS_DEFAULT,
    ALLOWED_CLIP_LENGTH_TARGETS, CUDA_AVAILABLE, MAX_CONCURRENT_JOBS,
    DISABLE_YOUTUBE_URL
)
from ..core.database import (
    jobs, job_queue, _persist_job_state, _ensure_job_context,
    _load_all_persisted_jobs, _delete_persisted_job, _discover_job_ids_on_disk,
    _purge_job_artifacts
)
from ..core.worker import running_processes
from ..utils.text import _normalize_space, _safe_float
from ..utils.media import _safe_input_filename
from ..services.job_history import build_job_result, get_job_summary, list_project_history
from ..services.job_manifest import read_job_manifest, write_job_manifest
from s3_uploader import delete_job_artifacts, download_job_artifacts

router = APIRouter(prefix="/api")

def _parse_form_bool(value: Any, default: bool = False) -> bool:
    if value is None: return default
    if isinstance(value, bool): return value
    raw = str(value).strip().lower()
    return raw in {"1", "true", "t", "yes", "y", "on"}

def normalize_aspect_ratio(val: Optional[str], default: str = "9:16") -> str:
    raw = str(val or default).strip().replace(" ", "")
    return raw if raw in {"9:16", "16:9"} else default

@router.post("/process")
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
    tight_edit_preset: Optional[str] = Form(None),
    ownership_attested: Optional[str] = Form(None),
    enable_diarization: Optional[str] = Form(None),
    groq_api_key: Optional[str] = Form(None),
    huggingface_token: Optional[str] = Form(None),
    max_auto_retries: Optional[int] = Form(None),
    retry_delay_seconds: Optional[int] = Form(None)
):
    x_gemini_key = request.headers.get("X-Gemini-Key")
    x_groq_key = request.headers.get("X-Groq-Key")
    
    # JSON body handling for dynamic payloads
    if "application/json" in request.headers.get("content-type", ""):
        body = await request.json()
        url = body.get("url") or url
        language = body.get("language", language)
        max_clips = body.get("max_clips", body.get("clipCount", max_clips))
        whisper_backend = body.get("whisper_backend", body.get("whisperBackend", whisper_backend))
        whisper_model = body.get("whisper_model", body.get("whisperModel", whisper_model))
        word_timestamps = body.get("word_timestamps", body.get("wordTimestamps", word_timestamps))
        ffmpeg_preset = body.get("ffmpeg_preset", body.get("ffmpegPreset", ffmpeg_preset))
        ffmpeg_crf = body.get("ffmpeg_crf", body.get("ffmpegCrf", ffmpeg_crf))
        aspect_ratio = body.get("aspect_ratio", body.get("aspectRatio", aspect_ratio))
        clip_length_target = body.get("clip_length_target", body.get("clipLengthTarget", clip_length_target))
        style_template = body.get("style_template", body.get("styleTemplate", style_template))
        content_profile = body.get("content_profile", body.get("contentPreset", content_profile))
        llm_model = body.get("llm_model", body.get("llmModel", llm_model))
        llm_provider = body.get("llm_provider", body.get("llmProvider", llm_provider))
        generation_mode = body.get("generation_mode", body.get("generationMode", generation_mode))
        build_trailer = body.get("build_trailer", body.get("buildTrailer", build_trailer))
        trailer_fragments_target = body.get("trailer_fragments_target", body.get("trailerFragmentsTarget", trailer_fragments_target))
        tight_edit_preset = body.get("tight_edit_preset", body.get("tightEditPreset", tight_edit_preset))
        ownership_attested = body.get("ownership_attested", body.get("ownershipAttested", ownership_attested))
        enable_diarization = body.get("enable_diarization", body.get("enableDiarization", enable_diarization))
    
    if not url and not file:
        raise HTTPException(status_code=400, detail="Must provide URL or File")
    if url and DISABLE_YOUTUBE_URL:
        raise HTTPException(status_code=403, detail="YouTube URL ingest is disabled on this deployment. Upload a file you own instead.")
    
    job_id = str(uuid.uuid4())
    job_output_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_output_dir, exist_ok=True)
    
    # Command Construction
    python_bin = sys.executable or "python3"
    cmd = [python_bin, "-u", "main.py"]
    env = os.environ.copy()
    if x_gemini_key:
        env["GEMINI_API_KEY"] = x_gemini_key
    if x_groq_key:
        env["GROQ_API_KEY"] = x_groq_key
    if groq_api_key:
        env["GROQ_API_KEY"] = groq_api_key
    if huggingface_token:
        env["HF_TOKEN"] = huggingface_token
    
    if url:
        cmd.extend(["-u", url, "--keep-original"])
        input_path = None
    else:
        input_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        cmd.extend(["-i", input_path])

    # Add all args (aspect ratio, language, etc.)
    cmd.extend(["-o", job_output_dir])
    normalized_aspect = normalize_aspect_ratio(aspect_ratio)
    cmd.extend(["--aspect-ratio", normalized_aspect])

    if language and str(language).strip() != "auto":
        cmd.extend(["--language", str(language).strip()])
    if max_clips:
        cmd.extend(["--max-clips", str(max(1, min(15, int(max_clips))))])
    if whisper_backend:
        cmd.extend(["--whisper-backend", str(whisper_backend)])
    if whisper_model:
        cmd.extend(["--whisper-model", str(whisper_model)])
    if word_timestamps is not None:
        cmd.extend(["--word-timestamps", "true" if _parse_form_bool(word_timestamps, True) else "false"])
    if ffmpeg_preset:
        cmd.extend(["--ffmpeg-preset", str(ffmpeg_preset)])
    if ffmpeg_crf:
        cmd.extend(["--ffmpeg-crf", str(ffmpeg_crf)])
    if clip_length_target in ALLOWED_CLIP_LENGTH_TARGETS:
        cmd.extend(["--clip-length-target", str(clip_length_target)])
    if style_template:
        cmd.extend(["--style-template", str(style_template)])
    if content_profile:
        cmd.extend(["--content-profile", str(content_profile)])
    if llm_model:
        cmd.extend(["--llm-model", str(llm_model)])
    if llm_provider:
        cmd.extend(["--llm-provider", str(llm_provider)])
    if _parse_form_bool(build_trailer, False) or generation_mode == "trailer":
        cmd.append("--build-trailer")
    if generation_mode == "trailer":
        cmd.append("--trailer-only")
    if trailer_fragments_target:
        cmd.extend(["--trailer-fragments-target", str(max(2, min(12, int(trailer_fragments_target))))])
    if tight_edit_preset:
        cmd.extend(["--tight-edit-preset", str(tight_edit_preset)])
    if _parse_form_bool(enable_diarization, False):
        cmd.append("--enable-diarization")

    jobs[job_id] = {
        'status': 'queued',
        'logs': [f"Job {job_id} queued."],
        'cmd': cmd,
        'env': env,
        'output_dir': job_output_dir,
        'input_path': input_path,
        'created_at': time.time(),
        'aspect_ratio': normalized_aspect,
        'clip_count_target': max_clips,
        'source_kind': 'youtube' if url else 'file',
        'source_url': url or '',
        'source_label': 'YouTube' if url else (file.filename if file else 'Archivo local'),
        'ownership_attested': True,
        'attestation': {
            'accepted': True,
            'source': 'url' if url else 'file',
            'user_agent': request.headers.get('user-agent', ''),
            'ip': request.headers.get('x-forwarded-for', request.client.host if request.client else ''),
            'timestamp': time.time()
        },
        'video_type': 'Super trailer' if generation_mode == 'trailer' else 'Topic-clips',
        'updated_at': time.time()
    }
    _persist_job_state(job_id)
    write_job_manifest(job_id, jobs[job_id], "queued")
    await job_queue.put(job_id)
    
    return {"job_id": job_id, "status": "queued"}

@router.get("/status/{job_id}")
async def get_status(job_id: str):
    job = _ensure_job_context(job_id)
    if not job: raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") == "completed" and not job.get("result"):
        result = build_job_result(job.get("output_dir") or os.path.join(OUTPUT_DIR, job_id), job_id)
        if result:
            job["result"] = result
            _persist_job_state(job_id)
    return job

@router.get("/jobs/recent")
async def list_jobs(limit: int = 20):
    all_jobs = _load_all_persisted_jobs(limit=limit)
    return {"items": list(all_jobs.values())}

@router.get("/projects/history")
async def projects_history(limit: int = 48, refresh: bool = True):
    projects = list_project_history(limit=limit)
    seen = {project.get("job_id") for project in projects}
    for active_job_id, active_job in list(jobs.items()):
        if active_job_id in seen:
            continue
        summary = get_job_summary(active_job_id, active_job)
        if summary:
            summary["storage"] = "memory"
            projects.insert(0, summary)
            seen.add(active_job_id)
    projects.sort(key=lambda item: item.get("updated_at") or item.get("created_at") or 0, reverse=True)
    projects = projects[: max(1, int(limit or 48))]
    return {"projects": projects, "total": len(projects)}

@router.get("/projects/clips/{job_id}")
async def project_clips(job_id: str, refresh: bool = True):
    job = _ensure_job_context(job_id)
    output_dir = (job or {}).get("output_dir") or os.path.join(OUTPUT_DIR, job_id)
    result = build_job_result(output_dir, job_id)
    if not result:
        restored_count = download_job_artifacts(job_id, output_dir)
        if restored_count:
            result = build_job_result(output_dir, job_id)
    if not result:
        raise HTTPException(status_code=404, detail="Project clips not found")
    if job is not None:
        job["result"] = result
        _persist_job_state(job_id)
    return {"job_id": job_id, "clips": result.get("clips", []), "result": result}

@router.get("/projects/manifest/{job_id}")
async def project_manifest(job_id: str):
    job = _ensure_job_context(job_id)
    output_dir = (job or {}).get("output_dir") or os.path.join(OUTPUT_DIR, job_id)
    manifest = read_job_manifest(output_dir)
    if not manifest:
        restored_count = download_job_artifacts(job_id, output_dir)
        if restored_count:
            manifest = read_job_manifest(output_dir)
    if not manifest:
        raise HTTPException(status_code=404, detail="Project manifest not found")
    return {"job_id": job_id, "manifest": manifest}

@router.post("/projects/{job_id}/cancel")
async def cancel_project(job_id: str):
    job = _ensure_job_context(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    process = running_processes.get(job_id)
    if process and process.poll() is None:
        try:
            process.terminate()
            try:
                process.wait(timeout=5)
            except Exception:
                process.kill()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not cancel job: {exc}") from exc

    job["status"] = "paused"
    job.setdefault("logs", []).append("Job paused/cancelled by user.")
    _persist_job_state(job_id)
    write_job_manifest(job_id, job, "paused")
    return {"success": True, "job_id": job_id, "status": "paused"}

@router.delete("/projects/{job_id}")
async def delete_project(job_id: str):
    process = running_processes.get(job_id)
    if process and process.poll() is None:
        try:
            process.terminate()
            try:
                process.wait(timeout=5)
            except Exception:
                process.kill()
        except Exception:
            pass
    _delete_persisted_job(job_id)
    jobs.pop(job_id, None)
    local_deleted = _purge_job_artifacts(job_id)
    try:
        s3_deleted = delete_job_artifacts(job_id)
    except Exception:
        s3_deleted = 0
    return {"success": True, "job_id": job_id, "local_deleted": local_deleted, "s3_deleted_count": s3_deleted}
