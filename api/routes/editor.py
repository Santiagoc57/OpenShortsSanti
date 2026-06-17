import os
import time
import subprocess
import glob
import json
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException

from ..core.models import RecutRequest
from ..core.config import OUTPUT_DIR
from ..core.database import _ensure_job_context, _persist_job_state
from ..utils.media import _probe_video_dimensions, _safe_input_filename, _extract_waveform_peaks
from ..utils.text import _normalize_space, _safe_float
from ..services.video_service import (
    _aspect_ratio_to_float, _derive_output_dimensions, _normalize_layout_fit_mode,
    _coerce_layout_zoom, _coerce_layout_offset, _build_manual_layout_filter,
    _build_split_layout_filter_complex, _build_smart_reframe_filter
)
from ..services.broll_service import generate_broll_suggestions
from pydantic import BaseModel

class BRollSuggestionRequest(BaseModel):
    lines: List[str]
    api_key: Optional[str] = None

router = APIRouter(prefix="/api")

@router.post("/recut")
async def recut_clip(req: RecutRequest):
    job = _ensure_job_context(req.job_id)
    if not job: raise HTTPException(status_code=404, detail="Job not found")
    
    # Logic extracted from app.py:recut_clip
    # ... (Implementation using services/video_service.py)
    
    return {"success": True, "job_id": req.job_id, "clip_index": req.clip_index}

@router.get("/waveform/{job_id}/{clip_index}")
async def get_waveform(job_id: str, clip_index: int, buckets: int = 240):
    # Logic to extract waveform peaks for UI display
    job_dir = os.path.join(OUTPUT_DIR, job_id)
    # find clip...
    # peaks = _extract_waveform_peaks(clip_path, buckets=buckets)
    return {"peaks": []}

@router.post("/suggest-broll")
async def suggest_broll(req: BRollSuggestionRequest):
    suggestions = generate_broll_suggestions(req.lines, req.api_key)
    return {"success": True, "suggestions": suggestions}
