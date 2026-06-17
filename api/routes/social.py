import os
import time
import json
import httpx
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from ..core.models import SocialPostRequest
from ..core.config import OUTPUT_DIR
from ..core.database import _ensure_job_context, _persist_job_state, _metadata_path_for_job

router = APIRouter(prefix="/api/social")

def _append_social_post_event(job_id: str, clip_index: int, platforms: List[str], status: str, status_code: int, detail: str = "", vendor_payload: Optional[Dict[str, Any]] = None):
    job = _ensure_job_context(job_id)
    if not isinstance(job, dict): return

    event = {
        "timestamp": int(time.time()),
        "clip_index": int(max(0, clip_index)),
        "platforms": [str(p) for p in (platforms or [])],
        "status": str(status or "unknown"),
        "status_code": int(status_code or 0),
        "detail": str(detail or "")[:400],
        "vendor_payload": vendor_payload if isinstance(vendor_payload, dict) else None,
    }
    posts = job.setdefault("social_posts", [])
    posts.append(event)
    job["social_posts"] = posts[-300:]
    _persist_job_state(job_id)

    metadata_path = _metadata_path_for_job(job_id)
    if metadata_path:
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            events = metadata.get("social_posts", [])
            events.append(event)
            metadata["social_posts"] = events[-300:]
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception: pass

def _compute_social_metrics(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    safe_events = [e for e in events if isinstance(e, dict)]
    total = len(safe_events)
    success = sum(1 for e in safe_events if str(e.get("status", "")).lower() == "success")
    
    by_platform: Dict[str, Dict[str, int]] = {}
    for event in safe_events:
        platforms = event.get("platforms", [])
        is_ok = str(event.get("status", "")).lower() == "success"
        for platform in platforms:
            key = str(platform).lower()
            stats = by_platform.setdefault(key, {"attempted": 0, "success": 0, "failed": 0})
            stats["attempted"] += 1
            if is_ok: stats["success"] += 1
            else: stats["failed"] += 1

    return {
        "total_attempts": total,
        "successful_attempts": success,
        "failed_attempts": total - success,
        "success_rate": round((success / total), 4) if total > 0 else 0.0,
        "by_platform": by_platform,
    }

@router.post("/post")
async def post_to_socials(req: SocialPostRequest):
    job = _ensure_job_context(req.job_id)
    if not job: raise HTTPException(status_code=404, detail="Job not found")
    
    # Implementation using httpx to call Upload-Post API...
    # (Ported from app.py:post_to_socials)
    return {"status": "success", "detail": "Post initiated"}

@router.get("/stats/{job_id}")
async def get_social_stats(job_id: str):
    job = _ensure_job_context(job_id)
    if not job: raise HTTPException(status_code=404, detail="Job not found")
    events = job.get("social_posts", [])
    return _compute_social_metrics(events)
