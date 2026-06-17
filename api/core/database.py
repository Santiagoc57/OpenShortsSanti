import os
import json
import time
import sqlite3
import shutil
import glob
import uuid
import threading
import asyncio
from typing import Dict, Any, Optional, List, Set
from .config import OUTPUT_DIR, UPLOAD_DIR, JOBS_DB_PATH, MAX_AUTO_RETRIES_DEFAULT, JOB_RETRY_DELAY_SECONDS_DEFAULT

# Global state
jobs: Dict[str, Dict[str, Any]] = {}
job_queue: asyncio.Queue = asyncio.Queue()
_JOBS_DB_LOCK = threading.Lock()

def _jobs_db_connect():
    return sqlite3.connect(JOBS_DB_PATH, timeout=15)

def _init_jobs_store():
    with _JOBS_DB_LOCK:
        conn = _jobs_db_connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS job_state (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                """
            )
            conn.commit()
        finally:
            conn.close()

def _safe_job_snapshot(job: Dict[str, Any]) -> Dict[str, Any]:
    snapshot = dict(job or {})
    logs = snapshot.get("logs")
    if isinstance(logs, list):
        snapshot["logs"] = [str(item) for item in logs][-1600:]
    else:
        snapshot["logs"] = []

    social_posts = snapshot.get("social_posts")
    if isinstance(social_posts, list):
        snapshot["social_posts"] = social_posts[-300:]
    else:
        snapshot["social_posts"] = []

    env_payload = snapshot.get("env")
    if isinstance(env_payload, dict):
        snapshot["env"] = {str(k): str(v) for k, v in env_payload.items()}

    return snapshot

def _persist_job_state(job_id: str):
    job = jobs.get(job_id)
    if not isinstance(job, dict):
        return
    now = time.time()
    job["updated_at"] = now
    snapshot = _safe_job_snapshot(job)
    status = str(snapshot.get("status", "unknown"))
    payload = json.dumps(snapshot, ensure_ascii=False, default=str)

    with _JOBS_DB_LOCK:
        conn = _jobs_db_connect()
        try:
            conn.execute(
                """
                INSERT INTO job_state (job_id, status, payload_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status = excluded.status,
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (job_id, status, payload, now),
            )
            conn.commit()
        finally:
            conn.close()

def _delete_persisted_job(job_id: str):
    with _JOBS_DB_LOCK:
        conn = _jobs_db_connect()
        try:
            conn.execute("DELETE FROM job_state WHERE job_id = ?", (job_id,))
            conn.commit()
        finally:
            conn.close()

def _load_persisted_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _JOBS_DB_LOCK:
        conn = _jobs_db_connect()
        try:
            row = conn.execute(
                "SELECT payload_json FROM job_state WHERE job_id = ? LIMIT 1",
                (job_id,)
            ).fetchone()
        finally:
            conn.close()
    if not row:
        return None
    try:
        payload = json.loads(row[0])
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None

def _load_all_persisted_jobs(limit: int = 400) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    safe_limit = max(10, min(4000, int(limit)))
    with _JOBS_DB_LOCK:
        conn = _jobs_db_connect()
        try:
            rows = conn.execute(
                "SELECT job_id, payload_json FROM job_state ORDER BY updated_at DESC LIMIT ?",
                (safe_limit,)
            ).fetchall()
        finally:
            conn.close()
    for row in rows:
        try:
            job_id = str(row[0])
            payload = json.loads(row[1])
            if isinstance(payload, dict):
                out[job_id] = payload
        except Exception:
            continue
    return out

def _metadata_path_for_job(job_id: str) -> Optional[str]:
    output_dir = os.path.join(OUTPUT_DIR, job_id)
    if not os.path.isdir(output_dir):
        return None
    candidates = sorted(glob.glob(os.path.join(output_dir, "*_metadata.json")))
    return candidates[0] if candidates else None

def _is_uuid_like_job_id(value: str) -> bool:
    candidate = str(value or "").strip()
    if not candidate:
        return False
    try:
        uuid.UUID(candidate)
        return True
    except Exception:
        return False

def _discover_job_ids_on_disk(limit: int = 400) -> List[str]:
    if not os.path.isdir(OUTPUT_DIR):
        return []
    entries = []
    for name in os.listdir(OUTPUT_DIR):
        if not _is_uuid_like_job_id(name):
            continue
        path = os.path.join(OUTPUT_DIR, name)
        if not os.path.isdir(path):
            continue
        try:
            updated_at = os.path.getmtime(path)
        except OSError:
            updated_at = 0
        entries.append((updated_at, name))
    entries.sort(reverse=True)
    return [name for _, name in entries[: max(1, int(limit or 400))]]

def _purge_job_artifacts(job_id: str) -> bool:
    safe_job_id = os.path.basename(str(job_id or "").strip())
    if not safe_job_id or safe_job_id != str(job_id or "").strip():
        return False
    output_dir = os.path.abspath(os.path.join(OUTPUT_DIR, safe_job_id))
    uploads_prefix = os.path.abspath(UPLOAD_DIR)
    output_prefix = os.path.abspath(OUTPUT_DIR)
    if not output_dir.startswith(output_prefix + os.sep):
        return False

    removed = False
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
        removed = True

    for path in glob.glob(os.path.join(uploads_prefix, f"{safe_job_id}_*")):
        try:
            if os.path.isfile(path):
                os.remove(path)
                removed = True
        except OSError:
            continue
    return removed

def _ensure_job_context(job_id: str) -> Optional[Dict[str, Any]]:
    if job_id in jobs:
        return jobs[job_id]
    persisted = _load_persisted_job(job_id)
    if persisted:
        jobs[job_id] = persisted
        return persisted
    return None
