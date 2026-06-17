import os
import sys
import uuid
import shutil
import subprocess
import threading
import time
import glob
import json
import asyncio
from typing import Dict, Any, Tuple, Optional, List

from .config import (
    MAX_CONCURRENT_JOBS, MAX_AUTO_RETRIES_DEFAULT, 
    JOB_RETRY_DELAY_SECONDS_DEFAULT
)
from .database import (
    jobs, job_queue, _persist_job_state, _ensure_job_context
)
from ..services.job_history import build_job_result
from ..services.job_manifest import write_job_manifest
from s3_uploader import upload_job_artifacts
from ..utils.media import _probe_media_duration_seconds, _safe_input_filename
from ..utils.text import _normalize_space, _safe_float

running_processes: Dict[str, subprocess.Popen] = {}


def enqueue_output(out, job_id):
    persist_tick = 0
    try:
        for line in iter(out.readline, b''):
            decoded_line = line.decode('utf-8').strip()
            if decoded_line:
                if job_id in jobs:
                    jobs[job_id]['logs'].append(decoded_line)
                    persist_tick += 1
                    if persist_tick % 12 == 0:
                        _persist_job_state(job_id)
    except Exception: pass
    finally: out.close()

async def _queue_job_retry(job_id: str, reason: str, trigger: str = "auto", delay_seconds: Optional[int] = None):
    job = _ensure_job_context(job_id)
    if not job: return False

    retry_count = int(job.get('auto_retry_count') or 0)
    max_retries = int(job.get('max_auto_retries', MAX_AUTO_RETRIES_DEFAULT) or 0)
    if retry_count >= max_retries:
        job['status'] = 'error'
        job['last_error'] = reason
        job['logs'].append(f"Retries exhausted ({retry_count}/{max_retries}): {reason}")
        write_job_manifest(job_id, job, "failed", error=reason)
        _persist_job_state(job_id)
        return False

    delay = delay_seconds if delay_seconds is not None else job.get('retry_delay_seconds', JOB_RETRY_DELAY_SECONDS_DEFAULT)
    delay = max(0, int(delay or 0))
    job['auto_retry_count'] = retry_count + 1
    job['last_error'] = reason
    job['status'] = 'queued'
    job['logs'].append(f"Retry scheduled ({job['auto_retry_count']}/{max_retries}): {reason}")
    _persist_job_state(job_id)
    if delay:
        await asyncio.sleep(delay)
    await job_queue.put(job_id)
    return True

async def run_job(job_id: str):
    job_data = jobs.get(job_id)
    if not job_data: return
    
    job_data['status'] = 'processing'
    job_data['logs'].append(f"Job started at {time.ctime()}")
    write_job_manifest(job_id, job_data, "started")
    _persist_job_state(job_id)
    
    try:
        process_env = dict(os.environ)
        process_env.update(job_data.get('env') or {})
        process_env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        process_env.setdefault("OMP_NUM_THREADS", "1")
        process_env.setdefault("MKL_NUM_THREADS", "1")
        process_env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
        process_env.setdefault("OPENBLAS_NUM_THREADS", "1")
        process = subprocess.Popen(
            job_data['cmd'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=process_env,
            cwd=os.getcwd()
        )
        running_processes[job_id] = process
        t_log = threading.Thread(target=enqueue_output, args=(process.stdout, job_id))
        t_log.daemon = True
        t_log.start()
        
        while process.poll() is None:
            await asyncio.sleep(2)
            # (Logic for partial metadata updates here)
            
        if job_data.get("status") == "paused":
            job_data['logs'].append("Process paused/cancelled by user.")
            write_job_manifest(job_id, job_data, "paused", returncode=process.returncode)
        elif process.returncode == 0:
            job_data['status'] = 'completed'
            job_data['logs'].append("Process finished successfully.")
            result = build_job_result(job_data.get("output_dir", ""), job_id)
            if result:
                job_data["result"] = result
            write_job_manifest(job_id, job_data, "completed", returncode=process.returncode)
            try:
                upload_job_artifacts(job_data.get("output_dir", ""), job_id)
                job_data['logs'].append("Artifacts persisted to S3/MinIO when configured.")
            except Exception as upload_error:
                job_data['logs'].append(f"Artifact upload warning: {upload_error}")
        else:
            job_data['status'] = 'error'
            write_job_manifest(job_id, job_data, "failed", returncode=process.returncode)
            await _queue_job_retry(job_id, f"Process failed with code {process.returncode}")
            
    except Exception as e:
        job_data['status'] = 'error'
        write_job_manifest(job_id, job_data, "error", error=str(e))
        await _queue_job_retry(job_id, str(e))
    finally:
        running_processes.pop(job_id, None)
        _persist_job_state(job_id)

async def job_worker():
    while True:
        job_id = await job_queue.get()
        try:
            await run_job(job_id)
        finally:
            job_queue.task_done()
