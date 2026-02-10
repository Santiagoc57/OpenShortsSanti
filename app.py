import os
import uuid
import subprocess
import threading
import json
import shutil
import glob
import time
import asyncio
import re
import csv
import io
import zipfile
from dotenv import load_dotenv
from typing import Dict, Optional, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from s3_uploader import upload_job_artifacts

load_dotenv()

# Constants
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration
# Default to 1 if not set, but user can set higher for powerful servers
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "5"))
MAX_FILE_SIZE_MB = 500  # 500 MB limit
JOB_RETENTION_SECONDS = 3600  # 1 hour retention

# Application State
job_queue = asyncio.Queue()
jobs: Dict[str, Dict] = {}
# Semester to limit concurrency to MAX_CONCURRENT_JOBS
concurrency_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

def _relocate_root_job_artifacts(job_id: str, job_output_dir: str) -> bool:
    """
    Backward-compat rescue:
    If main.py accidentally wrote metadata/clips into OUTPUT_DIR root (e.g. output/<jobid>_...),
    move them into output/<job_id>/ so the API can find and serve them.
    """
    try:
        os.makedirs(job_output_dir, exist_ok=True)
        root = OUTPUT_DIR
        pattern = os.path.join(root, f"{job_id}_*_metadata.json")
        meta_candidates = sorted(glob.glob(pattern), key=lambda p: os.path.getmtime(p), reverse=True)
        if not meta_candidates:
            return False

        # Move the newest metadata and its associated clips.
        metadata_path = meta_candidates[0]
        base_name = os.path.basename(metadata_path).replace("_metadata.json", "")

        # Move metadata
        dest_metadata = os.path.join(job_output_dir, os.path.basename(metadata_path))
        if os.path.abspath(metadata_path) != os.path.abspath(dest_metadata):
            shutil.move(metadata_path, dest_metadata)

        # Move any clips that match the same base_name into the job folder
        clip_pattern = os.path.join(root, f"{base_name}_clip_*.mp4")
        for clip_path in glob.glob(clip_pattern):
            dest_clip = os.path.join(job_output_dir, os.path.basename(clip_path))
            if os.path.abspath(clip_path) != os.path.abspath(dest_clip):
                shutil.move(clip_path, dest_clip)

        # Also move any temp_ clips that might remain
        temp_clip_pattern = os.path.join(root, f"temp_{base_name}_clip_*.mp4")
        for clip_path in glob.glob(temp_clip_pattern):
            dest_clip = os.path.join(job_output_dir, os.path.basename(clip_path))
            if os.path.abspath(clip_path) != os.path.abspath(dest_clip):
                shutil.move(clip_path, dest_clip)

        return True
    except Exception:
        return False

def _default_score_by_rank(rank: int) -> int:
    return max(55, 92 - (rank * 6))

def _score_band(score: int) -> str:
    if score >= 80:
        return "top"
    if score >= 65:
        return "medium"
    return "low"

def _normalize_confidence(raw_confidence, score: int) -> float:
    try:
        conf = float(raw_confidence)
    except (TypeError, ValueError):
        conf = score / 100.0
    return round(max(0.0, min(1.0, conf)), 2)

def _normalize_topic_tags(raw_tags) -> List[str]:
    if isinstance(raw_tags, str):
        raw_tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
    if not isinstance(raw_tags, list):
        return []

    out: List[str] = []
    seen = set()
    for tag in raw_tags:
        if not isinstance(tag, str):
            continue
        clean = tag.strip().lstrip("#").lower()[:24]
        if not clean or clean in seen:
            continue
        seen.add(clean)
        out.append(clean)
        if len(out) >= 5:
            break
    return out

def _default_topic_tags(clip: Dict) -> List[str]:
    text = " ".join([
        str(clip.get("video_title_for_youtube_short", "")),
        str(clip.get("video_description_for_tiktok", "")),
        str(clip.get("video_description_for_instagram", "")),
    ]).lower()
    words = re.findall(r"[a-zA-ZÀ-ÿ0-9]{4,}", text)
    stop = {
        "this", "that", "with", "para", "como", "este", "esta", "from",
        "about", "your", "have", "will", "they", "porque", "cuando",
        "donde", "video", "viral", "short", "shorts", "follow", "comment"
    }
    tags: List[str] = []
    seen = set()
    for w in words:
        if w in stop or w in seen:
            continue
        seen.add(w)
        tags.append(w[:24])
        if len(tags) >= 3:
            break
    return tags

def _tokenize_query(text: str) -> List[str]:
    if not text:
        return []
    words = re.findall(r"[a-zA-ZÀ-ÿ0-9]{3,}", text.lower())
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "como", "para",
        "cuando", "donde", "sobre", "porque", "video", "clip", "short", "shorts"
    }
    out: List[str] = []
    seen = set()
    for w in words:
        if w in stop or w in seen:
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= 8:
            break
    return out

def _normalize_clip_payload(clip: Dict, rank: int) -> Dict:
    """
    Ensure clip carries stable metadata used by the dashboard:
    - clip_index
    - virality_score
    - score_reason
    """
    if not isinstance(clip, dict):
        clip = {}

    clip['clip_index'] = int(clip.get('clip_index', rank))

    raw_score = clip.get('virality_score')
    try:
        score = int(round(float(raw_score)))
    except (TypeError, ValueError):
        score = _default_score_by_rank(rank)
    clip['virality_score'] = max(0, min(100, score))
    clip['score_band'] = _score_band(clip['virality_score'])
    clip['selection_confidence'] = _normalize_confidence(clip.get('selection_confidence'), clip['virality_score'])

    reason = clip.get('score_reason')
    if not reason:
        reason = f"AI ranking position #{rank+1} based on hook and retention potential."
    clip['score_reason'] = str(reason).strip()[:220]

    tags = _normalize_topic_tags(clip.get('topic_tags'))
    if not tags:
        tags = _default_topic_tags(clip)
    clip['topic_tags'] = tags

    return clip

async def cleanup_jobs():
    """Background task to remove old jobs and files."""
    import time
    print("🧹 Cleanup task started.")
    while True:
        try:
            await asyncio.sleep(300) # Check every 5 minutes
            now = time.time()
            
            # Simple directory cleanup based on modification time
            # Check OUTPUT_DIR
            for job_id in os.listdir(OUTPUT_DIR):
                job_path = os.path.join(OUTPUT_DIR, job_id)
                if os.path.isdir(job_path):
                    if now - os.path.getmtime(job_path) > JOB_RETENTION_SECONDS:
                        print(f"🧹 Purging old job: {job_id}")
                        shutil.rmtree(job_path, ignore_errors=True)
                        if job_id in jobs:
                            del jobs[job_id]

            # Cleanup Uploads
            for filename in os.listdir(UPLOAD_DIR):
                file_path = os.path.join(UPLOAD_DIR, filename)
                try:
                    if now - os.path.getmtime(file_path) > JOB_RETENTION_SECONDS:
                         os.remove(file_path)
                except Exception: pass

        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")

async def process_queue():
    """Background worker to process jobs from the queue with concurrency limit."""
    print(f"🚀 Job Queue Worker started with {MAX_CONCURRENT_JOBS} concurrent slots.")
    while True:
        try:
            # Wait for a job
            job_id = await job_queue.get()
            
            # Acquire semaphore slot (waits if max jobs are running)
            await concurrency_semaphore.acquire()
            print(f"🔄 Acquired slot for job: {job_id}")

            # Process in background task to not block the loop (allowing other slots to fill)
            asyncio.create_task(run_job_wrapper(job_id))
            
        except Exception as e:
            print(f"❌ Queue dispatch error: {e}")
            await asyncio.sleep(1)

async def run_job_wrapper(job_id):
    """Wrapper to run job and release semaphore"""
    try:
        job = jobs.get(job_id)
        if job:
            await run_job(job_id, job)
    except Exception as e:
         print(f"❌ Job wrapper error {job_id}: {e}")
    finally:
        # Always release semaphore and mark queue task done
        concurrency_semaphore.release()
        job_queue.task_done()
        print(f"✅ Released slot for job: {job_id}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start worker and cleanup
    worker_task = asyncio.create_task(process_queue())
    cleanup_task = asyncio.create_task(cleanup_jobs())
    yield
    # Cleanup (optional: cancel worker)

app = FastAPI(lifespan=lifespan)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving videos
app.mount("/videos", StaticFiles(directory=OUTPUT_DIR), name="videos")

class ProcessRequest(BaseModel):
    url: str

def enqueue_output(out, job_id):
    """Reads output from a subprocess and appends it to jobs logs."""
    try:
        for line in iter(out.readline, b''):
            decoded_line = line.decode('utf-8').strip()
            if decoded_line:
                print(f"📝 [Job Output] {decoded_line}")
                if job_id in jobs:
                    jobs[job_id]['logs'].append(decoded_line)
    except Exception as e:
        print(f"Error reading output for job {job_id}: {e}")
    finally:
        out.close()

async def run_job(job_id, job_data):
    """Executes the subprocess for a specific job."""
    
    cmd = job_data['cmd']
    env = job_data['env']
    output_dir = job_data['output_dir']
    
    jobs[job_id]['status'] = 'processing'
    jobs[job_id]['logs'].append("Job started by worker.")
    print(f"🎬 [run_job] Executing command for {job_id}: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Merge stderr to stdout
            env=env,
            cwd=os.getcwd()
        )
        
        # We need to capture logs in a thread because Popen isn't async
        t_log = threading.Thread(target=enqueue_output, args=(process.stdout, job_id))
        t_log.daemon = True
        t_log.start()
        
        # Async wait for process with incremental updates
        start_wait = time.time()
        while process.poll() is None:
            await asyncio.sleep(2)
            
            # Check for partial results every 2 seconds
            # Look for metadata file
            try:
                json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
                if json_files:
                    target_json = json_files[0]
                    # Read metadata (it might be being written to, so simple try/except or just read)
                    # Use a lock or just robust read? json.load might fail if file is partial.
                    # Usually main.py writes it once at start (based on my review).
                    if os.path.getsize(target_json) > 0:
                        with open(target_json, 'r') as f:
                            data = json.load(f)
                            
                        base_name = os.path.basename(target_json).replace('_metadata.json', '')
                        clips = data.get('shorts', [])
                        cost_analysis = data.get('cost_analysis')
                        
                        # Check which clips actually exist on disk
                        ready_clips = []
                        for i, clip in enumerate(clips):
                             clip = _normalize_clip_payload(clip, i)
                             clip_filename = f"{base_name}_clip_{i+1}.mp4"
                             clip_path = os.path.join(output_dir, clip_filename)
                             if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
                                 # Checking if file is growing? For now assume if it exists and main.py moves it there, it's done.
                                 # main.py writes to temp_... then moves to final name. So presence means ready!
                                 clip['video_url'] = f"/videos/{job_id}/{clip_filename}"
                                 ready_clips.append(clip)
                        
                        if ready_clips:
                             jobs[job_id]['result'] = {'clips': ready_clips, 'cost_analysis': cost_analysis}
            except Exception as e:
                # Ignore read errors during processing
                pass

        returncode = process.returncode
        
        if returncode == 0:
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['logs'].append("Process finished successfully.")
            
            # Start S3 upload in background (silent, non-blocking)
            loop = asyncio.get_event_loop()
            loop.run_in_executor(None, upload_job_artifacts, output_dir, job_id)
            
            # Find result JSON
            json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
            if not json_files:
                # Backward-compat rescue if outputs were written to OUTPUT_DIR root
                if _relocate_root_job_artifacts(job_id, output_dir):
                    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
            if json_files:
                target_json = json_files[0] 
                with open(target_json, 'r') as f:
                    data = json.load(f)
                
                # Enhance result with video URLs
                base_name = os.path.basename(target_json).replace('_metadata.json', '')
                clips = data.get('shorts', [])
                cost_analysis = data.get('cost_analysis')

                for i, clip in enumerate(clips):
                     clip = _normalize_clip_payload(clip, i)
                     clip_filename = f"{base_name}_clip_{i+1}.mp4"
                     clip['video_url'] = f"/videos/{job_id}/{clip_filename}"
                
                jobs[job_id]['result'] = {'clips': clips, 'cost_analysis': cost_analysis}
            else:
                 jobs[job_id]['status'] = 'failed'
                 jobs[job_id]['logs'].append("No metadata file generated.")
        else:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['logs'].append(f"Process failed with exit code {returncode}")
            
    except Exception as e:
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['logs'].append(f"Execution error: {str(e)}")

@app.post("/api/process")
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
    ffmpeg_crf: Optional[int] = Form(None)
):
    api_key = request.headers.get("X-Gemini-Key")
    if not api_key:
        raise HTTPException(status_code=400, detail="Missing X-Gemini-Key header")
    
    # Handle JSON body manually for URL payload
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        url = body.get("url")
        language = body.get("language") or language
        max_clips = body.get("max_clips") or max_clips
        whisper_backend = body.get("whisper_backend") or whisper_backend
        whisper_model = body.get("whisper_model") or whisper_model
        word_timestamps = body.get("word_timestamps") or word_timestamps
        ffmpeg_preset = body.get("ffmpeg_preset") or ffmpeg_preset
        ffmpeg_crf = body.get("ffmpeg_crf") or ffmpeg_crf
    
    if not url and not file:
        raise HTTPException(status_code=400, detail="Must provide URL or File")

    job_id = str(uuid.uuid4())
    job_output_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_output_dir, exist_ok=True)
    
    # Prepare Command
    cmd = ["python", "-u", "main.py"] # -u for unbuffered
    env = os.environ.copy()
    env["GEMINI_API_KEY"] = api_key # Override with key from request
    
    if url:
        cmd.extend(["-u", url])
    else:
        # Save uploaded file with size limit check
        input_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
        
        # Read file in chunks to check size
        size = 0
        limit_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
        
        with open(input_path, "wb") as buffer:
            while content := await file.read(1024 * 1024): # Read 1MB chunks
                size += len(content)
                if size > limit_bytes:
                    os.remove(input_path)
                    shutil.rmtree(job_output_dir)
                    raise HTTPException(status_code=413, detail=f"File too large. Max size {MAX_FILE_SIZE_MB}MB")
                buffer.write(content)
                
        cmd.extend(["-i", input_path])

    if language:
        cmd.extend(["--language", str(language)])
    if max_clips:
        cmd.extend(["--max-clips", str(max_clips)])
    if whisper_backend:
        cmd.extend(["--whisper-backend", str(whisper_backend)])
    if whisper_model:
        cmd.extend(["--whisper-model", str(whisper_model)])
    if word_timestamps is not None:
        cmd.extend(["--word-timestamps", str(word_timestamps)])
    if ffmpeg_preset:
        cmd.extend(["--ffmpeg-preset", str(ffmpeg_preset)])
    if ffmpeg_crf:
        cmd.extend(["--ffmpeg-crf", str(ffmpeg_crf)])

    cmd.extend(["-o", job_output_dir])

    # Enqueue Job
    jobs[job_id] = {
        'status': 'queued',
        'logs': [f"Job {job_id} queued."],
        'cmd': cmd,
        'env': env,
        'output_dir': job_output_dir,
        'input_path': input_path if not url else None
    }
    
    await job_queue.put(job_id)
    
    return {"job_id": job_id, "status": "queued"}

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return {
        "status": job['status'],
        "logs": job['logs'],
        "result": job.get('result')
    }

from editor import VideoEditor
from subtitles import generate_srt, burn_subtitles

class EditRequest(BaseModel):
    job_id: str
    clip_index: int
    api_key: Optional[str] = None
    input_filename: Optional[str] = None

@app.post("/api/edit")
async def edit_clip(
    req: EditRequest,
    x_gemini_key: Optional[str] = Header(None, alias="X-Gemini-Key")
):
    # Determine API Key
    final_api_key = req.api_key or x_gemini_key or os.environ.get("GEMINI_API_KEY")
    
    if not final_api_key:
        raise HTTPException(status_code=400, detail="Missing Gemini API Key (Header or Body)")

    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[req.job_id]
    if 'result' not in job or 'clips' not in job['result']:
        raise HTTPException(status_code=400, detail="Job result not available")
        
    try:
        # Resolve Input Path: Prefer explict input_filename from frontend (chaining edits)
        if req.input_filename:
            # Security: Ensure just a filename, no paths
            safe_name = os.path.basename(req.input_filename)
            input_path = os.path.join(OUTPUT_DIR, req.job_id, safe_name)
            filename = safe_name
        else:
            # Fallback to original clip
            clip = job['result']['clips'][req.clip_index]
            filename = clip['video_url'].split('/')[-1]
            input_path = os.path.join(OUTPUT_DIR, req.job_id, filename)
        
        if not os.path.exists(input_path):
             raise HTTPException(status_code=404, detail=f"Video file not found: {input_path}")

        # Define output path for edited video
        base_name = os.path.splitext(filename)[0]
        edited_filename = f"edited_{base_name}.mp4"
        output_path = os.path.join(OUTPUT_DIR, req.job_id, edited_filename)
        
        # Run editing in a thread to avoid blocking main loop
        # Since VideoEditor uses blocking calls (subprocess, API wait)
        def run_edit():
            editor = VideoEditor(api_key=final_api_key)
            
            # SAFE FILE RENAMING STRATEGY (Avoid UnicodeEncodeError in Docker)
            # Create a safe ASCII filename in the same directory
            safe_filename = f"temp_input_{req.job_id}.mp4"
            safe_input_path = os.path.join(OUTPUT_DIR, req.job_id, safe_filename)
            
            # Copy original file to safe path
            # (Copy is safer than rename if something crashes, we keep original)
            shutil.copy(input_path, safe_input_path)
            
            try:
                # 1. Upload (using safe path)
                vid_file = editor.upload_video(safe_input_path)
                
                # 2. Get duration
                import cv2
                cap = cv2.VideoCapture(safe_input_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = frame_count / fps if fps else 0
                cap.release()
                
                # Load transcript from metadata
                transcript = None
                try:
                    meta_files = glob.glob(os.path.join(OUTPUT_DIR, req.job_id, "*_metadata.json"))
                    if meta_files:
                        with open(meta_files[0], 'r') as f:
                            data = json.load(f)
                            transcript = data.get('transcript')
                except Exception as e:
                    print(f"⚠️ Could not load transcript for editing context: {e}")

                # 3. Get Plan (Filter String)
                filter_data = editor.get_ffmpeg_filter(vid_file, duration, fps=fps, width=width, height=height, transcript=transcript)
                
                # 4. Apply
                # Use safe output name first
                safe_output_path = os.path.join(OUTPUT_DIR, req.job_id, f"temp_output_{req.job_id}.mp4")
                editor.apply_edits(safe_input_path, safe_output_path, filter_data)
                
                # Move result to final destination (rename works even if dest name has unicode if filesystem supports it, 
                # but python might still struggle if locale is broken? No, os.rename usually handles it better than subprocess args)
                # Actually, output_path is defined above: f"edited_{filename}"
                # If filename has unicode, output_path has unicode.
                # Let's hope shutil.move / os.rename works.
                if os.path.exists(safe_output_path):
                    shutil.move(safe_output_path, output_path)
                
                return filter_data
            finally:
                # Cleanup temp safe input
                if os.path.exists(safe_input_path):
                    os.remove(safe_input_path)

        # Run in thread pool
        loop = asyncio.get_event_loop()
        plan = await loop.run_in_executor(None, run_edit)
        
        # Update clip URL in the job result? 
        # Or return new URL and let frontend handle it?
        # Updating job result allows persistence if page refreshes.
        
        new_video_url = f"/videos/{req.job_id}/{edited_filename}"
        
        # Start a new "edited" clip entry or just update the current one?
        # Let's update the current one's video_url but keep backup?
        # Or return the new URL to the frontend to display.
        
        return {
            "success": True, 
            "new_video_url": new_video_url,
            "edit_plan": plan
        }

    except Exception as e:
        print(f"❌ Edit Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class SubtitleRequest(BaseModel):
    job_id: str
    clip_index: int
    position: str = "bottom" # top, middle, bottom
    font_size: int = 16
    font_family: str = "Verdana"
    font_color: str = "#FFFFFF"
    stroke_color: str = "#000000"
    stroke_width: int = 2
    bold: bool = True
    box_color: str = "#000000"
    box_opacity: int = 60
    srt_content: Optional[str] = None
    input_filename: Optional[str] = None

@app.post("/api/subtitle")
async def add_subtitles(req: SubtitleRequest):
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Reload job data from disk just in case metadata was updated
    job = jobs[req.job_id]
    
    # We need to access metadata.json to get the transcript
    output_dir = os.path.join(OUTPUT_DIR, req.job_id)
    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
    
    if not json_files:
        raise HTTPException(status_code=404, detail="Metadata not found")
        
    with open(json_files[0], 'r') as f:
        data = json.load(f)
        
    transcript = data.get('transcript')
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript not found in metadata. Please process a new video.")
        
    clips = data.get('shorts', [])
    if req.clip_index >= len(clips):
        raise HTTPException(status_code=404, detail="Clip not found")
        
    clip_data = clips[req.clip_index]
    
    # Video Path
    if req.input_filename:
        # Use chained file
        filename = os.path.basename(req.input_filename)
    else:
        # Fallback to standard naming
        filename = clip_data.get('video_url', '').split('/')[-1]
        if not filename:
             base_name = os.path.basename(json_files[0]).replace('_metadata.json', '')
             filename = f"{base_name}_clip_{req.clip_index+1}.mp4"
         
    input_path = os.path.join(output_dir, filename)
    if not os.path.exists(input_path):
        # Try looking for edited version if url implied it?
        # Just fail if not found.
        raise HTTPException(status_code=404, detail=f"Video file not found: {input_path}")
        
    # Define outputs
    srt_filename = f"subs_{req.clip_index}_{int(time.time())}.srt"
    srt_path = os.path.join(output_dir, srt_filename)
    
    # Output video
    # We create a new file "subtitled_..."
    base_name = os.path.splitext(filename)[0]
    output_filename = f"subtitled_{base_name}.mp4"
    output_path = os.path.join(output_dir, output_filename)
    
    try:
        # 1. Generate or use provided SRT
        if req.srt_content:
             with open(srt_path, 'w', encoding='utf-8') as f:
                  f.write(req.srt_content)
        else:
             success = generate_srt(transcript, clip_data['start'], clip_data['end'], srt_path)
             if not success:
                  raise HTTPException(status_code=400, detail="No words found for this clip range.")
             
        # 2. Burn Subtitles
        # Run in thread pool
        def run_burn():
             burn_subtitles(
                 input_path,
                 srt_path,
                 output_path,
                 alignment=req.position,
                 fontsize=req.font_size,
                 font_name=req.font_family,
                 font_color=req.font_color,
                 stroke_color=req.stroke_color,
                 stroke_width=req.stroke_width,
                 bold=req.bold,
                 box_color=req.box_color,
                 box_opacity=req.box_opacity
             )
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, run_burn)
        
        # 3. Update Result?
        # We return the new URL.
        new_video_url = f"/videos/{req.job_id}/{output_filename}"
        
        return {
            "success": True,
            "new_video_url": new_video_url
        }
        
    except Exception as e:
        print(f"❌ Subtitle Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class SubtitlePreviewRequest(BaseModel):
    job_id: str
    clip_index: int

@app.post("/api/subtitle/preview")
async def preview_subtitles(req: SubtitlePreviewRequest):
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    output_dir = os.path.join(OUTPUT_DIR, req.job_id)
    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
    if not json_files:
        raise HTTPException(status_code=404, detail="Metadata not found")

    with open(json_files[0], 'r') as f:
        data = json.load(f)

    transcript = data.get('transcript')
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript not found in metadata.")

    clips = data.get('shorts', [])
    if req.clip_index >= len(clips):
        raise HTTPException(status_code=404, detail="Clip not found")

    clip_data = clips[req.clip_index]
    temp_name = f"subs_preview_{req.clip_index}_{int(time.time())}.srt"
    temp_path = os.path.join(output_dir, temp_name)
    success = generate_srt(transcript, clip_data['start'], clip_data['end'], temp_path)
    if not success:
        raise HTTPException(status_code=400, detail="No words found for this clip range.")

    with open(temp_path, 'r', encoding='utf-8') as f:
        content = f.read()
    try:
        os.remove(temp_path)
    except Exception:
        pass

    return {"srt": content}

class RecutRequest(BaseModel):
    job_id: str
    clip_index: int
    start: float
    end: float

@app.post("/api/recut")
async def recut_clip(req: RecutRequest):
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[req.job_id]
    input_path = job.get('input_path')
    if not input_path or not os.path.exists(input_path):
        raise HTTPException(status_code=400, detail="Original source video not available for recut.")

    if req.start < 0 or req.end <= req.start:
        raise HTTPException(status_code=400, detail="Invalid start/end times.")

    output_dir = os.path.join(OUTPUT_DIR, req.job_id)
    os.makedirs(output_dir, exist_ok=True)

    # Cut from original source
    temp_cut = os.path.join(output_dir, f"temp_recut_{req.clip_index}_{int(time.time())}.mp4")
    out_name = f"reclip_{req.clip_index+1}_{int(time.time())}.mp4"
    out_path = os.path.join(output_dir, out_name)

    cut_cmd = [
        'ffmpeg', '-y',
        '-ss', str(req.start),
        '-to', str(req.end),
        '-i', input_path,
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23', '-preset', 'fast',
        '-c:a', 'aac',
        '-movflags', '+faststart',
        temp_cut
    ]
    res = subprocess.run(cut_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if res.returncode != 0:
        raise HTTPException(status_code=500, detail=res.stderr.decode())

    # Process vertical
    try:
        from main import process_video_to_vertical
        success = process_video_to_vertical(temp_cut, out_path, "fast", 23)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to process vertical video.")
    finally:
        if os.path.exists(temp_cut):
            os.remove(temp_cut)

    # Update metadata
    json_files = glob.glob(os.path.join(output_dir, "*_metadata.json"))
    if json_files:
        try:
            with open(json_files[0], 'r') as f:
                data = json.load(f)
            clips = data.get('shorts', [])
            if req.clip_index < len(clips):
                clips[req.clip_index]['start'] = req.start
                clips[req.clip_index]['end'] = req.end
                with open(json_files[0], 'w') as f:
                    json.dump(data, f, indent=2)
        except Exception:
            pass

    # Update job result
    if 'result' in job and 'clips' in job['result'] and req.clip_index < len(job['result']['clips']):
        job['result']['clips'][req.clip_index]['video_url'] = f"/videos/{req.job_id}/{out_name}"
        job['result']['clips'][req.clip_index]['start'] = req.start
        job['result']['clips'][req.clip_index]['end'] = req.end

    return {"success": True, "new_video_url": f"/videos/{req.job_id}/{out_name}"}

class ExportPackRequest(BaseModel):
    job_id: str
    include_video_files: bool = True
    include_srt_files: bool = True

@app.post("/api/export/pack")
async def export_pack(req: ExportPackRequest):
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    output_dir = os.path.join(OUTPUT_DIR, req.job_id)
    if not os.path.isdir(output_dir):
        raise HTTPException(status_code=404, detail="Job output directory not found")

    metadata_files = sorted(glob.glob(os.path.join(output_dir, "*_metadata.json")))
    metadata_path = metadata_files[0] if metadata_files else None
    metadata = {}
    if metadata_path:
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}

    job = jobs[req.job_id]
    clips = []
    if isinstance(job.get("result"), dict):
        clips = job["result"].get("clips", []) or []
    if not clips:
        clips = metadata.get("shorts", []) if isinstance(metadata, dict) else []

    normalized_clips = []
    for i, clip in enumerate(clips):
        normalized = _normalize_clip_payload(dict(clip) if isinstance(clip, dict) else {}, i)
        if not normalized.get("video_url"):
            clip_filename = f"{req.job_id}_clip_{i+1}.mp4"
            # Best effort fallback if current naming is unknown.
            candidates = sorted(glob.glob(os.path.join(output_dir, f"*_clip_{i+1}.mp4")))
            if candidates:
                clip_filename = os.path.basename(candidates[0])
            normalized["video_url"] = f"/videos/{req.job_id}/{clip_filename}"
        normalized_clips.append(normalized)

    manifest = {
        "job_id": req.job_id,
        "generated_at": int(time.time()),
        "clips": normalized_clips
    }

    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf)
    writer.writerow([
        "clip_number",
        "clip_index",
        "virality_score",
        "score_band",
        "selection_confidence",
        "start_seconds",
        "end_seconds",
        "youtube_title",
        "caption_tiktok",
        "caption_instagram",
        "topic_tags"
    ])
    for i, clip in enumerate(normalized_clips):
        writer.writerow([
            i + 1,
            clip.get("clip_index", i),
            clip.get("virality_score", ""),
            clip.get("score_band", ""),
            clip.get("selection_confidence", ""),
            clip.get("start", ""),
            clip.get("end", ""),
            clip.get("video_title_for_youtube_short", ""),
            clip.get("video_description_for_tiktok", ""),
            clip.get("video_description_for_instagram", ""),
            ",".join(clip.get("topic_tags", []) or [])
        ])

    zip_name = f"agency_pack_{req.job_id}_{int(time.time())}.zip"
    zip_path = os.path.join(output_dir, zip_name)
    clip_files_added = 0
    srt_files_added = 0

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        zf.writestr("copies.csv", csv_buf.getvalue())

        if metadata_path and os.path.exists(metadata_path):
            zf.write(metadata_path, arcname=f"metadata/{os.path.basename(metadata_path)}")

        if req.include_video_files:
            seen = set()
            for clip in normalized_clips:
                video_url = clip.get("video_url", "")
                filename = os.path.basename(video_url)
                if not filename:
                    continue
                file_path = os.path.join(output_dir, filename)
                if file_path in seen or not os.path.exists(file_path):
                    continue
                seen.add(file_path)
                zf.write(file_path, arcname=f"clips/{filename}")
                clip_files_added += 1

        if req.include_srt_files:
            srt_paths = sorted(glob.glob(os.path.join(output_dir, "*.srt")))
            for srt_path in srt_paths:
                zf.write(srt_path, arcname=f"srt/{os.path.basename(srt_path)}")
                srt_files_added += 1

    return {
        "success": True,
        "pack_url": f"/videos/{req.job_id}/{zip_name}",
        "clips_in_manifest": len(normalized_clips),
        "video_files_added": clip_files_added,
        "srt_files_added": srt_files_added
    }

class ClipSearchRequest(BaseModel):
    job_id: str
    query: str
    limit: int = 5

@app.post("/api/search/clips")
async def search_clips(req: ClipSearchRequest):
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    output_dir = os.path.join(OUTPUT_DIR, req.job_id)
    json_files = sorted(glob.glob(os.path.join(output_dir, "*_metadata.json")))
    if not json_files:
        raise HTTPException(status_code=404, detail="Metadata not found")

    try:
        with open(json_files[0], "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read metadata: {e}")

    transcript = (data.get("transcript") or {})
    segments = transcript.get("segments") or []
    if not isinstance(segments, list) or not segments:
        raise HTTPException(status_code=400, detail="Transcript segments not available")

    query = (req.query or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    keywords = _tokenize_query(query)
    if not keywords:
        raise HTTPException(status_code=400, detail="Query too short or unsupported")

    transcript_text = str(transcript.get("text", "")).lower()
    segments_scored = []
    for idx, seg in enumerate(segments):
        seg_text = str(seg.get("text", "")).lower()
        if not seg_text:
            continue
        # Weighted keyword matching with phrase bonus.
        score = 0.0
        matched = []
        for kw in keywords:
            if kw in seg_text:
                matched.append(kw)
                score += 1.0
        if query.lower() in seg_text:
            score += 1.5
        elif query.lower() in transcript_text:
            # Global mention exists; local segment still gets a soft boost.
            score += 0.25

        if score <= 0:
            continue
        segments_scored.append((score, idx, matched, seg))

    if not segments_scored:
        return {"matches": [], "keywords": keywords}

    segments_scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    limit = max(1, min(20, int(req.limit)))

    # Estimate video duration from transcript.
    duration = 0.0
    for seg in segments:
        try:
            duration = max(duration, float(seg.get("end", 0.0)))
        except Exception:
            pass

    matches = []
    used_ranges = []
    for score, idx, matched, seg in segments_scored:
        start = max(0.0, float(seg.get("start", 0.0)) - 3.0)
        end = float(seg.get("end", 0.0)) + 12.0
        if end - start < 15.0:
            end = start + 15.0
        if end - start > 60.0:
            end = start + 60.0
        if duration > 0:
            end = min(end, duration)

        overlap = False
        for s0, e0 in used_ranges:
            if not (end <= s0 or start >= e0):
                overlap = True
                break
        if overlap:
            continue
        used_ranges.append((start, end))

        matches.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
            "match_score": round(float(score), 3),
            "keywords": matched,
            "snippet": str(seg.get("text", "")).strip()[:240]
        })
        if len(matches) >= limit:
            break

    return {
        "matches": matches,
        "keywords": keywords
    }

class SocialPostRequest(BaseModel):
    job_id: str
    clip_index: int
    api_key: str
    user_id: str
    platforms: List[str] # ["tiktok", "instagram", "youtube"]
    # Optional overrides if frontend wants to edit them
    title: Optional[str] = None
    description: Optional[str] = None
    scheduled_date: Optional[str] = None # ISO-8601 string
    timezone: Optional[str] = "UTC"

import httpx

@app.post("/api/social/post")
async def post_to_socials(req: SocialPostRequest):
    if req.job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[req.job_id]
    if 'result' not in job or 'clips' not in job['result']:
        raise HTTPException(status_code=400, detail="Job result not available")
        
    try:
        clip = job['result']['clips'][req.clip_index]
        # Video URL is relative /videos/..., we need absolute file path
        # clip['video_url'] is like "/videos/{job_id}/{filename}"
        # We constructed it as: f"/videos/{job_id}/{clip_filename}"
        # And file is at f"{OUTPUT_DIR}/{job_id}/{clip_filename}"
        
        filename = clip['video_url'].split('/')[-1]
        file_path = os.path.join(OUTPUT_DIR, req.job_id, filename)
        
        if not os.path.exists(file_path):
             raise HTTPException(status_code=404, detail=f"Video file not found: {file_path}")

        # Construct parameters for Upload-Post API
        # Fallbacks
        final_title = req.title or clip.get('title', 'Viral Short')
        final_description = req.description or clip.get('video_description_for_instagram') or clip.get('video_description_for_tiktok') or "Check this out!"
        
        # Prepare form data
        url = "https://api.upload-post.com/api/upload"
        headers = {
            "Authorization": f"Apikey {req.api_key}"
        }
        
        # Prepare data as dict (httpx handles lists for multiple values)
        data_payload = {
            "user": req.user_id,
            "title": final_title,
            "platform[]": req.platforms # Pass list directly
        }

        # Add scheduling if present
        if req.scheduled_date:
            data_payload["scheduled_date"] = req.scheduled_date
            if req.timezone:
                data_payload["timezone"] = req.timezone
        
        # Add Platform specifics
        if "tiktok" in req.platforms:
             data_payload["tiktok_title"] = final_description
             
        if "instagram" in req.platforms:
             data_payload["instagram_title"] = final_description
             data_payload["media_type"] = "REELS"

        if "youtube" in req.platforms:
             yt_title = req.title or clip.get('video_title_for_youtube_short', final_title)
             data_payload["youtube_title"] = yt_title
             data_payload["youtube_description"] = final_description
             data_payload["privacyStatus"] = "public"

        # Send File
        # httpx AsyncClient requires async file reading or bytes. 
        # Since we have MAX_FILE_SIZE_MB, reading into memory is safe-ish.
        with open(file_path, "rb") as f:
            file_content = f.read()
            
        files = {
            "video": (filename, file_content, "video/mp4")
        }

        # Switch to synchronous Client to avoid "sync request with AsyncClient" error with multipart/files
        with httpx.Client(timeout=120.0) as client:
            print(f"📡 Sending to Upload-Post for platforms: {req.platforms}")
            response = client.post(url, headers=headers, data=data_payload, files=files)
            
        if response.status_code not in [200, 201, 202]: # Added 201
             print(f"❌ Upload-Post Error: {response.text}")
             raise HTTPException(status_code=response.status_code, detail=f"Vendor API Error: {response.text}")

        return response.json()

    except Exception as e:
        print(f"❌ Social Post Exception: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/social/user")
async def get_social_user(api_key: str = Header(..., alias="X-Upload-Post-Key")):
    """Proxy to fetch user ID from Upload-Post"""
    if not api_key:
         raise HTTPException(status_code=400, detail="Missing X-Upload-Post-Key header")
         
    url = "https://api.upload-post.com/api/uploadposts/users"
    print(f"🔍 Fetching User ID from: {url}")
    headers = {"Authorization": f"Apikey {api_key}"}
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                print(f"❌ Upload-Post User Fetch Error: {resp.text}")
                raise HTTPException(status_code=resp.status_code, detail=f"Failed to fetch user: {resp.text}")
            
            data = resp.json()
            print(f"🔍 Upload-Post User Response: {data}")
            
            user_id = None
            # The structure is {'success': True, 'profiles': [{'username': '...'}, ...]}
            profiles_list = []
            if isinstance(data, dict):
                 raw_profiles = data.get('profiles', [])
                 if isinstance(raw_profiles, list):
                     for p in raw_profiles:
                         username = p.get('username')
                         if username:
                             # Determine connected platforms
                             socials = p.get('social_accounts', {})
                             connected = []
                             # Check typical platforms
                             for platform in ['tiktok', 'instagram', 'youtube']:
                                 account_info = socials.get(platform)
                                 # If it's a dict and typically has data, or just not empty string
                                 if isinstance(account_info, dict):
                                     connected.append(platform)
                             
                             profiles_list.append({
                                 "username": username,
                                 "connected": connected
                             })
            
            if not profiles_list:
                # Fallback if no profiles found
                return {"profiles": [], "error": "No profiles found"}
                
            return {"profiles": profiles_list}
            
        except Exception as e:
             raise HTTPException(status_code=500, detail=str(e))
