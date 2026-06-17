import subprocess
import re
import math
import struct
import os
from typing import Tuple, List, Optional
from urllib.parse import unquote

def _safe_input_filename(value: Optional[str]) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    filename = os.path.basename(raw)
    filename = unquote(filename)
    filename = unquote(filename)
    return os.path.basename(filename)

def _probe_media_duration_seconds(media_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        media_path
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode == 0:
            return max(0.0, float((proc.stdout or "").strip() or 0.0))
    except (FileNotFoundError, Exception):
        try:
            cmd_fallback = ["ffmpeg", "-i", media_path]
            proc_fb = subprocess.run(cmd_fallback, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            err_out = proc_fb.stderr or ""
            match = re.search(r"Duration:\s+(\d+):(\d+):(\d+\.\d+)", err_out)
            if match:
                h, m, s = match.groups()
                return float(h)*3600 + float(m)*60 + float(s)
        except Exception:
            pass
    return 0.0

def _probe_video_dimensions(media_path: str) -> Tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x",
        media_path
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode == 0:
            raw = str(proc.stdout or "").strip()
            if "x" in raw:
                w_raw, h_raw = raw.split("x", 1)
                return int(float(w_raw)), int(float(h_raw))
    except (FileNotFoundError, Exception):
        try:
            cmd_fallback = ["ffmpeg", "-i", media_path]
            proc_fb = subprocess.run(cmd_fallback, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            err_out = proc_fb.stderr or ""
            match = re.search(r"Video:.*?[\s,](\d+)x(\d+)[\s,]", err_out)
            if match:
                w, h = int(match.group(1)), int(match.group(2))
                if w > 0 and h > 0:
                    return w, h
        except Exception:
            pass
    return 0, 0

def _extract_waveform_peaks(media_path: str, buckets: int = 240, sample_rate: int = 11025) -> List[float]:
    safe_buckets = max(32, min(2000, int(buckets)))
    safe_rate = max(2000, min(48000, int(sample_rate)))

    cmd = [
        "ffmpeg",
        "-v", "error",
        "-i", media_path,
        "-ac", "1",
        "-ar", str(safe_rate),
        "-vn",
        "-f", "f32le",
        "pipe:1"
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="ignore") or "ffmpeg waveform extraction failed")

    raw = proc.stdout or b""
    sample_count = len(raw) // 4
    if sample_count <= 0:
        return [0.0] * safe_buckets

    floats = struct.unpack(f"<{sample_count}f", raw[:sample_count * 4])
    chunk_size = max(1, int(math.ceil(sample_count / safe_buckets)))
    peaks: List[float] = []

    for i in range(safe_buckets):
        start = i * chunk_size
        if start >= sample_count:
            peaks.append(0.0)
            continue
        end = min(sample_count, start + chunk_size)
        peak = 0.0
        for value in floats[start:end]:
            amp = abs(float(value))
            if amp > peak:
                peak = amp
        peaks.append(peak)

    max_peak = max(peaks) if peaks else 0.0
    if max_peak <= 0:
        return [0.0] * len(peaks)
    return [round(min(1.0, p / max_peak), 4) for p in peaks]
