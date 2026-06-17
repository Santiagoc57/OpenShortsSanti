import os
import subprocess
import re
import math
import uuid
import time
from typing import Tuple, List, Optional, Dict, Any
from ..utils.media import _probe_media_duration_seconds, _probe_video_dimensions, _safe_input_filename
from ..utils.text import _safe_float

# Module-level globals for lazy-loaded models
_SMART_REF_YOLO_MODEL = None
_SMART_REF_YOLO_UNAVAILABLE = False
_SMART_REF_FACE_CASCADE = None
_LAYOUT_OFFSET_FACTOR = 1.0

# 1. Geometry and Coordinate Utilities
def _even_int(value: float) -> int:
    iv = int(round(float(value)))
    if iv < 2: iv = 2
    if iv % 2 != 0: iv -= 1
    return max(2, iv)

def _aspect_ratio_to_float(aspect_ratio: str) -> float:
    ratio_map = {"9:16": 9.0 / 16.0, "16:9": 16.0 / 9.0}
    return ratio_map.get(str(aspect_ratio or "9:16"), 9.0 / 16.0)

def _derive_output_dimensions(in_w: int, in_h: int, target_ratio: float) -> Tuple[int, int]:
    source_ratio = (in_w / in_h) if in_w > 0 and in_h > 0 else target_ratio
    if source_ratio >= target_ratio:
        out_h = _even_int(in_h)
        out_w = _even_int(out_h * target_ratio)
    else:
        out_w = _even_int(in_w)
        out_h = _even_int(out_w / max(1e-6, target_ratio))
    return out_w, out_h

# 2. Layout Normalization
def _normalize_layout_fit_mode(fit_mode: Optional[str]) -> str:
    fit = str(fit_mode or "cover").strip().lower()
    return fit if fit in {"cover", "contain", "blur"} else "cover"

def _coerce_layout_zoom(value: Any, default: float = 1.0, fit_mode: str = "cover") -> float:
    fit = _normalize_layout_fit_mode(fit_mode)
    min_zoom = 1.0 if fit == "cover" else 0.5
    fallback = max(min_zoom, _safe_float(default, 1.0))
    return max(min_zoom, min(2.5, _safe_float(value, fallback)))

def _coerce_layout_offset(value: Any, default: float = 0.0) -> float:
    return max(-100.0, min(100.0, _safe_float(value, default)))

# 3. FFmpeg Filter Composition
def _build_manual_layout_ops_for_target(in_w: int, in_h: int, out_w: int, out_h: int, fit_mode: str = "cover", zoom: float = 1.0, offset_x: float = 0.0, offset_y: float = 0.0, input_label: Optional[str] = None) -> str:
    fit = _normalize_layout_fit_mode(fit_mode)
    z = _coerce_layout_zoom(zoom, 1.0, fit)
    ox_raw = _coerce_layout_offset(offset_x, 0.0) / 100.0
    oy_raw = _coerce_layout_offset(offset_y, 0.0) / 100.0
    if fit == "cover" and z <= 1.0001 and (abs(ox_raw) > 1e-6 or abs(oy_raw) > 1e-6): z = 1.06
    
    out_w, out_h = _even_int(out_w), _even_int(out_h)
    if fit in {"cover", "contain"}:
        base_scale = max(out_w/max(1,in_w), out_h/max(1,in_h)) if fit=="cover" else min(out_w/max(1,in_w), out_h/max(1,in_h))
        scale_factor = max(0.1, base_scale * z)
        capture_w, capture_h = int(out_w/scale_factor), int(out_h/scale_factor)
        filters, stage_w, stage_h = [], in_w, in_h
        if capture_w < stage_w or capture_h < stage_h:
            cw, ch = min(stage_w, capture_w), min(stage_h, capture_h)
            px, py = max(0.0, min(1.0, 0.5 + ox_raw)), max(0.0, min(1.0, 0.5 + oy_raw))
            cx = max(0, min(stage_w-cw, int(round(px*(stage_w-cw*z)+0.5*cw*(z-1)*(1+ox_raw)))))
            cy = max(0, min(stage_h-ch, int(round(py*(stage_h-ch*z)+0.5*ch*(z-1)*(1+oy_raw)))))
            filters.append(f"crop={cw}:{ch}:{cx}:{cy}")
            stage_w, stage_h = cw, ch
        sw, sh = _even_int(stage_w*scale_factor), _even_int(stage_h*scale_factor)
        filters.append(f"scale={sw}:{sh}")
        if sw < out_w or sh < out_h:
            px, py = max(0.0, min(1.0, 0.5 + ox_raw)), max(0.0, min(1.0, 0.5 + oy_raw))
            filters.append(f"pad={out_w}:{out_h}:{int(round(px*(out_w-sw)))}:{int(round(py*(out_h-sh)))}:black")
        return ",".join(filters)
    
    lbl = f"blur_{uuid.uuid4().hex[:4]}"
    bgs = max(out_w/max(1,in_w), out_h/max(1,in_h))
    fgs = min(out_w/max(1,in_w), out_h/max(1,in_h)) * z
    px, py = max(0.0, min(1.0, 0.5 + ox_raw)), max(0.0, min(1.0, 0.5 + oy_raw))
    fgx = f"W/2 + {z}*({px}*(W - w/{z}) - W/2) - {ox_raw*(z-1)/2.0}*W"
    fgy = f"H/2 + {z}*({py}*(H - h/{z}) - H/2) - {oy_raw*(z-1)/2.0}*H"
    in_prefix = f"[{input_label}]" if input_label else ""
    return f"{in_prefix}split[v_bg][v_fg];[v_bg]scale={_even_int(in_w*bgs)}:{_even_int(in_h*bgs)},crop={out_w}:{out_h},boxblur=40:10[v_out_bg];[v_fg]scale={_even_int(in_w*fgs)}:{_even_int(in_h*fgs)}[v_out_fg];[v_out_bg][v_out_fg]overlay={fgx}:{fgy}"

def _build_manual_layout_filter(input_video_path: str, aspect_ratio: str, fit_mode: str = "cover", zoom: float = 1.0, offset_x: float = 0.0, offset_y: float = 0.0) -> str:
    in_w, in_h = _probe_video_dimensions(input_video_path)
    if in_w <= 0 or in_h <= 0:
        in_w, in_h = 1920, 1080
    out_w, out_h = _derive_output_dimensions(in_w, in_h, _aspect_ratio_to_float(aspect_ratio))
    return _build_manual_layout_ops_for_target(in_w, in_h, out_w, out_h, fit_mode, zoom, offset_x, offset_y)

def _build_split_layout_filter_complex(input_video_path: str, aspect_ratio: str, fit_mode: str = "cover", zoom_a: float = 1.0, offset_a_x: float = 0.0, offset_a_y: float = 0.0, zoom_b: float = 1.0, offset_b_x: float = 0.0, offset_b_y: float = 0.0) -> str:
    in_w, in_h = _probe_video_dimensions(input_video_path)
    if in_w <= 0 or in_h <= 0:
        in_w, in_h = 1920, 1080
    out_w, out_h = _derive_output_dimensions(in_w, in_h, _aspect_ratio_to_float(aspect_ratio))
    half_h = _even_int(out_h / 2)
    top_ops = _build_manual_layout_ops_for_target(in_w, in_h, out_w, half_h, fit_mode, zoom_a, offset_a_x, offset_a_y, input_label="top")
    bottom_ops = _build_manual_layout_ops_for_target(in_w, in_h, out_w, half_h, fit_mode, zoom_b, offset_b_x, offset_b_y, input_label="bottom")
    return f"[0:v]split=2[top][bottom];{top_ops}[top_out];{bottom_ops}[bottom_out];[top_out][bottom_out]vstack=inputs=2"

# 4. Smart Reframe (Vision Tools) - Simplified for service
def _smart_ref_get_yolo_model():
    global _SMART_REF_YOLO_MODEL, _SMART_REF_YOLO_UNAVAILABLE
    if _SMART_REF_YOLO_UNAVAILABLE: return None
    if _SMART_REF_YOLO_MODEL is None:
        try:
            from ultralytics import YOLO
            _SMART_REF_YOLO_MODEL = YOLO("yolov8n.pt")
        except Exception:
            _SMART_REF_YOLO_UNAVAILABLE = True
            return None
    return _SMART_REF_YOLO_MODEL

def _smart_ref_get_face_cascade():
    global _SMART_REF_FACE_CASCADE
    if _SMART_REF_FACE_CASCADE: return _SMART_REF_FACE_CASCADE
    try:
        import cv2
        _SMART_REF_FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    except Exception: pass
    return _SMART_REF_FACE_CASCADE

# 5. Scene Analysis & Smooth Flow Generation
def _build_smart_reframe_filter(input_video_path: str, aspect_ratio: str, scene_frame_skip: int = 30, scene_downscale: int = 0) -> Tuple[str, Dict[str, Any]]:
    target_ratio = _aspect_ratio_to_float(aspect_ratio)
    in_w, in_h = _probe_video_dimensions(input_video_path)
    if in_w == 0 or in_h == 0:
        return "crop=iw/2:ih:iw/4:0", {"scene_count": 0}

    out_w, out_h = _derive_output_dimensions(in_w, in_h, target_ratio)
    avg_x, avg_y = in_w / 2, in_h / 2

    try:
        import numpy as np
        import cv2

        cap = cv2.VideoCapture(input_video_path)
        if cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or fps * 30)
            
            model = _smart_ref_get_yolo_model()
            cascade = _smart_ref_get_face_cascade()
            
            frame_centers = []
            frame_idx = 0
            # Sample up to 15 frames distributed evenly in the video
            step = max(1, total_frames // 15)
            
            while cap.isOpened() and frame_idx < total_frames and len(frame_centers) < 15:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret: break
                
                center_x, center_y = in_w / 2, in_h / 2
                found = False
                
                # Use YOLO primary detection (find subject)
                if model:
                    try:
                        results = model(frame, verbose=False)
                        for r in results:
                            boxes = r.boxes
                            if boxes is not None and len(boxes) > 0:
                                b = boxes[0].xywh[0].cpu().numpy()
                                center_x, center_y = float(b[0]), float(b[1])
                                found = True
                                break
                    except Exception: pass
                
                # Fallback to Haar cascade if YOLO failed
                if not found and cascade is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = cascade.detectMultiScale(gray, 1.1, 4)
                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        center_x, center_y = x + w/2.0, y + h/2.0
                        found = True
                        
                if found:
                    frame_centers.append((center_x, center_y))
                    
                frame_idx += step

            cap.release()
            
            if len(frame_centers) > 0:
                avg_x = float(np.mean([c[0] for c in frame_centers]))
                avg_y = float(np.mean([c[1] for c in frame_centers]))
    except Exception as e:
        print(f"Hotspot failure: {e}")

    # Ensure crop boundaries do not exceed the video resolution bounds
    cx = max(0, min(in_w - out_w, int(avg_x - out_w / 2)))
    cy = max(0, min(in_h - out_h, int(avg_y - out_h / 2)))
    
    return f"crop={out_w}:{out_h}:{cx}:{cy}", {"scene_count": 1, "avg_x": avg_x, "avg_y": avg_y}
