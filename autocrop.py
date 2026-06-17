import os
import time
import cv2
from tqdm import tqdm

# --- Constants ---
ASPECT_RATIO = 9 / 16

# Lazy-loaded models
_yolo_model = None
_mp_face_detection = None
_face_detector = None
_face_detector_unavailable = False

def get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        print("🤖 Cargando modelo YOLOv8 para rastreo inteligente...")
        from ultralytics import YOLO
        import sys
        
        # Suppress YOLO output initially
        original_stdout = sys.stdout
        with open(os.devnull, 'w') as f:
            sys.stdout = f
            try:
                # YOLOv8 nano is very fast and sufficient for tracking bodies
                _yolo_model = YOLO('yolov8n.pt')
            finally:
                sys.stdout = original_stdout
    return _yolo_model

def get_face_detector():
    global _mp_face_detection, _face_detector, _face_detector_unavailable
    if _face_detector_unavailable:
        return None
    if _face_detector is None:
        try:
            import mediapipe as mp
            _mp_face_detection = mp.solutions.face_detection
            _face_detector = _mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        except Exception as exc:
            _face_detector_unavailable = True
            print(f"⚠️ MediaPipe face detector unavailable; falling back to YOLO/general crop: {exc}")
            return None
    return _face_detector

class _DummyTime:
    def __init__(self, frame, fps):
        self._f = frame
        self._fps = max(fps, 1e-6)
    def get_frames(self): return self._f
    def get_seconds(self): return self._f / self._fps

def detect_scenes(video_path, downscale=0, frame_skip=0):
    """Detect scene boundaries using PySceneDetect (Modern API)."""
    scene_list = []
    fps = 30.0
    total_frames = 1
    
    import cv2
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
        cap.release()

    try:
        try:
            from scenedetect import detect_scenes as sd_detect_scenes
        except ImportError:
            from scenedetect import detect as sd_detect_scenes
        try:
            from scenedetect import ContentDetector
        except ImportError:
            from scenedetect.detectors import ContentDetector
        
        print("🎬 Detectando escenas para un mejor recorte (PySceneDetect)...")
        # detect_scenes is the high-level API in scenedetect 0.6.0+
        scene_list = sd_detect_scenes(video_path, ContentDetector(), show_progress=False)
        
    except Exception as e:
        print(f"❌ Error in PySceneDetect: {str(e)}")
        
    if not scene_list:
        # Fallback: treat the entire video as one scene
        scene_list = [(_DummyTime(0, fps), _DummyTime(total_frames, fps))]
        
    return scene_list, fps

class SmoothedCameraman:
    def __init__(self, output_width, output_height, video_width, video_height):
        self.output_width = output_width
        self.output_height = output_height
        self.video_width = video_width
        self.video_height = video_height
        
        self.current_center_x = video_width / 2
        self.target_center_x = video_width / 2
        
        self.crop_height = video_height
        self.crop_width = int(self.crop_height * ASPECT_RATIO)
        if self.crop_width > video_width:
             self.crop_width = video_width
             self.crop_height = int(self.crop_width / ASPECT_RATIO)
             
        self.safe_zone_radius = self.crop_width * 0.25

    def update_target(self, box):
        if box:
            x, y, w, h = box
            self.target_center_x = x + w / 2
    
    # Exponential smoothing tuned for a "tripod" feel: speeds are relative to
    # the video width so panning behaves the same at 720p and 4K.
    SMOOTHING_FACTOR = 0.10   # fraction of remaining distance covered per frame
    MAX_STEP_RATIO = 0.015    # max pan speed: 1.5% of width per frame
    MIN_STEP_RATIO = 0.002    # avoids an asymptotic crawl near the target

    def get_crop_box(self, force_snap=False):
        if force_snap:
            self.current_center_x = self.target_center_x
        else:
            diff = self.target_center_x - self.current_center_x
            if abs(diff) > self.safe_zone_radius:
                step = diff * self.SMOOTHING_FACTOR
                max_step = self.video_width * self.MAX_STEP_RATIO
                min_step = self.video_width * self.MIN_STEP_RATIO
                if abs(step) > max_step:
                    step = max_step if step > 0 else -max_step
                elif abs(step) < min_step:
                    step = min_step if step > 0 else -min_step
                if abs(step) >= abs(diff):
                    self.current_center_x = self.target_center_x
                else:
                    self.current_center_x += step
                
        half_crop = self.crop_width / 2
        
        if self.current_center_x - half_crop < 0:
            self.current_center_x = half_crop
        if self.current_center_x + half_crop > self.video_width:
            self.current_center_x = self.video_width - half_crop
            
        x1 = int(self.current_center_x - half_crop)
        x2 = int(self.current_center_x + half_crop)
        
        x1 = max(0, x1)
        x2 = min(self.video_width, x2)
        
        y1 = 0
        y2 = self.video_height
        
        return x1, y1, x2, y2

class SpeakerTracker:
    def __init__(self, stabilization_frames=15, cooldown_frames=30):
        self.active_speaker_id = None
        self.speaker_scores = {}
        self.last_seen = {}
        self.locked_counter = 0
        self.stabilization_threshold = stabilization_frames
        self.switch_cooldown = cooldown_frames
        self.last_switch_frame = -1000
        
        self.next_id = 0
        self.known_faces = []

    def get_target(self, face_candidates, frame_number, width):
        current_candidates = []
        for face in face_candidates:
            x, y, w, h = face['box']
            center_x = x + w / 2
            
            best_match_id = -1
            min_dist = width * 0.15
            
            for kf in self.known_faces:
                if frame_number - kf['last_frame'] > 30:
                    continue
                    
                dist = abs(center_x - kf['center'])
                if dist < min_dist:
                    min_dist = dist
                    best_match_id = kf['id']
            
            if best_match_id == -1:
                best_match_id = self.next_id
                self.next_id += 1
            
            self.known_faces = [kf for kf in self.known_faces if kf['id'] != best_match_id]
            self.known_faces.append({'id': best_match_id, 'center': center_x, 'last_frame': frame_number})
            
            current_candidates.append({
                'id': best_match_id,
                'box': face['box'],
                'score': face['score']
            })

        for pid in list(self.speaker_scores.keys()):
             self.speaker_scores[pid] *= 0.85
             if self.speaker_scores[pid] < 0.1:
                 del self.speaker_scores[pid]

        for cand in current_candidates:
            pid = cand['id']
            raw_score = cand['score'] / (width * width * 0.05)
            self.speaker_scores[pid] = self.speaker_scores.get(pid, 0) + raw_score

        if not current_candidates:
            return None 
            
        best_candidate = None
        max_score = -1
        
        for cand in current_candidates:
            pid = cand['id']
            total_score = self.speaker_scores.get(pid, 0)
            if pid == self.active_speaker_id:
                total_score *= 3.0
                
            if total_score > max_score:
                max_score = total_score
                best_candidate = cand

        if best_candidate:
            target_id = best_candidate['id']
            if target_id == self.active_speaker_id:
                self.locked_counter += 1
                return best_candidate['box']
            
            if frame_number - self.last_switch_frame < self.switch_cooldown:
                old_cand = next((c for c in current_candidates if c['id'] == self.active_speaker_id), None)
                if old_cand:
                    return old_cand['box']
            
            self.active_speaker_id = target_id
            self.last_switch_frame = frame_number
            self.locked_counter = 0
            return best_candidate['box']
            
        return None

# Detection runs on a downscaled copy: MediaPipe/YOLO accuracy is unaffected at
# this size, but color conversion + preprocessing on 1080p/4K frames dominates
# per-frame cost otherwise.
DETECTION_MAX_WIDTH = 640

def _detection_frame(frame):
    height, width = frame.shape[:2]
    if width <= DETECTION_MAX_WIDTH:
        return frame
    scale = DETECTION_MAX_WIDTH / width
    return cv2.resize(
        frame,
        (DETECTION_MAX_WIDTH, max(2, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )

def detect_face_candidates(frame):
    # Boxes are computed from MediaPipe's relative coords scaled by the
    # ORIGINAL frame size, so downscaling the detection input is transparent.
    height, width = frame.shape[:2]
    rgb_frame = cv2.cvtColor(_detection_frame(frame), cv2.COLOR_BGR2RGB)
    detector = get_face_detector()
    if detector is None:
        return []
    results = detector.process(rgb_frame)
    
    candidates = []
    if not results.detections:
        return []
        
    for detection in results.detections:
        bboxC = detection.location_data.relative_bounding_box
        x = int(bboxC.xmin * width)
        y = int(bboxC.ymin * height)
        w = int(bboxC.width * width)
        h = int(bboxC.height * height)
        
        candidates.append({
            'box': [x, y, w, h],
            'score': w * h
        })
    return candidates

def detect_person_yolo(frame):
    model = get_yolo_model()
    # imgsz=480 caps YOLO's internal letterbox size; boxes come back in
    # original-frame coordinates regardless.
    results = model(frame, verbose=False, classes=[0], imgsz=480)

    if not results:
        return None
        
    best_box = None
    max_area = 0
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
            w = x2 - x1
            h = y2 - y1
            area = w * h
            
            if area > max_area:
                max_area = area
                face_h = int(h * 0.4)
                best_box = [x1, y1, w, face_h]
                
    return best_box

def create_general_frame(frame, output_width, output_height):
    """
    'General Shot' frame:
    - Background: blurred cover-crop of the original.
    - Foreground: original scaled to fit, centered.
    The background is blurred at quarter resolution and upscaled: visually
    identical to a large-kernel blur at full size, ~10x faster per frame.
    """
    orig_h, orig_w = frame.shape[:2]

    small_w = max(2, output_width // 4)
    small_h = max(2, output_height // 4)
    bg_scale = max(small_w / orig_w, small_h / orig_h)
    bg_w = max(small_w, int(orig_w * bg_scale))
    bg_h = max(small_h, int(orig_h * bg_scale))
    bg_small = cv2.resize(frame, (bg_w, bg_h), interpolation=cv2.INTER_AREA)

    start_x = max(0, (bg_w - small_w) // 2)
    start_y = max(0, (bg_h - small_h) // 2)
    bg_small = bg_small[start_y:start_y + small_h, start_x:start_x + small_w]
    if bg_small.shape[0] != small_h or bg_small.shape[1] != small_w:
        bg_small = cv2.resize(bg_small, (small_w, small_h))

    bg_small = cv2.GaussianBlur(bg_small, (13, 13), 0)
    background = cv2.resize(bg_small, (output_width, output_height), interpolation=cv2.INTER_LINEAR)

    scale = min(output_width / orig_w, output_height / orig_h)
    fg_w = max(2, int(orig_w * scale))
    fg_h = max(2, int(orig_h * scale))
    foreground = cv2.resize(frame, (fg_w, fg_h), interpolation=cv2.INTER_AREA)

    x_offset = (output_width - fg_w) // 2
    y_offset = (output_height - fg_h) // 2
    background[y_offset:y_offset + fg_h, x_offset:x_offset + fg_w] = foreground

    return background

def _count_persons_yolo(frame, min_confidence=0.4):
    model = get_yolo_model()
    results = model(frame, verbose=False, classes=[0], imgsz=480)
    count = 0
    for result in results:
        for box in result.boxes:
            if float(box.conf[0]) >= min_confidence:
                count += 1
    return count

def analyze_scenes_strategy(video_path, scenes):
    cap = cv2.VideoCapture(video_path)
    strategies = []

    if not cap.isOpened():
        return ['TRACK'] * len(scenes)

    for start, end in tqdm(scenes, desc="   Analyzing Scenes"):
        frames_to_check = [
            start.get_frames() + 5,
            int((start.get_frames() + end.get_frames()) / 2),
            end.get_frames() - 5
        ]

        face_counts = []
        middle_frame = None
        for f_idx in frames_to_check:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret: continue

            candidates = detect_face_candidates(frame)
            face_counts.append(len(candidates))
            if middle_frame is None or f_idx == frames_to_check[1]:
                middle_frame = frame

        if not face_counts:
            avg_faces = 0
        else:
            avg_faces = sum(face_counts) / len(face_counts)

        if avg_faces > 1.2:
            strategies.append('GENERAL')
        elif avg_faces < 0.5:
            # Face detection misses profiles/backlit subjects. If YOLO sees
            # exactly one person, TRACK (with its YOLO fallback) frames better
            # than the blurred GENERAL layout.
            person_count = _count_persons_yolo(middle_frame) if middle_frame is not None else 0
            strategies.append('TRACK' if person_count == 1 else 'GENERAL')
        else:
            strategies.append('TRACK')

    cap.release()
    return strategies

def is_variable_frame_rate(video_path):
    import subprocess
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=r_frame_rate,avg_frame_rate',
             '-of', 'csv=p=0', video_path],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            return False
            
        parts = result.stdout.strip().split(',')
        if len(parts) < 2:
            return False
            
        def parse_rate(s):
            nums = s.strip().split('/')
            if len(nums) == 2 and int(nums[1]) != 0:
                return int(nums[0]) / int(nums[1])
            return float(nums[0])
            
        r_fps = parse_rate(parts[0])
        avg_fps = parse_rate(parts[1])
        return abs(r_fps - avg_fps) > 0.5
    except (FileNotFoundError, ValueError, ZeroDivisionError):
        return False

def normalize_to_cfr(video_path, output_path):
    import subprocess
    print("  Normalizing variable frame rate to constant frame rate...")
    command = [
        'ffmpeg', '-y', '-i', video_path,
        '-vsync', 'cfr', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        '-c:a', 'copy', output_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Warning: VFR normalization failed, proceeding with original file.")
        print("  Stderr:", e.stderr.decode())
        return False
