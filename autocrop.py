import os
import time

# --- Constants ---
ASPECT_RATIO = 9 / 16

# Lazy-loaded models
_yolo_model = None
_face_cascade = None

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

def get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        import cv2
        _face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return _face_cascade

def detect_scenes(video_path, downscale=0, frame_skip=0):
    """Detect scene boundaries using PySceneDetect (Modern API)."""
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
        
        # We still need fps for some calculations, let's get it via cv2 to avoid deprecated VideoManager
        import cv2
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        return scene_list, fps
    except Exception as e:
        print(f"❌ Error in PySceneDetect: {str(e)}")
        return [], 0.0

def analyze_scene_content(video_path, scene_start_time, scene_end_time):
    """
    Analyzes the middle frame of a scene to detect people and faces.
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: No se pudo abrir el video {video_path}")
        return []

    start_frame = scene_start_time.get_frames()
    end_frame = scene_end_time.get_frames()
    middle_frame_number = int(start_frame + (end_frame - start_frame) / 2)

    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_number)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return []

    results = get_yolo_model()([frame], verbose=False)
    detected_objects = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            if box.cls[0] == 0:  # Class 0 is 'person' in YOLO coco
                x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
                person_box = [x1, y1, x2, y2]

                # Look for a face inside the detected person bounding box
                person_roi_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                faces = get_face_cascade().detectMultiScale(person_roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                face_box = None
                if len(faces) > 0:
                    fx, fy, fw, fh = faces[0]
                    face_box = [x1 + fx, y1 + fy, x1 + fx + fw, y1 + fy + fh]

                detected_objects.append({'person_box': person_box, 'face_box': face_box})

    cap.release()
    return detected_objects

def get_enclosing_box(boxes):
    if not boxes:
        return None
    min_x = min(box[0] for box in boxes)
    min_y = min(box[1] for box in boxes)
    max_x = max(box[2] for box in boxes)
    max_y = max(box[3] for box in boxes)
    return [min_x, min_y, max_x, max_y]

def decide_cropping_strategy(scene_analysis, frame_height):
    """
    Decides whether to crop (TRACK) or pad (LETTERBOX) based on detected people.
    """
    num_people = len(scene_analysis)
    if num_people == 0:
        return 'LETTERBOX', None
        
    if num_people == 1:
        # If possible, focus on the face instead of the middle of the body
        target_box = scene_analysis[0]['face_box'] or scene_analysis[0]['person_box']
        return 'TRACK', target_box
        
    person_boxes = [obj['person_box'] for obj in scene_analysis]
    group_box = get_enclosing_box(person_boxes)
    group_width = group_box[2] - group_box[0]
    
    # If the group is narrower than 9:16 target width, we can track the group
    max_width_for_crop = frame_height * ASPECT_RATIO
    if group_width < max_width_for_crop:
        return 'TRACK', group_box
    else:
        # People are too far apart, pad the video to preserve everyone
        return 'LETTERBOX', None

def calculate_crop_box(target_box, frame_width, frame_height):
    target_center_x = (target_box[0] + target_box[2]) / 2
    crop_height = frame_height
    crop_width = int(crop_height * ASPECT_RATIO)
    x1 = int(target_center_x - crop_width / 2)
    y1 = 0
    x2 = int(target_center_x + crop_width / 2)
    y2 = frame_height
    if x1 < 0:
        x1 = 0
        x2 = crop_width
    if x2 > frame_width:
        x2 = frame_width
        x1 = frame_width - crop_width
    return x1, y1, x2, y2

def analyze_video_for_autocrop(video_path, start_time_secs=0, end_time_secs=None):
    """
    Main entrypoint.
    Provides FFmpeg filter parameters for each scene in the (clipped) video.
    Returns a list of dicts:
    [{
      'start_sec': float, 
      'end_sec': float, 
      'strategy': 'TRACK' or 'LETTERBOX', 
      'crop_coords': (x, y, w, h) or None
    }]
    """
    # 1. Detect scenes
    scenes, fps = detect_scenes(video_path)
    if not scenes:
        return []

    # Get video properties
    import cv2
    cap = cv2.VideoCapture(video_path)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # 2. Analyze each scene
    plan = []
    print(f"🔍 Analizando contenido en {len(scenes)} escenas...")
    
    for start, end in scenes:
        start_sec = start.get_seconds()
        end_sec = end.get_seconds()
        
        # If we only want a subset of the video (e.g. for a clip)
        if end_time_secs and start_sec > end_time_secs:
            # We are past the clip
            break
        if start_time_secs and end_sec < start_time_secs:
            # We are before the clip starts
            continue

        analysis = analyze_scene_content(video_path, start, end)
        strategy, target_box = decide_cropping_strategy(analysis, original_height)
        
        crop_coords = None
        if strategy == 'TRACK':
            x1, y1, x2, y2 = calculate_crop_box(target_box, original_width, original_height)
            crop_coords = (x1, y1, x2 - x1, y2 - y1) # x, y, width, height
            
        plan.append({
            'start_sec': start_sec,
            'end_sec': end_sec,
            'strategy': strategy,
            'crop_coords': crop_coords
        })
        
    return plan

def is_variable_frame_rate(video_path):
    import subprocess
    """Uses ffprobe to check if the video has a variable frame rate."""
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
    """Re-muxes a VFR video to constant frame rate using FFmpeg."""
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
