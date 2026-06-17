import os
import cv2
import json
from typing import List, Dict, Any, Optional
from ..core.config import GEMINI_API_KEY
from ..utils.text import _extract_generated_text

def extract_video_screenshots(video_path: str, output_dir: str, num_captures: int = 5) -> List[str]:
    """
    Extrae N capturas representativas del video para ser usadas como B-Roll.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    screenshots = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return screenshots

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if total_frames <= 0 or fps <= 0:
        cap.release()
        return screenshots

    step = max(1, total_frames // (num_captures + 1))
    
    for i in range(1, num_captures + 1):
        frame_idx = i * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Format: broll_capture_HHMMSSxxx.jpg
            time_sec = frame_idx / fps
            hours = int(time_sec // 3600)
            minutes = int((time_sec % 3600) // 60)
            seconds = int(time_sec % 60)
            ms = int((time_sec - int(time_sec)) * 1000)
            filename = f"broll_capture_{hours:02d}{minutes:02d}{seconds:02d}{ms:03d}.jpg"
            out_path = os.path.join(output_dir, filename)
            
            # Guardamos la imagen en calidad alta
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            screenshots.append(out_path)

    cap.release()
    return screenshots

def generate_broll_suggestions(script_lines: List[str], api_key: str = None) -> List[Dict[str, Any]]:
    """
    Dada una lista de lineas de guion o subtitulos, la IA devuelve sugerencias visuales
    indicando si debe usar Stock, Captura Real, o generar un Prompt de Midjourney.
    """
    key = api_key or GEMINI_API_KEY
    if not key or not script_lines:
        return []

    try:
        from google import genai
        client = genai.Client(api_key=key)
        
        script_text = "\n".join([f"{i+1}. {line}" for i, line in enumerate(script_lines)])
        
        prompt = (
            "Eres un Director Creativo especialista en Storytelling Visual (B-Roll) para TikTok/Reels.\n"
            "Dado el siguiente guion de un Short, genera sugerencias visuales (B-Roll) para cada línea.\n"
            "Tu respuesta debe ser un ARRAY de objetos JSON y NADA MÁS.\n\n"
            "Formato de objeto JSON:\n"
            "{\n"
            '  "line_index": [numero de linea],\n'
            '  "visual_theme": "[descripcion corta de la vibra de la imagen]",\n'
            '  "stock_keyword": "[palabras clave en INGLES para buscar en Pexels/Unsplash]",\n'
            '  "midjourney_prompt": "[Genera un Promp detallado y en INGLES optimizado para Midjourney v6/DALL-E 3. Incluye estilo, iluminacion, angulo de camara. Ej: Cinematic 4k, hyperrealistic photography, dark moody lighting...]",\n'
            '  "recommended_source": "[STOCK | MIDJOURNEY | ORIGINAL_VIDEO]"\n'
            "}\n\n"
            "Guion:\n"
            f"{script_text}"
        )
        
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt
        )
        
        raw_text = _extract_generated_text(response)
        
        # Limpiar y parsear JSON
        json_start = raw_text.find('[')
        json_end = raw_text.rfind(']') + 1
        if json_start != -1 and json_end != -1:
            json_str = raw_text[json_start:json_end]
            parsed = json.loads(json_str)
            return parsed
            
    except Exception as e:
        print(f"Error generando sugerencias B-Roll: {e}")
        
    return []
