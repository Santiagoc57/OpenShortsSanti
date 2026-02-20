import os
import re
import subprocess
import unicodedata
import shutil
from functools import lru_cache

_FONT_ALIAS_MAP = {
    # Keep canonical family names to avoid libass fallback on style/fullname mismatches.
    # Prefer styled fullnames first (when available), then family fallback.
    "montserrat": ("Montserrat Bold", "Montserrat SemiBold", "Montserrat"),
    "montserrat bold": ("Montserrat Bold", "Montserrat"),
    "montserrat semibold": ("Montserrat SemiBold", "Montserrat"),
    "montserrat extrabold": ("Montserrat ExtraBold", "Montserrat Black", "Montserrat"),
    "montserrat black": ("Montserrat Black", "Montserrat ExtraBold", "Montserrat"),
    "anton": ("Anton",),
    "archivo black": ("Archivo Black",),
    "bebas neue": ("Bebas Neue",),
    "bebas": ("Bebas Neue",),
    "oswald": ("Oswald Bold", "Oswald SemiBold", "Oswald"),
    "oswald bold": ("Oswald Bold", "Oswald"),
    "oswald semibold": ("Oswald SemiBold", "Oswald"),
    "teko": ("Teko Bold", "Teko SemiBold", "Teko"),
    "teko light": ("Teko", "Teko Regular"),
    "teko bold": ("Teko Bold", "Teko"),
    "teko semibold": ("Teko SemiBold", "Teko"),
    "arial": "Arial",
    "arial black": "Anton",
    "verdana": "Verdana",
    "georgia": "Georgia",
    "impact": "Anton",
}

def _normalize_font_key(value: str) -> str:
    text = str(value or "").strip().lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"\s+", " ", text)
    return text

def _bundled_fonts_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "dashboard", "public", "fonts")

def _register_bundled_fonts_for_fontconfig() -> None:
    """
    Make bundled fonts discoverable in headless environments (e.g. Colab).
    Copying to user font dir avoids relying only on subtitles/ass fontsdir behavior.
    """
    src_dir = _bundled_fonts_dir()
    if not os.path.isdir(src_dir):
        return
    dst_dir = os.path.expanduser("~/.local/share/fonts/openshorts")
    try:
        os.makedirs(dst_dir, exist_ok=True)
    except Exception:
        return

    copied_any = False
    for name in os.listdir(src_dir):
        if not name.lower().endswith((".ttf", ".otf")):
            continue
        src = os.path.join(src_dir, name)
        if not os.path.isfile(src):
            continue
        dst = os.path.join(dst_dir, name)
        try:
            src_size = os.path.getsize(src)
            dst_size = os.path.getsize(dst) if os.path.isfile(dst) else -1
            if src_size != dst_size:
                shutil.copy2(src, dst)
                copied_any = True
        except Exception:
            continue

    try:
        # Run quietly; if unavailable, caller still has fontsdir fallback in ffmpeg filter.
        if copied_any:
            subprocess.run(["fc-cache", "-f", dst_dir], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except Exception:
        pass

@lru_cache(maxsize=1)
def _load_available_font_families():
    _register_bundled_fonts_for_fontconfig()
    families = set()
    try:
        proc = subprocess.run(
            ["fc-list", "-f", "%{family}\n%{fullname}\n"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False
        )
        for raw in (proc.stdout or "").splitlines():
            for part in raw.split(","):
                name = str(part or "").strip()
                if name:
                    families.add(_normalize_font_key(name))
    except Exception:
        pass
    # Include bundled families shipped with the project for deterministic exports.
    fonts_dir = _bundled_fonts_dir()
    bundled = {
        "Montserrat-Bold.ttf": ["montserrat", "montserrat regular", "montserrat semibold", "montserrat bold", "montserrat extrabold", "montserrat black"],
        "Anton-Regular.ttf": ["anton", "anton regular"],
        "ArchivoBlack-Regular.ttf": ["archivo black", "archivo black regular"],
        "BebasNeue-Regular.ttf": ["bebas neue", "bebas neue regular"],
        "Oswald-Variable.ttf": ["oswald", "oswald regular", "oswald semibold", "oswald bold"],
        "Teko-Variable.ttf": ["teko", "teko regular", "teko semibold", "teko bold"],
    }
    for filename, family_names in bundled.items():
        if os.path.isfile(os.path.join(fonts_dir, filename)):
            for family in family_names:
                families.add(_normalize_font_key(family))
    return families

def _sanitize_font_name(font_name: str, fallback: str = "Anton") -> str:
    requested = str(font_name or "").strip()
    available = _load_available_font_families()

    def pick_available(name: str) -> str:
        candidate_name = str(name or "").strip()
        if not candidate_name:
            return ""
        key = _normalize_font_key(candidate_name)
        mapped = _FONT_ALIAS_MAP.get(key, candidate_name)
        mapped_candidates = mapped if isinstance(mapped, (tuple, list)) else (mapped,)
        for mapped_name in mapped_candidates:
            mapped_key = _normalize_font_key(mapped_name)
            if mapped_key in available:
                return str(mapped_name)
        return ""

    if not requested:
        # Never return an unavailable font name; that can trigger decorative system fallback in libass.
        for pref in [fallback, "Anton", "Archivo Black", "Bebas Neue", "Oswald", "Teko", "Montserrat", "Arial", "Verdana", "DejaVu Sans", "Noto Sans"]:
            chosen = pick_available(pref)
            if chosen:
                return chosen
        return "Arial"

    chosen_requested = pick_available(requested)
    if chosen_requested:
        return chosen_requested

    # Stable fallback chain to avoid random decorative system fonts in headless envs.
    for pref in [fallback, "Anton", "Archivo Black", "Bebas Neue", "Oswald", "Teko", "Montserrat", "Arial", "Verdana", "DejaVu Sans", "Noto Sans"]:
        chosen = pick_available(pref)
        if chosen:
            return chosen
    return "Arial"

def generate_srt(transcript, clip_start, clip_end, output_path, max_chars=20, max_duration=2.0):
    """
    Generates an SRT file from the transcript for a specific time range.
    Groups words into short lines suitable for vertical video.
    """
    
    words = []
    # 1. Extract and flatten words within range
    for segment in transcript.get('segments', []):
        for word_info in segment.get('words', []):
            # Check overlap
            if word_info['end'] > clip_start and word_info['start'] < clip_end:
                words.append(word_info)
    
    if not words:
        return False

    srt_content = ""
    index = 1
    
    current_block = []
    block_start = None
    
    for i, word in enumerate(words):
        # Adjust times relative to clip
        start = max(0, word['start'] - clip_start)
        end = max(0, word['end'] - clip_start)
        
        # Clip to video duration logic handled by ffmpeg usually, but good to be safe
        
        if not current_block:
            current_block.append(word)
            block_start = start
        else:
            # Decide whether to close block
            current_text_len = sum(len(w['word']) + 1 for w in current_block)
            duration = end - block_start
            
            if current_text_len + len(word['word']) > max_chars or duration > max_duration:
                # Finalize current block
                # End time of block is start of this word (gap) or end of last word?
                # Usually end of last word.
                block_end = current_block[-1]['end'] - clip_start
                
                text = " ".join([w['word'] for w in current_block]).strip()
                srt_content += format_srt_block(index, block_start, block_end, text)
                index += 1
                
                current_block = [word]
                block_start = start
            else:
                current_block.append(word)
    
    # Final block
    if current_block:
        block_end = current_block[-1]['end'] - clip_start
        text = " ".join([w['word'] for w in current_block]).strip()
        srt_content += format_srt_block(index, block_start, block_end, text)
        
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(srt_content)
        
    return True

def format_srt_block(index, start, end, text):
    def format_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
        
    return f"{index}\n{format_time(start)} --> {format_time(end)}\n{text}\n\n"

def _parse_srt_timestamp_to_seconds(ts: str) -> float:
    raw = str(ts or "").strip().replace(",", ".")
    parts = raw.split(":")
    if len(parts) != 3:
        return 0.0
    try:
        hh = float(parts[0])
        mm = float(parts[1])
        ss = float(parts[2])
        return max(0.0, hh * 3600 + mm * 60 + ss)
    except Exception:
        return 0.0

def _format_ass_timestamp(seconds: float) -> str:
    total = max(0.0, float(seconds or 0.0))
    hh = int(total // 3600)
    mm = int((total % 3600) // 60)
    ss = int(total % 60)
    cs = int(round((total - int(total)) * 100))
    if cs >= 100:
        cs = 0
        ss += 1
        if ss >= 60:
            ss = 0
            mm += 1
            if mm >= 60:
                mm = 0
                hh += 1
    return f"{hh}:{mm:02d}:{ss:02d}.{cs:02d}"

def _escape_ass_text(text: str) -> str:
    return str(text or "").replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")

def _parse_srt_entries(srt_text: str):
    entries = []
    blocks = re.split(r"\n\s*\n", str(srt_text or "").strip())
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        if re.match(r"^\d+$", lines[0]):
            timeline = lines[1] if len(lines) > 1 else ""
            text_lines = lines[2:]
        else:
            timeline = lines[0]
            text_lines = lines[1:]

        if "-->" not in timeline:
            continue
        start_raw, end_raw = [part.strip() for part in timeline.split("-->", 1)]
        start_s = _parse_srt_timestamp_to_seconds(start_raw)
        end_s = _parse_srt_timestamp_to_seconds(end_raw)
        text = "\n".join(text_lines).strip()
        if end_s <= start_s or not text:
            continue
        entries.append({"start": start_s, "end": end_s, "text": text})
    return entries

def _ass_alignment_from_position(alignment) -> int:
    # We now strictly use ASS alignment 5 (middle-center) for all positions,
    # and achieve vertical placement matching the CSS by overriding the Y coordinate.
    # This exactly mimics `transform: translate(-50%, -50%)` in the UI.
    return 5

def _normalize_text(value: str) -> str:
    text = str(value or "").lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text

def _pick_emotion_color(text: str, fallback: str = "#39FF14") -> str:
    normalized = _normalize_text(text)
    rules = [
        ("#FF4D4D", ["criminal", "ataque", "guerra", "odio", "corrupcion", "rabia", "violencia"]),
        ("#39FF14", ["dinero", "plata", "trading", "mercado", "finanzas", "bitcoin", "crypto", "exito"]),
        ("#FFC400", ["alerta", "riesgo", "peligro", "crisis", "grave", "urgente"]),
        ("#00E5FF", ["tip", "clave", "estrategia", "tutorial", "paso", "metodo"]),
        ("#B266FF", ["mindset", "increible", "wow", "sorpresa", "impacto"])
    ]
    for color, keywords in rules:
        if any(kw in normalized for kw in keywords):
            return color
    if "!" in str(text or ""):
        return "#FFC400"
    if "?" in str(text or ""):
        return "#00E5FF"
    return fallback

def _entry_anchor_for_alignment(ass_alignment: int):
    # Deprecated/unused mostly, but preserved for signature
    return 540, 1000, 972

def _clamp_percent(value: float) -> float:
    try:
        raw = float(value)
    except Exception:
        raw = 0.0
    return max(-100.0, min(100.0, raw))

def _caption_anchor_point(alignment_str: str, offset_x: float = 0.0, offset_y: float = 0.0):
    # Match the CSS percentages exactly:
    # top: 20%, middle: 50%, bottom: 80%
    align_lower = str(alignment_str).lower()
    if align_lower == "top":
        base_y_percent = 20.0
    elif align_lower == "middle":
        base_y_percent = 50.0
    else:
        base_y_percent = 80.0

    # UI offsets are percentages of the container. 
    # CSS: left: calc(50% + {offset_x}%)
    #      top: calc({base_y}% + {offset_y}%)
    final_x_percent = 50.0 + _clamp_percent(offset_x)
    final_y_percent = base_y_percent + _clamp_percent(offset_y)

    # Convert to 1080x1920 canvas
    x = int(round((final_x_percent / 100.0) * 1080.0))
    y = int(round((final_y_percent / 100.0) * 1920.0))
    
    # Clamp to reasonable bounds so it doesn't disappear completely off-screen
    x = max(80, min(1000, x))
    y = max(80, min(1840, y))
    return x, y

def _entry_anchor_for_alignment_with_offset(alignment_str: str, offset_x: float = 0.0, offset_y: float = 0.0):
    x, y = _caption_anchor_point(alignment_str, offset_x=offset_x, offset_y=offset_y)
    return x, y + 24, y

def generate_styled_ass_from_srt(
    srt_text: str,
    output_path: str,
    alignment="bottom",
    font_size: int = 40,
    font_name: str = "Montserrat",
    font_color: str = "#FFFFFF",
    stroke_color: str = "#000000",
    stroke_width: int = 4,
    bold: bool = True,
    box_color: str = "#000000",
    box_opacity: int = 0,
    offset_x: float = 0.0,
    offset_y: float = 0.0
) -> bool:
    entries = _parse_srt_entries(srt_text)
    if not entries:
        return False

    ass_alignment = _ass_alignment_from_position(alignment)
    # The frontend uses a narrow container (~420px) but the backend renders at 1080px.
    # To match the relative visual size in the React UI where font looks larger, we scale by 1.5.
    final_fontsize = max(10, int(float(font_size) * 1.5))
    safe_font_name = _sanitize_font_name(font_name)
    box_alpha = int(255 * (1 - max(0, min(100, int(box_opacity))) / 100))
    primary = _hex_to_ass_color(font_color, alpha=0)
    outline = _hex_to_ass_color(stroke_color, alpha=0)
    back = _hex_to_ass_color(box_color, alpha=box_alpha)
    border_style = 3 if int(box_opacity) > 0 else 1
    bold_flag = -1 if bool(bold) else 0

    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{primary},{primary},{outline},{back},{bold_flag},0,0,0,100,100,0,0,{border_style},{stroke_width},0,{alignment},30,30,38,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""".format(
        font_name=safe_font_name,
        font_size=final_fontsize,
        primary=primary,
        outline=outline,
        back=back,
        bold_flag=bold_flag,
        border_style=border_style,
        stroke_width=max(0, int(stroke_width)),
        alignment=ass_alignment
    )

    anchor_x, anchor_y = _caption_anchor_point(alignment, offset_x=offset_x, offset_y=offset_y)
    dialogue_lines = []
    for entry in entries:
        line_text = _escape_ass_text(str(entry.get("text", "")).strip()).replace("\n", r"\N")
        if not line_text:
            continue
        dialogue_lines.append(
            f"Dialogue: 0,{_format_ass_timestamp(entry['start'])},{_format_ass_timestamp(entry['end'])},Default,,0,0,0,,{{\\pos({anchor_x},{anchor_y})}}{line_text}"
        )

    if not dialogue_lines:
        return False

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n".join(dialogue_lines))
        f.write("\n")
    return True

def generate_karaoke_ass_from_srt(
    srt_text: str,
    output_path: str,
    alignment="bottom",
    font_size: int = 40,
    font_name: str = "Montserrat",
    font_color: str = "#FFFFFF",
    active_word_color: str = "#39FF14",
    stroke_color: str = "#000000",
    stroke_width: int = 3,
    bold: bool = True,
    box_color: str = "#000000",
    box_opacity: int = 0,
    pop_scale: int = 116,
    offset_x: float = 0.0,
    offset_y: float = 0.0
) -> bool:
    entries = _parse_srt_entries(srt_text)
    if not entries:
        return False

    ass_alignment = _ass_alignment_from_position(alignment)
    # Equivalent 1.5x scaling for karaoke to match React UI.
    final_fontsize = max(12, int(float(font_size) * 1.5))
    safe_font_name = _sanitize_font_name(font_name)
    box_alpha = int(255 * (1 - max(0, min(100, int(box_opacity))) / 100))
    primary = _hex_to_ass_color(font_color, alpha=0)
    use_auto_emotion = str(active_word_color or "").strip().lower() == "auto"
    base_active_hex = "#39FF14" if use_auto_emotion else str(active_word_color or "#39FF14")
    secondary = _hex_to_ass_color(base_active_hex, alpha=0)
    outline = _hex_to_ass_color(stroke_color, alpha=0)
    back = _hex_to_ass_color(box_color, alpha=box_alpha)
    border_style = 3 if int(box_opacity) > 0 else 1
    bold_flag = -1 if bool(bold) else 0
    safe_pop_scale = max(105, min(150, int(pop_scale)))
    # Slight extra letter spacing helps condensed display fonts (Anton/Bebas) breathe in karaoke.
    style_spacing = 1.2 if final_fontsize >= 34 else 0.8
    # Extra hard-space gap to avoid words visually "touching" when active word scales up.
    word_gap_token = r"\h\h" if safe_pop_scale >= 112 else r"\h"

    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{primary},{secondary},{outline},{back},{bold_flag},0,0,0,100,100,{style_spacing},0,{border_style},{stroke_width},0,{alignment},30,30,38,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""".format(
        font_name=safe_font_name,
        font_size=final_fontsize,
        primary=primary,
        secondary=secondary,
        outline=outline,
        back=back,
        bold_flag=bold_flag,
        style_spacing=f"{style_spacing:.2f}",
        border_style=border_style,
        stroke_width=max(0, int(stroke_width)),
        alignment=ass_alignment
    )

    dialogue_lines = []
    anchor_x, anchor_from_y, anchor_to_y = _entry_anchor_for_alignment_with_offset(
        alignment,
        offset_x=offset_x,
        offset_y=offset_y
    )
    for entry in entries:
        text = str(entry["text"]).strip()
        words = [w for w in re.split(r"\s+", text) if w]
        if not words:
            continue
        entry_active_hex = _pick_emotion_color(text, fallback=base_active_hex) if use_auto_emotion else base_active_hex
        entry_secondary = _hex_to_ass_color(entry_active_hex, alpha=0)
        duration_cs = max(1, int(round((entry["end"] - entry["start"]) * 100)))
        per_word = max(1, duration_cs // len(words))
        remainder = max(0, duration_cs - (per_word * len(words)))

        elapsed_cs = 0
        for idx, word in enumerate(words):
            k_value = per_word + (1 if idx < remainder else 0)
            start_cs = elapsed_cs
            end_cs = min(duration_cs, elapsed_cs + k_value)
            elapsed_cs = end_cs
            seg_start = entry["start"] + (start_cs / 100.0)
            seg_end = entry["start"] + (end_cs / 100.0)
            if seg_end <= seg_start:
                seg_end = seg_start + 0.03

            safe_word = _escape_ass_text(word)
            line_tokens = []
            for widx, w in enumerate(words):
                safe_w = _escape_ass_text(w)
                if widx == idx:
                    line_tokens.append(
                        f"{{\\c{entry_secondary}\\fscx{safe_pop_scale}\\fscy{safe_pop_scale}\\b1}}{safe_w}{{\\r}}"
                    )
                else:
                    line_tokens.append(safe_w)
            line_text = word_gap_token.join(line_tokens)
            intro_tag = f"{{\\fad(45,65)\\move({anchor_x},{anchor_from_y},{anchor_x},{anchor_to_y},0,130)}}"

            dialogue_lines.append(
                f"Dialogue: 0,{_format_ass_timestamp(seg_start)},{_format_ass_timestamp(seg_end)},Default,,0,0,0,,{intro_tag}{line_text}"
            )

    if not dialogue_lines:
        return False

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n".join(dialogue_lines))
        f.write("\n")
    return True

def _hex_to_ass_color(hex_color: str, alpha: int = 0) -> str:
    color = (hex_color or "#FFFFFF").strip()
    if not color.startswith("#") or len(color) != 7:
        color = "#FFFFFF"
    r = int(color[1:3], 16)
    g = int(color[3:5], 16)
    b = int(color[5:7], 16)
    alpha = max(0, min(255, int(alpha)))
    return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"

def burn_subtitles(
    video_path,
    srt_path,
    output_path,
    alignment=2,
    fontsize=16,
    font_name="Montserrat",
    font_color="#FFFFFF",
    stroke_color="#000000",
    stroke_width=2,
    bold=True,
    box_color="#000000",
    box_opacity=60
):
    """
    Burns subtitles into the video using FFmpeg.
    Alignment: 2 (Bottom), 5 (Middle), 8 (Top)
    """
    # Convert styling options to ASS format
    # FontSize is roughly pixels for 720p? It scales.
    # For 1080x1920, fontsize 16 is tiny. 
    # Let's assume standard vertical resolution (1080w). 
    # Try a larger default or let it be scaled.
    # We will accept a 'scale' factor or just big font.
    # Default 16 in ASS is small. 
    # Let's use a reasonable default if user says "small/medium/large".
    
    ass_alignment = _ass_alignment_from_position(alignment)
    safe_font_name = _sanitize_font_name(font_name)

    # Font size logic
    # Scale: Libass uses 384x288 virtual resolution unless PlayResX/Y set.
    # The frontend sends a value like 24 (pixels).
    # In 288p land, 24 is HUGE (approx 1/12th of screen height).
    # We want it to be smaller, around 10-12 units.
    # Factor: 0.5 is safe.
    final_fontsize = int(fontsize * 0.5) 
    if final_fontsize < 8: final_fontsize = 8

    # Path handling for filter string
    try:
        # Use absolute path but replace special chars for FFmpeg filter syntax
        # : -> \: and \ -> / (forward slash is safer on windows too usually in ffmpeg filters if escaped)
        # But for standard os paths on linux/mac: /path/to/file.srt
        # FFmpeg expects: subtitles='/path/to/file.srt'
        # If there are colons (e.g. C:), they need escaping: C\:
        safe_srt_path = srt_path.replace('\\', '/').replace(':', '\\:')
    except:
        safe_srt_path = srt_path

    # Style String
    # BorderStyle=3 (Opaque Box)
    # OutlineColour is Box Background. Alpha 60 (approx 40% opacity) -> &H60000000
    # Fontname: 'Verdana' or 'Arial' are safe. 'Verdana' is slightly more "modern/web".
    # Bold=1
    box_alpha = int(255 * (1 - max(0, min(100, int(box_opacity))) / 100))
    primary = _hex_to_ass_color(font_color, alpha=0)
    outline = _hex_to_ass_color(stroke_color, alpha=0)
    back = _hex_to_ass_color(box_color, alpha=box_alpha)
    border_style = 3 if int(box_opacity) > 0 else 1
    bold_flag = 1 if bool(bold) else 0

    style_string = (
        f"Alignment={ass_alignment},Fontname={safe_font_name},Fontsize={final_fontsize},"
        f"PrimaryColour={primary},OutlineColour={outline},BackColour={back},"
        f"BorderStyle={border_style},Outline={stroke_width},Shadow=0,MarginV=25,Bold={bold_flag}"
    )

    safe_fonts_dir = None
    fonts_dir = os.path.join(os.path.dirname(__file__), "dashboard", "public", "fonts")
    if os.path.isdir(fonts_dir):
        safe_fonts_dir = fonts_dir.replace('\\', '/').replace(':', '\\:')

    filter_candidates = []
    is_ass_input = str(srt_path or "").lower().endswith(".ass")
    if is_ass_input:
        # Prefer ASS-native filter for better style fidelity (font family, box, karaoke tags, spacing).
        vf_ass = f"ass='{safe_srt_path}'"
        if safe_fonts_dir:
            vf_ass = f"{vf_ass}:fontsdir='{safe_fonts_dir}'"
        filter_candidates.append(("ass", vf_ass))

        # Fallback for environments where ffmpeg was built without `ass` filter.
        vf_sub_fallback = f"subtitles='{safe_srt_path}'"
        if safe_fonts_dir:
            vf_sub_fallback = f"{vf_sub_fallback}:fontsdir='{safe_fonts_dir}'"
        filter_candidates.append(("subtitles-fallback", vf_sub_fallback))
    else:
        vf_subtitles = f"subtitles='{safe_srt_path}':force_style='{style_string}'"
        if safe_fonts_dir:
            vf_subtitles = f"{vf_subtitles}:fontsdir='{safe_fonts_dir}'"
        filter_candidates.append(("subtitles", vf_subtitles))

    last_err = ""
    for idx, (filter_name, vf_subtitles) in enumerate(filter_candidates):
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vf', vf_subtitles,
            '-c:a', 'copy',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'fast', '-crf', '23',
            '-movflags', '+faststart',
            output_path
        ]

        print(f"🎬 Burning subtitles ({filter_name}): {' '.join(cmd)}")
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if result.returncode == 0:
            return True

        last_err = result.stderr.decode(errors="ignore")
        print(f"⚠️ FFmpeg subtitle render failed with {filter_name}: {last_err}")
        if idx < len(filter_candidates) - 1:
            print("↪️ Trying subtitle render fallback...")
            continue

    raise Exception(f"FFmpeg failed: {last_err}")

    return True
