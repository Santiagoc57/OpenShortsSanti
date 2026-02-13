import os
import re
import subprocess
import unicodedata

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
    ass_alignment = 2
    align_lower = str(alignment).lower()
    if align_lower == "top":
        ass_alignment = 6
    elif align_lower == "middle":
        ass_alignment = 10
    elif align_lower == "bottom":
        ass_alignment = 2
    return ass_alignment

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
    if ass_alignment == 6:
        return 540, 230, 205
    if ass_alignment == 10:
        return 540, 1000, 972
    return 540, 1675, 1648

def _clamp_percent(value: float) -> float:
    try:
        raw = float(value)
    except Exception:
        raw = 0.0
    return max(-100.0, min(100.0, raw))

def _caption_anchor_point(ass_alignment: int, offset_x: float = 0.0, offset_y: float = 0.0):
    base_x = 540
    if ass_alignment == 6:
        base_y = 220
    elif ass_alignment == 10:
        base_y = 960
    else:
        base_y = 1680

    # Same perceptual scale used in editor preview for manual offset.
    shift_x = int(round((_clamp_percent(offset_x) / 100.0) * 378.0))
    shift_y = int(round((_clamp_percent(offset_y) / 100.0) * 672.0))

    x = max(80, min(1000, base_x + shift_x))
    y = max(120, min(1820, base_y + shift_y))
    return x, y

def _entry_anchor_for_alignment_with_offset(ass_alignment: int, offset_x: float = 0.0, offset_y: float = 0.0):
    x, y = _caption_anchor_point(ass_alignment, offset_x=offset_x, offset_y=offset_y)
    return x, y + 24, y

def generate_styled_ass_from_srt(
    srt_text: str,
    output_path: str,
    alignment="bottom",
    font_size: int = 34,
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
    final_fontsize = max(10, int(font_size))
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
        font_name=font_name,
        font_size=final_fontsize,
        primary=primary,
        outline=outline,
        back=back,
        bold_flag=bold_flag,
        border_style=border_style,
        stroke_width=max(0, int(stroke_width)),
        alignment=ass_alignment
    )

    anchor_x, anchor_y = _caption_anchor_point(ass_alignment, offset_x=offset_x, offset_y=offset_y)
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
    font_size: int = 44,
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
    final_fontsize = max(12, int(font_size))
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

    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{primary},{secondary},{outline},{back},{bold_flag},0,0,0,100,100,0,0,{border_style},{stroke_width},0,{alignment},30,30,38,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""".format(
        font_name=font_name,
        font_size=final_fontsize,
        primary=primary,
        secondary=secondary,
        outline=outline,
        back=back,
        bold_flag=bold_flag,
        border_style=border_style,
        stroke_width=max(0, int(stroke_width)),
        alignment=ass_alignment
    )

    dialogue_lines = []
    anchor_x, anchor_from_y, anchor_to_y = _entry_anchor_for_alignment_with_offset(
        ass_alignment,
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
            line_text = " ".join(line_tokens)
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
        f"Alignment={ass_alignment},Fontname={font_name},Fontsize={final_fontsize},"
        f"PrimaryColour={primary},OutlineColour={outline},BackColour={back},"
        f"BorderStyle={border_style},Outline={stroke_width},Shadow=0,MarginV=25,Bold={bold_flag}"
    )

    safe_fonts_dir = None
    fonts_dir = os.path.join(os.path.dirname(__file__), "dashboard", "public", "fonts")
    if os.path.isdir(fonts_dir):
        safe_fonts_dir = fonts_dir.replace('\\', '/').replace(':', '\\:')

    is_ass_input = str(srt_path or "").lower().endswith(".ass")
    if is_ass_input:
        vf_subtitles = f"subtitles='{safe_srt_path}'"
    else:
        vf_subtitles = f"subtitles='{safe_srt_path}':force_style='{style_string}'"
    if safe_fonts_dir:
        vf_subtitles = f"{vf_subtitles}:fontsdir='{safe_fonts_dir}'"

    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-vf', vf_subtitles,
        '-c:a', 'copy',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'fast', '-crf', '23',
        '-movflags', '+faststart',
        output_path
    ]
    
    print(f"🎬 Burning subtitles: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    
    if result.returncode != 0:
        print(f"❌ FFmpeg Subtitle Error: {result.stderr.decode()}")
        raise Exception(f"FFmpeg failed: {result.stderr.decode()}")

    return True
