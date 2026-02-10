import os
import subprocess

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
    font_name="Verdana",
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
    
    # Mapping alignment:
    # Top: 6 or 10? ASS: 8 = Top Center. 2 = Bottom Center. 5 = Middle Center.
    
    # Position mapping (Numpad)
    ass_alignment = 2 # Default Bottom Center
    align_lower = str(alignment).lower()
    if align_lower == 'top': 
        ass_alignment = 6 # 6 is Top-Center in libass (legacy mode usually 6, standard is 8. Let's force 2 (bottom) 10 (center) ? No. 
        # Actually libass follows SSA/ASS V4+.
        # 1=Left, 2=Center, 3=Right (Subtitles Filter treats these as "Bottom")
        # 5=Top-Left, 6=Top-Center, 7=Top-Right ??
        # 9=Mid-Left, 10=Mid-Center, 11=Mid-Right ??
        # Standard: 2=Bottom, 6=Top, 10=Middle
        ass_alignment = 6
    elif align_lower == 'middle': 
        ass_alignment = 10
    elif align_lower == 'bottom': 
        ass_alignment = 2

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
    
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-vf', f"subtitles='{safe_srt_path}':force_style='{style_string}'",
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
