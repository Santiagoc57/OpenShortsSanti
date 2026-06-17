from pydantic import BaseModel
from typing import List, Optional, Any, Dict

class RecutRequest(BaseModel):
    job_id: str
    clip_index: int
    start: float
    end: float
    aspect_ratio: Optional[str] = None
    layout_mode: Optional[str] = "single"  # single | split
    fit_mode: Optional[str] = "cover"  # cover | contain
    zoom: Optional[float] = 1.0        # contain: 0.5..2.5 | cover: 1.0..2.5
    offset_x: Optional[float] = 0.0    # -100 .. 100
    offset_y: Optional[float] = 0.0    # -100 .. 100
    split_zoom_a: Optional[float] = 1.0
    split_offset_a_x: Optional[float] = 0.0
    split_offset_a_y: Optional[float] = 0.0
    split_zoom_b: Optional[float] = 1.0
    split_offset_b_x: Optional[float] = 0.0
    split_offset_b_y: Optional[float] = 0.0
    auto_smart_reframe: Optional[bool] = False
    smart_scene_frame_skip: Optional[int] = 1
    smart_scene_downscale: Optional[int] = 0
    # Phase 3 Single-Pass Subtitle Options
    captions_on: Optional[bool] = False
    caption_position: Optional[str] = None
    caption_font_size: Optional[int] = None
    caption_font_family: Optional[str] = None
    caption_font_color: Optional[str] = None
    caption_stroke_color: Optional[str] = None
    caption_stroke_width: Optional[int] = None
    caption_bold: Optional[bool] = None
    caption_box_color: Optional[str] = None
    caption_box_opacity: Optional[int] = None
    caption_karaoke_mode: Optional[bool] = None
    caption_animation: Optional[str] = None
    caption_speaker_color_mode: Optional[bool] = None
    caption_speaker_color_palette: Optional[List[str]] = None
    caption_offset_x: Optional[float] = None
    caption_offset_y: Optional[float] = None
    srt_content: Optional[str] = None
    viral_hook_text: Optional[str] = None
    viral_hook_start: Optional[float] = None
    viral_hook_duration: Optional[float] = None
    viral_hook_font_size: Optional[int] = None
    viral_hook_font_family: Optional[str] = None
    viral_hook_font_color: Optional[str] = None
    viral_hook_stroke_color: Optional[str] = None
    viral_hook_stroke_width: Optional[int] = None
    viral_hook_bold: Optional[bool] = None
    viral_hook_box_color: Optional[str] = None
    viral_hook_box_opacity: Optional[int] = None
    viral_hook_line_spacing: Optional[int] = None

class ClipSearchRequest(BaseModel):
    job_id: str
    query: str
    limit: int = 5
    shortlist_limit: int = 5
    search_mode: str = "balanced"
    chapter_index: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    speaker: Optional[str] = None

class ClipSearchEvalCase(BaseModel):
    query: str
    expected_start: Optional[float] = None
    expected_end: Optional[float] = None
    chapter_index: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    speaker: Optional[str] = None
    min_match_score: Optional[float] = None

class ClipSearchEvalRequest(BaseModel):
    job_id: str
    cases: List[ClipSearchEvalCase]
    search_mode: str = "balanced"
    limit: int = 6
    shortlist_limit: int = 6
    expected_overlap_threshold: float = 0.35

class TranslateRequest(BaseModel):
    job_id: str
    clip_index: int
    target_language: str
    source_language: Optional[str] = None
    input_filename: Optional[str] = None

class SocialPostRequest(BaseModel):
    job_id: str
    clip_index: int
    api_key: str
    user_id: str
    platforms: List[str]
    title: Optional[str] = None
    description: Optional[str] = None
    scheduled_date: Optional[str] = None
    timezone: Optional[str] = "UTC"
