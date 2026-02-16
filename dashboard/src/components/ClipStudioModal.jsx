import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { X, FileText, Captions, Type, LayoutTemplate, Music2, Search, Sparkles, Loader2, Play, Pause, Pencil, SlidersHorizontal, ZoomOut, ZoomIn, Crosshair, Menu } from 'lucide-react';
import { apiFetch, getApiUrl } from '../config';

const CAPTION_PRESETS = [
  {
    id: 'viral_pop',
    name: 'Viral Pop',
    sample: 'THUNDER AGAIN\nLED BY',
    preview: {
      bg: 'linear-gradient(145deg, #0b1220 0%, #141b2a 55%, #090d16 100%)',
      highlightColor: '#39FF14',
      highlightWordIndex: 1
    },
    style: {
      position: 'bottom',
      fontSize: 34,
      fontFamily: 'Montserrat',
      fontColor: '#FFFFFF',
      strokeColor: '#0A0A0A',
      strokeWidth: 5,
      bold: true,
      boxColor: '#000000',
      boxOpacity: 0
    }
  },
  {
    id: 'classy',
    name: 'Elegante',
    sample: 'Aqui va tu\nsubtitulo',
    preview: {
      bg: 'linear-gradient(145deg, #1f2937 0%, #334155 100%)',
      highlightColor: '#E2E8F0',
      highlightWordIndex: 2
    },
    style: {
      position: 'bottom',
      fontSize: 24,
      fontFamily: 'Georgia',
      fontColor: '#F8FAFC',
      strokeColor: '#1F2937',
      strokeWidth: 2,
      bold: false,
      boxColor: '#000000',
      boxOpacity: 35
    }
  },
  {
    id: 'sleek',
    name: 'Nítido',
    sample: 'FRASE BREVE\nY CLARA',
    preview: {
      bg: 'linear-gradient(145deg, #111827 0%, #0b1120 100%)',
      highlightColor: '#FDE047',
      highlightWordIndex: 0
    },
    style: {
      position: 'middle',
      fontSize: 26,
      fontFamily: 'Arial Black',
      fontColor: '#FFFFFF',
      strokeColor: '#000000',
      strokeWidth: 2,
      bold: true,
      boxColor: '#000000',
      boxOpacity: 20
    }
  },
  {
    id: 'instant',
    name: 'Instantáneo',
    sample: 'Subtitulo\nrapido',
    preview: {
      bg: 'linear-gradient(145deg, #1e293b 0%, #0f172a 100%)',
      highlightColor: '#22d3ee',
      highlightWordIndex: 1
    },
    style: {
      position: 'bottom',
      fontSize: 24,
      fontFamily: 'Verdana',
      fontColor: '#FFFFFF',
      strokeColor: '#000000',
      strokeWidth: 2,
      bold: true,
      boxColor: '#000000',
      boxOpacity: 40
    }
  },
  {
    id: 'cinema',
    name: 'Cine',
    sample: 'Frase\ncinematica',
    preview: {
      bg: 'linear-gradient(145deg, #1c1917 0%, #292524 100%)',
      highlightColor: '#facc15',
      highlightWordIndex: 1
    },
    style: {
      position: 'middle',
      fontSize: 28,
      fontFamily: 'Arial',
      fontColor: '#FDE047',
      strokeColor: '#111827',
      strokeWidth: 3,
      bold: true,
      boxColor: '#111827',
      boxOpacity: 30
    }
  },
  {
    id: 'mrbeast_pro',
    name: 'MrBeast / Impact Pro',
    sample: 'NO TE LO\nPIERDAS',
    preview: {
      bg: 'linear-gradient(145deg, #0f172a 0%, #111827 60%, #1f2937 100%)',
      highlightColor: '#22c55e',
      highlightWordIndex: 3
    },
    style: {
      position: 'bottom',
      fontSize: 38,
      fontFamily: 'Montserrat',
      fontColor: '#FFFFFF',
      strokeColor: '#0A0A0A',
      strokeWidth: 6,
      bold: true,
      boxColor: '#000000',
      boxOpacity: 0
    }
  }
];

const SECTION_ITEMS = [
  { id: 'transcript', label: 'Transcripción', icon: FileText },
  { id: 'captions', label: 'Subtítulos', icon: Captions },
  { id: 'subtitle_edit', label: 'Editar subtítulos', icon: Type },
  { id: 'layout', label: 'Editar layout', icon: LayoutTemplate },
  { id: 'music', label: 'Música', icon: Music2 }
];

const SUBTITLE_EMOJIS = ['🔥', '😈', '🤯', '😂', '😱', '🚨', '✅', '💸', '🎯', '💥', '👏', '🙏'];
const ACTIVE_WORD_COLOR = '#39FF14';
const EMOJI_RULES = [
  { emoji: '💸', keywords: ['dinero', 'plata', 'inversion', 'inversión', 'trading', 'mercado', 'economia', 'economía', 'finanzas', 'bitcoin', 'crypto', 'cript', 'money', 'market', 'finance', 'profit', 'price', 'value'] },
  { emoji: '🔥', keywords: ['viral', 'brutal', 'explosivo', 'locura', 'increible', 'increíble', 'impacto', 'duro', 'fuerte', 'fire', 'hot', 'insane', 'crazy', 'epic'] },
  { emoji: '😱', keywords: ['miedo', 'peligro', 'riesgo', 'grave', 'crisis', 'caos', 'colapso', 'shock', 'fear', 'risk', 'danger', 'collapse'] },
  { emoji: '😈', keywords: ['criminal', 'poder', 'corrupcion', 'corrupción', 'ataque', 'enemigo', 'izquierda', 'derecha', 'evil', 'power', 'attack', 'enemy'] },
  { emoji: '😂', keywords: ['risa', 'chiste', 'gracioso', 'jaja', 'jajaja', 'humor', 'lol', 'haha', 'funny', 'joke'] },
  { emoji: '🎯', keywords: ['clave', 'tip', 'consejo', 'estrategia', 'enfoque', 'objetivo', 'exacto', 'preciso', 'key', 'tip', 'strategy', 'focus', 'goal'] },
  { emoji: '✅', keywords: ['listo', 'hecho', 'correcto', 'confirmado', 'funciona', 'ok', 'perfecto', 'done', 'ready', 'correct', 'confirmed', 'works'] },
  { emoji: '🤯', keywords: ['increible', 'increíble', 'mindblow', 'bestial', 'impresionante', 'sorpresa', 'unreal', 'shocking', 'mind', 'extraordinary'] },
  { emoji: '💥', keywords: ['boom', 'rompe', 'romper', 'estalla', 'estalló', 'revienta', 'break', 'blast', 'explode'] }
];
const EMOTION_COLOR_RULES = [
  { color: '#FF4D4D', keywords: ['criminal', 'ataque', 'guerra', 'odio', 'corrupcion', 'corrupción', 'rabia', 'violencia'] },
  { color: '#39FF14', keywords: ['dinero', 'plata', 'trading', 'mercado', 'finanzas', 'bitcoin', 'crypto', 'exito', 'éxito'] },
  { color: '#FFC400', keywords: ['alerta', 'riesgo', 'peligro', 'crisis', 'grave', 'urgente'] },
  { color: '#00E5FF', keywords: ['tip', 'clave', 'estrategia', 'tutorial', 'paso', 'metodo', 'método'] },
  { color: '#B266FF', keywords: ['mindset', 'increible', 'increíble', 'wow', 'sorpresa', 'impacto'] }
];

const parseTimeToSeconds = (raw) => {
  const value = String(raw || '').trim();
  const normalized = value.replace(',', '.');
  const [hh, mm, ss] = normalized.split(':');
  const h = Number(hh || 0);
  const m = Number(mm || 0);
  const s = Number(ss || 0);
  if (!Number.isFinite(h) || !Number.isFinite(m) || !Number.isFinite(s)) return 0;
  return (h * 3600) + (m * 60) + s;
};

const formatSrtTime = (seconds) => {
  const total = Math.max(0, Number(seconds) || 0);
  const hh = Math.floor(total / 3600);
  const mm = Math.floor((total % 3600) / 60);
  const ss = Math.floor(total % 60);
  const ms = Math.round((total - Math.floor(total)) * 1000);
  return `${String(hh).padStart(2, '0')}:${String(mm).padStart(2, '0')}:${String(ss).padStart(2, '0')},${String(ms).padStart(3, '0')}`;
};

const isEmojiOnly = (value) => {
  const clean = String(value || '').trim().replace(/\s+/g, '');
  if (!clean || clean.length > 10) return false;
  return /^[\p{Extended_Pictographic}\u200D\uFE0F]+$/u.test(clean);
};

const normalizeEmojiText = (value) => {
  return String(value || '')
    .toLowerCase()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '');
};

const suggestEmojiForText = (value) => {
  const text = normalizeEmojiText(value);
  if (!text.trim()) return '';
  for (const rule of EMOJI_RULES) {
    if (rule.keywords.some((kw) => text.includes(kw))) return rule.emoji;
  }
  if (/[0-9]/.test(text) || /\$|usd|eur|btc|eth/.test(text)) return '💸';
  if (text.split(/\s+/).length <= 2) return '🎯';
  if (text.includes('?')) return '🤔';
  if (text.includes('!')) return '🔥';
  // Fallback determinístico para evitar "no sugiere nada" en líneas neutras.
  const pool = ['🎯', '🔥', '✅', '💥', '🤯'];
  let hash = 0;
  for (let i = 0; i < text.length; i += 1) hash = ((hash << 5) - hash) + text.charCodeAt(i);
  const idx = Math.abs(hash) % pool.length;
  return pool[idx];
};

const suggestEmotionColorForText = (value) => {
  const text = normalizeEmojiText(value);
  if (!text.trim()) return ACTIVE_WORD_COLOR;
  for (const rule of EMOTION_COLOR_RULES) {
    if (rule.keywords.some((kw) => text.includes(kw))) return rule.color;
  }
  if (text.includes('!')) return '#FFC400';
  if (text.includes('?')) return '#00E5FF';
  return ACTIVE_WORD_COLOR;
};

const stripSubtitlePunctuation = (value) => {
  return String(value || '')
    .replace(/[^\p{L}\p{N}\s]/gu, '')
    .replace(/\s+/g, ' ')
    .trim();
};

const formatSubtitleText = (text, emphasize = false, punctuationOn = true) => {
  const base = punctuationOn ? String(text || '').trim() : stripSubtitlePunctuation(text);
  return emphasize ? base.toUpperCase() : base;
};

const truncateInline = (value, maxChars = 84) => {
  const clean = String(value || '').replace(/\s+/g, ' ').trim();
  if (!clean) return '';
  if (clean.length <= maxChars) return clean;
  return `${clean.slice(0, Math.max(1, maxChars - 3)).trim()}...`;
};

const SettingToggle = ({ label, checked, onChange }) => (
  <div className="flex items-center justify-between rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 px-3 py-2.5">
    <span className="text-sm text-slate-700 dark:text-slate-200">{label}</span>
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={onChange}
      className={`w-11 h-6 rounded-full transition-colors ${checked ? 'bg-violet-500' : 'bg-slate-300 dark:bg-slate-600'}`}
    >
      <span
        className={`block w-5 h-5 rounded-full bg-white shadow transition-transform ${checked ? 'translate-x-5' : 'translate-x-0.5'}`}
      />
    </button>
  </div>
);

const parseSrt = (srtText) => {
  const blocks = String(srtText || '').split(/\n\s*\n/).map((b) => b.trim()).filter(Boolean);
  const items = [];
  blocks.forEach((block, idx) => {
    const lines = block.split('\n').map((l) => l.trim()).filter(Boolean);
    if (lines.length < 2) return;
    const maybeIndex = /^\d+$/.test(lines[0]) ? Number(lines[0]) : idx + 1;
    const timeLine = /^\d+$/.test(lines[0]) ? lines[1] : lines[0];
    const textLines = /^\d+$/.test(lines[0]) ? lines.slice(2) : lines.slice(1);
    const normalizedTextLines = textLines.map((line) => String(line || '').trim()).filter(Boolean);
    const [rawStart, rawEnd] = String(timeLine).split('-->').map((v) => v.trim());
    if (!rawStart || !rawEnd) return;
    const hasEmojiLine = normalizedTextLines.length > 1 && isEmojiOnly(normalizedTextLines[0]);
    const emoji = hasEmojiLine ? normalizedTextLines[0] : '';
    const subtitleTextLines = hasEmojiLine ? normalizedTextLines.slice(1) : normalizedTextLines;
    items.push({
      id: `${maybeIndex}-${idx}`,
      index: maybeIndex,
      start: parseTimeToSeconds(rawStart),
      end: parseTimeToSeconds(rawEnd),
      text: subtitleTextLines.join(' ').trim(),
      emphasize: false,
      emoji
    });
  });
  return items;
};

const buildSrt = (entries, options = {}) => {
  const punctuationOn = options.punctuationOn !== false;
  const emojiOn = options.emojiOn !== false;
  if (!Array.isArray(entries)) return '';
  return entries
    .map((entry) => {
      if (!entry || !Number.isFinite(entry.start) || !Number.isFinite(entry.end)) return null;
      const text = formatSubtitleText(entry.text, entry.emphasize, punctuationOn);
      if (!text) return null;
      const emojiLine = emojiOn ? String(entry.emoji || '').trim() : '';
      return {
        start: entry.start,
        end: entry.end,
        text,
        emojiLine
      };
    })
    .filter(Boolean)
    .map((entry, idx) => {
      const subtitleBlock = entry.emojiLine ? `${entry.emojiLine}\n${entry.text}` : entry.text;
      return `${idx + 1}\n${formatSrtTime(entry.start)} --> ${formatSrtTime(entry.end)}\n${subtitleBlock}`;
    })
    .join('\n\n');
};

const toRgba = (hex, opacityPercent) => {
  const clean = String(hex || '#000000').replace('#', '');
  const h = clean.length === 3
    ? clean.split('').map((ch) => `${ch}${ch}`).join('')
    : clean.padEnd(6, '0').slice(0, 6);
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  const alpha = Math.max(0, Math.min(100, Number(opacityPercent) || 0)) / 100;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
};

const extractFilename = (urlOrPath) => {
  const raw = String(urlOrPath || '').trim();
  if (!raw) return '';
  try {
    const url = new URL(raw);
    const cleanPath = url.pathname || '';
    const last = cleanPath.split('/').pop() || '';
    try {
      return decodeURIComponent(last);
    } catch (_) {
      return last;
    }
  } catch (_) {
    const clean = raw.split('?')[0].split('#')[0];
    const last = clean.split('/').pop() || '';
    try {
      return decodeURIComponent(last);
    } catch (__){
      return last;
    }
  }
};

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const LAYOUT_OFFSET_FACTOR = 0.35;
const LAYOUT_PAN_SENSITIVITY = 120;
const CAPTION_OFFSET_FACTOR = 0.35;
const TIMELINE_ZOOM_MIN = 0.55;
const TIMELINE_ZOOM_MAX = 2.2;
const TIMELINE_ZOOM_DEFAULT = 0.9;
const TIMELINE_MODE_MINI = 'mini';
const TIMELINE_MODE_ADVANCED = 'advanced';

const formatTimelineTime = (seconds) => {
  const safe = Math.max(0, Number(seconds || 0));
  const mins = Math.floor(safe / 60);
  const secs = safe - (mins * 60);
  return `${String(mins).padStart(2, '0')}:${secs.toFixed(1).padStart(4, '0')}`;
};

const formatAbsoluteClock = (seconds) => {
  const safe = Math.max(0, Number(seconds || 0));
  const hours = Math.floor(safe / 3600);
  const mins = Math.floor((safe % 3600) / 60);
  const secs = safe % 60;
  if (hours > 0) {
    return `${String(hours).padStart(2, '0')}:${String(mins).padStart(2, '0')}:${secs.toFixed(1).padStart(4, '0')}`;
  }
  return `${String(mins).padStart(2, '0')}:${secs.toFixed(1).padStart(4, '0')}`;
};

export default function ClipStudioModal({
  isOpen,
  standalone = false,
  onClose,
  jobId,
  clipIndex,
  clip,
  currentVideoUrl,
  onApplied
}) {
  const [section, setSection] = useState('transcript');
  const [isApplying, setIsApplying] = useState(false);
  const [isLoadingTranscript, setIsLoadingTranscript] = useState(false);
  const [isLoadingSrt, setIsLoadingSrt] = useState(false);
  const [error, setError] = useState('');
  const [savedPulse, setSavedPulse] = useState(false);

  const [transcriptSegments, setTranscriptSegments] = useState([]);
  const [transcriptQuery, setTranscriptQuery] = useState('');

  const [captionsOn, setCaptionsOn] = useState(true);
  const [showCaptionSettings, setShowCaptionSettings] = useState(false);
  const [selectedPreset, setSelectedPreset] = useState(CAPTION_PRESETS[0].id);
  const [position, setPosition] = useState(CAPTION_PRESETS[0].style.position);
  const [fontSize, setFontSize] = useState(CAPTION_PRESETS[0].style.fontSize);
  const [fontFamily, setFontFamily] = useState(CAPTION_PRESETS[0].style.fontFamily);
  const [fontColor, setFontColor] = useState(CAPTION_PRESETS[0].style.fontColor);
  const [strokeColor, setStrokeColor] = useState('#0A0A0A');
  const [strokeWidth, setStrokeWidth] = useState(3);
  const [bold, setBold] = useState(true);
  const [boxColor, setBoxColor] = useState('#000000');
  const [boxOpacity, setBoxOpacity] = useState(20);
  const [karaokeMode, setKaraokeMode] = useState(false);
  const [punctuationOn, setPunctuationOn] = useState(true);
  const [emojiOn, setEmojiOn] = useState(true);
  const [captionOffsetX, setCaptionOffsetX] = useState(clamp(Number(clip?.caption_offset_x || 0), -100, 100));
  const [captionOffsetY, setCaptionOffsetY] = useState(clamp(Number(clip?.caption_offset_y || 0), -100, 100));
  const [isDraggingCaption, setIsDraggingCaption] = useState(false);

  const [subtitleEntries, setSubtitleEntries] = useState([]);
  const [subtitleSearch, setSubtitleSearch] = useState('');
  const [emojiPickerForId, setEmojiPickerForId] = useState('');
  const [emojiSuggestFeedback, setEmojiSuggestFeedback] = useState('');

  const [layoutAspect, setLayoutAspect] = useState(clip?.aspect_ratio === '16:9' ? '16:9' : '9:16');
  const [layoutStart, setLayoutStart] = useState(Number(clip?.start || 0));
  const [layoutEnd, setLayoutEnd] = useState(Number(clip?.end || 0));
  const [layoutPreRoll, setLayoutPreRoll] = useState(0);
  const [layoutPostRoll, setLayoutPostRoll] = useState(0);
  const [layoutMode, setLayoutMode] = useState(String(clip?.layout_mode || 'single').toLowerCase() === 'split' ? 'split' : 'single');
  const [layoutAutoSmart, setLayoutAutoSmart] = useState(Boolean(clip?.layout_auto_smart));
  const [layoutFitMode, setLayoutFitMode] = useState(String(clip?.layout_fit_mode || 'cover').toLowerCase() === 'contain' ? 'contain' : 'cover');
  const [layoutZoom, setLayoutZoom] = useState(clamp(Number(clip?.layout_zoom || 1), 0.5, 2.5));
  const [layoutOffsetX, setLayoutOffsetX] = useState(clamp(Number(clip?.layout_offset_x || 0), -100, 100));
  const [layoutOffsetY, setLayoutOffsetY] = useState(clamp(Number(clip?.layout_offset_y || 0), -100, 100));
  const [layoutSplitZoomA, setLayoutSplitZoomA] = useState(clamp(Number(clip?.layout_split_zoom_a ?? clip?.layout_zoom ?? 1), 0.5, 2.5));
  const [layoutSplitOffsetAX, setLayoutSplitOffsetAX] = useState(clamp(Number(clip?.layout_split_offset_a_x ?? clip?.layout_offset_x ?? 0), -100, 100));
  const [layoutSplitOffsetAY, setLayoutSplitOffsetAY] = useState(clamp(Number(clip?.layout_split_offset_a_y ?? clip?.layout_offset_y ?? 0), -100, 100));
  const [layoutSplitZoomB, setLayoutSplitZoomB] = useState(clamp(Number(clip?.layout_split_zoom_b ?? clip?.layout_zoom ?? 1), 0.5, 2.5));
  const [layoutSplitOffsetBX, setLayoutSplitOffsetBX] = useState(clamp(Number(clip?.layout_split_offset_b_x ?? (-(Number(clip?.layout_offset_x || 0)))), -100, 100));
  const [layoutSplitOffsetBY, setLayoutSplitOffsetBY] = useState(clamp(Number(clip?.layout_split_offset_b_y ?? clip?.layout_offset_y ?? 0), -100, 100));
  const [isPanningLayout, setIsPanningLayout] = useState(false);
  const isSplitLayout = layoutMode === 'split';
  const effectiveLayoutOffsetX = Number(layoutOffsetX || 0) * LAYOUT_OFFSET_FACTOR;
  const effectiveLayoutOffsetY = Number(layoutOffsetY || 0) * LAYOUT_OFFSET_FACTOR;
  const manualLayoutObjectPosition = useMemo(() => {
    const x = clamp(50 + Number(effectiveLayoutOffsetX || 0), 0, 100);
    const y = clamp(50 + Number(effectiveLayoutOffsetY || 0), 0, 100);
    return `${x.toFixed(3)}% ${y.toFixed(3)}%`;
  }, [effectiveLayoutOffsetX, effectiveLayoutOffsetY]);
  const effectiveSplitOffsetAX = Number(layoutSplitOffsetAX || 0) * LAYOUT_OFFSET_FACTOR;
  const effectiveSplitOffsetAY = Number(layoutSplitOffsetAY || 0) * LAYOUT_OFFSET_FACTOR;
  const effectiveSplitOffsetBX = Number(layoutSplitOffsetBX || 0) * LAYOUT_OFFSET_FACTOR;
  const effectiveSplitOffsetBY = Number(layoutSplitOffsetBY || 0) * LAYOUT_OFFSET_FACTOR;
  const splitObjectPositionA = useMemo(() => {
    const x = clamp(50 + Number(effectiveSplitOffsetAX || 0), 0, 100);
    const y = clamp(50 + Number(effectiveSplitOffsetAY || 0), 0, 100);
    return `${x.toFixed(3)}% ${y.toFixed(3)}%`;
  }, [effectiveSplitOffsetAX, effectiveSplitOffsetAY]);
  const splitObjectPositionB = useMemo(() => {
    const x = clamp(50 + Number(effectiveSplitOffsetBX || 0), 0, 100);
    const y = clamp(50 + Number(effectiveSplitOffsetBY || 0), 0, 100);
    return `${x.toFixed(3)}% ${y.toFixed(3)}%`;
  }, [effectiveSplitOffsetBX, effectiveSplitOffsetBY]);
  const effectiveCaptionOffsetX = Number(captionOffsetX || 0) * CAPTION_OFFSET_FACTOR;
  const effectiveCaptionOffsetY = Number(captionOffsetY || 0) * CAPTION_OFFSET_FACTOR;

  const [musicEnabled, setMusicEnabled] = useState(false);
  const [musicFile, setMusicFile] = useState(null);
  const [musicVolume, setMusicVolume] = useState(0.18);
  const [duckVoice, setDuckVoice] = useState(true);

  const [previewPlaying, setPreviewPlaying] = useState(false);
  const [previewCurrentTime, setPreviewCurrentTime] = useState(0);
  const [previewDuration, setPreviewDuration] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [timelineZoom, setTimelineZoom] = useState(TIMELINE_ZOOM_DEFAULT);
  const [timelineMode, setTimelineMode] = useState(TIMELINE_MODE_MINI);
  const [snapEnabled, setSnapEnabled] = useState(true);
  const [previewVideoUrl, setPreviewVideoUrl] = useState(String(currentVideoUrl || ''));
  const [videoLoadError, setVideoLoadError] = useState('');
  const previewVideoRef = useRef(null);
  const previewSplitVideoRef = useRef(null);
  const previewSurfaceRef = useRef(null);
  const subtitleListRef = useRef(null);
  const transcriptListRef = useRef(null);
  const previewBlobUrlRef = useRef(null);
  const timelineViewportRef = useRef(null);
  const timelineTrackRef = useRef(null);
  const subtitleDragRef = useRef(null);
  const selectionDragRef = useRef(null);
  const panDragRef = useRef(null);
  const captionDragRef = useRef(null);
  const subtitleEntryRefs = useRef(new Map());
  const transcriptEntryRefs = useRef(new Map());
  const lastFocusedSubtitleIdRef = useRef('');
  const lastFocusedTranscriptIdRef = useRef('');

  const activeSubtitleEntry = useMemo(() => {
    const t = Number(previewCurrentTime || 0);
    if (!Array.isArray(subtitleEntries) || subtitleEntries.length === 0) return null;
    return subtitleEntries.find((entry) => {
      const start = Number(entry?.start || 0);
      const end = Number(entry?.end || start);
      const hasText = String(entry?.text || '').trim().length > 0;
      return hasText && t >= start && t <= (end + 0.05);
    }) || null;
  }, [subtitleEntries, previewCurrentTime]);

  const previewText = useMemo(() => {
    if (activeSubtitleEntry) {
      return formatSubtitleText(activeSubtitleEntry.text, activeSubtitleEntry.emphasize, punctuationOn);
    }
    // Estado inicial: mostrar una muestra breve antes de reproducir.
    if (!previewPlaying && previewCurrentTime <= 0.05) {
      const first = subtitleEntries.find((entry) => String(entry?.text || '').trim());
      if (first) return formatSubtitleText(first.text, first.emphasize, punctuationOn);
    }
    return '';
  }, [activeSubtitleEntry, subtitleEntries, previewCurrentTime, previewPlaying, punctuationOn]);

  const previewEmoji = useMemo(() => {
    if (!emojiOn) return '';
    const activeEmoji = String(activeSubtitleEntry?.emoji || '').trim();
    if (activeEmoji) return activeEmoji;
    if (!previewPlaying && previewCurrentTime <= 0.05) {
      const first = subtitleEntries.find((entry) => String(entry?.text || '').trim());
      if (first?.emoji) return String(first.emoji).trim();
    }
    return '';
  }, [activeSubtitleEntry, subtitleEntries, previewCurrentTime, previewPlaying, emojiOn]);

  const karaokePreview = useMemo(() => {
    if (!karaokeMode || !activeSubtitleEntry) return null;
    const rawText = formatSubtitleText(activeSubtitleEntry.text, activeSubtitleEntry.emphasize, punctuationOn);
    if (!rawText) return null;
    const words = rawText.split(/\s+/).filter(Boolean);
    if (words.length === 0) return null;
    const start = Number(activeSubtitleEntry.start || 0);
    const end = Number(activeSubtitleEntry.end || start);
    const duration = Math.max(0.2, end - start);
    const progress = Math.max(0, Math.min(0.9999, (Number(previewCurrentTime || 0) - start) / duration));
    const activeIndex = Math.min(words.length - 1, Math.floor(progress * words.length));
    return {
      words,
      activeIndex,
      activeColor: suggestEmotionColorForText(rawText)
    };
  }, [karaokeMode, activeSubtitleEntry, previewCurrentTime, punctuationOn]);
  const captionDragEnabled = captionsOn && (section === 'captions' || section === 'subtitle_edit');
  const captionAnchorTopPercent = position === 'top' ? 12 : position === 'middle' ? 50 : 86;

  const baseClipStart = Number(clip?.start || 0);
  const baseClipEnd = Number(clip?.end || baseClipStart);

  const filteredTranscript = useMemo(() => {
    const q = String(transcriptQuery || '').trim().toLowerCase();
    const start = Number(layoutStart || 0);
    const end = Number(layoutEnd || start);
    return (transcriptSegments || []).filter((seg) => {
      const segStart = Number(seg?.start || 0);
      const segEnd = Number(seg?.end || segStart);
      const inRange = segEnd > start && segStart < end;
      if (!inRange) return false;
      if (!q) return true;
      const text = String(seg?.text || '').toLowerCase();
      return text.includes(q);
    });
  }, [transcriptSegments, transcriptQuery, layoutStart, layoutEnd]);

  const previewAbsoluteTime = useMemo(
    () => Number(baseClipStart || 0) + Number(previewCurrentTime || 0),
    [baseClipStart, previewCurrentTime]
  );

  const activeTranscriptSegment = useMemo(() => {
    if (!Array.isArray(filteredTranscript) || filteredTranscript.length === 0) return null;
    const t = Number(previewAbsoluteTime || 0);
    return filteredTranscript.find((seg) => {
      const start = Number(seg?.start || 0);
      const end = Number(seg?.end || start);
      const text = String(seg?.text || '').trim();
      return text && t >= start && t <= (end + 0.04);
    }) || null;
  }, [filteredTranscript, previewAbsoluteTime]);

  const previewClipTitle = useMemo(() => {
    const socialTitle = String(clip?.video_title_for_youtube_short || '').trim();
    if (socialTitle) return socialTitle;
    const genericTitle = String(clip?.title || '').trim();
    if (genericTitle) return genericTitle;
    const filename = extractFilename(currentVideoUrl) || extractFilename(clip?.video_url || '');
    if (filename) return filename;
    return `Clip n.o ${Number(clipIndex || 0) + 1}`;
  }, [clip?.video_title_for_youtube_short, clip?.title, clip?.video_url, currentVideoUrl, clipIndex]);

  const previewSocialDescription = useMemo(() => {
    return [
      String(clip?.video_description_for_tiktok || '').trim(),
      String(clip?.video_description_for_instagram || '').trim(),
      String(clip?.score_reason || '').trim()
    ].find(Boolean) || '';
  }, [clip?.video_description_for_tiktok, clip?.video_description_for_instagram, clip?.score_reason]);

  const previewHeaderSubline = useMemo(() => {
    if (!previewSocialDescription) return 'Sin descripción social para este clip.';
    return truncateInline(previewSocialDescription, 220);
  }, [previewSocialDescription]);

  const filteredSubtitleEntries = useMemo(() => {
    const q = String(subtitleSearch || '').trim().toLowerCase();
    if (!q) return subtitleEntries;
    return subtitleEntries.filter((entry) => String(entry?.text || '').toLowerCase().includes(q));
  }, [subtitleEntries, subtitleSearch]);

  useEffect(() => {
    if (section !== 'subtitle_edit') return;
    const activeId = String(activeSubtitleEntry?.id || '');
    if (!activeId) return;
    if (lastFocusedSubtitleIdRef.current === activeId) return;
    lastFocusedSubtitleIdRef.current = activeId;

    const container = subtitleListRef.current;
    const node = subtitleEntryRefs.current.get(activeId);
    if (!container || !node) return;

    const containerRect = container.getBoundingClientRect();
    const nodeRect = node.getBoundingClientRect();
    const outsideTop = nodeRect.top < containerRect.top + 8;
    const outsideBottom = nodeRect.bottom > containerRect.bottom - 8;
    if (outsideTop || outsideBottom) {
      node.scrollIntoView({
        block: 'nearest',
        behavior: previewPlaying ? 'smooth' : 'auto'
      });
    }
  }, [section, activeSubtitleEntry?.id, previewPlaying]);

  useEffect(() => {
    if (section !== 'transcript') return;
    const activeKey = activeTranscriptSegment
      ? `${activeTranscriptSegment.segment_index}-${activeTranscriptSegment.start}`
      : '';
    if (!activeKey) return;
    if (lastFocusedTranscriptIdRef.current === activeKey) return;
    lastFocusedTranscriptIdRef.current = activeKey;

    const container = transcriptListRef.current;
    const node = transcriptEntryRefs.current.get(activeKey);
    if (!container || !node) return;

    const containerRect = container.getBoundingClientRect();
    const nodeRect = node.getBoundingClientRect();
    const outsideTop = nodeRect.top < containerRect.top + 8;
    const outsideBottom = nodeRect.bottom > containerRect.bottom - 8;
    if (outsideTop || outsideBottom) {
      node.scrollIntoView({
        block: 'nearest',
        behavior: previewPlaying ? 'smooth' : 'auto'
      });
    }
  }, [section, activeTranscriptSegment, previewPlaying]);

  const srtContent = useMemo(
    () => buildSrt(subtitleEntries, { punctuationOn, emojiOn }),
    [subtitleEntries, punctuationOn, emojiOn]
  );
  const timelineDuration = useMemo(() => {
    const d = Number(previewDuration || 0);
    if (d > 0) return d;
    const fallback = Math.max(1, baseClipEnd - baseClipStart);
    return fallback;
  }, [previewDuration, baseClipStart, baseClipEnd]);

  const selectionStartRel = useMemo(() => {
    return clamp(Number(layoutStart || baseClipStart) - baseClipStart, 0, timelineDuration);
  }, [layoutStart, baseClipStart, timelineDuration]);

  const selectionEndRel = useMemo(() => {
    const raw = Number(layoutEnd || baseClipEnd) - baseClipStart;
    return clamp(raw, selectionStartRel + 0.08, timelineDuration);
  }, [layoutEnd, baseClipEnd, baseClipStart, selectionStartRel, timelineDuration]);

  const timelineTicks = useMemo(() => {
    const duration = Math.max(1, Number(timelineDuration || 0));
    const targetTicks = 5;
    const rawStep = duration / targetTicks;
    const step = rawStep <= 2 ? 2 : rawStep <= 5 ? 5 : rawStep <= 7 ? 7 : rawStep <= 10 ? 10 : 14;
    const ticks = [];
    for (let t = 0; t <= duration + 0.001; t += step) {
      ticks.push(Number(t.toFixed(2)));
    }
    if (ticks[ticks.length - 1] < duration) ticks.push(duration);
    return ticks;
  }, [timelineDuration]);

  const timelineDensityBars = useMemo(() => {
    const bars = clamp(Math.round(84 * timelineZoom), 56, 240);
    const duration = Math.max(0.1, Number(timelineDuration || 1));
    const entries = Array.isArray(subtitleEntries) ? subtitleEntries : [];
    return Array.from({ length: bars }).map((_, idx) => {
      const start = (idx / bars) * duration;
      const end = ((idx + 1) / bars) * duration;
      let overlap = 0;
      entries.forEach((entry) => {
        const es = Number(entry?.start || 0);
        const ee = Number(entry?.end || es);
        overlap += Math.max(0, Math.min(end, ee) - Math.max(start, es));
      });
      const normalized = Math.min(1, overlap / Math.max(0.08, (end - start)));
      return 0.15 + (normalized * 0.85);
    });
  }, [timelineDuration, subtitleEntries, timelineZoom]);

  const snapPoints = useMemo(() => {
    const points = new Set();
    points.add(0);
    points.add(Number(timelineDuration.toFixed(3)));

    (transcriptSegments || []).forEach((seg) => {
      const segStart = clamp(Number(seg?.start || 0) - baseClipStart, 0, timelineDuration);
      const segEnd = clamp(Number(seg?.end || segStart) - baseClipStart, 0, timelineDuration);
      points.add(Number(segStart.toFixed(3)));
      points.add(Number(segEnd.toFixed(3)));

      const words = Array.isArray(seg?.words) ? seg.words : [];
      words.forEach((word) => {
        const ws = clamp(Number(word?.start || 0) - baseClipStart, 0, timelineDuration);
        const we = clamp(Number(word?.end || ws) - baseClipStart, 0, timelineDuration);
        points.add(Number(ws.toFixed(3)));
        points.add(Number(we.toFixed(3)));
      });
    });

    subtitleEntries.forEach((entry) => {
      points.add(Number(clamp(Number(entry?.start || 0), 0, timelineDuration).toFixed(3)));
      points.add(Number(clamp(Number(entry?.end || 0), 0, timelineDuration).toFixed(3)));
    });

    return Array.from(points).sort((a, b) => a - b);
  }, [transcriptSegments, subtitleEntries, baseClipStart, timelineDuration]);

  const snapThreshold = useMemo(
    () => clamp(0.16 / Math.max(TIMELINE_ZOOM_MIN, timelineZoom), 0.025, 0.12),
    [timelineZoom]
  );

  const snapToNearest = useCallback((timeValue) => {
    const raw = clamp(Number(timeValue || 0), 0, timelineDuration);
    if (!snapEnabled || snapPoints.length === 0) return raw;

    let nearest = raw;
    let best = snapThreshold;
    for (let i = 0; i < snapPoints.length; i += 1) {
      const point = snapPoints[i];
      const diff = Math.abs(point - raw);
      if (diff <= best) {
        best = diff;
        nearest = point;
      }
      if (point > raw + snapThreshold) break;
    }
    return nearest;
  }, [snapEnabled, snapPoints, snapThreshold, timelineDuration]);

  const applyPreset = (presetId) => {
    const preset = CAPTION_PRESETS.find((p) => p.id === presetId);
    if (!preset) return;
    setSelectedPreset(presetId);
    setPosition(preset.style.position);
    setFontSize(preset.style.fontSize);
    setFontFamily(preset.style.fontFamily);
    setFontColor(preset.style.fontColor);
    setStrokeColor(preset.style.strokeColor);
    setStrokeWidth(preset.style.strokeWidth);
    setBold(Boolean(preset.style.bold));
    setBoxColor(preset.style.boxColor);
    setBoxOpacity(preset.style.boxOpacity);
  };

  const loadTranscript = async () => {
    if (!jobId) return;
    setIsLoadingTranscript(true);
    try {
      const res = await apiFetch(`/api/transcript/${jobId}?limit=2000&include_words=1`);
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      const segments = Array.isArray(data?.segments) ? data.segments : [];
      setTranscriptSegments(segments);
    } catch (e) {
      setError(`No se pudo cargar transcript: ${e.message}`);
      setTimeout(() => setError(''), 3500);
    } finally {
      setIsLoadingTranscript(false);
    }
  };

  const loadSrt = async () => {
    if (!jobId) return;
    setIsLoadingSrt(true);
    try {
      const res = await apiFetch('/api/subtitle/preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_id: jobId, clip_index: clipIndex })
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      const parsed = parseSrt(data?.srt || '');
      setSubtitleEntries(parsed);
    } catch (e) {
      setError(`No se pudo cargar subtítulos: ${e.message}`);
      setTimeout(() => setError(''), 3500);
    } finally {
      setIsLoadingSrt(false);
    }
  };

  useEffect(() => {
    if (!isOpen) return;
    setSection('transcript');
    setLayoutAspect(clip?.aspect_ratio === '16:9' ? '16:9' : '9:16');
    setLayoutStart(Number(clip?.start || 0));
    setLayoutEnd(Number(clip?.end || 0));
    setLayoutPreRoll(0);
    setLayoutPostRoll(0);
    setLayoutMode(String(clip?.layout_mode || 'single').toLowerCase() === 'split' ? 'split' : 'single');
    setLayoutAutoSmart(Boolean(clip?.layout_auto_smart));
    setLayoutFitMode(String(clip?.layout_fit_mode || 'cover').toLowerCase() === 'contain' ? 'contain' : 'cover');
    setLayoutZoom(clamp(Number(clip?.layout_zoom || 1), 0.5, 2.5));
    setLayoutOffsetX(clamp(Number(clip?.layout_offset_x || 0), -100, 100));
    setLayoutOffsetY(clamp(Number(clip?.layout_offset_y || 0), -100, 100));
    setLayoutSplitZoomA(clamp(Number(clip?.layout_split_zoom_a ?? clip?.layout_zoom ?? 1), 0.5, 2.5));
    setLayoutSplitOffsetAX(clamp(Number(clip?.layout_split_offset_a_x ?? clip?.layout_offset_x ?? 0), -100, 100));
    setLayoutSplitOffsetAY(clamp(Number(clip?.layout_split_offset_a_y ?? clip?.layout_offset_y ?? 0), -100, 100));
    setLayoutSplitZoomB(clamp(Number(clip?.layout_split_zoom_b ?? clip?.layout_zoom ?? 1), 0.5, 2.5));
    setLayoutSplitOffsetBX(clamp(Number(clip?.layout_split_offset_b_x ?? (-(Number(clip?.layout_offset_x || 0)))), -100, 100));
    setLayoutSplitOffsetBY(clamp(Number(clip?.layout_split_offset_b_y ?? clip?.layout_offset_y ?? 0), -100, 100));
    setMusicEnabled(false);
    setMusicFile(null);
    setMusicVolume(0.18);
    setDuckVoice(true);
    setPunctuationOn(true);
    setEmojiOn(true);
    setShowCaptionSettings(false);
    setEmojiPickerForId('');
    setEmojiSuggestFeedback('');
    setTimelineZoom(TIMELINE_ZOOM_DEFAULT);
    setTimelineMode(TIMELINE_MODE_MINI);
    applyPreset(CAPTION_PRESETS[0].id);
    const savedCaptionPosition = String(clip?.caption_position || '').toLowerCase();
    if (savedCaptionPosition === 'top' || savedCaptionPosition === 'middle' || savedCaptionPosition === 'bottom') {
      setPosition(savedCaptionPosition);
    }
    setCaptionOffsetX(clamp(Number(clip?.caption_offset_x || 0), -100, 100));
    setCaptionOffsetY(clamp(Number(clip?.caption_offset_y || 0), -100, 100));
    loadTranscript();
    loadSrt();
  }, [
    isOpen,
    jobId,
    clipIndex,
    clip?.start,
    clip?.end,
    clip?.aspect_ratio,
    clip?.layout_mode,
    clip?.layout_auto_smart,
    clip?.layout_fit_mode,
    clip?.layout_zoom,
    clip?.layout_offset_x,
    clip?.layout_offset_y,
    clip?.layout_split_zoom_a,
    clip?.layout_split_offset_a_x,
    clip?.layout_split_offset_a_y,
    clip?.layout_split_zoom_b,
    clip?.layout_split_offset_b_x,
    clip?.layout_split_offset_b_y,
    clip?.caption_position,
    clip?.caption_offset_x,
    clip?.caption_offset_y
  ]);

  useEffect(() => {
    const cleanupBlobUrl = () => {
      if (previewBlobUrlRef.current) {
        URL.revokeObjectURL(previewBlobUrlRef.current);
        previewBlobUrlRef.current = null;
      }
    };

    const sourceUrl = String(currentVideoUrl || '').trim();
    if (!isOpen) return () => {};
    if (!sourceUrl) {
      cleanupBlobUrl();
      setPreviewVideoUrl('');
      setVideoLoadError('');
      return () => {};
    }

    const isHttp = /^https?:\/\//i.test(sourceUrl);
    const isBlobOrData = /^blob:|^data:/i.test(sourceUrl);
    if (!isHttp || isBlobOrData) {
      cleanupBlobUrl();
      setPreviewVideoUrl(sourceUrl);
      setVideoLoadError('');
      return () => {};
    }

    const isNgrokSource = /ngrok/i.test(sourceUrl);
    if (!isNgrokSource) {
      cleanupBlobUrl();
      setPreviewVideoUrl(sourceUrl);
      setVideoLoadError('');
      return () => {};
    }

    let cancelled = false;
    setVideoLoadError('');
    (async () => {
      try {
        const res = await apiFetch(sourceUrl, { method: 'GET' });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const blob = await res.blob();
        const blobType = String(blob.type || '').toLowerCase();
        if (blobType && !blobType.startsWith('video/') && !blobType.includes('mp4') && !blobType.includes('octet-stream')) {
          throw new Error(`Tipo recibido: ${blob.type || 'desconocido'}`);
        }
        if (cancelled) return;
        cleanupBlobUrl();
        const objectUrl = URL.createObjectURL(blob);
        previewBlobUrlRef.current = objectUrl;
        setPreviewVideoUrl(objectUrl);
      } catch (err) {
        if (cancelled) return;
        // Fallback silencioso: aun si falla la descarga manual, el elemento <video>
        // puede cargar la URL remota directamente.
        setPreviewVideoUrl(sourceUrl);
        setVideoLoadError('');
      }
    })();

    return () => {
      cancelled = true;
      cleanupBlobUrl();
    };
  }, [isOpen, currentVideoUrl]);

  useEffect(() => {
    const video = previewVideoRef.current;
    if (!video) return;
    video.playbackRate = Number(playbackRate || 1);
  }, [playbackRate, isOpen]);

  useEffect(() => {
    const splitVideo = previewSplitVideoRef.current;
    if (!splitVideo) return;
    splitVideo.playbackRate = Number(playbackRate || 1);
  }, [playbackRate, isOpen, isSplitLayout]);

  useEffect(() => {
    const primary = previewVideoRef.current;
    const secondary = previewSplitVideoRef.current;
    if (!secondary) return;
    if (!isSplitLayout) {
      secondary.pause();
      return;
    }
    if (!primary) return;

    try {
      const targetTime = Number(primary.currentTime || previewCurrentTime || 0);
      if (Math.abs(Number(secondary.currentTime || 0) - targetTime) > 0.05) {
        secondary.currentTime = targetTime;
      }
    } catch (_) {
      // Ignore sync errors from browsers while seeking.
    }

    if (primary.paused) {
      secondary.pause();
      return;
    }
    const playPromise = secondary.play();
    if (playPromise && typeof playPromise.catch === 'function') {
      playPromise.catch(() => {});
    }
  }, [isSplitLayout, previewCurrentTime, previewPlaying, previewVideoUrl, isOpen]);

  const onSubtitleEntryChange = (entryId, nextText) => {
    setSubtitleEntries((prev) => prev.map((entry) => (
      entry.id === entryId ? { ...entry, text: nextText } : entry
    )));
    setSavedPulse(false);
  };

  const onSubtitleToggleEmphasis = (entryId) => {
    setSubtitleEntries((prev) => prev.map((entry) => (
      entry.id === entryId ? { ...entry, emphasize: !entry.emphasize } : entry
    )));
    setSavedPulse(false);
  };

  const onSubtitleEntryEmojiChange = (entryId, nextEmoji) => {
    setSubtitleEntries((prev) => prev.map((entry) => (
      entry.id === entryId ? { ...entry, emoji: String(nextEmoji || '').trim() } : entry
    )));
    setSavedPulse(false);
  };

  const autoSuggestEmojis = () => {
    let updatedCount = 0;
    setSubtitleEntries((prev) => prev.map((entry) => {
      if (String(entry?.emoji || '').trim()) return entry;
      const suggested = suggestEmojiForText(entry?.text || '');
      if (!suggested) return entry;
      updatedCount += 1;
      return { ...entry, emoji: suggested };
    }));
    setEmojiSuggestFeedback(
      updatedCount > 0
        ? `IA local sugirió emojis en ${updatedCount} línea${updatedCount === 1 ? '' : 's'}.`
        : 'No había líneas disponibles para sugerir (ya tenían emoji o estaban vacías).'
    );
    setTimeout(() => setEmojiSuggestFeedback(''), 3200);
    setSavedPulse(false);
  };

  const handleApply = async () => {
    if (!jobId) return;
    setIsApplying(true);
    setError('');
    let workingFile = extractFilename(currentVideoUrl);
    let resultingUrl = currentVideoUrl;
    const normalizedLayoutMode = isSplitLayout ? 'split' : 'single';
    const autoSmartRequest = normalizedLayoutMode === 'single' ? Boolean(layoutAutoSmart) : false;
    let appliedLayoutMode = normalizedLayoutMode;
    let appliedLayoutAutoSmart = autoSmartRequest;

    try {
      const clipStart = Number(clip?.start || 0);
      const clipEnd = Number(clip?.end || clipStart);
      let appliedClipStart = clipStart;
      let appliedClipEnd = clipEnd;
      const safePreRoll = clamp(Number(layoutPreRoll || 0), 0, 3);
      const safePostRoll = clamp(Number(layoutPostRoll || 0), 0, 3);
      const requestedStart = Math.max(0, Number(layoutStart || 0) - safePreRoll);
      const requestedEnd = Math.max(requestedStart + 0.08, Number(layoutEnd || requestedStart) + safePostRoll);
      if (!Number.isFinite(requestedStart) || !Number.isFinite(requestedEnd) || requestedEnd <= requestedStart) {
        throw new Error('Rango inválido. Ajusta inicio/fin o pre/post roll.');
      }
      const originalLayoutMode = String(clip?.layout_mode || 'single').toLowerCase() === 'split' ? 'split' : 'single';
      const originalAutoSmart = originalLayoutMode === 'single' ? Boolean(clip?.layout_auto_smart) : false;
      const originalFitMode = String(clip?.layout_fit_mode || 'cover').toLowerCase() === 'contain' ? 'contain' : 'cover';
      const originalZoom = clamp(Number(clip?.layout_zoom || 1), 0.5, 2.5);
      const originalOffsetX = clamp(Number(clip?.layout_offset_x || 0), -100, 100);
      const originalOffsetY = clamp(Number(clip?.layout_offset_y || 0), -100, 100);
      const originalSplitZoomA = clamp(Number(clip?.layout_split_zoom_a ?? clip?.layout_zoom ?? 1), 0.5, 2.5);
      const originalSplitOffsetAX = clamp(Number(clip?.layout_split_offset_a_x ?? clip?.layout_offset_x ?? 0), -100, 100);
      const originalSplitOffsetAY = clamp(Number(clip?.layout_split_offset_a_y ?? clip?.layout_offset_y ?? 0), -100, 100);
      const originalSplitZoomB = clamp(Number(clip?.layout_split_zoom_b ?? clip?.layout_zoom ?? 1), 0.5, 2.5);
      const originalSplitOffsetBX = clamp(Number(clip?.layout_split_offset_b_x ?? (-(Number(clip?.layout_offset_x || 0)))), -100, 100);
      const originalSplitOffsetBY = clamp(Number(clip?.layout_split_offset_b_y ?? clip?.layout_offset_y ?? 0), -100, 100);
      const needsRecut = layoutAspect !== (clip?.aspect_ratio === '16:9' ? '16:9' : '9:16')
        || Math.abs(requestedStart - clipStart) > 0.01
        || Math.abs(requestedEnd - clipEnd) > 0.01
        || normalizedLayoutMode !== originalLayoutMode
        || autoSmartRequest !== originalAutoSmart
        || layoutFitMode !== originalFitMode
        || Math.abs(Number(layoutZoom) - originalZoom) > 0.001
        || Math.abs(Number(layoutOffsetX) - originalOffsetX) > 0.01
        || Math.abs(Number(layoutOffsetY) - originalOffsetY) > 0.01
        || Math.abs(Number(layoutSplitZoomA) - originalSplitZoomA) > 0.001
        || Math.abs(Number(layoutSplitOffsetAX) - originalSplitOffsetAX) > 0.01
        || Math.abs(Number(layoutSplitOffsetAY) - originalSplitOffsetAY) > 0.01
        || Math.abs(Number(layoutSplitZoomB) - originalSplitZoomB) > 0.001
        || Math.abs(Number(layoutSplitOffsetBX) - originalSplitOffsetBX) > 0.01
        || Math.abs(Number(layoutSplitOffsetBY) - originalSplitOffsetBY) > 0.01;

      if (needsRecut) {
        const recutRes = await apiFetch('/api/recut', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            job_id: jobId,
            clip_index: clipIndex,
            start: requestedStart,
            end: requestedEnd,
            aspect_ratio: layoutAspect,
            layout_mode: normalizedLayoutMode,
            auto_smart_reframe: autoSmartRequest,
            fit_mode: layoutFitMode,
            zoom: Number(layoutZoom),
            offset_x: Number(layoutOffsetX),
            offset_y: Number(layoutOffsetY),
            split_zoom_a: Number(layoutSplitZoomA),
            split_offset_a_x: Number(layoutSplitOffsetAX),
            split_offset_a_y: Number(layoutSplitOffsetAY),
            split_zoom_b: Number(layoutSplitZoomB),
            split_offset_b_x: Number(layoutSplitOffsetBX),
            split_offset_b_y: Number(layoutSplitOffsetBY)
          })
        });
        if (!recutRes.ok) throw new Error(await recutRes.text());
        const recutData = await recutRes.json();
        if (recutData?.new_video_url) {
          resultingUrl = getApiUrl(recutData.new_video_url);
          workingFile = extractFilename(recutData.new_video_url);
        }
        if (Number.isFinite(Number(recutData?.start)) && Number.isFinite(Number(recutData?.end))) {
          appliedClipStart = Number(recutData.start);
          appliedClipEnd = Number(recutData.end);
        } else {
          appliedClipStart = requestedStart;
          appliedClipEnd = requestedEnd;
        }
        if (typeof recutData?.layout_mode === 'string') {
          appliedLayoutMode = String(recutData.layout_mode).toLowerCase() === 'split' ? 'split' : 'single';
          setLayoutMode(appliedLayoutMode);
        }
        if (typeof recutData?.auto_smart_reframe_applied === 'boolean') {
          appliedLayoutAutoSmart = Boolean(recutData.auto_smart_reframe_applied);
          setLayoutAutoSmart(appliedLayoutAutoSmart);
        }
      }

      if (captionsOn) {
        const subtitleRes = await apiFetch('/api/subtitle', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            job_id: jobId,
            clip_index: clipIndex,
            position,
            font_size: Number(fontSize),
            font_family: fontFamily,
            font_color: fontColor,
            stroke_color: strokeColor,
            stroke_width: Number(strokeWidth),
            bold: Boolean(bold),
            box_color: boxColor,
            box_opacity: Number(boxOpacity),
            karaoke_mode: Boolean(karaokeMode),
            caption_offset_x: Number(captionOffsetX),
            caption_offset_y: Number(captionOffsetY),
            srt_content: srtContent || null,
            input_filename: workingFile || undefined
          })
        });
        if (!subtitleRes.ok) throw new Error(await subtitleRes.text());
        const subtitleData = await subtitleRes.json();
        if (subtitleData?.new_video_url) {
          resultingUrl = getApiUrl(subtitleData.new_video_url);
          workingFile = extractFilename(subtitleData.new_video_url);
        }
      }

      if (musicEnabled && musicFile) {
        const formData = new FormData();
        formData.append('job_id', String(jobId));
        formData.append('clip_index', String(clipIndex));
        if (workingFile) formData.append('input_filename', workingFile);
        formData.append('music_volume', String(musicVolume));
        formData.append('duck_voice', String(duckVoice));
        formData.append('file', musicFile);

        const musicRes = await apiFetch('/api/music', {
          method: 'POST',
          body: formData
        });
        if (!musicRes.ok) throw new Error(await musicRes.text());
        const musicData = await musicRes.json();
        if (musicData?.new_video_url) {
          resultingUrl = getApiUrl(musicData.new_video_url);
          workingFile = extractFilename(musicData.new_video_url);
        }
      }

      setSavedPulse(true);
      onApplied && onApplied({
        newVideoUrl: resultingUrl,
        clipPatch: {
          start: appliedClipStart,
          end: appliedClipEnd,
          aspect_ratio: layoutAspect,
          layout_mode: appliedLayoutMode,
          layout_auto_smart: appliedLayoutAutoSmart,
          layout_fit_mode: layoutFitMode,
          layout_zoom: Number(layoutZoom),
          layout_offset_x: Number(layoutOffsetX),
          layout_offset_y: Number(layoutOffsetY),
          layout_split_zoom_a: Number(layoutSplitZoomA),
          layout_split_offset_a_x: Number(layoutSplitOffsetAX),
          layout_split_offset_a_y: Number(layoutSplitOffsetAY),
          layout_split_zoom_b: Number(layoutSplitZoomB),
          layout_split_offset_b_x: Number(layoutSplitOffsetBX),
          layout_split_offset_b_y: Number(layoutSplitOffsetBY),
          caption_position: position,
          caption_offset_x: Number(captionOffsetX),
          caption_offset_y: Number(captionOffsetY)
        }
      });
      onClose && onClose();
    } catch (e) {
      setError(`No se pudo aplicar cambios: ${e.message}`);
    } finally {
      setIsApplying(false);
    }
  };

  const togglePreviewPlayback = () => {
    const video = previewVideoRef.current;
    if (!video) return;
    if (video.paused) {
      video.play().catch(() => {});
    } else {
      video.pause();
    }
  };

  const seekTo = (nextTime) => {
    const video = previewVideoRef.current;
    if (!video) return;
    const bounded = clamp(Number(nextTime || 0), 0, Math.max(0, timelineDuration));
    video.currentTime = bounded;
    setPreviewCurrentTime(bounded);
  };

  const getTimelineTimeFromClientX = (clientX) => {
    const el = timelineTrackRef.current;
    if (!el) return null;
    const rect = el.getBoundingClientRect();
    const ratio = clamp((clientX - rect.left) / Math.max(1, rect.width), 0, 1);
    return ratio * timelineDuration;
  };

  const handleTimelinePointerSeek = (clientX) => {
    const t = getTimelineTimeFromClientX(clientX);
    if (t === null) return;
    seekTo(snapToNearest(t));
  };

  const cyclePlaybackRate = () => {
    const presets = [0.75, 1, 1.25, 1.5];
    const idx = presets.findIndex((v) => Math.abs(v - playbackRate) < 0.001);
    const next = presets[(idx + 1) % presets.length];
    setPlaybackRate(next);
  };

  const startSubtitleDrag = (event, entry, mode = 'move') => {
    event.preventDefault();
    event.stopPropagation();
    const pointerTime = getTimelineTimeFromClientX(event.clientX);
    if (pointerTime === null) return;
    subtitleDragRef.current = {
      entryId: entry.id,
      mode,
      pointerTime,
      originalStart: Number(entry.start || 0),
      originalEnd: Number(entry.end || entry.start || 0)
    };
  };

  const startSelectionDrag = (event, mode = 'move') => {
    event.preventDefault();
    event.stopPropagation();
    const pointerTime = getTimelineTimeFromClientX(event.clientX);
    if (pointerTime === null) return;
    selectionDragRef.current = {
      mode,
      pointerTime,
      startRel: selectionStartRel,
      endRel: selectionEndRel
    };
  };

  const startLayoutPan = (event) => {
    if (section !== 'layout') return;
    if (isSplitLayout) return;
    if (layoutAutoSmart) return;
    if (event.button !== 0) return;
    const surface = previewSurfaceRef.current;
    if (!surface) return;
    const rect = surface.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    event.preventDefault();
    panDragRef.current = {
      startClientX: event.clientX,
      startClientY: event.clientY,
      startOffsetX: Number(layoutOffsetX || 0),
      startOffsetY: Number(layoutOffsetY || 0),
      width: rect.width,
      height: rect.height
    };
    setIsPanningLayout(true);
  };

  const startCaptionDrag = (event) => {
    if (!captionDragEnabled) return;
    if (event.button !== 0) return;
    const surface = previewSurfaceRef.current;
    if (!surface) return;
    const rect = surface.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    event.preventDefault();
    event.stopPropagation();
    captionDragRef.current = {
      startClientX: event.clientX,
      startClientY: event.clientY,
      startOffsetX: Number(captionOffsetX || 0),
      startOffsetY: Number(captionOffsetY || 0),
      width: rect.width,
      height: rect.height
    };
    setIsDraggingCaption(true);
  };

  useEffect(() => {
    const onMove = (event) => {
      if (subtitleDragRef.current) {
        const drag = subtitleDragRef.current;
        const pointerTime = getTimelineTimeFromClientX(event.clientX);
        if (pointerTime === null) return;
        const delta = pointerTime - drag.pointerTime;
        const minDuration = 0.08;
        setSubtitleEntries((prev) => prev.map((entry) => {
          if (entry.id !== drag.entryId) return entry;
          const origStart = Number(drag.originalStart || 0);
          const origEnd = Number(drag.originalEnd || origStart);
          const duration = Math.max(minDuration, origEnd - origStart);
          let nextStart = origStart;
          let nextEnd = origEnd;

          if (drag.mode === 'move') {
            nextStart = clamp(origStart + delta, 0, Math.max(0, timelineDuration - duration));
            nextStart = snapToNearest(nextStart);
            nextStart = clamp(nextStart, 0, Math.max(0, timelineDuration - duration));
            nextEnd = nextStart + duration;
          } else if (drag.mode === 'start') {
            nextStart = clamp(origStart + delta, 0, origEnd - minDuration);
            nextStart = snapToNearest(nextStart);
            nextStart = clamp(nextStart, 0, origEnd - minDuration);
            nextEnd = origEnd;
          } else {
            nextStart = origStart;
            nextEnd = clamp(origEnd + delta, origStart + minDuration, timelineDuration);
            nextEnd = snapToNearest(nextEnd);
            nextEnd = clamp(nextEnd, origStart + minDuration, timelineDuration);
          }
          return {
            ...entry,
            start: Number(nextStart.toFixed(3)),
            end: Number(nextEnd.toFixed(3))
          };
        }));
      }

      if (selectionDragRef.current) {
        const drag = selectionDragRef.current;
        const pointerTime = getTimelineTimeFromClientX(event.clientX);
        if (pointerTime === null) return;
        const delta = pointerTime - drag.pointerTime;
        const minSpan = 0.2;
        let nextStart = drag.startRel;
        let nextEnd = drag.endRel;

        if (drag.mode === 'move') {
          const span = Math.max(minSpan, drag.endRel - drag.startRel);
          nextStart = clamp(drag.startRel + delta, 0, Math.max(0, timelineDuration - span));
          nextStart = snapToNearest(nextStart);
          nextStart = clamp(nextStart, 0, Math.max(0, timelineDuration - span));
          nextEnd = nextStart + span;
        } else if (drag.mode === 'start') {
          nextStart = clamp(drag.startRel + delta, 0, drag.endRel - minSpan);
          nextStart = snapToNearest(nextStart);
          nextStart = clamp(nextStart, 0, drag.endRel - minSpan);
          nextEnd = drag.endRel;
        } else {
          nextStart = drag.startRel;
          nextEnd = clamp(drag.endRel + delta, drag.startRel + minSpan, timelineDuration);
          nextEnd = snapToNearest(nextEnd);
          nextEnd = clamp(nextEnd, drag.startRel + minSpan, timelineDuration);
        }

        setLayoutStart(Number((baseClipStart + nextStart).toFixed(3)));
        setLayoutEnd(Number((baseClipStart + nextEnd).toFixed(3)));
      }

      if (panDragRef.current) {
        const drag = panDragRef.current;
        const dx = Number(event.clientX || 0) - drag.startClientX;
        const dy = Number(event.clientY || 0) - drag.startClientY;
        const nextX = drag.startOffsetX + ((dx / Math.max(1, drag.width)) * LAYOUT_PAN_SENSITIVITY);
        const nextY = drag.startOffsetY + ((dy / Math.max(1, drag.height)) * LAYOUT_PAN_SENSITIVITY);
        setLayoutOffsetX(clamp(nextX, -100, 100));
        setLayoutOffsetY(clamp(nextY, -100, 100));
      }

      if (captionDragRef.current) {
        const drag = captionDragRef.current;
        const dx = Number(event.clientX || 0) - drag.startClientX;
        const dy = Number(event.clientY || 0) - drag.startClientY;
        const deltaEffectiveX = (dx / Math.max(1, drag.width)) * 100;
        const deltaEffectiveY = (dy / Math.max(1, drag.height)) * 100;
        const nextX = drag.startOffsetX + (deltaEffectiveX / CAPTION_OFFSET_FACTOR);
        const nextY = drag.startOffsetY + (deltaEffectiveY / CAPTION_OFFSET_FACTOR);
        setCaptionOffsetX(clamp(nextX, -100, 100));
        setCaptionOffsetY(clamp(nextY, -100, 100));
        setSavedPulse(false);
      }
    };

    const onUp = () => {
      if (subtitleDragRef.current) {
        subtitleDragRef.current = null;
        setSavedPulse(false);
      }
      if (selectionDragRef.current) {
        selectionDragRef.current = null;
      }
      if (panDragRef.current) {
        panDragRef.current = null;
      }
      if (isPanningLayout) {
        setIsPanningLayout(false);
      }
      if (captionDragRef.current) {
        captionDragRef.current = null;
      }
      if (isDraggingCaption) {
        setIsDraggingCaption(false);
      }
    };

    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [
    timelineDuration,
    baseClipStart,
    selectionStartRel,
    selectionEndRel,
    snapToNearest,
    isPanningLayout,
    isDraggingCaption
  ]);

  if (!isOpen) return null;

  const aspectRatioClass = layoutAspect === '16:9' ? 'aspect-video max-w-[760px]' : 'aspect-[9/16] max-w-[420px]';
  const shellClass = standalone
    ? 'w-full h-full'
    : 'fixed inset-0 z-[110] bg-black/45 backdrop-blur-sm p-3 md:p-6';
  const frameClass = standalone
    ? 'w-full h-full rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-sm overflow-hidden flex flex-col'
    : 'w-full h-full rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-2xl overflow-hidden flex flex-col';

  return (
    <div className={shellClass}>
      <div className={frameClass}>
        <div className="flex items-center justify-between px-4 md:px-6 py-3 border-b border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900">
          <div className="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-200">
            <Pencil size={16} />
            <span className="font-semibold">Modo edición de clip</span>
          </div>
          <div className="flex items-center gap-3">
            <span className={`text-xs ${savedPulse ? 'text-emerald-600 dark:text-emerald-400' : 'text-slate-500'}`}>{savedPulse ? 'Guardado' : 'Sin aplicar'}</span>
            <button
              type="button"
              onClick={onClose}
              className="p-2 rounded-full border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700"
            >
              <X size={16} />
            </button>
            <button
              type="button"
              onClick={handleApply}
              disabled={isApplying}
              className="px-4 py-2 rounded-full bg-primary text-white text-sm font-semibold hover:bg-primary/90 disabled:opacity-60 inline-flex items-center gap-2"
            >
              {isApplying ? <Loader2 size={15} className="animate-spin" /> : <Sparkles size={15} />}
              {isApplying ? 'Aplicando...' : 'Aplicar'}
            </button>
          </div>
        </div>

        <div className="flex-1 min-h-0 flex">
          <aside className="w-[88px] md:w-[94px] border-r border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 p-2 space-y-2 overflow-y-auto">
            {SECTION_ITEMS.map((item) => {
              const Icon = item.icon;
              const active = section === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => setSection(item.id)}
                  className={`w-full rounded-xl py-3 px-1 text-center border transition-colors ${active
                    ? 'bg-primary/10 border-primary/40 text-primary'
                    : 'bg-white border-slate-200 text-slate-600 hover:bg-slate-100 dark:bg-slate-800 dark:border-slate-700 dark:text-slate-300 dark:hover:bg-slate-700'
                  }`}
                >
                  <Icon size={15} className="mx-auto mb-1" />
                  <div className="text-[11px] leading-tight font-medium">{item.label}</div>
                </button>
              );
            })}
          </aside>

          <div className="flex-1 min-w-0 grid grid-cols-1 xl:grid-cols-[420px_1fr] gap-0">
            <section className="border-r border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 p-4 overflow-y-auto custom-scrollbar">
              {section === 'transcript' && (
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-lg font-semibold text-slate-800 dark:text-slate-100">Transcripción</h3>
                    <button
                      type="button"
                      onClick={loadTranscript}
                      className="text-xs px-2 py-1 rounded-md border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800"
                    >
                      {isLoadingTranscript ? 'Cargando...' : 'Recargar'}
                    </button>
                  </div>
                  <div className="relative mb-3">
                    <Search size={14} className="absolute left-2.5 top-2.5 text-slate-400" />
                    <input
                      value={transcriptQuery}
                      onChange={(e) => setTranscriptQuery(e.target.value)}
                      placeholder="Buscar en transcript"
                      className="w-full rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 pl-8 pr-3 py-2 text-sm text-slate-700 dark:text-slate-200"
                    />
                  </div>
                  <div ref={transcriptListRef} className="space-y-2 max-h-[68vh] overflow-y-auto custom-scrollbar pr-1">
                    {filteredTranscript.map((seg) => {
                      const segmentKey = `${seg.segment_index}-${seg.start}`;
                      const isActive = Boolean(
                        activeTranscriptSegment
                        && Number(activeTranscriptSegment.start) === Number(seg.start)
                        && Number(activeTranscriptSegment.end) === Number(seg.end)
                      );
                      const clipRelativeStart = Math.max(0, Number(seg.start || 0) - Number(baseClipStart || 0));
                      const clipRelativeEnd = Math.max(0, Number(seg.end || 0) - Number(baseClipStart || 0));
                      return (
                        <div
                          key={segmentKey}
                          ref={(el) => {
                            if (!el) {
                              transcriptEntryRefs.current.delete(segmentKey);
                              return;
                            }
                            transcriptEntryRefs.current.set(segmentKey, el);
                          }}
                          onClick={() => {
                            seekTo(clipRelativeStart);
                          }}
                          className={`rounded-lg border p-2.5 cursor-pointer transition-colors ${
                            isActive
                              ? 'border-violet-400 bg-violet-50 dark:bg-violet-900/25 ring-1 ring-violet-300/70 dark:ring-violet-500/40'
                              : 'border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700/70'
                          }`}
                        >
                          <div className="flex items-center justify-between gap-2 mb-1">
                            <div className={`text-[11px] ${isActive ? 'text-violet-700 dark:text-violet-300 font-semibold' : 'text-slate-500'}`}>
                              {`${formatAbsoluteClock(seg.start)} - ${formatAbsoluteClock(seg.end)}`}
                            </div>
                            {isActive && (
                              <span className="text-[10px] px-1.5 py-0.5 rounded-full border border-violet-300 bg-violet-100 dark:border-violet-600 dark:bg-violet-900/35 text-violet-700 dark:text-violet-300">
                                Hablando
                              </span>
                            )}
                          </div>
                          <div className="text-[10px] text-slate-500 dark:text-slate-400 mb-1">
                            {`Clip ${formatTimelineTime(clipRelativeStart)} - ${formatTimelineTime(clipRelativeEnd)}`}
                          </div>
                          <p className="text-sm text-slate-700 dark:text-slate-200 leading-relaxed">{seg.text}</p>
                        </div>
                      );
                    })}
                    {!isLoadingTranscript && filteredTranscript.length === 0 && <p className="text-sm text-slate-500">No hay segmentos para este rango.</p>}
                  </div>
                </div>
              )}

              {section === 'captions' && (
                <div className="space-y-4">
                  <div>
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <h3 className="text-lg font-semibold text-zinc-800 dark:text-zinc-100">Subtítulos</h3>
                        {karaokeMode && (
                          <span className="text-[11px] px-2 py-0.5 rounded-full border border-violet-300 bg-violet-100 dark:border-violet-600 dark:bg-violet-900/35 text-violet-700 dark:text-violet-300">
                            Karaoke activo
                          </span>
                        )}
                      </div>
                      <button
                        type="button"
                        onClick={() => setShowCaptionSettings((v) => !v)}
                        className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border text-xs font-medium transition-colors ${
                          showCaptionSettings
                            ? 'border-violet-400 bg-violet-100 dark:bg-violet-900/25 text-violet-700 dark:text-violet-300'
                            : 'border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700'
                        }`}
                        title={showCaptionSettings ? 'Ocultar opciones de subtítulos' : 'Mostrar opciones de subtítulos'}
                      >
                        <Menu size={14} />
                        Opciones
                      </button>
                    </div>
                    <div className="mt-3 rounded-2xl border border-slate-200 dark:border-slate-700 bg-slate-50/80 dark:bg-slate-900/50 p-3">
                      <p className="text-xs font-semibold text-slate-500 dark:text-slate-400">Ajustes de subtítulos</p>
                      {!showCaptionSettings && (
                        <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-1">
                          Haz clic en <span className="font-medium">Opciones</span> para editar visibilidad, puntuación, emojis y karaoke.
                        </p>
                      )}
                      {showCaptionSettings && (
                        <div className="space-y-2 mt-3">
                          <SettingToggle label="Mostrar subtítulos" checked={captionsOn} onChange={() => setCaptionsOn((v) => !v)} />
                          <SettingToggle label="Puntuación" checked={punctuationOn} onChange={() => setPunctuationOn((v) => !v)} />
                          <SettingToggle label="Emoji" checked={emojiOn} onChange={() => setEmojiOn((v) => !v)} />
                          <SettingToggle label="Modo karaoke" checked={karaokeMode} onChange={() => setKaraokeMode((v) => !v)} />
                        </div>
                      )}
                    </div>
                  </div>

                  <div>
                    <p className="text-xs font-semibold text-zinc-500 mb-2">Presets</p>
                    <div className="grid grid-cols-2 lg:grid-cols-3 gap-2">
                      {CAPTION_PRESETS.map((preset) => {
                        const previewBg = preset?.preview?.bg || 'linear-gradient(145deg, #111827 0%, #1f2937 100%)';
                        const highlightColor = preset?.preview?.highlightColor || ACTIVE_WORD_COLOR;
                        const highlightWordIndex = Number.isFinite(preset?.preview?.highlightWordIndex)
                          ? Number(preset.preview.highlightWordIndex)
                          : -1;
                        let wordCounter = 0;
                        const sampleLines = String(preset.sample || '').split('\n').filter(Boolean);
                        const previewFontSize = Math.max(11, Math.min(18, Math.round((preset.style?.fontSize || 24) * 0.4)));
                        const previewStroke = Math.max(0, Number(preset.style?.strokeWidth || 0) * 0.65);
                        const sampleBoxColor = Number(preset.style?.boxOpacity || 0) > 0
                          ? toRgba(preset.style?.boxColor || '#000000', Math.min(90, Number(preset.style?.boxOpacity || 0)))
                          : 'transparent';
                        return (
                          <button
                            key={preset.id}
                            type="button"
                            onClick={() => applyPreset(preset.id)}
                            className={`rounded-xl border p-2 text-left transition-colors ${selectedPreset === preset.id
                              ? 'border-violet-400 bg-violet-50 dark:bg-violet-500/10 shadow-[0_0_0_1px_rgba(139,92,246,0.2)]'
                              : 'border-black/10 dark:border-white/10 bg-white/70 dark:bg-white/5 hover:bg-white dark:hover:bg-white/10'
                            }`}
                            title={`Aplicar preset ${preset.name}`}
                          >
                            <div
                              className="relative rounded-lg h-[86px] overflow-hidden border border-white/10"
                              style={{ background: previewBg }}
                            >
                              <div className="absolute inset-0 bg-[radial-gradient(circle_at_25%_20%,rgba(255,255,255,0.12),transparent_45%)]" />
                              <div className={`absolute inset-x-2 ${preset.style?.position === 'top' ? 'top-2' : preset.style?.position === 'middle' ? 'top-1/2 -translate-y-1/2' : 'bottom-2'} text-center`}>
                                <span
                                  className="inline-block rounded-md px-1.5 py-0.5 leading-tight"
                                  style={{
                                    fontFamily: preset.style?.fontFamily || 'Montserrat',
                                    fontSize: `${previewFontSize}px`,
                                    fontWeight: preset.style?.bold ? 700 : 500,
                                    color: preset.style?.fontColor || '#FFFFFF',
                                    textShadow: `0 0 ${previewStroke}px ${preset.style?.strokeColor || '#000000'}`,
                                    backgroundColor: sampleBoxColor
                                  }}
                                >
                                  {sampleLines.map((line, lineIdx) => {
                                    const words = String(line).split(/\s+/).filter(Boolean);
                                    return (
                                      <div key={`${preset.id}-sample-line-${lineIdx}`}>
                                        {words.map((word, idx) => {
                                          const currentWordIndex = wordCounter;
                                          wordCounter += 1;
                                          const isHighlight = currentWordIndex === highlightWordIndex;
                                          return (
                                            <span
                                              key={`${preset.id}-sample-word-${lineIdx}-${idx}`}
                                              style={isHighlight ? { color: highlightColor } : undefined}
                                            >
                                              {word}
                                              {idx < words.length - 1 ? ' ' : ''}
                                            </span>
                                          );
                                        })}
                                      </div>
                                    );
                                  })}
                                </span>
                              </div>
                            </div>
                            <div className="mt-1.5 text-[11px] font-semibold text-zinc-700 dark:text-zinc-100 truncate">{preset.name}</div>
                          </button>
                        );
                      })}
                    </div>
                  </div>

                  <div>
                    <p className="text-xs font-semibold text-zinc-500 mb-2">Posición</p>
                    <div className="grid grid-cols-3 gap-2">
                      {['top', 'middle', 'bottom'].map((opt) => (
                        <button
                          key={opt}
                          type="button"
                          onClick={() => {
                            setPosition(opt);
                            setSavedPulse(false);
                          }}
                          className={`rounded-lg px-2 py-2 text-xs border capitalize ${position === opt
                            ? 'border-violet-400 bg-violet-100 dark:bg-violet-900/20 text-violet-700 dark:text-violet-300'
                            : 'border-black/10 dark:border-white/10 text-zinc-600 dark:text-zinc-300'
                          }`}
                        >
                          {opt === 'top' ? 'Arriba' : opt === 'middle' ? 'Centro' : 'Abajo'}
                        </button>
                      ))}
                    </div>
                    <p className="mt-2 text-[11px] text-zinc-500">Tip: arrastra el subtítulo sobre el video para ubicarlo con el mouse.</p>
                  </div>

                  <div className="space-y-2">
                    <label className="text-xs text-zinc-600 dark:text-zinc-300 block">
                      Mover horizontal ({Math.round(effectiveCaptionOffsetX)}% efectivo)
                      <input
                        type="range"
                        min="-100"
                        max="100"
                        step="1"
                        value={captionOffsetX}
                        onChange={(e) => {
                          setCaptionOffsetX(clamp(Number(e.target.value || 0), -100, 100));
                          setSavedPulse(false);
                        }}
                        className="w-full mt-2"
                      />
                    </label>
                    <label className="text-xs text-zinc-600 dark:text-zinc-300 block">
                      Mover vertical ({Math.round(effectiveCaptionOffsetY)}% efectivo)
                      <input
                        type="range"
                        min="-100"
                        max="100"
                        step="1"
                        value={captionOffsetY}
                        onChange={(e) => {
                          setCaptionOffsetY(clamp(Number(e.target.value || 0), -100, 100));
                          setSavedPulse(false);
                        }}
                        className="w-full mt-2"
                      />
                    </label>
                    <button
                      type="button"
                      onClick={() => {
                        setCaptionOffsetX(0);
                        setCaptionOffsetY(0);
                        setSavedPulse(false);
                      }}
                      className="w-full rounded-lg border border-black/10 dark:border-white/10 px-3 py-2 text-xs text-zinc-600 dark:text-zinc-200 hover:bg-black/5 dark:hover:bg-white/5"
                    >
                      Reset posición de subtítulo
                    </button>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <label className="text-xs text-zinc-600 dark:text-zinc-300">Tamaño
                      <input type="number" min="12" max="84" value={fontSize} onChange={(e) => setFontSize(Number(e.target.value || 24))} className="mt-1 w-full rounded-md border border-black/10 dark:border-white/10 bg-white/80 dark:bg-black/20 p-2 text-sm" />
                    </label>
                    <label className="text-xs text-zinc-600 dark:text-zinc-300">Fuente
                      <input value={fontFamily} onChange={(e) => setFontFamily(e.target.value)} className="mt-1 w-full rounded-md border border-black/10 dark:border-white/10 bg-white/80 dark:bg-black/20 p-2 text-sm" />
                    </label>
                    <label className="text-xs text-zinc-600 dark:text-zinc-300">Color texto
                      <input type="color" value={fontColor} onChange={(e) => setFontColor(e.target.value)} className="mt-1 h-10 w-full rounded-md border border-black/10 dark:border-white/10 bg-white/80 dark:bg-black/20 p-1" />
                    </label>
                    <label className="text-xs text-zinc-600 dark:text-zinc-300">Color contorno
                      <input type="color" value={strokeColor} onChange={(e) => setStrokeColor(e.target.value)} className="mt-1 h-10 w-full rounded-md border border-black/10 dark:border-white/10 bg-white/80 dark:bg-black/20 p-1" />
                    </label>
                    <label className="text-xs text-zinc-600 dark:text-zinc-300">Grosor contorno
                      <input type="number" min="0" max="8" value={strokeWidth} onChange={(e) => setStrokeWidth(Number(e.target.value || 0))} className="mt-1 w-full rounded-md border border-black/10 dark:border-white/10 bg-white/80 dark:bg-black/20 p-2 text-sm" />
                    </label>
                    <label className="text-xs text-zinc-600 dark:text-zinc-300">Caja (%)
                      <input type="number" min="0" max="100" value={boxOpacity} onChange={(e) => setBoxOpacity(Number(e.target.value || 0))} className="mt-1 w-full rounded-md border border-black/10 dark:border-white/10 bg-white/80 dark:bg-black/20 p-2 text-sm" />
                    </label>
                  </div>
                  <label className="inline-flex items-center gap-2 text-xs text-zinc-600 dark:text-zinc-300">
                    <input type="checkbox" checked={bold} onChange={(e) => setBold(e.target.checked)} /> Negrita
                  </label>
                </div>
              )}

              {section === 'subtitle_edit' && (
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-lg font-semibold text-zinc-800 dark:text-zinc-100">Editar subtítulos</h3>
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={autoSuggestEmojis}
                        className="text-xs px-2 py-1 rounded-md border border-emerald-300 bg-emerald-100/80 dark:border-emerald-700 dark:bg-emerald-900/20 text-emerald-700 dark:text-emerald-300 hover:bg-emerald-100 dark:hover:bg-emerald-900/30 inline-flex items-center gap-1"
                        title="Auto-sugerencia local por contenido"
                      >
                        <Sparkles size={12} />
                        IA local
                      </button>
                      <button
                        type="button"
                        onClick={loadSrt}
                        className="text-xs px-2 py-1 rounded-md border border-black/10 dark:border-white/10 text-zinc-600 dark:text-zinc-300 hover:bg-black/5 dark:hover:bg-white/5"
                      >
                        {isLoadingSrt ? 'Cargando...' : 'Recargar SRT'}
                      </button>
                    </div>
                  </div>
                  {emojiSuggestFeedback && (
                    <div className="mb-2 rounded-md border border-emerald-300 bg-emerald-50/90 dark:border-emerald-700 dark:bg-emerald-900/20 px-2.5 py-1.5 text-[11px] text-emerald-700 dark:text-emerald-300">
                      {emojiSuggestFeedback}
                    </div>
                  )}
                  <div className="relative mb-3">
                    <Search size={14} className="absolute left-2.5 top-2.5 text-zinc-400" />
                    <input
                      value={subtitleSearch}
                      onChange={(e) => setSubtitleSearch(e.target.value)}
                      placeholder="Buscar subtítulo"
                      className="w-full rounded-lg border border-black/10 dark:border-white/10 bg-white/80 dark:bg-black/20 pl-8 pr-3 py-2 text-sm"
                    />
                  </div>
                  <div ref={subtitleListRef} className="space-y-2 max-h-[68vh] overflow-y-auto custom-scrollbar pr-1">
                    {filteredSubtitleEntries.map((entry) => (
                      <div
                        key={entry.id}
                        ref={(el) => {
                          if (!el) {
                            subtitleEntryRefs.current.delete(entry.id);
                            return;
                          }
                          subtitleEntryRefs.current.set(entry.id, el);
                        }}
                        onClick={() => {
                          const entryStart = Number(entry?.start || 0);
                          seekTo(entryStart);
                        }}
                        className={`rounded-lg border p-2.5 transition-colors cursor-pointer ${
                          activeSubtitleEntry?.id === entry.id
                            ? 'border-violet-400 bg-violet-50 dark:bg-violet-900/25 ring-1 ring-violet-300/70 dark:ring-violet-500/40'
                            : 'border-black/10 dark:border-white/10 bg-white/70 dark:bg-black/20'
                        }`}
                      >
                        {(() => {
                          const suggestedEmoji = suggestEmojiForText(entry.text);
                          return (
                            <>
                        <div className="flex items-center justify-between gap-2 mb-1">
                          <div className="flex items-center gap-2 min-w-0">
                            <span className={`text-[11px] ${activeSubtitleEntry?.id === entry.id ? 'text-violet-700 dark:text-violet-300 font-semibold' : 'text-zinc-500'}`}>
                              {`${formatSrtTime(entry.start)} - ${formatSrtTime(entry.end)}`}
                            </span>
                            {activeSubtitleEntry?.id === entry.id && (
                              <span className="text-[10px] px-1.5 py-0.5 rounded-full border border-violet-300 bg-violet-100 dark:border-violet-600 dark:bg-violet-900/35 text-violet-700 dark:text-violet-300">
                                Reproduciendo
                              </span>
                            )}
                          </div>
                          <div className="flex items-center gap-1.5">
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                setEmojiPickerForId((prev) => (prev === entry.id ? '' : entry.id));
                              }}
                              className={`text-[11px] px-2 py-1 rounded-md border ${emojiPickerForId === entry.id
                                ? 'border-violet-400 bg-violet-100 dark:bg-violet-500/15 text-violet-700 dark:text-violet-300'
                                : 'border-black/10 dark:border-white/10 text-zinc-600 dark:text-zinc-300'
                              }`}
                              title="Añadir emoji"
                            >
                              {entry.emoji ? `Emoji ${entry.emoji}` : 'Emoji'}
                            </button>
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                onSubtitleToggleEmphasis(entry.id);
                              }}
                              className={`text-[11px] px-2 py-1 rounded-md border ${entry.emphasize
                                ? 'border-amber-400 bg-amber-100 dark:bg-amber-500/15 text-amber-700 dark:text-amber-300'
                                : 'border-black/10 dark:border-white/10 text-zinc-600 dark:text-zinc-300'
                              }`}
                            >
                              Énfasis
                            </button>
                            {suggestedEmoji && (!entry.emoji || entry.emoji !== suggestedEmoji) && (
                              <button
                                type="button"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  onSubtitleEntryEmojiChange(entry.id, suggestedEmoji);
                                }}
                                className="text-[11px] px-2 py-1 rounded-md border border-emerald-300 bg-emerald-100/70 dark:border-emerald-700 dark:bg-emerald-900/20 text-emerald-700 dark:text-emerald-300"
                                title="Sugerencia local por contenido"
                              >
                                Sugerir {suggestedEmoji}
                              </button>
                            )}
                          </div>
                        </div>
                        {emojiPickerForId === entry.id && (
                          <div className="mb-2 rounded-md border border-violet-200 bg-violet-50/80 dark:border-violet-800 dark:bg-violet-900/10 p-2">
                            <div className="flex flex-wrap gap-1.5">
                              <button
                                type="button"
                                onMouseDown={(e) => e.stopPropagation()}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  onSubtitleEntryEmojiChange(entry.id, '');
                                  setEmojiPickerForId('');
                                }}
                                className="text-[11px] px-2 py-1 rounded-md border border-black/10 dark:border-white/10 bg-white/70 dark:bg-black/20 text-zinc-600 dark:text-zinc-300"
                              >
                                Sin emoji
                              </button>
                              {SUBTITLE_EMOJIS.map((emoji) => (
                                <button
                                  key={`${entry.id}-${emoji}`}
                                  type="button"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    onSubtitleEntryEmojiChange(entry.id, emoji);
                                    setEmojiPickerForId('');
                                  }}
                                  onMouseDown={(e) => e.stopPropagation()}
                                  className={`w-8 h-8 rounded-md border text-base ${entry.emoji === emoji
                                    ? 'border-violet-400 bg-violet-100 dark:bg-violet-900/20'
                                    : 'border-black/10 dark:border-white/10 bg-white/70 dark:bg-black/20'
                                  }`}
                                >
                                  {emoji}
                                </button>
                              ))}
                            </div>
                          </div>
                        )}
                        <textarea
                          value={entry.text}
                          onChange={(e) => onSubtitleEntryChange(entry.id, e.target.value)}
                          onClick={(e) => e.stopPropagation()}
                          rows={2}
                          className="w-full rounded-md border border-black/10 dark:border-white/10 bg-white dark:bg-black/20 p-2 text-sm text-zinc-700 dark:text-zinc-200"
                        />
                            </>
                          );
                        })()}
                      </div>
                    ))}
                    {!isLoadingSrt && filteredSubtitleEntries.length === 0 && (
                      <p className="text-sm text-zinc-500">No hay líneas de subtítulo para editar.</p>
                    )}
                  </div>
                </div>
              )}

              {section === 'layout' && (
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold text-zinc-800 dark:text-zinc-100">Editar layout</h3>
                  <p className="text-xs text-zinc-500">Ajusta formato/rango y elige entre encuadre manual o auto smart reframe por escena.</p>

                  <div>
                    <p className="text-xs font-semibold text-zinc-500 mb-2">Modo de layout</p>
                    <div className="grid grid-cols-2 gap-2">
                      <button
                        type="button"
                        onClick={() => {
                          setLayoutMode('single');
                          setSavedPulse(false);
                        }}
                        className={`rounded-lg px-3 py-2 text-sm border ${layoutMode === 'single'
                          ? 'border-violet-400 bg-violet-100 dark:bg-violet-900/20 text-violet-700 dark:text-violet-300'
                          : 'border-black/10 dark:border-white/10 text-zinc-700 dark:text-zinc-200'
                        }`}
                      >
                        Single
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          setLayoutMode('split');
                          setLayoutAutoSmart(false);
                          setLayoutFitMode('cover');
                          setSavedPulse(false);
                        }}
                        className={`rounded-lg px-3 py-2 text-sm border ${layoutMode === 'split'
                          ? 'border-violet-400 bg-violet-100 dark:bg-violet-900/20 text-violet-700 dark:text-violet-300'
                          : 'border-black/10 dark:border-white/10 text-zinc-700 dark:text-zinc-200'
                        }`}
                      >
                        Split (2 personas)
                      </button>
                    </div>
                    <p className="mt-1 text-[11px] text-zinc-500">
                      {layoutMode === 'split'
                        ? 'Split usa dos paneles del mismo video con paneo independiente por panel.'
                        : 'Single mantiene un único encuadre.'}
                    </p>
                  </div>

                  {layoutMode === 'single' && (
                    <SettingToggle
                      label="Auto smart reframe (beta)"
                      checked={layoutAutoSmart}
                      onChange={() => {
                        setLayoutAutoSmart((v) => !v);
                        setSavedPulse(false);
                      }}
                    />
                  )}

                  <div>
                    <p className="text-xs font-semibold text-zinc-500 mb-2">Formato</p>
                    <div className="grid grid-cols-2 gap-2">
                      {['9:16', '16:9'].map((ratio) => (
                        <button
                          key={ratio}
                          type="button"
                          onClick={() => setLayoutAspect(ratio)}
                          className={`rounded-lg px-3 py-2 text-sm border ${layoutAspect === ratio
                            ? 'border-violet-400 bg-violet-100 dark:bg-violet-900/20 text-violet-700 dark:text-violet-300'
                            : 'border-black/10 dark:border-white/10 text-zinc-700 dark:text-zinc-200'
                          }`}
                        >
                          {ratio}
                        </button>
                      ))}
                    </div>
                  </div>

                  {layoutMode === 'single' && !layoutAutoSmart && (
                    <>
                      <div>
                        <p className="text-xs font-semibold text-zinc-500 mb-2">Ajuste de video</p>
                        <div className="grid grid-cols-2 gap-2">
                          {['cover', 'contain'].map((mode) => (
                            <button
                              key={mode}
                              type="button"
                              onClick={() => setLayoutFitMode(mode)}
                              className={`rounded-lg px-3 py-2 text-sm border capitalize ${layoutFitMode === mode
                                ? 'border-violet-400 bg-violet-100 dark:bg-violet-900/20 text-violet-700 dark:text-violet-300'
                                : 'border-black/10 dark:border-white/10 text-zinc-700 dark:text-zinc-200'
                              }`}
                            >
                              {mode === 'cover' ? 'Cover' : 'Contain'}
                            </button>
                          ))}
                        </div>
                      </div>

                      <label className="text-xs text-zinc-600 dark:text-zinc-300 block">
                        Zoom ({layoutZoom.toFixed(2)}x)
                        <input
                          type="range"
                          min="0.5"
                          max="2.5"
                          step="0.01"
                          value={layoutZoom}
                          onChange={(e) => setLayoutZoom(clamp(Number(e.target.value || 1), 0.5, 2.5))}
                          className="w-full mt-2"
                        />
                      </label>

                      <label className="text-xs text-zinc-600 dark:text-zinc-300 block">
                        Mover horizontal ({Math.round(effectiveLayoutOffsetX)}% efectivo)
                        <input
                          type="range"
                          min="-100"
                          max="100"
                          step="1"
                          value={layoutOffsetX}
                          onChange={(e) => setLayoutOffsetX(clamp(Number(e.target.value || 0), -100, 100))}
                          className="w-full mt-2"
                        />
                      </label>

                      <label className="text-xs text-zinc-600 dark:text-zinc-300 block">
                        Mover vertical ({Math.round(effectiveLayoutOffsetY)}% efectivo)
                        <input
                          type="range"
                          min="-100"
                          max="100"
                          step="1"
                          value={layoutOffsetY}
                          onChange={(e) => setLayoutOffsetY(clamp(Number(e.target.value || 0), -100, 100))}
                          className="w-full mt-2"
                        />
                      </label>

                      <button
                        type="button"
                        onClick={() => {
                          setLayoutFitMode('cover');
                          setLayoutZoom(1);
                          setLayoutOffsetX(0);
                          setLayoutOffsetY(0);
                        }}
                        className="w-full rounded-lg border border-black/10 dark:border-white/10 px-3 py-2 text-sm text-zinc-600 dark:text-zinc-200 hover:bg-black/5 dark:hover:bg-white/5"
                      >
                        Reset encuadre manual
                      </button>
                    </>
                  )}

                  {layoutMode === 'split' && (
                    <>
                      <p className="text-[11px] text-zinc-500">
                        {layoutAspect === '9:16'
                          ? 'Split activo: panel superior + panel inferior.'
                          : 'Split activo: panel izquierdo + panel derecho.'}
                      </p>

                      <div className="space-y-3">
                        <div className="rounded-xl border border-black/10 dark:border-white/10 p-3 space-y-2">
                          <p className="text-xs font-semibold text-zinc-600 dark:text-zinc-300">Panel A</p>
                          <label className="text-xs text-zinc-600 dark:text-zinc-300 block">
                            Zoom A ({layoutSplitZoomA.toFixed(2)}x)
                            <input
                              type="range"
                              min="0.5"
                              max="2.5"
                              step="0.01"
                              value={layoutSplitZoomA}
                              onChange={(e) => setLayoutSplitZoomA(clamp(Number(e.target.value || 1), 0.5, 2.5))}
                              className="w-full mt-2"
                            />
                          </label>
                          <label className="text-xs text-zinc-600 dark:text-zinc-300 block">
                            Mover horizontal A ({Math.round(effectiveSplitOffsetAX)}% efectivo)
                            <input
                              type="range"
                              min="-100"
                              max="100"
                              step="1"
                              value={layoutSplitOffsetAX}
                              onChange={(e) => setLayoutSplitOffsetAX(clamp(Number(e.target.value || 0), -100, 100))}
                              className="w-full mt-2"
                            />
                          </label>
                          <label className="text-xs text-zinc-600 dark:text-zinc-300 block">
                            Mover vertical A ({Math.round(effectiveSplitOffsetAY)}% efectivo)
                            <input
                              type="range"
                              min="-100"
                              max="100"
                              step="1"
                              value={layoutSplitOffsetAY}
                              onChange={(e) => setLayoutSplitOffsetAY(clamp(Number(e.target.value || 0), -100, 100))}
                              className="w-full mt-2"
                            />
                          </label>
                        </div>

                        <div className="rounded-xl border border-black/10 dark:border-white/10 p-3 space-y-2">
                          <p className="text-xs font-semibold text-zinc-600 dark:text-zinc-300">Panel B</p>
                          <label className="text-xs text-zinc-600 dark:text-zinc-300 block">
                            Zoom B ({layoutSplitZoomB.toFixed(2)}x)
                            <input
                              type="range"
                              min="0.5"
                              max="2.5"
                              step="0.01"
                              value={layoutSplitZoomB}
                              onChange={(e) => setLayoutSplitZoomB(clamp(Number(e.target.value || 1), 0.5, 2.5))}
                              className="w-full mt-2"
                            />
                          </label>
                          <label className="text-xs text-zinc-600 dark:text-zinc-300 block">
                            Mover horizontal B ({Math.round(effectiveSplitOffsetBX)}% efectivo)
                            <input
                              type="range"
                              min="-100"
                              max="100"
                              step="1"
                              value={layoutSplitOffsetBX}
                              onChange={(e) => setLayoutSplitOffsetBX(clamp(Number(e.target.value || 0), -100, 100))}
                              className="w-full mt-2"
                            />
                          </label>
                          <label className="text-xs text-zinc-600 dark:text-zinc-300 block">
                            Mover vertical B ({Math.round(effectiveSplitOffsetBY)}% efectivo)
                            <input
                              type="range"
                              min="-100"
                              max="100"
                              step="1"
                              value={layoutSplitOffsetBY}
                              onChange={(e) => setLayoutSplitOffsetBY(clamp(Number(e.target.value || 0), -100, 100))}
                              className="w-full mt-2"
                            />
                          </label>
                        </div>
                      </div>

                      <button
                        type="button"
                        onClick={() => {
                          setLayoutSplitZoomA(1);
                          setLayoutSplitOffsetAX(0);
                          setLayoutSplitOffsetAY(0);
                          setLayoutSplitZoomB(1);
                          setLayoutSplitOffsetBX(0);
                          setLayoutSplitOffsetBY(0);
                        }}
                        className="w-full rounded-lg border border-black/10 dark:border-white/10 px-3 py-2 text-sm text-zinc-600 dark:text-zinc-200 hover:bg-black/5 dark:hover:bg-white/5"
                      >
                        Reset split
                      </button>
                    </>
                  )}

                  <div className="grid grid-cols-2 gap-3">
                    <label className="text-xs text-zinc-600 dark:text-zinc-300">Inicio (s)
                      <input type="number" step="0.1" value={layoutStart} onChange={(e) => setLayoutStart(Number(e.target.value || 0))} className="mt-1 w-full rounded-md border border-black/10 dark:border-white/10 bg-white/80 dark:bg-black/20 p-2 text-sm" />
                    </label>
                    <label className="text-xs text-zinc-600 dark:text-zinc-300">Fin (s)
                      <input type="number" step="0.1" value={layoutEnd} onChange={(e) => setLayoutEnd(Number(e.target.value || 0))} className="mt-1 w-full rounded-md border border-black/10 dark:border-white/10 bg-white/80 dark:bg-black/20 p-2 text-sm" />
                    </label>
                  </div>
                  <div className="grid grid-cols-2 gap-3">
                    <label className="text-xs text-zinc-600 dark:text-zinc-300">
                      Empezar antes (s)
                      <input
                        type="number"
                        min="0"
                        max="3"
                        step="0.1"
                        value={layoutPreRoll}
                        onChange={(e) => setLayoutPreRoll(clamp(Number(e.target.value || 0), 0, 3))}
                        className="mt-1 w-full rounded-md border border-black/10 dark:border-white/10 bg-white/80 dark:bg-black/20 p-2 text-sm"
                      />
                    </label>
                    <label className="text-xs text-zinc-600 dark:text-zinc-300">
                      Terminar después (s)
                      <input
                        type="number"
                        min="0"
                        max="3"
                        step="0.1"
                        value={layoutPostRoll}
                        onChange={(e) => setLayoutPostRoll(clamp(Number(e.target.value || 0), 0, 3))}
                        className="mt-1 w-full rounded-md border border-black/10 dark:border-white/10 bg-white/80 dark:bg-black/20 p-2 text-sm"
                      />
                    </label>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <button
                      type="button"
                      onClick={() => setLayoutPreRoll((v) => clamp(Number(v || 0) + 0.2, 0, 3))}
                      className="rounded-lg border border-black/10 dark:border-white/10 px-3 py-2 text-xs text-zinc-600 dark:text-zinc-200 hover:bg-black/5 dark:hover:bg-white/5"
                    >
                      +0.2s al inicio
                    </button>
                    <button
                      type="button"
                      onClick={() => setLayoutPostRoll((v) => clamp(Number(v || 0) + 0.2, 0, 3))}
                      className="rounded-lg border border-black/10 dark:border-white/10 px-3 py-2 text-xs text-zinc-600 dark:text-zinc-200 hover:bg-black/5 dark:hover:bg-white/5"
                    >
                      +0.2s al final
                    </button>
                  </div>
                  <p className="text-[11px] text-zinc-500">
                    Rango final estimado: {Math.max(0, Number(layoutStart || 0) - Number(layoutPreRoll || 0)).toFixed(2)}s - {(Math.max(0, Number(layoutEnd || 0)) + Math.max(0, Number(layoutPostRoll || 0))).toFixed(2)}s
                  </p>
                  <p className="text-[11px] text-zinc-500">
                    {layoutMode === 'split'
                      ? 'Tip: usa offsets opuestos entre Panel A y B para separar interlocutores.'
                      : layoutAutoSmart
                        ? 'Smart reframe: detecta personas por escena y decide recorte/letterbox automáticamente.'
                        : 'Tip: el recorte y el layout se aplican antes de subtítulos y música.'}
                  </p>
                </div>
              )}

              {section === 'music' && (
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold text-zinc-800 dark:text-zinc-100">Música</h3>
                  <label className="inline-flex items-center gap-2 text-sm text-zinc-700 dark:text-zinc-200">
                    <input type="checkbox" checked={musicEnabled} onChange={(e) => setMusicEnabled(e.target.checked)} />
                    Activar música de fondo
                  </label>

                  <div>
                    <label className="block text-xs text-zinc-500 mb-1">Archivo de música (mp3/wav/m4a)</label>
                    <input
                      type="file"
                      accept="audio/*"
                      onChange={(e) => setMusicFile(e.target.files?.[0] || null)}
                      className="w-full rounded-md border border-black/10 dark:border-white/10 bg-white/80 dark:bg-black/20 p-2 text-xs"
                    />
                    {musicFile && <p className="mt-1 text-[11px] text-zinc-500">{musicFile.name}</p>}
                  </div>

                  <label className="text-xs text-zinc-600 dark:text-zinc-300 block">Volumen música
                    <input
                      type="range"
                      min="0"
                      max="0.8"
                      step="0.01"
                      value={musicVolume}
                      onChange={(e) => setMusicVolume(Number(e.target.value))}
                      className="w-full mt-2"
                    />
                    <span className="text-[11px] text-zinc-500">{Math.round(musicVolume * 100)}%</span>
                  </label>

                  <label className="inline-flex items-center gap-2 text-sm text-zinc-700 dark:text-zinc-200">
                    <input type="checkbox" checked={duckVoice} onChange={(e) => setDuckVoice(e.target.checked)} />
                    Bajar música cuando habla la voz (ducking)
                  </label>
                </div>
              )}

              {error && (
                <div className="mt-4 rounded-lg border border-red-300 bg-red-50 text-red-700 dark:bg-red-900/20 dark:text-red-300 dark:border-red-700 px-3 py-2 text-sm">
                  {error}
                </div>
              )}
            </section>

            <section className="bg-slate-100 dark:bg-slate-900 p-4 md:p-6 flex flex-col">
              <div className="flex-1 rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-4 flex flex-col items-center justify-start gap-3">
                <div className="w-full px-1">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <p className="text-[15px] md:text-[20px] font-semibold leading-tight text-slate-900 dark:text-white" title={previewClipTitle}>
                        {previewClipTitle}
                      </p>
                      <p
                        className="mt-1.5 text-[10px] md:text-[11px] leading-snug text-slate-600 dark:text-slate-300"
                        style={{
                          display: '-webkit-box',
                          WebkitLineClamp: 2,
                          WebkitBoxOrient: 'vertical',
                          overflow: 'hidden'
                        }}
                        title={previewHeaderSubline}
                      >
                        {previewHeaderSubline}
                      </p>
                    </div>
                  </div>
                </div>
                <div
                  ref={previewSurfaceRef}
                  onMouseDown={startLayoutPan}
                  className={`w-full ${aspectRatioClass} rounded-md bg-black overflow-hidden relative mx-auto ${section === 'layout' && !layoutAutoSmart && !isSplitLayout ? (isPanningLayout ? 'cursor-grabbing' : 'cursor-grab') : ''}`}
                  title={section === 'layout' && !layoutAutoSmart && !isSplitLayout ? 'Arrastra para mover el encuadre manualmente' : undefined}
                >
                  {isSplitLayout ? (
                    <div className={`absolute inset-0 ${layoutAspect === '16:9' ? 'flex flex-row' : 'flex flex-col'}`}>
                      <div className={`${layoutAspect === '16:9' ? 'w-1/2 h-full' : 'w-full h-1/2'} overflow-hidden relative`}>
                        <video
                          ref={previewVideoRef}
                          src={previewVideoUrl || currentVideoUrl}
                          className="w-full h-full"
                          style={{
                            objectFit: layoutFitMode === 'contain' ? 'contain' : 'cover',
                            objectPosition: splitObjectPositionA,
                            transform: `scale(${Number(layoutSplitZoomA || 1)})`,
                            transformOrigin: 'center center'
                          }}
                          controls={section !== 'layout' && !isSplitLayout}
                          playsInline
                          onPlay={() => setPreviewPlaying(true)}
                          onPause={() => setPreviewPlaying(false)}
                          onTimeUpdate={(e) => {
                            const nextTime = Number(e?.currentTarget?.currentTime || 0);
                            setPreviewCurrentTime(nextTime);
                          }}
                          onSeeked={(e) => {
                            const nextTime = Number(e?.currentTarget?.currentTime || 0);
                            setPreviewCurrentTime(nextTime);
                          }}
                          onLoadedMetadata={(e) => {
                            const duration = Number(e?.currentTarget?.duration || 0);
                            const nextTime = Number(e?.currentTarget?.currentTime || 0);
                            setPreviewDuration(Number.isFinite(duration) ? duration : 0);
                            setPreviewCurrentTime(nextTime);
                          }}
                          onCanPlay={() => {
                            setVideoLoadError('');
                          }}
                          onLoadedData={() => {
                            setVideoLoadError('');
                          }}
                          onEnded={() => {
                            setPreviewPlaying(false);
                            setPreviewCurrentTime(0);
                          }}
                          onError={(e) => {
                            const videoEl = e?.currentTarget;
                            const hasFrame = Number(videoEl?.readyState || 0) >= 2 && Number(videoEl?.videoWidth || 0) > 0;
                            if (hasFrame) return;
                            setVideoLoadError('El navegador no pudo reproducir este archivo en la vista previa.');
                          }}
                        />
                      </div>
                      <div className={`${layoutAspect === '16:9' ? 'w-1/2 h-full' : 'w-full h-1/2'} overflow-hidden relative`}>
                        <video
                          ref={previewSplitVideoRef}
                          src={previewVideoUrl || currentVideoUrl}
                          className="w-full h-full pointer-events-none"
                          style={{
                            objectFit: layoutFitMode === 'contain' ? 'contain' : 'cover',
                            objectPosition: splitObjectPositionB,
                            transform: `scale(${Number(layoutSplitZoomB || 1)})`,
                            transformOrigin: 'center center'
                          }}
                          muted
                          playsInline
                          controls={false}
                          tabIndex={-1}
                          onLoadedData={() => {
                            // Keep this silent to avoid duplicate warning behavior.
                          }}
                        />
                      </div>
                      <div
                        className={`pointer-events-none absolute bg-white/45 ${layoutAspect === '16:9' ? 'top-0 bottom-0 left-1/2 w-px -translate-x-1/2' : 'left-0 right-0 top-1/2 h-px -translate-y-1/2'}`}
                      />
                    </div>
                  ) : (
                    <video
                      ref={previewVideoRef}
                      src={previewVideoUrl || currentVideoUrl}
                      className="w-full h-full"
                      style={{
                        objectFit: layoutFitMode === 'contain' ? 'contain' : 'cover',
                        objectPosition: layoutAutoSmart ? '50% 50%' : manualLayoutObjectPosition,
                        transform: layoutAutoSmart
                          ? 'scale(1)'
                          : `scale(${Number(layoutZoom || 1)})`,
                        transformOrigin: 'center center'
                      }}
                      controls={section !== 'layout' && !isSplitLayout}
                      playsInline
                      onPlay={() => setPreviewPlaying(true)}
                      onPause={() => setPreviewPlaying(false)}
                      onTimeUpdate={(e) => {
                        const nextTime = Number(e?.currentTarget?.currentTime || 0);
                        setPreviewCurrentTime(nextTime);
                      }}
                      onSeeked={(e) => {
                        const nextTime = Number(e?.currentTarget?.currentTime || 0);
                        setPreviewCurrentTime(nextTime);
                      }}
                      onLoadedMetadata={(e) => {
                        const duration = Number(e?.currentTarget?.duration || 0);
                        const nextTime = Number(e?.currentTarget?.currentTime || 0);
                        setPreviewDuration(Number.isFinite(duration) ? duration : 0);
                        setPreviewCurrentTime(nextTime);
                      }}
                      onCanPlay={() => {
                        setVideoLoadError('');
                      }}
                      onLoadedData={() => {
                        setVideoLoadError('');
                      }}
                      onEnded={() => {
                        setPreviewPlaying(false);
                        setPreviewCurrentTime(0);
                      }}
                      onError={(e) => {
                        const videoEl = e?.currentTarget;
                        const hasFrame = Number(videoEl?.readyState || 0) >= 2 && Number(videoEl?.videoWidth || 0) > 0;
                        if (hasFrame) return;
                        setVideoLoadError('El navegador no pudo reproducir este archivo en la vista previa.');
                      }}
                    />
                  )}

                  {section === 'layout' && !layoutAutoSmart && !isSplitLayout && (
                    <div className="absolute top-2 right-2 rounded-md bg-black/55 text-white text-[10px] px-2 py-1 pointer-events-none border border-white/20">
                      Arrastra para mover
                    </div>
                  )}
                  {captionDragEnabled && (
                    <div className="absolute top-2 left-2 rounded-md bg-black/55 text-white text-[10px] px-2 py-1 pointer-events-none border border-white/20">
                      Arrastra subtítulo
                    </div>
                  )}

                  {captionsOn && Boolean(previewText) && (
                    <div className="absolute inset-0 pointer-events-none">
                      <div
                        className="absolute px-6 text-center"
                        style={{
                          left: `calc(50% + ${effectiveCaptionOffsetX}%)`,
                          top: `calc(${captionAnchorTopPercent}% + ${effectiveCaptionOffsetY}%)`,
                          transform: 'translate(-50%, -50%)',
                          width: 'min(94%, 960px)'
                        }}
                      >
                        <span
                          className={`inline-block rounded-md px-2 py-1 ${captionDragEnabled ? (isDraggingCaption ? 'cursor-grabbing' : 'cursor-grab') : ''}`}
                          onMouseDown={startCaptionDrag}
                          style={{
                            fontSize: `${Math.max(12, Math.round(fontSize * 0.58))}px`,
                            fontFamily,
                            fontWeight: bold ? 700 : 400,
                            color: fontColor,
                            textShadow: `0 0 ${strokeWidth}px ${strokeColor}`,
                            backgroundColor: boxOpacity > 0 ? toRgba(boxColor, boxOpacity) : 'transparent',
                            pointerEvents: captionDragEnabled ? 'auto' : 'none',
                            userSelect: 'none'
                          }}
                        >
                          {previewEmoji && (
                            <span
                              className="block leading-none mb-1"
                              style={{ fontSize: `${Math.max(16, Math.round(fontSize * 0.66))}px` }}
                            >
                              {previewEmoji}
                            </span>
                          )}
                          {karaokeMode && karaokePreview ? (
                            karaokePreview.words.map((word, idx) => (
                              <span
                                key={`${word}-${idx}-${idx === karaokePreview.activeIndex ? 'active' : 'idle'}`}
                                className={idx === karaokePreview.activeIndex ? 'rounded px-1' : ''}
                                style={idx === karaokePreview.activeIndex
                                  ? {
                                      color: karaokePreview.activeColor,
                                      display: 'inline-block',
                                      marginRight: idx < karaokePreview.words.length - 1 ? '0.28em' : 0,
                                      transform: 'scale(1.16)',
                                      fontWeight: 800,
                                      textShadow: `0 0 ${Math.max(2, Number(strokeWidth || 0) + 1)}px ${strokeColor}`,
                                      animation: 'subtitleWordIn 180ms cubic-bezier(0.2, 0.7, 0.2, 1) both'
                                    }
                                  : {
                                      opacity: 0.92,
                                      display: 'inline-block',
                                      marginRight: idx < karaokePreview.words.length - 1 ? '0.28em' : 0,
                                      transform: 'translateY(0px) scale(1)',
                                      transition: 'opacity 120ms ease'
                                    }}
                              >
                                {word}
                              </span>
                            ))
                          ) : (
                            previewText
                          )}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
                {videoLoadError && (
                  <div className="mt-3 rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-700 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-300">
                    {videoLoadError}
                  </div>
                )}
              </div>

              <div className="mt-3 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 px-3 py-3 text-slate-700 dark:text-slate-200">
                <div className="flex items-center justify-between gap-3 text-sm mb-2.5">
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      className="w-8 h-8 rounded-md border border-slate-200 dark:border-slate-700 inline-flex items-center justify-center text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700"
                      title="Controles"
                    >
                      <SlidersHorizontal size={15} />
                    </button>
                    <button
                      type="button"
                      onClick={() => setTimelineZoom((z) => clamp(z - 0.15, TIMELINE_ZOOM_MIN, TIMELINE_ZOOM_MAX))}
                      className="w-8 h-8 rounded-md border border-slate-200 dark:border-slate-700 inline-flex items-center justify-center text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700"
                      title="Alejar timeline"
                    >
                      <ZoomOut size={15} />
                    </button>
                    <input
                      type="range"
                      min={String(TIMELINE_ZOOM_MIN)}
                      max={String(TIMELINE_ZOOM_MAX)}
                      step="0.05"
                      value={timelineZoom}
                      onChange={(e) => setTimelineZoom(clamp(Number(e.target.value || TIMELINE_ZOOM_DEFAULT), TIMELINE_ZOOM_MIN, TIMELINE_ZOOM_MAX))}
                      className="w-28 accent-primary"
                      title="Zoom timeline"
                    />
                    <span className="text-[11px] text-slate-500 dark:text-slate-400 tabular-nums min-w-[44px]">
                      {`${Math.round(timelineZoom * 100)}%`}
                    </span>
                    <button
                      type="button"
                      onClick={() => setTimelineZoom((z) => clamp(z + 0.15, TIMELINE_ZOOM_MIN, TIMELINE_ZOOM_MAX))}
                      className="w-8 h-8 rounded-md border border-slate-200 dark:border-slate-700 inline-flex items-center justify-center text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700"
                      title="Acercar timeline"
                    >
                      <ZoomIn size={15} />
                    </button>
                    <button
                      type="button"
                      onClick={() => setTimelineZoom(TIMELINE_ZOOM_DEFAULT)}
                      className="h-8 px-2.5 rounded-md border border-slate-200 dark:border-slate-700 text-xs font-semibold text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700"
                      title="Ajustar vista"
                    >
                      Ajustar
                    </button>
                    <div className="inline-flex items-center rounded-md border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 overflow-hidden">
                      <button
                        type="button"
                        onClick={() => setTimelineMode(TIMELINE_MODE_MINI)}
                        className={`h-8 px-2.5 text-xs font-semibold transition-colors ${
                          timelineMode === TIMELINE_MODE_MINI
                            ? 'bg-violet-100 text-violet-700 dark:bg-violet-900/30 dark:text-violet-300'
                            : 'text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800'
                        }`}
                        title="Vista mini"
                      >
                        Mini
                      </button>
                      <button
                        type="button"
                        onClick={() => setTimelineMode(TIMELINE_MODE_ADVANCED)}
                        className={`h-8 px-2.5 text-xs font-semibold transition-colors ${
                          timelineMode === TIMELINE_MODE_ADVANCED
                            ? 'bg-violet-100 text-violet-700 dark:bg-violet-900/30 dark:text-violet-300'
                            : 'text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800'
                        }`}
                        title="Vista avanzada"
                      >
                        Avanzado
                      </button>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-slate-600 dark:text-slate-300 tabular-nums">{formatTimelineTime(previewCurrentTime)}</span>
                    <button
                      type="button"
                      onClick={togglePreviewPlayback}
                      className="w-8 h-8 rounded-full border border-slate-200 dark:border-slate-700 inline-flex items-center justify-center hover:bg-slate-100 dark:hover:bg-slate-700"
                    >
                      {previewPlaying ? <Pause size={14} /> : <Play size={14} />}
                    </button>
                    <span className="text-slate-500 dark:text-slate-400 tabular-nums">{formatTimelineTime(timelineDuration)}</span>
                    <button
                      type="button"
                      onClick={cyclePlaybackRate}
                      className="min-w-[44px] px-2 h-8 rounded-md border border-slate-200 dark:border-slate-700 text-xs font-semibold hover:bg-slate-100 dark:hover:bg-slate-700"
                    >
                      {`${playbackRate}x`}
                    </button>
                    <button
                      type="button"
                      onClick={() => setSnapEnabled((v) => !v)}
                      className={`h-8 px-2.5 rounded-md border text-xs font-semibold inline-flex items-center gap-1.5 ${
                        snapEnabled
                          ? 'border-emerald-300 bg-emerald-100 text-emerald-700 dark:border-emerald-700/60 dark:bg-emerald-900/30 dark:text-emerald-300'
                          : 'border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700'
                      }`}
                      title="Snap magnético a transcript"
                    >
                      <Crosshair size={13} />
                      {snapEnabled ? 'Snap ON' : 'Snap OFF'}
                    </button>
                  </div>
                </div>

                <div className="mb-2 flex items-center justify-between text-[11px]">
                  <div className="text-slate-500 dark:text-slate-400">
                    Timeline de edición
                  </div>
                  <div className="flex items-center gap-1.5">
                    <button
                      type="button"
                      onClick={() => {
                        const snapped = snapToNearest(previewCurrentTime);
                        const nextRel = clamp(snapped, 0, Math.max(0, selectionEndRel - 0.2));
                        setLayoutStart(Number((baseClipStart + nextRel).toFixed(3)));
                      }}
                      className="px-2 py-1 rounded border border-slate-200 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-700"
                    >
                      Marcar IN
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        const snapped = snapToNearest(previewCurrentTime);
                        const nextRel = clamp(snapped, selectionStartRel + 0.2, timelineDuration);
                        setLayoutEnd(Number((baseClipStart + nextRel).toFixed(3)));
                      }}
                      className="px-2 py-1 rounded border border-slate-200 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-700"
                    >
                      Marcar OUT
                    </button>
                  </div>
                </div>

                <div ref={timelineViewportRef} className="overflow-x-auto custom-scrollbar">
                  <div
                    ref={timelineTrackRef}
                    className="relative select-none rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900/40 overflow-hidden cursor-pointer"
                    style={{ width: `${Math.max(65, Math.round(timelineZoom * 100))}%`, minWidth: '520px' }}
                    onClick={(e) => handleTimelinePointerSeek(e.clientX)}
                  >
                    <div className="px-3 pt-2 pb-1 border-b border-slate-200 dark:border-slate-700">
                      <div className="relative h-5">
                        {timelineTicks.map((tick) => {
                          const left = `${(tick / Math.max(0.001, timelineDuration)) * 100}%`;
                          return (
                            <div key={`tick-${tick}`} className="absolute top-0 -translate-x-1/2" style={{ left }}>
                              <div className="w-px h-2 bg-slate-300 dark:bg-slate-600 mx-auto" />
                              <div className="mt-0.5 text-[10px] text-slate-500 dark:text-slate-400 tabular-nums">
                                {`${Math.round(tick)}s`}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>

                    <div className="px-3 py-2 border-b border-slate-200 dark:border-slate-700">
                      <div className="relative h-8 rounded-md border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800/50">
                        <div
                          className="absolute inset-y-0 left-0 bg-slate-100 dark:bg-slate-800"
                          style={{ width: `${(selectionStartRel / Math.max(0.001, timelineDuration)) * 100}%` }}
                        />
                        <div
                          className="absolute inset-y-0 right-0 bg-slate-100 dark:bg-slate-800"
                          style={{ width: `${(1 - (selectionEndRel / Math.max(0.001, timelineDuration))) * 100}%` }}
                        />
                        <div
                          className="absolute inset-y-0 rounded-md border border-emerald-400 bg-emerald-200/50 dark:bg-emerald-500/20 cursor-grab"
                          style={{
                            left: `${(selectionStartRel / Math.max(0.001, timelineDuration)) * 100}%`,
                            width: `${((selectionEndRel - selectionStartRel) / Math.max(0.001, timelineDuration)) * 100}%`
                          }}
                          onMouseDown={(e) => startSelectionDrag(e, 'move')}
                          title="Rango de recorte"
                        >
                          <button
                            type="button"
                            className="absolute left-0 top-0 bottom-0 w-2 bg-emerald-500/80 cursor-ew-resize"
                            onMouseDown={(e) => startSelectionDrag(e, 'start')}
                            onClick={(e) => e.stopPropagation()}
                            aria-label="Mover inicio de recorte"
                          />
                          <button
                            type="button"
                            className="absolute right-0 top-0 bottom-0 w-2 bg-emerald-500/80 cursor-ew-resize"
                            onMouseDown={(e) => startSelectionDrag(e, 'end')}
                            onClick={(e) => e.stopPropagation()}
                            aria-label="Mover fin de recorte"
                          />
                        </div>
                      </div>
                    </div>

                    <div className="px-3 py-2 border-b border-slate-200 dark:border-slate-700">
                      <div className="relative h-10 rounded-md border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800/50 overflow-hidden">
                        {subtitleEntries
                          .filter((entry) => String(entry?.text || '').trim())
                          .map((entry) => {
                            const start = clamp(Number(entry.start || 0), 0, timelineDuration);
                            const end = clamp(Number(entry.end || start), start + 0.08, timelineDuration);
                            const left = (start / Math.max(0.001, timelineDuration)) * 100;
                            const width = Math.max(0.8, ((end - start) / Math.max(0.001, timelineDuration)) * 100);
                            return (
                              <div
                                key={`lane-${entry.id}`}
                                className="absolute top-1 bottom-1 rounded-md border border-violet-400/70 bg-violet-500/20 text-[10px] text-violet-800 dark:text-violet-200 px-2 flex items-center cursor-grab"
                                style={{ left: `${left}%`, width: `${width}%` }}
                                onMouseDown={(e) => startSubtitleDrag(e, entry, 'move')}
                                onDoubleClick={(e) => {
                                  e.stopPropagation();
                                  seekTo(start);
                                }}
                                title={entry.text}
                              >
                                <button
                                  type="button"
                                  className="absolute left-0 top-0 bottom-0 w-1.5 bg-violet-500/80 cursor-ew-resize"
                                  onMouseDown={(e) => startSubtitleDrag(e, entry, 'start')}
                                  onClick={(e) => e.stopPropagation()}
                                  aria-label="Mover inicio de subtítulo"
                                />
                                <span className="truncate">{entry.text}</span>
                                <button
                                  type="button"
                                  className="absolute right-0 top-0 bottom-0 w-1.5 bg-violet-500/80 cursor-ew-resize"
                                  onMouseDown={(e) => startSubtitleDrag(e, entry, 'end')}
                                  onClick={(e) => e.stopPropagation()}
                                  aria-label="Mover fin de subtítulo"
                                />
                              </div>
                            );
                          })}
                      </div>
                    </div>

                    {timelineMode === TIMELINE_MODE_ADVANCED && (
                      <>
                        <div className="px-3 py-2.5 border-b border-slate-200 dark:border-slate-700">
                          <div className="h-8 rounded-md bg-violet-100 dark:bg-violet-900/25 border border-violet-200 dark:border-violet-800 flex items-center px-3">
                            <Type size={14} className="text-violet-500 mr-2 shrink-0" />
                            <span className="text-sm text-violet-700 dark:text-violet-300 truncate">
                              {clip?.video_title_for_youtube_short || `Clip n.º ${clipIndex + 1}`}
                            </span>
                          </div>
                        </div>
                        <div className="px-3 py-3 border-b border-slate-200 dark:border-slate-700">
                          <div className="h-12 rounded-md bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 flex items-end gap-[2px] px-2">
                            {timelineDensityBars.map((amp, idx) => (
                              <div
                                key={`density-${idx}`}
                                className="flex-1 rounded-t-sm bg-slate-500/80 dark:bg-slate-400/80"
                                style={{ height: `${Math.max(7, amp * 100)}%` }}
                              />
                            ))}
                          </div>
                        </div>
                      </>
                    )}

                    <div
                      className="absolute top-0 bottom-0 w-[2px] bg-slate-800/85 dark:bg-white/90 pointer-events-none"
                      style={{ left: `${(clamp(previewCurrentTime, 0, timelineDuration) / Math.max(0.001, timelineDuration)) * 100}%` }}
                    />
                    <div
                      className="absolute top-1.5 w-3 h-3 rounded-full bg-slate-800 dark:bg-white border-2 border-white dark:border-slate-900 pointer-events-none -translate-x-1/2"
                      style={{ left: `${(clamp(previewCurrentTime, 0, timelineDuration) / Math.max(0.001, timelineDuration)) * 100}%` }}
                    />
                  </div>
                </div>

                <div className="mt-2 text-[11px] text-slate-500 dark:text-slate-400">
                  Clip n.º {clipIndex + 1}
                </div>
              </div>
            </section>
          </div>
        </div>
      </div>
    </div>
  );
}
