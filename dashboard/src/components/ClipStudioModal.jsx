import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { X, FileText, Captions, Type, LayoutTemplate, Music2, Search, Sparkles, Loader2, Play, Pause, Pencil, SlidersHorizontal, ZoomOut, ZoomIn, Crosshair, Menu, Lightbulb, Download, Languages, RotateCcw } from 'lucide-react';
import { apiFetch, getApiUrl } from '../config';
import SubtitleRenderer from './SubtitleRenderer';

const CAPTION_PRESETS = [
  {
    id: 'deep_diver',
    name: 'Deep Diver',
    sample: 'TO GET\nSTARTED',
    preview: {
      bg: 'linear-gradient(145deg, #0f172a 0%, #111827 60%, #1f2937 100%)',
      highlightColor: '#E2E8F0',
      highlightWordIndex: 2
    },
    style: {
      position: 'bottom',
      fontSize: 60,
      fontFamily: 'Montserrat',
      fontColor: '#FFFFFF',
      strokeColor: '#111827',
      strokeWidth: 0,
      bold: true,
      boxColor: '#111827',
      boxOpacity: 72,
      animation: 'slide'
    }
  },
  {
    id: 'karaoke_pro',
    name: 'Karaoke Pro',
    sample: 'TO GET\nSTARTED',
    preview: {
      bg: 'linear-gradient(145deg, #090f1d 0%, #111827 55%, #0b1020 100%)',
      highlightColor: '#39FF14',
      highlightWordIndex: 1
    },
    style: {
      position: 'bottom',
      fontSize: 60,
      fontFamily: 'Montserrat',
      fontColor: '#FFFFFF',
      strokeColor: '#000000',
      strokeWidth: 0,
      bold: true,
      boxColor: '#000000',
      boxOpacity: 0,
      animation: 'bounce'
    },
    karaokeMode: true
  },
  {
    id: 'mozi_pop',
    name: 'Mozi Pop',
    sample: 'TO GET\nSTARTED',
    preview: {
      bg: 'linear-gradient(145deg, #0b1220 0%, #111827 60%, #131925 100%)',
      highlightColor: '#22C55E',
      highlightWordIndex: 1
    },
    style: {
      position: 'bottom',
      fontSize: 60,
      fontFamily: 'Archivo Black',
      fontColor: '#FFFFFF',
      strokeColor: '#0A0A0A',
      strokeWidth: 0,
      bold: true,
      boxColor: '#000000',
      boxOpacity: 0,
      animation: 'pop'
    },
    karaokeMode: true
  },
  {
    id: 'think_media',
    name: 'Think Media',
    sample: 'TO GET\nSTARTED',
    preview: {
      bg: 'linear-gradient(145deg, #1c1917 0%, #292524 100%)',
      highlightColor: '#FACC15',
      highlightWordIndex: 1
    },
    style: {
      position: 'bottom',
      fontSize: 75,
      fontFamily: 'Bebas Neue',
      fontColor: '#FFFFFF',
      strokeColor: '#0A0A0A',
      strokeWidth: 5,
      bold: true,
      boxColor: '#000000',
      boxOpacity: 0,
      animation: 'pop'
    },
    karaokeMode: true
  },
  {
    id: 'highlighter_box',
    name: 'Highlighter Box',
    sample: 'TO GET\nSTARTED',
    preview: {
      bg: 'linear-gradient(145deg, #1e293b 0%, #0f172a 100%)',
      highlightColor: '#38BDF8',
      highlightWordIndex: 0
    },
    style: {
      position: 'bottom',
      fontSize: 55,
      fontFamily: 'Oswald',
      fontColor: '#E0F2FE',
      strokeColor: '#0F172A',
      strokeWidth: 0,
      bold: true,
      boxColor: '#1D4ED8',
      boxOpacity: 78,
      animation: 'slide'
    }
  },
  {
    id: 'white_card',
    name: 'Caja Blanca',
    sample: "Tal Wilkenfeld's\nJeff Beck\nAudition",
    preview: {
      bg: 'linear-gradient(145deg, #0f172a 0%, #111827 100%)',
      highlightColor: '#111827',
      highlightWordIndex: 1
    },
    style: {
      position: 'bottom',
      fontSize: 50,
      fontFamily: 'Montserrat',
      fontColor: '#111111',
      strokeColor: '#111111',
      strokeWidth: 0,
      bold: true,
      boxColor: '#FFFFFF',
      boxOpacity: 100,
      animation: 'none'
    }
  },
  {
    id: 'focus_bold',
    name: 'Focus',
    sample: 'TO GET\nSTARTED',
    preview: {
      bg: 'linear-gradient(145deg, #1a1a1a 0%, #101010 100%)',
      highlightColor: '#EAFB23',
      highlightWordIndex: 1
    },
    style: {
      position: 'bottom',
      fontSize: 60,
      fontFamily: 'Teko',
      fontColor: '#EAFB23',
      strokeColor: '#101010',
      strokeWidth: 0,
      bold: true,
      boxColor: '#000000',
      boxOpacity: 0,
      animation: 'bounce'
    }
  }
];

const DEFAULT_FONT_OPTIONS = [
  { value: 'Montserrat', label: 'Montserrat', available: true },
  { value: 'Anton', label: 'Anton', available: true },
  { value: 'Archivo Black', label: 'Archivo Black', available: true },
  { value: 'Bebas Neue', label: 'Bebas Neue', available: true },
  { value: 'Oswald', label: 'Oswald', available: true },
  { value: 'Teko', label: 'Teko', available: true },
  { value: 'Arial', label: 'Arial', available: true },
  { value: 'Verdana', label: 'Verdana', available: true }
];

const SECTION_ITEMS = [
  { id: 'transcript', label: 'Transcripción', icon: FileText },
  { id: 'captions', label: 'Subtítulos', icon: Captions },
  { id: 'subtitle_edit', label: 'Editar subtítulos', icon: Type },
  { id: 'viral_hook', label: 'Hook Viral', icon: Sparkles },
  { id: 'layout', label: 'Editar layout', icon: LayoutTemplate },
  { id: 'music', label: 'Música', icon: Music2 },
  { id: 'dubbing', label: 'Doblaje', icon: Languages }
];

const DEFAULT_DUBBING_LANGUAGES = {
  en: 'English',
  es: 'Spanish',
  fr: 'French',
  de: 'German',
  it: 'Italian',
  pt: 'Portuguese',
  pl: 'Polish',
  hi: 'Hindi',
  ja: 'Japanese',
  ko: 'Korean',
  zh: 'Chinese',
  ar: 'Arabic',
  ru: 'Russian',
  tr: 'Turkish',
  nl: 'Dutch',
  sv: 'Swedish',
  id: 'Indonesian',
  fil: 'Filipino',
  ms: 'Malay',
  vi: 'Vietnamese',
  th: 'Thai',
  uk: 'Ukrainian',
  el: 'Greek',
  cs: 'Czech',
  fi: 'Finnish',
  ro: 'Romanian',
  da: 'Danish',
  bg: 'Bulgarian',
  hr: 'Croatian',
  sk: 'Slovak',
  ta: 'Tamil'
};

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
const SPEAKER_COLOR_PALETTE = ['#39FF14', '#00E5FF', '#FFC400', '#FF4D4D', '#B266FF', '#22D3EE', '#F97316', '#F43F5E'];
const BRAND_KIT_STORAGE_KEY = 'brandKitV1';

const normalizeSubtitleFontFamily = (value) => {
  const key = String(value || '').trim().toLowerCase().replace(/\s+/g, ' ');
  if (key.startsWith('montserrat')) return 'Montserrat';
  if (key.startsWith('anton') || key === 'impact' || key === 'arial black') return 'Anton';
  if (key.startsWith('archivo black')) return 'Archivo Black';
  if (key.startsWith('bebas neue') || key === 'bebas') return 'Bebas Neue';
  if (key.startsWith('oswald')) return 'Oswald';
  if (key.startsWith('teko')) return 'Teko';
  if (key === 'arial') return 'Arial';
  if (key === 'verdana') return 'Verdana';
  return 'Anton';
};

const clampNum = (value, fallback, min, max) => {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, n));
};

const readStoredBrandKitSubtitleStyle = () => {
  if (typeof window === 'undefined') return null;
  try {
    const raw = window.localStorage.getItem(BRAND_KIT_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== 'object') return null;
    return {
      position: ['top', 'middle', 'bottom'].includes(parsed.subtitle_position) ? parsed.subtitle_position : null,
      fontSize: clampNum(parsed.subtitle_font_size, 50, 12, 84),
      fontFamily: normalizeSubtitleFontFamily(parsed.subtitle_font_family),
      fontColor: String(parsed.subtitle_font_color || '#FFFFFF'),
      strokeColor: String(parsed.subtitle_stroke_color || '#000000'),
      strokeWidth: clampNum(parsed.subtitle_stroke_width, 3, 0, 8),
      bold: typeof parsed.subtitle_bold === 'boolean' ? parsed.subtitle_bold : true,
      boxColor: String(parsed.subtitle_box_color || '#000000'),
      boxOpacity: clampNum(parsed.subtitle_box_opacity, 0, 0, 100),
      karaokeMode: false,
      subtitleAnimation: 'none',
      speakerColorMode: false
    };
  } catch (_) {
    return null;
  }
};

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

const pickSpeakerColorForLabel = (label) => {
  const safe = normalizeEmojiText(label);
  if (!safe) return ACTIVE_WORD_COLOR;
  let digest = 0;
  for (let i = 0; i < safe.length; i += 1) {
    digest = ((digest * 33) + safe.charCodeAt(i)) % 1000003;
  }
  return SPEAKER_COLOR_PALETTE[Math.abs(digest) % SPEAKER_COLOR_PALETTE.length];
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

const SettingToggle = ({ label, checked, onChange, tooltip }) => (
  <div
    className="flex items-center justify-between rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 px-3 py-2.5"
    title={tooltip || undefined}
  >
    <span className="inline-flex items-center gap-1 text-sm text-slate-700 dark:text-slate-200">
      {label}
      {tooltip && (
        <span
          className="inline-flex h-4 w-4 items-center justify-center rounded-full border border-slate-300 dark:border-slate-600 text-[10px] text-slate-500 dark:text-slate-300 cursor-help"
          title={tooltip}
          aria-label={tooltip}
        >
          ?
        </span>
      )}
    </span>
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={`${label}: ${checked ? 'activado' : 'desactivado'}`}
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
    } catch (__) {
      return last;
    }
  }
};

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const getMinZoomForFitMode = (fitMode) => {
  const mode = String(fitMode || 'cover').toLowerCase();
  return mode === 'contain' || mode === 'blur' ? 0.3 : 1.0;
};
const LAYOUT_OFFSET_FACTOR = 1.0;
const LAYOUT_PAN_MIN_ZOOM = 1.06;
const LAYOUT_PAN_SENSITIVITY = 120;
const CAPTION_OFFSET_FACTOR = 0.35;
const CAPTION_CENTER_SNAP_THRESHOLD = 2.2;
const VIRAL_HOOK_DEFAULT_DURATION = 3.0;
const VIRAL_HOOK_FONT_SIZE_MIN = 12;
const VIRAL_HOOK_FONT_SIZE_MAX = 84;
const DEFAULT_VIRAL_HOOK_STYLE = {
  fontSize: 60,
  fontFamily: 'Montserrat',
  fontColor: '#FFFFFF',
  strokeColor: '#111827',
  strokeWidth: 0,
  bold: true,
  boxColor: '#111827',
  boxOpacity: 72,
  lineSpacing: 0
};
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

const resolveDefaultViralHookText = (clip, clipIndex, currentVideoUrl = '') => {
  const explicitHook = String(clip?.viral_hook_text || '').trim();
  if (explicitHook) return explicitHook;
  const socialTitle = String(clip?.video_title_for_youtube_short || '').trim();
  if (socialTitle) return socialTitle;
  const genericTitle = String(clip?.title || '').trim();
  if (genericTitle) return genericTitle;
  const filename = extractFilename(currentVideoUrl) || extractFilename(clip?.video_url || '');
  if (filename) return filename;
  return `Clip n.o ${Number(clipIndex || 0) + 1}`;
};

const resolveViralHookFontSize = (clip, fallback = DEFAULT_VIRAL_HOOK_STYLE.fontSize) => {
  const explicitSize = Number(clip?.viral_hook_font_size);
  if (Number.isFinite(explicitSize) && explicitSize > 0) {
    return clamp(explicitSize, VIRAL_HOOK_FONT_SIZE_MIN, VIRAL_HOOK_FONT_SIZE_MAX);
  }
  return clamp(Number(fallback || DEFAULT_VIRAL_HOOK_STYLE.fontSize), VIRAL_HOOK_FONT_SIZE_MIN, VIRAL_HOOK_FONT_SIZE_MAX);
};

const resolveViralHookStyle = (clip, fallbackStyle = DEFAULT_VIRAL_HOOK_STYLE) => {
  const style = fallbackStyle && typeof fallbackStyle === 'object' ? fallbackStyle : DEFAULT_VIRAL_HOOK_STYLE;
  const resolvedFontSize = resolveViralHookFontSize(clip, style.fontSize);
  const rawStrokeWidth = Number(clip?.viral_hook_stroke_width ?? style.strokeWidth ?? 0);
  const rawBoxOpacity = Number(clip?.viral_hook_box_opacity ?? style.boxOpacity ?? 0);
  const rawBold = (typeof clip?.viral_hook_bold === 'boolean')
    ? clip.viral_hook_bold
    : Boolean(style.bold);

  return {
    fontSize: resolvedFontSize,
    fontFamily: normalizeSubtitleFontFamily(clip?.viral_hook_font_family || style.fontFamily || DEFAULT_VIRAL_HOOK_STYLE.fontFamily),
    fontColor: String(clip?.viral_hook_font_color || style.fontColor || DEFAULT_VIRAL_HOOK_STYLE.fontColor),
    strokeColor: String(clip?.viral_hook_stroke_color || style.strokeColor || DEFAULT_VIRAL_HOOK_STYLE.strokeColor),
    strokeWidth: clamp(rawStrokeWidth, 0, 8),
    bold: Boolean(rawBold),
    boxColor: String(clip?.viral_hook_box_color || style.boxColor || DEFAULT_VIRAL_HOOK_STYLE.boxColor),
    boxOpacity: clamp(rawBoxOpacity, 0, 100)
  };
};

export default function ClipStudioModal({
  isOpen,
  standalone = false,
  onClose,
  jobId,
  clipIndex,
  clip,
  currentVideoUrl,
  onApplied,
  onClipPatched,
  fontCatalog = DEFAULT_FONT_OPTIONS,
  elevenLabsKey = ''
}) {
  const [section, setSection] = useState('transcript');
  const [isApplying, setIsApplying] = useState(false);
  const [applyAction, setApplyAction] = useState('apply');
  const [isRenderingFastPreview, setIsRenderingFastPreview] = useState(false);
  const [isLoadingTranscript, setIsLoadingTranscript] = useState(false);
  const [isLoadingSrt, setIsLoadingSrt] = useState(false);
  const [error, setError] = useState('');
  const [previewTitleOverride, setPreviewTitleOverride] = useState('');
  const [savedPulse, setSavedPulse] = useState(false);

  const [transcriptSegments, setTranscriptSegments] = useState([]);
  const [transcriptQuery, setTranscriptQuery] = useState('');
  const [transcriptPlainMode, setTranscriptPlainMode] = useState(false);
  const [transcriptSceneDescriptionsOn, setTranscriptSceneDescriptionsOn] = useState(true);

  const [captionsOn, setCaptionsOn] = useState(true);
  const [showCaptionSettings, setShowCaptionSettings] = useState(false);
  const [selectedPreset, setSelectedPreset] = useState(CAPTION_PRESETS[0].id);
  const [selectedViralHookPreset, setSelectedViralHookPreset] = useState('');
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
  const [subtitleAnimation, setSubtitleAnimation] = useState('none');
  const [speakerColorMode, setSpeakerColorMode] = useState(false);
  const [punctuationOn, setPunctuationOn] = useState(true);
  const [emojiOn, setEmojiOn] = useState(true);
  const [captionOffsetX, setCaptionOffsetX] = useState(clamp(Number(clip?.caption_offset_x || 0), -100, 100));
  const [captionOffsetY, setCaptionOffsetY] = useState(clamp(Number(clip?.caption_offset_y ?? 10), -100, 100));
  const [isDraggingCaption, setIsDraggingCaption] = useState(false);
  const [captionCenterGuides, setCaptionCenterGuides] = useState({ x: false, y: false });

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
  const initialLayoutFitMode = String(clip?.layout_fit_mode || 'cover').toLowerCase() === 'contain' ? 'contain' : 'cover';
  const initialLayoutZoomMin = getMinZoomForFitMode(initialLayoutFitMode);
  const [layoutFitMode, setLayoutFitMode] = useState(initialLayoutFitMode);
  const [layoutZoom, setLayoutZoom] = useState(clamp(Number(clip?.layout_zoom || 1), initialLayoutZoomMin, 2.5));
  const [layoutOffsetX, setLayoutOffsetX] = useState(clamp(Number(clip?.layout_offset_x || 0), -100, 100));
  const [layoutOffsetY, setLayoutOffsetY] = useState(clamp(Number(clip?.layout_offset_y || 0), -100, 100));
  const [layoutSplitZoomA, setLayoutSplitZoomA] = useState(clamp(Number(clip?.layout_split_zoom_a ?? clip?.layout_zoom ?? 1), initialLayoutZoomMin, 2.5));
  const [layoutSplitOffsetAX, setLayoutSplitOffsetAX] = useState(clamp(Number(clip?.layout_split_offset_a_x ?? clip?.layout_offset_x ?? 0), -100, 100));
  const [layoutSplitOffsetAY, setLayoutSplitOffsetAY] = useState(clamp(Number(clip?.layout_split_offset_a_y ?? clip?.layout_offset_y ?? 0), -100, 100));
  const [layoutSplitZoomB, setLayoutSplitZoomB] = useState(clamp(Number(clip?.layout_split_zoom_b ?? clip?.layout_zoom ?? 1), initialLayoutZoomMin, 2.5));
  const [layoutSplitOffsetBX, setLayoutSplitOffsetBX] = useState(clamp(Number(clip?.layout_split_offset_b_x ?? (-(Number(clip?.layout_offset_x || 0)))), -100, 100));
  const [layoutSplitOffsetBY, setLayoutSplitOffsetBY] = useState(clamp(Number(clip?.layout_split_offset_b_y ?? clip?.layout_offset_y ?? 0), -100, 100));
  const [isPanningLayout, setIsPanningLayout] = useState(false);
  const isSplitLayout = layoutMode === 'split';
  const layoutZoomMin = getMinZoomForFitMode(layoutFitMode);
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
  const captionFontOptions = useMemo(() => {
    const source = Array.isArray(fontCatalog) && fontCatalog.length > 0
      ? fontCatalog
      : DEFAULT_FONT_OPTIONS;
    const out = [];
    const seen = new Set();
    source.forEach((item) => {
      const value = String(item?.value || '').trim();
      if (!value || seen.has(value)) return;
      seen.add(value);
      out.push({
        value,
        label: String(item?.label || value).trim() || value,
        available: item?.available !== false
      });
    });
    return out.length > 0 ? out : DEFAULT_FONT_OPTIONS;
  }, [fontCatalog]);
  const resolvedElevenLabsKey = useMemo(() => {
    const direct = String(elevenLabsKey || '').trim();
    if (direct) return direct;
    if (typeof window === 'undefined') return '';
    try {
      return String(window.localStorage.getItem('elevenlabs_key') || '').trim();
    } catch (_) {
      return '';
    }
  }, [elevenLabsKey]);
  const hasElevenLabsKey = Boolean(resolvedElevenLabsKey);

  const handleLayoutAspectChange = useCallback((nextAspectRaw) => {
    const nextAspect = nextAspectRaw === '16:9' ? '16:9' : '9:16';
    if (nextAspect === layoutAspect) return false;
    setLayoutAspect(nextAspect);
    setSavedPulse(false);

    // Avoid carrying stale pan/zoom values between radically different canvases.
    if (layoutMode === 'single') {
      if (!layoutAutoSmart) {
        const nextFitMode = (nextAspect === '16:9' && layoutFitMode === 'cover')
          ? 'contain'
          : layoutFitMode;
        if (nextFitMode !== layoutFitMode) {
          setLayoutFitMode(nextFitMode);
        }
        const minZoom = getMinZoomForFitMode(nextFitMode);
        setLayoutZoom(clamp(1, minZoom, 2.5));
      }
      setLayoutOffsetX(0);
      setLayoutOffsetY(0);
      return true;
    }

    setLayoutSplitZoomA(1);
    setLayoutSplitZoomB(1);
    setLayoutSplitOffsetAX(0);
    setLayoutSplitOffsetAY(0);
    setLayoutSplitOffsetBX(0);
    setLayoutSplitOffsetBY(0);
    return true;
  }, [
    layoutAspect,
    layoutMode,
    layoutAutoSmart,
    layoutFitMode
  ]);

  // No automatic zoom effects on pan.

  useEffect(() => {
    const minZoom = getMinZoomForFitMode(layoutFitMode);
    setLayoutZoom((prev) => clamp(Number(prev || 1), minZoom, 2.5));
    setLayoutSplitZoomA((prev) => clamp(Number(prev || 1), minZoom, 2.5));
    setLayoutSplitZoomB((prev) => clamp(Number(prev || 1), minZoom, 2.5));
  }, [layoutFitMode]);

  const [musicEnabled, setMusicEnabled] = useState(false);
  const [musicFile, setMusicFile] = useState(null);
  const [musicVolume, setMusicVolume] = useState(0.18);
  const [duckVoice, setDuckVoice] = useState(true);
  const [dubbingEnabled, setDubbingEnabled] = useState(false);
  const [dubbingTargetLanguage, setDubbingTargetLanguage] = useState('es');
  const [dubbingSourceLanguage, setDubbingSourceLanguage] = useState('auto');
  const [dubbingLanguages, setDubbingLanguages] = useState(DEFAULT_DUBBING_LANGUAGES);
  const [isLoadingDubbingLanguages, setIsLoadingDubbingLanguages] = useState(false);
  const dubbingLanguageOptions = useMemo(
    () => Object.entries(dubbingLanguages || {}).sort((a, b) => String(a[1]).localeCompare(String(b[1]))),
    [dubbingLanguages]
  );

  const [viralHookText, setViralHookText] = useState(() => resolveDefaultViralHookText(clip, clipIndex, currentVideoUrl));
  const [viralHookEnabled, setViralHookEnabled] = useState(() => Boolean(resolveDefaultViralHookText(clip, clipIndex, currentVideoUrl)));
  const [viralHookStart, setViralHookStart] = useState(Math.max(0, Number(clip?.viral_hook_start || 0)));
  const [viralHookDuration, setViralHookDuration] = useState(VIRAL_HOOK_DEFAULT_DURATION);
  const [viralHookFontSize, setViralHookFontSize] = useState(() => resolveViralHookStyle(clip).fontSize);
  const [viralHookFontFamily, setViralHookFontFamily] = useState(() => resolveViralHookStyle(clip).fontFamily);
  const [viralHookFontColor, setViralHookFontColor] = useState(() => resolveViralHookStyle(clip).fontColor);
  const [viralHookStrokeColor, setViralHookStrokeColor] = useState(() => resolveViralHookStyle(clip).strokeColor);
  const [viralHookStrokeWidth, setViralHookStrokeWidth] = useState(() => resolveViralHookStyle(clip).strokeWidth);
  const [viralHookBold, setViralHookBold] = useState(() => resolveViralHookStyle(clip).bold);
  const [viralHookBoxColor, setViralHookBoxColor] = useState(() => resolveViralHookStyle(clip).boxColor);
  const [viralHookBoxOpacity, setViralHookBoxOpacity] = useState(() => resolveViralHookStyle(clip).boxOpacity);
  const [viralHookLineSpacing, setViralHookLineSpacing] = useState(0);

  const [previewPlaying, setPreviewPlaying] = useState(false);
  const [previewCurrentTime, setPreviewCurrentTime] = useState(0);
  const [previewDuration, setPreviewDuration] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [timelineZoom, setTimelineZoom] = useState(TIMELINE_ZOOM_DEFAULT);
  const [timelineMode, setTimelineMode] = useState(TIMELINE_MODE_ADVANCED);
  const [snapEnabled, setSnapEnabled] = useState(true);
  const [previewVideoUrl, setPreviewVideoUrl] = useState('');
  const [fastPreviewCaptionsBurned, setFastPreviewCaptionsBurned] = useState(false);
  const [uncutFailed, setUncutFailed] = useState(false);

  // If the user uses manual framing (Auto smart reframe OFF), prefer _uncut.mp4 to show the whole video
  const uncutVideoUrl = currentVideoUrl ? currentVideoUrl.replace('.mp4', '_uncut.mp4') : '';
  const isManualLayout = !layoutAutoSmart;
  const activeSourceUrl = (isManualLayout && !uncutFailed) ? uncutVideoUrl : currentVideoUrl;

  const [videoLoadError, setVideoLoadError] = useState('');
  const previewVideoRef = useRef(null);
  const previewSplitVideoRef = useRef(null);
  const previewSurfaceRef = useRef(null);
  const subtitleListRef = useRef(null);
  const transcriptListRef = useRef(null);
  const previewBlobUrlRef = useRef(null);
  const fastPreviewStopTimerRef = useRef(null);
  const timelineViewportRef = useRef(null);
  const timelineTrackRef = useRef(null);
  const subtitleDragRef = useRef(null);
  const viralHookDragRef = useRef(null);
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

  const baseClipStart = Number(clip?.start || 0);
  const baseClipEnd = Number(clip?.end || baseClipStart);

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
    const absoluteTime = Number(baseClipStart || 0) + Number(previewCurrentTime || 0);
    const activeSpeaker = speakerColorMode && Array.isArray(transcriptSegments)
      ? (transcriptSegments.find((seg) => {
        const segStart = Number(seg?.start || 0);
        const segEnd = Number(seg?.end || segStart);
        return absoluteTime >= segStart && absoluteTime <= (segEnd + 0.05);
      })?.speaker || '')
      : '';
    return {
      words,
      activeIndex,
      activeColor: activeSpeaker ? pickSpeakerColorForLabel(activeSpeaker) : suggestEmotionColorForText(rawText)
    };
  }, [karaokeMode, activeSubtitleEntry, previewCurrentTime, punctuationOn, baseClipStart, transcriptSegments, speakerColorMode]);
  const captionDragEnabled = captionsOn && (section === 'captions' || section === 'subtitle_edit');
  const captionAnchorTopPercent = position === 'top' ? 12 : position === 'middle' ? 50 : 86;

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
      const sceneDescription = String(seg?.scene_description || '').toLowerCase();
      return text.includes(q) || sceneDescription.includes(q);
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
    const overriddenTitle = String(previewTitleOverride || '').trim();
    if (overriddenTitle) return overriddenTitle;
    const socialTitle = String(clip?.video_title_for_youtube_short || '').trim();
    if (socialTitle) return socialTitle;
    const genericTitle = String(clip?.title || '').trim();
    if (genericTitle) return genericTitle;
    const filename = extractFilename(currentVideoUrl) || extractFilename(clip?.video_url || '');
    if (filename) return filename;
    return `Clip n.o ${Number(clipIndex || 0) + 1}`;
  }, [previewTitleOverride, clip?.video_title_for_youtube_short, clip?.title, clip?.video_url, currentVideoUrl, clipIndex]);

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
  const viralHookTimelineDuration = useMemo(() => clamp(Number(viralHookDuration || 0), 0.4, Math.max(0.4, timelineDuration)), [viralHookDuration, timelineDuration]);
  const viralHookTimelineStart = useMemo(
    () => clamp(Number(viralHookStart || 0), 0, Math.max(0, timelineDuration - viralHookTimelineDuration)),
    [viralHookStart, timelineDuration, viralHookTimelineDuration]
  );
  const viralHookTimelineEnd = useMemo(
    () => clamp(viralHookTimelineStart + viralHookTimelineDuration, viralHookTimelineStart + 0.4, timelineDuration),
    [viralHookTimelineStart, viralHookTimelineDuration, timelineDuration]
  );
  const previewViralHookText = useMemo(
    () => (viralHookEnabled ? String(viralHookText || '').trim() : ''),
    [viralHookEnabled, viralHookText]
  );
  const previewViralHookBoxBg = useMemo(() => {
    const normalizedOpacity = clamp(Number(viralHookBoxOpacity || 0), 0, 100);
    const effectiveOpacity = normalizedOpacity > 0 ? normalizedOpacity : 68;
    return toRgba(viralHookBoxColor, effectiveOpacity);
  }, [viralHookBoxColor, viralHookBoxOpacity]);
  const previewViralHookStrokeWidth = useMemo(
    () => clamp(Number(viralHookStrokeWidth || 0) * 0.45, 0, 4),
    [viralHookStrokeWidth]
  );
  const previewViralHookFontSize = useMemo(
    () => clamp(Math.round(Number(viralHookFontSize || DEFAULT_VIRAL_HOOK_STYLE.fontSize) * 0.62), 14, 42),
    [viralHookFontSize]
  );
  const showPreviewViralHook = useMemo(() => {
    if (!previewViralHookText) return false;
    const t = Number(previewCurrentTime || 0);
    return t >= viralHookTimelineStart && t <= viralHookTimelineEnd;
  }, [previewViralHookText, previewCurrentTime, viralHookTimelineStart, viralHookTimelineEnd]);

  const selectionStartRel = useMemo(() => {
    return clamp(Number(layoutStart || baseClipStart) - baseClipStart, 0, timelineDuration);
  }, [layoutStart, baseClipStart, timelineDuration]);

  const selectionEndRel = useMemo(() => {
    const raw = Number(layoutEnd || baseClipEnd) - baseClipStart;
    return clamp(raw, selectionStartRel + 0.08, timelineDuration);
  }, [layoutEnd, baseClipEnd, baseClipStart, selectionStartRel, timelineDuration]);

  useEffect(() => {
    setViralHookDuration((prev) => clamp(Number(prev || VIRAL_HOOK_DEFAULT_DURATION), 0.4, Math.max(0.4, timelineDuration)));
  }, [timelineDuration]);

  useEffect(() => {
    setViralHookStart((prev) => clamp(Number(prev || 0), 0, Math.max(0, timelineDuration - viralHookTimelineDuration)));
  }, [timelineDuration, viralHookTimelineDuration]);

  const transitionPoints = useMemo(() => {
    const points = Array.isArray(clip?.transition_points) ? clip.transition_points : [];
    const normalized = points
      .map((value) => Number(value))
      .filter((value) => Number.isFinite(value))
      .map((value) => clamp(value, 0, timelineDuration))
      .sort((a, b) => a - b);
    const deduped = [];
    normalized.forEach((value) => {
      if (deduped.length === 0 || Math.abs(deduped[deduped.length - 1] - value) > 0.03) {
        deduped.push(value);
      }
    });
    return deduped;
  }, [clip?.transition_points, timelineDuration]);

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
    setKaraokeMode(Boolean(preset.karaokeMode));
    setSubtitleAnimation(String(preset.style?.animation || 'none'));
    setSpeakerColorMode(Boolean(preset.style?.speakerColorMode));
    // Sync viral hook to the same style automatically
    applyViralHookPreset(presetId);
  };

  const applyViralHookPreset = (presetId) => {
    const preset = CAPTION_PRESETS.find((p) => p.id === presetId);
    if (!preset) return;
    setSelectedViralHookPreset(presetId);
    setViralHookFontSize(clamp(Number(preset.style?.fontSize || CAPTION_PRESETS[0].style.fontSize), VIRAL_HOOK_FONT_SIZE_MIN, VIRAL_HOOK_FONT_SIZE_MAX));
    setViralHookFontFamily(normalizeSubtitleFontFamily(preset.style?.fontFamily || CAPTION_PRESETS[0].style.fontFamily));
    setViralHookFontColor(String(preset.style?.fontColor || CAPTION_PRESETS[0].style.fontColor));
    setViralHookStrokeColor(String(preset.style?.strokeColor || CAPTION_PRESETS[0].style.strokeColor));
    setViralHookStrokeWidth(clamp(Number(preset.style?.strokeWidth ?? CAPTION_PRESETS[0].style.strokeWidth), 0, 8));
    setViralHookBold(Boolean(preset.style?.bold ?? CAPTION_PRESETS[0].style.bold));
    setViralHookBoxColor(String(preset.style?.boxColor || CAPTION_PRESETS[0].style.boxColor));
    setViralHookBoxOpacity(clamp(Number(preset.style?.boxOpacity ?? CAPTION_PRESETS[0].style.boxOpacity), 0, 100));
  };

  useEffect(() => {
    if (!isOpen) return;
    let cancelled = false;
    const loadDubbingLanguages = async () => {
      setIsLoadingDubbingLanguages(true);
      try {
        const res = await apiFetch('/api/translate/languages');
        if (!res.ok) return;
        const data = await res.json();
        if (cancelled) return;
        if (data?.languages && typeof data.languages === 'object' && !Array.isArray(data.languages)) {
          setDubbingLanguages(data.languages);
        }
      } catch (_) {
        // Keep fallback list when endpoint is unavailable.
      } finally {
        if (!cancelled) setIsLoadingDubbingLanguages(false);
      }
    };
    loadDubbingLanguages();
    return () => {
      cancelled = true;
    };
  }, [isOpen]);

  const loadTranscript = async () => {
    const clipTranscriptSegments = Array.isArray(clip?.transcript_segments) ? clip.transcript_segments : [];
    setIsLoadingTranscript(true);
    if (clipTranscriptSegments.length > 0) {
      const clipTimebase = String(clip?.transcript_timebase || '').trim().toLowerCase();
      const baseOffset = clipTimebase === 'clip' ? Number(clip?.start || 0) : 0;
      const normalized = clipTranscriptSegments
        .map((seg, idx) => {
          if (!seg || typeof seg !== 'object') return null;
          const rawStart = Number(seg.start ?? 0);
          const rawEnd = Number(seg.end ?? rawStart);
          const segStart = Number.isFinite(rawStart) ? Math.max(0, rawStart + baseOffset) : 0;
          const segEnd = Number.isFinite(rawEnd) ? Math.max(segStart, rawEnd + baseOffset) : segStart;
          const words = Array.isArray(seg.words)
            ? seg.words
              .map((wordItem) => {
                if (!wordItem || typeof wordItem !== 'object') return null;
                const wsRaw = Number(wordItem.start ?? segStart);
                const weRaw = Number(wordItem.end ?? wsRaw);
                const ws = Number.isFinite(wsRaw) ? Math.max(0, wsRaw + baseOffset) : segStart;
                const we = Number.isFinite(weRaw) ? Math.max(ws, weRaw + baseOffset) : ws;
                const token = String(wordItem.word || wordItem.text || '').trim();
                if (!token) return null;
                return {
                  start: Number(ws.toFixed(3)),
                  end: Number(we.toFixed(3)),
                  word: token
                };
              })
              .filter(Boolean)
            : [];
          const text = String(seg.text || '').trim() || words.map((w) => w.word).join(' ').trim();
          if (!text) return null;
          const sceneDescription = String(
            seg.scene_description
            || seg.sceneDescription
            || seg.visual_description
            || seg.description
            || ''
          ).trim();
          return {
            segment_index: Number.isFinite(Number(seg.segment_index)) ? Number(seg.segment_index) : idx,
            start: Number(segStart.toFixed(3)),
            end: Number(segEnd.toFixed(3)),
            duration: Number(Math.max(0, segEnd - segStart).toFixed(3)),
            speaker: seg.speaker ? String(seg.speaker).trim() : null,
            word_count: words.length,
            text,
            words,
            scene_description: sceneDescription || null
          };
        })
        .filter(Boolean);
      setTranscriptSegments(normalized);
      setIsLoadingTranscript(false);
      return;
    }
    if (!jobId) {
      setIsLoadingTranscript(false);
      return;
    }
    try {
      const res = await apiFetch(`/api/transcript/${jobId}?limit=2000&include_words=1`);
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      const segments = Array.isArray(data?.segments) ? data.segments : [];
      const normalizedSegments = segments.map((seg, idx) => {
        const sceneDescription = String(
          seg?.scene_description
          || seg?.sceneDescription
          || seg?.visual_description
          || seg?.description
          || ''
        ).trim();
        return {
          ...seg,
          segment_index: Number.isFinite(Number(seg?.segment_index)) ? Number(seg.segment_index) : idx,
          scene_description: sceneDescription || null
        };
      });
      setTranscriptSegments(normalizedSegments);
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
      const rawParsed = parseSrt(data?.srt || '');

      // Enrich with rich words and speakers from transcriptSegments if available
      const enrichedParsed = rawParsed.map(entry => {
        let entryWords = [];
        let speakerCounts = {};

        if (transcriptSegments && transcriptSegments.length > 0) {
          transcriptSegments.forEach(seg => {
            if (seg.words && Array.isArray(seg.words)) {
              seg.words.forEach(w => {
                // Approximate overlap check (Word falls inside SRT bound)
                if (w.end > entry.start && w.start < entry.end) {
                  entryWords.push({
                    word: w.word,
                    start: w.start,
                    end: w.end,
                    speaker: seg.speaker || null
                  });
                  if (seg.speaker) {
                    speakerCounts[seg.speaker] = (speakerCounts[seg.speaker] || 0) + 1;
                  }
                }
              });
            }
          });
        }

        let dominantSpeaker = null;
        let maxCount = 0;
        for (const [spk, count] of Object.entries(speakerCounts)) {
          if (count > maxCount) {
            maxCount = count;
            dominantSpeaker = spk;
          }
        }

        return {
          ...entry,
          words: entryWords.length > 0 ? entryWords : undefined,
          speaker: dominantSpeaker || undefined
        };
      });

      setSubtitleEntries(enrichedParsed);
    } catch (e) {
      setError(`No se pudo cargar subtítulos: ${e.message}`);
      setTimeout(() => setError(''), 3500);
    } finally {
      setIsLoadingSrt(false);
    }
  };

  const parseApiErrorDetail = async (res) => {
    const raw = await res.text();
    let detail = '';
    try {
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed.detail === 'string') detail = parsed.detail;
    } catch (_) {
      // Non-JSON error body.
    }
    return detail || raw || `HTTP ${res.status}`;
  };

  const buildDownloadFilename = useCallback(() => {
    const fallbackIndex = Number.isFinite(Number(clipIndex)) ? Number(clipIndex) + 1 : 1;
    const raw = String(
      previewTitleOverride
      || clip?.video_title_for_youtube_short
      || clip?.title
      || `clip-${fallbackIndex}`
    );
    const safe = raw
      .normalize('NFD')
      .replace(/[\u0300-\u036f]/g, '')
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '')
      .slice(0, 72);
    return `${safe || `clip-${fallbackIndex}`}.mp4`;
  }, [clip?.title, clip?.video_title_for_youtube_short, clipIndex, previewTitleOverride]);

  const triggerAnchorDownload = useCallback((href, filename) => {
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = href;
    if (filename) a.download = filename;
    a.rel = 'noopener noreferrer';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }, []);

  const openDownloadFallback = useCallback((sourceUrl) => {
    const safeUrl = String(sourceUrl || '').trim();
    if (!safeUrl) return;
    const popup = window.open(safeUrl, '_blank', 'noopener,noreferrer');
    if (!popup) {
      triggerAnchorDownload(safeUrl, undefined);
    }
  }, [triggerAnchorDownload]);

  const downloadVideoFromUrl = useCallback(async (sourceUrl) => {
    const safeUrl = String(sourceUrl || '').trim();
    if (!safeUrl) throw new Error('URL de video vacía');

    const useDirectFetch = /^blob:|^data:/i.test(safeUrl);
    const response = await (useDirectFetch
      ? fetch(safeUrl, { method: 'GET' })
      : apiFetch(safeUrl, { method: 'GET' }));

    if (!response.ok) {
      throw new Error(`Descarga fallida (${response.status})`);
    }

    const contentType = String(response.headers.get('content-type') || '').toLowerCase();
    if (contentType.includes('text/html')) {
      throw new Error('El servidor devolvió HTML en lugar de video');
    }

    const blob = await response.blob();
    const blobType = String(blob.type || '').toLowerCase();
    if (blobType.includes('text/html')) {
      throw new Error('Respuesta inválida al descargar video');
    }

    const objectUrl = window.URL.createObjectURL(blob);
    try {
      triggerAnchorDownload(objectUrl, buildDownloadFilename());
    } finally {
      setTimeout(() => {
        window.URL.revokeObjectURL(objectUrl);
      }, 1500);
    }
  }, [buildDownloadFilename, triggerAnchorDownload]);

  const downloadVideoWithFallback = useCallback(async (sourceUrl) => {
    const safeUrl = String(sourceUrl || '').trim();
    if (!safeUrl) return;
    try {
      await downloadVideoFromUrl(safeUrl);
    } catch (downloadErr) {
      console.warn('No se pudo descargar automáticamente el clip editado:', downloadErr);
      openDownloadFallback(safeUrl);
    }
  }, [downloadVideoFromUrl, openDownloadFallback]);

  useEffect(() => {
    if (!isOpen) return;
    setSection('transcript');
    setPreviewTitleOverride('');
    setLayoutAspect(clip?.aspect_ratio === '16:9' ? '16:9' : '9:16');
    setLayoutStart(Number(clip?.start || 0));
    setLayoutEnd(Number(clip?.end || 0));
    setLayoutPreRoll(0);
    setLayoutPostRoll(0);
    setLayoutMode(String(clip?.layout_mode || 'single').toLowerCase() === 'split' ? 'split' : 'single');
    setLayoutAutoSmart(Boolean(clip?.layout_auto_smart));
    const nextFitMode = String(clip?.layout_fit_mode || 'cover').toLowerCase() === 'contain' ? 'contain' : 'cover';
    const nextZoomMin = getMinZoomForFitMode(nextFitMode);
    setLayoutFitMode(nextFitMode);
    setLayoutZoom(clamp(Number(clip?.layout_zoom || 1), nextZoomMin, 2.5));
    setLayoutOffsetX(clamp(Number(clip?.layout_offset_x || 0), -100, 100));
    setLayoutOffsetY(clamp(Number(clip?.layout_offset_y || 0), -100, 100));
    setLayoutSplitZoomA(clamp(Number(clip?.layout_split_zoom_a ?? clip?.layout_zoom ?? 1), nextZoomMin, 2.5));
    setLayoutSplitOffsetAX(clamp(Number(clip?.layout_split_offset_a_x ?? clip?.layout_offset_x ?? 0), -100, 100));
    setLayoutSplitOffsetAY(clamp(Number(clip?.layout_split_offset_a_y ?? clip?.layout_offset_y ?? 0), -100, 100));
    setLayoutSplitZoomB(clamp(Number(clip?.layout_split_zoom_b ?? clip?.layout_zoom ?? 1), nextZoomMin, 2.5));
    setLayoutSplitOffsetBX(clamp(Number(clip?.layout_split_offset_b_x ?? (-(Number(clip?.layout_offset_x || 0)))), -100, 100));
    setLayoutSplitOffsetBY(clamp(Number(clip?.layout_split_offset_b_y ?? clip?.layout_offset_y ?? 0), -100, 100));
    setMusicEnabled(false);
    setMusicFile(null);
    setMusicVolume(0.18);
    setDuckVoice(true);
    setDubbingEnabled(false);
    setDubbingTargetLanguage(String(clip?.dub_target_language || 'es').trim().toLowerCase() || 'es');
    setDubbingSourceLanguage(String(clip?.dub_source_language || 'auto').trim().toLowerCase() || 'auto');
    const nextViralHookDefault = resolveDefaultViralHookText(clip, clipIndex, currentVideoUrl);
    const nextViralHookStyle = resolveViralHookStyle(clip);
    setViralHookText(nextViralHookDefault);
    setViralHookEnabled(Boolean(nextViralHookDefault));
    setViralHookStart(Math.max(0, Number(clip?.viral_hook_start || 0)));
    setViralHookDuration(clamp(Number(clip?.viral_hook_duration || VIRAL_HOOK_DEFAULT_DURATION), 0.4, 8));
    setViralHookFontSize(nextViralHookStyle.fontSize);
    setViralHookFontFamily(nextViralHookStyle.fontFamily);
    setViralHookFontColor(nextViralHookStyle.fontColor);
    setViralHookStrokeColor(nextViralHookStyle.strokeColor);
    setViralHookStrokeWidth(nextViralHookStyle.strokeWidth);
    setViralHookBold(nextViralHookStyle.bold);
    setViralHookBoxColor(nextViralHookStyle.boxColor);
    setViralHookBoxOpacity(nextViralHookStyle.boxOpacity);
    setSelectedViralHookPreset('');
    setPunctuationOn(true);
    setEmojiOn(true);
    setShowCaptionSettings(false);
    setTranscriptPlainMode(false);
    setTranscriptSceneDescriptionsOn(true);
    setEmojiPickerForId('');
    setEmojiSuggestFeedback('');
    setTimelineZoom(TIMELINE_ZOOM_DEFAULT);
    setTimelineMode(TIMELINE_MODE_MINI);

    const hasClipStyle = [
      clip?.caption_font_size,
      clip?.caption_font_family,
      clip?.caption_font_color,
      clip?.caption_stroke_color,
      clip?.caption_stroke_width,
      clip?.caption_bold,
      clip?.caption_box_color,
      clip?.caption_box_opacity,
      clip?.caption_karaoke_mode,
      clip?.caption_animation,
      clip?.caption_speaker_color_mode
    ].some((v) => v !== undefined && v !== null && String(v) !== '');

    if (hasClipStyle) {
      const clipPosition = String(clip?.caption_position || '').toLowerCase();
      setPosition(['top', 'middle', 'bottom'].includes(clipPosition) ? clipPosition : CAPTION_PRESETS[0].style.position);
      setFontSize(clamp(Number(clip?.caption_font_size ?? CAPTION_PRESETS[0].style.fontSize), 12, 84));
      setFontFamily(normalizeSubtitleFontFamily(clip?.caption_font_family || CAPTION_PRESETS[0].style.fontFamily));
      setFontColor(String(clip?.caption_font_color || CAPTION_PRESETS[0].style.fontColor));
      setStrokeColor(String(clip?.caption_stroke_color || CAPTION_PRESETS[0].style.strokeColor));
      setStrokeWidth(clamp(Number(clip?.caption_stroke_width ?? CAPTION_PRESETS[0].style.strokeWidth), 0, 8));
      setBold(typeof clip?.caption_bold === 'boolean' ? clip.caption_bold : Boolean(CAPTION_PRESETS[0].style.bold));
      setBoxColor(String(clip?.caption_box_color || CAPTION_PRESETS[0].style.boxColor));
      setBoxOpacity(clamp(Number(clip?.caption_box_opacity ?? CAPTION_PRESETS[0].style.boxOpacity), 0, 100));
      setKaraokeMode(Boolean(clip?.caption_karaoke_mode));
      setSubtitleAnimation(['none', 'pop', 'bounce', 'slide'].includes(String(clip?.caption_animation || '').toLowerCase())
        ? String(clip?.caption_animation || '').toLowerCase()
        : 'none');
      setSpeakerColorMode(Boolean(clip?.caption_speaker_color_mode));
      setSelectedPreset('');
    } else {
      const brandKitStyle = readStoredBrandKitSubtitleStyle();
      if (brandKitStyle) {
        if (brandKitStyle.position) setPosition(brandKitStyle.position);
        setFontSize(clamp(Number(brandKitStyle.fontSize), 12, 84));
        setFontFamily(normalizeSubtitleFontFamily(brandKitStyle.fontFamily));
        setFontColor(String(brandKitStyle.fontColor || '#FFFFFF'));
        setStrokeColor(String(brandKitStyle.strokeColor || '#000000'));
        setStrokeWidth(clamp(Number(brandKitStyle.strokeWidth), 0, 8));
        setBold(Boolean(brandKitStyle.bold));
        setBoxColor(String(brandKitStyle.boxColor || '#000000'));
        setBoxOpacity(clamp(Number(brandKitStyle.boxOpacity), 0, 100));
        setKaraokeMode(Boolean(brandKitStyle.karaokeMode));
        setSubtitleAnimation(String(brandKitStyle.subtitleAnimation || 'none'));
        setSpeakerColorMode(Boolean(brandKitStyle.speakerColorMode));
        setSelectedPreset('');
      } else {
        applyPreset(CAPTION_PRESETS[0].id);
      }
    }

    setCaptionOffsetX(clamp(Number(clip?.caption_offset_x || 0), -100, 100));
    setCaptionOffsetY(clamp(Number(clip?.caption_offset_y ?? 10), -100, 100));
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
    clip?.caption_offset_y,
    clip?.caption_font_size,
    clip?.caption_font_family,
    clip?.caption_font_color,
    clip?.caption_stroke_color,
    clip?.caption_stroke_width,
    clip?.caption_bold,
    clip?.caption_box_color,
    clip?.caption_box_opacity,
    clip?.caption_karaoke_mode,
    clip?.caption_animation,
    clip?.caption_speaker_color_mode,
    clip?.viral_hook_text,
    clip?.viral_hook_duration,
    clip?.viral_hook_start,
    clip?.viral_hook_font_size,
    clip?.viral_hook_font_family,
    clip?.viral_hook_font_color,
    clip?.viral_hook_stroke_color,
    clip?.viral_hook_stroke_width,
    clip?.viral_hook_bold,
    clip?.viral_hook_box_color,
    clip?.viral_hook_box_opacity,
    clip?.dub_target_language,
    clip?.dub_source_language,
    clip?.transcript_segments,
    clip?.transcript_timebase,
    clip?.video_title_for_youtube_short,
    clip?.title,
    clip?.video_url,
    currentVideoUrl
  ]);

  useEffect(() => {
    const cleanupBlobUrl = () => {
      if (previewBlobUrlRef.current) {
        URL.revokeObjectURL(previewBlobUrlRef.current);
        previewBlobUrlRef.current = null;
      }
    };

    const sourceUrl = String(currentVideoUrl || '').trim();
    if (!isOpen) return () => { };
    if (!sourceUrl) {
      cleanupBlobUrl();
      setPreviewVideoUrl('');
      setFastPreviewCaptionsBurned(false);
      setVideoLoadError('');
      return () => { };
    }

    const isHttp = /^https?:\/\//i.test(sourceUrl);
    const isBlobOrData = /^blob:|^data:/i.test(sourceUrl);
    if (!isHttp || isBlobOrData) {
      cleanupBlobUrl();
      setPreviewVideoUrl(sourceUrl);
      setFastPreviewCaptionsBurned(false);
      setVideoLoadError('');
      return () => { };
    }

    const isNgrokSource = /ngrok/i.test(sourceUrl);
    if (!isNgrokSource) {
      cleanupBlobUrl();
      setPreviewVideoUrl(sourceUrl);
      setFastPreviewCaptionsBurned(false);
      setVideoLoadError('');
      return () => { };
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
        setFastPreviewCaptionsBurned(false);
      } catch (err) {
        if (cancelled) return;
        // Fallback silencioso: aun si falla la descarga manual, el elemento <video>
        // puede cargar la URL remota directamente.
        setPreviewVideoUrl(sourceUrl);
        setFastPreviewCaptionsBurned(false);
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
      playPromise.catch(() => { });
    }
  }, [isSplitLayout, previewCurrentTime, previewPlaying, previewVideoUrl, isOpen]);

  useEffect(() => {
    return () => {
      if (fastPreviewStopTimerRef.current) {
        clearTimeout(fastPreviewStopTimerRef.current);
        fastPreviewStopTimerRef.current = null;
      }
    };
  }, []);

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

  const handleApply = async ({ downloadAfter = false } = {}) => {
    if (!jobId) return;
    setApplyAction(downloadAfter ? 'apply_download' : 'apply');
    setIsApplying(true);
    setError('');
    let workingFile = extractFilename(currentVideoUrl);
    let resultingUrl = currentVideoUrl;
    const normalizedLayoutMode = isSplitLayout ? 'split' : 'single';
    const autoSmartRequest = normalizedLayoutMode === 'single' ? Boolean(layoutAutoSmart) : false;
    let appliedLayoutMode = normalizedLayoutMode;
    let appliedLayoutAutoSmart = autoSmartRequest;
    let appliedCaptionPosition = position;
    let appliedCaptionOffsetX = Number(captionOffsetX);
    let appliedCaptionOffsetY = Number(captionOffsetY);
    let appliedCaptionFontSize = Number(fontSize);
    let appliedCaptionFontFamily = normalizeSubtitleFontFamily(fontFamily);
    let appliedCaptionFontColor = fontColor;
    let appliedCaptionStrokeColor = strokeColor;
    let appliedCaptionStrokeWidth = Number(strokeWidth);
    let appliedCaptionBold = Boolean(bold);
    let appliedCaptionBoxColor = boxColor;
    let appliedCaptionBoxOpacity = Number(boxOpacity);
    let appliedCaptionKaraokeMode = Boolean(karaokeMode);
    let appliedCaptionAnimation = String(subtitleAnimation || 'none');
    let appliedCaptionSpeakerColorMode = Boolean(speakerColorMode);
    let appliedDubTargetLanguage = dubbingEnabled ? String(dubbingTargetLanguage || 'es').trim().toLowerCase() : '';
    let appliedDubSourceLanguage = dubbingEnabled ? String(dubbingSourceLanguage || 'auto').trim().toLowerCase() : 'auto';
    let appliedViralHookText = viralHookEnabled ? String(viralHookText || '').trim() : '';
    let appliedViralHookStart = viralHookEnabled ? Number(viralHookTimelineStart || 0) : 0;
    let appliedViralHookDuration = viralHookEnabled ? Number(viralHookTimelineDuration || 0) : 0;
    let appliedViralHookFontSize = clamp(Number(viralHookFontSize || DEFAULT_VIRAL_HOOK_STYLE.fontSize), VIRAL_HOOK_FONT_SIZE_MIN, VIRAL_HOOK_FONT_SIZE_MAX);
    let appliedViralHookFontFamily = normalizeSubtitleFontFamily(viralHookFontFamily);
    let appliedViralHookFontColor = String(viralHookFontColor || '#FFFFFF');
    let appliedViralHookStrokeColor = String(viralHookStrokeColor || '#000000');
    let appliedViralHookStrokeWidth = clamp(Number(viralHookStrokeWidth || 0), 0, 8);
    let appliedViralHookBold = Boolean(viralHookBold);
    let appliedViralHookBoxColor = String(viralHookBoxColor || '#000000');
    let appliedViralHookBoxOpacity = clamp(Number(viralHookBoxOpacity || 0), 0, 100);
    let subtitleSrtPayload = srtContent || null;

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
      if (viralHookEnabled && !appliedViralHookText) {
        appliedViralHookText = resolveDefaultViralHookText(clip, clipIndex, currentVideoUrl);
        setViralHookText(appliedViralHookText);
      }
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

      // We now determine if a new render is required based on layout OR caption changes.
      // With Single-Pass Optimization, we send everything to /api/recut.
      const captionsChanged = captionsOn && (
        appliedCaptionPosition !== String(clip?.caption_position || appliedCaptionPosition)
        || Math.abs(appliedCaptionOffsetX - Number(clip?.caption_offset_x || 0)) > 0.01
        || Math.abs(appliedCaptionOffsetY - Number(clip?.caption_offset_y || 0)) > 0.01
        || Math.abs(appliedCaptionFontSize - Number(clip?.caption_font_size || appliedCaptionFontSize)) > 0.01
        || appliedCaptionFontFamily !== String(clip?.caption_font_family || appliedCaptionFontFamily)
        || appliedCaptionFontColor !== String(clip?.caption_font_color || appliedCaptionFontColor)
        || appliedCaptionStrokeColor !== String(clip?.caption_stroke_color || appliedCaptionStrokeColor)
        || Math.abs(appliedCaptionStrokeWidth - Number(clip?.caption_stroke_width || appliedCaptionStrokeWidth)) > 0.01
        || appliedCaptionBold !== Boolean(clip?.caption_bold ?? appliedCaptionBold)
        || appliedCaptionBoxColor !== String(clip?.caption_box_color || appliedCaptionBoxColor)
        || Math.abs(appliedCaptionBoxOpacity - Number(clip?.caption_box_opacity || appliedCaptionBoxOpacity)) > 0.01
        || appliedCaptionKaraokeMode !== Boolean(clip?.caption_karaoke_mode ?? appliedCaptionKaraokeMode)
        || appliedCaptionAnimation !== String(clip?.caption_animation || appliedCaptionAnimation)
        || appliedCaptionSpeakerColorMode !== Boolean(clip?.caption_speaker_color_mode ?? appliedCaptionSpeakerColorMode)
      );
      const existingViralHookText = String(clip?.viral_hook_text || '').trim();
      const existingViralHookStart = Math.max(0, Number(clip?.viral_hook_start || 0));
      const existingViralHookDuration = Math.max(0, Number(clip?.viral_hook_duration || 0));
      const existingViralHookStyle = resolveViralHookStyle(clip, {
        fontSize: appliedViralHookFontSize,
        fontFamily: appliedViralHookFontFamily,
        fontColor: appliedViralHookFontColor,
        strokeColor: appliedViralHookStrokeColor,
        strokeWidth: appliedViralHookStrokeWidth,
        bold: appliedViralHookBold,
        boxColor: appliedViralHookBoxColor,
        boxOpacity: appliedViralHookBoxOpacity
      });
      const viralHookChanged = (
        appliedViralHookText !== existingViralHookText
        || Math.abs(appliedViralHookStart - existingViralHookStart) > 0.01
        || Math.abs(appliedViralHookDuration - existingViralHookDuration) > 0.01
        || (Boolean(appliedViralHookText) && (
          Math.abs(appliedViralHookFontSize - Number(existingViralHookStyle.fontSize || appliedViralHookFontSize)) > 0.01
          || appliedViralHookFontFamily !== String(existingViralHookStyle.fontFamily || appliedViralHookFontFamily)
          || appliedViralHookFontColor !== String(existingViralHookStyle.fontColor || appliedViralHookFontColor)
          || appliedViralHookStrokeColor !== String(existingViralHookStyle.strokeColor || appliedViralHookStrokeColor)
          || Math.abs(appliedViralHookStrokeWidth - Number(existingViralHookStyle.strokeWidth || appliedViralHookStrokeWidth)) > 0.01
          || appliedViralHookBold !== Boolean(existingViralHookStyle.bold)
          || appliedViralHookBoxColor !== String(existingViralHookStyle.boxColor || appliedViralHookBoxColor)
          || Math.abs(appliedViralHookBoxOpacity - Number(existingViralHookStyle.boxOpacity || appliedViralHookBoxOpacity)) > 0.01
        ))
      );

      // We always attempt to fetch fresh subtitle SRT data if we are overriding it or if timestamps changed
      const clipRangeChangedByRecut = (
        Math.abs(Number(appliedClipStart || 0) - Number(clipStart || 0)) > 0.01
        || Math.abs(Number(appliedClipEnd || 0) - Number(clipEnd || 0)) > 0.01
      );
      if (captionsOn && (clipRangeChangedByRecut || srtContent)) {
        try {
          const refreshedSrtRes = await apiFetch('/api/subtitle/preview', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ job_id: jobId, clip_index: clipIndex })
          });
          if (refreshedSrtRes.ok) {
            const refreshedSrtData = await refreshedSrtRes.json();
            const refreshedEntries = parseSrt(refreshedSrtData?.srt || '');
            if (Array.isArray(refreshedEntries) && refreshedEntries.length > 0) {
              setSubtitleEntries(refreshedEntries);
              subtitleSrtPayload = buildSrt(refreshedEntries, { punctuationOn, emojiOn }) || null;
            } else {
              subtitleSrtPayload = null;
            }
          } else {
            subtitleSrtPayload = null;
          }
        } catch (_) {
          subtitleSrtPayload = null;
        }
      }

      if (needsRecut || captionsChanged || viralHookChanged) {
        const payload = {
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
          split_offset_b_y: Number(layoutSplitOffsetBY),
          // Subtitle parameters injected directly into the Recut API step
          captions_on: captionsOn,
          caption_position: appliedCaptionPosition,
          caption_font_size: appliedCaptionFontSize,
          caption_font_family: appliedCaptionFontFamily,
          caption_font_color: appliedCaptionFontColor,
          caption_stroke_color: appliedCaptionStrokeColor,
          caption_stroke_width: appliedCaptionStrokeWidth,
          caption_bold: appliedCaptionBold,
          caption_box_color: appliedCaptionBoxColor,
          caption_box_opacity: appliedCaptionBoxOpacity,
          caption_karaoke_mode: appliedCaptionKaraokeMode,
          caption_animation: appliedCaptionAnimation,
          caption_speaker_color_mode: appliedCaptionSpeakerColorMode,
          caption_offset_x: appliedCaptionOffsetX,
          caption_offset_y: appliedCaptionOffsetY,
          srt_content: subtitleSrtPayload,
          viral_hook_text: appliedViralHookText,
          viral_hook_start: appliedViralHookStart,
          viral_hook_duration: appliedViralHookDuration,
          viral_hook_font_size: appliedViralHookFontSize,
          viral_hook_font_family: appliedViralHookFontFamily,
          viral_hook_font_color: appliedViralHookFontColor,
          viral_hook_stroke_color: appliedViralHookStrokeColor,
          viral_hook_stroke_width: appliedViralHookStrokeWidth,
          viral_hook_bold: appliedViralHookBold,
          viral_hook_box_color: appliedViralHookBoxColor,
          viral_hook_box_opacity: appliedViralHookBoxOpacity,
          viral_hook_line_spacing: Number(viralHookLineSpacing || 0)
        };

        const recutRes = await apiFetch('/api/recut', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        if (!recutRes.ok) throw new Error(await recutRes.text());
        const recutData = await recutRes.json();

        // Single Output processing
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
        if (typeof recutData?.viral_hook_text === 'string') {
          appliedViralHookText = String(recutData.viral_hook_text || '').trim();
          setViralHookText(appliedViralHookText);
          setViralHookEnabled(Boolean(appliedViralHookText));
        }
        if (Number.isFinite(Number(recutData?.viral_hook_start))) {
          appliedViralHookStart = Math.max(0, Number(recutData.viral_hook_start));
          setViralHookStart(appliedViralHookStart);
        }
        if (Number.isFinite(Number(recutData?.viral_hook_duration))) {
          appliedViralHookDuration = Math.max(0, Number(recutData.viral_hook_duration));
          setViralHookDuration(appliedViralHookDuration);
        }
        if (Number.isFinite(Number(recutData?.viral_hook_font_size))) {
          appliedViralHookFontSize = clamp(Number(recutData.viral_hook_font_size), VIRAL_HOOK_FONT_SIZE_MIN, VIRAL_HOOK_FONT_SIZE_MAX);
          setViralHookFontSize(appliedViralHookFontSize);
        }
        if (typeof recutData?.viral_hook_font_family === 'string' && String(recutData.viral_hook_font_family).trim()) {
          appliedViralHookFontFamily = normalizeSubtitleFontFamily(recutData.viral_hook_font_family);
          setViralHookFontFamily(appliedViralHookFontFamily);
        }
        if (typeof recutData?.viral_hook_font_color === 'string' && String(recutData.viral_hook_font_color).trim()) {
          appliedViralHookFontColor = String(recutData.viral_hook_font_color).trim();
          setViralHookFontColor(appliedViralHookFontColor);
        }
        if (typeof recutData?.viral_hook_stroke_color === 'string' && String(recutData.viral_hook_stroke_color).trim()) {
          appliedViralHookStrokeColor = String(recutData.viral_hook_stroke_color).trim();
          setViralHookStrokeColor(appliedViralHookStrokeColor);
        }
        if (Number.isFinite(Number(recutData?.viral_hook_stroke_width))) {
          appliedViralHookStrokeWidth = clamp(Number(recutData.viral_hook_stroke_width), 0, 8);
          setViralHookStrokeWidth(appliedViralHookStrokeWidth);
        }
        if (typeof recutData?.viral_hook_bold === 'boolean') {
          appliedViralHookBold = Boolean(recutData.viral_hook_bold);
          setViralHookBold(appliedViralHookBold);
        }
        if (typeof recutData?.viral_hook_box_color === 'string' && String(recutData.viral_hook_box_color).trim()) {
          appliedViralHookBoxColor = String(recutData.viral_hook_box_color).trim();
          setViralHookBoxColor(appliedViralHookBoxColor);
        }
        if (Number.isFinite(Number(recutData?.viral_hook_box_opacity))) {
          appliedViralHookBoxOpacity = clamp(Number(recutData.viral_hook_box_opacity), 0, 100);
          setViralHookBoxOpacity(appliedViralHookBoxOpacity);
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

      if (dubbingEnabled) {
        if (!hasElevenLabsKey) {
          throw new Error('Falta la API Key de ElevenLabs en Configuración para aplicar doblaje.');
        }
        if (!appliedDubTargetLanguage) {
          appliedDubTargetLanguage = 'es';
        }

        const translateRes = await apiFetch('/api/translate', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-ElevenLabs-Key': resolvedElevenLabsKey
          },
          body: JSON.stringify({
            job_id: jobId,
            clip_index: Number(clipIndex),
            target_language: appliedDubTargetLanguage,
            source_language: appliedDubSourceLanguage !== 'auto' ? appliedDubSourceLanguage : null,
            input_filename: workingFile || null
          })
        });
        if (!translateRes.ok) throw new Error(await parseApiErrorDetail(translateRes));
        const translateData = await translateRes.json();
        if (translateData?.new_video_url) {
          resultingUrl = getApiUrl(translateData.new_video_url);
          workingFile = extractFilename(translateData.new_video_url);
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
          caption_position: appliedCaptionPosition,
          caption_offset_x: appliedCaptionOffsetX,
          caption_offset_y: appliedCaptionOffsetY,
          caption_font_size: appliedCaptionFontSize,
          caption_font_family: appliedCaptionFontFamily,
          caption_font_color: appliedCaptionFontColor,
          caption_stroke_color: appliedCaptionStrokeColor,
          caption_stroke_width: appliedCaptionStrokeWidth,
          caption_bold: appliedCaptionBold,
          caption_box_color: appliedCaptionBoxColor,
          caption_box_opacity: appliedCaptionBoxOpacity,
          caption_karaoke_mode: appliedCaptionKaraokeMode,
          caption_animation: appliedCaptionAnimation,
          caption_speaker_color_mode: appliedCaptionSpeakerColorMode,
          viral_hook_text: appliedViralHookText,
          viral_hook_start: appliedViralHookStart,
          viral_hook_duration: appliedViralHookDuration,
          viral_hook_font_size: appliedViralHookFontSize,
          viral_hook_font_family: appliedViralHookFontFamily,
          viral_hook_font_color: appliedViralHookFontColor,
          viral_hook_stroke_color: appliedViralHookStrokeColor,
          viral_hook_stroke_width: appliedViralHookStrokeWidth,
          viral_hook_bold: appliedViralHookBold,
          viral_hook_box_color: appliedViralHookBoxColor,
          viral_hook_box_opacity: appliedViralHookBoxOpacity,
          viral_hook_line_spacing: Number(viralHookLineSpacing || 0),
          ...(dubbingEnabled
            ? {
              dub_target_language: appliedDubTargetLanguage,
              dub_source_language: appliedDubSourceLanguage !== 'auto' ? appliedDubSourceLanguage : null
            }
            : {})
        }
      });
      if (downloadAfter && resultingUrl) {
        void downloadVideoWithFallback(resultingUrl);
      }
      onClose && onClose();
    } catch (e) {
      setError(`No se pudo aplicar cambios: ${e.message}`);
    } finally {
      setIsApplying(false);
      setApplyAction('apply');
    }
  };

  const handleFastPreview = async (targetAspect = layoutAspect) => {
    if (!jobId) return;
    setIsRenderingFastPreview(true);
    setError('');

    const runLocalFastPreview = async () => {
      const primary = previewVideoRef.current;
      if (!primary) throw new Error('Preview local no disponible en este momento.');
      const clipStartAbs = Number(clip?.start || 0);
      const requestedStartAbs = Number.isFinite(Number(layoutStart)) ? Number(layoutStart) : clipStartAbs;
      const relativeStart = clamp(
        requestedStartAbs - clipStartAbs,
        0,
        Math.max(0, Number(timelineDuration || 0) - 0.05)
      );
      try {
        primary.currentTime = relativeStart;
      } catch (_) {
        // ignore seek issues
      }
      const secondary = previewSplitVideoRef.current;
      if (secondary) {
        try {
          secondary.currentTime = relativeStart;
        } catch (_) {
          // ignore seek issues
        }
      }
      const playPromise = primary.play();
      if (playPromise && typeof playPromise.catch === 'function') {
        await playPromise.catch(() => { });
      }
      setFastPreviewCaptionsBurned(false);
      setVideoLoadError('');
      setPreviewCurrentTime(relativeStart);
      setPreviewPlaying(true);
      setSavedPulse(false);
      if (fastPreviewStopTimerRef.current) {
        clearTimeout(fastPreviewStopTimerRef.current);
      }
      fastPreviewStopTimerRef.current = setTimeout(() => {
        const pv = previewVideoRef.current;
        const sv = previewSplitVideoRef.current;
        if (pv) pv.pause();
        if (sv) sv.pause();
        setPreviewPlaying(false);
        fastPreviewStopTimerRef.current = null;
      }, 3200);
    };

    try {
      const startCandidate = Number.isFinite(Number(layoutStart)) ? Number(layoutStart) : Number(clip?.start || 0);
      const payload = {
        job_id: jobId,
        clip_index: clipIndex,
        input_filename: null,
        start: Math.max(0, Number(startCandidate || 0)),
        duration: 3.2,
        aspect_ratio: targetAspect === '16:9' ? '16:9' : '9:16',
        fit_mode: layoutFitMode,
        zoom: Number(layoutZoom),
        offset_x: Number(layoutOffsetX),
        offset_y: Number(layoutOffsetY),
        captions_on: Boolean(captionsOn),
        caption_position: position,
        caption_font_size: Number(fontSize),
        caption_font_family: String(fontFamily || 'Montserrat'),
        caption_font_color: String(fontColor || '#FFFFFF'),
        caption_stroke_color: String(strokeColor || '#000000'),
        caption_stroke_width: Number(strokeWidth),
        caption_bold: Boolean(bold),
        caption_box_color: String(boxColor || '#000000'),
        caption_box_opacity: Number(boxOpacity),
        caption_karaoke_mode: Boolean(karaokeMode),
        caption_animation: String(subtitleAnimation || 'none'),
        caption_speaker_color_mode: Boolean(speakerColorMode),
        caption_offset_x: Number(captionOffsetX),
        caption_offset_y: Number(captionOffsetY),
        srt_content: captionsOn ? (srtContent || null) : null
      };
      const res = await apiFetch('/api/clip/fast-preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!res.ok) throw new Error(await parseApiErrorDetail(res));
      const data = await res.json();
      const nextUrl = getApiUrl(data?.preview_video_url || '');
      if (!nextUrl) throw new Error('No se recibió URL de preview.');
      setPreviewVideoUrl(nextUrl);
      setFastPreviewCaptionsBurned(Boolean(data?.captions_burned));
      setVideoLoadError('');
      setPreviewCurrentTime(0);
      setPreviewPlaying(false);
      setSavedPulse(false);
    } catch (e) {
      const msg = String(e?.message || 'error desconocido');
      const lower = msg.toLowerCase();
      const canFallbackLocal = (
        lower.includes('not found')
        || lower.includes('404')
        || lower.includes('failed to fetch')
        || lower.includes('networkerror')
      );
      if (canFallbackLocal) {
        try {
          await runLocalFastPreview();
          setError('');
          return;
        } catch (localErr) {
          const localMsg = String(localErr?.message || 'error local desconocido');
          setError(`No se pudo generar preview rápido: ${localMsg}`);
          setTimeout(() => setError(''), 3500);
          return;
        }
      }
      setError(`No se pudo generar preview rápido: ${msg}`);
      setTimeout(() => setError(''), 3500);
    } finally {
      setIsRenderingFastPreview(false);
    }
  };

  const togglePreviewPlayback = () => {
    const video = previewVideoRef.current;
    if (!video) return;
    if (video.paused) {
      video.play().catch(() => { });
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

  const startViralHookDrag = (event, mode = 'move') => {
    if (!viralHookEnabled) return;
    event.preventDefault();
    event.stopPropagation();
    const pointerTime = getTimelineTimeFromClientX(event.clientX);
    if (pointerTime === null) return;
    viralHookDragRef.current = {
      mode,
      pointerTime,
      originalStart: Number(viralHookTimelineStart || 0),
      originalEnd: Number(viralHookTimelineEnd || viralHookTimelineDuration)
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

      if (viralHookDragRef.current) {
        const drag = viralHookDragRef.current;
        const pointerTime = getTimelineTimeFromClientX(event.clientX);
        if (pointerTime !== null) {
          const delta = pointerTime - drag.pointerTime;
          const minDuration = 0.4;
          const origStart = Number(drag.originalStart || 0);
          const origEnd = Number(drag.originalEnd || origStart + minDuration);
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

          setViralHookStart(Number(nextStart.toFixed(3)));
          setViralHookDuration(Number(Math.max(minDuration, nextEnd - nextStart).toFixed(3)));
          setSavedPulse(false);
        }
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
        let nextX = drag.startOffsetX + (deltaEffectiveX / CAPTION_OFFSET_FACTOR);
        let nextY = drag.startOffsetY + (deltaEffectiveY / CAPTION_OFFSET_FACTOR);

        let guideX = false;
        let guideY = false;
        const centeredYRaw = (50 - captionAnchorTopPercent) / CAPTION_OFFSET_FACTOR;
        if (Math.abs(nextX * CAPTION_OFFSET_FACTOR) <= CAPTION_CENTER_SNAP_THRESHOLD) {
          nextX = 0;
          guideX = true;
        }
        if (Math.abs((nextY - centeredYRaw) * CAPTION_OFFSET_FACTOR) <= CAPTION_CENTER_SNAP_THRESHOLD) {
          nextY = centeredYRaw;
          guideY = true;
        }

        setCaptionCenterGuides({ x: guideX, y: guideY });
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
      if (viralHookDragRef.current) {
        viralHookDragRef.current = null;
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
      setCaptionCenterGuides({ x: false, y: false });
    };

    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', onUp);
    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
    };
  }, [
    timelineDuration,
    viralHookEnabled,
    viralHookTimelineDuration,
    viralHookTimelineStart,
    viralHookTimelineEnd,
    baseClipStart,
    selectionStartRel,
    selectionEndRel,
    snapToNearest,
    captionAnchorTopPercent,
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
              onClick={() => {
                // Reset all settings to factory defaults
                applyPreset(CAPTION_PRESETS[0].id);
                setCaptionOffsetX(0);
                setCaptionOffsetY(10);
                setViralHookText('');
                setViralHookEnabled(false);
                setViralHookStart(0);
                setViralHookDuration(VIRAL_HOOK_DEFAULT_DURATION);
                setViralHookLineSpacing(0);
                setLayoutZoom(1.0);
                setLayoutOffsetX(0);
                setLayoutOffsetY(0);
                setLayoutFitMode('cover');
                setLayoutMode('single');
                setLayoutAutoSmart(false);
                setEmojiOn(true);
                setPunctuationOn(true);
                setSpeakerColorMode(false);
                setSavedPulse(false);
                setSelectedPreset(CAPTION_PRESETS[0].id);
              }}
              className="px-3 py-2 rounded-full border border-amber-300 dark:border-amber-600 bg-white dark:bg-slate-800 text-amber-700 dark:text-amber-300 text-xs font-medium hover:bg-amber-50 dark:hover:bg-amber-900/20 inline-flex items-center gap-1.5"
              title="Restaurar todos los ajustes a los valores por defecto"
            >
              <RotateCcw size={13} />
              Reset
            </button>
            <button
              type="button"
              onClick={() => handleApply({ downloadAfter: false })}
              disabled={isApplying}
              className="px-4 py-2 rounded-full bg-violet-600 hover:bg-violet-700 !text-white text-sm font-semibold shadow-sm shadow-violet-900/20 disabled:opacity-60 inline-flex items-center gap-2 transition-colors"
            >
              {isApplying && applyAction === 'apply' ? <Loader2 size={15} className="animate-spin" /> : <Sparkles size={15} />}
              {isApplying ? (applyAction === 'apply' ? 'Aplicando...' : 'Procesando...') : 'Aplicar'}
            </button>
            <button
              type="button"
              onClick={() => handleApply({ downloadAfter: true })}
              disabled={isApplying}
              className="px-4 py-2 rounded-full bg-emerald-600 hover:bg-emerald-700 !text-white text-sm font-semibold shadow-sm shadow-emerald-900/20 disabled:opacity-60 inline-flex items-center gap-2 transition-colors"
              title="Aplica cambios y descarga el clip final"
            >
              {isApplying && applyAction === 'apply_download' ? <Loader2 size={15} className="animate-spin" /> : <Download size={15} />}
              {isApplying
                ? (applyAction === 'apply_download' ? 'Aplicando y descargando...' : 'Procesando...')
                : 'Aplicar y descargar'}
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
                  <div className="mb-3 space-y-2">
                    <h3 className="text-lg font-semibold text-slate-800 dark:text-slate-100">Transcripción</h3>
                    <div className="grid grid-cols-1 gap-2">
                      <SettingToggle
                        label="Solo transcripción"
                        tooltip="Vista limpia en texto continuo, sin tarjetas por segmento."
                        checked={transcriptPlainMode}
                        onChange={() => setTranscriptPlainMode((v) => !v)}
                      />
                      <SettingToggle
                        label="Descripciones de escena"
                        tooltip="Muestra descripciones visuales si vienen en el transcript del backend."
                        checked={transcriptSceneDescriptionsOn}
                        onChange={() => setTranscriptSceneDescriptionsOn((v) => !v)}
                      />
                    </div>
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
                      const sceneDescription = String(seg?.scene_description || '').trim();
                      const showSceneDescription = transcriptSceneDescriptionsOn && sceneDescription;
                      if (transcriptPlainMode) {
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
                            className={`cursor-pointer px-1 py-1.5 ${isActive ? 'bg-violet-50/80 dark:bg-violet-900/20 rounded-md' : ''}`}
                          >
                            <p className={`text-[13px] leading-relaxed ${isActive ? 'text-violet-800 dark:text-violet-200' : 'text-slate-700 dark:text-slate-200'}`}>
                              {seg.text}
                            </p>
                            {showSceneDescription && (
                              <p className="mt-1 rounded-md bg-slate-900/90 px-2 py-1 text-[12px] leading-snug text-slate-200">
                                {`${Math.max(0, Number(seg.duration || 0)).toFixed(1)}s: ${sceneDescription}`}
                              </p>
                            )}
                          </div>
                        );
                      }
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
                          className={`rounded-lg border p-2.5 cursor-pointer transition-colors ${isActive
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
                          {showSceneDescription && (
                            <p className="mt-1.5 rounded-md bg-slate-900/90 px-2 py-1 text-[12px] leading-snug text-slate-200">
                              {`${Math.max(0, Number(seg.duration || 0)).toFixed(1)}s: ${sceneDescription}`}
                            </p>
                          )}
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
                        <span
                          className="inline-flex items-center justify-center rounded-full border border-amber-300/80 bg-amber-100/80 px-1.5 py-1 text-amber-700 dark:border-amber-700/80 dark:bg-amber-900/25 dark:text-amber-300"
                          title="Tip: usa Opciones para editar visibilidad, puntuación, emojis y karaoke."
                          aria-label="Tip de subtítulos"
                        >
                          <Lightbulb size={12} />
                        </span>
                        {karaokeMode && (
                          <span className="text-[11px] px-2 py-0.5 rounded-full border border-violet-300 bg-violet-100 dark:border-violet-600 dark:bg-violet-900/35 text-violet-700 dark:text-violet-300">
                            Karaoke activo
                          </span>
                        )}
                      </div>
                      <button
                        type="button"
                        onClick={() => setShowCaptionSettings((v) => !v)}
                        className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border text-xs font-medium transition-colors ${showCaptionSettings
                          ? 'border-violet-400 bg-violet-100 dark:bg-violet-900/25 text-violet-700 dark:text-violet-300'
                          : 'border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700'
                          }`}
                        title={showCaptionSettings ? 'Ocultar opciones de subtítulos' : 'Mostrar opciones de subtítulos'}
                      >
                        <Menu size={14} />
                        Opciones
                      </button>
                    </div>
                    {showCaptionSettings && (
                      <div className="mt-3 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50/80 dark:bg-slate-900/50 p-3">
                        <div className="grid grid-cols-2 gap-2">
                          <SettingToggle
                            label="Mostrar subtítulos"
                            tooltip="Activa u oculta los subtítulos en la vista previa y en la exportación."
                            checked={captionsOn}
                            onChange={() => setCaptionsOn((v) => !v)}
                          />
                          <SettingToggle
                            label="Puntuación"
                            tooltip="Si lo desactivas, se eliminan signos como comas y puntos para un estilo más limpio."
                            checked={punctuationOn}
                            onChange={() => setPunctuationOn((v) => !v)}
                          />
                          <SettingToggle
                            label="Emoji"
                            tooltip="Muestra emojis sugeridos al inicio de cada línea cuando están disponibles."
                            checked={emojiOn}
                            onChange={() => setEmojiOn((v) => !v)}
                          />
                          <SettingToggle
                            label="Modo karaoke"
                            tooltip="Resalta palabra por palabra durante la reproducción para dar efecto karaoke."
                            checked={karaokeMode}
                            onChange={() => setKaraokeMode((v) => !v)}
                          />
                        </div>
                      </div>
                    )}
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
                    <p className="text-xs font-semibold text-zinc-500 mb-2">Tamaño de fuente</p>
                    <div className="flex items-center gap-3">
                      <input
                        type="range"
                        min={12}
                        max={84}
                        step={1}
                        value={fontSize}
                        onChange={(e) => {
                          setFontSize(Number(e.target.value));
                          setSavedPulse(false);
                        }}
                        className="flex-1 accent-violet-500 h-1.5"
                      />
                      <span className="text-xs font-mono text-zinc-400 w-8 text-right">{fontSize}</span>
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

                  <details className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50/70 dark:bg-slate-900/40 px-3 py-2">
                    <summary className="cursor-pointer select-none text-xs font-semibold text-slate-600 dark:text-slate-300">
                      Ajustes avanzados
                    </summary>
                    <div className="mt-3 space-y-3">
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
                              setPreviewVideoUrl('');
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
                              setPreviewVideoUrl('');
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
                          <input type="number" min="12" max="84" value={fontSize} onChange={(e) => setFontSize(Number(e.target.value || 50))} className="mt-1 w-full rounded-md border border-black/10 dark:border-white/10 bg-white/80 dark:bg-black/20 p-2 text-sm" />
                        </label>
                        <label className="text-xs text-zinc-600 dark:text-zinc-300">Fuente
                          <select
                            value={fontFamily}
                            onChange={(e) => setFontFamily(normalizeSubtitleFontFamily(e.target.value))}
                            className="mt-1 w-full rounded-md border border-black/10 dark:border-white/10 bg-white/80 dark:bg-black/20 p-2 text-sm"
                          >
                            {captionFontOptions.map((font) => (
                              <option key={font.value} value={font.value}>
                                {font.label}
                                {font.available === false ? ' (no disponible)' : ''}
                              </option>
                            ))}
                          </select>
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
                      <div className="grid grid-cols-2 gap-3">
                        <label className="text-xs text-zinc-600 dark:text-zinc-300">Animación
                          <select
                            value={subtitleAnimation}
                            onChange={(e) => {
                              const next = String(e.target.value || 'none').toLowerCase();
                              setSubtitleAnimation(['none', 'pop', 'bounce', 'slide'].includes(next) ? next : 'none');
                              setSavedPulse(false);
                            }}
                            className="mt-1 w-full rounded-md border border-black/10 dark:border-white/10 bg-white/80 dark:bg-black/20 p-2 text-sm"
                          >
                            <option value="none">Sin animación</option>
                            <option value="pop">Pop</option>
                            <option value="bounce">Bounce</option>
                            <option value="slide">Slide</option>
                          </select>
                        </label>
                        <div className="text-xs text-zinc-600 dark:text-zinc-300">
                          Color por hablante
                          <button
                            type="button"
                            role="switch"
                            aria-checked={speakerColorMode}
                            onClick={() => {
                              setSpeakerColorMode((v) => !v);
                              setSavedPulse(false);
                            }}
                            className={`mt-1 flex h-10 w-full items-center justify-between rounded-md border px-2 text-xs transition-colors ${speakerColorMode
                              ? 'border-violet-400 bg-violet-100/70 dark:bg-violet-900/25 text-violet-700 dark:text-violet-300'
                              : 'border-black/10 dark:border-white/10 bg-white/80 dark:bg-black/20 text-zinc-600 dark:text-zinc-300'
                              }`}
                            title="Asigna color de resaltado según el hablante detectado en transcript."
                          >
                            <span>{speakerColorMode ? 'Activo' : 'Inactivo'}</span>
                            <span className={`inline-block h-4 w-7 rounded-full ${speakerColorMode ? 'bg-violet-500' : 'bg-zinc-300 dark:bg-zinc-600'}`}>
                              <span className={`block h-3 w-3 rounded-full bg-white transition-transform ${speakerColorMode ? 'translate-x-3.5' : 'translate-x-0.5'} mt-0.5`} />
                            </span>
                          </button>
                        </div>
                      </div>
                      <label className="inline-flex items-center gap-2 text-xs text-zinc-600 dark:text-zinc-300">
                        <input type="checkbox" checked={bold} onChange={(e) => setBold(e.target.checked)} /> Negrita
                      </label>
                    </div>
                  </details>
                </div>
              )}

              {section === 'subtitle_edit' && (
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-lg font-semibold text-zinc-800 dark:text-zinc-100">Editar subtítulos</h3>
                    <button
                      type="button"
                      onClick={autoSuggestEmojis}
                      className="text-xs px-2 py-1 rounded-md border border-emerald-300 bg-emerald-100/80 dark:border-emerald-700 dark:bg-emerald-900/20 text-emerald-700 dark:text-emerald-300 hover:bg-emerald-100 dark:hover:bg-emerald-900/30 inline-flex items-center gap-1"
                      title="Auto-sugerencia local por contenido"
                    >
                      <Sparkles size={12} />
                      IA local
                    </button>
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
                        className={`rounded-lg border p-2.5 transition-colors cursor-pointer ${activeSubtitleEntry?.id === entry.id
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

              {section === 'viral_hook' && (
                <div className="space-y-4">
                  <div>
                    <h3 className="text-lg font-semibold text-zinc-800 dark:text-zinc-100 flex items-center gap-2">
                      <Sparkles size={18} className="text-amber-500" /> Viral Hook Overlay
                    </h3>
                    <p className="text-xs text-zinc-500 mt-1">
                      Muestra un título-resumen en la parte superior del video (en caja) para captar atención. Tiene estilo independiente de subtítulos. Por defecto sale de 0.0s a 3.0s.
                    </p>
                  </div>

                  <SettingToggle
                    label="Activar hook viral"
                    tooltip="Cuando está apagado, no se renderiza el hook viral en el export."
                    checked={viralHookEnabled}
                    onChange={() => {
                      const hookFallback = resolveDefaultViralHookText(clip, clipIndex, currentVideoUrl);
                      setViralHookEnabled((v) => {
                        const next = !v;
                        if (next && !String(viralHookText || '').trim() && hookFallback) {
                          setViralHookText(hookFallback);
                        }
                        return next;
                      });
                      setSavedPulse(false);
                    }}
                  />

                  <div className="space-y-3 p-4 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800/80">
                    <div className="flex flex-col gap-2">
                      <p className="text-xs font-semibold text-zinc-500 mb-1">Preset de estilo</p>
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
                              key={`viral-preset-${preset.id}`}
                              type="button"
                              onClick={() => {
                                applyViralHookPreset(preset.id);
                                setSavedPulse(false);
                              }}
                              className={`rounded-xl border p-2 text-left transition-colors ${selectedViralHookPreset === preset.id
                                ? 'border-violet-400 bg-violet-50 dark:bg-violet-500/10 shadow-[0_0_0_1px_rgba(139,92,246,0.2)]'
                                : 'border-black/10 dark:border-white/10 bg-white/70 dark:bg-white/5 hover:bg-white dark:hover:bg-white/10'
                                }`}
                              title={`Aplicar preset ${preset.name} al hook`}
                            >
                              <div
                                className="relative rounded-lg h-[72px] overflow-hidden border border-white/10"
                                style={{ background: previewBg }}
                              >
                                <div className="absolute inset-0 bg-[radial-gradient(circle_at_25%_20%,rgba(255,255,255,0.12),transparent_45%)]" />
                                <div className="absolute inset-x-2 top-1.5 text-center">
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
                                        <div key={`viral-${preset.id}-sample-line-${lineIdx}`}>
                                          {words.map((word, idx) => {
                                            const currentWordIndex = wordCounter;
                                            wordCounter += 1;
                                            const isHighlight = currentWordIndex === highlightWordIndex;
                                            return (
                                              <span
                                                key={`viral-${preset.id}-sample-word-${lineIdx}-${idx}`}
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
                              <div className="mt-1 text-[11px] font-semibold text-zinc-700 dark:text-zinc-100 truncate">{preset.name}</div>
                            </button>
                          );
                        })}
                      </div>
                    </div>

                    <div className="flex flex-col gap-2">
                      <div className="flex justify-between items-center">
                        <label className="text-sm font-medium text-slate-700 dark:text-slate-200 text-left">Texto del Hook</label>
                        <button
                          className="text-xs text-violet-600 dark:text-violet-400 font-medium hover:underline bg-transparent"
                          onClick={() => {
                            if (clip?.transcript && typeof clip.transcript === 'object' && clip.transcript.text) {
                              const sentenceMatch = clip.transcript.text.split(/[.?!]/)[0];
                              if (sentenceMatch) {
                                setViralHookText(sentenceMatch.trim() + " 🤯");
                              }
                            } else if (subtitleEntries && subtitleEntries.length > 0) {
                              setViralHookText(subtitleEntries[0].text + " 🤯");
                            }
                          }}
                        >
                          Generar con IA (Transcript)
                        </button>
                      </div>
                      <textarea
                        value={viralHookText}
                        onChange={(e) => {
                          setViralHookText(e.target.value);
                          if (!viralHookEnabled && String(e.target.value || '').trim()) {
                            setViralHookEnabled(true);
                          }
                          setSavedPulse(false);
                        }}
                        placeholder="Ej: Un pasajero me reconoció en el avión 🤯"
                        rows={2}
                        disabled={!viralHookEnabled}
                        className="w-full rounded-md border border-slate-300 dark:border-slate-600 bg-transparent px-3 py-2 text-sm text-slate-800 dark:text-slate-100 focus:border-violet-500 focus:outline-none"
                      />
                    </div>

                    <div className="flex flex-col gap-2">
                      <label className="text-sm font-medium text-slate-700 dark:text-slate-200 text-left">
                        Inicio: <span className="text-violet-600 dark:text-violet-400">{viralHookTimelineStart.toFixed(1)}s</span>
                      </label>
                      <input
                        type="range"
                        min="0"
                        max={String(Math.max(0, timelineDuration - 0.4))}
                        step="0.1"
                        value={viralHookTimelineStart}
                        disabled={!viralHookEnabled}
                        onChange={(e) => {
                          const nextStart = clamp(Number(e.target.value), 0, Math.max(0, timelineDuration - 0.4));
                          setViralHookStart(nextStart);
                          setSavedPulse(false);
                        }}
                        className="w-full accent-violet-600"
                      />
                    </div>

                    <div className="flex flex-col gap-2">
                      <label className="text-sm font-medium text-slate-700 dark:text-slate-200 text-left">
                        Duración: <span className="text-violet-600 dark:text-violet-400">{viralHookTimelineDuration.toFixed(1)}s</span>
                      </label>
                      <input
                        type="range"
                        min="0.4"
                        max={String(Math.max(0.4, timelineDuration - viralHookTimelineStart))}
                        step="0.1"
                        value={viralHookTimelineDuration}
                        disabled={!viralHookEnabled}
                        onChange={(e) => {
                          setViralHookDuration(Number(e.target.value));
                          setSavedPulse(false);
                        }}
                        className="w-full accent-violet-600"
                      />
                    </div>

                    <div className="flex flex-col gap-2">
                      <label className="text-sm font-medium text-slate-700 dark:text-slate-200 text-left">
                        Tamaño: <span className="text-violet-600 dark:text-violet-400">{Math.round(viralHookFontSize)} px</span>
                      </label>
                      <input
                        type="range"
                        min={String(VIRAL_HOOK_FONT_SIZE_MIN)}
                        max={String(VIRAL_HOOK_FONT_SIZE_MAX)}
                        step="1"
                        value={viralHookFontSize}
                        disabled={!viralHookEnabled}
                        onChange={(e) => {
                          setViralHookFontSize(clamp(Number(e.target.value), VIRAL_HOOK_FONT_SIZE_MIN, VIRAL_HOOK_FONT_SIZE_MAX));
                          setSelectedViralHookPreset('');
                          setSavedPulse(false);
                        }}
                        className="w-full accent-violet-600"
                      />
                    </div>

                    <div className="flex flex-col gap-2">
                      <label className="text-sm font-medium text-slate-700 dark:text-slate-200 text-left">
                        Interlineado: <span className="text-violet-600 dark:text-violet-400">{viralHookLineSpacing}</span>
                      </label>
                      <input
                        type="range"
                        min="-10"
                        max="20"
                        step="1"
                        value={viralHookLineSpacing}
                        disabled={!viralHookEnabled}
                        onChange={(e) => {
                          setViralHookLineSpacing(Number(e.target.value));
                          setSavedPulse(false);
                        }}
                        className="w-full accent-violet-600"
                      />
                    </div>

                    <details className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50/70 dark:bg-slate-900/40 px-3 py-2">
                      <summary className="cursor-pointer select-none text-xs font-semibold text-slate-600 dark:text-slate-300">
                        Ajustes avanzados del Hook
                      </summary>
                      <div className="mt-3 space-y-3">
                        <div className="grid grid-cols-2 gap-3">
                          <label className="text-xs text-zinc-600 dark:text-zinc-300">Fuente
                            <select
                              value={viralHookFontFamily}
                              onChange={(e) => {
                                setViralHookFontFamily(normalizeSubtitleFontFamily(e.target.value));
                                setSelectedViralHookPreset('');
                                setSavedPulse(false);
                              }}
                              className="mt-1 w-full rounded-md border border-black/10 dark:border-white/10 bg-white/80 dark:bg-black/20 p-2 text-sm"
                            >
                              {captionFontOptions.map((font) => (
                                <option key={`viral-hook-font-${font.value}`} value={font.value}>
                                  {font.label}
                                  {font.available === false ? ' (no disponible)' : ''}
                                </option>
                              ))}
                            </select>
                          </label>
                          <label className="text-xs text-zinc-600 dark:text-zinc-300">Grosor contorno
                            <input
                              type="number"
                              min="0"
                              max="8"
                              value={viralHookStrokeWidth}
                              onChange={(e) => {
                                setViralHookStrokeWidth(clamp(Number(e.target.value || 0), 0, 8));
                                setSelectedViralHookPreset('');
                                setSavedPulse(false);
                              }}
                              className="mt-1 w-full rounded-md border border-black/10 dark:border-white/10 bg-white/80 dark:bg-black/20 p-2 text-sm"
                            />
                          </label>
                          <label className="text-xs text-zinc-600 dark:text-zinc-300">Color texto
                            <input
                              type="color"
                              value={viralHookFontColor}
                              onChange={(e) => {
                                setViralHookFontColor(e.target.value);
                                setSelectedViralHookPreset('');
                                setSavedPulse(false);
                              }}
                              className="mt-1 h-10 w-full rounded-md border border-black/10 dark:border-white/10 bg-white/80 dark:bg-black/20 p-1"
                            />
                          </label>
                          <label className="text-xs text-zinc-600 dark:text-zinc-300">Color contorno
                            <input
                              type="color"
                              value={viralHookStrokeColor}
                              onChange={(e) => {
                                setViralHookStrokeColor(e.target.value);
                                setSelectedViralHookPreset('');
                                setSavedPulse(false);
                              }}
                              className="mt-1 h-10 w-full rounded-md border border-black/10 dark:border-white/10 bg-white/80 dark:bg-black/20 p-1"
                            />
                          </label>
                          <label className="text-xs text-zinc-600 dark:text-zinc-300">Color caja
                            <input
                              type="color"
                              value={viralHookBoxColor}
                              onChange={(e) => {
                                setViralHookBoxColor(e.target.value);
                                setSelectedViralHookPreset('');
                                setSavedPulse(false);
                              }}
                              className="mt-1 h-10 w-full rounded-md border border-black/10 dark:border-white/10 bg-white/80 dark:bg-black/20 p-1"
                            />
                          </label>
                          <label className="text-xs text-zinc-600 dark:text-zinc-300">Caja (%)
                            <input
                              type="number"
                              min="0"
                              max="100"
                              value={viralHookBoxOpacity}
                              onChange={(e) => {
                                setViralHookBoxOpacity(clamp(Number(e.target.value || 0), 0, 100));
                                setSelectedViralHookPreset('');
                                setSavedPulse(false);
                              }}
                              className="mt-1 w-full rounded-md border border-black/10 dark:border-white/10 bg-white/80 dark:bg-black/20 p-2 text-sm"
                            />
                          </label>
                        </div>
                        <label className="inline-flex items-center gap-2 text-xs text-zinc-600 dark:text-zinc-300">
                          <input
                            type="checkbox"
                            checked={viralHookBold}
                            onChange={(e) => {
                              setViralHookBold(e.target.checked);
                              setSelectedViralHookPreset('');
                              setSavedPulse(false);
                            }}
                          />
                          Negrita
                        </label>
                      </div>
                    </details>
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
                          onClick={() => {
                            const changed = handleLayoutAspectChange(ratio);
                            if (changed) void handleFastPreview(ratio);
                          }}
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
                          {['cover', 'contain', 'blur'].map((mode) => (
                            <button
                              key={mode}
                              type="button"
                              onClick={() => {
                                setLayoutFitMode(mode);
                                setPreviewVideoUrl('');
                                setSavedPulse(false);
                              }}
                              className={`rounded-lg px-3 py-2 text-sm border capitalize ${layoutFitMode === mode
                                ? 'border-violet-400 bg-violet-100 dark:bg-violet-900/20 text-violet-700 dark:text-violet-300'
                                : 'border-black/10 dark:border-white/10 text-zinc-700 dark:text-zinc-200'
                                }`}
                            >
                              {mode === 'cover' ? 'Cover' : mode === 'contain' ? 'Contain' : 'Blur'}
                            </button>
                          ))}
                        </div>
                        <p className="mt-1 text-[11px] text-zinc-500">
                          {layoutFitMode === 'contain'
                            ? 'Contain conserva todo el video y deja barras negras. Usa Cover para llenar todo el ancho/alto.'
                            : layoutFitMode === 'blur'
                              ? 'Blur conserva todo el video y rellena los lados con un fondo difuminado.'
                              : 'Cover rellena todo el cuadro y recorta excedentes para evitar barras.'}
                        </p>
                      </div>

                      <label className="text-xs text-zinc-600 dark:text-zinc-300 block">
                        Zoom ({layoutZoom.toFixed(2)}x)
                        <input
                          type="range"
                          min={String(layoutZoomMin)}
                          max="2.5"
                          step="0.01"
                          value={layoutZoom}
                          onChange={(e) => {
                            const val = clamp(Number(e.target.value || 1), layoutZoomMin, 2.5);
                            setLayoutZoom(val);
                            setPreviewVideoUrl('');
                            setSavedPulse(false);
                          }}
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
                          onChange={(e) => {
                            const next = clamp(Number(e.target.value || 0), -100, 100);
                            setLayoutOffsetX(next);
                            setPreviewVideoUrl('');
                            setSavedPulse(false);
                          }}
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
                          onChange={(e) => {
                            const next = clamp(Number(e.target.value || 0), -100, 100);
                            setLayoutOffsetY(next);
                            setPreviewVideoUrl('');
                            setSavedPulse(false);
                          }}
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
                              min={String(layoutZoomMin)}
                              max="2.5"
                              step="0.01"
                              value={layoutSplitZoomA}
                              onChange={(e) => setLayoutSplitZoomA(clamp(Number(e.target.value || 1), layoutZoomMin, 2.5))}
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
                              min={String(layoutZoomMin)}
                              max="2.5"
                              step="0.01"
                              value={layoutSplitZoomB}
                              onChange={(e) => setLayoutSplitZoomB(clamp(Number(e.target.value || 1), layoutZoomMin, 2.5))}
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

              {section === 'dubbing' && (
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold text-zinc-800 dark:text-zinc-100">Doblaje</h3>
                  <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50/70 dark:bg-slate-900/40 p-3 space-y-3">
                    <div className="flex items-center justify-between gap-3">
                      <div className="flex items-center gap-2">
                        <Languages size={15} className="text-emerald-600 dark:text-emerald-400" />
                        <p className="text-sm font-semibold text-slate-700 dark:text-slate-200">Doblaje ElevenLabs</p>
                      </div>
                      <span className={`text-[11px] px-2 py-0.5 rounded-full border ${hasElevenLabsKey
                        ? 'border-emerald-300 bg-emerald-100 text-emerald-700 dark:border-emerald-700/60 dark:bg-emerald-900/30 dark:text-emerald-300'
                        : 'border-amber-300 bg-amber-100 text-amber-700 dark:border-amber-700/60 dark:bg-amber-900/30 dark:text-amber-300'
                        }`}>
                        {hasElevenLabsKey ? 'API key OK' : 'Falta API key'}
                      </span>
                    </div>

                    <SettingToggle
                      label="Activar doblaje IA"
                      tooltip="Traduce la voz del clip con ElevenLabs al idioma destino al aplicar cambios."
                      checked={dubbingEnabled}
                      onChange={() => {
                        setDubbingEnabled((v) => !v);
                        setSavedPulse(false);
                      }}
                    />

                    <div className="grid grid-cols-1 gap-3">
                      <label className="text-xs text-zinc-600 dark:text-zinc-300">
                        Idioma destino
                        <select
                          value={dubbingTargetLanguage}
                          onChange={(e) => {
                            setDubbingTargetLanguage(String(e.target.value || 'es'));
                            setSavedPulse(false);
                          }}
                          disabled={!dubbingEnabled}
                          className="mt-1 w-full rounded-md border border-black/10 dark:border-white/10 bg-white/80 dark:bg-black/20 p-2 text-sm"
                        >
                          {dubbingLanguageOptions.map(([code, name]) => (
                            <option key={`dub-target-${code}`} value={code}>{name}</option>
                          ))}
                        </select>
                      </label>

                      <label className="text-xs text-zinc-600 dark:text-zinc-300">
                        Idioma origen (opcional)
                        <select
                          value={dubbingSourceLanguage}
                          onChange={(e) => {
                            setDubbingSourceLanguage(String(e.target.value || 'auto'));
                            setSavedPulse(false);
                          }}
                          disabled={!dubbingEnabled}
                          className="mt-1 w-full rounded-md border border-black/10 dark:border-white/10 bg-white/80 dark:bg-black/20 p-2 text-sm"
                        >
                          <option value="auto">Auto detectar</option>
                          {dubbingLanguageOptions.map(([code, name]) => (
                            <option key={`dub-source-${code}`} value={code}>{name}</option>
                          ))}
                        </select>
                      </label>
                    </div>

                    <p className="text-[11px] text-zinc-500">
                      {isLoadingDubbingLanguages
                        ? 'Cargando idiomas de doblaje...'
                        : 'El doblaje se procesa al pulsar Aplicar. Si no eliges idioma origen, ElevenLabs lo detecta automáticamente.'}
                    </p>
                    {!hasElevenLabsKey && (
                      <p className="text-[11px] text-amber-600 dark:text-amber-300">
                        Configura tu API key de ElevenLabs en Configuración para habilitar el doblaje.
                      </p>
                    )}
                  </div>
                </div>
              )}

              {error && (
                <div className="mt-4 rounded-lg border border-red-300 bg-red-50 text-red-700 dark:bg-red-900/20 dark:text-red-300 dark:border-red-700 px-3 py-2 text-sm">
                  {error}
                </div>
              )}
            </section>

            <section className="bg-slate-100 dark:bg-slate-900 p-4 md:p-6 flex flex-col min-h-0 overflow-y-auto custom-scrollbar">
              <div className="flex-1 rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-4 flex flex-col items-center justify-start gap-3">
                <div className="w-full px-1">
                  <div className="flex items-start gap-3">
                    <div className="min-w-0">
                      <p className="text-[15px] md:text-[20px] font-semibold leading-tight text-slate-900 dark:text-white" title={previewClipTitle}>
                        {previewClipTitle}
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
                          src={previewVideoUrl || activeSourceUrl}
                          className="w-full h-full"
                          style={{
                            ...(() => {
                              const zNum = Number(layoutSplitZoomA || 1);
                              const effectX = clamp(Number(layoutSplitOffsetAX || 0), -100, 100);
                              const effectY = clamp(Number(layoutSplitOffsetAY || 0), -100, 100);
                              const isFastPreview = Boolean(previewVideoUrl);

                              const transX = -(effectX / 100) * ((zNum - 1) / 2) * 100;
                              const transY = -(effectY / 100) * ((zNum - 1) / 2) * 100;

                              return {
                                objectFit: isFastPreview ? 'contain' : 'cover',
                                objectPosition: isFastPreview ? '50% 50%' : splitObjectPositionA,
                                transform: isFastPreview
                                  ? 'scale(1)'
                                  : `scale(${zNum}) translate(${transX}%, ${transY}%)`,
                                transformOrigin: 'center center'
                              };
                            })()
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
                          src={previewVideoUrl || activeSourceUrl}
                          className="w-full h-full pointer-events-none"
                          style={{
                            ...(() => {
                              const zNum = Number(layoutSplitZoomB || 1);
                              const effectX = clamp(Number(layoutSplitOffsetBX || 0), -100, 100);
                              const effectY = clamp(Number(layoutSplitOffsetBY || 0), -100, 100);
                              const isFastPreview = Boolean(previewVideoUrl);

                              const transX = -(effectX / 100) * ((zNum - 1) / 2) * 100;
                              const transY = -(effectY / 100) * ((zNum - 1) / 2) * 100;

                              return {
                                objectFit: isFastPreview ? 'contain' : 'cover',
                                objectPosition: isFastPreview ? '50% 50%' : splitObjectPositionB,
                                transform: isFastPreview
                                  ? 'scale(1)'
                                  : `scale(${zNum}) translate(${transX}%, ${transY}%)`,
                                transformOrigin: 'center center'
                              };
                            })()
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
                      src={previewVideoUrl || activeSourceUrl}
                      className="w-full h-full"
                      style={{
                        ...(() => {
                          const zNum = Number(layoutZoom || 1);
                          const effectX = clamp(Number(layoutOffsetX || 0), -100, 100);
                          const effectY = clamp(Number(layoutOffsetY || 0), -100, 100);
                          const isFastPreview = Boolean(previewVideoUrl);

                          // transX/Y enables panning through the extra overhang created by zNum > 1
                          // A scale of 1.5 creates a 25% overhang on each edge ((1.5-1)/2).
                          // We map effect from -100 to 100 to move from +overhang to -overhang.
                          const transX = -(effectX / 100) * ((zNum - 1) / 2) * 100;
                          const transY = -(effectY / 100) * ((zNum - 1) / 2) * 100;

                          return {
                            objectFit: isFastPreview ? 'contain' : 'cover',
                            objectPosition: (layoutAutoSmart || isFastPreview) ? '50% 50%' : manualLayoutObjectPosition,
                            transform: (layoutAutoSmart || isFastPreview)
                              ? 'scale(1)'
                              : `scale(${zNum}) translate(${transX}%, ${transY}%)`,
                            transformOrigin: 'center center'
                          };
                        })()
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

                        // If we tried to load uncut and it failed (e.g. old clip without uncut), fallback
                        if (activeSourceUrl === uncutVideoUrl && !uncutFailed) {
                          setUncutFailed(true);
                          return;
                        }

                        setVideoLoadError('El navegador no pudo reproducir este archivo en la vista previa.');
                      }}
                    />
                  )}

                  {section === 'layout' && !layoutAutoSmart && !isSplitLayout && (
                    <div className="absolute top-2 right-2 rounded-md bg-white/92 dark:bg-black/70 text-slate-900 dark:text-white text-[10px] px-2 py-1 pointer-events-none border border-slate-200/90 dark:border-white/20 shadow-sm">
                      Arrastra para mover
                    </div>
                  )}
                  {captionDragEnabled && (
                    <div className="absolute top-2 left-2 rounded-md bg-white/92 dark:bg-black/70 text-slate-900 dark:text-white text-[10px] px-2 py-1 pointer-events-none border border-slate-200/90 dark:border-white/20 shadow-sm">
                      Arrastra subtítulo
                    </div>
                  )}
                  {captionDragEnabled && isDraggingCaption && (captionCenterGuides.x || captionCenterGuides.y) && (
                    <>
                      {captionCenterGuides.x && (
                        <div className="pointer-events-none absolute top-0 bottom-0 left-1/2 -translate-x-1/2 w-[1px] bg-cyan-300/90 shadow-[0_0_0_1px_rgba(34,211,238,0.18)]" />
                      )}
                      {captionCenterGuides.y && (
                        <div className="pointer-events-none absolute left-0 right-0 top-1/2 -translate-y-1/2 h-[1px] bg-cyan-300/90 shadow-[0_0_0_1px_rgba(34,211,238,0.18)]" />
                      )}
                    </>
                  )}
                  {showPreviewViralHook && (
                    <div className="pointer-events-none absolute left-1/2 top-[8%] -translate-x-1/2 w-[86%] z-[4] flex justify-center">
                      <div className="max-w-full rounded-md border border-white/20 px-3 py-2 shadow-lg" style={{ backgroundColor: previewViralHookBoxBg }}>
                        <p
                          className="text-center break-words"
                          style={{
                            color: viralHookFontColor,
                            fontFamily: String(viralHookFontFamily || 'Montserrat'),
                            fontWeight: viralHookBold ? 700 : 600,
                            fontSize: `${previewViralHookFontSize}px`,
                            lineHeight: `${1.2 + (Number(viralHookLineSpacing || 0) * 0.02)}`,
                            WebkitTextStroke: `${previewViralHookStrokeWidth}px ${viralHookStrokeColor}`
                          }}
                        >
                          {previewViralHookText}
                        </p>
                      </div>
                    </div>
                  )}

                  {captionsOn && !fastPreviewCaptionsBurned && subtitleEntries && subtitleEntries.length > 0 && (
                    <SubtitleRenderer
                      currentTime={previewCurrentTime}
                      srtEntries={subtitleEntries}
                      fontSize={fontSize}
                      fontFamily={fontFamily}
                      fontColor={fontColor}
                      strokeColor={strokeColor}
                      strokeWidth={strokeWidth}
                      bold={bold}
                      boxColor={boxColor}
                      boxOpacity={boxOpacity}
                      karaokeMode={karaokeMode}
                      position={position}
                      offsetX={effectiveCaptionOffsetX}
                      offsetY={effectiveCaptionOffsetY}
                      animation={subtitleAnimation}
                      speakerColorMode={speakerColorMode}
                      isDragging={isDraggingCaption}
                      onMouseDown={startCaptionDrag}
                      captionDragEnabled={captionDragEnabled}
                    />
                  )}
                </div>
                {videoLoadError && (
                  <div className="mt-3 rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-700 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-300">
                    {videoLoadError}
                  </div>
                )}
              </div>

              <div className="mt-3 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 px-3 py-3 text-slate-700 dark:text-slate-200">
                {/* ── Controls Bar ── */}
                <div className="flex items-center justify-between gap-3 text-sm mb-2.5">
                  <div className="flex items-center gap-2">
                    <button type="button" onClick={() => setTimelineZoom((z) => clamp(z - 0.15, TIMELINE_ZOOM_MIN, TIMELINE_ZOOM_MAX))} className="w-8 h-8 rounded-md border border-slate-200 dark:border-slate-700 inline-flex items-center justify-center text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700" title="Alejar">
                      <ZoomOut size={15} />
                    </button>
                    <input type="range" min={String(TIMELINE_ZOOM_MIN)} max={String(TIMELINE_ZOOM_MAX)} step="0.05" value={timelineZoom} onChange={(e) => setTimelineZoom(clamp(Number(e.target.value || TIMELINE_ZOOM_DEFAULT), TIMELINE_ZOOM_MIN, TIMELINE_ZOOM_MAX))} className="w-28 accent-primary" title="Zoom" />
                    <span className="text-[11px] text-slate-500 dark:text-slate-400 tabular-nums min-w-[44px]">{`${Math.round(timelineZoom * 100)}%`}</span>
                    <button type="button" onClick={() => setTimelineZoom((z) => clamp(z + 0.15, TIMELINE_ZOOM_MIN, TIMELINE_ZOOM_MAX))} className="w-8 h-8 rounded-md border border-slate-200 dark:border-slate-700 inline-flex items-center justify-center text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700" title="Acercar">
                      <ZoomIn size={15} />
                    </button>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-slate-600 dark:text-slate-300 tabular-nums">{formatTimelineTime(previewCurrentTime)}</span>
                    <button type="button" onClick={togglePreviewPlayback} className="w-8 h-8 rounded-full border border-slate-200 dark:border-slate-700 inline-flex items-center justify-center hover:bg-slate-100 dark:hover:bg-slate-700">
                      {previewPlaying ? <Pause size={14} /> : <Play size={14} />}
                    </button>
                    <span className="text-slate-500 dark:text-slate-400 tabular-nums">{formatTimelineTime(timelineDuration)}</span>
                    <button type="button" onClick={cyclePlaybackRate} className="min-w-[44px] px-2 h-8 rounded-md border border-slate-200 dark:border-slate-700 text-xs font-semibold hover:bg-slate-100 dark:hover:bg-slate-700">{`${playbackRate}x`}</button>
                    <button type="button" onClick={() => setSnapEnabled((v) => !v)} className={`h-8 px-2.5 rounded-md border text-xs font-semibold inline-flex items-center gap-1.5 ${snapEnabled ? 'border-emerald-300 bg-emerald-100 text-emerald-700 dark:border-emerald-700/60 dark:bg-emerald-900/30 dark:text-emerald-300' : 'border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700'}`} title="Snap magnético: cuando está ON el cursor se adhiere automáticamente a los puntos del transcript al arrastrar">
                      <Crosshair size={13} />{snapEnabled ? 'Snap ON' : 'Snap OFF'}
                    </button>
                    <div className="w-px h-5 bg-slate-200 dark:bg-slate-600" />
                    <button type="button" onClick={() => { const s = snapToNearest(previewCurrentTime); setLayoutStart(Number((baseClipStart + clamp(s, 0, Math.max(0, selectionEndRel - 0.2))).toFixed(3))); }} className="h-8 px-2.5 rounded-md border border-slate-200 dark:border-slate-700 text-xs font-semibold text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700" title="Marcar IN: establece el punto de inicio del clip en la posición actual del cursor (tecla I en editores profesionales)">‹ Marcar IN</button>
                    <button type="button" onClick={() => { const s = snapToNearest(previewCurrentTime); setLayoutEnd(Number((baseClipStart + clamp(s, selectionStartRel + 0.2, timelineDuration)).toFixed(3))); }} className="h-8 px-2.5 rounded-md border border-slate-200 dark:border-slate-700 text-xs font-semibold text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700" title="Marcar OUT: establece el punto final del clip en la posición actual del cursor (tecla O en editores profesionales)">Marcar OUT ›</button>
                  </div>
                </div>

                {/* ── NLE Layout: sticky labels + scrollable tracks ── */}
                <div className="flex rounded-lg border border-slate-200 dark:border-slate-700 overflow-hidden">

                  {/* Fixed label sidebar */}
                  <div className="shrink-0 border-r border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900/60 flex flex-col w-[52px]">
                    {/* Tick ruler spacer */}
                    <div className="h-[28px] border-b border-slate-200 dark:border-slate-700" />
                    {/* Video label */}
                    <div className="h-[52px] border-b border-slate-200 dark:border-slate-700 flex items-center justify-center px-0" title="Track de video">
                      <div className="w-4 h-4 rounded-sm bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center shrink-0">
                        <svg width="8" height="8" viewBox="0 0 8 8" fill="white"><polygon points="2,1 7,4 2,7" /></svg>
                      </div>
                    </div>
                    {/* Subs label */}
                    <div className="h-[58px] border-b border-slate-200 dark:border-slate-700 flex items-center justify-center px-0" title="Track de subtítulos">
                      <div className="w-4 h-4 rounded-sm bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center shrink-0">
                        <svg width="9" height="7" viewBox="0 0 9 7" fill="none"><rect x="0" y="0" width="9" height="2" rx="1" fill="white" /><rect x="0" y="4" width="6" height="2" rx="1" fill="white" /></svg>
                      </div>
                    </div>
                    {/* Hook label */}
                    <div className="h-[58px] border-b border-slate-200 dark:border-slate-700 flex items-center justify-center px-0" title="Track de hook viral">
                      <div className="w-4 h-4 rounded-sm bg-gradient-to-br from-amber-500 to-orange-500 flex items-center justify-center shrink-0">
                        <Sparkles size={9} className="text-white" />
                      </div>
                    </div>
                    {/* Waveform label */}
                    {timelineMode === TIMELINE_MODE_ADVANCED && (
                      <div className="h-[72px] flex items-center justify-center px-0" title="Track de audio">
                        <svg width="10" height="10" viewBox="0 0 10 10" fill="none"><rect x="1" y="4" width="2" height="6" rx="1" fill="#64748b" /><rect x="4" y="2" width="2" height="8" rx="1" fill="#64748b" /><rect x="7" y="5" width="2" height="5" rx="1" fill="#64748b" /></svg>
                      </div>
                    )}
                  </div>

                  {/* Scrollable track area */}
                  <div ref={timelineViewportRef} className="flex-1 overflow-x-auto custom-scrollbar">
                    <div
                      ref={timelineTrackRef}
                      className="relative select-none bg-slate-50 dark:bg-slate-900/40 cursor-col-resize"
                      style={{ minWidth: '400px', width: `${Math.max(100, Math.round(timelineZoom * 100))}%` }}
                      onPointerDown={(e) => {
                        if (e.button !== 0) return;
                        e.currentTarget.setPointerCapture(e.pointerId);
                        handleTimelinePointerSeek(e.clientX);
                      }}
                      onPointerMove={(e) => {
                        if (!e.currentTarget.hasPointerCapture(e.pointerId)) return;
                        handleTimelinePointerSeek(e.clientX);
                      }}
                      onPointerUp={(e) => { e.currentTarget.releasePointerCapture(e.pointerId); }}
                    >
                      {/* Tick ruler */}
                      <div className="px-3 pt-2 pb-1 border-b border-slate-200 dark:border-slate-700">
                        <div className="relative h-5">
                          {timelineTicks.map((tick) => {
                            const left = `${(tick / Math.max(0.001, timelineDuration)) * 100}%`;
                            return (
                              <div key={`tick-${tick}`} className="absolute top-0 -translate-x-1/2" style={{ left }}>
                                <div className="w-px h-2 bg-slate-300 dark:bg-slate-600 mx-auto" />
                                <div className="mt-0.5 text-[10px] text-slate-500 dark:text-slate-400 tabular-nums">{`${Math.round(tick)}s`}</div>
                              </div>
                            );
                          })}
                        </div>
                      </div>

                      {/* Video track */}
                      <div className="border-b border-slate-200 dark:border-slate-700 py-2 px-2">
                        <div className="relative h-8 rounded-md overflow-hidden" style={{ background: 'linear-gradient(90deg,#1e3a5f 0%,#1a4a7a 40%,#1e3a5f 100%)' }}>
                          <div className="absolute inset-y-0 left-0 bg-black/40" style={{ width: `${(selectionStartRel / Math.max(0.001, timelineDuration)) * 100}%` }} />
                          <div className="absolute inset-y-0 right-0 bg-black/40" style={{ width: `${(1 - (selectionEndRel / Math.max(0.001, timelineDuration))) * 100}%` }} />
                          <div className="absolute inset-y-0 border-t-2 border-b-2 border-cyan-400/80 cursor-grab" style={{ left: `${(selectionStartRel / Math.max(0.001, timelineDuration)) * 100}%`, width: `${((selectionEndRel - selectionStartRel) / Math.max(0.001, timelineDuration)) * 100}%`, background: 'linear-gradient(90deg,rgba(34,211,238,.18) 0%,rgba(56,189,248,.12) 100%)' }} onMouseDown={(e) => startSelectionDrag(e, 'move')} title="Rango de recorte">
                            <button type="button" className="absolute left-0 top-0 bottom-0 w-2 bg-cyan-400/80 cursor-ew-resize" onMouseDown={(e) => startSelectionDrag(e, 'start')} onClick={(e) => e.stopPropagation()} aria-label="Inicio" />
                            <button type="button" className="absolute right-0 top-0 bottom-0 w-2 bg-cyan-400/80 cursor-ew-resize" onMouseDown={(e) => startSelectionDrag(e, 'end')} onClick={(e) => e.stopPropagation()} aria-label="Fin" />
                          </div>
                          <span className="absolute left-3 top-1/2 -translate-y-1/2 text-[10px] font-semibold text-white/70 pointer-events-none truncate max-w-[60%]">{clip?.video_title_for_youtube_short || `Clip ${clipIndex + 1}`}</span>
                        </div>
                      </div>

                      {/* Subtitles track */}
                      <div className="border-b border-slate-200 dark:border-slate-700 py-2 px-2">
                        <div className="relative h-10 rounded-md bg-slate-900/10 dark:bg-slate-900/40 border border-slate-200 dark:border-slate-700 overflow-hidden">
                          {subtitleEntries.filter((e) => String(e?.text || '').trim()).map((entry) => {
                            const start = clamp(Number(entry.start || 0), 0, timelineDuration);
                            const end = clamp(Number(entry.end || start), start + 0.08, timelineDuration);
                            const left = (start / Math.max(0.001, timelineDuration)) * 100;
                            const width = Math.max(0.8, ((end - start) / Math.max(0.001, timelineDuration)) * 100);
                            const accent = fontColor || '#a78bfa';
                            return (
                              <div key={`lane-${entry.id}`} className="absolute top-1 bottom-1 rounded-md text-[10px] px-2 flex items-center cursor-grab select-none transition-shadow hover:shadow-md" style={{ left: `${left}%`, width: `${width}%`, background: `${accent}22`, border: `1px solid ${accent}88`, color: accent }} onMouseDown={(e) => startSubtitleDrag(e, entry, 'move')} onDoubleClick={(e) => { e.stopPropagation(); seekTo(start); }} title={entry.text}>
                                <button type="button" className="absolute left-0 top-0 bottom-0 w-1.5 cursor-ew-resize rounded-l-md" style={{ background: `${accent}99` }} onMouseDown={(e) => startSubtitleDrag(e, entry, 'start')} onClick={(e) => e.stopPropagation()} aria-label="Inicio" />
                                <span className="truncate pl-0.5 font-medium">{entry.text}</span>
                                <button type="button" className="absolute right-0 top-0 bottom-0 w-1.5 cursor-ew-resize rounded-r-md" style={{ background: `${accent}99` }} onMouseDown={(e) => startSubtitleDrag(e, entry, 'end')} onClick={(e) => e.stopPropagation()} aria-label="Fin" />
                              </div>
                            );
                          })}
                        </div>
                      </div>

                      {/* Viral Hook track */}
                      <div className="border-b border-slate-200 dark:border-slate-700 py-2 px-2">
                        <div className="relative h-10 rounded-md bg-amber-50/60 dark:bg-amber-900/10 border border-amber-200/80 dark:border-amber-700/40 overflow-hidden">
                          {viralHookEnabled && String(viralHookText || '').trim() ? (
                            (() => {
                              const start = clamp(Number(viralHookTimelineStart || 0), 0, timelineDuration);
                              const end = clamp(Number(viralHookTimelineEnd || (start + 0.4)), start + 0.4, timelineDuration);
                              const left = (start / Math.max(0.001, timelineDuration)) * 100;
                              const width = Math.max(0.8, ((end - start) / Math.max(0.001, timelineDuration)) * 100);
                              return (
                                <div
                                  className="absolute top-1 bottom-1 rounded-md text-[10px] px-2 flex items-center cursor-grab select-none transition-shadow hover:shadow-md"
                                  style={{
                                    left: `${left}%`,
                                    width: `${width}%`,
                                    background: 'rgba(245, 158, 11, 0.18)',
                                    border: '1px solid rgba(217, 119, 6, 0.6)',
                                    color: '#b45309'
                                  }}
                                  onMouseDown={(e) => startViralHookDrag(e, 'move')}
                                  title={viralHookText}
                                >
                                  <button
                                    type="button"
                                    className="absolute left-0 top-0 bottom-0 w-1.5 cursor-ew-resize rounded-l-md"
                                    style={{ background: 'rgba(217, 119, 6, 0.85)' }}
                                    onMouseDown={(e) => startViralHookDrag(e, 'start')}
                                    onClick={(e) => e.stopPropagation()}
                                    aria-label="Inicio hook viral"
                                  />
                                  <span className="truncate pl-0.5 font-medium">{viralHookText}</span>
                                  <button
                                    type="button"
                                    className="absolute right-0 top-0 bottom-0 w-1.5 cursor-ew-resize rounded-r-md"
                                    style={{ background: 'rgba(217, 119, 6, 0.85)' }}
                                    onMouseDown={(e) => startViralHookDrag(e, 'end')}
                                    onClick={(e) => e.stopPropagation()}
                                    aria-label="Fin hook viral"
                                  />
                                </div>
                              );
                            })()
                          ) : (
                            <div className="absolute inset-0 flex items-center px-2 text-[10px] text-amber-700/80 dark:text-amber-200/70">
                              Hook viral desactivado
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Waveform (Advanced mode) */}
                      {timelineMode === TIMELINE_MODE_ADVANCED && (
                        <div className="py-2 px-2">
                          <div className="h-12 rounded-md bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 flex items-end gap-[2px] px-2">
                            {timelineDensityBars.map((amp, idx) => (
                              <div key={`density-${idx}`} className="flex-1 rounded-t-sm bg-slate-500/80 dark:bg-slate-400/80" style={{ height: `${Math.max(7, amp * 100)}%` }} />
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Playhead */}
                      <div className="absolute top-0 bottom-0 w-[2px] bg-slate-800/85 dark:bg-white/90 pointer-events-none" style={{ left: `${(clamp(previewCurrentTime, 0, timelineDuration) / Math.max(0.001, timelineDuration)) * 100}%` }} />
                      <div className="absolute top-1.5 w-3 h-3 rounded-full bg-slate-800 dark:bg-white border-2 border-white dark:border-slate-900 pointer-events-none -translate-x-1/2" style={{ left: `${(clamp(previewCurrentTime, 0, timelineDuration) / Math.max(0.001, timelineDuration)) * 100}%` }} />
                    </div>
                  </div>
                </div>

                <div className="mt-2 text-[11px] text-slate-500 dark:text-slate-400">Clip n.º {clipIndex + 1}</div>
              </div>
            </section>
          </div>
        </div>
      </div >
    </div >
  );
}
