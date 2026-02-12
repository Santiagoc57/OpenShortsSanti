import React, { useEffect, useMemo, useRef, useState } from 'react';
import { X, FileText, Captions, Type, LayoutTemplate, Music2, Search, Sparkles, Loader2, Play, Pause, Pencil } from 'lucide-react';
import { apiFetch, getApiUrl } from '../config';

const CAPTION_PRESETS = [
  {
    id: 'shouter',
    name: 'Impacto',
    sample: 'AQUI VA\nTU FRASE',
    style: {
      position: 'bottom',
      fontSize: 30,
      fontFamily: 'Impact',
      fontColor: '#FFE600',
      strokeColor: '#0A0A0A',
      strokeWidth: 4,
      bold: true,
      boxColor: '#000000',
      boxOpacity: 0
    }
  },
  {
    id: 'classy',
    name: 'Elegante',
    sample: 'Aqui va tu\nsubtitulo',
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
  }
];

const SECTION_ITEMS = [
  { id: 'transcript', label: 'Transcripción', icon: FileText },
  { id: 'captions', label: 'Subtítulos', icon: Captions },
  { id: 'subtitle_edit', label: 'Editar subtítulos', icon: Type },
  { id: 'layout', label: 'Editar layout', icon: LayoutTemplate },
  { id: 'music', label: 'Música', icon: Music2 }
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

const parseSrt = (srtText) => {
  const blocks = String(srtText || '').split(/\n\s*\n/).map((b) => b.trim()).filter(Boolean);
  const items = [];
  blocks.forEach((block, idx) => {
    const lines = block.split('\n').map((l) => l.trim()).filter(Boolean);
    if (lines.length < 2) return;
    const maybeIndex = /^\d+$/.test(lines[0]) ? Number(lines[0]) : idx + 1;
    const timeLine = /^\d+$/.test(lines[0]) ? lines[1] : lines[0];
    const textLines = /^\d+$/.test(lines[0]) ? lines.slice(2) : lines.slice(1);
    const [rawStart, rawEnd] = String(timeLine).split('-->').map((v) => v.trim());
    if (!rawStart || !rawEnd) return;
    items.push({
      id: `${maybeIndex}-${idx}`,
      index: maybeIndex,
      start: parseTimeToSeconds(rawStart),
      end: parseTimeToSeconds(rawEnd),
      text: textLines.join(' ').trim(),
      emphasize: false
    });
  });
  return items;
};

const buildSrt = (entries) => {
  if (!Array.isArray(entries)) return '';
  return entries
    .filter((entry) => entry && Number.isFinite(entry.start) && Number.isFinite(entry.end) && String(entry.text || '').trim())
    .map((entry, idx) => {
      const text = entry.emphasize ? String(entry.text).toUpperCase() : String(entry.text);
      return `${idx + 1}\n${formatSrtTime(entry.start)} --> ${formatSrtTime(entry.end)}\n${text}`;
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
    return cleanPath.split('/').pop() || '';
  } catch (_) {
    const clean = raw.split('?')[0].split('#')[0];
    return clean.split('/').pop() || '';
  }
};

export default function ClipStudioModal({
  isOpen,
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
  const [selectedPreset, setSelectedPreset] = useState(CAPTION_PRESETS[0].id);
  const [position, setPosition] = useState('bottom');
  const [fontSize, setFontSize] = useState(26);
  const [fontFamily, setFontFamily] = useState('Impact');
  const [fontColor, setFontColor] = useState('#FFE600');
  const [strokeColor, setStrokeColor] = useState('#0A0A0A');
  const [strokeWidth, setStrokeWidth] = useState(3);
  const [bold, setBold] = useState(true);
  const [boxColor, setBoxColor] = useState('#000000');
  const [boxOpacity, setBoxOpacity] = useState(20);

  const [subtitleEntries, setSubtitleEntries] = useState([]);
  const [subtitleSearch, setSubtitleSearch] = useState('');

  const [layoutAspect, setLayoutAspect] = useState(clip?.aspect_ratio === '16:9' ? '16:9' : '9:16');
  const [layoutStart, setLayoutStart] = useState(Number(clip?.start || 0));
  const [layoutEnd, setLayoutEnd] = useState(Number(clip?.end || 0));

  const [musicEnabled, setMusicEnabled] = useState(false);
  const [musicFile, setMusicFile] = useState(null);
  const [musicVolume, setMusicVolume] = useState(0.18);
  const [duckVoice, setDuckVoice] = useState(true);

  const [previewPlaying, setPreviewPlaying] = useState(false);
  const [previewVideoUrl, setPreviewVideoUrl] = useState(String(currentVideoUrl || ''));
  const [videoLoadError, setVideoLoadError] = useState('');
  const previewVideoRef = useRef(null);
  const previewBlobUrlRef = useRef(null);

  const previewText = useMemo(() => {
    const first = subtitleEntries.find((entry) => String(entry?.text || '').trim());
    if (first) return first.emphasize ? String(first.text).toUpperCase() : String(first.text);
    return 'Así se verán tus subtítulos';
  }, [subtitleEntries]);

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

  const filteredSubtitleEntries = useMemo(() => {
    const q = String(subtitleSearch || '').trim().toLowerCase();
    if (!q) return subtitleEntries;
    return subtitleEntries.filter((entry) => String(entry?.text || '').toLowerCase().includes(q));
  }, [subtitleEntries, subtitleSearch]);

  const srtContent = useMemo(() => buildSrt(subtitleEntries), [subtitleEntries]);

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
      const res = await apiFetch(`/api/transcript/${jobId}?limit=2000`);
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
    setMusicEnabled(false);
    setMusicFile(null);
    setMusicVolume(0.18);
    setDuckVoice(true);
    applyPreset(CAPTION_PRESETS[0].id);
    loadTranscript();
    loadSrt();
  }, [isOpen, jobId, clipIndex, clip?.start, clip?.end, clip?.aspect_ratio]);

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
        setPreviewVideoUrl(sourceUrl);
        setVideoLoadError(`No se pudo cargar vista previa remota (${err.message}).`);
      }
    })();

    return () => {
      cancelled = true;
      cleanupBlobUrl();
    };
  }, [isOpen, currentVideoUrl]);

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

  const handleApply = async () => {
    if (!jobId) return;
    setIsApplying(true);
    setError('');
    let workingFile = extractFilename(currentVideoUrl);
    let resultingUrl = currentVideoUrl;

    try {
      const clipStart = Number(clip?.start || 0);
      const clipEnd = Number(clip?.end || clipStart);
      const needsRecut = layoutAspect !== (clip?.aspect_ratio === '16:9' ? '16:9' : '9:16')
        || Math.abs(Number(layoutStart) - clipStart) > 0.01
        || Math.abs(Number(layoutEnd) - clipEnd) > 0.01;

      if (needsRecut) {
        const recutRes = await apiFetch('/api/recut', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            job_id: jobId,
            clip_index: clipIndex,
            start: Number(layoutStart),
            end: Number(layoutEnd),
            aspect_ratio: layoutAspect
          })
        });
        if (!recutRes.ok) throw new Error(await recutRes.text());
        const recutData = await recutRes.json();
        if (recutData?.new_video_url) {
          resultingUrl = getApiUrl(recutData.new_video_url);
          workingFile = extractFilename(recutData.new_video_url);
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
      onApplied && onApplied({ newVideoUrl: resultingUrl });
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

  if (!isOpen) return null;

  const aspectRatioClass = layoutAspect === '16:9' ? 'aspect-video max-w-[760px]' : 'aspect-[9/16] max-w-[420px]';

  return (
    <div className="fixed inset-0 z-[110] bg-black/45 backdrop-blur-sm p-3 md:p-6">
      <div className="w-full h-full rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-2xl overflow-hidden flex flex-col">
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
                  <div className="space-y-2 max-h-[68vh] overflow-y-auto custom-scrollbar pr-1">
                    {filteredTranscript.map((seg) => (
                      <div key={`${seg.segment_index}-${seg.start}`} className="rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 p-2.5">
                        <div className="text-[11px] text-slate-500 mb-1">{`${seg.start.toFixed(2)}s - ${seg.end.toFixed(2)}s`}</div>
                        <p className="text-sm text-slate-700 dark:text-slate-200 leading-relaxed">{seg.text}</p>
                      </div>
                    ))}
                    {!isLoadingTranscript && filteredTranscript.length === 0 && <p className="text-sm text-slate-500">No hay segmentos para este rango.</p>}
                  </div>
                </div>
              )}

              {section === 'captions' && (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold text-zinc-800 dark:text-zinc-100">Subtítulos</h3>
                    <label className="inline-flex items-center gap-2 text-sm text-zinc-600 dark:text-zinc-300">
                      <span>{captionsOn ? 'Activos' : 'Pausados'}</span>
                      <input type="checkbox" checked={captionsOn} onChange={(e) => setCaptionsOn(e.target.checked)} />
                    </label>
                  </div>

                  <div>
                    <p className="text-xs font-semibold text-zinc-500 mb-2">Presets</p>
                    <div className="grid grid-cols-2 gap-2">
                      {CAPTION_PRESETS.map((preset) => (
                        <button
                          key={preset.id}
                          type="button"
                          onClick={() => applyPreset(preset.id)}
                          className={`rounded-xl border p-2 text-left ${selectedPreset === preset.id
                            ? 'border-orange-400 bg-orange-50 dark:bg-orange-500/10'
                            : 'border-black/10 dark:border-white/10 bg-white/70 dark:bg-white/5'
                          }`}
                        >
                          <div className="text-[11px] font-semibold text-zinc-700 dark:text-zinc-100">{preset.name}</div>
                          <div className="text-[10px] mt-1 text-zinc-500 dark:text-zinc-400 whitespace-pre-line leading-tight">{preset.sample}</div>
                        </button>
                      ))}
                    </div>
                  </div>

                  <div>
                    <p className="text-xs font-semibold text-zinc-500 mb-2">Posición</p>
                    <div className="grid grid-cols-3 gap-2">
                      {['top', 'middle', 'bottom'].map((opt) => (
                        <button
                          key={opt}
                          type="button"
                          onClick={() => setPosition(opt)}
                          className={`rounded-lg px-2 py-2 text-xs border capitalize ${position === opt
                            ? 'border-violet-400 bg-violet-100 dark:bg-violet-900/20 text-violet-700 dark:text-violet-300'
                            : 'border-black/10 dark:border-white/10 text-zinc-600 dark:text-zinc-300'
                          }`}
                        >
                          {opt === 'top' ? 'Arriba' : opt === 'middle' ? 'Centro' : 'Abajo'}
                        </button>
                      ))}
                    </div>
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
                    <button
                      type="button"
                      onClick={loadSrt}
                      className="text-xs px-2 py-1 rounded-md border border-black/10 dark:border-white/10 text-zinc-600 dark:text-zinc-300 hover:bg-black/5 dark:hover:bg-white/5"
                    >
                      {isLoadingSrt ? 'Cargando...' : 'Recargar SRT'}
                    </button>
                  </div>
                  <div className="relative mb-3">
                    <Search size={14} className="absolute left-2.5 top-2.5 text-zinc-400" />
                    <input
                      value={subtitleSearch}
                      onChange={(e) => setSubtitleSearch(e.target.value)}
                      placeholder="Buscar subtítulo"
                      className="w-full rounded-lg border border-black/10 dark:border-white/10 bg-white/80 dark:bg-black/20 pl-8 pr-3 py-2 text-sm"
                    />
                  </div>
                  <div className="space-y-2 max-h-[68vh] overflow-y-auto custom-scrollbar pr-1">
                    {filteredSubtitleEntries.map((entry) => (
                      <div key={entry.id} className="rounded-lg border border-black/10 dark:border-white/10 bg-white/70 dark:bg-black/20 p-2.5">
                        <div className="flex items-center justify-between gap-2 mb-1">
                          <span className="text-[11px] text-zinc-500">{`${formatSrtTime(entry.start)} - ${formatSrtTime(entry.end)}`}</span>
                          <button
                            type="button"
                            onClick={() => onSubtitleToggleEmphasis(entry.id)}
                            className={`text-[11px] px-2 py-1 rounded-md border ${entry.emphasize
                              ? 'border-amber-400 bg-amber-100 dark:bg-amber-500/15 text-amber-700 dark:text-amber-300'
                              : 'border-black/10 dark:border-white/10 text-zinc-600 dark:text-zinc-300'
                            }`}
                          >
                            Énfasis
                          </button>
                        </div>
                        <textarea
                          value={entry.text}
                          onChange={(e) => onSubtitleEntryChange(entry.id, e.target.value)}
                          rows={2}
                          className="w-full rounded-md border border-black/10 dark:border-white/10 bg-white dark:bg-black/20 p-2 text-sm text-zinc-700 dark:text-zinc-200"
                        />
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
                  <p className="text-xs text-zinc-500">Ajusta formato y rango para evitar overlays de publicidad o marcos incómodos.</p>

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

                  <div className="grid grid-cols-2 gap-3">
                    <label className="text-xs text-zinc-600 dark:text-zinc-300">Inicio (s)
                      <input type="number" step="0.1" value={layoutStart} onChange={(e) => setLayoutStart(Number(e.target.value || 0))} className="mt-1 w-full rounded-md border border-black/10 dark:border-white/10 bg-white/80 dark:bg-black/20 p-2 text-sm" />
                    </label>
                    <label className="text-xs text-zinc-600 dark:text-zinc-300">Fin (s)
                      <input type="number" step="0.1" value={layoutEnd} onChange={(e) => setLayoutEnd(Number(e.target.value || 0))} className="mt-1 w-full rounded-md border border-black/10 dark:border-white/10 bg-white/80 dark:bg-black/20 p-2 text-sm" />
                    </label>
                  </div>
                  <p className="text-[11px] text-zinc-500">Tip: el recorte y el layout se aplican antes de subtítulos y música.</p>
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
              <div className="flex-1 rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 p-4 flex items-center justify-center">
                <div className={`w-full ${aspectRatioClass} rounded-md bg-black overflow-hidden relative mx-auto`}>
                  <video
                    ref={previewVideoRef}
                    src={previewVideoUrl || currentVideoUrl}
                    className="w-full h-full object-cover"
                    controls
                    playsInline
                    onPlay={() => setPreviewPlaying(true)}
                    onPause={() => setPreviewPlaying(false)}
                    onError={() => {
                      setVideoLoadError('El navegador no pudo reproducir este archivo en la vista previa.');
                    }}
                  />

                  {captionsOn && (
                    <div
                      className={`absolute left-0 right-0 px-6 text-center pointer-events-none ${position === 'top' ? 'top-8' : position === 'middle' ? 'top-1/2 -translate-y-1/2' : 'bottom-8'}`}
                    >
                      <span
                        className="inline-block rounded-md px-2 py-1"
                        style={{
                          fontSize: `${Math.max(12, Math.round(fontSize * 0.58))}px`,
                          fontFamily,
                          fontWeight: bold ? 700 : 400,
                          color: fontColor,
                          textShadow: `0 0 ${strokeWidth}px ${strokeColor}`,
                          backgroundColor: boxOpacity > 0 ? toRgba(boxColor, boxOpacity) : 'transparent'
                        }}
                      >
                        {previewText}
                      </span>
                    </div>
                  )}
                </div>
                {videoLoadError && (
                  <div className="mt-3 rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-700 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-300">
                    {videoLoadError}
                  </div>
                )}
              </div>

              <div className="mt-3 rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 px-3 py-2 flex items-center gap-3 text-sm text-slate-700 dark:text-slate-200">
                <button
                  type="button"
                  onClick={togglePreviewPlayback}
                  className="w-7 h-7 rounded-full border border-slate-200 dark:border-slate-700 inline-flex items-center justify-center"
                >
                  {previewPlaying ? <Pause size={14} /> : <Play size={14} />}
                </button>
                <span className="text-xs text-slate-500">Clip n.º {clipIndex + 1}</span>
              </div>
            </section>
          </div>
        </div>
      </div>
    </div>
  );
}
