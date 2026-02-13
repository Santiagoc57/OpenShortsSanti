import React, { useState, useEffect, useRef } from 'react';
import { Download, Share2, Instagram, Youtube, Video, CheckCircle, AlertCircle, X, Loader2, Copy, Wand2, Calendar, Clock, Scissors, Play, Pause, Pencil } from 'lucide-react';
import { getApiUrl, apiFetch } from '../config';
import ClipStudioModal from './ClipStudioModal';

const transcriptSegmentsCache = new Map();

const buildTranscriptExcerpt = (segments, clipStart, clipEnd, maxChars = 420) => {
    if (!Array.isArray(segments) || segments.length === 0) return '';
    const start = Number(clipStart || 0);
    const end = Number(clipEnd || start);
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return '';

    const matched = segments.filter((seg) => {
        const ss = Number(seg?.start || 0);
        const se = Number(seg?.end || ss);
        const text = String(seg?.text || '').trim();
        return text && se > start && ss < end;
    });
    if (matched.length === 0) return '';

    const excerpt = matched.map((seg) => String(seg.text || '').trim()).join(' ').replace(/\s+/g, ' ').trim();
    if (!excerpt) return '';
    if (excerpt.length > maxChars) return `${excerpt.slice(0, maxChars - 3).trim()}...`;
    return excerpt;
};

export default function ResultCard({ clip, displayIndex = 0, clipIndex = 0, jobId, uploadPostKey, uploadUserId, geminiApiKey, onPlay, onPause, onOpenStudio }) {
    const [showModal, setShowModal] = useState(false);
    const [showStudioModal, setShowStudioModal] = useState(false);
    const [showRecutModal, setShowRecutModal] = useState(false);
    const videoRef = React.useRef(null);
    const editVideoRef = useRef(null);
    const [currentVideoUrl, setCurrentVideoUrl] = useState(getApiUrl(clip.video_url));
    const [baseVideoUrl, setBaseVideoUrl] = useState(getApiUrl(clip.video_url));
    const [subtitledVideoUrl, setSubtitledVideoUrl] = useState(null);
    const [subtitlesEnabled, setSubtitlesEnabled] = useState(false);
    const [playbackVideoUrl, setPlaybackVideoUrl] = useState(getApiUrl(clip.video_url));
    const [videoLoadError, setVideoLoadError] = useState(null);
    const blobUrlRef = useRef(null);
    const clipAspectRatio = clip?.aspect_ratio === '16:9' ? '16:9' : '9:16';
    const isLandscape = clipAspectRatio === '16:9';

    const [platforms, setPlatforms] = useState({
        tiktok: true,
        instagram: true,
        youtube: true
    });
    const [postTitle, setPostTitle] = useState("");
    const [postDescription, setPostDescription] = useState("");
    const [isScheduling, setIsScheduling] = useState(false);
    const [scheduleDate, setScheduleDate] = useState("");

    const [posting, setPosting] = useState(false);

    const formatTime = (seconds) => {
        if (typeof seconds !== 'number' || Number.isNaN(seconds)) return '0:00';
        const total = Math.max(0, Math.floor(seconds));
        const mins = Math.floor(total / 60);
        const secs = total % 60;
        return `${mins}:${String(secs).padStart(2, '0')}`;
    };
    const rawScore = Number(clip?.virality_score);
    const clipScore = Number.isFinite(rawScore) ? Math.max(0, Math.min(100, Math.round(rawScore))) : 0;
    const rawConfidence = Number(clip?.selection_confidence);
    const clipConfidence = Number.isFinite(rawConfidence) ? Math.max(0, Math.min(1, rawConfidence)) : clipScore / 100;
    const topicTags = Array.isArray(clip?.topic_tags) ? clip.topic_tags.filter((t) => typeof t === 'string' && t.trim() !== '') : [];
    const scoreBadgeClass = clipScore >= 80
        ? 'bg-emerald-500/15 border-emerald-500/40 text-emerald-300'
        : clipScore >= 65
            ? 'bg-amber-500/15 border-amber-500/40 text-amber-300'
            : 'bg-zinc-500/15 border-zinc-500/30 text-zinc-300';
    const [postResult, setPostResult] = useState(null);

    const [isEditing, setIsEditing] = useState(false);
    const [isRecutting, setIsRecutting] = useState(false);
    const [editError, setEditError] = useState(null);
    const [activeTextTab, setActiveTextTab] = useState('social');
    const [transcriptFallback, setTranscriptFallback] = useState('');
    const [recutStart, setRecutStart] = useState(clip.start);
    const [recutEnd, setRecutEnd] = useState(clip.end);
    const [editDuration, setEditDuration] = useState(0);
    const [editCurrentTime, setEditCurrentTime] = useState(0);
    const [editPlaying, setEditPlaying] = useState(false);

    useEffect(() => {
        const cleanupBlobUrl = () => {
            if (blobUrlRef.current) {
                URL.revokeObjectURL(blobUrlRef.current);
                blobUrlRef.current = null;
            }
        };

        const sourceUrl = String(currentVideoUrl || '');
        if (!sourceUrl) {
            cleanupBlobUrl();
            setPlaybackVideoUrl('');
            setVideoLoadError(null);
            return () => {};
        }

        const isNgrokSource = /ngrok/i.test(sourceUrl);
        if (!isNgrokSource) {
            cleanupBlobUrl();
            setPlaybackVideoUrl(sourceUrl);
            setVideoLoadError(null);
            return () => {};
        }

        let cancelled = false;
        setVideoLoadError(null);

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
                blobUrlRef.current = objectUrl;
                setPlaybackVideoUrl(objectUrl);
            } catch (err) {
                if (cancelled) return;
                setPlaybackVideoUrl(sourceUrl);
                setVideoLoadError(`No se pudo cargar video remoto (${err.message}).`);
            }
        })();

        return () => {
            cancelled = true;
            cleanupBlobUrl();
        };
    }, [currentVideoUrl]);

    // Initialize/Reset form when modal opens
    useEffect(() => {
        if (showModal) {
            setPostTitle(clip.video_title_for_youtube_short || "Corto viral");
            setPostDescription(clip.video_description_for_instagram || clip.video_description_for_tiktok || "");
            setIsScheduling(false);
            setScheduleDate("");
            setPostResult(null);
        }
    }, [showModal, clip]);

    useEffect(() => {
        if (!showRecutModal || !editVideoRef.current) return;
        const vid = editVideoRef.current;
        const onLoaded = () => setEditDuration(vid.duration || 0);
        const onTime = () => setEditCurrentTime(vid.currentTime || 0);
        const onPlay = () => setEditPlaying(true);
        const onPause = () => setEditPlaying(false);
        vid.addEventListener('loadedmetadata', onLoaded);
        vid.addEventListener('timeupdate', onTime);
        vid.addEventListener('play', onPlay);
        vid.addEventListener('pause', onPause);
        return () => {
            vid.removeEventListener('loadedmetadata', onLoaded);
            vid.removeEventListener('timeupdate', onTime);
            vid.removeEventListener('play', onPlay);
            vid.removeEventListener('pause', onPause);
        };
    }, [showRecutModal]);

    const clipDuration = Math.max(0, Number(clip?.end || 0) - Number(clip?.start || 0));
    const viralityTen = (clipScore / 10).toFixed(1);
    const socialText = clip.video_description_for_tiktok || clip.video_description_for_instagram || clip.video_title_for_youtube_short || "Sin descripción.";
    const baseTranscriptText = [
        clip?.transcript_excerpt,
        clip?.transcript_text,
        clip?.transcription,
        typeof clip?.transcript === 'string' ? clip.transcript : null
    ].find((value) => typeof value === 'string' && value.trim()) || '';
    const transcriptDisplay = baseTranscriptText || transcriptFallback || "No hay transcripción disponible para este clip todavía.";
    const viralityText = clip.score_reason || 'Este clip tiene buena combinación de claridad, gancho y potencial de engagement.';
    const hashtagsText = topicTags.length > 0 ? topicTags.map((tag) => `#${tag}`).join(' ') : 'Sin hashtags sugeridos.';
    const activePanelText = activeTextTab === 'transcript'
        ? transcriptDisplay
        : activeTextTab === 'virality'
            ? viralityText
            : activeTextTab === 'hashtags'
                ? hashtagsText
                : socialText;

    const handleCopyActiveText = async () => {
        try {
            if (activePanelText) await navigator.clipboard.writeText(activePanelText);
        } catch (_) {
            // Ignore clipboard permission errors.
        }
    };

    useEffect(() => {
        let cancelled = false;
        const loadFallback = async () => {
            if (baseTranscriptText || !jobId) {
                setTranscriptFallback('');
                return;
            }
            const cachedSegments = transcriptSegmentsCache.get(jobId);
            if (cachedSegments) {
                const text = buildTranscriptExcerpt(cachedSegments, clip?.start, clip?.end);
                if (!cancelled) setTranscriptFallback(text);
                return;
            }
            try {
                const res = await apiFetch(`/api/transcript/${jobId}?limit=2000`);
                if (!res.ok) return;
                const data = await res.json();
                const segments = Array.isArray(data?.segments) ? data.segments : [];
                transcriptSegmentsCache.set(jobId, segments);
                const text = buildTranscriptExcerpt(segments, clip?.start, clip?.end);
                if (!cancelled) setTranscriptFallback(text);
            } catch (_) {
                // No-op: keep fallback empty.
            }
        };
        loadFallback();
        return () => {
            cancelled = true;
        };
    }, [baseTranscriptText, jobId, clip?.start, clip?.end]);

    const handleDownload = async (e) => {
        e.preventDefault();
        try {
            const response = await fetch(currentVideoUrl);
            if (!response.ok) throw new Error('Descarga fallida');
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = `clip-${displayIndex + 1}.mp4`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } catch (err) {
            console.error('Error de descarga:', err);
            window.open(currentVideoUrl, '_blank');
        }
    };

    const handleAutoEdit = async () => {
        setIsEditing(true);
        setEditError(null);
        try {
             // Use passed prop or fallback
             const apiKey = geminiApiKey || localStorage.getItem('gemini_key');
             
             if (!apiKey) {
                 throw new Error("Falta la API Key de Gemini. Configúrala en Configuración.");
             }

             const res = await apiFetch('/api/edit', {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'X-Gemini-Key': apiKey 
                },
                body: JSON.stringify({
                    job_id: jobId,
                    clip_index: clipIndex,
                    input_filename: currentVideoUrl.split('/').pop()
                })
             });

             if (!res.ok) {
                 const errText = await res.text();
                 try {
                     const jsonErr = JSON.parse(errText);
                     throw new Error(jsonErr.detail || errText);
                 } catch (e) {
                     throw new Error(errText);
                 }
             }

             const data = await res.json();
             if (data.new_video_url) {
                 const nextUrl = getApiUrl(data.new_video_url);
                 setCurrentVideoUrl(nextUrl);
                 setBaseVideoUrl(nextUrl);
                 setSubtitledVideoUrl(null);
                 setSubtitlesEnabled(false);
                 // Reload video
                 if (videoRef.current) {
                     videoRef.current.load();
                 }
             }

        } catch (e) {
            setEditError(e.message);
            setTimeout(() => setEditError(null), 5000);
        } finally {
            setIsEditing(false);
        }
    };

    const handleRecut = async () => {
        setIsRecutting(true);
        setEditError(null);
        try {
            const res = await apiFetch('/api/recut', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    job_id: jobId,
                    clip_index: clipIndex,
                    start: Number(recutStart),
                    end: Number(recutEnd),
                    aspect_ratio: clipAspectRatio
                })
            });
            if (!res.ok) {
                const errText = await res.text();
                throw new Error(errText);
            }
            const data = await res.json();
            if (data.new_video_url) {
                const nextUrl = getApiUrl(data.new_video_url);
                setCurrentVideoUrl(nextUrl);
                setBaseVideoUrl(nextUrl);
                setSubtitledVideoUrl(null);
                setSubtitlesEnabled(false);
                if (videoRef.current) {
                    videoRef.current.load();
                }
                setShowRecutModal(false);
            }
        } catch (e) {
            setEditError(e.message);
            setTimeout(() => setEditError(null), 5000);
        } finally {
            setIsRecutting(false);
        }
    };

    const handlePost = async () => {
        if (!uploadPostKey || !uploadUserId) {
            setPostResult({ success: false, msg: "Falta API Key o User ID." });
            return;
        }

        const selectedPlatforms = Object.keys(platforms).filter(k => platforms[k]);
        if (selectedPlatforms.length === 0) {
            setPostResult({ success: false, msg: "Selecciona al menos una plataforma." });
            return;
        }

        if (isScheduling && !scheduleDate) {
            setPostResult({ success: false, msg: "Selecciona fecha y hora." });
            return;
        }

        setPosting(true);
        setPostResult(null);

        try {
            const payload = {
                job_id: jobId,
                clip_index: clipIndex,
                api_key: uploadPostKey,
                user_id: uploadUserId,
                platforms: selectedPlatforms,
                title: postTitle,
                description: postDescription
            };

            if (isScheduling && scheduleDate) {
                // Convert to ISO-8601
                payload.scheduled_date = new Date(scheduleDate).toISOString();
                // Optional: pass timezone if needed, backend defaults to UTC or we can send user's timezone
                payload.timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
            }

            const res = await apiFetch('/api/social/post', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!res.ok) {
                const errText = await res.text();
                try {
                    const jsonErr = JSON.parse(errText);
                    throw new Error(jsonErr.detail || errText);
                } catch (e) {
                    throw new Error(errText);
                }
            }

            setPostResult({ success: true, msg: isScheduling ? "Programado correctamente." : "Publicado correctamente." });
            setTimeout(() => {
                setShowModal(false);
                setPostResult(null);
            }, 3000);

        } catch (e) {
            setPostResult({ success: false, msg: `Falló: ${e.message}` });
        } finally {
            setPosting(false);
        }
    };

    return (
        <div
            className="bg-surface border border-white/10 rounded-2xl overflow-hidden flex flex-col md:flex-row group hover:border-primary/30 hover:shadow-xl transition-all animate-[fadeIn_0.5s_ease-out]"
            style={{ animationDelay: `${displayIndex * 0.08}s` }}
        >
            <div
                className="w-full md:w-[220px] lg:w-[240px] bg-black relative shrink-0"
                style={{ aspectRatio: isLandscape ? '16 / 9' : '9 / 16' }}
            >
                <video
                    ref={videoRef}
                    src={playbackVideoUrl || currentVideoUrl}
                    controls
                    className="w-full h-full object-cover"
                    playsInline
                    onError={() => {
                        setVideoLoadError('El navegador no pudo reproducir este archivo (codec o respuesta no válida).');
                    }}
                    onPlay={() => {
                        const currentTime = videoRef.current ? videoRef.current.currentTime : 0;
                        onPlay && onPlay(clip.start + currentTime);
                    }}
                    onPause={() => onPause && onPause()}
                    onEnded={() => {
                        if (videoRef.current) {
                            videoRef.current.currentTime = 0;
                            videoRef.current.play();
                        }
                    }}
                />
                <div className="absolute top-3 left-3 bg-black/65 backdrop-blur-sm text-white text-[10px] font-bold px-2 py-1 rounded-md border border-white/10">
                    #{displayIndex + 1}
                </div>
                <div className="absolute top-3 right-3 bg-black/65 backdrop-blur-sm text-white text-[10px] font-semibold px-2 py-1 rounded-md border border-white/10">
                    {clipAspectRatio}
                </div>
                <div className="absolute bottom-3 right-3 bg-black/70 text-white text-[10px] font-mono px-1.5 py-0.5 rounded">
                    {formatTime(clipDuration)}
                </div>

                {isEditing && (
                    <div className="absolute inset-0 bg-black/60 backdrop-blur-sm flex flex-col items-center justify-center z-10 p-4 text-center">
                        <Loader2 size={30} className="text-primary animate-spin mb-2" />
                        <span className="text-xs font-bold text-white uppercase tracking-wider">IA editando</span>
                        <span className="text-[10px] text-zinc-300 mt-1">Aplicando mejoras visuales</span>
                    </div>
                )}
                {videoLoadError && (
                    <div className="absolute inset-x-2 bottom-10 z-10 rounded-md border border-amber-400/40 bg-amber-500/15 px-2 py-1 text-[10px] leading-tight text-amber-100">
                        {videoLoadError}
                    </div>
                )}
            </div>

            <div className="flex-1 p-4 md:p-5 flex flex-col min-w-0 bg-[#121214]">
                <div className="flex items-start justify-between gap-4">
                    <div className="min-w-0">
                        <h3 className="text-lg font-bold text-white leading-tight line-clamp-2 break-words" title={clip.video_title_for_youtube_short}>
                            {clip.video_title_for_youtube_short || "Clip viral generado"}
                        </h3>
                        <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-zinc-400">
                            <span className={`px-2 py-1 rounded-md border ${scoreBadgeClass}`}>
                                Puntaje {clipScore}/100
                            </span>
                            <span className="px-2 py-1 rounded-md border border-white/10 bg-white/5">
                                Confianza {Math.round(clipConfidence * 100)}%
                            </span>
                            <span className="px-2 py-1 rounded-md border border-white/10 bg-white/5">
                                {formatTime(Number(clip.start || 0))} - {formatTime(Number(clip.end || 0))}
                            </span>
                        </div>
                    </div>
                    <div className="shrink-0 text-right px-3 py-2 rounded-lg border border-white/10 bg-white/5">
                        <div className="text-3xl leading-none font-bold text-white">{viralityTen}</div>
                        <div className="text-[10px] uppercase tracking-wider text-zinc-400 mt-1">Virality</div>
                    </div>
                </div>

                <div className="mt-4 rounded-xl border border-white/10 bg-black/20 p-3">
                    <div className="flex items-center justify-between gap-3 mb-3">
                        <div className="flex items-center gap-5 md:gap-6 border-b border-white/10 w-full">
                            <button
                                onClick={() => setActiveTextTab('social')}
                                className={`pb-1.5 px-0.5 text-xs font-semibold transition-colors border-b-2 ${
                                    activeTextTab === 'social'
                                        ? 'text-primary border-primary'
                                        : 'text-zinc-500 border-transparent hover:text-zinc-300'
                                }`}
                            >
                                Social
                            </button>
                            <button
                                onClick={() => setActiveTextTab('transcript')}
                                className={`pb-1.5 px-0.5 text-xs font-semibold transition-colors border-b-2 ${
                                    activeTextTab === 'transcript'
                                        ? 'text-primary border-primary'
                                        : 'text-zinc-500 border-transparent hover:text-zinc-300'
                                }`}
                            >
                                Transcripción
                            </button>
                            <button
                                onClick={() => setActiveTextTab('virality')}
                                className={`pb-1.5 px-0.5 text-xs font-semibold transition-colors border-b-2 ${
                                    activeTextTab === 'virality'
                                        ? 'text-primary border-primary'
                                        : 'text-zinc-500 border-transparent hover:text-zinc-300'
                                }`}
                            >
                                Puntaje viral
                            </button>
                            <button
                                onClick={() => setActiveTextTab('hashtags')}
                                className={`pb-1.5 px-0.5 text-xs font-semibold transition-colors border-b-2 ${
                                    activeTextTab === 'hashtags'
                                        ? 'text-primary border-primary'
                                        : 'text-zinc-500 border-transparent hover:text-zinc-300'
                                }`}
                            >
                                Etiquetas
                            </button>
                        </div>
                        <button
                            onClick={handleCopyActiveText}
                            className="text-zinc-400 hover:text-zinc-200 p-1 rounded border border-white/10 bg-white/5"
                            title="Copiar texto"
                        >
                            <Copy size={13} />
                        </button>
                    </div>
                    {activeTextTab === 'virality' ? (
                        <div className="space-y-3">
                            <div className="flex items-center justify-between gap-3">
                                <div className="text-xs text-zinc-400">Puntaje de viralidad</div>
                                <div className="text-sm font-semibold text-white">{clipScore}/100</div>
                            </div>
                            <div className="h-2 w-full rounded-full bg-white/10 overflow-hidden">
                                <div
                                    className="h-full bg-gradient-to-r from-emerald-500 via-amber-400 to-rose-500"
                                    style={{ width: `${clipScore}%` }}
                                />
                            </div>
                            <div className="text-[11px] text-zinc-400">
                                Confianza del modelo: <span className="text-zinc-200 font-semibold">{Math.round(clipConfidence * 100)}%</span>
                            </div>
                            <p className="text-sm text-zinc-300 leading-relaxed break-words">{viralityText}</p>
                        </div>
                    ) : activeTextTab === 'hashtags' ? (
                        <div className="space-y-3">
                            <p className="text-xs text-zinc-400">Hashtags sugeridos para publicar este clip:</p>
                            {topicTags.length > 0 ? (
                                <div className="flex flex-wrap gap-2">
                                    {topicTags.map((tag) => (
                                        <span key={`hash-${tag}`} className="text-xs px-2.5 py-1 rounded-md border border-primary/30 bg-primary/10 text-primary">
                                            #{tag}
                                        </span>
                                    ))}
                                </div>
                            ) : (
                                <p className="text-sm text-zinc-300">Sin hashtags sugeridos.</p>
                            )}
                        </div>
                    ) : (
                        <p className={`text-sm text-zinc-300 leading-relaxed break-words ${activeTextTab === 'transcript' ? 'font-mono text-xs' : ''}`}>
                            {activePanelText}
                        </p>
                    )}
                </div>

                {subtitledVideoUrl && (
                    <div className="mt-3 flex items-center gap-2 text-[11px] text-zinc-400">
                        <span>Subtítulos:</span>
                        <button
                            onClick={() => {
                                const next = !subtitlesEnabled;
                                setSubtitlesEnabled(next);
                                setCurrentVideoUrl(next ? subtitledVideoUrl : baseVideoUrl);
                                if (videoRef.current) {
                                    videoRef.current.load();
                                }
                            }}
                            className={`px-2 py-1 rounded-full border transition-colors ${
                                subtitlesEnabled
                                    ? 'bg-yellow-500/20 border-yellow-500/40 text-yellow-300'
                                    : 'bg-white/5 border-white/10 text-zinc-400'
                            }`}
                        >
                            {subtitlesEnabled ? 'ON' : 'OFF'}
                        </button>
                    </div>
                )}

                {editError && (
                    <div className="mt-3 p-2 bg-red-500/10 border border-red-500/20 text-red-400 text-xs rounded-lg flex items-center gap-2">
                        <AlertCircle size={13} className="shrink-0" />
                        {editError}
                    </div>
                )}

                <div className="mt-4 pt-4 border-t border-white/10 flex flex-wrap items-center justify-between gap-3">
                    <div className="flex flex-wrap items-center gap-2">
                        <button
                            onClick={handleDownload}
                            className="py-2.5 px-5 bg-gradient-to-r from-emerald-600 to-green-500 hover:from-emerald-500 hover:to-green-400 text-white rounded-lg text-sm font-semibold transition-all active:scale-[0.98] flex items-center gap-2 shadow-lg shadow-emerald-900/35"
                        >
                            <Download size={15} /> Descargar
                        </button>
                    </div>
                    <div className="flex items-center gap-2">
                        <button
                            onClick={handleAutoEdit}
                            disabled={isEditing}
                            className="p-2.5 rounded-lg border border-white/10 bg-white/5 hover:bg-white/10 text-zinc-200 disabled:opacity-60"
                            title={isEditing ? 'Editando...' : 'Edición automática IA'}
                        >
                            {isEditing ? <Loader2 size={15} className="animate-spin" /> : <Wand2 size={15} />}
                        </button>
                        <button
                            onClick={() => {
                                if (onOpenStudio) {
                                    onOpenStudio({ clip, clipIndex, currentVideoUrl });
                                    return;
                                }
                                setShowStudioModal(true);
                            }}
                            className="py-2 px-3 rounded-lg border border-violet-300 bg-violet-100 text-violet-700 hover:bg-violet-200 dark:border-violet-400/40 dark:bg-violet-500/15 dark:hover:bg-violet-500/25 dark:text-violet-200 transition-colors inline-flex items-center gap-2 shadow-sm"
                            title="Editar clip"
                        >
                            <Pencil size={14} />
                            <span className="text-sm font-medium">Editar</span>
                        </button>
                        <button
                            onClick={() => setShowRecutModal(true)}
                            className="p-2.5 rounded-lg border border-white/10 bg-white/5 hover:bg-white/10 text-zinc-200"
                            title="Recortar clip"
                        >
                            <Scissors size={15} />
                        </button>
                    </div>
                </div>
            </div>

            {/* Post Modal */}
            {showModal && (
                <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-[fadeIn_0.2s_ease-out]">
                    <div className="bg-[#121214] border border-white/10 p-6 rounded-2xl w-full max-w-md shadow-2xl relative max-h-[90vh] overflow-y-auto custom-scrollbar">
                        <button
                            onClick={() => setShowModal(false)}
                            className="absolute top-4 right-4 text-zinc-500 hover:text-white"
                        >
                            <X size={20} />
                        </button>

                        <h3 className="text-lg font-bold text-white mb-4">Publicar / Programar</h3>

                        {!uploadPostKey && (
                            <div className="mb-4 p-3 bg-yellow-500/10 border border-yellow-500/20 text-yellow-200 text-xs rounded-lg flex items-start gap-2">
                                <AlertCircle size={14} className="mt-0.5 shrink-0" />
                                <div>Configura primero la API Key en Configuración.</div>
                            </div>
                        )}

                        <div className="space-y-4 mb-6">
                            {/* Title & Description */}
                            <div>
                                <label className="block text-xs font-bold text-zinc-400 mb-1">Título del video</label>
                                <input 
                                    type="text" 
                                    value={postTitle}
                                    onChange={(e) => setPostTitle(e.target.value)}
                                    className="w-full bg-black/40 border border-white/10 rounded-lg p-2 text-sm text-white focus:outline-none focus:border-primary/50 placeholder-zinc-600"
                                    placeholder="Escribe un título llamativo..."
                                />
                            </div>

                            <div>
                                <label className="block text-xs font-bold text-zinc-400 mb-1">Texto / Descripción</label>
                                <textarea 
                                    value={postDescription}
                                    onChange={(e) => setPostDescription(e.target.value)}
                                    rows={4}
                                    className="w-full bg-black/40 border border-white/10 rounded-lg p-2 text-sm text-white focus:outline-none focus:border-primary/50 placeholder-zinc-600 resize-none"
                                    placeholder="Escribe un texto para tu publicación..."
                                />
                            </div>

                            {/* Scheduling */}
                            <div className="p-3 bg-white/5 rounded-lg border border-white/5">
                                <div className="flex items-center justify-between mb-2">
                                    <div className="flex items-center gap-2 text-sm text-white font-medium">
                                        <Calendar size={16} className="text-purple-400" /> Programar publicación
                                    </div>
                                    <label className="relative inline-flex items-center cursor-pointer">
                                        <input type="checkbox" checked={isScheduling} onChange={(e) => setIsScheduling(e.target.checked)} className="sr-only peer" />
                                        <div className="w-9 h-5 bg-zinc-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-purple-600"></div>
                                    </label>
                                </div>
                                
                                {isScheduling && (
                                    <div className="mt-3 animate-[fadeIn_0.2s_ease-out]">
                                        <label className="block text-xs text-zinc-400 mb-1">Selecciona fecha y hora</label>
                                        <div className="relative">
                                            <input 
                                                type="datetime-local" 
                                                value={scheduleDate}
                                                onChange={(e) => setScheduleDate(e.target.value)}
                                                className="w-full bg-black/40 border border-white/10 rounded-lg p-2 pl-9 text-sm text-white focus:outline-none focus:border-purple-500/50 [color-scheme:dark]"
                                            />
                                            <Clock size={14} className="absolute left-3 top-2.5 text-zinc-500" />
                                        </div>
                                    </div>
                                )}
                            </div>

                            {/* Platforms */}
                            <div>
                                <label className="block text-xs font-bold text-zinc-400 mb-2">Seleccionar plataformas</label>
                                <div className="grid grid-cols-1 gap-2">
                                    <label className="flex items-center gap-3 p-3 bg-white/5 rounded-lg cursor-pointer hover:bg-white/10 transition-colors border border-white/5">
                                        <input type="checkbox" checked={platforms.tiktok} onChange={e => setPlatforms({ ...platforms, tiktok: e.target.checked })} className="w-4 h-4 rounded border-zinc-600 bg-black/50 text-primary focus:ring-primary" />
                                        <div className="flex items-center gap-2 text-sm text-white"><Video size={16} className="text-cyan-400" /> TikTok</div>
                                    </label>
                                    <label className="flex items-center gap-3 p-3 bg-white/5 rounded-lg cursor-pointer hover:bg-white/10 transition-colors border border-white/5">
                                        <input type="checkbox" checked={platforms.instagram} onChange={e => setPlatforms({ ...platforms, instagram: e.target.checked })} className="w-4 h-4 rounded border-zinc-600 bg-black/50 text-primary focus:ring-primary" />
                                        <div className="flex items-center gap-2 text-sm text-white"><Instagram size={16} className="text-pink-400" /> Instagram</div>
                                    </label>
                                    <label className="flex items-center gap-3 p-3 bg-white/5 rounded-lg cursor-pointer hover:bg-white/10 transition-colors border border-white/5">
                                        <input type="checkbox" checked={platforms.youtube} onChange={e => setPlatforms({ ...platforms, youtube: e.target.checked })} className="w-4 h-4 rounded border-zinc-600 bg-black/50 text-primary focus:ring-primary" />
                                        <div className="flex items-center gap-2 text-sm text-white"><Youtube size={16} className="text-red-400" /> YouTube Shorts</div>
                                    </label>
                                </div>
                            </div>
                        </div>

                        {postResult && (
                            <div className={`mb-4 p-3 rounded-lg text-xs flex items-start gap-2 ${postResult.success ? 'bg-green-500/10 text-green-400' : 'bg-red-500/10 text-red-400'}`}>
                                {postResult.success ? <CheckCircle size={14} className="mt-0.5 shrink-0" /> : <AlertCircle size={14} className="mt-0.5 shrink-0" />}
                                <div>{postResult.msg}</div>
                            </div>
                        )}

                        <button
                            onClick={handlePost}
                            disabled={posting || !uploadPostKey}
                            className="w-full py-3 bg-primary hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed rounded-xl text-white font-bold transition-all flex items-center justify-center gap-2"
                        >
                            {posting ? <><Loader2 size={16} className="animate-spin" /> {isScheduling ? 'Programando...' : 'Publicando...'}</> : <><Share2 size={16} /> {isScheduling ? 'Programar publicación' : 'Publicar ahora'}</>}
                        </button>
                    </div>
                </div>
            )}

            {!onOpenStudio && (
                <ClipStudioModal
                    isOpen={showStudioModal}
                    onClose={() => setShowStudioModal(false)}
                    jobId={jobId}
                    clipIndex={clipIndex}
                    clip={clip}
                    currentVideoUrl={currentVideoUrl}
                    onApplied={({ newVideoUrl }) => {
                        if (newVideoUrl) {
                            setCurrentVideoUrl(newVideoUrl);
                            setBaseVideoUrl(newVideoUrl);
                            setSubtitledVideoUrl(null);
                            setSubtitlesEnabled(false);
                            if (videoRef.current) {
                                videoRef.current.load();
                            }
                        }
                        setShowStudioModal(false);
                    }}
                />
            )}

            {showRecutModal && (
                <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-[fadeIn_0.2s_ease-out]">
                    <div className="bg-[#121214] border border-white/10 p-6 rounded-2xl w-full max-w-3xl shadow-2xl relative">
                        <button
                            onClick={() => setShowRecutModal(false)}
                            className="absolute top-4 right-4 text-zinc-500 hover:text-white"
                        >
                            <X size={20} />
                        </button>
                        <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                            <Scissors size={16} className="text-primary" /> Editar video
                        </h3>
                        <div className="grid grid-cols-1 md:grid-cols-[1.4fr_1fr] gap-6">
                            <div className="bg-black/60 rounded-xl border border-white/10 p-3">
                                <video
                                    ref={editVideoRef}
                                    src={currentVideoUrl}
                                    controls
                                    className={`w-full rounded-lg ${isLandscape ? 'aspect-video' : 'aspect-[9/16]'}`}
                                />
                                <div className="mt-3 flex items-center justify-between text-xs text-zinc-400">
                                    <div className="flex items-center gap-2">
                                        <button
                                            onClick={() => {
                                                const v = editVideoRef.current;
                                                if (!v) return;
                                                if (v.paused) v.play();
                                                else v.pause();
                                            }}
                                            className="px-3 py-1 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 flex items-center gap-2"
                                        >
                                            {editPlaying ? <Pause size={12} /> : <Play size={12} />}
                                            {editPlaying ? 'Pausar' : 'Reproducir'}
                                        </button>
                                        <span>{formatTime(editCurrentTime)} / {formatTime(editDuration)}</span>
                                    </div>
                                    <button
                                        onClick={async () => {
                                            try {
                                                const response = await fetch(currentVideoUrl);
                                                if (!response.ok) throw new Error('Descarga fallida');
                                                const blob = await response.blob();
                                                const url = window.URL.createObjectURL(blob);
                                                const a = document.createElement('a');
                                                a.style.display = 'none';
                                                a.href = url;
                                                a.download = `clip-${displayIndex + 1}.mp4`;
                                                document.body.appendChild(a);
                                                a.click();
                                                window.URL.revokeObjectURL(url);
                                                document.body.removeChild(a);
                                            } catch (err) {
                                                window.open(currentVideoUrl, '_blank');
                                            }
                                        }}
                                        className="px-3 py-1 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 flex items-center gap-2"
                                    >
                                        <Download size={12} /> Descargar
                                    </button>
                                </div>
                            </div>
                            <div className="space-y-4">
                                <div>
                                    <label className="block text-xs font-bold text-zinc-400 mb-2">Rango de corte</label>
                                    <div className="space-y-3">
                                        <div>
                                            <div className="flex justify-between text-[10px] text-zinc-500 mb-1">
                                                <span>Inicio: {formatTime(Number(recutStart))}</span>
                                                <button
                                                    onClick={() => setRecutStart(editCurrentTime.toFixed(1))}
                                                    className="text-primary hover:text-primary/80"
                                                >
                                                    Fijar al playhead
                                                </button>
                                            </div>
                                            <input
                                                type="range"
                                                min="0"
                                                max={editDuration || 0}
                                                step="0.1"
                                                value={recutStart}
                                                onChange={(e) => {
                                                    const val = Number(e.target.value);
                                                    setRecutStart(val);
                                                    if (val >= Number(recutEnd)) setRecutEnd(val + 0.5);
                                                }}
                                                className="w-full"
                                            />
                                        </div>
                                        <div>
                                            <div className="flex justify-between text-[10px] text-zinc-500 mb-1">
                                                <span>Fin: {formatTime(Number(recutEnd))}</span>
                                                <button
                                                    onClick={() => setRecutEnd(editCurrentTime.toFixed(1))}
                                                    className="text-primary hover:text-primary/80"
                                                >
                                                    Fijar al playhead
                                                </button>
                                            </div>
                                            <input
                                                type="range"
                                                min="0"
                                                max={editDuration || 0}
                                                step="0.1"
                                                value={recutEnd}
                                                onChange={(e) => setRecutEnd(Number(e.target.value))}
                                                className="w-full"
                                            />
                                        </div>
                                    </div>
                                </div>
                                <div className="grid grid-cols-2 gap-3">
                                    <div>
                                        <label className="block text-xs font-bold text-zinc-400 mb-1">Inicio (segundos)</label>
                                        <input
                                            type="number"
                                            step="0.1"
                                            value={recutStart}
                                            onChange={(e) => setRecutStart(e.target.value)}
                                            className="w-full bg-black/40 border border-white/10 rounded-lg p-2 text-sm text-white focus:outline-none focus:border-primary/50"
                                        />
                                    </div>
                                    <div>
                                        <label className="block text-xs font-bold text-zinc-400 mb-1">Fin (segundos)</label>
                                        <input
                                            type="number"
                                            step="0.1"
                                            value={recutEnd}
                                            onChange={(e) => setRecutEnd(e.target.value)}
                                            className="w-full bg-black/40 border border-white/10 rounded-lg p-2 text-sm text-white focus:outline-none focus:border-primary/50"
                                        />
                                    </div>
                                </div>
                                <button
                                    onClick={handleRecut}
                                    disabled={isRecutting}
                                    className="w-full py-3 mt-2 bg-primary hover:bg-primary/90 rounded-xl text-white font-bold transition-all flex items-center justify-center gap-2"
                                >
                                    {isRecutting ? <Loader2 size={16} className="animate-spin" /> : <Scissors size={16} />}
                                    {isRecutting ? 'Procesando...' : 'Aplicar corte'}
                                </button>
                                <p className="text-[10px] text-zinc-500 text-center">
                                    Usa el playhead para marcar inicio/fin y ajustar el rango.
                                </p>
                            </div>
                        </div>
                        <p className="text-[10px] text-zinc-500 text-center mt-4">
                            El recorte requiere que el video original subido esté disponible.
                        </p>
                    </div>
                </div>
            )}
        </div>
    );
}
