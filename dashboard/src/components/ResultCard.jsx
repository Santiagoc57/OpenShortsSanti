import React, { useState, useEffect, useRef } from 'react';
import { Download, Share2, Instagram, Youtube, Video, CheckCircle, AlertCircle, X, Loader2, Copy, Wand2, Type, Calendar, Clock, Scissors, Play, Pause } from 'lucide-react';
import { getApiUrl, apiFetch } from '../config';
import SubtitleModal from './SubtitleModal';

export default function ResultCard({ clip, displayIndex = 0, clipIndex = 0, jobId, uploadPostKey, uploadUserId, geminiApiKey, onPlay, onPause }) {
    const [showModal, setShowModal] = useState(false);
    const [showSubtitleModal, setShowSubtitleModal] = useState(false);
    const [showRecutModal, setShowRecutModal] = useState(false);
    const videoRef = React.useRef(null);
    const editVideoRef = useRef(null);
    const [currentVideoUrl, setCurrentVideoUrl] = useState(getApiUrl(clip.video_url));
    const [baseVideoUrl, setBaseVideoUrl] = useState(getApiUrl(clip.video_url));
    const [subtitledVideoUrl, setSubtitledVideoUrl] = useState(null);
    const [subtitlesEnabled, setSubtitlesEnabled] = useState(false);

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
    const [isSubtitling, setIsSubtitling] = useState(false);
    const [isRecutting, setIsRecutting] = useState(false);
    const [editError, setEditError] = useState(null);
    const [recutStart, setRecutStart] = useState(clip.start);
    const [recutEnd, setRecutEnd] = useState(clip.end);
    const [editDuration, setEditDuration] = useState(0);
    const [editCurrentTime, setEditCurrentTime] = useState(0);
    const [editPlaying, setEditPlaying] = useState(false);

    // Initialize/Reset form when modal opens
    useEffect(() => {
        if (showModal) {
            setPostTitle(clip.video_title_for_youtube_short || "Viral Short");
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

    const handleAutoEdit = async () => {
        setIsEditing(true);
        setEditError(null);
        try {
             // Use passed prop or fallback
             const apiKey = geminiApiKey || localStorage.getItem('gemini_key');
             
             if (!apiKey) {
                 throw new Error("Gemini API Key is missing. Please set it in Settings.");
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

    const handleSubtitle = async (options) => {
        setIsSubtitling(true);
        setEditError(null);
        try {
            const res = await apiFetch('/api/subtitle', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    job_id: jobId,
                    clip_index: clipIndex,
                    position: options.position,
                    font_size: options.fontSize,
                    font_family: options.fontFamily,
                    font_color: options.fontColor,
                    stroke_color: options.strokeColor,
                    stroke_width: options.strokeWidth,
                    bold: options.bold,
                    box_color: options.boxColor,
                    box_opacity: options.boxOpacity,
                    srt_content: options.srtContent || null,
                    input_filename: currentVideoUrl.split('/').pop()
                })
            });

            if (!res.ok) {
                const errText = await res.text();
                throw new Error(errText);
            }

            const data = await res.json();
            if (data.new_video_url) {
                const baseUrl = currentVideoUrl;
                const subUrl = getApiUrl(data.new_video_url);
                setBaseVideoUrl(baseUrl);
                setSubtitledVideoUrl(subUrl);
                setSubtitlesEnabled(true);
                setCurrentVideoUrl(subUrl);
                if (videoRef.current) {
                    videoRef.current.load();
                }
                setShowSubtitleModal(false);
            }

        } catch (e) {
            setEditError(e.message);
            setTimeout(() => setEditError(null), 5000);
        } finally {
            setIsSubtitling(false);
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
                    end: Number(recutEnd)
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
            setPostResult({ success: false, msg: "Missing API Key or User ID." });
            return;
        }

        const selectedPlatforms = Object.keys(platforms).filter(k => platforms[k]);
        if (selectedPlatforms.length === 0) {
            setPostResult({ success: false, msg: "Select at least one platform." });
            return;
        }

        if (isScheduling && !scheduleDate) {
            setPostResult({ success: false, msg: "Please select a date and time." });
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

            setPostResult({ success: true, msg: isScheduling ? "Scheduled successfully!" : "Posted successfully!" });
            setTimeout(() => {
                setShowModal(false);
                setPostResult(null);
            }, 3000);

        } catch (e) {
            setPostResult({ success: false, msg: `Failed: ${e.message}` });
        } finally {
            setPosting(false);
        }
    };

    return (
        <div className="bg-surface border border-white/5 rounded-2xl overflow-hidden flex flex-col md:flex-row group hover:border-white/10 transition-all animate-[fadeIn_0.5s_ease-out] min-h-[300px] h-auto" style={{ animationDelay: `${displayIndex * 0.1}s` }}>
            {/* Left: Video Preview (Responsive Width) */}
            <div className="w-full md:w-[180px] lg:w-[200px] bg-black relative shrink-0 aspect-[9/16] md:aspect-auto group/video">
                <video
                    ref={videoRef}
                    src={currentVideoUrl}
                    controls
                    className="w-full h-full object-cover"
                    playsInline
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
                <div className="absolute top-3 left-3 flex gap-2">
                    <span className="bg-black/60 backdrop-blur-md text-white text-[10px] font-bold px-2 py-1 rounded-md border border-white/10 uppercase tracking-wide">
                        Clip {displayIndex + 1}
                    </span>
                </div>
                
                {/* Auto Edit Overlay if Processing */}
                {isEditing && (
                    <div className="absolute inset-0 bg-black/60 backdrop-blur-sm flex flex-col items-center justify-center z-10 p-4 text-center">
                        <Loader2 size={32} className="text-primary animate-spin mb-3" />
                        <span className="text-xs font-bold text-white uppercase tracking-wider">AI Magic in Progress...</span>
                        <span className="text-[10px] text-zinc-400 mt-1">Applying viral edits & zooms</span>
                    </div>
                )}
            </div>

            {/* Right: Content & Details */}
            <div className="flex-1 p-4 md:p-5 flex flex-col bg-[#121214] overflow-hidden min-w-0">
                <div className="mb-4">
                     <h3 className="text-base font-bold text-white leading-tight line-clamp-2 mb-2 break-words" title={clip.video_title_for_youtube_short}>
                        {clip.video_title_for_youtube_short || "Viral Clip Generated"}
                    </h3>
                    <div className="flex flex-wrap gap-2 text-[10px] text-zinc-500 font-mono">
                        <span className={`px-1.5 py-0.5 rounded border shrink-0 ${scoreBadgeClass}`} title={clip.score_reason || 'Predicted performance score'}>
                            Score {clipScore}/100
                        </span>
                        <span className="bg-white/5 px-1.5 py-0.5 rounded border border-white/5 shrink-0" title="Selection confidence">
                            Conf {Math.round(clipConfidence * 100)}%
                        </span>
                        <span className="bg-white/5 px-1.5 py-0.5 rounded border border-white/5 shrink-0">{Math.floor(clip.end - clip.start)}s</span>
                        <span className="bg-white/5 px-1.5 py-0.5 rounded border border-white/5 shrink-0">
                            {formatTime(clip.start)} - {formatTime(clip.end)}
                        </span>
                        <span className="bg-white/5 px-1.5 py-0.5 rounded border border-white/5 shrink-0">#shorts</span>
                        <span className="bg-white/5 px-1.5 py-0.5 rounded border border-white/5 shrink-0">#viral</span>
                    </div>
                    {clip.score_reason && (
                        <p className="mt-2 text-[10px] text-zinc-500 line-clamp-2" title={clip.score_reason}>
                            {clip.score_reason}
                        </p>
                    )}
                    {topicTags.length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-1">
                            {topicTags.slice(0, 5).map((tag) => (
                                <span key={tag} className="text-[10px] bg-white/5 border border-white/10 rounded px-1.5 py-0.5 text-zinc-300">
                                    #{tag}
                                </span>
                            ))}
                        </div>
                    )}
                    {subtitledVideoUrl && (
                        <div className="mt-2 flex items-center gap-2 text-[10px] text-zinc-400">
                            <span>Subtitles</span>
                            <button
                                onClick={() => {
                                    const next = !subtitlesEnabled;
                                    setSubtitlesEnabled(next);
                                    setCurrentVideoUrl(next ? subtitledVideoUrl : baseVideoUrl);
                                    if (videoRef.current) {
                                        videoRef.current.load();
                                    }
                                }}
                                className={`px-2 py-1 rounded-full border transition-colors ${subtitlesEnabled ? 'bg-yellow-500/20 border-yellow-500/40 text-yellow-300' : 'bg-white/5 border-white/10 text-zinc-400'}`}
                            >
                                {subtitlesEnabled ? 'ON' : 'OFF'}
                            </button>
                        </div>
                    )}
                </div>

                {/* Scrollable Descriptions Area */}
                <div className="flex-1 overflow-y-auto custom-scrollbar space-y-3 pr-2 mb-4">
                     {/* YouTube */}
                     <div className="bg-black/20 rounded-lg p-3 border border-white/5">
                        <div className="flex items-center gap-2 text-[10px] font-bold text-red-400 mb-1.5 uppercase tracking-wider">
                            <Youtube size={12} className="shrink-0" /> <span className="truncate">YouTube Title</span>
                        </div>
                        <p className="text-xs text-zinc-300 select-all break-words">
                            {clip.video_title_for_youtube_short || "Viral Short Video"}
                        </p>
                     </div>

                     {/* TikTok / IG */}
                     <div className="bg-black/20 rounded-lg p-3 border border-white/5">
                        <div className="flex items-center gap-2 text-[10px] font-bold text-zinc-400 mb-1.5 uppercase tracking-wider">
                            <Video size={12} className="text-cyan-400 shrink-0" /> 
                            <span className="text-zinc-500">/</span>
                            <Instagram size={12} className="text-pink-400 shrink-0" />
                            <span className="truncate">Caption</span>
                        </div>
                        <p className="text-xs text-zinc-300 line-clamp-3 hover:line-clamp-none transition-all cursor-pointer select-all break-words">
                            {clip.video_description_for_tiktok || clip.video_description_for_instagram}
                        </p>
                     </div>
                </div>

                {/* Error Message */}
                {editError && (
                    <div className="mb-3 p-2 bg-red-500/10 border border-red-500/20 text-red-400 text-[10px] rounded-lg flex items-center gap-2">
                        <AlertCircle size={12} className="shrink-0" />
                        {editError}
                    </div>
                )}

                {/* Actions Footer */}
                <div className="grid grid-cols-2 gap-3 mt-auto pt-4 border-t border-white/5">
                    <button
                        onClick={handleAutoEdit}
                        disabled={isEditing}
                        className="col-span-1 py-2 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white rounded-lg text-xs font-bold shadow-lg shadow-purple-500/20 transition-all active:scale-[0.98] flex items-center justify-center gap-2 mb-1 truncate px-1"
                    >
                        {isEditing ? <Loader2 size={14} className="animate-spin" /> : <Wand2 size={14} />} 
                        {isEditing ? 'Editing...' : 'Auto Edit'}
                    </button>

                    <button
                        onClick={() => setShowSubtitleModal(true)}
                        disabled={isSubtitling}
                        className="col-span-1 py-2 bg-gradient-to-r from-yellow-600 to-orange-600 hover:from-yellow-500 hover:to-orange-500 text-white rounded-lg text-xs font-bold shadow-lg shadow-orange-500/20 transition-all active:scale-[0.98] flex items-center justify-center gap-2 mb-1 truncate px-1"
                    >
                        {isSubtitling ? <Loader2 size={14} className="animate-spin" /> : <Type size={14} />} 
                        {isSubtitling ? 'Adding...' : 'Subtitles'}
                    </button>

                    <button
                        onClick={() => setShowRecutModal(true)}
                        className="col-span-1 py-2 bg-white/10 hover:bg-white/20 text-white rounded-lg text-xs font-bold shadow-lg shadow-white/10 transition-all active:scale-[0.98] flex items-center justify-center gap-2 mb-1 truncate px-1"
                    >
                        <Scissors size={14} />
                        Edit Video
                    </button>

                    <button
                        onClick={() => setShowModal(true)}
                        className="col-span-1 py-2 bg-primary hover:bg-blue-600 text-white rounded-lg text-xs font-bold shadow-lg shadow-primary/20 transition-all active:scale-[0.98] flex items-center justify-center gap-2 truncate px-2"
                    >
                        <Share2 size={14} className="shrink-0" /> Post
                    </button>
                    <button
                        onClick={async (e) => {
                            e.preventDefault();
                            try {
                                const response = await fetch(currentVideoUrl);
                                if (!response.ok) throw new Error('Download failed');
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
                                console.error('Download error:', err);
                                window.open(currentVideoUrl, '_blank');
                            }
                        }}
                        className="col-span-1 py-2 bg-white/5 hover:bg-white/10 text-zinc-300 hover:text-white rounded-lg text-xs font-medium transition-colors flex items-center justify-center gap-2 border border-white/5 truncate px-2"
                    >
                        <Download size={14} className="shrink-0" /> Download
                    </button>
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

                        <h3 className="text-lg font-bold text-white mb-4">Post / Schedule</h3>

                        {!uploadPostKey && (
                            <div className="mb-4 p-3 bg-yellow-500/10 border border-yellow-500/20 text-yellow-200 text-xs rounded-lg flex items-start gap-2">
                                <AlertCircle size={14} className="mt-0.5 shrink-0" />
                                <div>Configure API Key in Settings first.</div>
                            </div>
                        )}

                        <div className="space-y-4 mb-6">
                            {/* Title & Description */}
                            <div>
                                <label className="block text-xs font-bold text-zinc-400 mb-1">Video Title</label>
                                <input 
                                    type="text" 
                                    value={postTitle}
                                    onChange={(e) => setPostTitle(e.target.value)}
                                    className="w-full bg-black/40 border border-white/10 rounded-lg p-2 text-sm text-white focus:outline-none focus:border-primary/50 placeholder-zinc-600"
                                    placeholder="Enter a catchy title..."
                                />
                            </div>

                            <div>
                                <label className="block text-xs font-bold text-zinc-400 mb-1">Caption / Description</label>
                                <textarea 
                                    value={postDescription}
                                    onChange={(e) => setPostDescription(e.target.value)}
                                    rows={4}
                                    className="w-full bg-black/40 border border-white/10 rounded-lg p-2 text-sm text-white focus:outline-none focus:border-primary/50 placeholder-zinc-600 resize-none"
                                    placeholder="Write a caption for your post..."
                                />
                            </div>

                            {/* Scheduling */}
                            <div className="p-3 bg-white/5 rounded-lg border border-white/5">
                                <div className="flex items-center justify-between mb-2">
                                    <div className="flex items-center gap-2 text-sm text-white font-medium">
                                        <Calendar size={16} className="text-purple-400" /> Schedule Post
                                    </div>
                                    <label className="relative inline-flex items-center cursor-pointer">
                                        <input type="checkbox" checked={isScheduling} onChange={(e) => setIsScheduling(e.target.checked)} className="sr-only peer" />
                                        <div className="w-9 h-5 bg-zinc-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-purple-600"></div>
                                    </label>
                                </div>
                                
                                {isScheduling && (
                                    <div className="mt-3 animate-[fadeIn_0.2s_ease-out]">
                                        <label className="block text-xs text-zinc-400 mb-1">Select Date & Time</label>
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
                                <label className="block text-xs font-bold text-zinc-400 mb-2">Select Platforms</label>
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
                            {posting ? <><Loader2 size={16} className="animate-spin" /> {isScheduling ? 'Scheduling...' : 'Publishing...'}</> : <><Share2 size={16} /> {isScheduling ? 'Schedule Post' : 'Publish Now'}</>}
                        </button>
                    </div>
                </div>
            )}

            <SubtitleModal 
                isOpen={showSubtitleModal}
                onClose={() => setShowSubtitleModal(false)}
                onGenerate={handleSubtitle}
                isProcessing={isSubtitling}
                videoUrl={currentVideoUrl}
                onLoadSrt={async () => {
                    const res = await apiFetch('/api/subtitle/preview', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ job_id: jobId, clip_index: clipIndex })
                    });
                    if (!res.ok) {
                        const errText = await res.text();
                        throw new Error(errText);
                    }
                    const data = await res.json();
                    return data.srt || '';
                }}
            />

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
                            <Scissors size={16} className="text-primary" /> Edit Video
                        </h3>
                        <div className="grid grid-cols-1 md:grid-cols-[1.4fr_1fr] gap-6">
                            <div className="bg-black/60 rounded-xl border border-white/10 p-3">
                                <video
                                    ref={editVideoRef}
                                    src={currentVideoUrl}
                                    controls
                                    className="w-full aspect-video rounded-lg"
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
                                            {editPlaying ? 'Pause' : 'Play'}
                                        </button>
                                        <span>{formatTime(editCurrentTime)} / {formatTime(editDuration)}</span>
                                    </div>
                                    <button
                                        onClick={async () => {
                                            try {
                                                const response = await fetch(currentVideoUrl);
                                                if (!response.ok) throw new Error('Download failed');
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
                                        <Download size={12} /> Download
                                    </button>
                                </div>
                            </div>
                            <div className="space-y-4">
                                <div>
                                    <label className="block text-xs font-bold text-zinc-400 mb-2">Trim Range</label>
                                    <div className="space-y-3">
                                        <div>
                                            <div className="flex justify-between text-[10px] text-zinc-500 mb-1">
                                                <span>Start: {formatTime(Number(recutStart))}</span>
                                                <button
                                                    onClick={() => setRecutStart(editCurrentTime.toFixed(1))}
                                                    className="text-primary hover:text-primary/80"
                                                >
                                                    Set to playhead
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
                                                <span>End: {formatTime(Number(recutEnd))}</span>
                                                <button
                                                    onClick={() => setRecutEnd(editCurrentTime.toFixed(1))}
                                                    className="text-primary hover:text-primary/80"
                                                >
                                                    Set to playhead
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
                                        <label className="block text-xs font-bold text-zinc-400 mb-1">Start (seconds)</label>
                                        <input
                                            type="number"
                                            step="0.1"
                                            value={recutStart}
                                            onChange={(e) => setRecutStart(e.target.value)}
                                            className="w-full bg-black/40 border border-white/10 rounded-lg p-2 text-sm text-white focus:outline-none focus:border-primary/50"
                                        />
                                    </div>
                                    <div>
                                        <label className="block text-xs font-bold text-zinc-400 mb-1">End (seconds)</label>
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
                                    {isRecutting ? 'Processing...' : 'Apply Cut'}
                                </button>
                                <p className="text-[10px] text-zinc-500 text-center">
                                    Usa el playhead para marcar inicio/fin y ajustar el rango.
                                </p>
                            </div>
                        </div>
                        <p className="text-[10px] text-zinc-500 text-center mt-4">
                            Recut requires the original uploaded video to be available.
                        </p>
                    </div>
                </div>
            )}
        </div>
    );
}
