import React, { useState } from 'react';
import { Youtube, Upload, FileVideo, X } from 'lucide-react';

export default function MediaInput({ onProcess, isProcessing }) {
    const [mode, setMode] = useState('url'); // 'url' | 'file'
    const [url, setUrl] = useState('');
    const [file, setFile] = useState(null);
    const [language, setLanguage] = useState('auto');
    const [clipCount, setClipCount] = useState(6);
    const [whisperBackend, setWhisperBackend] = useState('openai');
    const [whisperModel, setWhisperModel] = useState('base');
    const [wordTimestamps, setWordTimestamps] = useState(true);
    const [ffmpegPreset, setFfmpegPreset] = useState('fast');
    const [ffmpegCrf, setFfmpegCrf] = useState(23);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (mode === 'url' && url) {
            onProcess({
                type: 'url',
                payload: url,
                language,
                clipCount,
                whisperBackend,
                whisperModel,
                wordTimestamps,
                ffmpegPreset,
                ffmpegCrf
            });
        } else if (mode === 'file' && file) {
            onProcess({
                type: 'file',
                payload: file,
                language,
                clipCount,
                whisperBackend,
                whisperModel,
                wordTimestamps,
                ffmpegPreset,
                ffmpegCrf
            });
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            setFile(e.dataTransfer.files[0]);
            setMode('file');
        }
    };

    return (
        <div className="bg-surface border border-white/5 rounded-2xl p-6 animate-[fadeIn_0.6s_ease-out]">
            <div className="flex gap-4 mb-6 border-b border-white/5 pb-4">
                <button
                    onClick={() => setMode('url')}
                    className={`flex items-center gap-2 pb-2 px-2 transition-all ${mode === 'url'
                        ? 'text-primary border-b-2 border-primary -mb-[17px]'
                        : 'text-zinc-400 hover:text-white'
                        }`}
                >
                    <Youtube size={18} />
                    YouTube URL
                </button>
                <button
                    onClick={() => setMode('file')}
                    className={`flex items-center gap-2 pb-2 px-2 transition-all ${mode === 'file'
                        ? 'text-primary border-b-2 border-primary -mb-[17px]'
                        : 'text-zinc-400 hover:text-white'
                        }`}
                >
                    <Upload size={18} />
                    Upload File
                </button>
            </div>

            <form onSubmit={handleSubmit}>
                {mode === 'url' ? (
                    <div className="space-y-4">
                        <input
                            type="url"
                            value={url}
                            onChange={(e) => setUrl(e.target.value)}
                            placeholder="https://www.youtube.com/watch?v=..."
                            className="input-field"
                            required
                        />
                    </div>
                ) : (
                    <div
                        className={`border-2 border-dashed rounded-xl p-8 text-center transition-all ${file ? 'border-primary/50 bg-primary/5' : 'border-zinc-700 hover:border-zinc-500 bg-white/5'
                            }`}
                        onDragOver={(e) => e.preventDefault()}
                        onDrop={handleDrop}
                    >
                        {file ? (
                            <div className="flex items-center justify-center gap-3 text-white">
                                <FileVideo className="text-primary" />
                                <span className="font-medium">{file.name}</span>
                                <button
                                    type="button"
                                    onClick={() => setFile(null)}
                                    className="p-1 hover:bg-white/10 rounded-full"
                                >
                                    <X size={16} />
                                </button>
                            </div>
                        ) : (
                            <label className="cursor-pointer block">
                                <input
                                    type="file"
                                    accept="video/*"
                                    onChange={(e) => setFile(e.target.files?.[0] || null)}
                                    className="hidden"
                                />
                                <Upload className="mx-auto mb-3 text-zinc-500" size={24} />
                                <p className="text-zinc-400">Click to upload or drag and drop</p>
                                <p className="text-xs text-zinc-600 mt-1">MP4, MOV up to 500MB</p>
                            </label>
                        )}
                    </div>
                )}

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
                    <div className="space-y-2">
                        <label className="block text-sm text-zinc-400">Idioma del video</label>
                        <select
                            value={language}
                            onChange={(e) => setLanguage(e.target.value)}
                            className="input-field"
                        >
                            <option value="auto">Auto detectar</option>
                            <option value="es">Español</option>
                            <option value="en">Inglés</option>
                            <option value="pt">Portugués</option>
                            <option value="fr">Francés</option>
                            <option value="de">Alemán</option>
                            <option value="it">Italiano</option>
                        </select>
                    </div>
                    <div className="space-y-2">
                        <label className="block text-sm text-zinc-400">Número de clips</label>
                        <input
                            type="number"
                            min="1"
                            max="15"
                            value={clipCount}
                            onChange={(e) => setClipCount(Number(e.target.value || 1))}
                            className="input-field"
                        />
                    </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
                    <div className="space-y-2">
                        <label className="block text-sm text-zinc-400">Whisper backend</label>
                        <select
                            value={whisperBackend}
                            onChange={(e) => setWhisperBackend(e.target.value)}
                            className="input-field"
                        >
                            <option value="openai">openai-whisper (más estable)</option>
                            <option value="faster">faster-whisper (más rápido, menos estable)</option>
                        </select>
                    </div>
                    <div className="space-y-2">
                        <label className="block text-sm text-zinc-400">Modelo</label>
                        <select
                            value={whisperModel}
                            onChange={(e) => setWhisperModel(e.target.value)}
                            className="input-field"
                        >
                            <option value="tiny">tiny (muy rápido)</option>
                            <option value="base">base (balance)</option>
                            <option value="small">small (mejor accuracy)</option>
                        </select>
                    </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
                    <div className="space-y-2">
                        <label className="block text-sm text-zinc-400">Subtítulos precisos (word timestamps)</label>
                        <select
                            value={wordTimestamps ? 'yes' : 'no'}
                            onChange={(e) => setWordTimestamps(e.target.value === 'yes')}
                            className="input-field"
                        >
                            <option value="yes">Sí (más lento)</option>
                            <option value="no">No (más rápido)</option>
                        </select>
                    </div>
                    <div className="space-y-2">
                        <label className="block text-sm text-zinc-400">FFmpeg preset</label>
                        <select
                            value={ffmpegPreset}
                            onChange={(e) => setFfmpegPreset(e.target.value)}
                            className="input-field"
                        >
                            <option value="ultrafast">ultrafast (más rápido)</option>
                            <option value="fast">fast</option>
                            <option value="medium">medium (mejor calidad)</option>
                        </select>
                    </div>
                </div>

                <div className="mt-6 space-y-2">
                    <label className="block text-sm text-zinc-400">Calidad de video (CRF)</label>
                    <input
                        type="number"
                        min="18"
                        max="30"
                        value={ffmpegCrf}
                        onChange={(e) => setFfmpegCrf(Number(e.target.value || 23))}
                        className="input-field"
                    />
                </div>

                <button
                    type="submit"
                    disabled={isProcessing || (mode === 'url' && !url) || (mode === 'file' && !file)}
                    className="w-full btn-primary mt-6 flex items-center justify-center gap-2"
                >
                    {isProcessing ? (
                        <>
                            <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                            Processing Video...
                        </>
                    ) : (
                        <>
                            Generate Clips
                        </>
                    )}
                </button>
            </form>
        </div>
    );
}
