import React, { useEffect, useState } from 'react';
import { Youtube, Upload, FileVideo, X, CheckCircle2, Settings2 } from 'lucide-react';

const MEDIA_INPUT_STORAGE_KEY = 'mediaInputPresetV1';

const ALLOWED_VALUES = {
    language: ['auto', 'es', 'en', 'pt', 'fr', 'de', 'it'],
    whisperBackend: ['openai', 'faster'],
    whisperModel: ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3', 'distil-large-v3'],
    ffmpegPreset: ['ultrafast', 'fast', 'medium'],
    aspectRatio: ['9:16', '16:9'],
    clipLengthTarget: ['short', 'balanced', 'long']
};

const MODEL_OPTIONS_BY_BACKEND = {
    openai: [
        { value: 'tiny', label: 'tiny (muy rápido)' },
        { value: 'base', label: 'base (equilibrado)' },
        { value: 'small', label: 'small (mejor precisión)' },
        { value: 'medium', label: 'medium (alta precisión)' },
        { value: 'large', label: 'large (máxima precisión)' }
    ],
    faster: [
        { value: 'tiny', label: 'tiny (muy rápido)' },
        { value: 'base', label: 'base (equilibrado)' },
        { value: 'small', label: 'small (mejor precisión)' },
        { value: 'medium', label: 'medium (alta precisión)' },
        { value: 'large-v2', label: 'large-v2 (muy alta precisión)' },
        { value: 'large-v3', label: 'large-v3 (recomendado Colab)' },
        { value: 'distil-large-v3', label: 'distil-large-v3 (rápido + preciso)' }
    ]
};

const CONTENT_PRESETS = [
    {
        id: 'general',
        name: 'General',
        subtitle: 'Todoterreno',
        settings: {
            clipLengthTarget: 'balanced',
            clipCount: 6,
            whisperBackend: 'openai',
            whisperModel: 'base',
            wordTimestamps: true
        }
    },
    {
        id: 'podcast',
        name: 'Podcast',
        subtitle: 'Conversación y contexto',
        settings: {
            clipLengthTarget: 'long',
            clipCount: 5,
            whisperBackend: 'openai',
            whisperModel: 'small',
            wordTimestamps: true,
            ffmpegPreset: 'medium',
            ffmpegCrf: 21
        }
    },
    {
        id: 'tutorial',
        name: 'Tutorial',
        subtitle: 'Explicación clara',
        settings: {
            clipLengthTarget: 'balanced',
            clipCount: 6,
            whisperBackend: 'openai',
            whisperModel: 'small',
            wordTimestamps: true,
            ffmpegPreset: 'medium',
            ffmpegCrf: 22
        }
    },
    {
        id: 'entrevista',
        name: 'Entrevista',
        subtitle: 'Momentos y frases',
        settings: {
            clipLengthTarget: 'balanced',
            clipCount: 7,
            whisperBackend: 'openai',
            whisperModel: 'base',
            wordTimestamps: true,
            ffmpegPreset: 'fast',
            ffmpegCrf: 22
        }
    }
];

const TEMPLATE_PRESETS = [
    {
        id: 'default',
        name: 'Predeterminado',
        subtitle: 'Balanceado general',
        gradient: 'from-zinc-500 to-zinc-800',
        settings: {
            aspectRatio: '9:16',
            clipLengthTarget: 'balanced',
            clipCount: 6,
            whisperBackend: 'openai',
            whisperModel: 'base',
            wordTimestamps: true,
            ffmpegPreset: 'fast',
            ffmpegCrf: 23
        }
    },
    {
        id: 'modern',
        name: 'Moderno',
        subtitle: 'Más ritmo y claridad',
        gradient: 'from-sky-400 to-indigo-600',
        settings: {
            aspectRatio: '9:16',
            clipLengthTarget: 'short',
            clipCount: 7,
            whisperBackend: 'openai',
            whisperModel: 'small',
            wordTimestamps: true,
            ffmpegPreset: 'medium',
            ffmpegCrf: 21
        }
    },
    {
        id: 'bouncy',
        name: 'Dinámico',
        subtitle: 'Cortes cortos dinámicos',
        gradient: 'from-fuchsia-500 to-purple-700',
        settings: {
            aspectRatio: '9:16',
            clipLengthTarget: 'short',
            clipCount: 8,
            whisperBackend: 'faster',
            whisperModel: 'base',
            wordTimestamps: false,
            ffmpegPreset: 'fast',
            ffmpegCrf: 24
        }
    },
    {
        id: 'mrbeast',
        name: 'MrBeast',
        subtitle: 'Retención agresiva',
        gradient: 'from-cyan-400 to-blue-600',
        settings: {
            aspectRatio: '9:16',
            clipLengthTarget: 'short',
            clipCount: 9,
            whisperBackend: 'openai',
            whisperModel: 'small',
            wordTimestamps: true,
            ffmpegPreset: 'fast',
            ffmpegCrf: 20
        }
    },
    {
        id: 'business',
        name: 'Negocios',
        subtitle: 'Más formal, menos jumpy',
        gradient: 'from-amber-400 to-orange-600',
        settings: {
            aspectRatio: '16:9',
            clipLengthTarget: 'long',
            clipCount: 5,
            whisperBackend: 'openai',
            whisperModel: 'base',
            wordTimestamps: true,
            ffmpegPreset: 'medium',
            ffmpegCrf: 21
        }
    }
];

const WHISPER_OPTION_PRESETS = [
    {
        id: 'colab_pro',
        name: 'Colab Pro',
        subtitle: 'faster + large-v3 + timestamps',
        settings: {
            whisperBackend: 'faster',
            whisperModel: 'large-v3',
            wordTimestamps: true,
            ffmpegPreset: 'fast',
            ffmpegCrf: 22
        }
    },
    {
        id: 'estable',
        name: 'Estable',
        subtitle: 'openai + base + timestamps',
        settings: {
            whisperBackend: 'openai',
            whisperModel: 'base',
            wordTimestamps: true,
            ffmpegPreset: 'fast',
            ffmpegCrf: 23
        }
    },
    {
        id: 'rapido',
        name: 'Rápido',
        subtitle: 'faster + tiny + sin timestamps',
        settings: {
            whisperBackend: 'faster',
            whisperModel: 'tiny',
            wordTimestamps: false,
            ffmpegPreset: 'ultrafast',
            ffmpegCrf: 25
        }
    },
    {
        id: 'preciso',
        name: 'Preciso',
        subtitle: 'openai + small + mayor calidad',
        settings: {
            whisperBackend: 'openai',
            whisperModel: 'small',
            wordTimestamps: true,
            ffmpegPreset: 'medium',
            ffmpegCrf: 21
        }
    }
];

function pickAllowed(value, allowed, fallback) {
    if (!value) return fallback;
    return allowed.includes(value) ? value : fallback;
}

function clampNumber(value, fallback, min, max) {
    const n = Number(value);
    if (!Number.isFinite(n)) return fallback;
    return Math.min(max, Math.max(min, n));
}

function normalizeWhisperModelForBackend(backend, model) {
    const backendKey = backend === 'faster' ? 'faster' : 'openai';
    const options = MODEL_OPTIONS_BY_BACKEND[backendKey] || MODEL_OPTIONS_BY_BACKEND.openai;
    if (options.some((opt) => opt.value === model)) return model;
    const fallback = backendKey === 'faster' ? 'large-v3' : 'base';
    if (options.some((opt) => opt.value === fallback)) return fallback;
    return options[0]?.value || 'base';
}

function loadStoredMediaInputConfig() {
    if (typeof window === 'undefined') return null;
    try {
        const raw = window.localStorage.getItem(MEDIA_INPUT_STORAGE_KEY);
        if (!raw) return null;
        const parsed = JSON.parse(raw);
        if (!parsed || typeof parsed !== 'object') return null;
        return {
            language: pickAllowed(parsed.language, ALLOWED_VALUES.language, 'auto'),
            clipCount: clampNumber(parsed.clipCount, 6, 1, 15),
            whisperBackend: pickAllowed(parsed.whisperBackend, ALLOWED_VALUES.whisperBackend, 'faster'),
            whisperModel: normalizeWhisperModelForBackend(
                pickAllowed(parsed.whisperBackend, ALLOWED_VALUES.whisperBackend, 'faster'),
                pickAllowed(parsed.whisperModel, ALLOWED_VALUES.whisperModel, 'large-v3')
            ),
            wordTimestamps: typeof parsed.wordTimestamps === 'boolean' ? parsed.wordTimestamps : true,
            ffmpegPreset: pickAllowed(parsed.ffmpegPreset, ALLOWED_VALUES.ffmpegPreset, 'fast'),
            ffmpegCrf: clampNumber(parsed.ffmpegCrf, 23, 18, 30),
            aspectRatio: pickAllowed(parsed.aspectRatio, ALLOWED_VALUES.aspectRatio, '9:16'),
            clipLengthTarget: pickAllowed(parsed.clipLengthTarget, ALLOWED_VALUES.clipLengthTarget, 'balanced'),
            selectedTemplate: TEMPLATE_PRESETS.some((p) => p.id === parsed.selectedTemplate) ? parsed.selectedTemplate : 'default',
            selectedContentPreset: CONTENT_PRESETS.some((p) => p.id === parsed.selectedContentPreset) ? parsed.selectedContentPreset : 'general',
            selectedWhisperOption: (WHISPER_OPTION_PRESETS.some((p) => p.id === parsed.selectedWhisperOption) || parsed.selectedWhisperOption === 'custom')
                ? parsed.selectedWhisperOption
                : 'colab_pro'
        };
    } catch (error) {
        console.warn('Failed to load media input config:', error);
        return null;
    }
}

export default function MediaInput({ onProcess, isProcessing }) {
    const [initialConfig] = useState(() => loadStoredMediaInputConfig());
    const [mode, setMode] = useState('file'); // 'url' | 'file'
    const [url, setUrl] = useState('');
    const [file, setFile] = useState(null);
    const [language, setLanguage] = useState(initialConfig?.language ?? 'auto');
    const [clipCount, setClipCount] = useState(initialConfig?.clipCount ?? 6);
    const [whisperBackend, setWhisperBackend] = useState(initialConfig?.whisperBackend ?? 'faster');
    const [whisperModel, setWhisperModel] = useState(initialConfig?.whisperModel ?? 'large-v3');
    const [wordTimestamps, setWordTimestamps] = useState(initialConfig?.wordTimestamps ?? true);
    const [ffmpegPreset, setFfmpegPreset] = useState(initialConfig?.ffmpegPreset ?? 'fast');
    const [ffmpegCrf, setFfmpegCrf] = useState(initialConfig?.ffmpegCrf ?? 23);
    const [aspectRatio, setAspectRatio] = useState(initialConfig?.aspectRatio ?? '9:16');
    const [clipLengthTarget, setClipLengthTarget] = useState(initialConfig?.clipLengthTarget ?? 'balanced');
    const [selectedTemplate, setSelectedTemplate] = useState(initialConfig?.selectedTemplate ?? 'default');
    const [selectedContentPreset, setSelectedContentPreset] = useState(initialConfig?.selectedContentPreset ?? 'general');
    const [selectedWhisperOption, setSelectedWhisperOption] = useState(initialConfig?.selectedWhisperOption ?? 'colab_pro');
    const [showConfigModal, setShowConfigModal] = useState(false);
    const whisperModelOptions = MODEL_OPTIONS_BY_BACKEND[whisperBackend] || MODEL_OPTIONS_BY_BACKEND.openai;

    useEffect(() => {
        if (typeof window === 'undefined') return;
        const payload = {
            language,
            clipCount,
            whisperBackend,
            whisperModel,
            wordTimestamps,
            ffmpegPreset,
            ffmpegCrf,
            aspectRatio,
            clipLengthTarget,
            selectedTemplate,
            selectedContentPreset,
            selectedWhisperOption
        };
        window.localStorage.setItem(MEDIA_INPUT_STORAGE_KEY, JSON.stringify(payload));
    }, [
        language,
        clipCount,
        whisperBackend,
        whisperModel,
        wordTimestamps,
        ffmpegPreset,
        ffmpegCrf,
        aspectRatio,
        clipLengthTarget,
        selectedTemplate,
        selectedContentPreset,
        selectedWhisperOption
    ]);

    const applySettings = (settings) => {
        if (!settings || typeof settings !== 'object') return;
        const nextBackend = settings.whisperBackend || whisperBackend;
        const nextModel = normalizeWhisperModelForBackend(nextBackend, settings.whisperModel || whisperModel);
        if (settings.aspectRatio) setAspectRatio(settings.aspectRatio);
        if (settings.clipLengthTarget) setClipLengthTarget(settings.clipLengthTarget);
        if (typeof settings.clipCount === 'number') setClipCount(settings.clipCount);
        setWhisperBackend(nextBackend);
        setWhisperModel(nextModel);
        if (typeof settings.wordTimestamps === 'boolean') setWordTimestamps(settings.wordTimestamps);
        if (settings.ffmpegPreset) setFfmpegPreset(settings.ffmpegPreset);
        if (typeof settings.ffmpegCrf === 'number') setFfmpegCrf(settings.ffmpegCrf);
    };

    useEffect(() => {
        const normalizedModel = normalizeWhisperModelForBackend(whisperBackend, whisperModel);
        if (normalizedModel !== whisperModel) {
            setWhisperModel(normalizedModel);
        }
    }, [whisperBackend, whisperModel]);

    const applyTemplate = (templateId) => {
        const preset = TEMPLATE_PRESETS.find((p) => p.id === templateId);
        if (!preset) return;
        setSelectedTemplate(templateId);
        setSelectedWhisperOption('custom');
        applySettings(preset.settings);
    };

    const applyContentPreset = (presetId) => {
        const preset = CONTENT_PRESETS.find((p) => p.id === presetId);
        if (!preset) return;
        setSelectedContentPreset(presetId);
        setSelectedWhisperOption('custom');
        applySettings(preset.settings);
    };

    const applyWhisperOptionPreset = (presetId) => {
        const preset = WHISPER_OPTION_PRESETS.find((p) => p.id === presetId);
        if (!preset) return;
        setSelectedWhisperOption(presetId);
        applySettings(preset.settings);
    };

    const canConfigure = (mode === 'url' && url.trim()) || (mode === 'file' && file);
    const modalInputClass = 'w-full rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 px-4 py-3.5 text-sm text-slate-900 dark:text-white shadow-sm focus:border-primary focus:ring-primary';
    const modalLabelClass = 'block text-sm font-medium text-slate-700 dark:text-slate-300';
    const modalCardClass = 'rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900/60 p-5 shadow-sm';

    const handleGenerate = () => {
        if (!canConfigure) return;

        const payloadBase = {
            language,
            clipCount,
            whisperBackend,
            whisperModel,
            wordTimestamps,
            ffmpegPreset,
            ffmpegCrf,
            aspectRatio,
            clipLengthTarget,
            styleTemplate: selectedTemplate,
            contentPreset: selectedContentPreset
        };

        if (mode === 'url' && url) {
            onProcess({
                type: 'url',
                payload: url,
                ...payloadBase
            });
        } else if (mode === 'file' && file) {
            onProcess({
                type: 'file',
                payload: file,
                ...payloadBase
            });
        }

        setShowConfigModal(false);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            setFile(e.dataTransfer.files[0]);
            setMode('file');
            setShowConfigModal(true);
        }
    };

    return (
        <>
            <div className="bg-white/90 dark:bg-surface border border-slate-200 dark:border-white/10 rounded-2xl p-6 shadow-sm animate-[fadeIn_0.6s_ease-out]">
                <div className="flex gap-4 mb-6 border-b border-slate-200 dark:border-white/5 pb-4">
                    <button
                        onClick={() => setMode('url')}
                        className={`flex items-center gap-2 pb-2 px-2 transition-all ${mode === 'url'
                            ? 'text-primary border-b-2 border-primary -mb-[17px]'
                            : 'text-zinc-500 hover:text-slate-900 dark:hover:text-white'
                            }`}
                    >
                        <Youtube size={18} />
                        URL de YouTube
                    </button>
                    <button
                        onClick={() => setMode('file')}
                        className={`flex items-center gap-2 pb-2 px-2 transition-all ${mode === 'file'
                            ? 'text-primary border-b-2 border-primary -mb-[17px]'
                            : 'text-zinc-500 hover:text-slate-900 dark:hover:text-white'
                            }`}
                    >
                        <Upload size={18} />
                        Subir archivo
                    </button>
                </div>

                {mode === 'url' ? (
                    <div className="space-y-4">
                        <input
                            type="url"
                            value={url}
                            onChange={(e) => setUrl(e.target.value)}
                            placeholder="https://www.youtube.com/watch?v=..."
                            className="input-field"
                        />
                    </div>
                ) : (
                    <div
                        className={`border-2 border-dashed rounded-xl p-8 text-center transition-all ${file ? 'border-primary/50 bg-primary/5' : 'border-slate-300 dark:border-zinc-700 hover:border-primary/40 bg-slate-50/70 dark:bg-white/5'}`}
                        onDragOver={(e) => e.preventDefault()}
                        onDrop={handleDrop}
                    >
                        {file ? (
                            <div className="flex items-center justify-center gap-3 text-slate-800 dark:text-white">
                                <FileVideo className="text-primary" />
                                <span className="font-medium">{file.name}</span>
                                <button
                                    type="button"
                                    onClick={() => setFile(null)}
                                    className="p-1 hover:bg-slate-200 dark:hover:bg-white/10 rounded-full"
                                >
                                    <X size={16} />
                                </button>
                            </div>
                        ) : (
                            <label className="cursor-pointer block">
                                <input
                                    type="file"
                                    accept="video/*,audio/*"
                                    onChange={(e) => {
                                        const nextFile = e.target.files?.[0] || null;
                                        setFile(nextFile);
                                        if (nextFile) {
                                            setMode('file');
                                            setShowConfigModal(true);
                                        }
                                    }}
                                    className="hidden"
                                />
                                <Upload className="mx-auto mb-3 text-zinc-500" size={24} />
                                <p className="text-zinc-600 dark:text-zinc-400">Haz clic para subir o arrastra y suelta</p>
                                <p className="text-xs text-zinc-500 dark:text-zinc-600 mt-1">MP4, MOV, MP3, WAV, M4A hasta 500MB</p>
                            </label>
                        )}
                    </div>
                )}

                <button
                    type="button"
                    disabled={isProcessing || !canConfigure}
                    onClick={() => setShowConfigModal(true)}
                    className="w-full btn-primary mt-6 flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    <Settings2 size={16} />
                    {mode === 'file' ? 'Configurar y generar' : 'Continuar a configuración'}
                </button>
            </div>

            {showConfigModal && (
                <div className="fixed inset-0 z-[110] bg-black/60 backdrop-blur-sm overflow-y-auto p-4 md:p-6">
                    <div className="mx-auto my-4 md:my-8 w-full max-w-5xl rounded-2xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-950 p-5 md:p-8 shadow-2xl">
                        <div className="mb-8 flex items-start justify-between gap-4">
                            <div>
                                <h3 className="text-2xl md:text-3xl font-bold text-slate-900 dark:text-white">Configura tu video antes de generar</h3>
                                <p className="text-sm text-slate-500 dark:text-slate-400 mt-2">
                                    Fuente: <span className="font-semibold text-slate-900 dark:text-white">{mode === 'file' ? (file?.name || 'Archivo local') : 'URL de YouTube'}</span>
                                </p>
                            </div>
                            <button
                                type="button"
                                onClick={() => setShowConfigModal(false)}
                                className="p-2 rounded-full text-slate-400 hover:text-slate-700 dark:hover:text-white hover:bg-slate-200/80 dark:hover:bg-slate-800 transition-colors"
                            >
                                <X size={20} />
                            </button>
                        </div>

                        <div className="space-y-8">
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                <div className="space-y-2">
                                    <label className={modalLabelClass}>Formato de salida</label>
                                    <select
                                        value={aspectRatio}
                                        onChange={(e) => setAspectRatio(e.target.value)}
                                        className={modalInputClass}
                                    >
                                        <option value="9:16">9:16 (Vertical Shorts/Reels/TikTok)</option>
                                        <option value="16:9">16:9 (Horizontal YouTube/Presentación)</option>
                                    </select>
                                </div>
                                <div className="space-y-2">
                                    <label className={modalLabelClass}>Duración objetivo del clip</label>
                                    <select
                                        value={clipLengthTarget}
                                        onChange={(e) => setClipLengthTarget(e.target.value)}
                                        className={modalInputClass}
                                    >
                                        <option value="balanced">30-45s (equilibrado)</option>
                                        <option value="short">&lt;30s (hook rápido)</option>
                                        <option value="long">45-60s (contexto)</option>
                                    </select>
                                </div>
                            </div>

                            <div className={modalCardClass}>
                                <div className="flex items-center justify-between gap-2 mb-4">
                                    <div>
                                        <h4 className="text-base font-bold text-slate-900 dark:text-white">Tipo de contenido</h4>
                                        <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">Preajuste rápido para adaptar el motor al formato del video.</p>
                                    </div>
                                    <span className="px-3 py-1 bg-violet-100 dark:bg-violet-900/30 text-violet-700 dark:text-violet-300 text-[10px] font-bold tracking-wider uppercase rounded-full">Perfil</span>
                                </div>
                                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                                    {CONTENT_PRESETS.map((preset) => {
                                        const selected = selectedContentPreset === preset.id;
                                        return (
                                            <button
                                                key={preset.id}
                                                type="button"
                                                onClick={() => applyContentPreset(preset.id)}
                                                className={`relative rounded-xl p-3 text-left transition-all ${
                                                    selected
                                                        ? 'border-2 border-primary bg-violet-50 dark:bg-violet-900/10 shadow-sm'
                                                        : 'border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:border-violet-300 dark:hover:border-violet-700 hover:shadow-md'
                                                }`}
                                            >
                                                <h5 className={`font-bold text-sm ${selected ? 'text-primary' : 'text-slate-800 dark:text-slate-200'}`}>{preset.name}</h5>
                                                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">{preset.subtitle}</p>
                                                {selected && (
                                                    <div className="absolute top-3 right-3 text-primary">
                                                        <CheckCircle2 size={16} />
                                                    </div>
                                                )}
                                            </button>
                                        );
                                    })}
                                </div>
                            </div>

                            <div className={modalCardClass}>
                                <div className="flex items-center justify-between gap-2 mb-5">
                                    <div>
                                        <h4 className="text-base font-bold text-slate-900 dark:text-white">Plantillas rápidas</h4>
                                        <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">Un clic para cargar configuración completa del proyecto.</p>
                                    </div>
                                    <span className="px-3 py-1 bg-violet-100 dark:bg-violet-900/30 text-violet-700 dark:text-violet-300 text-[10px] font-bold tracking-wider uppercase rounded-full">Preajuste</span>
                                </div>
                                <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
                                    {TEMPLATE_PRESETS.map((preset) => {
                                        const selected = selectedTemplate === preset.id;
                                        return (
                                            <button
                                                key={preset.id}
                                                type="button"
                                                onClick={() => applyTemplate(preset.id)}
                                                className="group text-left"
                                            >
                                                <div className={`relative aspect-[9/16] rounded-xl overflow-hidden transition-all ${
                                                    selected
                                                        ? 'border-[3px] border-primary shadow-lg ring-4 ring-primary/10'
                                                        : 'border border-slate-200 dark:border-slate-700 group-hover:border-primary group-hover:shadow-lg'
                                                }`}>
                                                    <div className={`absolute inset-0 bg-gradient-to-br ${preset.gradient} opacity-90`} />
                                                    <div className="absolute top-2 left-2 bg-black/50 backdrop-blur text-white text-[10px] font-bold px-1.5 py-0.5 rounded">1.00</div>
                                                    <div className="absolute bottom-6 left-2 right-2">
                                                        <div className="bg-black/70 backdrop-blur-sm text-white text-[10px] p-1.5 text-center rounded-lg">Aquí va tu subtítulo</div>
                                                    </div>
                                                    {selected && (
                                                        <div className="absolute top-2 right-2 bg-primary text-white rounded-full w-5 h-5 flex items-center justify-center shadow-md">
                                                            <CheckCircle2 size={14} />
                                                        </div>
                                                    )}
                                                </div>
                                                <div className="mt-2 px-1">
                                                    <h5 className={`font-bold text-sm ${selected ? 'text-primary' : 'text-slate-900 dark:text-white group-hover:text-primary'}`}>{preset.name}</h5>
                                                    <p className="text-[11px] text-slate-500 dark:text-slate-400 leading-tight truncate">{preset.subtitle}</p>
                                                </div>
                                            </button>
                                        );
                                    })}
                                </div>
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                <div className="space-y-2">
                                    <label className={modalLabelClass}>Idioma del video</label>
                                    <select
                                        value={language}
                                        onChange={(e) => setLanguage(e.target.value)}
                                        className={modalInputClass}
                                    >
                                        <option value="auto">Detectar automáticamente</option>
                                        <option value="es">Español</option>
                                        <option value="en">Inglés</option>
                                        <option value="pt">Portugués</option>
                                        <option value="fr">Francés</option>
                                        <option value="de">Alemán</option>
                                        <option value="it">Italiano</option>
                                    </select>
                                </div>
                                <div className="space-y-2">
                                    <label className={modalLabelClass}>Número de clips</label>
                                    <input
                                        type="number"
                                        min="1"
                                        max="15"
                                        value={clipCount}
                                        onChange={(e) => setClipCount(Number(e.target.value || 1))}
                                        className={modalInputClass}
                                    />
                                </div>
                            </div>

                            <div className={modalCardClass}>
                                <div className="flex items-center justify-between gap-2 mb-4">
                                    <div>
                                        <h4 className="text-base font-bold text-slate-900 dark:text-white">Whisper opciones</h4>
                                        <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">Preajustes para velocidad y precisión de transcripción.</p>
                                    </div>
                                    <span className="px-3 py-1 bg-violet-100 dark:bg-violet-900/30 text-violet-700 dark:text-violet-300 text-[10px] font-bold tracking-wider uppercase rounded-full">Preajuste</span>
                                </div>

                                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-4">
                                    {WHISPER_OPTION_PRESETS.map((preset) => {
                                        const selected = selectedWhisperOption === preset.id;
                                        return (
                                            <button
                                                key={preset.id}
                                                type="button"
                                                onClick={() => applyWhisperOptionPreset(preset.id)}
                                                className={`rounded-xl p-3 text-left transition-all ${
                                                    selected
                                                        ? 'border-2 border-primary bg-violet-50 dark:bg-violet-900/10 shadow-sm'
                                                        : 'border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:border-violet-300 dark:hover:border-violet-700 hover:shadow-md'
                                                }`}
                                            >
                                                <p className={`text-sm font-bold ${selected ? 'text-primary' : 'text-slate-800 dark:text-slate-200'}`}>{preset.name}</p>
                                                <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-1">{preset.subtitle}</p>
                                            </button>
                                        );
                                    })}
                                </div>
                                {selectedWhisperOption === 'custom' && (
                                    <p className="mb-4 text-[11px] text-slate-500 dark:text-slate-400">Modo personalizado activo (ajustes manuales).</p>
                                )}

                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    <div className="space-y-2">
                                        <label className={modalLabelClass}>Whisper backend</label>
                                        <select
                                            value={whisperBackend}
                                            onChange={(e) => {
                                                const nextBackend = e.target.value;
                                                setWhisperBackend(nextBackend);
                                                setWhisperModel(normalizeWhisperModelForBackend(nextBackend, whisperModel));
                                                setSelectedWhisperOption('custom');
                                            }}
                                            className={modalInputClass}
                                        >
                                            <option value="openai">openai-whisper (más estable)</option>
                                            <option value="faster">faster-whisper (recomendado Colab)</option>
                                        </select>
                                    </div>
                                    <div className="space-y-2">
                                        <label className={modalLabelClass}>Modelo Whisper</label>
                                        <select
                                            value={whisperModel}
                                            onChange={(e) => {
                                                setWhisperModel(e.target.value);
                                                setSelectedWhisperOption('custom');
                                            }}
                                            className={modalInputClass}
                                        >
                                            {whisperModelOptions.map((modelOpt) => (
                                                <option key={modelOpt.value} value={modelOpt.value}>
                                                    {modelOpt.label}
                                                </option>
                                            ))}
                                        </select>
                                    </div>
                                    <div className="space-y-2">
                                        <label className={modalLabelClass}>Subtítulos precisos (marcas por palabra)</label>
                                        <select
                                            value={wordTimestamps ? 'yes' : 'no'}
                                            onChange={(e) => {
                                                setWordTimestamps(e.target.value === 'yes');
                                                setSelectedWhisperOption('custom');
                                            }}
                                            className={modalInputClass}
                                        >
                                            <option value="yes">Sí (más lento)</option>
                                            <option value="no">No (más rápido)</option>
                                        </select>
                                    </div>
                                    <div className="space-y-2">
                                        <label className={modalLabelClass}>FFmpeg preset</label>
                                        <select
                                            value={ffmpegPreset}
                                            onChange={(e) => {
                                                setFfmpegPreset(e.target.value);
                                                setSelectedWhisperOption('custom');
                                            }}
                                            className={modalInputClass}
                                        >
                                            <option value="ultrafast">ultrafast (más rápido)</option>
                                            <option value="fast">fast</option>
                                            <option value="medium">medium (mejor calidad)</option>
                                        </select>
                                    </div>
                                </div>

                                <div className="mt-5 space-y-2">
                                    <label className={modalLabelClass}>Calidad de video (CRF)</label>
                                    <input
                                        type="number"
                                        min="18"
                                        max="30"
                                        value={ffmpegCrf}
                                        onChange={(e) => {
                                            setFfmpegCrf(Number(e.target.value || 23));
                                            setSelectedWhisperOption('custom');
                                        }}
                                        className={modalInputClass}
                                    />
                                </div>
                            </div>

                            <div className="flex flex-col-reverse sm:flex-row justify-end items-center gap-3 pt-2 pb-1">
                                <button
                                    type="button"
                                    onClick={() => setShowConfigModal(false)}
                                    className="w-full sm:w-auto px-8 py-3 rounded-xl border border-slate-300 dark:border-slate-600 text-slate-700 dark:text-white font-medium hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
                                >
                                    Cancelar
                                </button>
                                <button
                                    type="button"
                                    onClick={handleGenerate}
                                    disabled={isProcessing || !canConfigure}
                                    className="w-full sm:w-auto btn-primary flex items-center justify-center gap-2 px-10 py-3 disabled:opacity-50 disabled:cursor-not-allowed"
                                >
                                    {isProcessing ? (
                                        <>
                                            <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                            Procesando video...
                                        </>
                                    ) : (
                                        <>Generar clips</>
                                    )}
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </>
    );
}
