import React, { useEffect, useState } from 'react';
import { Youtube, Upload, FileVideo, X, CheckCircle2, Settings2, Clapperboard, Zap, Bot, SlidersHorizontal, Smartphone, Monitor } from 'lucide-react';
import { createPortal } from 'react-dom';

const MEDIA_INPUT_STORAGE_KEY = 'mediaInputPresetV2';

const ALLOWED_VALUES = {
    language: ['es', 'en', 'fr', 'de', 'it', 'pt', 'auto'],
    whisperBackend: ['openai', 'faster', 'whisperx'],
    whisperModel: ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
    ffmpegPreset: ['ultrafast', 'fast', 'medium'],
    aspectRatio: ['9:16', '16:9'],
    clipLengthTarget: ['short', 'balanced', 'long'],
    generationMode: ['clips', 'trailer'],
    llmModel: [
        'gemini-2.5-flash-lite', 'gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-1.5-flash'
    ]
};

const LLM_MODEL_OPTIONS = {
    gemini: [
        { value: 'gemini-2.5-flash-lite', label: 'gemini-2.5-flash-lite | Gemini 2.5 Flash-Lite' },
        { value: 'gemini-2.5-flash', label: 'gemini-2.5-flash | Gemini 2.5 Flash' },
        { value: 'gemini-2.0-flash', label: 'gemini-2.0-flash | Gemini 2.0 Flash' },
        { value: 'gemini-1.5-flash', label: 'gemini-1.5-flash | Gemini 1.5 Flash' }
    ]
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
        { value: 'large-v3', label: 'large-v3 (recomendado Colab / Español)' }
    ],
    whisperx: [
        { value: 'tiny', label: 'tiny (muy rápido)' },
        { value: 'base', label: 'base (equilibrado)' },
        { value: 'small', label: 'small (mejor precisión)' },
        { value: 'medium', label: 'medium (alta precisión)' },
        { value: 'large-v2', label: 'large-v2 (muy alta precisión)' },
        { value: 'large-v3', label: 'large-v3 (WhisperX PRO)' }
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
            whisperBackend: 'faster',
            whisperModel: 'large-v3',
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
            whisperBackend: 'faster',
            whisperModel: 'large-v3',
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
            whisperBackend: 'faster',
            whisperModel: 'large-v3',
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
            whisperBackend: 'faster',
            whisperModel: 'large-v3',
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
            whisperBackend: 'faster',
            whisperModel: 'large-v3',
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
            whisperBackend: 'faster',
            whisperModel: 'large-v3',
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
            wordTimestamps: true,
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
            whisperBackend: 'faster',
            whisperModel: 'large-v3',
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
            whisperBackend: 'faster',
            whisperModel: 'large-v3',
            wordTimestamps: true,
            ffmpegPreset: 'medium',
            ffmpegCrf: 21
        }
    }
];

const WHISPER_OPTION_PRESETS = [
    {
        id: 'faster_balanced',
        name: 'Faster Balanced (Default)',
        subtitle: 'faster + large-v3 + timestamps',
        settings: {
            whisperBackend: 'faster',
            whisperModel: 'large-v3',
            wordTimestamps: true,
            ffmpegPreset: 'fast',
            ffmpegCrf: 23
        }
    },
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
        subtitle: 'faster + tiny + transcripción',
        settings: {
            whisperBackend: 'faster',
            whisperModel: 'tiny',
            wordTimestamps: true,
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
    },
    {
        id: 'whisperx_pro',
        name: 'WhisperX Pro',
        subtitle: 'whisperx + large-v3 + Diarización',
        settings: {
            whisperBackend: 'whisperx',
            whisperModel: 'large-v3',
            wordTimestamps: true,
            ffmpegPreset: 'fast',
            ffmpegCrf: 22
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
            language: pickAllowed(parsed.language, ALLOWED_VALUES.language, 'es'),
            clipCount: clampNumber(parsed.clipCount, 6, 1, 15),
            trailerFragmentsTarget: clampNumber(parsed.trailerFragmentsTarget, 6, 2, 12),
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
                : 'faster_balanced',
            generationMode: pickAllowed(parsed.generationMode, ALLOWED_VALUES.generationMode, 'clips'),
            llmModel: pickAllowed(parsed.llmModel, ALLOWED_VALUES.llmModel, 'gemini-2.5-flash-lite')
        };
    } catch (error) {
        console.warn('Failed to load media input config:', error);
        return null;
    }
}

export default function MediaInput({
    onProcess,
    isProcessing,
    apiKey = '',
    prefillFile = null
}) {
    const [initialConfig] = useState(() => loadStoredMediaInputConfig());
    const [mode, setMode] = useState('file'); // 'url' | 'file'
    const [url, setUrl] = useState('');
    const [file, setFile] = useState(null);
    const [language, setLanguage] = useState(initialConfig?.language ?? 'es');
    const [clipCount, setClipCount] = useState(initialConfig?.clipCount ?? 6);
    const [trailerFragmentsTarget, setTrailerFragmentsTarget] = useState(initialConfig?.trailerFragmentsTarget ?? 6);
    const [whisperBackend, setWhisperBackend] = useState(initialConfig?.whisperBackend ?? 'faster');
    const [whisperModel, setWhisperModel] = useState(initialConfig?.whisperModel ?? 'large-v3');
    const [wordTimestamps, setWordTimestamps] = useState(initialConfig?.wordTimestamps ?? true);
    const [enableDiarization, setEnableDiarization] = useState(initialConfig?.enableDiarization ?? false);
    const [ffmpegPreset, setFfmpegPreset] = useState(initialConfig?.ffmpegPreset ?? 'fast');
    const [ffmpegCrf, setFfmpegCrf] = useState(initialConfig?.ffmpegCrf ?? 23);
    const [aspectRatio, setAspectRatio] = useState(initialConfig?.aspectRatio ?? '9:16');
    const [clipLengthTarget, setClipLengthTarget] = useState(initialConfig?.clipLengthTarget ?? 'balanced');
    const [selectedTemplate, setSelectedTemplate] = useState(initialConfig?.selectedTemplate ?? 'default');
    const [selectedContentPreset, setSelectedContentPreset] = useState(initialConfig?.selectedContentPreset ?? 'general');
    const [selectedWhisperOption, setSelectedWhisperOption] = useState(initialConfig?.selectedWhisperOption ?? 'faster_balanced');
    const [generationMode, setGenerationMode] = useState(initialConfig?.generationMode ?? 'clips');
    const [llmModel, setLlmModel] = useState(initialConfig?.llmModel ?? 'gemini-2.5-flash-lite');
    const [showConfigModal, setShowConfigModal] = useState(false);
    const [configTab, setConfigTab] = useState('general');
    const whisperModelOptions = MODEL_OPTIONS_BY_BACKEND[whisperBackend] || MODEL_OPTIONS_BY_BACKEND.openai;

    useEffect(() => {
        if (typeof window === 'undefined') return;
        const payload = {
            language,
            clipCount,
            trailerFragmentsTarget,
            whisperBackend,
            whisperModel,
            wordTimestamps,
            ffmpegPreset,
            ffmpegCrf,
            aspectRatio,
            clipLengthTarget,
            selectedTemplate,
            selectedContentPreset,
            selectedWhisperOption,
            generationMode,
            enableDiarization,
            llmModel
        };
        window.localStorage.setItem(MEDIA_INPUT_STORAGE_KEY, JSON.stringify(payload));
    }, [
        language,
        clipCount,
        trailerFragmentsTarget,
        whisperBackend,
        whisperModel,
        wordTimestamps,
        enableDiarization,
        ffmpegPreset,
        ffmpegCrf,
        aspectRatio,
        clipLengthTarget,
        selectedTemplate,
        selectedContentPreset,
        selectedWhisperOption,
        generationMode,
        llmModel
    ]);

    const applySettings = (settings) => {
        if (!settings || typeof settings !== 'object') return;
        const nextBackend = settings.whisperBackend || whisperBackend;
        const nextModel = normalizeWhisperModelForBackend(nextBackend, settings.whisperModel || whisperModel);
        if (settings.aspectRatio) setAspectRatio(settings.aspectRatio);
        if (settings.clipLengthTarget) setClipLengthTarget(settings.clipLengthTarget);
        if (settings.llmModel) setLlmModel(settings.llmModel);
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

    useEffect(() => {
        if (showConfigModal) setConfigTab('general');
    }, [showConfigModal]);

    useEffect(() => {
        const injectedFile = prefillFile?.file;
        if (!injectedFile) return;
        const type = String(injectedFile.type || '').toLowerCase();
        const name = String(injectedFile.name || '').toLowerCase();
        const looksLikeMedia = type.startsWith('video/')
            || type.startsWith('audio/')
            || /\.(mp4|mov|m4a|mp3|wav|mkv|webm)$/i.test(name);
        if (!looksLikeMedia) return;
        setMode('file');
        setFile(injectedFile);
        setShowConfigModal(true);
    }, [prefillFile]);

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
            trailer_fragments_target: trailerFragmentsTarget,
            whisperBackend,
            whisperModel,
            wordTimestamps,
            enableDiarization: whisperBackend === 'whisperx' ? enableDiarization : false,
            ffmpegPreset,
            ffmpegCrf,
            aspectRatio,
            clipLengthTarget,
            styleTemplate: selectedTemplate,
            contentPreset: selectedContentPreset,
            llm_provider: 'gemini',
            llm_model: llmModel,
            generation_mode: generationMode,
            build_trailer: generationMode === 'trailer'
        };

        const headers = {};
        if (apiKey?.trim()) headers['X-Gemini-Key'] = apiKey.trim();

        if (mode === 'url' && url) {
            onProcess({
                type: 'url',
                payload: url,
                ...payloadBase
            }, headers);
        } else if (mode === 'file' && file) {
            onProcess({
                type: 'file',
                payload: file,
                ...payloadBase
            }, headers);
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
            <div className="w-full max-w-3xl mx-auto bg-white/95 dark:bg-surface border border-slate-100 dark:border-white/10 rounded-2xl p-4 md:p-5 shadow-[0_18px_50px_rgba(139,92,246,0.14)] dark:shadow-none animate-[fadeIn_0.6s_ease-out]">
                <div className="flex gap-6 px-2 mb-5 border-b border-slate-200 dark:border-white/10">
                    <button
                        onClick={() => setMode('url')}
                        className={`flex items-center gap-1.5 pb-3 px-0.5 text-sm font-medium transition-all border-b-2 ${mode === 'url'
                            ? 'text-primary border-primary'
                            : 'text-slate-500 dark:text-zinc-400 border-transparent hover:text-slate-900 dark:hover:text-white hover:border-slate-300 dark:hover:border-slate-600'
                            }`}
                    >
                        <Youtube size={14} />
                        URL de YouTube
                    </button>
                    <button
                        onClick={() => setMode('file')}
                        className={`flex items-center gap-1.5 pb-3 px-0.5 text-sm font-medium transition-all border-b-2 ${mode === 'file'
                            ? 'text-primary border-primary'
                            : 'text-slate-500 dark:text-zinc-400 border-transparent hover:text-slate-900 dark:hover:text-white hover:border-slate-300 dark:hover:border-slate-600'
                            }`}
                    >
                        <Upload size={14} />
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
                        className={`border-2 border-dashed rounded-xl p-10 md:p-12 text-center transition-all ${file ? 'border-primary/50 bg-primary/5' : 'border-violet-200 dark:border-zinc-700 hover:border-primary/40 bg-slate-50/60 dark:bg-white/5'}`}
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
                            <label className="cursor-pointer block group">
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
                                <div className="mx-auto mb-4 w-10 h-10 rounded-full bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 shadow-sm flex items-center justify-center text-primary group-hover:scale-105 transition-transform">
                                    <Upload size={18} />
                                </div>
                                <p className="text-slate-800 dark:text-slate-200 font-medium">Haz clic para subir o arrastra y suelta</p>
                                <p className="text-xs text-slate-500 dark:text-zinc-500 mt-1">MP4, MOV, MP3, WAV, M4A hasta 500MB</p>
                            </label>
                        )}
                    </div>
                )}

                <button
                    type="button"
                    disabled={isProcessing || !canConfigure}
                    onClick={() => setShowConfigModal(true)}
                    className="w-full mt-5 rounded-full bg-gradient-to-r from-[#ba9df8] via-[#aa8cf5] to-[#946ff1] hover:brightness-95 !text-white font-medium py-3.5 px-6 shadow-[0_10px_24px_rgba(139,92,246,0.35)] transition-all duration-200 flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    <Settings2 size={16} />
                    {mode === 'file' ? 'Configurar y generar' : 'Continuar a configuración'}
                </button>
            </div>

            {showConfigModal && createPortal(
                <div className="fixed inset-0 z-[110] p-4 md:p-6 overflow-y-auto flex items-center justify-center">
                    {/* Simple Blurred Overlay */}
                    <div className="fixed inset-0 bg-slate-900/40 backdrop-blur-sm" />

                    {/* Modal Container with Entrance Animation */}
                    <div className="relative w-full max-w-4xl rounded-3xl border border-white/20 bg-white/90 dark:bg-slate-900/90 backdrop-blur-2xl shadow-[0_32px_120px_rgba(0,0,0,0.5)] overflow-hidden flex flex-col max-h-[90vh] animate-modal-in">
                        <div className="px-6 md:px-8 py-6 border-b border-white/10 bg-white/40 dark:bg-slate-800/40 backdrop-blur-md">
                            <div className="flex items-start justify-between gap-4">
                                <div>
                                    <h3 className="text-2xl font-bold text-slate-900 dark:text-white">Configura tu video</h3>
                                    <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">Define los parámetros antes de generar el contenido</p>
                                </div>
                                <button
                                    type="button"
                                    onClick={() => setShowConfigModal(false)}
                                    className="p-2 rounded-full text-slate-400 hover:text-slate-700 dark:hover:text-white hover:bg-slate-200/80 dark:hover:bg-slate-700 transition-colors"
                                    title="Cerrar"
                                >
                                    <X size={18} />
                                </button>
                            </div>
                        </div>

                        <div className="px-6 md:px-8 border-b border-white/10 bg-white/20 dark:bg-slate-900/20">
                            <div className="flex items-center gap-1 overflow-x-auto">
                                <button
                                    type="button"
                                    onClick={() => setConfigTab('general')}
                                    className={`inline-flex items-center gap-2 whitespace-nowrap px-4 py-3 text-sm font-semibold border-b-2 transition-colors ${configTab === 'general'
                                        ? 'border-primary text-primary bg-primary/5'
                                        : 'border-transparent text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200'
                                        }`}
                                >
                                    <Settings2 size={15} />
                                    General
                                </button>
                                <button
                                    type="button"
                                    onClick={() => setConfigTab('content')}
                                    className={`inline-flex items-center gap-2 whitespace-nowrap px-4 py-3 text-sm font-semibold border-b-2 transition-colors ${configTab === 'content'
                                        ? 'border-primary text-primary bg-primary/5'
                                        : 'border-transparent text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200'
                                        }`}
                                >
                                    <Bot size={15} />
                                    Contenido e IA
                                </button>
                                <button
                                    type="button"
                                    onClick={() => setConfigTab('advanced')}
                                    className={`inline-flex items-center gap-2 whitespace-nowrap px-4 py-3 text-sm font-semibold border-b-2 transition-colors ${configTab === 'advanced'
                                        ? 'border-primary text-primary bg-primary/5'
                                        : 'border-transparent text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200'
                                        }`}
                                >
                                    <SlidersHorizontal size={15} />
                                    Avanzado
                                </button>
                            </div>
                        </div>

                        <div className="flex-1 overflow-y-auto custom-scrollbar px-6 md:px-8 py-6 space-y-6">
                            {configTab === 'general' && (
                                <div className="space-y-6">
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                        <button
                                            type="button"
                                            onClick={() => setGenerationMode('clips')}
                                            className={`relative rounded-xl border p-4 text-left transition-all ${generationMode === 'clips'
                                                ? 'border-primary bg-primary/10 ring-1 ring-primary/20'
                                                : 'border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:bg-slate-50 dark:hover:bg-slate-700/60'
                                                }`}
                                        >
                                            <div className="flex items-start gap-3">
                                                <div className={`w-9 h-9 rounded-full flex items-center justify-center ${generationMode === 'clips' ? 'bg-primary/15 text-primary' : 'bg-slate-100 dark:bg-slate-700 text-slate-500 dark:text-slate-300'}`}>
                                                    <Clapperboard size={16} />
                                                </div>
                                                <div className="min-w-0">
                                                    <h4 className="font-bold text-slate-900 dark:text-white flex items-center gap-2">
                                                        Clips Virales
                                                    </h4>
                                                    <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">Detecta y extrae automáticamente los momentos más atractivos para TikTok/Reels.</p>
                                                </div>
                                            </div>
                                            {generationMode === 'clips' && (
                                                <div className="absolute top-3 right-3 text-primary">
                                                    <CheckCircle2 size={16} />
                                                </div>
                                            )}
                                        </button>

                                        <button
                                            type="button"
                                            onClick={() => setGenerationMode('trailer')}
                                            className={`relative rounded-xl border p-4 text-left transition-all ${generationMode === 'trailer'
                                                ? 'border-primary bg-primary/10 ring-1 ring-primary/20'
                                                : 'border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 hover:bg-slate-50 dark:hover:bg-slate-700/60'
                                                }`}
                                        >
                                            <div className="flex items-start gap-3">
                                                <div className={`w-9 h-9 rounded-full flex items-center justify-center ${generationMode === 'trailer' ? 'bg-primary/15 text-primary' : 'bg-slate-100 dark:bg-slate-700 text-slate-500 dark:text-slate-300'}`}>
                                                    <Zap size={16} />
                                                </div>
                                                <div>
                                                    <h4 className="font-bold text-slate-900 dark:text-white flex items-center gap-2">
                                                        Super Trailer
                                                    </h4>
                                                    <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">Condensa todo el video en un trailer dinámico de alta retención.</p>
                                                </div>
                                            </div>
                                        </button>
                                    </div>

                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                        <div>
                                            <label className={`${modalLabelClass} mb-3`}>Formato de salida</label>
                                            <div className="grid grid-cols-2 gap-2">
                                                <button
                                                    type="button"
                                                    onClick={() => setAspectRatio('9:16')}
                                                    className={`inline-flex items-center justify-center gap-2 rounded-lg border px-3 py-2.5 text-sm font-medium transition-colors ${aspectRatio === '9:16'
                                                        ? 'border-primary bg-primary/10 text-primary'
                                                        : 'border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 bg-white dark:bg-slate-800'
                                                        }`}
                                                >
                                                    <Smartphone size={15} />
                                                    9:16 Vertical
                                                </button>
                                                <button
                                                    type="button"
                                                    onClick={() => setAspectRatio('16:9')}
                                                    className={`inline-flex items-center justify-center gap-2 rounded-lg border px-3 py-2.5 text-sm font-medium transition-colors ${aspectRatio === '16:9'
                                                        ? 'border-primary bg-primary/10 text-primary'
                                                        : 'border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 bg-white dark:bg-slate-800'
                                                        }`}
                                                >
                                                    <Monitor size={15} />
                                                    16:9 Horizontal
                                                </button>
                                            </div>
                                        </div>

                                        <div>
                                            <label className={`${modalLabelClass} mb-3`}>Duración objetivo</label>
                                            <div className="flex rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-100 dark:bg-slate-800 p-1">
                                                {[
                                                    { key: 'short', label: 'Corto' },
                                                    { key: 'balanced', label: 'Equilibrado' },
                                                    { key: 'long', label: 'Largo' }
                                                ].map((item) => (
                                                    <button
                                                        key={item.key}
                                                        type="button"
                                                        onClick={() => setClipLengthTarget(item.key)}
                                                        className={`flex-1 px-2 py-1.5 text-sm rounded-md transition-colors ${clipLengthTarget === item.key
                                                            ? 'bg-white dark:bg-slate-700 text-primary shadow-sm border border-slate-200 dark:border-slate-600 font-semibold'
                                                            : 'text-slate-500 dark:text-slate-400'
                                                            }`}
                                                    >
                                                        {item.label}
                                                    </button>
                                                ))}
                                            </div>
                                            <p className="text-[11px] text-slate-500 mt-2">Generará clips entre 30 y 60 segundos.</p>
                                        </div>
                                    </div>

                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                        <div className="space-y-2">
                                            <label className={modalLabelClass}>Número de clips</label>
                                            <input
                                                type="number"
                                                min="1"
                                                max="15"
                                                value={clipCount}
                                                onChange={(e) => setClipCount(Number(e.target.value || 1))}
                                                disabled={generationMode === 'trailer'}
                                                className={modalInputClass}
                                            />
                                            {generationMode === 'trailer' && (
                                                <p className="text-[11px] text-amber-600 dark:text-amber-400">En Super Trailer no se usa este valor.</p>
                                            )}
                                        </div>
                                        <div className="space-y-2">
                                            <label className={modalLabelClass}>Segmentos destacados (Super Trailer)</label>
                                            <input
                                                type="number"
                                                min="2"
                                                max="12"
                                                value={trailerFragmentsTarget}
                                                onChange={(e) => {
                                                    const nextValue = Number(e.target.value || 6);
                                                    setTrailerFragmentsTarget(Math.max(2, Math.min(12, Math.round(nextValue))));
                                                }}
                                                disabled={generationMode !== 'trailer'}
                                                className={modalInputClass}
                                            />
                                            <p className="text-[11px] text-slate-500">Cantidad de momentos incluidos en el trailer.</p>
                                        </div>
                                    </div>
                                </div>
                            )}

                            {configTab === 'content' && (
                                <div className="space-y-6">
                                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                        <div className="space-y-2">
                                            <label className={modalLabelClass}>Modelo (Gemini)</label>
                                            <select
                                                value={llmModel}
                                                onChange={(e) => setLlmModel(e.target.value)}
                                                className={modalInputClass}
                                            >
                                                {LLM_MODEL_OPTIONS.gemini.map((opt) => (
                                                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                                                ))}
                                            </select>
                                        </div>
                                        <div className="space-y-2">
                                            <label className={modalLabelClass}>Idioma del video</label>
                                            <select
                                                value={language}
                                                onChange={(e) => setLanguage(e.target.value)}
                                                className={modalInputClass}
                                            >
                                                <option value="es">Español</option>
                                                <option value="en">Inglés</option>
                                                <option value="fr">Francés</option>
                                                <option value="de">Alemán</option>
                                                <option value="it">Italiano</option>
                                                <option value="pt">Portugués</option>
                                                <option value="auto">Detectar automáticamente</option>
                                            </select>
                                        </div>
                                    </div>

                                    <div className={modalCardClass}>
                                        <div className="flex items-center justify-between gap-2 mb-4">
                                            <div>
                                                <h4 className="text-base font-bold text-slate-900 dark:text-white">Tipo de contenido</h4>
                                                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">Preajuste para adaptar el motor al formato del video.</p>
                                            </div>
                                            <span className="px-3 py-1 bg-violet-100 dark:bg-violet-900/30 text-violet-700 dark:text-violet-300 text-[10px] font-bold tracking-wider uppercase rounded-full">Perfil</span>
                                        </div>
                                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                                            {CONTENT_PRESETS.map((preset) => {
                                                const selected = selectedContentPreset === preset.id;
                                                return (
                                                    <button
                                                        key={preset.id}
                                                        type="button"
                                                        onClick={() => applyContentPreset(preset.id)}
                                                        className={`relative rounded-xl p-3 text-left transition-all ${selected
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
                                        <div className="flex items-center justify-between gap-2 mb-4">
                                            <div>
                                                <h4 className="text-base font-bold text-slate-900 dark:text-white">Plantillas rápidas</h4>
                                                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">Carga una configuración completa con un clic.</p>
                                            </div>
                                        </div>
                                        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
                                            {TEMPLATE_PRESETS.map((preset) => {
                                                const selected = selectedTemplate === preset.id;
                                                return (
                                                    <button
                                                        key={preset.id}
                                                        type="button"
                                                        onClick={() => applyTemplate(preset.id)}
                                                        className="group text-left"
                                                    >
                                                        <div className={`relative aspect-[9/16] rounded-xl overflow-hidden transition-all ${selected
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
                                </div>
                            )}

                            {configTab === 'advanced' && (
                                <div className="space-y-6">
                                    <div className={modalCardClass}>
                                        <div className="flex items-center justify-between gap-2 mb-4">
                                            <div>
                                                <h4 className="text-base font-bold text-slate-900 dark:text-white">Whisper opciones</h4>
                                                <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">Preajustes para velocidad y precisión de transcripción.</p>
                                            </div>
                                            <span className="px-3 py-1 bg-violet-100 dark:bg-violet-900/30 text-violet-700 dark:text-violet-300 text-[10px] font-bold tracking-wider uppercase rounded-full">Preajuste</span>
                                        </div>

                                        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 mb-4">
                                            {WHISPER_OPTION_PRESETS.map((preset) => {
                                                const selected = selectedWhisperOption === preset.id;
                                                return (
                                                    <button
                                                        key={preset.id}
                                                        type="button"
                                                        onClick={() => applyWhisperOptionPreset(preset.id)}
                                                        className={`rounded-xl p-3 text-left transition-all ${selected
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
                                            <p className="mb-4 text-[11px] text-slate-500 dark:text-slate-400">Modo personalizado activo.</p>
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
                                                    <option value="faster">faster-whisper (predeterminado)</option>
                                                    <option value="openai">openai-whisper (compatibilidad)</option>
                                                    <option value="whisperx">whisperx (Speaker Diarization)</option>
                                                </select>

                                                {/* Diarization opt-in — only visible when whisperx is selected */}
                                                {whisperBackend === 'whisperx' && (
                                                    <div className="mt-3 rounded-xl border border-amber-300 dark:border-amber-700/60 bg-amber-50 dark:bg-amber-900/20 px-4 py-3 flex items-start gap-3">
                                                        <input
                                                            id="enable-diarization"
                                                            type="checkbox"
                                                            checked={enableDiarization}
                                                            onChange={(e) => setEnableDiarization(e.target.checked)}
                                                            className="mt-0.5 h-4 w-4 rounded accent-amber-500 cursor-pointer"
                                                        />
                                                        <label htmlFor="enable-diarization" className="cursor-pointer">
                                                            <span className="block text-sm font-semibold text-amber-800 dark:text-amber-300">👥 Detectar hablantes (Diarización)</span>
                                                            <span className="block text-[11px] text-amber-700 dark:text-amber-400 mt-0.5">
                                                                Muy lento en CPU (~5–10 min). Actívalo solo si tienes múltiples personas y necesitas colores por hablante.
                                                            </span>
                                                        </label>
                                                    </div>
                                                )}
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
                                            <div className="space-y-2 md:col-span-2">
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
                                    </div>
                                </div>
                            )}
                        </div>

                        <div className="px-6 md:px-8 py-4 border-t border-slate-200 dark:border-slate-700 bg-white/90 dark:bg-slate-800/70 backdrop-blur-sm flex flex-col-reverse sm:flex-row justify-end items-center gap-3">
                            <button
                                type="button"
                                onClick={() => setShowConfigModal(false)}
                                className="w-full sm:w-auto px-6 py-2.5 rounded-full border border-slate-300 dark:border-slate-600 text-slate-700 dark:text-white font-medium hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
                            >
                                Cancelar
                            </button>
                            <button
                                type="button"
                                onClick={handleGenerate}
                                disabled={isProcessing || !canConfigure}
                                className="w-full sm:w-auto btn-primary flex items-center justify-center gap-2 px-8 py-2.5 disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                {isProcessing ? (
                                    <>
                                        <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                        Procesando video...
                                    </>
                                ) : (
                                    <>{generationMode === 'trailer' ? 'Generar Super Trailer' : 'Generar clips'}</>
                                )}
                            </button>
                        </div>
                    </div>
                </div>,
                document.body
            )}
        </>
    );
}
