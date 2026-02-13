import React, { useEffect, useState } from 'react';
import { X, Type, Loader2 } from 'lucide-react';

const BRAND_KIT_STORAGE_KEY = 'brandKitV1';
const DEFAULT_SUBTITLE_STYLE = {
    name: 'Predeterminado',
    subtitle_position: 'bottom',
    subtitle_font_size: 24,
    subtitle_font_family: 'Impact',
    subtitle_font_color: '#FFFFFF',
    subtitle_stroke_color: '#000000',
    subtitle_stroke_width: 3,
    subtitle_bold: true,
    subtitle_box_color: '#000000',
    subtitle_box_opacity: 60
};

const CAPTION_STYLE_PRESETS = [
    {
        id: 'bold_center',
        name: 'Negrita Centro',
        subtitle_position: 'middle',
        subtitle_font_size: 32,
        subtitle_font_family: 'Impact',
        subtitle_font_color: '#FFFFFF',
        subtitle_stroke_color: '#000000',
        subtitle_stroke_width: 4,
        subtitle_bold: true,
        subtitle_box_color: '#000000',
        subtitle_box_opacity: 45
    },
    {
        id: 'neon_pop',
        name: 'Neon Pop',
        subtitle_position: 'bottom',
        subtitle_font_size: 30,
        subtitle_font_family: 'Arial Black',
        subtitle_font_color: '#00F5FF',
        subtitle_stroke_color: '#091933',
        subtitle_stroke_width: 3,
        subtitle_bold: true,
        subtitle_box_color: '#0B1020',
        subtitle_box_opacity: 35
    },
    {
        id: 'typewriter',
        name: 'Máquina de escribir',
        subtitle_position: 'bottom',
        subtitle_font_size: 24,
        subtitle_font_family: 'Verdana',
        subtitle_font_color: '#F8FAFC',
        subtitle_stroke_color: '#111827',
        subtitle_stroke_width: 2,
        subtitle_bold: false,
        subtitle_box_color: '#111827',
        subtitle_box_opacity: 25
    },
    {
        id: 'bubble',
        name: 'Burbuja',
        subtitle_position: 'middle',
        subtitle_font_size: 30,
        subtitle_font_family: 'Arial Black',
        subtitle_font_color: '#111827',
        subtitle_stroke_color: '#FFFFFF',
        subtitle_stroke_width: 4,
        subtitle_bold: true,
        subtitle_box_color: '#FDE047',
        subtitle_box_opacity: 75
    },
    {
        id: 'minimal_clean',
        name: 'Minimal limpio',
        subtitle_position: 'bottom',
        subtitle_font_size: 22,
        subtitle_font_family: 'Arial',
        subtitle_font_color: '#FFFFFF',
        subtitle_stroke_color: '#000000',
        subtitle_stroke_width: 1,
        subtitle_bold: false,
        subtitle_box_color: '#000000',
        subtitle_box_opacity: 20
    }
];

const normalizeBrandKitStyle = (raw) => {
    const src = raw && typeof raw === 'object' ? raw : {};
    const asNum = (v, fallback, min, max) => {
        const n = Number(v);
        if (!Number.isFinite(n)) return fallback;
        return Math.max(min, Math.min(max, Math.round(n)));
    };
    return {
        name: String(src.name || DEFAULT_SUBTITLE_STYLE.name).slice(0, 48),
        subtitle_position: ['top', 'middle', 'bottom'].includes(src.subtitle_position) ? src.subtitle_position : DEFAULT_SUBTITLE_STYLE.subtitle_position,
        subtitle_font_size: asNum(src.subtitle_font_size, DEFAULT_SUBTITLE_STYLE.subtitle_font_size, 12, 84),
        subtitle_font_family: String(src.subtitle_font_family || DEFAULT_SUBTITLE_STYLE.subtitle_font_family).slice(0, 48),
        subtitle_font_color: String(src.subtitle_font_color || DEFAULT_SUBTITLE_STYLE.subtitle_font_color),
        subtitle_stroke_color: String(src.subtitle_stroke_color || DEFAULT_SUBTITLE_STYLE.subtitle_stroke_color),
        subtitle_stroke_width: asNum(src.subtitle_stroke_width, DEFAULT_SUBTITLE_STYLE.subtitle_stroke_width, 0, 8),
        subtitle_bold: typeof src.subtitle_bold === 'boolean' ? src.subtitle_bold : DEFAULT_SUBTITLE_STYLE.subtitle_bold,
        subtitle_box_color: String(src.subtitle_box_color || DEFAULT_SUBTITLE_STYLE.subtitle_box_color),
        subtitle_box_opacity: asNum(src.subtitle_box_opacity, DEFAULT_SUBTITLE_STYLE.subtitle_box_opacity, 0, 100)
    };
};

const loadBrandKitStyle = () => {
    try {
        const raw = localStorage.getItem(BRAND_KIT_STORAGE_KEY);
        if (!raw) return DEFAULT_SUBTITLE_STYLE;
        return normalizeBrandKitStyle(JSON.parse(raw));
    } catch (_) {
        return DEFAULT_SUBTITLE_STYLE;
    }
};

export default function SubtitleModal({ isOpen, onClose, onGenerate, isProcessing, videoUrl, aspectRatio = '9:16', onLoadSrt }) {
    const [position, setPosition] = useState(DEFAULT_SUBTITLE_STYLE.subtitle_position); // bottom, middle, top
    const [fontSize, setFontSize] = useState(DEFAULT_SUBTITLE_STYLE.subtitle_font_size);
    const [fontFamily, setFontFamily] = useState(DEFAULT_SUBTITLE_STYLE.subtitle_font_family);
    const [fontColor, setFontColor] = useState(DEFAULT_SUBTITLE_STYLE.subtitle_font_color);
    const [strokeColor, setStrokeColor] = useState(DEFAULT_SUBTITLE_STYLE.subtitle_stroke_color);
    const [strokeWidth, setStrokeWidth] = useState(DEFAULT_SUBTITLE_STYLE.subtitle_stroke_width);
    const [bold, setBold] = useState(DEFAULT_SUBTITLE_STYLE.subtitle_bold);
    const [boxColor, setBoxColor] = useState(DEFAULT_SUBTITLE_STYLE.subtitle_box_color);
    const [boxOpacity, setBoxOpacity] = useState(DEFAULT_SUBTITLE_STYLE.subtitle_box_opacity);
    const [brandKitName, setBrandKitName] = useState(DEFAULT_SUBTITLE_STYLE.name);
    const [srtText, setSrtText] = useState('');
    const [loadingSrt, setLoadingSrt] = useState(false);
    const isLandscape = aspectRatio === '16:9';

    const toRgba = (hex, opacity) => {
        const clean = (hex || '#000000').replace('#', '');
        const r = parseInt(clean.substring(0, 2), 16);
        const g = parseInt(clean.substring(2, 4), 16);
        const b = parseInt(clean.substring(4, 6), 16);
        return `rgba(${r}, ${g}, ${b}, ${opacity / 100})`;
    };

    const applyBrandKit = (kitRaw) => {
        const kit = normalizeBrandKitStyle(kitRaw);
        setPosition(kit.subtitle_position);
        setFontSize(kit.subtitle_font_size);
        setFontFamily(kit.subtitle_font_family);
        setFontColor(kit.subtitle_font_color);
        setStrokeColor(kit.subtitle_stroke_color);
        setStrokeWidth(kit.subtitle_stroke_width);
        setBold(kit.subtitle_bold);
        setBoxColor(kit.subtitle_box_color);
        setBoxOpacity(kit.subtitle_box_opacity);
        setBrandKitName(kit.name || 'Predeterminado');
    };

    const applyCaptionPreset = (presetId) => {
        const preset = CAPTION_STYLE_PRESETS.find((p) => p.id === presetId);
        if (!preset) return;
        applyBrandKit({
            name: preset.name,
            subtitle_position: preset.subtitle_position,
            subtitle_font_size: preset.subtitle_font_size,
            subtitle_font_family: preset.subtitle_font_family,
            subtitle_font_color: preset.subtitle_font_color,
            subtitle_stroke_color: preset.subtitle_stroke_color,
            subtitle_stroke_width: preset.subtitle_stroke_width,
            subtitle_bold: preset.subtitle_bold,
            subtitle_box_color: preset.subtitle_box_color,
            subtitle_box_opacity: preset.subtitle_box_opacity
        });
    };

    useEffect(() => {
        if (!isOpen) return;
        applyBrandKit(loadBrandKitStyle());
    }, [isOpen]);

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-[fadeIn_0.2s_ease-out]">
            <div className="bg-[#121214] border border-white/10 p-6 rounded-2xl w-full max-w-4xl shadow-2xl relative flex flex-col md:flex-row gap-6 max-h-[90vh]">
                <button
                    onClick={onClose}
                    className="absolute top-4 right-4 text-zinc-500 hover:text-white z-10"
                >
                    <X size={20} />
                </button>

                {/* Left: Preview */}
                <div
                    className={`flex-1 flex flex-col items-center justify-center bg-black rounded-lg border border-white/5 overflow-hidden relative w-full mx-auto max-h-[600px] ${isLandscape ? 'max-w-[640px]' : 'max-w-[360px]'}`}
                    style={{ aspectRatio: isLandscape ? '16 / 9' : '9 / 16' }}
                >
                     <video src={videoUrl} className="w-full h-full object-contain opacity-50" muted playsInline />
                     
                     {/* Subtitle Overlay Preview */}
                     <div className={`absolute w-full px-8 text-center transition-all duration-300 pointer-events-none flex flex-col items-center justify-center
                        ${position === 'top' ? 'top-20' : ''}
                        ${position === 'middle' ? 'top-0 bottom-0' : ''}
                        ${position === 'bottom' ? 'bottom-20' : ''}
                     `}>
                        <span 
                            className="bg-black/50 text-white font-bold px-2 py-1 rounded shadow-lg backdrop-blur-sm border border-white/10 text-center"
                            style={{ 
                                fontSize: `${Math.max(12, fontSize * 0.6)}px`,
                                fontFamily,
                                fontWeight: bold ? 700 : 400,
                                color: fontColor,
                                backgroundColor: boxOpacity > 0 ? toRgba(boxColor, boxOpacity) : 'transparent',
                                textShadow: `0 0 ${strokeWidth}px ${strokeColor}`,
                                maxWidth: '80%' 
                            }} 
                        >
                            Así se verán tus subtítulos<br/>en el video
                        </span>
                     </div>
                </div>

                {/* Right: Controls */}
                <div className="w-full md:w-80 flex flex-col">
                    <div className="flex items-start justify-between gap-3 mb-6">
                        <h3 className="text-xl font-bold text-white flex items-center gap-2">
                            <Type className="text-primary" /> Subtítulos automáticos
                        </h3>
                        <button
                            type="button"
                            onClick={() => applyBrandKit(loadBrandKitStyle())}
                            className="px-2 py-1 rounded-lg text-[10px] bg-white/10 border border-white/20 text-zinc-200 hover:bg-white/15 uppercase tracking-wider"
                            title="Recargar estilo de subtítulos desde el kit de marca"
                        >
                            {`Kit: ${brandKitName}`}
                        </button>
                    </div>

                    <div className="space-y-6 flex-1">
                        <div>
                            <label className="text-xs font-bold text-zinc-400 uppercase tracking-wider mb-2 block">Preajustes</label>
                            <div className="grid grid-cols-2 gap-2">
                                {CAPTION_STYLE_PRESETS.map((preset) => (
                                    <button
                                        key={preset.id}
                                        type="button"
                                        onClick={() => applyCaptionPreset(preset.id)}
                                        className="text-[11px] px-2 py-2 rounded-lg border border-white/10 bg-white/5 text-zinc-200 hover:bg-white/10 text-left"
                                    >
                                        {preset.name}
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Position Selector */}
                        <div>
                            <label className="text-xs font-bold text-zinc-400 uppercase tracking-wider mb-3 block">Posición</label>
                            <div className="grid grid-cols-1 gap-2">
                                <button 
                                    onClick={() => setPosition('top')}
                                    className={`p-3 rounded-xl border flex items-center gap-3 transition-all ${position === 'top' ? 'bg-primary/20 border-primary text-white' : 'bg-white/5 border-white/5 text-zinc-400 hover:bg-white/10'}`}
                                >
                                    <div className="w-8 h-8 rounded-lg bg-black/50 border border-white/10 flex items-start justify-center pt-1">
                                        <div className="w-4 h-0.5 bg-white/50 rounded-full"></div>
                                    </div>
                                    <span className="font-medium">Arriba</span>
                                </button>
                                
                                <button 
                                    onClick={() => setPosition('middle')}
                                    className={`p-3 rounded-xl border flex items-center gap-3 transition-all ${position === 'middle' ? 'bg-primary/20 border-primary text-white' : 'bg-white/5 border-white/5 text-zinc-400 hover:bg-white/10'}`}
                                >
                                    <div className="w-8 h-8 rounded-lg bg-black/50 border border-white/10 flex items-center justify-center">
                                        <div className="w-4 h-0.5 bg-white/50 rounded-full"></div>
                                    </div>
                                    <span className="font-medium">Centro</span>
                                </button>
                                
                                <button 
                                    onClick={() => setPosition('bottom')}
                                    className={`p-3 rounded-xl border flex items-center gap-3 transition-all ${position === 'bottom' ? 'bg-primary/20 border-primary text-white' : 'bg-white/5 border-white/5 text-zinc-400 hover:bg-white/10'}`}
                                >
                                    <div className="w-8 h-8 rounded-lg bg-black/50 border border-white/10 flex items-end justify-center pb-1">
                                        <div className="w-4 h-0.5 bg-white/50 rounded-full"></div>
                                    </div>
                                    <span className="font-medium">Abajo</span>
                                </button>
                            </div>
                        </div>

                        <div>
                            <label className="text-xs font-bold text-zinc-400 uppercase tracking-wider mb-3 block">Estilo</label>
                            <div className="space-y-3">
                                <div className="grid grid-cols-2 gap-3">
                                    <div>
                                        <label className="text-[10px] text-zinc-500">Tipografía</label>
                                        <select
                                            value={fontFamily}
                                            onChange={(e) => setFontFamily(e.target.value)}
                                            className="input-field text-xs"
                                        >
                                            <option value="Impact">Impact</option>
                                            <option value="Arial Black">Arial Black</option>
                                            <option value="Arial">Arial</option>
                                            <option value="Verdana">Verdana</option>
                                        </select>
                                    </div>
                                    <div>
                                        <label className="text-[10px] text-zinc-500">Tamaño</label>
                                        <input
                                            type="number"
                                            min="16"
                                            max="80"
                                            value={fontSize}
                                            onChange={(e) => setFontSize(Number(e.target.value || 24))}
                                            className="input-field text-xs"
                                        />
                                    </div>
                                </div>
                                <div className="grid grid-cols-2 gap-3">
                                    <div>
                                        <label className="text-[10px] text-zinc-500">Color texto</label>
                                        <input
                                            type="color"
                                            value={fontColor}
                                            onChange={(e) => setFontColor(e.target.value)}
                                            className="w-full h-9 rounded-lg border border-white/10 bg-black/40 p-1"
                                        />
                                    </div>
                                    <div>
                                        <label className="text-[10px] text-zinc-500">Color borde</label>
                                        <input
                                            type="color"
                                            value={strokeColor}
                                            onChange={(e) => setStrokeColor(e.target.value)}
                                            className="w-full h-9 rounded-lg border border-white/10 bg-black/40 p-1"
                                        />
                                    </div>
                                </div>
                                <div className="grid grid-cols-2 gap-3">
                                    <div>
                                        <label className="text-[10px] text-zinc-500">Grosor borde</label>
                                        <input
                                            type="number"
                                            min="0"
                                            max="8"
                                            value={strokeWidth}
                                            onChange={(e) => setStrokeWidth(Number(e.target.value || 2))}
                                            className="input-field text-xs"
                                        />
                                    </div>
                                    <div className="flex items-end gap-2">
                                        <input
                                            type="checkbox"
                                            checked={bold}
                                            onChange={(e) => setBold(e.target.checked)}
                                            className="w-4 h-4 rounded border-zinc-600 bg-black/50 text-primary focus:ring-primary"
                                        />
                                        <label className="text-[10px] text-zinc-500">Negrita</label>
                                    </div>
                                </div>
                                <div className="grid grid-cols-2 gap-3">
                                    <div>
                                        <label className="text-[10px] text-zinc-500">Color caja</label>
                                        <input
                                            type="color"
                                            value={boxColor}
                                            onChange={(e) => setBoxColor(e.target.value)}
                                            className="w-full h-9 rounded-lg border border-white/10 bg-black/40 p-1"
                                        />
                                    </div>
                                    <div>
                                        <label className="text-[10px] text-zinc-500">Opacidad caja</label>
                                        <input
                                            type="number"
                                            min="0"
                                            max="100"
                                            value={boxOpacity}
                                            onChange={(e) => setBoxOpacity(Number(e.target.value || 0))}
                                            className="input-field text-xs"
                                        />
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div>
                            <label className="text-xs font-bold text-zinc-400 uppercase tracking-wider mb-2 block">Revisar subtítulos</label>
                            <button
                                type="button"
                                onClick={async () => {
                                    if (!onLoadSrt) return;
                                    setLoadingSrt(true);
                                    try {
                                        const text = await onLoadSrt();
                                        if (typeof text === 'string') setSrtText(text);
                                    } finally {
                                        setLoadingSrt(false);
                                    }
                                }}
                                className="mb-2 px-3 py-2 rounded-lg text-xs bg-white/5 hover:bg-white/10 text-zinc-300 border border-white/10"
                            >
                                {loadingSrt ? 'Cargando...' : 'Cargar subtítulos'}
                            </button>
                            <textarea
                                value={srtText}
                                onChange={(e) => setSrtText(e.target.value)}
                                rows={6}
                                className="w-full bg-black/40 border border-white/10 rounded-lg p-2 text-xs text-white focus:outline-none focus:border-primary/50 resize-none"
                                placeholder="Aquí puedes corregir tildes o texto antes de quemarlos."
                            />
                        </div>
                    </div>

                    <button
                        onClick={() => onGenerate({
                            position,
                            fontSize,
                            fontFamily,
                            fontColor,
                            strokeColor,
                            strokeWidth,
                            bold,
                            boxColor,
                            boxOpacity,
                            srtContent: srtText
                        })}
                        disabled={isProcessing}
                        className="w-full py-4 mt-6 bg-gradient-to-r from-yellow-500 to-orange-500 hover:from-yellow-400 hover:to-orange-400 text-black font-bold rounded-xl shadow-lg shadow-orange-500/20 transition-all active:scale-[0.98] flex items-center justify-center gap-2"
                    >
                        {isProcessing ? <Loader2 size={20} className="animate-spin" /> : <Type size={20} />}
                        {isProcessing ? 'Generando...' : 'Generar subtítulos'}
                    </button>
                    
                    <p className="text-[10px] text-zinc-500 text-center mt-3">
                        Usa timestamps por palabra para sincronizar con precisión.
                    </p>
                </div>
            </div>
        </div>
    );
}
