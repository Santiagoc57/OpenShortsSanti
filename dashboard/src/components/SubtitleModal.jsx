import React, { useState } from 'react';
import { X, Type, Loader2 } from 'lucide-react';

export default function SubtitleModal({ isOpen, onClose, onGenerate, isProcessing, videoUrl, onLoadSrt }) {
    const [position, setPosition] = useState('bottom'); // bottom, middle, top
    const [fontSize, setFontSize] = useState(24);
    const [fontFamily, setFontFamily] = useState('Impact');
    const [fontColor, setFontColor] = useState('#FFFFFF');
    const [strokeColor, setStrokeColor] = useState('#000000');
    const [strokeWidth, setStrokeWidth] = useState(3);
    const [bold, setBold] = useState(true);
    const [boxColor, setBoxColor] = useState('#000000');
    const [boxOpacity, setBoxOpacity] = useState(60);
    const [srtText, setSrtText] = useState('');
    const [loadingSrt, setLoadingSrt] = useState(false);

    const toRgba = (hex, opacity) => {
        const clean = (hex || '#000000').replace('#', '');
        const r = parseInt(clean.substring(0, 2), 16);
        const g = parseInt(clean.substring(2, 4), 16);
        const b = parseInt(clean.substring(4, 6), 16);
        return `rgba(${r}, ${g}, ${b}, ${opacity / 100})`;
    };

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
                <div className="flex-1 flex flex-col items-center justify-center bg-black rounded-lg border border-white/5 overflow-hidden relative aspect-[9/16] max-h-[600px]">
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
                            This is how your subtitles<br/>will appear on the video
                        </span>
                     </div>
                </div>

                {/* Right: Controls */}
                <div className="w-full md:w-80 flex flex-col">
                    <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-2">
                        <Type className="text-primary" /> Auto Subtitles
                    </h3>

                    <div className="space-y-6 flex-1">
                        {/* Position Selector */}
                        <div>
                            <label className="text-xs font-bold text-zinc-400 uppercase tracking-wider mb-3 block">Position</label>
                            <div className="grid grid-cols-1 gap-2">
                                <button 
                                    onClick={() => setPosition('top')}
                                    className={`p-3 rounded-xl border flex items-center gap-3 transition-all ${position === 'top' ? 'bg-primary/20 border-primary text-white' : 'bg-white/5 border-white/5 text-zinc-400 hover:bg-white/10'}`}
                                >
                                    <div className="w-8 h-8 rounded-lg bg-black/50 border border-white/10 flex items-start justify-center pt-1">
                                        <div className="w-4 h-0.5 bg-white/50 rounded-full"></div>
                                    </div>
                                    <span className="font-medium">Top</span>
                                </button>
                                
                                <button 
                                    onClick={() => setPosition('middle')}
                                    className={`p-3 rounded-xl border flex items-center gap-3 transition-all ${position === 'middle' ? 'bg-primary/20 border-primary text-white' : 'bg-white/5 border-white/5 text-zinc-400 hover:bg-white/10'}`}
                                >
                                    <div className="w-8 h-8 rounded-lg bg-black/50 border border-white/10 flex items-center justify-center">
                                        <div className="w-4 h-0.5 bg-white/50 rounded-full"></div>
                                    </div>
                                    <span className="font-medium">Center</span>
                                </button>
                                
                                <button 
                                    onClick={() => setPosition('bottom')}
                                    className={`p-3 rounded-xl border flex items-center gap-3 transition-all ${position === 'bottom' ? 'bg-primary/20 border-primary text-white' : 'bg-white/5 border-white/5 text-zinc-400 hover:bg-white/10'}`}
                                >
                                    <div className="w-8 h-8 rounded-lg bg-black/50 border border-white/10 flex items-end justify-center pb-1">
                                        <div className="w-4 h-0.5 bg-white/50 rounded-full"></div>
                                    </div>
                                    <span className="font-medium">Bottom</span>
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
                        {isProcessing ? 'Generating...' : 'Generate Subtitles'}
                    </button>
                    
                    <p className="text-[10px] text-zinc-500 text-center mt-3">
                        Uses AI word-level timestamps to sync perfectly.
                    </p>
                </div>
            </div>
        </div>
    );
}
