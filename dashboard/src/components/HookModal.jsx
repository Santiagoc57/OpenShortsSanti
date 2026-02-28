import React, { useState, useEffect } from 'react';
import { X, Loader2, Wand2, Type, AlertCircle } from 'lucide-react';

const POSITIONS = [
    { value: 'top', label: 'Top' },
    { value: 'center', label: 'Center' },
    { value: 'bottom', label: 'Bottom' },
];

const SIZES = [
    { value: 'S', label: 'Small' },
    { value: 'M', label: 'Medium' },
    { value: 'L', label: 'Large' },
];

export default function HookModal({ isOpen, onClose, onGenerate, isProcessing, videoUrl, initialText }) {
    const [text, setText] = useState('');
    const [position, setPosition] = useState('top');
    const [size, setSize] = useState('M');

    useEffect(() => {
        if (isOpen) {
            setText(initialText || '');
            setPosition('top');
            setSize('M');
        }
    }, [isOpen, initialText]);

    if (!isOpen) return null;

    const handleSubmit = () => {
        if (!text.trim()) return;
        onGenerate({ text: text.trim(), position, size });
    };

    return (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-[fadeIn_0.2s_ease-out]">
            <div className="bg-[#121214] border border-white/10 p-6 rounded-2xl w-full max-w-md shadow-2xl relative">
                <button
                    onClick={onClose}
                    disabled={isProcessing}
                    className="absolute top-4 right-4 text-zinc-500 hover:text-white disabled:opacity-50"
                >
                    <X size={20} />
                </button>

                <div className="flex items-center gap-3 mb-6">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-400 to-yellow-500 flex items-center justify-center">
                        <Wand2 size={20} className="text-black" />
                    </div>
                    <div>
                        <h3 className="text-lg font-bold text-white">Viral Hook</h3>
                        <p className="text-xs text-zinc-500">Add an attention-grabbing text overlay</p>
                    </div>
                </div>

                {/* Preview */}
                <div className="mb-6 rounded-xl overflow-hidden bg-black aspect-video relative">
                    <video
                        src={videoUrl}
                        className="w-full h-full object-contain"
                        muted
                        playsInline
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent pointer-events-none" />
                    {/* Hook text preview overlay */}
                    {text.trim() && (
                        <div
                            className="absolute left-0 right-0 flex justify-center pointer-events-none px-4"
                            style={{
                                top: position === 'top' ? '15%' : position === 'center' ? '45%' : undefined,
                                bottom: position === 'bottom' ? '15%' : undefined,
                            }}
                        >
                            <span
                                className="text-white font-black text-center px-4 py-2 rounded-lg bg-black/65 backdrop-blur-sm border border-white/10 max-w-[90%] break-words"
                                style={{
                                    fontSize: size === 'S' ? '12px' : size === 'L' ? '20px' : '16px',
                                }}
                            >
                                {text}
                            </span>
                        </div>
                    )}
                </div>

                {/* Hook Text */}
                <div className="mb-4">
                    <label className="block text-sm font-medium text-zinc-400 mb-2">
                        <Type size={14} className="inline mr-2" />
                        Hook Text
                    </label>
                    <textarea
                        value={text}
                        onChange={(e) => setText(e.target.value)}
                        rows={2}
                        maxLength={120}
                        className="w-full bg-black/40 border border-white/10 rounded-lg p-3 text-sm text-white focus:outline-none focus:border-yellow-500/50 placeholder-zinc-600 resize-none"
                        placeholder="e.g. Wait for it... 🔥"
                        disabled={isProcessing}
                    />
                    <p className="text-[10px] text-zinc-600 mt-1 text-right">{text.length}/120</p>
                </div>

                {/* Position & Size Row */}
                <div className="grid grid-cols-2 gap-4 mb-6">
                    <div>
                        <label className="block text-xs font-medium text-zinc-400 mb-2">Position</label>
                        <div className="flex gap-2">
                            {POSITIONS.map((p) => (
                                <button
                                    key={p.value}
                                    onClick={() => setPosition(p.value)}
                                    disabled={isProcessing}
                                    className={`flex-1 py-2 rounded-lg text-xs font-medium transition-all border ${position === p.value
                                            ? 'bg-yellow-500/20 border-yellow-500/50 text-yellow-300'
                                            : 'bg-white/5 border-white/5 text-zinc-400 hover:bg-white/10'
                                        }`}
                                >
                                    {p.label}
                                </button>
                            ))}
                        </div>
                    </div>
                    <div>
                        <label className="block text-xs font-medium text-zinc-400 mb-2">Size</label>
                        <div className="flex gap-2">
                            {SIZES.map((s) => (
                                <button
                                    key={s.value}
                                    onClick={() => setSize(s.value)}
                                    disabled={isProcessing}
                                    className={`flex-1 py-2 rounded-lg text-xs font-medium transition-all border ${size === s.value
                                            ? 'bg-yellow-500/20 border-yellow-500/50 text-yellow-300'
                                            : 'bg-white/5 border-white/5 text-zinc-400 hover:bg-white/10'
                                        }`}
                                >
                                    {s.label}
                                </button>
                            ))}
                        </div>
                    </div>
                </div>

                {/* Processing State */}
                {isProcessing && (
                    <div className="mb-4 p-4 bg-white/5 rounded-lg border border-white/10">
                        <div className="flex items-center gap-3">
                            <Loader2 size={20} className="text-yellow-400 animate-spin" />
                            <div>
                                <p className="text-sm text-white font-medium">Adding viral hook...</p>
                                <p className="text-xs text-zinc-500">Rendering text overlay</p>
                            </div>
                        </div>
                    </div>
                )}

                {/* Actions */}
                <div className="flex gap-3">
                    <button
                        onClick={onClose}
                        disabled={isProcessing}
                        className="flex-1 py-3 bg-white/5 hover:bg-white/10 text-zinc-300 rounded-xl font-medium transition-colors disabled:opacity-50"
                    >
                        Cancel
                    </button>
                    <button
                        onClick={handleSubmit}
                        disabled={isProcessing || !text.trim()}
                        className="flex-1 py-3 bg-gradient-to-r from-amber-400 to-yellow-500 hover:from-amber-300 hover:to-yellow-400 text-black rounded-xl font-bold transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                    >
                        {isProcessing ? (
                            <>
                                <Loader2 size={16} className="animate-spin" />
                                Adding...
                            </>
                        ) : (
                            <>
                                <Wand2 size={16} />
                                Add Hook
                            </>
                        )}
                    </button>
                </div>
            </div>
        </div>
    );
}
