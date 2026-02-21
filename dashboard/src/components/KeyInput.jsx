import React, { useState, useEffect } from 'react';
import { Key, Eye, EyeOff, Check } from 'lucide-react';

export default function KeyInput({ onKeySet, savedKey }) {
    const [key, setKey] = useState(savedKey || '');
    const [isVisible, setIsVisible] = useState(false);
    const [isSaved, setIsSaved] = useState(!!savedKey);

    useEffect(() => {
        if (savedKey) setKey(savedKey);
    }, [savedKey]);

    const handleSave = () => {
        if (key.trim().length > 0) {
            onKeySet(key);
            setIsSaved(true);
        }
    };

    return (
        <div className="bg-white/85 dark:bg-surface border border-slate-200 dark:border-white/10 rounded-2xl p-6 mb-8 shadow-sm animate-[fadeIn_0.5s_ease-out]">
            <div>
                <div className="flex items-center gap-3 mb-4">
                    <div className="p-2 bg-primary/10 rounded-lg text-primary">
                        <Key size={20} />
                    </div>
                    <h2 className="text-lg font-semibold text-slate-900 dark:text-white">API Key de Gemini</h2>
                </div>

                <div className="flex gap-3">
                    <div className="relative flex-1">
                        <input
                            type={isVisible ? "text" : "password"}
                            value={key}
                            onChange={(e) => {
                                setKey(e.target.value);
                                setIsSaved(false);
                            }}
                            placeholder="AIzaSy..."
                            className="input-field pr-12 font-mono"
                        />
                        <button
                            onClick={() => setIsVisible(!isVisible)}
                            className="absolute right-3 top-1/2 -translate-y-1/2 text-zinc-400 hover:text-slate-900 dark:hover:text-white transition-colors"
                        >
                            {isVisible ? <EyeOff size={18} /> : <Eye size={18} />}
                        </button>
                    </div>
                    <button
                        onClick={handleSave}
                        disabled={!key || isSaved}
                        className={`px-6 rounded-xl font-medium transition-all flex items-center gap-2 ${isSaved
                            ? 'bg-emerald-100 dark:bg-green-500/20 text-emerald-700 dark:text-green-400 cursor-default'
                            : 'bg-gradient-to-r from-primary to-indigo-500 hover:from-primary hover:to-indigo-600 text-white shadow-lg shadow-primary/20'
                            }`}
                    >
                        {isSaved ? <><Check size={18} /> Listo</> : 'Guardar'}
                    </button>
                </div>
            </div>

            <p className="mt-4 text-xs text-slate-500 dark:text-zinc-500 text-center">
                Tus claves se guardan localmente en tu navegador.
                <span className="mx-2">|</span>
                <a href="https://aistudio.google.com/app/apikey" target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">Clave Gemini</a>
            </p>
        </div>
    );
}
