import React from 'react';
import { Sparkles } from 'lucide-react';
import MediaInput from '../components/MediaInput';

const HomeView = ({ 
  status, 
  handleProcess, 
  apiKey, 
  homePrefillFile, 
  setActiveTab, 
  setProjectsViewMode 
}) => {
  return (
    <div className="animate-[fadeIn_0.3s_ease-out] py-6 md:py-12">
      <div className="relative max-w-6xl mx-auto min-h-[calc(100vh-220px)] -translate-y-[6%] md:-translate-y-[10%] flex flex-col items-center justify-center">
        <div className="max-w-4xl mx-auto text-center mb-8 md:mb-10">
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-violet-100/90 dark:bg-violet-900/30 text-violet-700 dark:text-violet-300 text-[11px] font-medium border border-violet-200 dark:border-violet-700/40 mb-4">
            <span className="w-1.5 h-1.5 rounded-full bg-violet-500 animate-pulse" />
            Motor de clips IA activo
          </span>
          <h1 className="text-4xl md:text-5xl lg:text-6xl font-extrabold text-slate-900 leading-[1.12] tracking-tight">
            Convierte videos largos en
            <span className="block pb-1 text-transparent bg-clip-text bg-gradient-to-r from-violet-600 to-indigo-500 dark:from-violet-400 dark:to-blue-400 dark:text-white">clips virales en segundos</span>
          </h1>
        </div>

        <MediaInput
          onProcess={handleProcess}
          isProcessing={status === 'processing'}
          apiKey={apiKey}
          prefillFile={homePrefillFile}
        />
      </div>
      
      {status === 'processing' && (
        <div className="max-w-6xl mx-auto mt-4">
          <div className="rounded-xl border border-primary/30 bg-primary/10 px-4 py-3 flex items-center justify-between gap-3">
            <p className="text-sm text-primary font-medium">
              Hay un proyecto en curso. Revisa el progreso en la pestaña Proyectos.
            </p>
            <button
              type="button"
              onClick={() => {
                setActiveTab('projects');
                setProjectsViewMode('detail');
              }}
              className="text-xs px-3 py-1.5 rounded-lg border border-primary/40 bg-white/70 dark:bg-black/20 text-primary hover:bg-white dark:hover:bg-black/30"
            >
              Ir a Proyectos
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default HomeView;
