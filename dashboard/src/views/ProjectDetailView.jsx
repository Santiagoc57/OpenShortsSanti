import React from 'react';
import { 
  ArrowLeft, Activity, RefreshCw, Trash2, Pause, Terminal, 
  ChevronDown, Sparkles, ChevronRight, List, LayoutGrid, FileVideo, 
  Scissors, LayoutDashboard, Search, Settings, History, Check, 
  RotateCcw 
} from 'lucide-react';
import ResultCard from '../components/ResultCard';
import ClipStudioModal from '../components/ClipStudioModal';
import { 
  getApiUrl, 
  formatTimelineTime, 
  strategyLabel, 
  scopeLabel, 
  queueStatusLabel, 
  projectStatusLabel, 
  projectSourceBadgeClass 
} from '../utils';

const ProjectDetailView = ({
  // Job Status & Manager
  jobId, status, results, logs, processingMedia, isPollingPaused, setIsPollingPaused, 
  handleRetryJob, handleCancelJob, removeProject, setLogs, processingTimeline, isRetryingJob,
  
  // Navigation
  setActiveTab, setProjectsViewMode,
  
  // Clips Search & Data
  clipSearch: {
    isSearchingClips, clipSearchQuery, setClipSearchQuery, handleClipSearch, 
    clipSearchModePreset, setClipSearchModePreset, clipSearchChapterFilter, setClipSearchChapterFilter,
    clipSearchStartTime, setClipSearchStartTime, clipSearchEndTime, setClipSearchEndTime,
    clipSearchSpeakerFilter, setClipSearchSpeakerFilter, clipSearchError, clipSearchResults,
    clipSearchChapters, clipHybridShortlist, clipSearchProvider, clipSearchMode, 
    clipSearchRelaxed, clipSearchKeywords, clipSearchPhrases, clipSearchScope,
    availableSearchSpeakers, loadTranscriptSegments, isLoadingTranscript, transcriptFilter,
    setTranscriptFilter, transcriptError, visibleTranscriptSegments, transcriptTotal,
    transcriptHasSpeakers, handleTranscriptSegmentPlay, transcriptSegments
  },
  
  // Exporter Actions
  clipExporter: {
    isExportingPack, handleExportPack, isGeneratingTrailer, handleGenerateTrailer,
    batchScheduleReport, handleBatchScheduleReport, packExportReport, isBatchScheduling
  },
  
  // UI States
  logsVisible, setLogsVisible,
  clipSort, setClipSort,
  clipFilter, setClipFilter,
  clipTagFilter, setClipTagFilter,
  clipsViewMode, setClipsViewMode,
  batchStrategy, applyBatchStrategy,
  batchTopCount, setBatchTopCount,
  batchStartDelayMinutes, setBatchStartDelayMinutes,
  batchIntervalMinutes, setBatchIntervalMinutes,
  batchScope, setBatchScope,
  
  // Modals & Contexts
  studioContext, closeClipStudio, handleStudioClipPatched, handleStudioApplied,
  captionFontOptions, elevenLabsKey,
  
  // Helpers & Computed
  processingProjectName, processingSourceLabel, visibleClips, sortedClips,
  showSettings, apiKey, setApiKey, setElevenLabsKey, handleReset,
  availableTags, outputModeLabel, handleClipPlay,
  showTrailerFocusLayout, trailerSegmentsCount, trailerInspectorTab, setTrailerInspectorTab,
  trailerPreviewRef, trailerPrimaryClip, isSuperTrailerMode, trailerVideoUrl,
  trailerScoreLabel, handleTranscriptSegmentPlay: globalHandleTranscriptSegmentPlay,
  uploadPostKey, uploadUserId, userProfiles
}) => {
  const costAnalysis = results?.cost_analysis || null;
  const totalCost = Number(costAnalysis?.total_cost);
  const hasCostAnalysis = Number.isFinite(totalCost);
  const inputTokens = Number.isFinite(Number(costAnalysis?.input_tokens)) ? Number(costAnalysis.input_tokens).toLocaleString() : '-';
  const outputTokens = Number.isFinite(Number(costAnalysis?.output_tokens)) ? Number(costAnalysis.output_tokens).toLocaleString() : '-';
  const safeLogs = Array.isArray(logs) ? logs : [];
  const safeAvailableTags = Array.isArray(availableTags) ? availableTags : [];
  const safeVisibleClips = Array.isArray(visibleClips) ? visibleClips : [];
  const safeSortedClips = Array.isArray(sortedClips) ? sortedClips : [];

  return (
    <div className="animate-[fadeIn_0.3s_ease-out]">
      {studioContext ? (
        <div className="h-[calc(100vh-7.8rem)] min-h-[620px] lg:min-h-[720px]">
          <ClipStudioModal
            isOpen
            standalone
            onClose={closeClipStudio}
            jobId={studioContext.jobId}
            clipIndex={studioContext.clipIndex}
            clip={studioContext.clip}
            currentVideoUrl={studioContext.currentVideoUrl}
            projectLanguage={studioContext.projectLanguage}
            defaultDubbingTargetLanguage={studioContext.defaultDubbingTargetLanguage}
            onClipPatched={handleStudioClipPatched}
            onApplied={handleStudioApplied}
            fontCatalog={captionFontOptions}
            elevenLabsKey={elevenLabsKey}
            transcriptSegments={transcriptSegments}
          />
        </div>
      ) : (
        <div className="min-h-[620px] flex flex-col gap-5">
          <div className="w-full border border-slate-200/90 dark:border-white/10 bg-white/95 dark:bg-black/20 rounded-2xl p-4 shadow-[0_8px_26px_rgba(15,23,42,0.08)]">
            <div className="flex flex-col gap-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-2 min-w-0">
                  <button
                    type="button"
                    onClick={() => setProjectsViewMode('list')}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-slate-200 dark:border-slate-600 text-slate-600 dark:text-slate-300 bg-slate-50/80 dark:bg-slate-800/50 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors text-xs"
                    title="Volver a lista de proyectos"
                  >
                    <ArrowLeft size={13} />
                    Proyectos
                  </button>
                  <Activity className={`text-primary ${status === 'processing' ? 'animate-pulse' : ''}`} size={18} />
                  <h2 className="text-base md:text-lg font-semibold text-slate-900 dark:text-white truncate">
                    Procesamiento del proyecto
                  </h2>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <button
                    type="button"
                    onClick={handleRetryJob}
                    disabled={!jobId || status === 'processing' || isRetryingJob}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-slate-200 dark:border-slate-600 bg-slate-50/80 dark:bg-slate-800/50 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors text-[11px] disabled:opacity-50 disabled:cursor-not-allowed"
                    title="Reprocesar proyecto"
                  >
                    <RefreshCw size={12} className={isRetryingJob ? 'animate-spin' : ''} />
                    {isRetryingJob ? 'Reprocesando...' : 'Recargar'}
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      Promise.resolve(jobId ? removeProject(jobId) : handleReset()).finally(() => {
                        setProjectsViewMode('list');
                      });
                    }}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-red-200 dark:border-red-700 text-red-600 dark:text-red-300 bg-red-50/40 dark:bg-red-900/10 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors text-[11px] disabled:opacity-50 disabled:cursor-not-allowed"
                    title="Eliminar proyecto"
                  >
                    <Trash2 size={12} />
                    Eliminar
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      if (status === 'processing') {
                        handleCancelJob();
                      }
                    }}
                    disabled={status !== 'processing' || !jobId}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-slate-200 dark:border-slate-600 bg-slate-50/80 dark:bg-slate-800/50 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors text-[11px] disabled:opacity-50 disabled:cursor-not-allowed"
                    title="Pausar/cancelar procesamiento"
                  >
                    <Pause size={12} />
                    Pausar
                  </button>
                  <span className={`text-[11px] px-2.5 py-1 rounded-full border font-semibold uppercase tracking-wide ${status === 'processing'
                    ? 'bg-primary/10 border-primary/20 text-primary'
                    : status === 'complete'
                      ? 'bg-emerald-100 dark:bg-green-500/10 border-emerald-200 dark:border-green-500/20 text-emerald-700 dark:text-green-400'
                      : 'bg-red-100 dark:bg-red-500/10 border-red-200 dark:border-red-500/20 text-red-700 dark:text-red-400'
                    }`}>
                    {status === 'processing' ? 'EN PROCESO' : status === 'paused' ? 'PAUSADO' : status === 'complete' ? 'COMPLETADO' : 'ERROR'}
                  </span>
                </div>
              </div>

              <div className="space-y-2">
                <div className="flex flex-col gap-1 md:flex-row md:items-center md:justify-between text-xs md:text-sm">
                  <span className="font-medium text-slate-700 dark:text-slate-200">{processingTimeline.headline}</span>
                  <span className="text-slate-500 dark:text-slate-400 md:text-right">{processingTimeline.stepProgressLabel}</span>
                </div>
                <div className="w-full h-2.5 rounded-full bg-slate-200 dark:bg-slate-700 overflow-hidden">
                  <div
                    className={`h-full transition-all duration-500 ${status === 'error' ? 'bg-red-400' : 'bg-gradient-to-r from-primary to-indigo-400'
                      }`}
                    style={{ width: `${processingTimeline.progressPercent}%` }}
                  />
                </div>
              </div>

              <div className="rounded-xl border border-slate-200 dark:border-white/10 overflow-hidden">
                <button
                  onClick={() => setLogsVisible(!logsVisible)}
                  className="w-full px-3 py-2.5 text-left flex items-center justify-between bg-slate-50/80 dark:bg-white/5 hover:bg-slate-100 dark:hover:bg-white/10 transition-colors"
                >
                  <span className="text-xs font-mono text-slate-600 dark:text-zinc-300 flex items-center gap-2">
                    <Terminal size={12} /> Logs del sistema (opcional)
                  </span>
                  <ChevronDown size={14} className={`text-zinc-500 transition-transform ${logsVisible ? '' : '-rotate-90'}`} />
                </button>
                {logsVisible && (
                  <div className="max-h-36 overflow-y-auto p-3 font-mono text-xs space-y-1.5 custom-scrollbar text-slate-600 dark:text-zinc-400 bg-white dark:bg-[#0c0c0e]">
                    {safeLogs.length === 0 && (
                      <div className="text-slate-400 dark:text-zinc-500">Aún no hay logs.</div>
                    )}
                    {safeLogs.map((log, i) => (
                      <div key={i} className={`flex gap-2 ${log.toLowerCase().includes('error') ? 'text-red-500 dark:text-red-400' : 'text-slate-600 dark:text-zinc-400'}`}>
                        <span className="text-slate-400 dark:text-zinc-600 shrink-0">{new Date().toLocaleTimeString()}</span>
                        <span>{log}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="flex flex-col gap-4">
            {/* Results Grid */}
            <div className="w-full flex flex-col border border-slate-200/90 dark:border-white/10 bg-white/95 dark:bg-background rounded-2xl p-5 transition-all duration-700 ease-in-out shadow-[0_8px_26px_rgba(15,23,42,0.08)]">
              <div className="mb-5 shrink-0">
                <div className="flex items-center gap-2 text-[11px] text-slate-500 dark:text-zinc-500 mb-1">
                  <span>Proyecto</span>
                  <ChevronRight size={12} />
                  <span className="truncate">{processingProjectName}</span>
                </div>
                <div className="flex flex-wrap items-center gap-2">
                  <h2 className="text-lg font-semibold flex items-center gap-2 text-slate-900 dark:text-white">
                    <Sparkles className="text-yellow-400" size={20} />
                    {showTrailerFocusLayout ? 'Super trailer' : 'Clips generados'}
                  </h2>
                  {!showTrailerFocusLayout && safeSortedClips.length > 0 && (
                    <div className="ml-auto flex items-center gap-2">
                      <span className="text-xs bg-slate-100 dark:bg-white/10 text-slate-700 dark:text-white px-2 py-0.5 rounded-full border border-slate-200 dark:border-transparent">
                        {safeVisibleClips.length}/{safeSortedClips.length} Clips
                      </span>
                      <div className="inline-flex items-center rounded-lg border border-slate-200 dark:border-white/15 bg-white dark:bg-white/5 p-1">
                        <button
                          type="button"
                          onClick={() => setClipsViewMode('list')}
                          className={`inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-[11px] font-medium transition-colors ${clipsViewMode === 'list'
                            ? 'bg-primary/15 text-primary'
                            : 'text-slate-500 dark:text-zinc-400 hover:bg-slate-100 dark:hover:bg-white/10'
                            }`}
                          title="Vista lista"
                        >
                          <List size={12} />
                          Lista
                        </button>
                        <button
                          type="button"
                          onClick={() => setClipsViewMode('gallery')}
                          className={`inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-[11px] font-medium transition-colors ${clipsViewMode === 'gallery'
                            ? 'bg-primary/15 text-primary'
                            : 'text-slate-500 dark:text-zinc-400 hover:bg-slate-100 dark:hover:bg-white/10'
                            }`}
                          title="Vista galería"
                        >
                          <LayoutGrid size={12} />
                          Galería
                        </button>
                      </div>
                    </div>
                  )}
                </div>
                <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
                  <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md border border-slate-200 dark:border-white/10 bg-slate-50 dark:bg-white/5 text-slate-700 dark:text-zinc-300">
                    <FileVideo size={12} />
                    {`Fuente: ${processingSourceLabel}`}
                  </span>
                  <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md border border-slate-200 dark:border-white/10 bg-slate-50 dark:bg-white/5 text-slate-700 dark:text-zinc-300">
                    <Scissors size={12} />
                    {showTrailerFocusLayout
                      ? `Segmentos: ${trailerSegmentsCount || '-'}`
                      : `Objetivo: ${processingMedia?.clipCount || '-'} clips`}
                  </span>
                  <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md border border-slate-200 dark:border-white/10 bg-slate-50 dark:bg-white/5 text-slate-700 dark:text-zinc-300">
                    <LayoutDashboard size={12} />
                    {`Formato: ${outputModeLabel(processingMedia?.aspectRatio)}`}
                  </span>
                  {hasCostAnalysis && (
                    <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md border border-green-300 dark:border-green-500/30 bg-green-50 dark:bg-green-500/10 text-green-700 dark:text-green-300" title={`Entrada: ${inputTokens} | Salida: ${outputTokens}`}>
                      {`Costo: $${totalCost.toFixed(5)}`}
                    </span>
                  )}
                </div>
              </div>

              {!showTrailerFocusLayout && safeSortedClips.length > 0 && (
                <div className="mb-4 shrink-0 rounded-xl border border-slate-200 dark:border-white/10 bg-slate-50/80 dark:bg-white/[0.02] p-3.5 space-y-3">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-xs font-medium text-slate-500 dark:text-zinc-500">Orden:</span>
                    <select
                      value={clipSort}
                      onChange={(e) => setClipSort(e.target.value)}
                      className="text-xs bg-white dark:bg-white/5 border border-slate-200 dark:border-white/10 rounded-md px-2 py-1.5 text-slate-700 dark:text-zinc-200"
                    >
                      <option value="top">Mayor puntaje</option>
                      <option value="balanced">Línea de tiempo</option>
                      <option value="safe">Más seguros</option>
                    </select>
                    <span className="text-xs font-medium text-slate-500 dark:text-zinc-500 ml-1">Filtro:</span>
                    <select
                      value={clipFilter}
                      onChange={(e) => setClipFilter(e.target.value)}
                      className="text-xs bg-white dark:bg-white/5 border border-slate-200 dark:border-white/10 rounded-md px-2 py-1.5 text-slate-700 dark:text-zinc-200"
                    >
                      <option value="all">Todos</option>
                      <option value="top">Alto (80+)</option>
                      <option value="medium">Medio (65-79)</option>
                      <option value="low">Bajo (&lt;65)</option>
                    </select>
                    <span className="text-xs font-medium text-slate-500 dark:text-zinc-500">Etiqueta:</span>
                    <select
                      value={clipTagFilter}
                      onChange={(e) => setClipTagFilter(e.target.value)}
                      className="text-xs bg-white dark:bg-white/5 border border-slate-200 dark:border-white/10 rounded-md px-2 py-1.5 text-slate-700 dark:text-zinc-200"
                    >
                      <option value="all">Todas</option>
                      {safeAvailableTags.map((tag) => (
                        <option key={tag} value={tag}>{tag}</option>
                      ))}
                    </select>
                    <span className="text-xs font-medium text-slate-500 dark:text-zinc-500">Estrategia:</span>
                    <select
                      value={batchStrategy}
                      onChange={(e) => applyBatchStrategy(e.target.value)}
                      className="text-xs bg-white dark:bg-white/5 border border-slate-200 dark:border-white/10 rounded-md px-2 py-1.5 text-slate-700 dark:text-zinc-200"
                    >
                      <option value="growth">Crecimiento</option>
                      <option value="balanced">Balanceada</option>
                      <option value="conservative">Conservadora</option>
                      <option value="custom">Personalizada</option>
                    </select>
                    <span className="text-xs font-medium text-slate-500 dark:text-zinc-500 ml-1">N clips:</span>
                    <input
                      type="number"
                      min="1"
                      max="10"
                      value={batchTopCount}
                      onChange={(e) => {
                        setBatchStrategy('custom');
                        setBatchTopCount(Math.max(1, Math.min(10, Number(e.target.value || 1))));
                      }}
                      className="w-16 text-xs bg-white dark:bg-white/5 border border-slate-200 dark:border-white/10 rounded-md px-2 py-1.5 text-slate-700 dark:text-zinc-200"
                    />
                    <span className="text-xs font-medium text-slate-500 dark:text-zinc-500">Inicia en:</span>
                    <select
                      value={batchStartDelayMinutes}
                      onChange={(e) => {
                        setBatchStrategy('custom');
                        setBatchStartDelayMinutes(Number(e.target.value));
                      }}
                      className="text-xs bg-white dark:bg-white/5 border border-slate-200 dark:border-white/10 rounded-md px-2 py-1.5 text-slate-700 dark:text-zinc-200"
                    >
                      <option value={0}>ahora</option>
                      <option value={5}>5m</option>
                      <option value={15}>15m</option>
                      <option value={30}>30m</option>
                      <option value={60}>60m</option>
                    </select>
                    <span className="text-xs font-medium text-slate-500 dark:text-zinc-500">Cada:</span>
                    <select
                      value={batchIntervalMinutes}
                      onChange={(e) => {
                        setBatchStrategy('custom');
                        setBatchIntervalMinutes(Number(e.target.value));
                      }}
                      className="text-xs bg-white dark:bg-white/5 border border-slate-200 dark:border-white/10 rounded-md px-2 py-1.5 text-slate-700 dark:text-zinc-200"
                    >
                      <option value={15}>15m</option>
                      <option value={30}>30m</option>
                      <option value={60}>60m</option>
                      <option value={120}>120m</option>
                      <option value={240}>240m</option>
                    </select>
                    <span className="text-xs font-medium text-slate-500 dark:text-zinc-500">Alcance:</span>
                    <select
                      value={batchScope}
                      onChange={(e) => {
                        setBatchStrategy('custom');
                        setBatchScope(e.target.value);
                      }}
                      className="text-xs bg-white dark:bg-white/5 border border-slate-200 dark:border-white/10 rounded-md px-2 py-1.5 text-slate-700 dark:text-zinc-200"
                    >
                      <option value="visible">Visible</option>
                      <option value="global">Global</option>
                    </select>
                    <button
                      onClick={() => handleBatchScheduleReport()}
                      disabled={isBatchScheduling || (batchScope === 'global' ? safeSortedClips.length === 0 : safeVisibleClips.length === 0)}
                      className="ml-1 text-xs bg-violet-100 dark:bg-primary/20 border border-violet-300 dark:border-primary/40 text-violet-700 dark:text-primary rounded-md px-2 py-1.5 hover:bg-violet-200 dark:hover:bg-primary/30 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {isBatchScheduling ? 'Encolando...' : `Encolar ${Math.max(1, Math.min(10, Number(batchTopCount) || 1))}`}
                    </button>
                    <button
                      onClick={handleExportPack}
                      disabled={isExportingPack || !jobId}
                      className="text-xs bg-emerald-100 dark:bg-emerald-500/20 border border-emerald-300 dark:border-emerald-500/40 text-emerald-700 dark:text-emerald-300 rounded-md px-2 py-1.5 hover:bg-emerald-200 dark:hover:bg-emerald-500/30 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {isExportingPack ? 'Exportando...' : 'Exportar paquete'}
                    </button>
                    <button
                      onClick={handleGenerateTrailer}
                      disabled={isGeneratingTrailer || !jobId || status !== 'complete'}
                      className="text-xs bg-amber-100 dark:bg-amber-500/20 border border-amber-300 dark:border-amber-500/40 text-amber-700 dark:text-amber-300 rounded-md px-2 py-1.5 hover:bg-amber-200 dark:hover:bg-amber-500/30 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {isGeneratingTrailer ? 'Generando trailer...' : 'Generar Super Trailer ⚡'}
                    </button>
                  </div>
                </div>
              )}

              {!showTrailerFocusLayout && safeSortedClips.length > 0 && (
                <div className="mb-4 rounded-xl border border-slate-200 dark:border-white/10 bg-white dark:bg-white/[0.02] p-3.5 shadow-sm">
                  <div className="flex items-center gap-2">
                    <Search size={14} className="text-slate-400" />
                    <select
                      value={clipSearchModePreset}
                      onChange={(e) => setClipSearchModePreset(e.target.value)}
                      className="text-xs bg-slate-50 dark:bg-black/30 border border-slate-200 dark:border-white/10 rounded-md px-2 py-1.5 text-slate-700 dark:text-zinc-200 shrink-0"
                    >
                      <option value="exact">Exacta</option>
                      <option value="balanced">Balanceada</option>
                      <option value="broad">Amplia</option>
                    </select>
                    <input
                      type="text"
                      value={clipSearchQuery}
                      onChange={(e) => setClipSearchQuery(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') handleClipSearch();
                      }}
                      placeholder="Clip Anything: ej. 'cuando habla de deuda' o 'momento polémico'"
                      className="flex-1 text-xs bg-slate-50 dark:bg-black/30 border border-slate-200 dark:border-white/10 rounded-md px-2 py-1.5 text-slate-700 dark:text-zinc-200"
                    />
                    <button
                      onClick={handleClipSearch}
                      disabled={isSearchingClips || !clipSearchQuery.trim()}
                      className="text-xs bg-slate-100 dark:bg-white/10 border border-slate-200 dark:border-white/20 rounded-md px-2 py-1.5 text-slate-700 dark:text-zinc-200 hover:bg-slate-200 dark:hover:bg-white/15 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {isSearchingClips ? 'Buscando...' : 'Buscar'}
                    </button>
                  </div>
                </div>
              )}

              {batchScheduleReport && (
                <div className={`mb-4 text-xs rounded-lg border px-3 py-2 ${batchScheduleReport.failures.length === 0
                  ? 'bg-green-50 dark:bg-green-500/10 border-green-200 dark:border-green-500/30 text-green-700 dark:text-green-300'
                  : 'bg-amber-50 dark:bg-amber-500/10 border-amber-200 dark:border-amber-500/30 text-amber-700 dark:text-amber-200'
                  }`}>
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <p>{`Lote programado: ${batchScheduleReport.success}/${batchScheduleReport.total} en cola.`}</p>
                    <button
                      onClick={handleBatchScheduleReport}
                      className="text-[11px] bg-white dark:bg-white/10 border border-slate-200 dark:border-white/20 rounded px-2 py-1 hover:bg-slate-50 dark:hover:bg-white/15"
                    >
                      Exportar CSV del lote
                    </button>
                  </div>
                </div>
              )}

              <div className="flex-1 overflow-y-auto custom-scrollbar p-1">
                {!showTrailerFocusLayout && (
                  <div className={clipsViewMode === 'gallery' 
                    ? "grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 pb-12"
                    : "flex flex-col gap-4 pb-12"
                  }>
                    {safeVisibleClips.length === 0 && (
                      <div className="flex flex-col items-center justify-center py-20 text-slate-400">
                        <Activity size={40} className="mb-4 opacity-20" />
                        <p className="text-sm font-medium">Aún no hay clips generados para mostrar.</p>
                      </div>
                    )}
                    {safeVisibleClips.map((clip, index) => (
                      <ResultCard
                        key={`${clip.clip_index || index}-${clip.start}`}
                        clip={clip}
                        displayIndex={index}
                        clipIndex={clip.clip_index}
                        viewMode={clipsViewMode}
                        jobId={jobId}
                        uploadPostKey={uploadPostKey}
                        uploadUserId={uploadUserId}
                        geminiApiKey={apiKey}
                        elevenLabsKey={elevenLabsKey}
                        onPlay={handleClipPlay}
                        onClipPatched={handleStudioClipPatched}
                      />
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ProjectDetailView;
