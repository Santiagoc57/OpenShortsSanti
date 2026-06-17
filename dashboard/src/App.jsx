import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { 
  Settings, Instagram, ChevronDown, Check, Monitor, 
  LayoutDashboard, Scissors, Youtube
} from 'lucide-react';

import { 
  outputModeLabel,
  formatProjectDate
} from './utils';
import { apiFetch } from './config';

import { useAuthSecurity } from './hooks/useAuthSecurity';
import { useJobManager } from './hooks/useJobManager';
import { useClipSearch } from './hooks/useClipSearch';
import { useClipExporter } from './hooks/useClipExporter';

import HomeView from './views/HomeView';
import ProjectsView from './views/ProjectsView';
import SettingsView from './views/SettingsView';
import ProjectDetailView from './views/ProjectDetailView';
import TikTokIcon from './components/TikTokIcon';
import MediaInput from './components/MediaInput';

// --- Sub-componentes locales de Layout ---

const UserProfileSelector = ({ profiles, selectedUserId, onSelect }) => {
  const [isOpen, setIsOpen] = useState(false);
  if (!profiles || profiles.length === 0) return null;
  const selectedProfile = profiles.find(p => p.username === selectedUserId) || profiles[0];

  return (
    <div className="relative z-50">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center justify-between bg-white dark:bg-slate-900 border border-slate-200 dark:border-white/10 rounded-lg px-3 py-2 text-sm text-slate-700 dark:text-zinc-300 hover:bg-slate-50 dark:hover:bg-white/5 transition-colors min-w-[180px]"
      >
        <span className="flex items-center gap-2">
          <div className="w-5 h-5 rounded-full bg-gradient-to-br from-primary to-purple-600 flex items-center justify-center text-[10px] font-bold text-white">
            {selectedProfile?.username?.substring(0, 1).toUpperCase() || "U"}
          </div>
          <span className="font-medium text-slate-900 dark:text-white truncate max-w-[100px]">{selectedProfile?.username || "Select User"}</span>
        </span>
        <ChevronDown size={14} className={`text-zinc-500 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {isOpen && (
        <div className="absolute top-full mt-2 right-0 w-64 bg-white dark:bg-[#1a1a1a] border border-slate-200 dark:border-white/10 rounded-xl shadow-2xl overflow-hidden">
          <div className="max-h-60 overflow-y-auto custom-scrollbar">
            {profiles.map((profile) => (
              <button
                key={profile.username}
                onClick={() => {
                  onSelect(profile.username);
                  setIsOpen(false);
                }}
                className="w-full flex items-center justify-between px-4 py-3 hover:bg-slate-50 dark:hover:bg-white/5 transition-colors text-left group border-b border-slate-100 dark:border-white/5 last:border-0"
              >
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-full bg-gradient-to-br from-primary/20 to-purple-500/20 flex items-center justify-center text-xs font-bold text-white border border-white/10 shrink-0">
                    {profile.username.substring(0, 2).toUpperCase()}
                  </div>
                  <div className="min-w-0">
                    <div className="text-sm font-medium text-slate-700 dark:text-zinc-200 group-hover:text-slate-900 dark:group-hover:text-white transition-colors truncate">
                      {profile.username}
                    </div>
                    <div className="flex gap-2 mt-0.5">
                      <div className={`flex items-center gap-1 text-[10px] ${profile.connected.includes('tiktok') ? 'text-zinc-300' : 'text-zinc-600'}`}>
                        <TikTokIcon size={10} />
                      </div>
                      <div className={`flex items-center gap-1 text-[10px] ${profile.connected.includes('instagram') ? 'text-pink-400' : 'text-zinc-600'}`}>
                        <Instagram size={10} />
                      </div>
                      <div className={`flex items-center gap-1 text-[10px] ${profile.connected.includes('youtube') ? 'text-red-400' : 'text-zinc-600'}`}>
                        <Youtube size={10} />
                      </div>
                    </div>
                  </div>
                </div>
                {selectedUserId === profile.username && <Check size={14} className="text-primary shrink-0" />}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// --- Main Application ---

const BRAND_KIT_STORAGE_KEY = 'brandKitV1';
const DEFAULT_BRAND_KIT = {
  name: 'Predeterminado',
  subtitle_position: 'bottom',
  subtitle_font_family: 'Anton',
  subtitle_font_size: 40,
  subtitle_font_color: '#FFFFFF',
  subtitle_stroke_color: '#000000',
  subtitle_stroke_width: 3,
  subtitle_bold: true,
  subtitle_box_color: '#000000',
  subtitle_box_opacity: 60
};

export default function App() {
  const [activeTab, setActiveTab] = useState('home');
  const [homePrefillFile, setHomePrefillFile] = useState(null);
  
  const [toasts, setToasts] = useState([]);
  const showToast = useCallback((msg, type = 'info') => {
    const id = Date.now() + Math.random();
    setToasts((prev) => [...prev.slice(-4), { id, msg, type }]);
    setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== id)), 4500);
  }, []);

  // --- Logic Hooks ---
  const auth = useAuthSecurity(showToast);
  const { 
    apiKey, elevenLabsKey, huggingfaceToken, uploadPostKey, uploadUserId, userProfiles, 
    setApiKey, setElevenLabsKey, setHuggingfaceToken, setUploadPostKey, setUploadUserId, fetchUserProfiles,
    apiBaseUrlInput, setApiBaseUrlInput, apiBaseUrlActive, apiBaseUrlMessage, apiBaseUrlMessageType,
    isTestingApiBaseUrl, handleSaveApiBaseUrl, runConnectivityCheck, handleResetApiBaseUrl,
    handleTestApiBaseUrl, connectivityStatus, isConnectivityChecking
  } = auth;

  const jobManager = useJobManager(showToast, auth);
  const { 
    jobId, status, results, isPollingPaused, isRetryingJob, logs, processingMedia, 
    setIsPollingPaused, setLogs, handleProcess, handleRetryJob, handleReset, 
    handleCancelJob,
    projects, setProjects, removeProject, removeAllProjects, favoriteProjectsCount, 
    toggleProjectFavorite, saveProjectTitle, beginEditProjectTitle, cancelEditProjectTitle,
    projectTitleEditJobId, projectTitleDraft, setProjectTitleDraft, openSavedProject
  } = jobManager;

  const [clipSearchModePreset, setClipSearchModePreset] = useState('balanced');
  const clipSearch = useClipSearch(jobId, status, results, null, setLogs, apiKey, clipSearchModePreset);
  const clipExporter = useClipExporter(jobId, showToast, setLogs);

  // --- UI State & Persistent Config ---
  const [projectFilter, setProjectFilter] = useState('all');
  const [projectsViewMode, setProjectsViewMode] = useState('list');
  const [clipsViewMode, setClipsViewMode] = useState('list');
  const [clipSort, setClipSort] = useState('top');
  const [clipFilter, setClipFilter] = useState('all');
  const [clipTagFilter, setClipTagFilter] = useState('all');
  const [logsVisible, setLogsVisible] = useState(false);
  const [captionFontOptions, setCaptionFontOptions] = useState([]);

  // Batch states
  const [batchTopCount, setBatchTopCount] = useState(3);
  const [batchStartDelayMinutes, setBatchStartDelayMinutes] = useState(0);
  const [batchIntervalMinutes, setBatchIntervalMinutes] = useState(60);
  const [batchScope, setBatchScope] = useState('visible');
  const [batchStrategy, setBatchStrategy] = useState('growth');

  const [brandKit, setBrandKit] = useState(() => {
    try {
      const saved = localStorage.getItem(BRAND_KIT_STORAGE_KEY);
      return saved ? JSON.parse(saved) : DEFAULT_BRAND_KIT;
    } catch (e) { return DEFAULT_BRAND_KIT; }
  });

  useEffect(() => {
    localStorage.setItem(BRAND_KIT_STORAGE_KEY, JSON.stringify(brandKit));
  }, [brandKit]);

  useEffect(() => {
    const fetchFonts = async () => {
      try {
        const response = await apiFetch('/api/fonts');
        if (response.ok) {
          const data = await response.json();
          setCaptionFontOptions(data.fonts || []);
        }
      } catch (e) { console.error("Error fetching fonts", e); }
    };
    fetchFonts();
  }, []);

  // --- Computed Views ---
  const visibleProjects = useMemo(() => {
    let list = [...projects];
    if (jobId && (status === 'processing' || status === 'complete' || status === 'error') && !list.some((project) => project.job_id === jobId)) {
      list.unshift({
        job_id: jobId,
        title: processingMedia?.sourceLabel || 'Proyecto en proceso',
        status: status === 'complete' ? 'complete' : status === 'error' ? 'error' : 'processing',
        created_at: Date.now(),
        updated_at: Date.now(),
        expires_at: null,
        clip_count_actual: Array.isArray(results?.clips) ? results.clips.length : 0,
        clip_count_target: processingMedia?.clipCount,
        ratio: processingMedia?.aspectRatio || '9:16',
        source_kind: processingMedia?.type === 'url' ? 'youtube' : 'file',
        source_label: processingMedia?.sourceLabel || 'Archivo local',
        video_type: processingMedia?.generationMode === 'trailer' ? 'Super trailer' : 'Topic-clips',
        preview_video_url: results?.clips?.[0]?.video_url || '',
        thumbnail_url: '',
        clips: Array.isArray(results?.clips) ? results.clips.slice(0, 3) : [],
      });
    }
    if (projectFilter === 'favorites') list = list.filter(p => p.favorite);
    return list.sort((a, b) => new Date(b.updated_at || b.created_at || 0) - new Date(a.updated_at || a.created_at || 0));
  }, [jobId, processingMedia, projectFilter, projects, results?.clips, status]);

  const sortedClips = useMemo(() => {
    if (!results?.clips) return [];
    let list = [...results.clips];
    if (clipFilter === 'top') list = list.filter(c => c.virality_score >= 80);
    else if (clipFilter === 'medium') list = list.filter(c => (c.virality_score >= 65 && c.virality_score < 80));
    else if (clipFilter === 'low') list = list.filter(c => c.virality_score < 65);
    if (clipTagFilter !== 'all') list = list.filter(c => Array.isArray(c.topic_tags) && c.topic_tags.includes(clipTagFilter));
    if (clipSort === 'top') return list.sort((a, b) => b.virality_score - a.virality_score);
    if (clipSort === 'balanced') return list.sort((a, b) => a.start - b.start);
    return list;
  }, [results?.clips, clipSort, clipFilter, clipTagFilter]);

  const processingTimeline = useMemo(() => {
    const phase = results?.ui_phase || 'init';
    const progress = results?.progress || 0;
    let headline = "Preparando motor de clips...";
    if (status === 'processing') {
      if (phase === 'downloading') headline = "Descargando medios...";
      else if (phase === 'transcribing') headline = "Analizando diálogos...";
      else if (phase === 'segmenting') headline = "Detectando momentos...";
      else if (phase === 'rendering') headline = "Generando cortes...";
    } else if (status === 'complete') headline = "¡Clips listos!";
    else if (status === 'paused') headline = "Procesamiento pausado.";
    return {
      headline,
      progressPercent: status === 'complete' ? 100 : Math.max(5, progress),
      stepProgressLabel: status === 'processing' ? `${progress}%` : status === 'complete' ? 'Listo' : status === 'paused' ? 'Pausado' : 'Error'
    };
  }, [status, results]);

  const applyBatchStrategy = (strategy) => {
    setBatchStrategy(strategy);
    if (strategy === 'growth') { setBatchTopCount(5); setBatchIntervalMinutes(30); }
    else if (strategy === 'balanced') { setBatchTopCount(3); setBatchIntervalMinutes(60); }
  };

  const TopBar = () => (
    <header className="sticky top-0 z-50 w-full bg-white/80 dark:bg-slate-900/80 backdrop-blur-md border-b border-slate-200 dark:border-white/10">
      <div className="max-w-[1500px] mx-auto px-4 h-16 flex items-center justify-between">
        <div className="flex items-center gap-8">
          <div className="flex items-center gap-2 cursor-pointer" onClick={() => setActiveTab('home')}>
            <div className="w-9 h-9 bg-primary rounded-xl flex items-center justify-center shadow-lg shadow-primary/20">
              <Scissors className="text-white" size={20} />
            </div>
            <span className="text-xl font-black text-slate-900 dark:text-white hidden sm:block">OpenShorts</span>
          </div>
          <nav className="hidden md:flex items-center gap-1 bg-slate-100 dark:bg-white/5 p-1 rounded-xl">
            {[
              { id: 'home', label: 'Inicio', icon: Monitor },
              { id: 'projects', label: 'Proyectos', icon: LayoutDashboard },
              { id: 'settings', label: 'Ajustes', icon: Settings },
            ].map((tab) => (
              <button key={tab.id} onClick={() => { setActiveTab(tab.id); if (tab.id === 'projects') setProjectsViewMode('list'); }}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold transition-all ${activeTab === tab.id ? 'bg-white dark:bg-slate-800 text-primary shadow-sm' : 'text-slate-500 hover:text-slate-800'}`}>
                <tab.icon size={16} /> {tab.label}
              </button>
            ))}
          </nav>
        </div>
        <div className="flex items-center gap-3">
          <UserProfileSelector profiles={userProfiles} selectedUserId={uploadUserId} onSelect={setUploadUserId} />
        </div>
      </div>
    </header>
  );

  return (
    <div className="theme-light min-h-screen bg-slate-50 font-sans text-slate-900 transition-colors duration-300">
      <TopBar />
      <main className="max-w-[1500px] mx-auto w-full px-4 sm:px-6 py-6 transition-all duration-300">
        {activeTab === 'home' && (
          <HomeView
            status={status}
            handleProcess={handleProcess}
            apiKey={apiKey}
            homePrefillFile={homePrefillFile}
            setActiveTab={setActiveTab}
            setProjectsViewMode={setProjectsViewMode}
          />
        )}

        {activeTab === 'projects' && projectsViewMode === 'list' && (
          <ProjectsView
            projects={projects}
            visibleProjects={visibleProjects}
            projectFilter={projectFilter} setProjectFilter={setProjectFilter}
            favoriteProjectsCount={favoriteProjectsCount}
            removeAllProjects={removeAllProjects}
            openSavedProject={openSavedProject}
            jobId={jobId}
            projectTitleEditJobId={projectTitleEditJobId}
            projectTitleDraft={projectTitleDraft} setProjectTitleDraft={setProjectTitleDraft}
            saveProjectTitle={saveProjectTitle} cancelEditProjectTitle={cancelEditProjectTitle}
            beginEditProjectTitle={beginEditProjectTitle}
            toggleProjectFavorite={toggleProjectFavorite}
            removeProject={removeProject}
            setActiveTab={setActiveTab} setProjectsViewMode={setProjectsViewMode}
            formatProjectDate={formatProjectDate} outputModeLabel={outputModeLabel}
          />
        )}

        {activeTab === 'projects' && projectsViewMode === 'detail' && (
          <ProjectDetailView
            jobId={jobId} status={status} results={results} logs={logs} processingMedia={processingMedia}
            isPollingPaused={isPollingPaused} setIsPollingPaused={setIsPollingPaused}
            handleRetryJob={handleRetryJob} handleCancelJob={handleCancelJob} removeProject={removeProject}
            setLogs={setLogs} processingTimeline={processingTimeline} isRetryingJob={isRetryingJob}
            setActiveTab={setActiveTab} setProjectsViewMode={setProjectsViewMode}
            clipSearch={clipSearch} clipExporter={clipExporter}
            logsVisible={logsVisible} setLogsVisible={setLogsVisible}
            clipSort={clipSort} setClipSort={setClipSort}
            clipFilter={clipFilter} setClipFilter={setClipFilter}
            clipTagFilter={clipTagFilter} setClipTagFilter={setClipTagFilter}
            clipsViewMode={clipsViewMode} setClipsViewMode={setClipsViewMode}
            batchStrategy={batchStrategy} applyBatchStrategy={applyBatchStrategy}
            batchTopCount={batchTopCount} setBatchTopCount={setBatchTopCount}
            batchStartDelayMinutes={batchStartDelayMinutes} setBatchStartDelayMinutes={setBatchStartDelayMinutes}
            batchIntervalMinutes={batchIntervalMinutes} setBatchIntervalMinutes={setBatchIntervalMinutes}
            batchScope={batchScope} setBatchScope={setBatchScope}
            captionFontOptions={captionFontOptions} elevenLabsKey={elevenLabsKey}
            processingProjectName={projects.find(p => p.job_id === jobId)?.title || "Sin nombre"}
            processingSourceLabel={processingMedia?.sourceLabel || "Archivo"}
            visibleClips={sortedClips} sortedClips={sortedClips} 
            apiKey={apiKey} setApiKey={setApiKey} setElevenLabsKey={setElevenLabsKey} handleReset={handleReset}
            availableTags={clipSearch.availableTags} outputModeLabel={outputModeLabel}
            uploadPostKey={uploadPostKey} uploadUserId={uploadUserId} userProfiles={userProfiles}
          />
        )}

        {activeTab === 'settings' && (
          <SettingsView
            apiKey={apiKey} setApiKey={setApiKey}
            elevenLabsKey={elevenLabsKey} setElevenLabsKey={setElevenLabsKey}
            huggingfaceToken={huggingfaceToken} setHuggingfaceToken={setHuggingfaceToken}
            uploadPostKey={uploadPostKey} setUploadPostKey={setUploadPostKey}
            uploadUserId={uploadUserId} setUploadUserId={setUploadUserId}
            userProfiles={userProfiles} fetchUserProfiles={fetchUserProfiles}
            isFetchingUserProfiles={auth.isFetchingUserProfiles}
            userProfilesMessage={auth.userProfilesMessage}
            userProfilesMessageType={auth.userProfilesMessageType}
            apiBaseUrlInput={apiBaseUrlInput} setApiBaseUrlInput={setApiBaseUrlInput}
            apiBaseUrlActive={apiBaseUrlActive} apiBaseUrlMessage={apiBaseUrlMessage}
            apiBaseUrlMessageType={apiBaseUrlMessageType}
            isTestingApiBaseUrl={isTestingApiBaseUrl} handleSaveApiBaseUrl={handleSaveApiBaseUrl}
            runConnectivityCheck={runConnectivityCheck} 
            brandKit={brandKit} handleBrandKitFieldChange={(f, v) => setBrandKit(p => ({ ...p, [f]: v }))}
            captionFontOptions={captionFontOptions} connectivityStatus={connectivityStatus}
            isConnectivityChecking={isConnectivityChecking} handleReset={handleReset}
            setActiveTab={setActiveTab}
          />
        )}

      </main>

      {/* Persistence overlay for toasts */}
      <div className="fixed bottom-6 right-6 z-[9999] flex flex-col gap-2">
        {toasts.map(t => (
          <div key={t.id} className={`px-4 py-3 rounded-xl shadow-2xl border backdrop-blur-md animate-[slideInRight_0.3s_ease-out] ${
            t.type === 'error' ? 'bg-red-500/10 border-red-500/50 text-red-500' : 'bg-slate-900/90 border-white/10 text-white'
          }`}>
            {t.msg}
          </div>
        ))}
      </div>
    </div>
  );
}
