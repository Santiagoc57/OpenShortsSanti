import { useState, useEffect, useCallback, useRef } from 'react';
import { apiFetch } from '../config';

const pollJob = async (jobId) => {
  const res = await apiFetch(`/api/status/${jobId}`);
  if (!res.ok) throw new Error('Status check failed');
  return res.json();
};

const PROJECTS_EXPIRE_DAYS = 14;
const PROJECT_PREFS_STORAGE_KEY = 'openshortsProjectPrefsV1';
const PROCESSING_PROJECTS_STORAGE_KEY = 'openshortsProcessingProjectsV1';

const loadProjectPrefs = () => {
  if (typeof window === 'undefined') return {};
  try {
    return JSON.parse(window.localStorage.getItem(PROJECT_PREFS_STORAGE_KEY) || '{}') || {};
  } catch {
    return {};
  }
};

const saveProjectPrefs = (prefs) => {
  if (typeof window === 'undefined') return;
  window.localStorage.setItem(PROJECT_PREFS_STORAGE_KEY, JSON.stringify(prefs || {}));
};

const loadProcessingProjects = () => {
  if (typeof window === 'undefined') return [];
  try {
    const items = JSON.parse(window.localStorage.getItem(PROCESSING_PROJECTS_STORAGE_KEY) || '[]');
    return Array.isArray(items) ? items : [];
  } catch {
    return [];
  }
};

const saveProcessingProjects = (projects) => {
  if (typeof window === 'undefined') return;
  const safe = (Array.isArray(projects) ? projects : [])
    .filter((project) => project?.job_id)
    .slice(0, 80);
  window.localStorage.setItem(PROCESSING_PROJECTS_STORAGE_KEY, JSON.stringify(safe));
};

const upsertProcessingProject = (project) => {
  if (!project?.job_id) return;
  const existing = loadProcessingProjects().filter((item) => item.job_id !== project.job_id);
  saveProcessingProjects([project, ...existing]);
};

const removeProcessingProject = (jobId) => {
  if (!jobId) return;
  saveProcessingProjects(loadProcessingProjects().filter((item) => item.job_id !== jobId));
};

const clearLocalProcessingProjects = () => {
  saveProcessingProjects(loadProcessingProjects().filter((item) => !isLocalProjectId(item.job_id)));
};

const mergeProjectLists = (primary, fallback) => {
  const out = [];
  const seen = new Set();
  [...(primary || []), ...(fallback || [])].forEach((project) => {
    if (!project?.job_id || seen.has(project.job_id)) return;
    seen.add(project.job_id);
    out.push(project);
  });
  return out;
};

const isLocalProjectId = (value) => String(value || '').startsWith('local-');

const mergeProjectPrefs = (projects) => {
  const prefs = loadProjectPrefs();
  return (projects || []).map((project) => {
    const pref = prefs[project.job_id] || {};
    return {
      ...project,
      title: pref.title || project.title,
      favorite: Boolean(pref.favorite),
    };
  });
};

const buildProcessBody = (input) => {
  const formData = new FormData();
  if (input.type === 'url') {
    formData.append('url', input.payload);
  } else {
    formData.append('file', input.payload);
  }

  const fields = {
    language: input.language,
    max_clips: input.clipCount,
    whisper_backend: input.whisperBackend,
    whisper_model: input.whisperModel,
    word_timestamps: input.wordTimestamps,
    ffmpeg_preset: input.ffmpegPreset,
    ffmpeg_crf: input.ffmpegCrf,
    aspect_ratio: input.aspectRatio,
    clip_length_target: input.clipLengthTarget,
    style_template: input.styleTemplate,
    content_profile: input.contentPreset,
    llm_provider: input.llm_provider,
    llm_model: input.llm_model,
    generation_mode: input.generation_mode,
    build_trailer: input.build_trailer,
    trailer_fragments_target: input.trailer_fragments_target,
    ownership_attested: input.ownership_attested,
    enable_diarization: input.enableDiarization,
  };

  Object.entries(fields).forEach(([key, value]) => {
    if (value === undefined || value === null || value === '') return;
    formData.append(key, String(value));
  });
  return formData;
};

export const useJobManager = (showToast, auth) => {
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState('idle'); // idle, processing, complete, error
  const [results, setResults] = useState(null);
  const [processUiPhase, setProcessUiPhase] = useState('idle');
  const [isPollingPaused, setIsPollingPaused] = useState(false);
  const [isRetryingJob, setIsRetryingJob] = useState(false);
  const [logs, setLogs] = useState([]);
  const [processingMedia, setProcessingMedia] = useState(null);
  const [projects, setProjects] = useState(() => mergeProjectPrefs(loadProcessingProjects()));
  const [projectTitleEditJobId, setProjectTitleEditJobId] = useState(null);
  const [projectTitleDraft, setProjectTitleDraft] = useState('');

  const refreshProjects = useCallback(async () => {
    try {
      const res = await apiFetch('/api/projects/history?limit=80&refresh=true');
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      const backendProjects = Array.isArray(data.projects) ? data.projects : [];
      const localProcessing = loadProcessingProjects();
      const merged = mergeProjectLists(backendProjects, localProcessing);
      setProjects(mergeProjectPrefs(merged));
    } catch (error) {
      console.error('Failed to refresh projects', error);
      setProjects((prev) => mergeProjectPrefs(mergeProjectLists(prev, loadProcessingProjects())));
    }
  }, []);

  useEffect(() => {
    refreshProjects();
  }, [refreshProjects]);

  const applyPolledJobData = useCallback((targetJobId, data) => {
    if (targetJobId !== jobId) return;

    // Update Project List (This ideally would be in a useProjects hook, 
    // but for now we'll handle the logic and let App.jsx update its state)
    // We'll return the 'projectUpdate' object.

    if (data.result) {
      setResults(data.result);
    }

    if (data.status === 'completed') {
      setProcessUiPhase('complete');
      setStatus('complete');
      removeProcessingProject(targetJobId);
      refreshProjects();
      return;
    }
    if (data.status === 'failed') {
      setProcessUiPhase('error');
      setStatus('error');
      removeProcessingProject(targetJobId);
      const errorMsg = data.error || (data.logs && data.logs.length > 0 ? data.logs[data.logs.length - 1] : "Proceso fallido");
      setLogs((prev) => [...prev, "Error: " + errorMsg]);
      refreshProjects();
      return;
    }
    if (data.status === 'paused') {
      setProcessUiPhase('paused');
      setStatus('paused');
      setLogs(data.logs || []);
      removeProcessingProject(targetJobId);
      refreshProjects();
      return;
    }

    if (data.logs) setLogs(data.logs);
    const readyClips = Array.isArray(data?.result?.clips) ? data.result.clips.length : 0;
    if (readyClips > 0) {
      setProcessUiPhase('rendering');
    } else if (data.status === 'queued') {
      setProcessUiPhase('queued');
    } else {
      setProcessUiPhase('running');
    }
  }, [jobId, refreshProjects]);

  useEffect(() => {
    if (!(status === 'processing' || status === 'completed') || !jobId || isLocalProjectId(jobId) || isPollingPaused) return;

    let timeoutId;
    let delay = 2000;
    const MAX_DELAY = 10000;
    const RAMP_AFTER_MS = 30000;
    const startTs = Date.now();
    let stopped = false;

    const tick = async () => {
      if (stopped) return;
      try {
        const data = await pollJob(jobId);
        if (import.meta.env.DEV) console.log('Job status:', data);
        applyPolledJobData(jobId, data);
      } catch (e) {
        console.error('Polling error', e);
      }
      if (stopped) return;
      const elapsed = Date.now() - startTs;
      if (elapsed > RAMP_AFTER_MS) {
        delay = Math.min(MAX_DELAY, delay * 1.4);
      }
      timeoutId = setTimeout(tick, delay);
    };

    timeoutId = setTimeout(tick, delay);
    return () => {
      stopped = true;
      clearTimeout(timeoutId);
    };
  }, [status, jobId, isPollingPaused, applyPolledJobData]);

  const handleReset = useCallback(() => {
    clearLocalProcessingProjects();
    setJobId(null);
    setStatus('idle');
    setResults(null);
    setProcessUiPhase('idle');
    setIsPollingPaused(false);
    setLogs([]);
    setProcessingMedia(null);
  }, []);

  const handleProcess = useCallback(async (input, headers = {}) => {
    const localJobId = `local-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const localProject = {
      job_id: localJobId,
      title: input.type === 'url' ? input.payload : input.payload?.name || 'Proyecto en proceso',
      status: 'processing',
      backend_status: 'local',
      created_at: Date.now(),
      updated_at: Date.now(),
      expires_at: null,
      clip_count: 0,
      clip_count_actual: 0,
      clip_count_target: input.clipCount,
      ratio: input.aspectRatio || '9:16',
      source_kind: input.type === 'url' ? 'youtube' : 'file',
      source_label: input.type === 'url' ? 'YouTube' : input.payload?.name || 'Archivo local',
      video_type: input.generation_mode === 'trailer' ? 'Super trailer' : 'Topic-clips',
      preview_video_url: '',
      thumbnail_url: '',
      clips: [],
    };
    setJobId(localJobId);
    setStatus('processing');
    setProcessUiPhase('queued');
    setResults(null);
    setLogs(['Preparando proyecto...']);
    setProcessingMedia({
      type: input.type,
      sourceLabel: input.type === 'url' ? 'YouTube' : input.payload?.name || 'Archivo local',
      aspectRatio: input.aspectRatio || '9:16',
      clipCount: input.clipCount,
      generationMode: input.generation_mode || 'clips',
    });
    upsertProcessingProject(localProject);
    setProjects((prev) => mergeProjectPrefs(mergeProjectLists([localProject], prev)));

    try {
      const res = await apiFetch('/api/process', {
        method: 'POST',
        headers,
        body: buildProcessBody(input),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setJobId(data.job_id);
      removeProcessingProject(localJobId);
      const processingProject = {
        job_id: data.job_id,
        title: input.type === 'url' ? input.payload : input.payload?.name || 'Proyecto en proceso',
        status: 'processing',
        backend_status: 'queued',
        created_at: Date.now(),
        updated_at: Date.now(),
        expires_at: null,
        clip_count: 0,
        clip_count_actual: 0,
        clip_count_target: input.clipCount,
        ratio: input.aspectRatio || '9:16',
        source_kind: input.type === 'url' ? 'youtube' : 'file',
        source_label: input.type === 'url' ? 'YouTube' : input.payload?.name || 'Archivo local',
        video_type: input.generation_mode === 'trailer' ? 'Super trailer' : 'Topic-clips',
        preview_video_url: '',
        thumbnail_url: '',
        clips: [],
      };
      upsertProcessingProject(processingProject);
      setProjects((prev) => mergeProjectPrefs(mergeProjectLists(
        [processingProject],
        prev.filter((project) => project.job_id !== localJobId)
      )));
      setLogs((prev) => [...prev, `Proyecto encolado: ${data.job_id}`]);
      await refreshProjects();
    } catch (error) {
      setStatus('error');
      setProcessUiPhase('error');
      const failedLocalProject = {
        ...localProject,
        status: 'error',
        backend_status: 'local_error',
        updated_at: Date.now(),
      };
      upsertProcessingProject(failedLocalProject);
      setProjects((prev) => mergeProjectPrefs(mergeProjectLists(
        [failedLocalProject],
        prev.filter((project) => project.job_id !== localJobId)
      )));
      setLogs((prev) => [...prev, `Error: ${error.message}`]);
      showToast(`No se pudo iniciar el proyecto: ${error.message}`, 'error');
    }
  }, [refreshProjects, showToast]);

  const handleRetryJob = useCallback(async () => {
    if (!jobId || isLocalProjectId(jobId)) return;
    setIsRetryingJob(true);
    setIsPollingPaused(false);
    setProcessUiPhase('queued');
    
    try {
      const res = await apiFetch(`/api/retry/${jobId}`, { method: 'POST' });
      if (!res.ok) throw new Error(await res.text());
      setResults(null);
      setStatus('processing');
      setProcessUiPhase('queued');
      setLogs((prev) => [...prev, 'Reintento manual encolado.']);
    } catch (e) {
      setLogs((prev) => [...prev, `Reintento fallido: ${e.message}`]);
      showToast(`Error al reintentar: ${e.message}`, 'error');
    } finally {
      setIsRetryingJob(false);
    }
  }, [jobId, showToast]);

  const handleCancelJob = useCallback(async () => {
    if (!jobId) return;
    if (isLocalProjectId(jobId)) {
      setIsPollingPaused(true);
      setStatus('paused');
      setProcessUiPhase('paused');
      setLogs((prev) => [...prev, 'Proyecto local pausado/cancelado.']);
      removeProcessingProject(jobId);
      setProjects((prev) => prev.map((project) => (
        project.job_id === jobId ? { ...project, status: 'paused', backend_status: 'local_paused' } : project
      )));
      return;
    }
    try {
      const res = await apiFetch(`/api/projects/${encodeURIComponent(jobId)}/cancel`, { method: 'POST' });
      if (!res.ok) throw new Error(await res.text());
      setIsPollingPaused(true);
      setStatus('paused');
      setProcessUiPhase('paused');
      setLogs((prev) => [...prev, 'Proyecto pausado/cancelado.']);
      removeProcessingProject(jobId);
      await refreshProjects();
    } catch (e) {
      setLogs((prev) => [...prev, `No se pudo pausar: ${e.message}`]);
      showToast(`No se pudo pausar: ${e.message}`, 'error');
    }
  }, [jobId, refreshProjects, showToast]);

  const openSavedProject = useCallback(async (project) => {
    if (!project?.job_id) return;
    setJobId(project.job_id);
    setStatus(project.status === 'complete' ? 'complete' : project.status === 'error' ? 'error' : 'processing');
    setProcessUiPhase(project.status === 'complete' ? 'complete' : project.status === 'error' ? 'error' : 'running');
    setProcessingMedia({
      sourceLabel: project.source_label || 'Archivo',
      aspectRatio: project.ratio || '9:16',
    });
    setLogs([]);
    try {
      const res = await apiFetch(`/api/projects/clips/${encodeURIComponent(project.job_id)}?refresh=true`);
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setResults(data.result || { clips: data.clips || [] });
    } catch (error) {
      setResults({ clips: project.clips || [] });
      setLogs((prev) => [...prev, `No se pudieron cargar todos los clips: ${error.message}`]);
    }
  }, []);

  const removeProject = useCallback(async (targetJobId) => {
    if (!targetJobId) return;
    if (!isLocalProjectId(targetJobId)) {
      try {
        await apiFetch(`/api/projects/${encodeURIComponent(targetJobId)}`, { method: 'DELETE' });
      } catch (error) {
        console.error('Failed to delete backend project', error);
      }
    }
    setProjects((prev) => prev.filter((project) => project.job_id !== targetJobId));
    removeProcessingProject(targetJobId);
    if (targetJobId === jobId) {
      clearLocalProcessingProjects();
    }
    const prefs = loadProjectPrefs();
    delete prefs[targetJobId];
    saveProjectPrefs(prefs);
    if (targetJobId === jobId) handleReset();
    return true;
  }, [handleReset, jobId]);

  const removeAllProjects = useCallback(async () => {
    const current = [...projects];
    for (const project of current) {
      // Keep this sequential to avoid hammering local disk/S3 deletes.
      await removeProject(project.job_id);
    }
  }, [projects, removeProject]);

  const toggleProjectFavorite = useCallback((targetJobId) => {
    const prefs = loadProjectPrefs();
    const current = prefs[targetJobId] || {};
    prefs[targetJobId] = { ...current, favorite: !current.favorite };
    saveProjectPrefs(prefs);
    setProjects((prev) => mergeProjectPrefs(prev));
  }, []);

  const beginEditProjectTitle = useCallback((project) => {
    setProjectTitleEditJobId(project.job_id);
    setProjectTitleDraft(project.title || '');
  }, []);

  const cancelEditProjectTitle = useCallback(() => {
    setProjectTitleEditJobId(null);
    setProjectTitleDraft('');
  }, []);

  const saveProjectTitle = useCallback((targetJobId) => {
    const prefs = loadProjectPrefs();
    const current = prefs[targetJobId] || {};
    prefs[targetJobId] = { ...current, title: projectTitleDraft.trim() };
    saveProjectPrefs(prefs);
    setProjects((prev) => mergeProjectPrefs(prev));
    cancelEditProjectTitle();
  }, [cancelEditProjectTitle, projectTitleDraft]);

  const favoriteProjectsCount = projects.filter((project) => project.favorite).length;

  return {
    jobId, setJobId,
    status, setStatus,
    results, setResults,
    processUiPhase, setProcessUiPhase,
    isPollingPaused, setIsPollingPaused,
    isRetryingJob,
    logs, setLogs,
    processingMedia, setProcessingMedia,
    handleProcess,
    handleReset,
    handleRetryJob,
    handleCancelJob,
    applyPolledJobData,
    projects, setProjects,
    removeProject,
    removeAllProjects,
    favoriteProjectsCount,
    toggleProjectFavorite,
    saveProjectTitle,
    beginEditProjectTitle,
    cancelEditProjectTitle,
    projectTitleEditJobId,
    projectTitleDraft,
    setProjectTitleDraft,
    openSavedProject,
    refreshProjects
  };
};
