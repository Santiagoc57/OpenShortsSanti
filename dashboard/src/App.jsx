import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { Upload, FileVideo, Sparkles, Youtube, Instagram, Share2, LogOut, ChevronDown, ChevronRight, Check, Activity, LayoutDashboard, Settings, PlusCircle, History, Menu, X, Terminal, Shield, Search, Sun, Moon, MoreHorizontal, Heart, Link2, Pencil, Scissors, RefreshCw, Pause, Play, Trash2, ArrowLeft } from 'lucide-react';
import KeyInput from './components/KeyInput';
import MediaInput from './components/MediaInput';
import ResultCard from './components/ResultCard';
import ClipStudioModal from './components/ClipStudioModal';
import { apiFetch, getApiUrl, getApiBaseUrl, normalizeApiBaseUrl, setApiBaseUrl } from './config';

// Enhanced "Encryption" using XOR + Base64 with a Salt
// This is better than plain Base64 but still client-side.
const SECRET_KEY = import.meta.env.VITE_ENCRYPTION_KEY || "OpenShorts-Static-Salt-Change-Me";
const ENCRYPTION_PREFIX = "ENC:";

const encrypt = (text) => {
  if (!text) return '';
  try {
    const xor = text.split('').map((c, i) =>
      String.fromCharCode(c.charCodeAt(0) ^ SECRET_KEY.charCodeAt(i % SECRET_KEY.length))
    ).join('');
    return ENCRYPTION_PREFIX + btoa(xor);
  } catch (e) {
    console.error("Encryption failed", e);
    return text;
  }
};

const decrypt = (text) => {
  if (!text) return '';
  if (text.startsWith(ENCRYPTION_PREFIX)) {
    try {
      const raw = text.slice(ENCRYPTION_PREFIX.length);
      // Check if it's plain base64 or our custom XOR (simple try)
      const xor = atob(raw);
      const result = xor.split('').map((c, i) =>
        String.fromCharCode(c.charCodeAt(0) ^ SECRET_KEY.charCodeAt(i % SECRET_KEY.length))
      ).join('');
      return result;
    } catch (e) {
      // Fallback if decryption fails (might be old plain text)
      return '';
    }
  }
  // Backward compatibility: If no prefix, assume old plain text (or return empty if you want to force re-login)
  // For migration: Return text as is, so it populates the field, and next save will encrypt it.
  return text;
};

// Simple TikTok icon sine Lucide might not have it or it varies
const TikTokIcon = ({ size = 16, className = "" }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" className={className}>
    <path d="M19.589 6.686a4.793 4.793 0 0 1-3.77-4.245V2h-3.445v13.672a2.896 2.896 0 0 1-5.201 1.743l-.002-.001.002.001a2.895 2.895 0 0 1 3.183-4.51v-3.5a6.329 6.329 0 0 0-5.394 10.692 6.33 6.33 0 0 0 10.857-4.424V8.687a8.182 8.182 0 0 0 4.773 1.526V6.79a4.831 4.831 0 0 1-1.003-.104z" />
  </svg>
);

const UserProfileSelector = ({ profiles, selectedUserId, onSelect }) => {
  const [isOpen, setIsOpen] = useState(false);

  if (!profiles || profiles.length === 0) return null;

  const selectedProfile = profiles.find(p => p.username === selectedUserId) || profiles[0];

  return (
    <div className="relative z-50">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center justify-between bg-white dark:bg-surface border border-slate-200 dark:border-white/10 rounded-lg px-3 py-2 text-sm text-slate-700 dark:text-zinc-300 hover:bg-slate-50 dark:hover:bg-white/5 transition-colors min-w-[180px]"
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
                      {/* Status indicators */}
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

// Mock polling function
const pollJob = async (jobId) => {
  const res = await apiFetch(`/api/status/${jobId}`);
  if (!res.ok) throw new Error('Status check failed');
  return res.json();
};

const BRAND_KIT_STORAGE_KEY = 'brandKitV1';
const PROJECTS_STORAGE_KEY = 'openshortsProjectsV1';
const PROJECTS_EXPIRE_DAYS = 14;
const DEFAULT_BRAND_KIT = {
  name: 'Predeterminado',
  subtitle_position: 'bottom',
  subtitle_font_family: 'Impact',
  subtitle_font_size: 24,
  subtitle_font_color: '#FFFFFF',
  subtitle_stroke_color: '#000000',
  subtitle_stroke_width: 3,
  subtitle_bold: true,
  subtitle_box_color: '#000000',
  subtitle_box_opacity: 60
};

const normalizeBrandKit = (raw) => {
  const src = raw && typeof raw === 'object' ? raw : {};
  const asNum = (v, fallback, min, max) => {
    const n = Number(v);
    if (!Number.isFinite(n)) return fallback;
    return Math.max(min, Math.min(max, Math.round(n)));
  };
  return {
    name: String(src.name || DEFAULT_BRAND_KIT.name).slice(0, 48),
    subtitle_position: ['top', 'middle', 'bottom'].includes(src.subtitle_position) ? src.subtitle_position : DEFAULT_BRAND_KIT.subtitle_position,
    subtitle_font_family: String(src.subtitle_font_family || DEFAULT_BRAND_KIT.subtitle_font_family).slice(0, 48),
    subtitle_font_size: asNum(src.subtitle_font_size, DEFAULT_BRAND_KIT.subtitle_font_size, 12, 84),
    subtitle_font_color: String(src.subtitle_font_color || DEFAULT_BRAND_KIT.subtitle_font_color),
    subtitle_stroke_color: String(src.subtitle_stroke_color || DEFAULT_BRAND_KIT.subtitle_stroke_color),
    subtitle_stroke_width: asNum(src.subtitle_stroke_width, DEFAULT_BRAND_KIT.subtitle_stroke_width, 0, 8),
    subtitle_bold: typeof src.subtitle_bold === 'boolean' ? src.subtitle_bold : DEFAULT_BRAND_KIT.subtitle_bold,
    subtitle_box_color: String(src.subtitle_box_color || DEFAULT_BRAND_KIT.subtitle_box_color),
    subtitle_box_opacity: asNum(src.subtitle_box_opacity, DEFAULT_BRAND_KIT.subtitle_box_opacity, 0, 100)
  };
};

const formatProjectDate = (isoString) => {
  if (!isoString) return '-';
  try {
    return new Date(isoString).toLocaleDateString('es-ES', { month: 'short', day: 'numeric' });
  } catch (_) {
    return '-';
  }
};

const projectVideoTypeLabel = (contentPreset) => {
  const key = String(contentPreset || '').toLowerCase();
  if (key === 'podcast') return 'Resumen y highlights';
  if (key === 'tutorial') return 'Tutorial-clips';
  if (key === 'entrevista') return 'Momentos de entrevista';
  return 'Topic-clips';
};

function App() {
  const getInitialClipSort = () => {
    const stored = localStorage.getItem('clipSortPreset');
    return ['top', 'balanced', 'safe'].includes(stored) ? stored : 'top';
  };
  const getInitialClipFilter = () => {
    const stored = localStorage.getItem('clipFilterPreset');
    return ['all', 'top', 'medium', 'low'].includes(stored) ? stored : 'all';
  };
  const getInitialTagFilter = () => {
    const stored = localStorage.getItem('clipTagFilterPreset');
    return stored || 'all';
  };
  const getInitialBatchTopCount = () => {
    const stored = Number(localStorage.getItem('batchTopCountPreset'));
    if (!Number.isFinite(stored)) return 3;
    return Math.max(1, Math.min(10, Math.round(stored)));
  };
  const getInitialBatchStartDelay = () => {
    const stored = Number(localStorage.getItem('batchStartDelayPreset'));
    if (!Number.isFinite(stored)) return 15;
    return Math.max(0, Math.min(180, Math.round(stored)));
  };
  const getInitialBatchInterval = () => {
    const stored = Number(localStorage.getItem('batchIntervalPreset'));
    if (!Number.isFinite(stored)) return 60;
    return Math.max(5, Math.min(720, Math.round(stored)));
  };
  const getInitialBatchScope = () => {
    const stored = localStorage.getItem('batchScopePreset');
    return ['visible', 'global'].includes(stored) ? stored : 'visible';
  };
  const getInitialBatchStrategy = () => {
    const stored = localStorage.getItem('batchStrategyPreset');
    return ['growth', 'balanced', 'conservative', 'custom'].includes(stored) ? stored : 'balanced';
  };
  const getInitialClipSearchModePreset = () => {
    const stored = localStorage.getItem('clipSearchModePreset');
    return ['exact', 'balanced', 'broad'].includes(stored) ? stored : 'balanced';
  };
  const getInitialTheme = () => {
    const stored = localStorage.getItem('uiTheme');
    if (stored === 'light' || stored === 'dark') return stored;
    return 'light';
  };
  const getInitialBrandKit = () => {
    try {
      const raw = localStorage.getItem(BRAND_KIT_STORAGE_KEY);
      if (!raw) return DEFAULT_BRAND_KIT;
      return normalizeBrandKit(JSON.parse(raw));
    } catch (_) {
      return DEFAULT_BRAND_KIT;
    }
  };
  const getInitialProjects = () => {
    try {
      const raw = localStorage.getItem(PROJECTS_STORAGE_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];
      return parsed
        .filter((item) => item && typeof item === 'object' && item.job_id)
        .slice(0, 40);
    } catch (_) {
      return [];
    }
  };

  const [apiKey, setApiKey] = useState(localStorage.getItem('gemini_key') || '');
  // Social API State - Load encrypted or plain
  const [uploadPostKey, setUploadPostKey] = useState(() => {
    const stored = localStorage.getItem('uploadPostKey_v3');
    if (stored) return decrypt(stored);
    return '';
  });

  const [uploadUserId, setUploadUserId] = useState(() => localStorage.getItem('uploadUserId') || '');
  const [userProfiles, setUserProfiles] = useState([]); // List of {username, connected: []}
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState('idle'); // idle, processing, complete, error
  const [results, setResults] = useState(null);
  const [clipSort, setClipSort] = useState(getInitialClipSort); // top, balanced, safe
  const [clipFilter, setClipFilter] = useState(getInitialClipFilter); // all, top, medium, low
  const [clipTagFilter, setClipTagFilter] = useState(getInitialTagFilter); // all | tag
  const [batchTopCount, setBatchTopCount] = useState(getInitialBatchTopCount);
  const [batchStartDelayMinutes, setBatchStartDelayMinutes] = useState(getInitialBatchStartDelay);
  const [batchIntervalMinutes, setBatchIntervalMinutes] = useState(getInitialBatchInterval);
  const [batchScope, setBatchScope] = useState(getInitialBatchScope); // visible | global
  const [batchStrategy, setBatchStrategy] = useState(getInitialBatchStrategy); // growth | balanced | conservative | custom
  const [isBatchScheduling, setIsBatchScheduling] = useState(false);
  const [batchScheduleReport, setBatchScheduleReport] = useState(null);
  const [isExportingPack, setIsExportingPack] = useState(false);
  const [packExportReport, setPackExportReport] = useState(null);
  const [clipSearchQuery, setClipSearchQuery] = useState('');
  const [isSearchingClips, setIsSearchingClips] = useState(false);
  const [clipSearchResults, setClipSearchResults] = useState([]);
  const [clipSearchKeywords, setClipSearchKeywords] = useState([]);
  const [clipSearchPhrases, setClipSearchPhrases] = useState([]);
  const [clipSearchChapters, setClipSearchChapters] = useState([]);
  const [clipSearchSpeakers, setClipSearchSpeakers] = useState([]);
  const [clipHybridShortlist, setClipHybridShortlist] = useState([]);
  const [clipSearchProvider, setClipSearchProvider] = useState('local');
  const [clipSearchMode, setClipSearchMode] = useState('topic');
  const [clipSearchRelaxed, setClipSearchRelaxed] = useState(false);
  const [clipSearchScope, setClipSearchScope] = useState(null);
  const [clipSearchChapterFilter, setClipSearchChapterFilter] = useState('-1');
  const [clipSearchStartTime, setClipSearchStartTime] = useState('');
  const [clipSearchEndTime, setClipSearchEndTime] = useState('');
  const [clipSearchSpeakerFilter, setClipSearchSpeakerFilter] = useState('all');
  const [transcriptSegments, setTranscriptSegments] = useState([]);
  const [transcriptFilter, setTranscriptFilter] = useState('');
  const [transcriptTotal, setTranscriptTotal] = useState(0);
  const [isLoadingTranscript, setIsLoadingTranscript] = useState(false);
  const [transcriptError, setTranscriptError] = useState(null);
  const [transcriptHasSpeakers, setTranscriptHasSpeakers] = useState(false);
  const [clipSearchModePreset, setClipSearchModePreset] = useState(getInitialClipSearchModePreset);
  const [clipSearchError, setClipSearchError] = useState(null);
  const [logs, setLogs] = useState([]);
  const [logsVisible, setLogsVisible] = useState(false);
  const [isPollingPaused, setIsPollingPaused] = useState(false);
  const [processUiPhase, setProcessUiPhase] = useState('idle'); // idle | uploading | queued | running | rendering | finalizing | complete | error
  const [isRetryingJob, setIsRetryingJob] = useState(false);
  const [processingMedia, setProcessingMedia] = useState(null);
  const [activeTab, setActiveTab] = useState('home'); // home, projects, settings
  const [theme, setTheme] = useState(getInitialTheme); // dark | light
  const [brandKit, setBrandKit] = useState(getInitialBrandKit);
  const [apiBaseUrlInput, setApiBaseUrlInput] = useState(() => getApiBaseUrl() || '');
  const [apiBaseUrlActive, setApiBaseUrlActive] = useState(() => getApiBaseUrl() || '');
  const [apiBaseUrlMessage, setApiBaseUrlMessage] = useState('');
  const [apiBaseUrlMessageType, setApiBaseUrlMessageType] = useState('neutral');
  const [isTestingApiBaseUrl, setIsTestingApiBaseUrl] = useState(false);
  const [connectivityStatus, setConnectivityStatus] = useState({
    api: 'checking',
    tunnel: 'idle',
    usingNgrok: false,
    lastCheckedAt: null,
    error: ''
  });
  const [isConnectivityChecking, setIsConnectivityChecking] = useState(false);
  const connectivityCheckInFlight = useRef(false);
  const [projects, setProjects] = useState(getInitialProjects);
  const [projectsViewMode, setProjectsViewMode] = useState('list'); // list | detail
  const [studioContext, setStudioContext] = useState(null);
  const [pendingReturnClipIndex, setPendingReturnClipIndex] = useState(null);
  const [focusedClipIndex, setFocusedClipIndex] = useState(null);
  const [projectFilter, setProjectFilter] = useState('all');
  const [projectMenuJobId, setProjectMenuJobId] = useState(null);
  const [projectTitleEditJobId, setProjectTitleEditJobId] = useState(null);
  const [projectTitleDraft, setProjectTitleDraft] = useState('');
  const pollingPauseBeforeStudioRef = useRef(false);
  const clipCardRefs = useRef(new Map());
  
  // Sync state for original video playback
  const [syncedTime, setSyncedTime] = useState(0);
  const [isSyncedPlaying, setIsSyncedPlaying] = useState(false);
  const [syncTrigger, setSyncTrigger] = useState(0);

  const handleClipPlay = (startTime) => {
    setSyncedTime(startTime);
    setIsSyncedPlaying(true);
    setSyncTrigger(prev => prev + 1);
  };

  const handleClipPause = () => {
    setIsSyncedPlaying(false);
  };

  const runConnectivityCheck = useCallback(async () => {
    if (connectivityCheckInFlight.current) return;
    connectivityCheckInFlight.current = true;
    setIsConnectivityChecking(true);

    const apiBase = getApiBaseUrl() || '';
    const usingNgrok = apiBase.includes('ngrok');
    let nextApi = 'error';
    let nextTunnel = usingNgrok ? 'error' : 'local';
    let nextError = '';
    let timeoutId;

    try {
      const controller = new AbortController();
      timeoutId = setTimeout(() => controller.abort(), 9000);

      const res = await apiFetch(`/api/status/__healthcheck__?ts=${Date.now()}`, {
        method: 'GET',
        cache: 'no-store',
        signal: controller.signal
      });

      // 404 también confirma que la API y el túnel están vivos.
      if (res.ok || res.status === 404) {
        nextApi = 'ok';
        nextTunnel = usingNgrok ? 'ok' : 'local';
      } else {
        nextError = `HTTP ${res.status}`;
      }
    } catch (e) {
      nextError = e?.name === 'AbortError' ? 'timeout' : String(e?.message || 'sin conexión');
    } finally {
      if (timeoutId) clearTimeout(timeoutId);
      setConnectivityStatus({
        api: nextApi,
        tunnel: nextTunnel,
        usingNgrok,
        lastCheckedAt: new Date().toISOString(),
        error: nextError
      });
      setIsConnectivityChecking(false);
      connectivityCheckInFlight.current = false;
    }
  }, []);

  const formatTimelineTime = (seconds) => {
    const val = Number(seconds);
    if (!Number.isFinite(val) || val < 0) return '0:00';
    const total = Math.floor(val);
    const mins = Math.floor(total / 60);
    const secs = total % 60;
    return `${mins}:${String(secs).padStart(2, '0')}`;
  };

  const strategyLabel = (value) => ({
    growth: 'Crecimiento',
    balanced: 'Balanceada',
    conservative: 'Conservadora',
    custom: 'Personalizada'
  }[value] || value || '-');

  const scopeLabel = (value) => ({
    visible: 'Visible',
    global: 'Global'
  }[value] || value || '-');

  const queueStatusLabel = (value) => ({
    scheduled: 'programado',
    failed: 'fallido'
  }[value] || value || '-');

  const projectStatusLabel = (value) => ({
    processing: 'Procesando',
    complete: 'Completado',
    error: 'Error'
  }[value] || 'Procesando');

  const projectSourceBadgeClass = (value) => {
    if (value === 'youtube') return 'bg-red-100 text-red-700 dark:bg-red-900/20 dark:text-red-300';
    if (value === 'url') return 'bg-sky-100 text-sky-700 dark:bg-sky-900/20 dark:text-sky-300';
    return 'bg-amber-100 text-amber-800 dark:bg-zinc-700 dark:text-zinc-200';
  };

  useEffect(() => {
    // Encrypt Gemini Key too for consistency if desired, but user asked specifically about Social integration not saving well.
    // For now keeping gemini plain for compatibility unless requested.
    if (apiKey) localStorage.setItem('gemini_key', apiKey);
  }, [apiKey]);

  useEffect(() => {
    if (uploadPostKey) {
      localStorage.setItem('uploadPostKey_v3', encrypt(uploadPostKey));
    }
    if (uploadUserId) {
      localStorage.setItem('uploadUserId', uploadUserId);
    }
  }, [uploadPostKey, uploadUserId]);

  useEffect(() => {
    if (uploadPostKey && userProfiles.length === 0) {
      fetchUserProfiles();
    }
  }, [uploadPostKey]);

  useEffect(() => {
    localStorage.setItem('uiTheme', theme);
    const root = document.documentElement;
    if (theme === 'light') {
      root.classList.add('theme-light');
      return;
    }
    root.classList.remove('theme-light');
  }, [theme]);

  useEffect(() => {
    localStorage.setItem(BRAND_KIT_STORAGE_KEY, JSON.stringify(normalizeBrandKit(brandKit)));
  }, [brandKit]);

  useEffect(() => {
    localStorage.setItem(PROJECTS_STORAGE_KEY, JSON.stringify((projects || []).slice(0, 40)));
  }, [projects]);

  useEffect(() => {
    if (typeof window === 'undefined') return undefined;
    const syncApiBaseUrl = () => {
      const current = getApiBaseUrl() || '';
      setApiBaseUrlActive(current);
    };
    syncApiBaseUrl();
    window.addEventListener('openshorts-api-base-url-changed', syncApiBaseUrl);
    return () => window.removeEventListener('openshorts-api-base-url-changed', syncApiBaseUrl);
  }, []);

  useEffect(() => {
    runConnectivityCheck();
    const interval = setInterval(() => runConnectivityCheck(), 3 * 60 * 1000);
    return () => clearInterval(interval);
  }, [runConnectivityCheck, apiBaseUrlActive]);

  useEffect(() => {
    localStorage.setItem('clipSortPreset', clipSort);
  }, [clipSort]);

  useEffect(() => {
    localStorage.setItem('clipFilterPreset', clipFilter);
  }, [clipFilter]);

  useEffect(() => {
    localStorage.setItem('clipTagFilterPreset', clipTagFilter);
  }, [clipTagFilter]);

  useEffect(() => {
    localStorage.setItem('batchTopCountPreset', String(batchTopCount));
  }, [batchTopCount]);

  useEffect(() => {
    localStorage.setItem('batchStartDelayPreset', String(batchStartDelayMinutes));
  }, [batchStartDelayMinutes]);

  useEffect(() => {
    localStorage.setItem('batchIntervalPreset', String(batchIntervalMinutes));
  }, [batchIntervalMinutes]);

  useEffect(() => {
    localStorage.setItem('batchScopePreset', batchScope);
  }, [batchScope]);

  useEffect(() => {
    localStorage.setItem('batchStrategyPreset', batchStrategy);
  }, [batchStrategy]);

  useEffect(() => {
    localStorage.setItem('clipSearchModePreset', clipSearchModePreset);
  }, [clipSearchModePreset]);

  const applyPolledJobData = (targetJobId, data) => {
    const normalizedStatus = data.status === 'completed'
      ? 'complete'
      : data.status === 'failed'
        ? 'error'
        : 'processing';

    setProjects((prev) => prev.map((project) => {
      if (project.job_id !== targetJobId) return project;
      const clips = Array.isArray(data?.result?.clips) ? data.result.clips : [];
      const firstClip = clips[0] || null;
      return {
        ...project,
        status: normalizedStatus,
        clip_count_actual: clips.length || project.clip_count_actual || null,
        ratio: firstClip?.aspect_ratio || project.ratio || '9:16',
        thumbnail_url: firstClip?.thumbnail_url || project.thumbnail_url || null,
        preview_video_url: firstClip?.video_url || project.preview_video_url || null,
        updated_at: new Date().toISOString(),
        last_error: data?.error || null
      };
    }));

    if (data.result) {
      setResults(data.result);
    }

    if (data.status === 'completed') {
      setProcessUiPhase('complete');
      setStatus('complete');
      return;
    }
    if (data.status === 'failed') {
      setProcessUiPhase('error');
      setStatus('error');
      const errorMsg = data.error || (data.logs && data.logs.length > 0 ? data.logs[data.logs.length - 1] : "Proceso fallido");
      setLogs((prev) => [...prev, "Error: " + errorMsg]);
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
  };

  useEffect(() => {
    let interval;
    if ((status === 'processing' || status === 'completed') && jobId && !isPollingPaused) {
      interval = setInterval(async () => {
        try {
          const data = await pollJob(jobId);
          console.log("Job status:", data);
          applyPolledJobData(jobId, data);
        } catch (e) {
          console.error("Polling error", e);
        }
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [status, jobId, isPollingPaused]);


  const fetchUserProfiles = async () => {
    if (!uploadPostKey) return;
    try {
      const res = await apiFetch('/api/social/user', {
        headers: { 'X-Upload-Post-Key': uploadPostKey }
      });
      if (!res.ok) throw new Error("Error al consultar");
      const data = await res.json();
      if (data.profiles && data.profiles.length > 0) {
        setUserProfiles(data.profiles);
        // Auto select first if none selected
        if (!uploadUserId) {
          setUploadUserId(data.profiles[0].username);
        }
      } else {
        alert("No se encontraron perfiles para esta API Key.");
      }
    } catch (e) {
      alert("Error consultando perfiles de usuario. Revisa la API Key.");
      console.error(e);
    }
  };

  const handleProcess = async (data) => {
    const createdAt = new Date();
    const expiresAt = new Date(createdAt.getTime() + (PROJECTS_EXPIRE_DAYS * 24 * 60 * 60 * 1000));
    const isUrlSource = data?.type === 'url';
    const parsedUrl = (() => {
      if (!isUrlSource || !data?.payload) return null;
      try {
        return new URL(String(data.payload));
      } catch (_) {
        return null;
      }
    })();
    const sourceKind = isUrlSource
      ? (parsedUrl && /youtube|youtu\.be/i.test(parsedUrl.hostname) ? 'youtube' : 'url')
      : 'local';
    const sourceLabel = isUrlSource
      ? (sourceKind === 'youtube' ? 'YouTube' : (parsedUrl?.hostname || 'Enlace'))
      : 'Archivo local';
    const projectTitle = isUrlSource
      ? (parsedUrl?.pathname ? decodeURIComponent(parsedUrl.pathname).replaceAll('/', '').trim() || parsedUrl.hostname : String(data.payload))
      : (data?.payload?.name || 'Video local');

    const projectDraft = {
      title: projectTitle,
      source_kind: sourceKind,
      source_label: sourceLabel,
      video_type: projectVideoTypeLabel(data?.contentPreset),
      ratio: data?.aspectRatio === '16:9' ? '16:9' : '9:16',
      clip_count_target: Number.isFinite(data?.clipCount) ? data.clipCount : null,
      created_at: createdAt.toISOString(),
      expires_at: expiresAt.toISOString()
    };

    setStudioContext(null);
    setStatus('processing');
    setActiveTab('projects');
    setProjectsViewMode('detail');
    setIsPollingPaused(false);
    setProcessUiPhase('uploading');
    setLogs([data?.type === 'file' ? "Subiendo archivo..." : "Enviando URL al backend..."]);
    setResults(null);
    setBatchScheduleReport(null);
    setPackExportReport(null);
    setClipSearchResults([]);
    setClipSearchKeywords([]);
    setClipSearchPhrases([]);
    setClipSearchChapters([]);
    setClipSearchSpeakers([]);
    setClipHybridShortlist([]);
    setClipSearchProvider('local');
    setClipSearchMode('topic');
    setClipSearchRelaxed(false);
    setClipSearchScope(null);
    setClipSearchChapterFilter('-1');
    setClipSearchStartTime('');
    setClipSearchEndTime('');
    setClipSearchSpeakerFilter('all');
    setTranscriptSegments([]);
    setTranscriptFilter('');
    setTranscriptTotal(0);
    setIsLoadingTranscript(false);
    setTranscriptError(null);
    setTranscriptHasSpeakers(false);
    setClipSearchError(null);
    setProcessingMedia(data);

    try {
      let body;
      const headers = { 'X-Gemini-Key': apiKey };
      const language = data.language && data.language !== 'auto' ? data.language : null;
      const clipCount = Number.isFinite(data.clipCount) ? data.clipCount : null;
      const whisperBackend = data.whisperBackend || null;
      const whisperModel = data.whisperModel || null;
      const wordTimestamps = typeof data.wordTimestamps === 'boolean' ? data.wordTimestamps : null;
      const ffmpegPreset = data.ffmpegPreset || null;
      const ffmpegCrf = Number.isFinite(data.ffmpegCrf) ? data.ffmpegCrf : null;
      const aspectRatio = data.aspectRatio === '16:9' ? '16:9' : '9:16';
      const clipLengthTarget = ['short', 'balanced', 'long'].includes(data.clipLengthTarget) ? data.clipLengthTarget : null;
      const styleTemplate = typeof data.styleTemplate === 'string' && data.styleTemplate.trim() ? data.styleTemplate.trim() : null;
      const contentPreset = typeof data.contentPreset === 'string' && data.contentPreset.trim() ? data.contentPreset.trim() : null;

      if (data.type === 'url') {
        headers['Content-Type'] = 'application/json';
        body = JSON.stringify({
          url: data.payload,
          language,
          max_clips: clipCount,
          whisper_backend: whisperBackend,
          whisper_model: whisperModel,
          word_timestamps: wordTimestamps,
          ffmpeg_preset: ffmpegPreset,
          ffmpeg_crf: ffmpegCrf,
          aspect_ratio: aspectRatio,
          clip_length_target: clipLengthTarget,
          style_template: styleTemplate,
          content_profile: contentPreset
        });
      } else {
        const formData = new FormData();
        formData.append('file', data.payload);
        if (language) formData.append('language', language);
        if (clipCount) formData.append('max_clips', String(clipCount));
        if (whisperBackend) formData.append('whisper_backend', whisperBackend);
        if (whisperModel) formData.append('whisper_model', whisperModel);
        if (wordTimestamps !== null) formData.append('word_timestamps', String(wordTimestamps));
        if (ffmpegPreset) formData.append('ffmpeg_preset', ffmpegPreset);
        if (ffmpegCrf) formData.append('ffmpeg_crf', String(ffmpegCrf));
        formData.append('aspect_ratio', aspectRatio);
        if (clipLengthTarget) formData.append('clip_length_target', clipLengthTarget);
        if (styleTemplate) formData.append('style_template', styleTemplate);
        if (contentPreset) formData.append('content_profile', contentPreset);
        body = formData;
      }

      const res = await apiFetch('/api/process', {
        method: 'POST',
        headers: data.type === 'url' ? headers : { 'X-Gemini-Key': apiKey },
        body
      });

      if (!res.ok) throw new Error(await res.text());
      const resData = await res.json();
      setJobId(resData.job_id);
      setProcessUiPhase('queued');
      setLogs((prev) => [...prev, `Proyecto en cola (${resData.job_id.slice(0, 8)}...)`]);
      setProjects((prev) => {
        const next = Array.isArray(prev) ? [...prev] : [];
        const withoutSame = next.filter((item) => item.job_id !== resData.job_id);
        return [
          {
            job_id: resData.job_id,
            favorite: false,
            status: 'processing',
            thumbnail_url: null,
            preview_video_url: null,
            clip_count_actual: null,
            ...projectDraft
          },
          ...withoutSame
        ].slice(0, 40);
      });

    } catch (e) {
      setStatus('error');
      setProcessUiPhase('error');
      const msg = String(e?.message || 'Error desconocido');
      const isNetworkIssue = /networkerror|failed to fetch|load failed|cors|fetch resource/i.test(msg);
      if (isNetworkIssue && (getApiBaseUrl() || '').includes('ngrok')) {
        setLogs((l) => [
          ...l,
          `Error iniciando job: ${msg}`,
          'Diagnóstico: el túnel ngrok remoto parece caído o caducado. Genera una URL nueva en Colab/ngrok y guárdala en Configuración.'
        ]);
      } else {
        setLogs((l) => [...l, `Error iniciando job: ${msg}`]);
      }
    }
  };

  const handleReset = () => {
    setActiveTab('home');
    setProjectsViewMode('list');
    setStudioContext(null);
    setPendingReturnClipIndex(null);
    setFocusedClipIndex(null);
    setStatus('idle');
    setIsPollingPaused(false);
    setProcessUiPhase('idle');
    setJobId(null);
    setResults(null);
    setBatchScheduleReport(null);
    setPackExportReport(null);
    setClipSearchResults([]);
    setClipSearchKeywords([]);
    setClipSearchPhrases([]);
    setClipSearchChapters([]);
    setClipSearchSpeakers([]);
    setClipHybridShortlist([]);
    setClipSearchProvider('local');
    setClipSearchMode('topic');
    setClipSearchRelaxed(false);
    setClipSearchScope(null);
    setClipSearchChapterFilter('-1');
    setClipSearchStartTime('');
    setClipSearchEndTime('');
    setClipSearchSpeakerFilter('all');
    setTranscriptSegments([]);
    setTranscriptFilter('');
    setTranscriptTotal(0);
    setIsLoadingTranscript(false);
    setTranscriptError(null);
    setTranscriptHasSpeakers(false);
    setClipSearchError(null);
    setLogs([]);
    setProcessingMedia(null);
  };

  const handleRetryJob = async () => {
    if (!jobId) return;
    setIsRetryingJob(true);
    setIsPollingPaused(false);
    setProcessUiPhase('queued');
    setProjects((prev) => prev.map((project) => (
      project.job_id === jobId
        ? { ...project, status: 'processing', last_error: null, updated_at: new Date().toISOString() }
        : project
    )));
    try {
      const res = await apiFetch(`/api/retry/${jobId}`, { method: 'POST' });
      if (!res.ok) throw new Error(await res.text());
      setResults(null);
      setBatchScheduleReport(null);
      setPackExportReport(null);
      setStatus('processing');
      setProcessUiPhase('queued');
      setClipSearchResults([]);
      setClipSearchKeywords([]);
      setClipSearchPhrases([]);
      setClipSearchChapters([]);
      setClipSearchSpeakers([]);
      setClipHybridShortlist([]);
      setClipSearchScope(null);
      setClipSearchChapterFilter('-1');
      setClipSearchStartTime('');
      setClipSearchEndTime('');
      setClipSearchSpeakerFilter('all');
      setClipSearchError(null);
      setTranscriptError(null);
      setLogs((prev) => [...prev, 'Reintento manual encolado.']);
    } catch (e) {
      setLogs((prev) => [...prev, `Reintento fallido: ${e.message}`]);
    } finally {
      setIsRetryingJob(false);
    }
  };

  const sortedClips = useMemo(() => {
    const clips = results?.clips;
    if (!Array.isArray(clips)) return [];

    const normalized = clips.map((clip, idx) => {
      const rawScore = Number(clip?.virality_score);
      const fallbackScore = Math.max(55, 92 - (idx * 6));
      const score = Number.isFinite(rawScore) ? Math.max(0, Math.min(100, Math.round(rawScore))) : fallbackScore;

      const rawClipIndex = Number(clip?.clip_index);
      const clipIndex = Number.isFinite(rawClipIndex) ? rawClipIndex : idx;

      const duration = Math.max(0, Number(clip?.end ?? 0) - Number(clip?.start ?? 0));
      const band = clip?.score_band || (score >= 80 ? 'top' : score >= 65 ? 'medium' : 'low');
      const rawConfidence = Number(clip?.selection_confidence);
      const confidence = Number.isFinite(rawConfidence)
        ? Math.max(0, Math.min(1, rawConfidence))
        : Number((score / 100).toFixed(2));
      const topicTags = Array.isArray(clip?.topic_tags)
        ? clip.topic_tags.filter((t) => typeof t === 'string' && t.trim() !== '').map((t) => t.trim().toLowerCase())
        : [];
      const aspectRatio = clip?.aspect_ratio === '16:9' ? '16:9' : '9:16';
      const start = Number(clip?.start ?? 0);
      const end = Number(clip?.end ?? start);
      const rawExcerpt = [
        clip?.transcript_excerpt,
        clip?.transcript_text,
        clip?.transcription,
        clip?.transcript
      ].find((value) => typeof value === 'string' && value.trim());

      let transcriptExcerpt = typeof rawExcerpt === 'string' ? rawExcerpt.trim() : '';
      if (!transcriptExcerpt && Array.isArray(transcriptSegments) && transcriptSegments.length > 0) {
        const pieces = transcriptSegments
          .filter((seg) => {
            const segStart = Number(seg?.start ?? 0);
            const segEnd = Number(seg?.end ?? segStart);
            return segEnd > start && segStart < end;
          })
          .map((seg) => String(seg?.text || '').trim())
          .filter(Boolean);
        transcriptExcerpt = pieces.join(' ').trim();
      }
      if (transcriptExcerpt.length > 1800) {
        transcriptExcerpt = `${transcriptExcerpt.slice(0, 1800)}...`;
      }

      return {
        ...clip,
        transcript_excerpt: transcriptExcerpt,
        virality_score: score,
        score_band: band,
        selection_confidence: confidence,
        topic_tags: topicTags,
        clip_index: clipIndex,
        aspect_ratio: aspectRatio,
        _duration: duration
      };
    });

    const sorted = [...normalized];
    if (clipSort === 'balanced') {
      sorted.sort((a, b) => a.clip_index - b.clip_index);
    } else if (clipSort === 'safe') {
      sorted.sort((a, b) => {
        if (b.virality_score !== a.virality_score) return b.virality_score - a.virality_score;
        return a._duration - b._duration;
      });
    } else {
      sorted.sort((a, b) => {
        if (b.virality_score !== a.virality_score) return b.virality_score - a.virality_score;
        return a.clip_index - b.clip_index;
      });
    }
    return sorted;
  }, [results, clipSort, transcriptSegments]);

  const availableTags = useMemo(() => {
    const counts = new Map();
    sortedClips.forEach((clip) => {
      (clip.topic_tags || []).forEach((tag) => {
        counts.set(tag, (counts.get(tag) || 0) + 1);
      });
    });
    return Array.from(counts.entries())
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .slice(0, 20)
      .map(([tag]) => tag);
  }, [sortedClips]);

  useEffect(() => {
    if (clipTagFilter !== 'all' && !availableTags.includes(clipTagFilter)) {
      setClipTagFilter('all');
    }
  }, [clipTagFilter, availableTags]);

  const visibleClips = useMemo(() => {
    if (!Array.isArray(sortedClips)) return [];
    return sortedClips.filter((clip) => {
      const matchBand = clipFilter === 'all' ? true : clip.score_band === clipFilter;
      const matchTag = clipTagFilter === 'all' ? true : (clip.topic_tags || []).includes(clipTagFilter);
      return matchBand && matchTag;
    });
  }, [sortedClips, clipFilter, clipTagFilter]);

  const applyBatchStrategy = (strategy) => {
    setBatchStrategy(strategy);
    if (strategy === 'growth') {
      setBatchTopCount(5);
      setBatchStartDelayMinutes(5);
      setBatchIntervalMinutes(30);
      setBatchScope('global');
      return;
    }
    if (strategy === 'conservative') {
      setBatchTopCount(2);
      setBatchStartDelayMinutes(60);
      setBatchIntervalMinutes(240);
      setBatchScope('visible');
      return;
    }
    if (strategy === 'balanced') {
      setBatchTopCount(3);
      setBatchStartDelayMinutes(15);
      setBatchIntervalMinutes(60);
      setBatchScope('visible');
      return;
    }
  };

  const handleBatchReportCsvDownload = () => {
    if (!batchScheduleReport || !Array.isArray(batchScheduleReport.timeline) || batchScheduleReport.timeline.length === 0) return;
    const lines = [
      [
        'scheduled_at',
        'clip_index',
        'clip_title',
        'virality_score',
        'platforms',
        'status',
        'error'
      ].join(',')
    ];
    batchScheduleReport.timeline.forEach((item) => {
      const safe = (v) => `"${String(v ?? '').replaceAll('"', '""')}"`;
      lines.push([
        safe(item.scheduled_at),
        safe(item.clip_index),
        safe(item.clip_title),
        safe(item.virality_score),
        safe((item.platforms || []).join('|')),
        safe(item.status),
        safe(item.error || '')
      ].join(','));
    });

    const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `batch_schedule_report_${jobId || 'job'}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleQueueTopClips = async () => {
    const candidatePool = batchScope === 'global' ? sortedClips : visibleClips;
    if (!jobId || candidatePool.length === 0) return;
    if (!uploadPostKey || !uploadUserId) {
      alert("Configura Upload-Post API Key y perfil de usuario en Configuración para usar cola batch.");
      return;
    }

    const topCount = Math.max(1, Math.min(10, Number(batchTopCount) || 3));
    const startDelay = Math.max(0, Math.min(180, Number(batchStartDelayMinutes) || 15));
    const interval = Math.max(5, Math.min(720, Number(batchIntervalMinutes) || 60));

    const candidates = [...candidatePool]
      .sort((a, b) => b.virality_score - a.virality_score || a.clip_index - b.clip_index)
      .slice(0, topCount);

    if (candidates.length === 0) return;

    const selectedProfile = userProfiles.find((p) => p.username === uploadUserId);
    const connectedPlatforms = Array.isArray(selectedProfile?.connected)
      ? selectedProfile.connected.filter((p) => ['tiktok', 'instagram', 'youtube'].includes(p))
      : [];
    const platforms = connectedPlatforms.length > 0 ? connectedPlatforms : ['tiktok', 'instagram', 'youtube'];

    setIsBatchScheduling(true);
    setBatchScheduleReport(null);
    setPackExportReport(null);
      setLogs((prev) => [...prev, `Encolando ${candidates.length} clips priorizados (${platforms.join(', ')}) | inicia +${startDelay}m, cada ${interval}m...`]);

    let success = 0;
    const failures = [];
    const timeline = [];
    const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC';

    for (let i = 0; i < candidates.length; i += 1) {
      const clip = candidates[i];
      const scheduledAt = new Date(Date.now() + (startDelay + (i * interval)) * 60 * 1000).toISOString();
      try {
        const payload = {
          job_id: jobId,
          clip_index: clip.clip_index,
          api_key: uploadPostKey,
          user_id: uploadUserId,
          platforms,
          title: clip.video_title_for_youtube_short || `Clip viral #${i + 1}`,
          description: clip.video_description_for_instagram || clip.video_description_for_tiktok || "Míralo aquí.",
          scheduled_date: scheduledAt,
          timezone
        };

        const res = await apiFetch('/api/social/post', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        if (!res.ok) {
          const errText = await res.text();
          failures.push(`Clip ${clip.clip_index + 1}: ${errText}`);
          timeline.push({
            clip_index: clip.clip_index,
            clip_title: clip.video_title_for_youtube_short || `Clip ${clip.clip_index + 1}`,
            virality_score: clip.virality_score,
            scheduled_at: scheduledAt,
            platforms,
            status: 'failed',
            error: errText
          });
        } else {
          success += 1;
          timeline.push({
            clip_index: clip.clip_index,
            clip_title: clip.video_title_for_youtube_short || `Clip ${clip.clip_index + 1}`,
            virality_score: clip.virality_score,
            scheduled_at: scheduledAt,
            platforms,
            status: 'scheduled',
            error: ''
          });
        }
      } catch (e) {
        failures.push(`Clip ${clip.clip_index + 1}: ${e.message}`);
        timeline.push({
          clip_index: clip.clip_index,
          clip_title: clip.video_title_for_youtube_short || `Clip ${clip.clip_index + 1}`,
          virality_score: clip.virality_score,
          scheduled_at: scheduledAt,
          platforms,
          status: 'failed',
          error: e.message
        });
      }
    }

    const report = {
      success,
      total: candidates.length,
      failures,
      timeline: timeline.sort((a, b) => new Date(a.scheduled_at) - new Date(b.scheduled_at)),
      strategy: batchStrategy,
      scope: batchScope,
      top_count: topCount,
      start_delay_minutes: startDelay,
      interval_minutes: interval
    };
    setBatchScheduleReport(report);
    if (failures.length === 0) {
      setLogs((prev) => [...prev, `Cola batch completada: ${success}/${candidates.length} programados.`]);
    } else {
      setLogs((prev) => [...prev, `Cola batch completada con incidencias: ${success}/${candidates.length} programados.`]);
    }
    setIsBatchScheduling(false);
  };

  const handleExportPack = async () => {
    if (!jobId) return;
    setIsExportingPack(true);
    setPackExportReport(null);
    try {
      const res = await apiFetch('/api/export/pack', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          job_id: jobId,
          include_video_files: true,
          include_srt_files: true,
          include_thumbnails: true,
          include_platform_variants: true,
          thumbnail_format: 'jpg',
          thumbnail_width: 1080
        })
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setPackExportReport(data);

      const href = getApiUrl(data.pack_url);
      const a = document.createElement('a');
      a.href = href;
      a.download = href.split('/').pop() || `agency_pack_${jobId}.zip`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setLogs((prev) => [...prev, `Pack listo (${data.video_files_added} videos, ${data.srt_files_added} srt, ${data.thumbnail_files_added || 0} miniaturas).`]);
    } catch (e) {
      setPackExportReport({ success: false, error: e.message });
      setLogs((prev) => [...prev, `Error exportando paquete: ${e.message}`]);
    } finally {
      setIsExportingPack(false);
    }
  };

  const handleClipSearch = async () => {
    if (!jobId) return;
    const query = clipSearchQuery.trim();
    if (!query) return;
    const startTimeNum = Number(clipSearchStartTime);
    const endTimeNum = Number(clipSearchEndTime);
    const hasStart = clipSearchStartTime.trim() !== '' && Number.isFinite(startTimeNum);
    const hasEnd = clipSearchEndTime.trim() !== '' && Number.isFinite(endTimeNum);
    const chapterIndexNum = Number(clipSearchChapterFilter);
    const hasChapter = Number.isFinite(chapterIndexNum) && chapterIndexNum >= 0;
    const speakerFilter = clipSearchSpeakerFilter !== 'all' ? clipSearchSpeakerFilter : null;

    setIsSearchingClips(true);
    setClipSearchError(null);
    try {
      const headers = { 'Content-Type': 'application/json' };
      if (apiKey?.trim()) {
        headers['X-Gemini-Key'] = apiKey.trim();
      }
      const res = await apiFetch('/api/search/clips', {
        method: 'POST',
        headers,
        body: JSON.stringify({
          job_id: jobId,
          query,
          limit: 6,
          shortlist_limit: 6,
          search_mode: clipSearchModePreset,
          chapter_index: hasChapter ? chapterIndexNum : null,
          start_time: hasStart ? startTimeNum : null,
          end_time: hasEnd ? endTimeNum : null,
          speaker: speakerFilter
        })
      });
      if (!res.ok) {
        throw new Error(await res.text());
      }
      const data = await res.json();
      setClipSearchResults(Array.isArray(data.matches) ? data.matches : []);
      setClipSearchKeywords(Array.isArray(data.keywords) ? data.keywords : []);
      setClipSearchPhrases(Array.isArray(data.phrases) ? data.phrases : []);
      setClipSearchChapters(Array.isArray(data.chapters) ? data.chapters : []);
      setClipSearchSpeakers(Array.isArray(data.speakers) ? data.speakers : []);
      setClipHybridShortlist(Array.isArray(data.hybrid_shortlist) ? data.hybrid_shortlist : []);
      setClipSearchProvider(data.semantic_provider === 'gemini' ? 'gemini' : 'local');
      setClipSearchMode(data.query_profile?.mode || 'topic');
      setClipSearchRelaxed(Boolean(data.query_profile?.relaxed || data.used_relaxed_profile));
      setClipSearchScope(data.search_scope || null);
      setLogs((prev) => [...prev, `Búsqueda de clips \"${query}\": ${(data.matches || []).length} coincidencias.`]);
    } catch (e) {
      setClipSearchError(e.message);
      setClipSearchResults([]);
      setClipSearchKeywords([]);
      setClipSearchPhrases([]);
      setClipSearchChapters([]);
      setClipSearchSpeakers([]);
      setClipHybridShortlist([]);
      setClipSearchProvider('local');
      setClipSearchMode('topic');
      setClipSearchRelaxed(false);
      setClipSearchScope(null);
      setLogs((prev) => [...prev, `Búsqueda de clips fallida: ${e.message}`]);
    } finally {
      setIsSearchingClips(false);
    }
  };

  const loadTranscriptSegments = async (targetJobId) => {
    if (!targetJobId) return;
    setIsLoadingTranscript(true);
    setTranscriptError(null);
    try {
      const res = await apiFetch(`/api/transcript/${targetJobId}?limit=1200`);
      if (!res.ok) {
        throw new Error(await res.text());
      }
      const data = await res.json();
      const segments = Array.isArray(data.segments) ? data.segments : [];
      setTranscriptSegments(segments);
      setTranscriptTotal(Number.isFinite(data.total) ? data.total : segments.length);
      setTranscriptHasSpeakers(Boolean(data.has_speaker_labels));
      setLogs((prev) => [...prev, `Transcript cargado: ${segments.length}/${data.total || segments.length} segmentos.`]);
    } catch (e) {
      setTranscriptError(e.message);
      setTranscriptSegments([]);
      setTranscriptTotal(0);
      setTranscriptHasSpeakers(false);
    } finally {
      setIsLoadingTranscript(false);
    }
  };

  useEffect(() => {
    if (status !== 'complete' || !jobId) return;
    if (transcriptSegments.length > 0) return;
    loadTranscriptSegments(jobId);
  }, [status, jobId]);

  const visibleTranscriptSegments = useMemo(() => {
    if (!Array.isArray(transcriptSegments) || transcriptSegments.length === 0) return [];
    const q = transcriptFilter.trim().toLowerCase();
    if (!q) return transcriptSegments;
    return transcriptSegments.filter((seg) => {
      const text = String(seg?.text || '').toLowerCase();
      const speaker = String(seg?.speaker || '').toLowerCase();
      return text.includes(q) || speaker.includes(q);
    });
  }, [transcriptSegments, transcriptFilter]);

  const availableSearchSpeakers = useMemo(() => {
    const merged = new Set();
    (clipSearchSpeakers || []).forEach((speaker) => {
      const value = String(speaker || '').trim();
      if (value) merged.add(value);
    });
    (transcriptSegments || []).forEach((seg) => {
      const value = String(seg?.speaker || '').trim();
      if (value) merged.add(value);
    });
    return Array.from(merged).sort((a, b) => a.localeCompare(b));
  }, [clipSearchSpeakers, transcriptSegments]);

  const favoriteProjectsCount = useMemo(
    () => (projects || []).filter((p) => p.favorite).length,
    [projects]
  );

  const visibleProjects = useMemo(() => {
    const list = Array.isArray(projects) ? projects : [];
    if (projectFilter === 'favorites') {
      return list.filter((p) => p.favorite);
    }
    return list;
  }, [projects, projectFilter]);

  const processingTimeline = useMemo(() => {
    const joinedLogs = (logs || []).join(' ').toLowerCase();
    const includesAny = (...tokens) => tokens.some((token) => joinedLogs.includes(token));
    const readyClips = Array.isArray(results?.clips) ? results.clips.length : 0;
    const targetClips = Number.isFinite(processingMedia?.clipCount) ? Math.max(1, processingMedia.clipCount) : null;

    const queuedSeen = includesAny('queued', 'encol', 'attempt');
    const processingSeen = includesAny('executing command', 'processing', 'transcrib', 'whisper', 'analiz');
    const discoverySeen = includesAny('metadata', 'topic', 'virality', 'score', 'find');
    const renderingSeen = readyClips > 0 || includesAny('clip_', 'clip ', 'render', 'export');
    const finishingSeen = includesAny('process finished successfully', 'finished successfully', 'completed');

    let phaseIndex = 0;
    if (status === 'complete' || processUiPhase === 'complete') {
      phaseIndex = 5;
    } else if (processUiPhase === 'uploading') {
      phaseIndex = 0;
    } else if (processUiPhase === 'queued' || queuedSeen) {
      phaseIndex = 1;
    } else if (renderingSeen || processUiPhase === 'rendering') {
      phaseIndex = 4;
    } else if (discoverySeen) {
      phaseIndex = 3;
    } else if (processingSeen || processUiPhase === 'running') {
      phaseIndex = 2;
    } else if (finishingSeen || processUiPhase === 'finalizing') {
      phaseIndex = 5;
    }

    const steps = [
      {
        key: 'upload',
        label: processingMedia?.type === 'file' ? 'Subiendo video' : 'Enviando enlace'
      },
      { key: 'project', label: 'Creando proyecto' },
      { key: 'process', label: 'Procesando video' },
      { key: 'best_parts', label: 'Buscando mejores momentos' },
      {
        key: 'clips',
        label: targetClips
          ? `Generando clips (${Math.min(readyClips, targetClips)}/${targetClips})`
          : readyClips > 0
            ? `Generando clips (${readyClips})`
            : 'Generando clips'
      },
      { key: 'finalize', label: 'Finalizando' }
    ];

    const withState = steps.map((step, idx) => {
      if (status === 'complete') return { ...step, state: 'done' };
      if (idx < phaseIndex) return { ...step, state: 'done' };
      if (idx === phaseIndex) return { ...step, state: status === 'error' ? 'error' : 'active' };
      return { ...step, state: 'todo' };
    }).map((step, idx) => ({ ...step, index: idx + 1 }));

    const visibleSteps = withState.filter((step, idx) => {
      if (status === 'complete') return true;
      return idx <= phaseIndex;
    });

    let progressPercent = 10;
    if (status === 'complete') {
      progressPercent = 100;
    } else if (phaseIndex === 0) {
      progressPercent = 12;
    } else if (phaseIndex === 1) {
      progressPercent = 24;
    } else if (phaseIndex === 2) {
      progressPercent = 44;
    } else if (phaseIndex === 3) {
      progressPercent = 62;
    } else if (phaseIndex === 4) {
      if (targetClips) {
        const ratio = Math.max(0, Math.min(1, readyClips / targetClips));
        progressPercent = Math.min(92, 70 + Math.round(ratio * 22));
      } else {
        progressPercent = Math.min(90, 70 + Math.min(18, readyClips * 6));
      }
    } else if (phaseIndex === 5) {
      progressPercent = 96;
    }

    return {
      steps: withState,
      visibleSteps,
      phaseIndex,
      totalSteps: withState.length,
      progressPercent: status === 'error' ? Math.max(8, progressPercent - 4) : progressPercent,
      headline: status === 'complete'
        ? 'Proceso completado'
        : status === 'error'
          ? 'Proceso interrumpido'
          : withState.find((step) => step.state === 'active')?.label || 'Procesando',
      readyClips,
      targetClips
    };
  }, [logs, results, processingMedia, processUiPhase, status]);

  const processingProjectName = useMemo(() => {
    if (!processingMedia) return 'Proyecto actual';
    if (processingMedia.type === 'file') {
      return processingMedia?.payload?.name || 'Archivo local';
    }
    const raw = String(processingMedia?.payload || '').trim();
    if (!raw) return 'Enlace';
    try {
      const url = new URL(raw);
      return url.hostname.replace(/^www\./, '');
    } catch (_) {
      return raw.length > 42 ? `${raw.slice(0, 42)}...` : raw;
    }
  }, [processingMedia]);

  const processingSourceLabel = processingMedia?.type === 'file' ? 'Archivo local' : 'Enlace';

  const handleBrandKitFieldChange = (field, value) => {
    setBrandKit((prev) => normalizeBrandKit({ ...prev, [field]: value }));
  };

  const handleBrandKitReset = () => {
    setBrandKit(DEFAULT_BRAND_KIT);
  };

  const handleSaveApiBaseUrl = () => {
    const rawInput = String(apiBaseUrlInput || '').trim();
    const normalizedInput = normalizeApiBaseUrl(rawInput);
    if (rawInput && !normalizedInput) {
      setApiBaseUrlMessage('URL inválida. Usa un formato como https://xxxx.ngrok-free.app');
      setApiBaseUrlMessageType('error');
      return;
    }
    const normalized = setApiBaseUrl(apiBaseUrlInput);
    const effective = getApiBaseUrl() || '';
    setApiBaseUrlInput(effective);
    setApiBaseUrlActive(effective);
    if (normalized) {
      setApiBaseUrlMessage(`API remota guardada: ${normalized}`);
      setApiBaseUrlMessageType('success');
      setLogs((prev) => [...prev, `API remota activa: ${normalized}`]);
      return;
    }
    setApiBaseUrlMessage('Sin URL remota. Se usará backend local/proxy.');
    setApiBaseUrlMessageType('neutral');
    setLogs((prev) => [...prev, 'API remota desactivada.']);
  };

  const handleTestApiBaseUrl = async () => {
    const target = normalizeApiBaseUrl(apiBaseUrlInput || apiBaseUrlActive);
    if (!target) {
      setApiBaseUrlMessage('Pega una URL válida primero (ej: https://xxxx.ngrok-free.app).');
      setApiBaseUrlMessageType('neutral');
      return;
    }

    setIsTestingApiBaseUrl(true);
    setApiBaseUrlMessage(`Probando conexión a ${target} ...`);
    setApiBaseUrlMessageType('neutral');

    try {
      const res = await fetch(`${target}/docs`, { method: 'GET' });
      if (res.ok) {
        setApiBaseUrlMessage(`Conexión OK con ${target}`);
        setApiBaseUrlMessageType('success');
        return;
      }
      setApiBaseUrlMessage(`La URL respondió ${res.status}. Verifica que OpenShorts esté corriendo en ese túnel.`);
      setApiBaseUrlMessageType('neutral');
    } catch (_) {
      setApiBaseUrlMessage('No se pudo conectar. En ngrok esto suele ser URL vencida/offline o túnel apuntando a otro servicio.');
      setApiBaseUrlMessageType('error');
    } finally {
      setIsTestingApiBaseUrl(false);
    }
  };

  const handleResetApiBaseUrl = () => {
    setApiBaseUrl('');
    const effective = getApiBaseUrl() || '';
    setApiBaseUrlInput(effective);
    setApiBaseUrlActive(effective);
    if (effective) {
      setApiBaseUrlMessage(`Se restauró la URL de entorno: ${effective}`);
      setApiBaseUrlMessageType('neutral');
      return;
    }
    setApiBaseUrlMessage('Restaurado a local/proxy (sin URL remota).');
    setApiBaseUrlMessageType('neutral');
  };

  const toggleProjectFavorite = (targetJobId) => {
    setProjects((prev) => prev.map((project) => (
      project.job_id === targetJobId
        ? { ...project, favorite: !project.favorite }
        : project
    )));
  };

  const removeProject = (targetJobId) => {
    setProjects((prev) => prev.filter((project) => project.job_id !== targetJobId));
    if (jobId === targetJobId) {
      handleReset();
      setActiveTab('projects');
      setProjectsViewMode('list');
    }
    if (projectTitleEditJobId === targetJobId) {
      setProjectTitleEditJobId(null);
      setProjectTitleDraft('');
    }
    setProjectMenuJobId(null);
  };

  const beginEditProjectTitle = (project) => {
    if (!project?.job_id) return;
    setProjectTitleEditJobId(project.job_id);
    setProjectTitleDraft(String(project.title || '').trim());
    setProjectMenuJobId(null);
  };

  const cancelEditProjectTitle = () => {
    setProjectTitleEditJobId(null);
    setProjectTitleDraft('');
  };

  const saveProjectTitle = (targetJobId) => {
    const nextTitle = String(projectTitleDraft || '').trim();
    if (!targetJobId || !nextTitle) {
      cancelEditProjectTitle();
      return;
    }
    setProjects((prev) => prev.map((project) => (
      project.job_id === targetJobId
        ? { ...project, title: nextTitle, updated_at: new Date().toISOString() }
        : project
    )));
    cancelEditProjectTitle();
  };

  const setClipCardRef = useCallback((clipIndex, node) => {
    const key = Number(clipIndex);
    if (!Number.isFinite(key)) return;
    if (node) {
      clipCardRefs.current.set(key, node);
      return;
    }
    clipCardRefs.current.delete(key);
  }, []);

  const openClipStudio = useCallback(({ clip, clipIndex, currentVideoUrl }) => {
    if (!jobId || !clip) return;
    pollingPauseBeforeStudioRef.current = Boolean(isPollingPaused);
    setIsPollingPaused(true);
    setFocusedClipIndex(null);
    setStudioContext({
      jobId,
      clip,
      clipIndex: Number.isFinite(clipIndex) ? clipIndex : Number(clip?.clip_index || 0),
      currentVideoUrl: currentVideoUrl || getApiUrl(clip?.video_url || '')
    });
  }, [jobId, isPollingPaused]);

  const closeClipStudio = useCallback((options = {}) => {
    const { restoreFocus = true } = options;
    const targetClipIndex = restoreFocus ? Number(studioContext?.clipIndex) : NaN;
    setStudioContext(null);
    if (Number.isFinite(targetClipIndex)) {
      setPendingReturnClipIndex(targetClipIndex);
      setFocusedClipIndex(targetClipIndex);
      setTimeout(() => setFocusedClipIndex(null), 2000);
    }
    if (status === 'processing') {
      setIsPollingPaused(Boolean(pollingPauseBeforeStudioRef.current));
    }
  }, [status, studioContext?.clipIndex]);

  const handleStudioApplied = useCallback(({ newVideoUrl, clipPatch }) => {
    if (!studioContext) return;
    const targetClipIndex = Number(studioContext.clipIndex);
    if (newVideoUrl || (clipPatch && typeof clipPatch === 'object')) {
      setResults((prev) => {
        if (!prev || !Array.isArray(prev.clips)) return prev;
        return {
          ...prev,
          clips: prev.clips.map((item) => {
            if (Number(item?.clip_index) !== targetClipIndex) return item;
            return {
              ...item,
              ...(newVideoUrl ? { video_url: newVideoUrl } : {}),
              ...(clipPatch || {})
            };
          })
        };
      });
    }
    if (newVideoUrl) {
      setProjects((prev) => prev.map((project) => (
        project.job_id === studioContext.jobId
          ? { ...project, preview_video_url: newVideoUrl, updated_at: new Date().toISOString() }
          : project
      )));
    }
    closeClipStudio();
  }, [studioContext, closeClipStudio]);

  const openSavedProject = async (project) => {
    if (!project?.job_id) return;
    setStudioContext(null);
    setPendingReturnClipIndex(null);
    setFocusedClipIndex(null);
    setActiveTab('projects');
    setProjectsViewMode('detail');
    setJobId(project.job_id);
    setStatus('processing');
    setIsPollingPaused(false);
    setProcessUiPhase('running');
    setResults(null);
    setLogs(['Cargando proyecto guardado...']);
    setProcessingMedia(null);
    try {
      const data = await pollJob(project.job_id);
      if (Array.isArray(data.logs) && data.logs.length > 0) {
        setLogs(data.logs);
      }
      if (data.result) {
        setResults(data.result);
      }
      if (data.status === 'completed') {
        setProcessUiPhase('complete');
        setStatus('complete');
      } else if (data.status === 'failed') {
        setProcessUiPhase('error');
        setStatus('error');
      } else {
        setProcessUiPhase(Array.isArray(data?.result?.clips) && data.result.clips.length > 0 ? 'rendering' : 'running');
        setStatus('processing');
      }
    } catch (e) {
      setProcessUiPhase('error');
      setStatus('error');
      setLogs((prev) => [...prev, `No se pudo abrir el proyecto: ${e.message}`]);
    } finally {
      setProjectMenuJobId(null);
    }
  };

  useEffect(() => {
    if ((activeTab !== 'projects' || projectsViewMode !== 'detail') && studioContext) {
      closeClipStudio({ restoreFocus: false });
    }
  }, [activeTab, projectsViewMode, studioContext, closeClipStudio]);

  useEffect(() => {
    if (studioContext) return;
    if (pendingReturnClipIndex === null) return;
    if (activeTab !== 'projects' || projectsViewMode !== 'detail') return;

    const focusClipCard = () => {
      const node = clipCardRefs.current.get(Number(pendingReturnClipIndex));
      if (!node) return false;
      node.scrollIntoView({ behavior: 'smooth', block: 'center' });
      if (typeof node.focus === 'function') {
        node.focus({ preventScroll: true });
      }
      return true;
    };

    setPendingReturnClipIndex(null);
    requestAnimationFrame(() => {
      if (!focusClipCard()) {
        setTimeout(() => {
          focusClipCard();
        }, 120);
      }
    });
  }, [studioContext, pendingReturnClipIndex, activeTab, projectsViewMode]);

  // --- UI Components ---

  const TopBar = () => (
    <header className="sticky top-0 z-30 border-b border-slate-200/80 dark:border-slate-700/60 bg-white/90 dark:bg-slate-900/80 backdrop-blur-md">
      <div className="max-w-[1500px] mx-auto px-4 sm:px-6">
        <div className="h-16 grid grid-cols-[auto_1fr_auto] items-center gap-3">
          <div className="flex items-center gap-3 min-w-0">
            <div className="w-9 h-9 rounded-xl overflow-hidden border border-slate-200 dark:border-slate-700 bg-white">
              <img src="/logo-openshorts.png" alt="OpenShorts" className="w-full h-full object-cover" />
            </div>
            <span className="font-bold text-slate-900 dark:text-white tracking-tight">OpenShorts</span>
          </div>

          <div className="hidden md:flex items-center justify-center gap-2">
            <button
              onClick={() => {
                setActiveTab('home');
                setProjectsViewMode('list');
              }}
              className={`px-3 py-1.5 rounded-full text-sm font-medium border transition-colors ${
                activeTab === 'home'
                  ? 'bg-primary/10 border-primary/30 text-primary'
                  : 'bg-transparent border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:border-slate-300 dark:hover:border-slate-500'
              }`}
            >
              <span className="inline-flex items-center gap-1.5">
                <LayoutDashboard size={14} />
                Home
              </span>
            </button>
            <button
              onClick={() => {
                setActiveTab('projects');
                setProjectsViewMode('list');
              }}
              className={`px-3 py-1.5 rounded-full text-sm font-medium border transition-colors ${
                activeTab === 'projects'
                  ? 'bg-primary/10 border-primary/30 text-primary'
                  : 'bg-transparent border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:border-slate-300 dark:hover:border-slate-500'
              }`}
            >
              <span className="inline-flex items-center gap-1.5">
                <History size={14} />
                Proyectos
              </span>
            </button>
            <button
              onClick={() => setActiveTab('settings')}
              className={`px-3 py-1.5 rounded-full text-sm font-medium border transition-colors ${
                activeTab === 'settings'
                  ? 'bg-primary/10 border-primary/30 text-primary'
                  : 'bg-transparent border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:border-slate-300 dark:hover:border-slate-500'
              }`}
            >
              <span className="inline-flex items-center gap-1.5">
                <Settings size={14} />
                Configuración
              </span>
            </button>
          </div>

          <div className="flex items-center gap-2 sm:gap-3">
            <div className="hidden lg:flex items-center gap-2 px-2.5 py-1.5 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-white/5">
              <span className="text-[10px] uppercase tracking-wider text-slate-500 dark:text-slate-400">Señal</span>
              <span
                className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] border ${
                  connectivityStatus.api === 'ok'
                    ? 'bg-emerald-100 text-emerald-700 border-emerald-200 dark:bg-emerald-900/20 dark:text-emerald-300 dark:border-emerald-800'
                    : connectivityStatus.api === 'checking'
                      ? 'bg-amber-100 text-amber-700 border-amber-200 dark:bg-amber-900/20 dark:text-amber-300 dark:border-amber-800'
                      : 'bg-red-100 text-red-700 border-red-200 dark:bg-red-900/20 dark:text-red-300 dark:border-red-800'
                }`}
                title={connectivityStatus.error ? `API: ${connectivityStatus.error}` : 'Estado de API'}
              >
                <span className={`w-1.5 h-1.5 rounded-full ${connectivityStatus.api === 'ok' ? 'bg-emerald-500 animate-pulse' : connectivityStatus.api === 'checking' ? 'bg-amber-500 animate-pulse' : 'bg-red-500'}`} />
                API
              </span>
              <span
                className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] border ${
                  connectivityStatus.tunnel === 'ok'
                    ? 'bg-emerald-100 text-emerald-700 border-emerald-200 dark:bg-emerald-900/20 dark:text-emerald-300 dark:border-emerald-800'
                    : connectivityStatus.tunnel === 'local'
                      ? 'bg-slate-100 text-slate-700 border-slate-200 dark:bg-slate-800 dark:text-slate-300 dark:border-slate-700'
                      : connectivityStatus.tunnel === 'checking'
                        ? 'bg-amber-100 text-amber-700 border-amber-200 dark:bg-amber-900/20 dark:text-amber-300 dark:border-amber-800'
                        : 'bg-red-100 text-red-700 border-red-200 dark:bg-red-900/20 dark:text-red-300 dark:border-red-800'
                }`}
                title={connectivityStatus.usingNgrok ? 'Estado del túnel ngrok' : 'Conexión local activa'}
              >
                <span className={`w-1.5 h-1.5 rounded-full ${connectivityStatus.tunnel === 'ok' ? 'bg-emerald-500 animate-pulse' : connectivityStatus.tunnel === 'local' ? 'bg-slate-500' : connectivityStatus.tunnel === 'checking' ? 'bg-amber-500 animate-pulse' : 'bg-red-500'}`} />
                {connectivityStatus.usingNgrok ? 'Ngrok' : 'Local'}
              </span>
              {connectivityStatus.lastCheckedAt && (
                <span className="text-[10px] text-slate-500 dark:text-slate-400">
                  {new Date(connectivityStatus.lastCheckedAt).toLocaleTimeString('es-ES', { hour12: false })}
                </span>
              )}
              <button
                type="button"
                onClick={runConnectivityCheck}
                className="p-1 rounded-md text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-white hover:bg-slate-200 dark:hover:bg-slate-700 transition-colors"
                title="Verificar conexión ahora"
              >
                <RefreshCw size={13} className={isConnectivityChecking ? 'animate-spin' : ''} />
              </button>
            </div>
            {userProfiles.length > 0 && (
              <UserProfileSelector
                profiles={userProfiles}
                selectedUserId={uploadUserId}
                onSelect={setUploadUserId}
              />
            )}
            {!apiKey && (
              <span className="hidden md:inline text-xs text-amber-600 dark:text-amber-300 bg-amber-100/70 dark:bg-amber-500/10 px-3 py-1 rounded-full border border-amber-200 dark:border-amber-500/20">
                Falta API Key
              </span>
            )}
            <button
              onClick={() => setTheme((prev) => prev === 'dark' ? 'light' : 'dark')}
              className="inline-flex items-center gap-2 text-xs bg-slate-100 dark:bg-white/10 border border-slate-200 dark:border-white/20 px-3 py-1.5 rounded-lg text-slate-700 dark:text-zinc-200 hover:bg-slate-200 dark:hover:bg-white/15 transition-colors"
              title={theme === 'dark' ? 'Cambiar a modo claro' : 'Cambiar a modo oscuro'}
            >
              {theme === 'dark' ? <Sun size={14} /> : <Moon size={14} />}
              <span className="hidden sm:inline">{theme === 'dark' ? 'Claro' : 'Oscuro'}</span>
            </button>
          </div>
        </div>
      </div>
    </header>
  );

  return (
    <div className="min-h-screen bg-background selection:bg-primary/20">
      <TopBar />

      <main className="max-w-[1500px] mx-auto w-full px-4 sm:px-6 py-6 relative">

        {/* Main Workspace */}
        <div className="flex-1 overflow-hidden relative">

          {/* View: Settings */}
          {activeTab === 'settings' && (
            <div className="animate-[fadeIn_0.3s_ease-out] max-w-3xl mx-auto pb-10">
              <div className="flex items-center justify-between mb-8">
                <h1 className="text-2xl font-bold text-slate-900 dark:text-white">Configuración</h1>
                <div className="px-3 py-1 bg-emerald-100 dark:bg-green-500/10 border border-emerald-200 dark:border-green-500/20 rounded-full text-[10px] text-emerald-700 dark:text-green-400 font-medium flex items-center gap-2">
                  <Shield size={12} /> Privacidad: las llaves viven en tu navegador (se envían al backend solo para procesar)
                </div>
              </div>
              <KeyInput onKeySet={setApiKey} savedKey={apiKey} />

              <div className="glass-panel p-6 mt-8">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold">Backend remoto (Colab / ngrok)</h2>
                  <span className="text-[10px] bg-white/5 border border-white/5 px-2 py-0.5 rounded text-zinc-500 uppercase tracking-wider">Recomendado</span>
                </div>
                <p className="text-xs text-zinc-500 mb-4 leading-relaxed">
                  Pega aquí tu URL pública de Colab/ngrok para que el frontend mande todas las peticiones a ese backend.
                  Así no tienes que cambiar scripts ni tocar variables cada vez.
                </p>
                <div className="space-y-3">
                  <label className="block text-sm text-zinc-400">URL API remota</label>
                  <input
                    type="url"
                    value={apiBaseUrlInput}
                    onChange={(e) => setApiBaseUrlInput(e.target.value)}
                    className="input-field"
                    placeholder="https://62cb-34-168-226-133.ngrok-free.app"
                  />
                  <div className="flex flex-wrap gap-2">
                    <button
                      type="button"
                      onClick={handleSaveApiBaseUrl}
                      className="btn-primary py-2 px-4 text-sm"
                    >
                      Guardar URL
                    </button>
                    <button
                      type="button"
                      onClick={handleTestApiBaseUrl}
                      disabled={isTestingApiBaseUrl}
                      className="py-2 px-4 text-sm rounded-lg border border-white/15 text-zinc-300 hover:bg-white/5 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {isTestingApiBaseUrl ? 'Probando...' : 'Probar conexión'}
                    </button>
                    <button
                      type="button"
                      onClick={handleResetApiBaseUrl}
                      className="py-2 px-4 text-sm rounded-lg border border-white/15 text-zinc-300 hover:bg-white/5"
                    >
                      Volver a local
                    </button>
                  </div>
                  <p className="text-[11px] text-zinc-500">
                    {apiBaseUrlActive
                      ? `Activa ahora: ${apiBaseUrlActive}`
                      : 'Activa ahora: local/proxy'}
                  </p>
                  {apiBaseUrlMessage && (
                    <p className={`text-[11px] ${
                      apiBaseUrlMessageType === 'success'
                        ? 'text-emerald-400'
                        : apiBaseUrlMessageType === 'error'
                          ? 'text-rose-400'
                          : 'text-zinc-500'
                    }`}>
                      {apiBaseUrlMessage}
                    </p>
                  )}
                </div>
              </div>

              <div className="glass-panel p-6 mt-8">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold">Integración social</h2>
                  <span className="text-[10px] bg-white/5 border border-white/5 px-2 py-0.5 rounded text-zinc-500 uppercase tracking-wider">Opcional</span>
                </div>
                <p className="text-xs text-zinc-500 mb-6 leading-relaxed">
                  Publica automáticamente tus clips en TikTok, Instagram Reels y YouTube Shorts con <strong>Upload-Post</strong>.
                  Incluye un <strong>plan gratuito</strong> (sin tarjeta de crédito).
                  Si prefieres, puedes omitir esta parte y descargar/subir tus videos manualmente.
                </p>
                <div className="space-y-4">
                  <label className="block text-sm text-zinc-400">API Key de Upload-Post</label>
                  <div className="flex gap-2">
                    <input
                      type="password"
                      value={uploadPostKey}
                      onChange={(e) => setUploadPostKey(e.target.value)}
                      className="input-field"
                      placeholder="ey..."
                    />
                    <button onClick={fetchUserProfiles} className="btn-primary py-2 px-4 text-sm">
                      Conectar
                    </button>
                  </div>
                  <p className="text-xs text-zinc-500 leading-relaxed">
                    Conecta tu cuenta de Upload-Post para publicar con un clic.
                    <div className="mt-3 grid grid-cols-1 sm:grid-cols-3 gap-2">
                      <a href="https://app.upload-post.com/login" target="_blank" rel="noopener noreferrer" className="p-2 border border-white/5 rounded-lg hover:bg-white/5 transition-colors flex flex-col gap-1">
                        <span className="text-zinc-400 font-medium">1. Login</span>
                        <span className="text-[10px] text-zinc-600">Registra tu cuenta</span>
                      </a>
                      <a href="https://app.upload-post.com/manage-users" target="_blank" rel="noopener noreferrer" className="p-2 border border-white/5 rounded-lg hover:bg-white/5 transition-colors flex flex-col gap-1">
                        <span className="text-zinc-400 font-medium">2. Perfiles</span>
                        <span className="text-[10px] text-zinc-600">Crear y conectar</span>
                      </a>
                      <a href="https://app.upload-post.com/api-keys" target="_blank" rel="noopener noreferrer" className="p-2 border border-white/5 rounded-lg hover:bg-white/5 transition-colors flex flex-col gap-1">
                        <span className="text-zinc-400 font-medium">3. API Key</span>
                        <span className="text-[10px] text-zinc-600">Generar llave</span>
                      </a>
                    </div>
                    <br />
                    <span className="text-zinc-600 italic">
                      Las llaves se guardan solo en tu navegador. Se envían al backend únicamente para procesar tu solicitud y no se almacenan del lado del servidor.
                    </span>
                  </p>
                </div>
              </div>

              <div className="glass-panel p-6 mt-8">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-lg font-semibold">Kit de marca (Subtítulos)</h2>
                  <span className="text-[10px] bg-white/5 border border-white/5 px-2 py-0.5 rounded text-zinc-500 uppercase tracking-wider">V1</span>
                </div>
                <p className="text-xs text-zinc-500 mb-6 leading-relaxed">
                  Define tu estilo de subtítulos por defecto una sola vez. Cada clip abrirá el modal de subtítulos con este preset cargado.
                </p>

                <div className="space-y-4">
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    <div>
                      <label className="block text-xs text-zinc-400 mb-1">Nombre del preset</label>
                      <input
                        type="text"
                        value={brandKit.name}
                        onChange={(e) => handleBrandKitFieldChange('name', e.target.value)}
                        className="input-field"
                        placeholder="Mi marca"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-zinc-400 mb-1">Posición</label>
                      <select
                        value={brandKit.subtitle_position}
                        onChange={(e) => handleBrandKitFieldChange('subtitle_position', e.target.value)}
                        className="input-field"
                      >
                        <option value="top">Arriba</option>
                        <option value="middle">Centro</option>
                        <option value="bottom">Abajo</option>
                      </select>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    <div>
                      <label className="block text-xs text-zinc-400 mb-1">Familia tipográfica</label>
                      <select
                        value={brandKit.subtitle_font_family}
                        onChange={(e) => handleBrandKitFieldChange('subtitle_font_family', e.target.value)}
                        className="input-field"
                      >
                        <option value="Impact">Impact</option>
                        <option value="Arial Black">Arial Black</option>
                        <option value="Arial">Arial</option>
                        <option value="Verdana">Verdana</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-xs text-zinc-400 mb-1">Tamaño de fuente</label>
                      <input
                        type="number"
                        min="12"
                        max="84"
                        value={brandKit.subtitle_font_size}
                        onChange={(e) => handleBrandKitFieldChange('subtitle_font_size', Number(e.target.value || 24))}
                        className="input-field"
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    <div>
                      <label className="block text-xs text-zinc-400 mb-1">Color de texto</label>
                      <input
                        type="color"
                        value={brandKit.subtitle_font_color}
                        onChange={(e) => handleBrandKitFieldChange('subtitle_font_color', e.target.value)}
                        className="w-full h-10 rounded-lg border border-white/10 bg-black/30 p-1"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-zinc-400 mb-1">Color del contorno</label>
                      <input
                        type="color"
                        value={brandKit.subtitle_stroke_color}
                        onChange={(e) => handleBrandKitFieldChange('subtitle_stroke_color', e.target.value)}
                        className="w-full h-10 rounded-lg border border-white/10 bg-black/30 p-1"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-zinc-400 mb-1">Color de caja</label>
                      <input
                        type="color"
                        value={brandKit.subtitle_box_color}
                        onChange={(e) => handleBrandKitFieldChange('subtitle_box_color', e.target.value)}
                        className="w-full h-10 rounded-lg border border-white/10 bg-black/30 p-1"
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    <div>
                      <label className="block text-xs text-zinc-400 mb-1">Grosor del contorno</label>
                      <input
                        type="number"
                        min="0"
                        max="8"
                        value={brandKit.subtitle_stroke_width}
                        onChange={(e) => handleBrandKitFieldChange('subtitle_stroke_width', Number(e.target.value || 0))}
                        className="input-field"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-zinc-400 mb-1">Opacidad de caja</label>
                      <input
                        type="number"
                        min="0"
                        max="100"
                        value={brandKit.subtitle_box_opacity}
                        onChange={(e) => handleBrandKitFieldChange('subtitle_box_opacity', Number(e.target.value || 0))}
                        className="input-field"
                      />
                    </div>
                    <div className="flex items-end">
                      <label className="inline-flex items-center gap-2 text-sm text-zinc-300">
                        <input
                          type="checkbox"
                          checked={brandKit.subtitle_bold}
                          onChange={(e) => handleBrandKitFieldChange('subtitle_bold', e.target.checked)}
                          className="w-4 h-4 rounded border-zinc-600 bg-black/50 text-primary focus:ring-primary"
                        />
                        Texto en negrita
                      </label>
                    </div>
                  </div>

                  <div className="flex items-center justify-between gap-3">
                    <div className="text-[11px] text-zinc-500">
                      Guardado automáticamente en el navegador como <span className="font-mono">{`"${brandKit.name || 'Predeterminado'}"`}</span>
                    </div>
                    <button
                      type="button"
                      onClick={handleBrandKitReset}
                      className="text-xs bg-white/10 border border-white/20 px-3 py-1.5 rounded-lg text-zinc-300 hover:bg-white/15"
                    >
                      Restablecer valores
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* View: Home / Projects */}
          {(activeTab === 'home' || (activeTab === 'projects' && projectsViewMode === 'list')) && (
            <div className="animate-[fadeIn_0.3s_ease-out] py-4 md:py-10">
              {activeTab === 'home' && (
                <>
                  <div className="max-w-4xl mx-auto text-center mb-8 md:mb-10">
                    <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 text-primary text-xs font-semibold border border-primary/20 mb-4">
                      <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
                      Motor de clips IA activo
                    </span>
                    <h1 className={theme === 'light'
                      ? 'text-4xl md:text-5xl font-extrabold text-slate-900 leading-tight'
                      : 'text-4xl md:text-5xl font-extrabold bg-gradient-to-r from-white to-white/70 bg-clip-text text-transparent leading-tight'
                    }>
                      Convierte videos largos en
                      <span className="block text-transparent bg-clip-text bg-gradient-to-r from-primary to-indigo-400">clips virales en segundos</span>
                    </h1>
                  </div>

                  <div className="max-w-6xl mx-auto">
                    <MediaInput onProcess={handleProcess} isProcessing={status === 'processing'} />
                  </div>
                  {status !== 'idle' && (
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
                </>
              )}

              {activeTab === 'projects' && (
              <div className="max-w-6xl mx-auto">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
                  <div className="flex items-center gap-3">
                    <h2 className="text-3xl font-bold text-slate-900 dark:text-white">Mis proyectos</h2>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => setProjectFilter('all')}
                      className={`px-4 py-1.5 rounded-full text-sm font-medium transition-colors border ${
                        projectFilter === 'all'
                          ? 'bg-slate-900 dark:bg-white text-white dark:text-slate-900 border-transparent'
                          : 'bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-700'
                      }`}
                    >
                      {`Todos (${projects.length})`}
                    </button>
                    <button
                      type="button"
                      onClick={() => setProjectFilter('favorites')}
                      className={`px-4 py-1.5 rounded-full text-sm font-medium transition-colors border ${
                        projectFilter === 'favorites'
                          ? 'bg-slate-900 dark:bg-white text-white dark:text-slate-900 border-transparent'
                          : 'bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-700'
                      }`}
                    >
                      {`Favoritos (${favoriteProjectsCount})`}
                    </button>
                  </div>
                </div>

                {visibleProjects.length > 0 ? (
                  <>
                    <div className="hidden md:grid grid-cols-12 gap-4 px-4 py-2 text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2">
                      <div className="col-span-5">Descripción</div>
                      <div className="col-span-2">Origen</div>
                      <div className="col-span-3">Tipo de video</div>
                      <div className="col-span-1">Ratio</div>
                      <div className="col-span-1" />
                    </div>
                    <div className="space-y-3">
                      {visibleProjects.map((project) => (
                        <div
                          key={project.job_id}
                          role="button"
                          tabIndex={0}
                          onClick={() => openSavedProject(project)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter' || e.key === ' ') {
                              e.preventDefault();
                              openSavedProject(project);
                            }
                          }}
                          className={`relative overflow-hidden rounded-2xl border bg-white dark:bg-slate-900/60 p-4 shadow-sm hover:shadow-md transition-shadow ${
                            project.job_id === jobId
                              ? 'border-primary/60 ring-2 ring-primary/20'
                              : 'border-slate-200 dark:border-slate-700'
                          }`}
                        >
                          <div className="grid grid-cols-1 md:grid-cols-12 gap-4 items-center">
                            <div className="col-span-1 md:col-span-5 flex items-start gap-4">
                              <div className="relative w-24 h-24 md:w-20 md:h-20 rounded-lg overflow-hidden border border-slate-200 dark:border-slate-700 bg-slate-100 dark:bg-slate-800 flex-shrink-0">
                                {project.thumbnail_url ? (
                                  <img src={getApiUrl(project.thumbnail_url)} alt={project.title || 'Proyecto'} className="w-full h-full object-cover" />
                                ) : project.preview_video_url ? (
                                  <video src={getApiUrl(project.preview_video_url)} className="w-full h-full object-cover" muted playsInline preload="metadata" />
                                ) : (
                                  <div className="w-full h-full flex items-center justify-center">
                                    {project.status === 'processing' ? (
                                      <Activity size={26} className="text-slate-400 animate-spin" />
                                    ) : (
                                      <FileVideo size={26} className="text-slate-400" />
                                    )}
                                  </div>
                                )}
                                <div className="absolute bottom-1 right-1 text-[10px] px-1.5 py-0.5 rounded bg-black/70 text-white backdrop-blur-sm">
                                  {project.ratio || '9:16'}
                                </div>
                              </div>
                              <div className="py-1 min-w-0">
                                {projectTitleEditJobId === project.job_id ? (
                                  <div className="flex items-center gap-2">
                                    <input
                                      type="text"
                                      value={projectTitleDraft}
                                      onChange={(e) => setProjectTitleDraft(e.target.value)}
                                      onClick={(e) => e.stopPropagation()}
                                      onKeyDown={(e) => {
                                        e.stopPropagation();
                                        if (e.key === 'Enter') {
                                          e.preventDefault();
                                          saveProjectTitle(project.job_id);
                                        } else if (e.key === 'Escape') {
                                          e.preventDefault();
                                          cancelEditProjectTitle();
                                        }
                                      }}
                                      className="input-field py-2 text-base"
                                      autoFocus
                                    />
                                    <button
                                      type="button"
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        saveProjectTitle(project.job_id);
                                      }}
                                      className="px-2 py-1 text-xs rounded-lg border border-emerald-300 bg-emerald-100 text-emerald-700 dark:bg-emerald-900/20 dark:border-emerald-700 dark:text-emerald-300"
                                    >
                                      Guardar
                                    </button>
                                    <button
                                      type="button"
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        cancelEditProjectTitle();
                                      }}
                                      className="px-2 py-1 text-xs rounded-lg border border-slate-300 text-slate-600 dark:border-slate-600 dark:text-slate-300"
                                    >
                                      Cancelar
                                    </button>
                                  </div>
                                ) : (
                                  <div className="flex items-center gap-2 min-w-0">
                                    <h3 className="text-xl md:text-2xl font-bold text-slate-900 dark:text-white truncate">{project.title || 'Proyecto'}</h3>
                                    <button
                                      type="button"
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        beginEditProjectTitle(project);
                                      }}
                                      className="p-1 rounded-md text-slate-400 hover:text-primary hover:bg-primary/10 transition-colors shrink-0"
                                      title="Editar título"
                                    >
                                      <Pencil size={14} />
                                    </button>
                                  </div>
                                )}
                                <div className="text-sm text-slate-500 dark:text-slate-400 space-y-0.5 mt-1">
                                  <p>{`Creado: ${formatProjectDate(project.created_at)}`}</p>
                                  <p>{`Expira: ${formatProjectDate(project.expires_at)}`}</p>
                                </div>
                              </div>
                            </div>

                            <div className="col-span-1 md:col-span-2">
                              <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium ${projectSourceBadgeClass(project.source_kind)}`}>
                                {project.source_kind === 'youtube' ? <Youtube size={13} /> : project.source_kind === 'url' ? <Link2 size={13} /> : <Upload size={13} />}
                                {project.source_label || 'Archivo local'}
                              </span>
                            </div>

                            <div className="col-span-1 md:col-span-3">
                              <div className="flex flex-col">
                                <span className="font-semibold text-slate-900 dark:text-white text-sm">{project.video_type || 'Topic-clips'}</span>
                                <span className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                                  {project.clip_count_actual
                                    ? `Número de clips: ${project.clip_count_actual}`
                                    : project.status === 'processing'
                                      ? 'Procesando...'
                                      : project.status === 'error'
                                        ? 'Error en procesamiento'
                                        : `Objetivo: ${project.clip_count_target || '-'} clips`}
                                </span>
                              </div>
                            </div>

                            <div className="col-span-1 md:col-span-1">
                              <span className="font-medium text-slate-700 dark:text-slate-300 text-sm">{project.ratio || '9:16'}</span>
                            </div>

                            <div className="col-span-1 md:col-span-1 flex items-center justify-end gap-1">
                              <span className={`text-[11px] px-2 py-0.5 rounded-full border ${
                                project.status === 'complete'
                                  ? 'bg-emerald-100 text-emerald-700 border-emerald-200 dark:bg-emerald-900/20 dark:text-emerald-300 dark:border-emerald-800'
                                  : project.status === 'error'
                                    ? 'bg-red-100 text-red-700 border-red-200 dark:bg-red-900/20 dark:text-red-300 dark:border-red-800'
                                    : 'bg-amber-100 text-amber-700 border-amber-200 dark:bg-amber-900/20 dark:text-amber-300 dark:border-amber-800'
                              }`}>
                                {projectStatusLabel(project.status)}
                              </span>
                              <button
                                type="button"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  toggleProjectFavorite(project.job_id);
                                }}
                                className={`p-2 rounded-full transition-colors ${
                                  project.favorite
                                    ? 'text-pink-500 bg-pink-50 dark:bg-pink-900/20'
                                    : 'text-slate-400 hover:text-pink-500 hover:bg-slate-100 dark:hover:bg-slate-800'
                                }`}
                                title={project.favorite ? 'Quitar de favoritos' : 'Agregar a favoritos'}
                              >
                                <Heart size={16} fill={project.favorite ? 'currentColor' : 'none'} />
                              </button>
                              <button
                                type="button"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  setProjectMenuJobId((prev) => prev === project.job_id ? null : project.job_id);
                                }}
                                className="p-2 rounded-full text-slate-400 hover:text-slate-600 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
                                title="Opciones"
                              >
                                <MoreHorizontal size={16} />
                              </button>
                              {projectMenuJobId === project.job_id && (
                                <div className="absolute right-2 top-11 z-10 min-w-[160px] rounded-xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-xl p-1">
                                  <button
                                    type="button"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      openSavedProject(project);
                                    }}
                                    className="w-full text-left px-3 py-2 text-sm text-slate-700 dark:text-slate-200 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800"
                                  >
                                    Abrir proyecto
                                  </button>
                                  <button
                                    type="button"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      toggleProjectFavorite(project.job_id);
                                      setProjectMenuJobId(null);
                                    }}
                                    className="w-full text-left px-3 py-2 text-sm text-slate-700 dark:text-slate-200 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800"
                                  >
                                    {project.favorite ? 'Quitar favorito' : 'Marcar favorito'}
                                  </button>
                                  <button
                                    type="button"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      beginEditProjectTitle(project);
                                    }}
                                    className="w-full text-left px-3 py-2 text-sm text-slate-700 dark:text-slate-200 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-800"
                                  >
                                    Editar título
                                  </button>
                                  <button
                                    type="button"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      removeProject(project.job_id);
                                    }}
                                    className="w-full text-left px-3 py-2 text-sm text-red-600 dark:text-red-300 rounded-lg hover:bg-red-50 dark:hover:bg-red-900/20"
                                  >
                                    Eliminar de la lista
                                  </button>
                                </div>
                              )}
                              <button
                                type="button"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  if (window.confirm('¿Eliminar este proyecto de la lista?')) {
                                    removeProject(project.job_id);
                                  }
                                }}
                                className="p-2 rounded-full text-slate-400 hover:text-red-600 dark:hover:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors"
                                title="Eliminar proyecto"
                              >
                                <Trash2 size={16} />
                              </button>
                              <button
                                type="button"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  openSavedProject(project);
                                }}
                                className="p-2 rounded-full text-slate-400 hover:text-primary hover:bg-primary/10 transition-colors"
                                title="Abrir proyecto"
                              >
                                <ChevronRight size={16} />
                              </button>
                            </div>
                          </div>
                          {project.status === 'processing' && (
                            <div className="absolute left-0 right-0 bottom-0 h-1 bg-violet-100 dark:bg-slate-700">
                              <div className="h-full w-1/3 bg-primary animate-pulse" />
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                    <button
                      type="button"
                      onClick={() => {
                        setActiveTab('home');
                        setProjectsViewMode('list');
                        window.scrollTo({ top: 0, behavior: 'smooth' });
                      }}
                      className="mt-6 w-full rounded-3xl border-2 border-dashed border-slate-300 dark:border-slate-700 p-8 text-center bg-white/60 dark:bg-slate-900/40 hover:bg-white dark:hover:bg-slate-900/70 transition-colors"
                    >
                      <div className="w-14 h-14 bg-violet-100 dark:bg-violet-900/20 rounded-full flex items-center justify-center mx-auto mb-3">
                        <PlusCircle size={24} className="text-primary" />
                      </div>
                      <p className="text-lg font-bold text-slate-900 dark:text-white">Crear nuevo proyecto</p>
                      <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">Sube un archivo o pega una URL para comenzar.</p>
                    </button>
                  </>
                ) : (
                  <div className="rounded-3xl border-2 border-dashed border-slate-300 dark:border-slate-700 p-10 text-center bg-white/60 dark:bg-slate-900/40">
                    <div className="w-16 h-16 bg-violet-100 dark:bg-violet-900/20 rounded-full flex items-center justify-center mx-auto mb-4">
                      <PlusCircle size={28} className="text-primary" />
                    </div>
                    <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-2">Crea tu primer proyecto</h3>
                    <p className="text-sm text-slate-500 dark:text-slate-400 max-w-md mx-auto">
                      Pega un enlace de YouTube o sube un video para generar clips virales automáticamente.
                    </p>
                  </div>
                )}
              </div>
              )}
            </div>
          )}

          {activeTab === 'projects' && projectsViewMode === 'detail' && studioContext && (
            <div className="h-[calc(100vh-7.8rem)] min-h-[720px] animate-[fadeIn_0.22s_ease-out]">
              <ClipStudioModal
                isOpen
                standalone
                onClose={closeClipStudio}
                jobId={studioContext.jobId}
                clipIndex={studioContext.clipIndex}
                clip={studioContext.clip}
                currentVideoUrl={studioContext.currentVideoUrl}
                onApplied={handleStudioApplied}
              />
            </div>
          )}

          {/* View: Processing / Results (Split View) */}
          {activeTab === 'projects' && projectsViewMode === 'detail' && !studioContext && (status === 'processing' || status === 'complete' || status === 'error') && (
            <div className="min-h-[620px] flex flex-col gap-4 animate-[fadeIn_0.3s_ease-out]">
              <div className="w-full border border-slate-200 dark:border-white/10 bg-white/90 dark:bg-black/20 rounded-2xl p-4 shadow-sm">
                <div className="flex flex-col gap-4">
                  <div className="flex items-center justify-between gap-3">
                    <div className="flex items-center gap-2 min-w-0">
                      <button
                        type="button"
                        onClick={() => setProjectsViewMode('list')}
                        className="inline-flex items-center gap-1.5 px-2 py-1 rounded-lg border border-slate-300 dark:border-slate-600 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors text-xs"
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
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={handleRetryJob}
                        disabled={!jobId || status === 'processing' || isRetryingJob}
                        className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border border-slate-300 dark:border-slate-600 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors text-[11px] disabled:opacity-50 disabled:cursor-not-allowed"
                        title="Reprocesar proyecto"
                      >
                        <RefreshCw size={12} className={isRetryingJob ? 'animate-spin' : ''} />
                        {isRetryingJob ? 'Reprocesando...' : 'Recargar'}
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          if (!jobId) return;
                          if (window.confirm('¿Eliminar este proyecto?')) {
                            removeProject(jobId);
                          }
                        }}
                        disabled={!jobId}
                        className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border border-red-300 dark:border-red-700 text-red-600 dark:text-red-300 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors text-[11px] disabled:opacity-50 disabled:cursor-not-allowed"
                        title="Eliminar proyecto"
                      >
                        <Trash2 size={12} />
                        Eliminar
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          setIsPollingPaused((prev) => {
                            const next = !prev;
                            setLogs((current) => [...current, next ? 'Actualización automática pausada.' : 'Actualización automática reanudada.']);
                            return next;
                          });
                        }}
                        disabled={status !== 'processing'}
                        className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border border-slate-300 dark:border-slate-600 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors text-[11px] disabled:opacity-50 disabled:cursor-not-allowed"
                        title={isPollingPaused ? 'Reanudar actualización automática' : 'Pausar actualización automática'}
                      >
                        {isPollingPaused ? <Play size={12} /> : <Pause size={12} />}
                        {isPollingPaused ? 'Reanudar' : 'Pausar'}
                      </button>
                      <span className={`text-[11px] px-2.5 py-1 rounded-full border font-semibold ${
                        status === 'processing'
                          ? 'bg-primary/10 border-primary/20 text-primary'
                          : status === 'complete'
                            ? 'bg-emerald-100 dark:bg-green-500/10 border-emerald-200 dark:border-green-500/20 text-emerald-700 dark:text-green-400'
                            : 'bg-red-100 dark:bg-red-500/10 border-red-200 dark:border-red-500/20 text-red-700 dark:text-red-400'
                      }`}>
                        {status === 'processing' ? (isPollingPaused ? 'PAUSADO' : 'EN PROCESO') : status === 'complete' ? 'COMPLETADO' : 'ERROR'}
                      </span>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-xs md:text-sm">
                      <span className="font-medium text-slate-700 dark:text-slate-200">{processingTimeline.headline}</span>
                      <span className="text-slate-500 dark:text-slate-400">{processingTimeline.progressPercent}%</span>
                    </div>
                    <div className="w-full h-2.5 rounded-full bg-slate-200 dark:bg-slate-700 overflow-hidden">
                      <div
                        className={`h-full transition-all duration-500 ${
                          status === 'error' ? 'bg-red-400' : 'bg-gradient-to-r from-primary to-indigo-400'
                        }`}
                        style={{ width: `${processingTimeline.progressPercent}%` }}
                      />
                    </div>
                  </div>

                  <div className="overflow-x-auto custom-scrollbar pb-1">
                    <div className="flex items-center gap-2 min-w-max">
                      {processingTimeline.visibleSteps.map((step) => {
                        const unit = 100 / Math.max(1, processingTimeline.totalSteps || 1);
                        const stepStart = (step.index - 1) * unit;
                        const activeStepProgress = Math.max(
                          8,
                          Math.min(98, ((processingTimeline.progressPercent - stepStart) / unit) * 100)
                        );
                        const ringPercent = step.state === 'done'
                          ? 100
                          : step.state === 'active'
                            ? activeStepProgress
                            : 0;
                        const ringCircumference = 2 * Math.PI * 11;
                        const ringOffset = ringCircumference * (1 - (ringPercent / 100));

                        return (
                          <div
                            key={step.key}
                            className={`px-3 py-2 rounded-xl border text-xs flex items-center gap-2 min-w-[210px] ${
                              step.state === 'done'
                                ? 'border-emerald-200 dark:border-emerald-800 bg-emerald-50 dark:bg-emerald-900/20 text-emerald-700 dark:text-emerald-300'
                                : step.state === 'active'
                                  ? 'border-primary/35 bg-primary/10 text-primary shadow-[0_0_0_1px_rgba(139,92,246,0.18)]'
                                  : step.state === 'error'
                                    ? 'border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300'
                                    : 'border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/60 text-slate-500 dark:text-slate-400'
                            }`}
                          >
                            <div className="relative w-6 h-6 shrink-0">
                              <svg className={`absolute inset-0 -rotate-90 ${step.state === 'active' ? 'animate-pulse' : ''}`} viewBox="0 0 24 24">
                                <circle
                                  cx="12"
                                  cy="12"
                                  r="11"
                                  fill="none"
                                  stroke="currentColor"
                                  strokeOpacity="0.2"
                                  strokeWidth="2"
                                />
                                <circle
                                  cx="12"
                                  cy="12"
                                  r="11"
                                  fill="none"
                                  stroke="currentColor"
                                  strokeWidth="2"
                                  strokeLinecap="round"
                                  strokeDasharray={ringCircumference}
                                  strokeDashoffset={ringOffset}
                                  style={{ transition: 'stroke-dashoffset 350ms ease-out' }}
                                />
                              </svg>
                              <div className="absolute inset-0 flex items-center justify-center text-[11px] font-semibold">
                                {step.state === 'done' ? <Check size={12} /> : step.index}
                              </div>
                            </div>
                            <span className="truncate">{step.label}</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  <div className="rounded-lg border border-slate-200 dark:border-white/10 overflow-hidden">
                    <button
                      onClick={() => setLogsVisible(!logsVisible)}
                      className="w-full px-3 py-2.5 text-left flex items-center justify-between bg-slate-50 dark:bg-white/5 hover:bg-slate-100 dark:hover:bg-white/10 transition-colors"
                    >
                      <span className="text-xs font-mono text-slate-600 dark:text-zinc-300 flex items-center gap-2">
                        <Terminal size={12} /> Logs del sistema (opcional)
                      </span>
                      <ChevronDown size={14} className={`text-zinc-500 transition-transform ${logsVisible ? '' : '-rotate-90'}`} />
                    </button>
                    {logsVisible && (
                      <div className="max-h-36 overflow-y-auto p-3 font-mono text-xs space-y-1.5 custom-scrollbar text-slate-600 dark:text-zinc-400 bg-white dark:bg-[#0c0c0e]">
                        {logs.length === 0 && (
                          <div className="text-slate-400 dark:text-zinc-500">Aún no hay logs.</div>
                        )}
                        {logs.map((log, i) => (
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
                <div className="w-full flex flex-col border border-slate-200 dark:border-white/10 bg-white/85 dark:bg-background rounded-2xl p-5 transition-all duration-700 ease-in-out shadow-sm">
                  <div className="mb-5 shrink-0">
                    <div className="flex items-center gap-2 text-[11px] text-slate-500 dark:text-zinc-500 mb-1">
                      <span>Proyecto</span>
                      <ChevronRight size={12} />
                      <span className="truncate">{processingProjectName}</span>
                    </div>
                    <h2 className="text-lg font-semibold flex items-center gap-2 text-slate-900 dark:text-white">
                      <Sparkles className="text-yellow-400" size={20} />
                      Clips generados
                      {sortedClips.length > 0 && (
                        <span className="text-xs bg-slate-100 dark:bg-white/10 text-slate-700 dark:text-white px-2 py-0.5 rounded-full ml-auto border border-slate-200 dark:border-transparent">
                          {visibleClips.length}/{sortedClips.length} Clips
                        </span>
                      )}
                    </h2>
                    <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
                      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md border border-white/10 bg-white/5 text-zinc-300">
                        <FileVideo size={12} />
                        {`Fuente: ${processingSourceLabel}`}
                      </span>
                      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md border border-white/10 bg-white/5 text-zinc-300">
                        <Scissors size={12} />
                        {`Objetivo: ${processingMedia?.clipCount || '-'} clips`}
                      </span>
                      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md border border-white/10 bg-white/5 text-zinc-300">
                        <LayoutDashboard size={12} />
                        {`Ratio: ${processingMedia?.aspectRatio === '16:9' ? '16:9' : '9:16'}`}
                      </span>
                      {results?.cost_analysis && (
                        <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md border border-green-500/30 bg-green-500/10 text-green-300" title={`Entrada: ${results.cost_analysis.input_tokens} | Salida: ${results.cost_analysis.output_tokens}`}>
                          {`Costo: $${results.cost_analysis.total_cost.toFixed(5)}`}
                        </span>
                      )}
                    </div>
                  </div>

                {sortedClips.length > 0 && (
                  <div className="mb-4 shrink-0 flex flex-wrap items-center gap-2">
                    <span className="text-xs text-zinc-500">Orden:</span>
                    <select
                      value={clipSort}
                      onChange={(e) => setClipSort(e.target.value)}
                      className="text-xs bg-white/5 border border-white/10 rounded-md px-2 py-1 text-zinc-200"
                    >
                      <option value="top">Mayor puntaje</option>
                      <option value="balanced">Línea de tiempo</option>
                      <option value="safe">Más seguros</option>
                    </select>
                    <span className="text-xs text-zinc-500 ml-1">Filtro:</span>
                    <select
                      value={clipFilter}
                      onChange={(e) => setClipFilter(e.target.value)}
                      className="text-xs bg-white/5 border border-white/10 rounded-md px-2 py-1 text-zinc-200"
                    >
                      <option value="all">Todos</option>
                      <option value="top">Alto (80+)</option>
                      <option value="medium">Medio (65-79)</option>
                      <option value="low">Bajo (&lt;65)</option>
                    </select>
                    <span className="text-xs text-zinc-500">Etiqueta:</span>
                    <select
                      value={clipTagFilter}
                      onChange={(e) => setClipTagFilter(e.target.value)}
                      className="text-xs bg-white/5 border border-white/10 rounded-md px-2 py-1 text-zinc-200"
                    >
                      <option value="all">Todas</option>
                      {availableTags.map((tag) => (
                        <option key={tag} value={tag}>{tag}</option>
                      ))}
                    </select>
                    <span className="text-xs text-zinc-500">Estrategia:</span>
                    <select
                      value={batchStrategy}
                      onChange={(e) => applyBatchStrategy(e.target.value)}
                      className="text-xs bg-white/5 border border-white/10 rounded-md px-2 py-1 text-zinc-200"
                    >
                      <option value="growth">Crecimiento</option>
                      <option value="balanced">Balanceada</option>
                      <option value="conservative">Conservadora</option>
                      <option value="custom">Personalizada</option>
                    </select>
                    <span className="text-xs text-zinc-500 ml-1">N clips:</span>
                    <input
                      type="number"
                      min="1"
                      max="10"
                      value={batchTopCount}
                      onChange={(e) => {
                        setBatchStrategy('custom');
                        setBatchTopCount(Math.max(1, Math.min(10, Number(e.target.value || 1))));
                      }}
                      className="w-16 text-xs bg-white/5 border border-white/10 rounded-md px-2 py-1 text-zinc-200"
                    />
                    <span className="text-xs text-zinc-500">Inicia en:</span>
                    <select
                      value={batchStartDelayMinutes}
                      onChange={(e) => {
                        setBatchStrategy('custom');
                        setBatchStartDelayMinutes(Number(e.target.value));
                      }}
                      className="text-xs bg-white/5 border border-white/10 rounded-md px-2 py-1 text-zinc-200"
                    >
                      <option value={0}>ahora</option>
                      <option value={5}>5m</option>
                      <option value={15}>15m</option>
                      <option value={30}>30m</option>
                      <option value={60}>60m</option>
                    </select>
                    <span className="text-xs text-zinc-500">Cada:</span>
                    <select
                      value={batchIntervalMinutes}
                      onChange={(e) => {
                        setBatchStrategy('custom');
                        setBatchIntervalMinutes(Number(e.target.value));
                      }}
                      className="text-xs bg-white/5 border border-white/10 rounded-md px-2 py-1 text-zinc-200"
                    >
                      <option value={15}>15m</option>
                      <option value={30}>30m</option>
                      <option value={60}>60m</option>
                      <option value={120}>120m</option>
                      <option value={240}>240m</option>
                    </select>
                    <span className="text-xs text-zinc-500">Alcance:</span>
                    <select
                      value={batchScope}
                      onChange={(e) => {
                        setBatchStrategy('custom');
                        setBatchScope(e.target.value);
                      }}
                      className="text-xs bg-white/5 border border-white/10 rounded-md px-2 py-1 text-zinc-200"
                    >
                      <option value="visible">Visible</option>
                      <option value="global">Global</option>
                    </select>
                    <button
                      onClick={handleQueueTopClips}
                      disabled={isBatchScheduling || (batchScope === 'global' ? sortedClips.length === 0 : visibleClips.length === 0)}
                      className="ml-1 text-xs bg-primary/20 border border-primary/40 text-primary rounded-md px-2 py-1 hover:bg-primary/30 disabled:opacity-50 disabled:cursor-not-allowed"
                      title="Programa lote usando N clips e intervalo actual"
                    >
                      {isBatchScheduling ? 'Encolando...' : `Encolar ${Math.max(1, Math.min(10, Number(batchTopCount) || 1))}`}
                    </button>
                    <button
                      onClick={handleExportPack}
                      disabled={isExportingPack || !jobId}
                      className="text-xs bg-emerald-500/20 border border-emerald-500/40 text-emerald-300 rounded-md px-2 py-1 hover:bg-emerald-500/30 disabled:opacity-50 disabled:cursor-not-allowed"
                      title="Exportar paquete para agencia (zip)"
                    >
                      {isExportingPack ? 'Exportando...' : 'Exportar paquete'}
                    </button>
                  </div>
                )}

                {sortedClips.length > 0 && (
                  <div className="mb-4 rounded-lg border border-white/10 bg-white/[0.02] p-3">
                    <div className="flex items-center gap-2">
                      <Search size={14} className="text-zinc-400" />
                      <select
                        value={clipSearchModePreset}
                        onChange={(e) => setClipSearchModePreset(e.target.value)}
                        className="text-xs bg-black/30 border border-white/10 rounded-md px-2 py-1.5 text-zinc-200 shrink-0"
                        title="Modo de búsqueda: Exacta prioriza precisión, Amplia prioriza cobertura."
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
                        className="flex-1 text-xs bg-black/30 border border-white/10 rounded-md px-2 py-1.5 text-zinc-200"
                      />
                      <button
                        onClick={handleClipSearch}
                        disabled={isSearchingClips || !clipSearchQuery.trim()}
                        className="text-xs bg-white/10 border border-white/20 rounded-md px-2 py-1.5 text-zinc-200 hover:bg-white/15 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {isSearchingClips ? 'Buscando...' : 'Buscar'}
                      </button>
                    </div>
                    <div className="mt-2 grid grid-cols-1 lg:grid-cols-4 gap-2">
                      <select
                        value={clipSearchChapterFilter}
                        onChange={(e) => setClipSearchChapterFilter(e.target.value)}
                        className="text-xs bg-black/30 border border-white/10 rounded-md px-2 py-1.5 text-zinc-200"
                        title="Limita búsqueda a un capítulo detectado automáticamente"
                      >
                        <option value="-1">Todos los capítulos</option>
                        {clipSearchChapters.map((chapter) => (
                          <option key={`chapter-filter-${chapter.chapter_index}`} value={String(chapter.chapter_index)}>
                            {`#${chapter.chapter_index + 1} ${chapter.title || 'Capítulo'} (${chapter.start}s-${chapter.end}s)`}
                          </option>
                        ))}
                      </select>
                      <input
                        type="number"
                        min="0"
                        step="1"
                        value={clipSearchStartTime}
                        onChange={(e) => setClipSearchStartTime(e.target.value)}
                        placeholder="Desde seg (opcional)"
                        className="text-xs bg-black/30 border border-white/10 rounded-md px-2 py-1.5 text-zinc-200"
                      />
                      <input
                        type="number"
                        min="0"
                        step="1"
                        value={clipSearchEndTime}
                        onChange={(e) => setClipSearchEndTime(e.target.value)}
                        placeholder="Hasta seg (opcional)"
                        className="text-xs bg-black/30 border border-white/10 rounded-md px-2 py-1.5 text-zinc-200"
                      />
                      <select
                        value={clipSearchSpeakerFilter}
                        onChange={(e) => setClipSearchSpeakerFilter(e.target.value)}
                        className="text-xs bg-black/30 border border-white/10 rounded-md px-2 py-1.5 text-zinc-200"
                        title="Filtra por hablante si hay diarización"
                      >
                        <option value="all">Todos los hablantes</option>
                        {availableSearchSpeakers.map((speaker) => (
                          <option key={`speaker-filter-${speaker}`} value={speaker}>{speaker}</option>
                        ))}
                      </select>
                    </div>

                    {clipSearchError && (
                      <p className="mt-2 text-[11px] text-red-300">{clipSearchError}</p>
                    )}

                    {!clipSearchError && (clipSearchResults.length > 0 || clipSearchChapters.length > 0 || clipHybridShortlist.length > 0) && (
                      <div className="mt-2 flex flex-wrap items-center gap-2 text-[10px] text-zinc-400">
                        <span className={`uppercase tracking-wider px-1.5 py-0.5 rounded border ${
                          clipSearchProvider === 'gemini'
                            ? 'bg-green-500/15 border-green-500/30 text-green-300'
                            : 'bg-zinc-500/15 border-zinc-500/30 text-zinc-300'
                        }`}>
                          {`semántica: ${clipSearchProvider}`}
                        </span>
                        <span className="uppercase tracking-wider px-1.5 py-0.5 rounded border bg-blue-500/15 border-blue-500/30 text-blue-300">
                          {`intención: ${clipSearchMode}`}
                        </span>
                        <span className="uppercase tracking-wider px-1.5 py-0.5 rounded border bg-cyan-500/15 border-cyan-500/30 text-cyan-300">
                          {`modo: ${clipSearchModePreset}`}
                        </span>
                        {clipSearchRelaxed && (
                          <span className="uppercase tracking-wider px-1.5 py-0.5 rounded border bg-amber-500/15 border-amber-500/30 text-amber-300">
                            relajado
                          </span>
                        )}
                        {clipSearchKeywords.length > 0 && (
                          <span className="truncate">
                            {`palabras clave: ${clipSearchKeywords.join(', ')}`}
                          </span>
                        )}
                        {clipSearchPhrases.length > 0 && (
                          <span className="truncate text-zinc-500">
                            {`frases: ${clipSearchPhrases.join(' | ')}`}
                          </span>
                        )}
                        {clipSearchScope?.applied && (
                          <span className="truncate text-emerald-300">
                            {`alcance: ${clipSearchScope.start ?? '-'}s-${clipSearchScope.end ?? '-'}s${clipSearchScope.speaker ? ` | hablante ${clipSearchScope.speaker}` : ''}${clipSearchScope.chapter?.chapter_index !== undefined ? ` | cap #${Number(clipSearchScope.chapter.chapter_index) + 1}` : ''}`}
                          </span>
                        )}
                      </div>
                    )}

                    {clipSearchResults.length > 0 && (
                      <div className="mt-2 max-h-36 overflow-y-auto border border-white/10 rounded bg-black/20 divide-y divide-white/5">
                        {clipSearchResults.map((m, i) => (
                          <div key={`${m.start}-${m.end}-${i}`} className="px-2 py-1.5 text-[11px] flex items-center justify-between gap-2">
                            <div className="min-w-0">
                              <div className="text-zinc-200 truncate">{m.snippet || `Coincidencia ${i + 1}`}</div>
                              <div className="text-zinc-500">{`${m.start}s - ${m.end}s | híbrido ${m.match_score} | sem ${m.semantic_score ?? '-'} | clave ${m.keyword_score ?? '-'} | viral ${m.virality_boost ?? '-'}`}</div>
                              {Array.isArray(m.speakers) && m.speakers.length > 0 && (
                                <div className="text-zinc-500 truncate">{`hablante: ${m.speakers.join(', ')}`}</div>
                              )}
                            </div>
                            <button
                              onClick={() => handleClipPlay(m.start)}
                              className="shrink-0 text-[11px] bg-primary/20 border border-primary/40 text-primary rounded px-2 py-1 hover:bg-primary/30"
                            >
                              Reproducir
                            </button>
                          </div>
                        ))}
                      </div>
                    )}

                    {clipSearchChapters.length > 0 && (
                      <div className="mt-2 border border-white/10 rounded bg-black/20">
                        <div className="px-2 py-1 text-[10px] uppercase tracking-wider text-zinc-400 border-b border-white/10">
                          Capítulos automáticos
                        </div>
                        <div className="max-h-36 overflow-y-auto divide-y divide-white/5">
                          {clipSearchChapters.map((chapter) => (
                            <div key={`chapter-${chapter.chapter_index}-${chapter.start}`} className="px-2 py-1.5 text-[11px] flex items-center justify-between gap-2">
                              <div className="min-w-0">
                                <div className="text-zinc-200 truncate">{chapter.title || `Capítulo ${chapter.chapter_index + 1}`}</div>
                                <div className="text-zinc-500 truncate">{`${chapter.start}s - ${chapter.end}s${Array.isArray(chapter.keywords) && chapter.keywords.length > 0 ? ` | ${chapter.keywords.join(', ')}` : ''}`}</div>
                              </div>
                              <div className="shrink-0 flex items-center gap-1.5">
                                <button
                                  onClick={() => {
                                    setClipSearchChapterFilter(String(chapter.chapter_index));
                                    setClipSearchStartTime('');
                                    setClipSearchEndTime('');
                                  }}
                                  className="text-[11px] bg-emerald-500/15 border border-emerald-500/30 text-emerald-300 rounded px-2 py-1 hover:bg-emerald-500/25"
                                >
                                  Acotar
                                </button>
                                <button
                                  onClick={() => handleClipPlay(chapter.start)}
                                  className="text-[11px] bg-white/10 border border-white/20 text-zinc-200 rounded px-2 py-1 hover:bg-white/15"
                                >
                                  Reproducir
                                </button>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {clipHybridShortlist.length > 0 && (
                      <div className="mt-2 border border-white/10 rounded bg-black/20">
                        <div className="px-2 py-1 text-[10px] uppercase tracking-wider text-zinc-400 border-b border-white/10">
                          Lista corta híbrida (Semántica + puntaje de viralidad)
                        </div>
                        <div className="max-h-36 overflow-y-auto divide-y divide-white/5">
                          {clipHybridShortlist.map((item, i) => (
                            <div key={`shortlist-${item.clip_index}-${i}`} className="px-2 py-1.5 text-[11px] flex items-center justify-between gap-2">
                              <div className="min-w-0">
                                <div className="text-zinc-200 truncate">{item.title || `Clip ${item.clip_index + 1}`}</div>
                                <div className="text-zinc-500 truncate">{`${item.start}s - ${item.end}s | híbrido ${item.hybrid_score} | viralidad ${item.virality_score}`}</div>
                              </div>
                              <button
                                onClick={() => handleClipPlay(item.start)}
                                className="shrink-0 text-[11px] bg-primary/15 border border-primary/30 text-primary rounded px-2 py-1 hover:bg-primary/25"
                              >
                                Reproducir
                              </button>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {status === 'complete' && jobId && (
                  <div className="mb-4 rounded-lg border border-white/10 bg-white/[0.02] p-3">
                    <div className="flex items-center gap-2 mb-2">
                      <History size={14} className="text-zinc-400" />
                      <span className="text-xs text-zinc-300 font-medium">Sincronía de transcript</span>
                      <span className="text-[10px] text-zinc-500">{`${visibleTranscriptSegments.length}/${transcriptTotal || transcriptSegments.length} segmentos`}</span>
                      {transcriptHasSpeakers && (
                        <span className="text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded border bg-violet-500/15 border-violet-500/30 text-violet-300">
                          etiquetas de speaker
                        </span>
                      )}
                      <button
                        onClick={() => loadTranscriptSegments(jobId)}
                        disabled={isLoadingTranscript}
                        className="ml-auto text-[11px] bg-white/10 border border-white/20 rounded px-2 py-1 text-zinc-200 hover:bg-white/15 disabled:opacity-50"
                      >
                        {isLoadingTranscript ? 'Cargando...' : 'Recargar'}
                      </button>
                    </div>
                    <div className="flex items-center gap-2">
                      <input
                        type="text"
                        value={transcriptFilter}
                        onChange={(e) => setTranscriptFilter(e.target.value)}
                        placeholder="Filtrar transcript..."
                        className="flex-1 text-xs bg-black/30 border border-white/10 rounded-md px-2 py-1.5 text-zinc-200"
                      />
                    </div>
                    {transcriptError && (
                      <p className="mt-2 text-[11px] text-red-300">{transcriptError}</p>
                    )}
                    {!transcriptError && visibleTranscriptSegments.length > 0 && (
                      <div className="mt-2 max-h-56 overflow-y-auto border border-white/10 rounded bg-black/20 divide-y divide-white/5">
                        {visibleTranscriptSegments.map((seg) => (
                          <div key={`seg-${seg.segment_index}-${seg.start}`} className="px-2 py-1.5 text-[11px] flex items-start justify-between gap-2">
                            <div className="min-w-0">
                              <div className="text-zinc-500">
                                {`${formatTimelineTime(seg.start)} - ${formatTimelineTime(seg.end)}`}
                                {seg.speaker ? ` | ${seg.speaker}` : ''}
                              </div>
                              <div className="text-zinc-200 line-clamp-2">{seg.text}</div>
                            </div>
                            <button
                              onClick={() => handleClipPlay(seg.start)}
                              className="shrink-0 text-[11px] bg-primary/20 border border-primary/40 text-primary rounded px-2 py-1 hover:bg-primary/30"
                            >
                              Reproducir
                            </button>
                          </div>
                        ))}
                      </div>
                    )}
                    {!transcriptError && !isLoadingTranscript && visibleTranscriptSegments.length === 0 && (
                      <p className="mt-2 text-[11px] text-zinc-500">No se encontraron segmentos para este filtro.</p>
                    )}
                  </div>
                )}

                {batchScheduleReport && (
                  <div className={`mb-4 text-xs rounded-lg border px-3 py-2 ${
                    batchScheduleReport.failures.length === 0
                      ? 'bg-green-500/10 border-green-500/30 text-green-300'
                      : 'bg-amber-500/10 border-amber-500/30 text-amber-200'
                  }`}>
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <p>{`Lote programado: ${batchScheduleReport.success}/${batchScheduleReport.total} en cola.`}</p>
                      <button
                        onClick={handleBatchReportCsvDownload}
                        className="text-[11px] bg-white/10 border border-white/20 rounded px-2 py-1 hover:bg-white/15"
                      >
                        Exportar CSV del lote
                      </button>
                    </div>
                    <p className="mt-1 text-[11px] text-zinc-300">
                      {`Estrategia: ${strategyLabel(batchScheduleReport.strategy || 'custom')} | Alcance: ${scopeLabel(batchScheduleReport.scope || 'visible')} | N clips: ${batchScheduleReport.top_count ?? '-'} | Cada: ${batchScheduleReport.interval_minutes ?? '-'}m`}
                    </p>
                    {batchScheduleReport.failures.length > 0 && (
                      <p className="mt-1 text-[11px] text-amber-300">
                        {batchScheduleReport.failures[0]}
                      </p>
                    )}
                    {Array.isArray(batchScheduleReport.timeline) && batchScheduleReport.timeline.length > 0 && (
                      <div className="mt-3 max-h-36 overflow-y-auto border border-white/10 rounded bg-black/20">
                        <div className="px-2 py-1 text-[10px] uppercase tracking-wider text-zinc-400 border-b border-white/10">
                          Calendario de cola
                        </div>
                        <div className="divide-y divide-white/5">
                          {batchScheduleReport.timeline.map((item, idx) => (
                            <div key={`${item.clip_index}-${item.scheduled_at}-${idx}`} className="px-2 py-1.5 text-[11px] flex items-center justify-between gap-2">
                              <div className="min-w-0">
                                <div className="text-zinc-200 truncate">{item.clip_title}</div>
                                <div className="text-zinc-500">{new Date(item.scheduled_at).toLocaleString()}</div>
                              </div>
                              <span className={`shrink-0 px-1.5 py-0.5 rounded border ${
                                item.status === 'scheduled'
                                  ? 'bg-green-500/15 border-green-500/30 text-green-300'
                                  : 'bg-red-500/15 border-red-500/30 text-red-300'
                              }`}>
                                {queueStatusLabel(item.status)}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
                {packExportReport && (
                  <div className={`mb-4 text-xs rounded-lg border px-3 py-2 ${
                    packExportReport.success === false
                      ? 'bg-red-500/10 border-red-500/30 text-red-300'
                      : 'bg-blue-500/10 border-blue-500/30 text-blue-300'
                  }`}>
                    {packExportReport.success === false
                      ? <p>{`Error exportando paquete: ${packExportReport.error}`}</p>
                      : <p>{`Paquete listo: ${packExportReport.video_files_added} videos, ${packExportReport.srt_files_added} SRT, ${packExportReport.thumbnail_files_added || 0} miniaturas, ${packExportReport.platform_variant_rows || 0} filas por plataforma, ${packExportReport.clips_in_manifest} clips en el manifiesto.`}</p>}
                  </div>
                )}

                 <div className="flex-1 overflow-y-auto custom-scrollbar p-1">
                    {visibleClips.length > 0 ? (
                       <div className="grid grid-cols-1 gap-4 pb-10">
                           {visibleClips.map((clip, i) => {
                             const clipRefIndex = Number(clip?.clip_index);
                             const focusMatch = Number.isFinite(clipRefIndex) && clipRefIndex === focusedClipIndex;
                             return (
                               <div
                                 key={`${clip.clip_index}-${clip.video_url || i}`}
                                 ref={(node) => setClipCardRef(clipRefIndex, node)}
                                 tabIndex={-1}
                                 data-clip-index={Number.isFinite(clipRefIndex) ? clipRefIndex : undefined}
                                 className={`rounded-2xl outline-none transition-shadow ${focusMatch ? 'ring-2 ring-primary/50 shadow-[0_0_0_4px_rgba(124,58,237,0.14)]' : ''}`}
                               >
                                 <ResultCard
                                   clip={clip}
                                   displayIndex={i}
                                   clipIndex={clip.clip_index}
                                   jobId={jobId}
                                   uploadPostKey={uploadPostKey}
                                   uploadUserId={uploadUserId}
                                   geminiApiKey={apiKey}
                                   onOpenStudio={openClipStudio}
                                   onPlay={(time) => handleClipPlay(time)}
                                   onPause={handleClipPause}
                                 />
                               </div>
                             );
                           })}
                       </div>
                    ) : (
                    status === 'processing' ? (
                      <div className="h-full flex flex-col items-center justify-center text-zinc-500 space-y-4 opacity-50">
                        <div className="w-12 h-12 rounded-full border-2 border-zinc-800 border-t-primary animate-spin" />
                        <p className="text-sm">Esperando clips...</p>
                      </div>
                    ) : sortedClips.length > 0 ? (
                      <div className="h-full flex flex-col items-center justify-center text-zinc-500 space-y-2">
                        <p>Ningún clip coincide con este filtro.</p>
                      </div>
                    ) : status === 'error' ? (
                      <div className="h-full flex flex-col items-center justify-center text-red-400 space-y-2">
                        <p>Generación fallida.</p>
                        {jobId && (
                          <button
                            onClick={handleRetryJob}
                            disabled={isRetryingJob}
                            className="text-xs bg-white/10 border border-white/20 rounded-md px-3 py-1.5 text-zinc-200 hover:bg-white/15 disabled:opacity-50 disabled:cursor-not-allowed"
                          >
                            {isRetryingJob ? 'Reintentando...' : 'Reintentar job'}
                          </button>
                        )}
                      </div>
                    ) : null
                  )}
                </div>
              </div>

            </div>
          </div>
          )}

        </div>
      </main>
    </div>
  );
}

export default App;
