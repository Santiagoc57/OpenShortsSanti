import { useState, useEffect, useMemo, useCallback } from 'react';
import { apiFetch } from '../config';

export const useClipSearch = (jobId, status, results, setResults, setLogs, apiKey, clipSearchModePreset) => {
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
  const [clipSearchError, setClipSearchError] = useState(null);

  const [transcriptSegments, setTranscriptSegments] = useState([]);
  const [transcriptFilter, setTranscriptFilter] = useState('');
  const [transcriptTotal, setTranscriptTotal] = useState(0);
  const [isLoadingTranscript, setIsLoadingTranscript] = useState(false);
  const [transcriptError, setTranscriptError] = useState(null);
  const [transcriptHasSpeakers, setTranscriptHasSpeakers] = useState(false);

  const [isGeneratingTrailer, setIsGeneratingTrailer] = useState(false);

  const handleClipSearch = useCallback(async () => {
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
      if (!res.ok) throw new Error(await res.text());
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
      setLogs((prev) => [...prev, `Búsqueda de clips "${query}": ${(data.matches || []).length} coincidencias.`]);
    } catch (e) {
      setClipSearchError(e.message);
      setLogs((prev) => [...prev, `Búsqueda de clips fallida: ${e.message}`]);
    } finally {
      setIsSearchingClips(false);
    }
  }, [jobId, clipSearchQuery, clipSearchStartTime, clipSearchEndTime, clipSearchChapterFilter, clipSearchSpeakerFilter, clipSearchModePreset, apiKey, setLogs]);

  const loadTranscriptSegments = useCallback(async (targetJobId) => {
    if (!targetJobId) return;
    setIsLoadingTranscript(true);
    setTranscriptError(null);
    try {
      const res = await apiFetch(`/api/transcript/${targetJobId}?limit=1200`);
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      const segments = Array.isArray(data.segments) ? data.segments : [];
      setTranscriptSegments(segments);
      setTranscriptTotal(Number.isFinite(data.total) ? data.total : segments.length);
      setTranscriptHasSpeakers(Boolean(data.has_speaker_labels));
    } catch (e) {
      setTranscriptError(e.message);
    } finally {
      setIsLoadingTranscript(false);
    }
  }, []);

  useEffect(() => {
    if (status === 'complete' && jobId && transcriptSegments.length === 0) {
      loadTranscriptSegments(jobId);
    }
  }, [status, jobId, transcriptSegments.length, loadTranscriptSegments]);

  const handleGenerateTrailer = useCallback(async (aspectRatio) => {
    if (!jobId) return;
    setIsGeneratingTrailer(true);
    setLogs((prev) => [...prev, "Generando Super Trailer ⚡ (Súper Resumen)..."]);

    try {
      const res = await apiFetch("/api/clip/trailer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ job_id: jobId, aspect_ratio: aspectRatio })
      });

      if (!res.ok) throw new Error(await res.text());

      const data = await res.json();
      if (data.success && data.trailer_url) {
        setResults((prev) => ({ ...prev, latest_trailer_url: data.trailer_url }));
        setLogs((prev) => [...prev, `Super Trailer listo ⚡: ${data.trailer_url}`]);
      }
    } catch (e) {
      setLogs((prev) => [...prev, `Error generando Super Trailer: ${e.message}`]);
    } finally {
      setIsGeneratingTrailer(false);
    }
  }, [jobId, setResults, setLogs]);

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

  return {
    clipSearchQuery, setClipSearchQuery,
    isSearchingClips,
    clipSearchResults,
    clipSearchKeywords,
    clipSearchPhrases,
    clipSearchChapters,
    clipSearchSpeakers,
    clipHybridShortlist,
    clipSearchProvider,
    clipSearchMode,
    clipSearchRelaxed,
    clipSearchScope,
    clipSearchChapterFilter, setClipSearchChapterFilter,
    clipSearchStartTime, setClipSearchStartTime,
    clipSearchEndTime, setClipSearchEndTime,
    clipSearchSpeakerFilter, setClipSearchSpeakerFilter,
    clipSearchError,
    handleClipSearch,
    transcriptSegments,
    transcriptFilter, setTranscriptFilter,
    transcriptTotal,
    isLoadingTranscript,
    transcriptError,
    transcriptHasSpeakers,
    visibleTranscriptSegments,
    handleGenerateTrailer,
    isGeneratingTrailer,
    availableSearchSpeakers: useMemo(() => {
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
    }, [clipSearchSpeakers, transcriptSegments])
  };
};

