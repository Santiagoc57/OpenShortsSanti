import { useState, useCallback } from 'react';
import { apiFetch, getApiUrl } from '../config';

export const useClipExporter = (jobId, showToast, setLogs) => {
  const [isExportingPack, setIsExportingPack] = useState(false);
  const [packExportReport, setPackExportReport] = useState(null);
  const [isBatchScheduling, setIsBatchScheduling] = useState(false);
  const [batchScheduleReport, setBatchScheduleReport] = useState(null);

  const triggerDownload = useCallback((href, filename) => {
    const a = document.createElement('a');
    a.href = href;
    if (filename) a.download = filename;
    a.rel = 'noopener noreferrer';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }, []);

  const downloadPackFile = useCallback(async (sourceUrl, filenameFallback) => {
    const safeUrl = String(sourceUrl || '').trim();
    if (!safeUrl) throw new Error('URL de paquete vacía');

    const response = await apiFetch(safeUrl, { method: 'GET' });
    if (!response.ok) {
      throw new Error(`Descarga de paquete fallida (${response.status})`);
    }

    const contentType = String(response.headers.get('content-type') || '').toLowerCase();
    if (contentType.includes('text/html') || contentType.includes('application/json')) {
      throw new Error('El servidor devolvió una respuesta no válida para descarga de ZIP');
    }

    const blob = await response.blob();
    const objectUrl = URL.createObjectURL(blob);
    try {
      triggerDownload(objectUrl, filenameFallback);
    } finally {
      setTimeout(() => URL.revokeObjectURL(objectUrl), 1500);
    }
  }, [triggerDownload]);

  const handleExportPack = useCallback(async () => {
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
          include_platform_video_variants: true,
          thumbnail_format: 'jpg',
          thumbnail_width: 1080
        })
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setPackExportReport(data);

      const href = getApiUrl(String(data?.pack_url || ''));
      const fallbackName = href.split('/').pop() || `agency_pack_${jobId}.zip`;
      try {
        await downloadPackFile(href, fallbackName);
      } catch (downloadErr) {
        setLogs((prev) => [...prev, `Descarga directa falló, abriendo enlace del pack: ${downloadErr.message}`]);
        const opened = window.open(href, '_blank', 'noopener,noreferrer');
        if (!opened) {
          triggerDownload(href, fallbackName);
        }
      }
      setLogs((prev) => [...prev, `Pack listo (${data.video_files_added} videos, ${data.srt_files_added} srt, ${data.thumbnail_files_added || 0} miniaturas, ${data.platform_video_variant_files_added || 0} variantes por plataforma).`]);
    } catch (e) {
      setPackExportReport({ success: false, error: e.message });
      setLogs((prev) => [...prev, `Error exportando paquete: ${e.message}`]);
    } finally {
      setIsExportingPack(false);
    }
  }, [jobId, downloadPackFile, triggerDownload, setLogs]);

  const handleBatchScheduleReport = useCallback(() => {
    if (!batchScheduleReport || !batchScheduleReport.timeline) return;
    const { timeline } = batchScheduleReport;
    const safe = (v) => `"${String(v || '').replaceAll('"', '""')}"`;
    const lines = [
      ['Clip #', 'Scheduled (ISO)', 'Platforms', 'Title', 'Virality', 'Status', 'Error'].join(',')
    ];
    timeline.forEach((item) => {
      lines.push([
        item.clip_index + 1,
        safe(item.scheduled_at),
        safe((item.platforms || []).join('|')),
        safe(item.clip_title),
        safe(item.virality_score),
        safe(item.status),
        safe(item.error || '')
      ].join(','));
    });

    const blob = new Blob([lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    triggerDownload(url, `batch_schedule_report_${jobId || 'job'}.csv`);
    URL.revokeObjectURL(url);
  }, [batchScheduleReport, jobId, triggerDownload]);

  const handleQueueTopClips = useCallback(async (
    candidatePool, 
    uploadPostKey, 
    uploadUserId, 
    userProfiles, 
    batchTopCount, 
    batchStartDelayMinutes, 
    batchIntervalMinutes, 
    batchScope,
    batchStrategy
  ) => {
    if (!jobId || !candidatePool || candidatePool.length === 0) return;
    if (!uploadPostKey || !uploadUserId) {
      showToast('Configura Upload-Post API Key y perfil de usuario en Configuración para usar cola batch.', 'warning');
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
  }, [jobId, showToast, setLogs]);

  return {
    isExportingPack,
    packExportReport,
    handleExportPack,
    isBatchScheduling,
    setIsBatchScheduling,
    batchScheduleReport,
    setBatchScheduleReport,
    handleBatchScheduleReport,
    handleQueueTopClips,
    downloadPackFile,
    triggerDownload
  };
};
