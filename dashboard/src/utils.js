export { getApiUrl } from './config';

export const formatTimelineTime = (seconds) => {
  const val = Number(seconds);
  if (!Number.isFinite(val) || val < 0) return '0:00';
  const total = Math.floor(val);
  const mins = Math.floor(total / 60);
  const secs = total % 60;
  return `${mins}:${String(secs).padStart(2, '0')}`;
};

export const formatProjectDate = (value) => {
  if (!value) return '-';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString();
};

export const outputModeLabel = (value) => ({
  '9:16': 'Vertical 9:16',
  '16:9': 'Horizontal 16:9',
  vertical: 'Vertical 9:16',
  horizontal: 'Horizontal 16:9'
}[value] || value || '-');

export const strategyLabel = (value) => ({
  growth: 'Crecimiento',
  balanced: 'Balanceada',
  conservative: 'Conservadora',
  custom: 'Personalizada'
}[value] || value || '-');

export const scopeLabel = (value) => ({
  visible: 'Visible',
  global: 'Global'
}[value] || value || '-');

export const queueStatusLabel = (value) => ({
  scheduled: 'programado',
  failed: 'fallido'
}[value] || value || '-');

export const projectStatusLabel = (value) => ({
  processing: 'Procesando',
  paused: 'Pausado',
  complete: 'Completado',
  error: 'Error'
}[value] || 'Procesando');

export const projectSourceBadgeClass = (value) => {
  if (value === 'youtube') return 'bg-red-100 text-red-700 dark:bg-red-900/20 dark:text-red-300';
  if (value === 'url') return 'bg-sky-100 text-sky-700 dark:bg-sky-900/20 dark:text-sky-300';
  return 'bg-amber-100 text-amber-800 dark:bg-zinc-700 dark:text-zinc-200';
};
