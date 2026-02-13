const API_BASE_OVERRIDE_KEY = 'openshorts_api_base_url';

const extractFirstUrlCandidate = (value) => {
    const raw = String(value || '').trim();
    if (!raw) return '';
    const match = raw.match(/https?:\/\/[^\s"')]+/i);
    if (match?.[0]) return match[0];
    if (/^[a-z0-9.-]+\.[a-z]{2,}(\/.*)?$/i.test(raw)) return `https://${raw}`;
    return raw;
};

export const normalizeApiBaseUrl = (value) => {
    const candidate = extractFirstUrlCandidate(value);
    if (!candidate) return '';
    try {
        const url = new URL(candidate);
        const pathname = url.pathname === '/' ? '' : url.pathname.replace(/\/+$/, '');
        return `${url.origin}${pathname}`;
    } catch (_) {
        if (/^https?:\/\//i.test(candidate)) {
            return candidate.replace(/\/+$/, '');
        }
        return '';
    }
};

const readStoredApiBaseUrl = () => {
    if (typeof window === 'undefined') return '';
    return normalizeApiBaseUrl(window.localStorage.getItem(API_BASE_OVERRIDE_KEY) || '');
};

const readEnvApiBaseUrl = () => normalizeApiBaseUrl(import.meta.env.VITE_API_URL || '');

export const getApiBaseUrl = () => {
    const stored = readStoredApiBaseUrl();
    if (stored) return stored;
    return readEnvApiBaseUrl();
};

export const setApiBaseUrl = (value) => {
    const normalized = normalizeApiBaseUrl(value);
    if (typeof window !== 'undefined') {
        if (normalized) {
            window.localStorage.setItem(API_BASE_OVERRIDE_KEY, normalized);
        } else {
            window.localStorage.removeItem(API_BASE_OVERRIDE_KEY);
        }
        window.dispatchEvent(new CustomEvent('openshorts-api-base-url-changed', { detail: { apiBaseUrl: normalized } }));
    }
    return normalized;
};

export const getApiUrl = (path) => {
    if (path.startsWith('http')) return path;
    const normalizedPath = path.startsWith('/') ? path : `/${path}`;
    const apiBase = getApiBaseUrl();
    return `${apiBase}${normalizedPath}`;
};

const isNgrokBaseUrl = (apiBase) => {
    if (!apiBase) return false;
    return apiBase.includes('ngrok');
};

export const apiFetch = (path, options = {}) => {
    const url = getApiUrl(path);
    const headers = new Headers(options.headers || {});
    const apiBase = getApiBaseUrl();

    if (isNgrokBaseUrl(apiBase)) {
        headers.set('ngrok-skip-browser-warning', 'true');
    }

    return fetch(url, {
        ...options,
        headers
    });
};
