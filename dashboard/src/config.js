// Configuration for API endpoints
// If VITE_API_URL is set (e.g. in production), use it.
// Otherwise, default to empty string which means relative paths (proxied in dev).

export const API_BASE_URL = import.meta.env.VITE_API_URL || '';

export const getApiUrl = (path) => {
    if (path.startsWith('http')) return path;
    // Ensure path starts with / if not present
    const normalizedPath = path.startsWith('/') ? path : `/${path}`;
    return `${API_BASE_URL}${normalizedPath}`;
};

const isNgrokBaseUrl = () => {
    if (!API_BASE_URL) return false;
    return API_BASE_URL.includes('ngrok');
};

export const apiFetch = (path, options = {}) => {
    const url = getApiUrl(path);
    const headers = new Headers(options.headers || {});

    // Avoid ngrok browser warning HTML page on XHR/fetch requests.
    if (isNgrokBaseUrl()) {
        headers.set('ngrok-skip-browser-warning', 'true');
    }

    return fetch(url, {
        ...options,
        headers
    });
};
