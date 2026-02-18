const normalizeSpace = (text) => String(text || '').replace(/\s+/g, ' ').trim();
const DEFAULT_GEMINI_MODELS = [
  'gemini-2.0-flash',
  'gemini-2.0-flash-lite',
  'gemini-1.5-flash-latest',
  'gemini-1.5-flash'
];

export const sanitizeShortTitle = (rawTitle, maxChars = 95) => {
  let text = normalizeSpace(rawTitle).replace(/[#@]/g, '').replace(/[\r\n\t]+/g, ' ').trim();
  text = text.replace(/^[-:;,.!?¡¿\s"'`]+/, '').replace(/["'`]+$/g, '').trim();
  if (text.length > maxChars) {
    const sliced = text.slice(0, maxChars);
    text = sliced.includes(' ') ? sliced.slice(0, sliced.lastIndexOf(' ')).trim() : sliced.trim();
  }
  return text;
};

const hash32 = (value) => {
  let h = 0;
  const str = String(value || '');
  for (let i = 0; i < str.length; i += 1) {
    h = ((h << 5) - h) + str.charCodeAt(i);
    h |= 0;
  }
  return Math.abs(h);
};

const FALLBACK_STOP_WORDS = new Set([
  'esto', 'esta', 'este', 'para', 'como', 'cuando', 'donde', 'sobre', 'porque',
  'video', 'clip', 'corte', 'tema', 'aqui', 'aquí', 'desde', 'entre', 'ellos',
  'ellas', 'usted', 'ustedes', 'tambien', 'también', 'siempre', 'nunca'
]);

export const buildFallbackTitleLocal = ({
  currentTitle = '',
  transcriptExcerpt = '',
  topicTags = [],
  avoidTitle = ''
}) => {
  const cleanCurrent = sanitizeShortTitle(currentTitle || 'Momento clave del video');
  const cleanAvoid = sanitizeShortTitle(avoidTitle || cleanCurrent).toLowerCase();
  const safeTags = (Array.isArray(topicTags) ? topicTags : [])
    .map((tag) => sanitizeShortTitle(tag, 24).toLowerCase())
    .filter(Boolean);

  let keyword = safeTags.find((tag) => tag.length >= 4) || '';
  if (!keyword) {
    const words = String(transcriptExcerpt || '')
      .toLowerCase()
      .normalize('NFD')
      .replace(/[\u0300-\u036f]/g, '')
      .match(/[a-z0-9]{4,}/g) || [];
    keyword = words.find((w) => !FALLBACK_STOP_WORDS.has(w)) || '';
  }

  const leads = [
    'Lo que no te contaron',
    'La parte mas fuerte',
    'El momento que explica todo',
    'Asi lo dijo sin filtro',
    'La frase que cambia el debate'
  ];
  const hooks = [
    'cambia el debate',
    'deja una alerta clara',
    'explica el punto clave',
    'abre una discusion fuerte',
    'resume lo mas importante'
  ];
  const seed = hash32(`${cleanCurrent}|${keyword}|${transcriptExcerpt}|${safeTags.join(',')}`);
  const lead = leads[seed % leads.length];
  const hook = hooks[Math.floor(seed / Math.max(1, leads.length)) % hooks.length];

  const candidates = [];
  if (keyword) {
    candidates.push(`${lead}: ${keyword} y por que ${hook}`);
    candidates.push(`${keyword}: ${hook} en este corte`);
  }
  candidates.push(`${cleanCurrent} | ${hook}`);
  candidates.push(`${lead} y por que ${hook}`);

  for (const candidate of candidates) {
    const clean = sanitizeShortTitle(candidate);
    if (!clean) continue;
    if (clean.toLowerCase() === cleanAvoid) continue;
    return clean;
  }
  return sanitizeShortTitle(cleanCurrent || 'Momento clave del video');
};

const extractGeminiText = (payload) => {
  const candidates = Array.isArray(payload?.candidates) ? payload.candidates : [];
  for (const candidate of candidates) {
    const parts = Array.isArray(candidate?.content?.parts) ? candidate.content.parts : [];
    const merged = parts
      .map((part) => normalizeSpace(part?.text || ''))
      .filter(Boolean)
      .join(' ')
      .trim();
    if (merged) return merged;
  }
  return '';
};

const parseGeminiError = async (response) => {
  const raw = await response.text();
  try {
    const data = JSON.parse(raw);
    const message = normalizeSpace(data?.error?.message || data?.message || '');
    if (message) return message;
  } catch (_) {
    // Non-JSON response body.
  }
  return normalizeSpace(raw) || `HTTP ${response.status}`;
};

const isModelUnavailableError = (message = '', status = 0) => {
  const msg = normalizeSpace(message).toLowerCase();
  if (status === 404) return true;
  if (msg.includes('models/') && msg.includes('not found')) return true;
  if (msg.includes('model') && msg.includes('not found')) return true;
  if (msg.includes('model') && msg.includes('not supported')) return true;
  if (msg.includes('unknown model')) return true;
  if (msg.includes('for api version') && msg.includes('model')) return true;
  return false;
};

export const regenerateTitleWithGemini = async ({
  apiKey,
  currentTitle,
  transcriptExcerpt = '',
  socialExcerpt = '',
  topicTags = [],
  model = ''
}) => {
  const key = normalizeSpace(apiKey);
  if (!key) {
    throw new Error('Missing Gemini API key');
  }

  const safeTitle = sanitizeShortTitle(currentTitle || 'Momento clave del video');
  const safeTranscript = normalizeSpace(transcriptExcerpt).slice(0, 420);
  const safeSocial = normalizeSpace(socialExcerpt).slice(0, 300);
  const tagsLine = (Array.isArray(topicTags) ? topicTags : [])
    .map((tag) => sanitizeShortTitle(tag, 24).toLowerCase())
    .filter(Boolean)
    .slice(0, 8)
    .join(', ');

  const prompt = [
    'Reescribe SOLO el título de este clip corto.',
    'Devuelve una sola línea sin comillas.',
    'Reglas: español neutro, 55-95 caracteres, gancho claro, sin emojis, sin hashtags.',
    `No repitas literalmente este título: ${safeTitle}.`,
    `Título actual: ${safeTitle}`,
    `Contexto social: ${safeSocial || 'n/a'}`,
    `Contexto transcript: ${safeTranscript || 'n/a'}`,
    `Etiquetas: ${tagsLine || 'n/a'}`
  ].join('\n');

  const modelCandidates = (() => {
    if (Array.isArray(model)) {
      return model.map((m) => normalizeSpace(m)).filter(Boolean);
    }
    const raw = normalizeSpace(model);
    if (!raw) return [...DEFAULT_GEMINI_MODELS];
    return raw
      .split(',')
      .map((m) => normalizeSpace(m))
      .filter(Boolean);
  })();

  let lastError = 'Gemini no devolvió un título válido.';
  for (const modelName of modelCandidates) {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(modelName)}:generateContent?key=${encodeURIComponent(key)}`;
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [{ role: 'user', parts: [{ text: prompt }] }],
        generationConfig: {
          temperature: 0.65,
          topP: 0.9,
          maxOutputTokens: 120
        }
      })
    });

    if (!response.ok) {
      const err = await parseGeminiError(response);
      if (isModelUnavailableError(err, response.status)) {
        lastError = err;
        continue;
      }
      throw new Error(err);
    }

    const data = await response.json();
    const generated = sanitizeShortTitle(extractGeminiText(data));
    if (generated) return generated;
    lastError = 'Gemini no devolvió un título válido.';
  }

  throw new Error(lastError);
};
