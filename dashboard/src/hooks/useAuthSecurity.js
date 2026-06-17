import { useState, useEffect, useCallback } from 'react';
import { apiFetch } from '../config';

// Simple XOR + Base64 encryption for client-side obfuscation
const SECRET_KEY = import.meta.env.VITE_ENCRYPTION_KEY || "OpenShorts-Static-Salt-Change-Me";

const encrypt = (text) => {
  if (!text) return '';
  const salted = text + '|' + SECRET_KEY;
  const xor = salted.split('').map((char, i) => 
    String.fromCharCode(char.charCodeAt(0) ^ SECRET_KEY.charCodeAt(i % SECRET_KEY.length))
  ).join('');
  return btoa(xor);
};

const decrypt = (encoded) => {
  if (!encoded) return '';
  try {
    const xor = atob(encoded);
    const salted = xor.split('').map((char, i) => 
      String.fromCharCode(char.charCodeAt(0) ^ SECRET_KEY.charCodeAt(i % SECRET_KEY.length))
    ).join('');
    const [text, salt] = salted.split('|');
    return salt === SECRET_KEY ? text : '';
  } catch (e) {
    return '';
  }
};

/**
 * Hook to manage API keys and security settings.
 * Sensitive keys are stored in sessionStorage as primary (cleared on tab close)
 * with an encrypted localStorage backup for user convenience.
 */
export const useAuthSecurity = (showToast) => {
  const _loadSensitiveKey = (sessionKey, localKey) => {
    const fromSession = sessionStorage.getItem(sessionKey);
    if (fromSession) return fromSession;
    const fromLocal = localStorage.getItem(localKey);
    return fromLocal ? decrypt(fromLocal) : '';
  };

  const _saveSensitiveKey = (sessionKey, localKey, value) => {
    if (value) {
      sessionStorage.setItem(sessionKey, value);
      localStorage.setItem(localKey, encrypt(value));
    } else {
      sessionStorage.removeItem(sessionKey);
      localStorage.removeItem(localKey);
    }
  };

  const [apiKey, setApiKey] = useState(() => _loadSensitiveKey('gemini_key_session', 'gemini_key_enc'));
  const [elevenLabsKey, setElevenLabsKey] = useState(() => _loadSensitiveKey('elevenlabs_key_session', 'elevenlabs_key_enc'));
  const [huggingfaceToken, setHuggingfaceToken] = useState(localStorage.getItem('hf_token') || '');
  
  const [uploadPostKey, setUploadPostKey] = useState(() => {
    const stored = localStorage.getItem('uploadPostKey_v3');
    if (stored) return decrypt(stored);
    return '';
  });

  const [uploadUserId, setUploadUserId] = useState(() => localStorage.getItem('uploadUserId') || '');
  const [userProfiles, setUserProfiles] = useState([]);

  // Persistance Sync
  useEffect(() => {
    _saveSensitiveKey('gemini_key_session', 'gemini_key_enc', apiKey);
    if (localStorage.getItem('gemini_key')) localStorage.removeItem('gemini_key');
  }, [apiKey]);

  useEffect(() => {
    _saveSensitiveKey('elevenlabs_key_session', 'elevenlabs_key_enc', elevenLabsKey);
    if (localStorage.getItem('elevenlabs_key')) localStorage.removeItem('elevenlabs_key');
  }, [elevenLabsKey]);

  useEffect(() => {
    if (huggingfaceToken) localStorage.setItem('hf_token', huggingfaceToken);
    else localStorage.removeItem('hf_token');
  }, [huggingfaceToken]);

  useEffect(() => {
    if (uploadPostKey) {
      localStorage.setItem('uploadPostKey_v3', encrypt(uploadPostKey));
    } else {
      localStorage.removeItem('uploadPostKey_v3');
    }
    if (uploadUserId) {
      localStorage.setItem('uploadUserId', uploadUserId);
    } else {
      localStorage.removeItem('uploadUserId');
    }
  }, [uploadPostKey, uploadUserId]);

  const fetchUserProfiles = useCallback(async () => {
    if (!uploadPostKey) return;
    try {
      const res = await apiFetch('/api/social/user', {
        headers: { 'X-Upload-Post-Key': uploadPostKey }
      });
      if (!res.ok) throw new Error("Error al consultar");
      const data = await res.json();
      if (data.profiles && data.profiles.length > 0) {
        setUserProfiles(data.profiles);
        if (!uploadUserId) setUploadUserId(data.profiles[0].username);
      } else {
        showToast('No se encontraron perfiles para esta API Key.', 'warning');
      }
    } catch (e) {
      showToast('Error consultando perfiles. Revisa la API Key.', 'error');
      console.error(e);
    }
  }, [uploadPostKey, uploadUserId, showToast]);

  useEffect(() => {
    if (uploadPostKey && userProfiles.length === 0) {
      fetchUserProfiles();
    }
  }, [uploadPostKey, userProfiles.length, fetchUserProfiles]);

  return {
    apiKey, setApiKey,
    elevenLabsKey, setElevenLabsKey,
    huggingfaceToken, setHuggingfaceToken,
    uploadPostKey, setUploadPostKey,
    uploadUserId, setUploadUserId,
    userProfiles,
    fetchUserProfiles,
    encrypt,
    decrypt
  };
};
