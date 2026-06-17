import React from 'react';
import { Settings, Shield, Globe, RefreshCw, Key, Sparkles, Check, RotateCcw } from 'lucide-react';

const SettingsView = ({
  apiKey, setApiKey,
  elevenLabsKey, setElevenLabsKey,
  huggingfaceToken, setHuggingfaceToken,
  uploadPostKey, setUploadPostKey,
  uploadUserId, setUploadUserId,
  userProfiles, fetchUserProfiles,
  apiBaseUrlInput, setApiBaseUrlInput,
  apiBaseUrlActive,
  apiBaseUrlMessage,
  apiBaseUrlMessageType,
  isTestingApiBaseUrl,
  handleSaveApiBaseUrl,
  runConnectivityCheck,
  brandKit,
  handleBrandKitFieldChange,
  captionFontOptions,
  connectivityStatus,
  isConnectivityChecking,
  handleReset,
  setActiveTab
}) => {
  const safeConnectivityStatus = connectivityStatus || { api: 'unknown' };
  const canEditApiBaseUrl = typeof setApiBaseUrlInput === 'function';
  const canCheckConnectivity = typeof runConnectivityCheck === 'function';
  const canSaveApiBaseUrl = typeof handleSaveApiBaseUrl === 'function';
  const canReset = typeof handleReset === 'function';

  return (
    <div className="max-w-6xl mx-auto py-6 md:py-12 animate-[fadeIn_0.3s_ease-out]">
      <div className="flex items-center gap-3 mb-8">
        <div className="w-12 h-12 rounded-2xl bg-primary/10 flex items-center justify-center text-primary">
          <Settings size={28} />
        </div>
        <div>
          <h2 className="text-3xl font-bold text-slate-900 dark:text-white">Configuración</h2>
          <p className="text-slate-500 dark:text-slate-400">Gestiona tus API Keys, conexiones y preferencias de marca</p>
        </div>
      </div>

      <div className="space-y-8">
        <section className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 overflow-hidden shadow-sm">
          <div className="px-6 py-4 border-b border-slate-200 dark:border-slate-700 bg-slate-50/80 dark:bg-slate-800/50 flex items-center gap-2">
            <Globe size={18} className="text-primary" />
            <h2 className="text-lg font-semibold text-slate-900 dark:text-white">Servidor y Conectividad</h2>
          </div>
          <div className="p-6 space-y-6">
            <div className="grid md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">Base URL del API (FastAPI)</label>
                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={apiBaseUrlInput || ''}
                      onChange={(e) => {
                        if (canEditApiBaseUrl) setApiBaseUrlInput(e.target.value);
                      }}
                      disabled={!canEditApiBaseUrl}
                      className="flex-1 py-2.5 px-3 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-sm focus:ring-2 focus:ring-primary/20 transition-all"
                      placeholder="http://localhost:8000"
                    />
                  </div>
                  {apiBaseUrlMessage && (
                    <p className={`mt-2 text-xs font-medium ${apiBaseUrlMessageType === 'error' ? 'text-red-500' : 'text-emerald-500'}`}>
                      {apiBaseUrlMessage}
                    </p>
                  )}
                </div>

                <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-100 dark:border-slate-700/50">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-sm font-medium text-slate-700 dark:text-slate-300">Estado del Tunel / API</span>
                    <button
                      onClick={runConnectivityCheck}
                      disabled={isConnectivityChecking || !canCheckConnectivity}
                      className="p-1.5 rounded-md hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-500 transition-colors disabled:opacity-50"
                    >
                      <RefreshCw size={14} className={isConnectivityChecking ? 'animate-spin' : ''} />
                    </button>
                  </div>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-slate-500">API Principal</span>
                      <span className={`font-semibold ${safeConnectivityStatus.api === 'online' ? 'text-emerald-500 text-white' : safeConnectivityStatus.api === 'offline' ? 'text-red-500' : 'text-amber-500'}`}>
                        {safeConnectivityStatus.api === 'online' ? 'EN LÍNEA' : safeConnectivityStatus.api === 'offline' ? 'DESCONECTADO' : 'SIN VERIFICAR'}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-slate-50 dark:bg-slate-800/40 rounded-xl p-5 border border-slate-100 dark:border-slate-700 text-sm space-y-3">
                <h3 className="font-semibold text-slate-900 dark:text-white flex items-center gap-2">
                  <Shield size={16} className="text-primary" />
                  Privacidad y Reset
                </h3>
                <p className="text-slate-500 dark:text-slate-400 text-xs leading-relaxed">
                  Tus llaves se guardan de forma segura en el navegador. Si usas una conexión vía túnel (ngrok), asegúrate de que sea segura.
                </p>
                <div className="pt-2">
                  <button
                    onClick={() => {
                      if (window.confirm('¿Estás seguro de que quieres limpiar todo el estado y logs? Esto no borrará tus API Keys.')) {
                        if (canReset) handleReset();
                      }
                    }}
                    disabled={!canReset}
                    className="inline-flex items-center gap-2 text-red-600 dark:text-red-400 font-medium hover:underline text-xs"
                  >
                    <RotateCcw size={14} />
                    Reiniciar estado de la aplicación
                  </button>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 overflow-hidden shadow-sm">
          <div className="px-6 py-4 border-b border-slate-200 dark:border-slate-700 bg-slate-50/80 dark:bg-slate-800/50 flex items-center gap-2">
            <Key size={18} className="text-primary" />
            <h2 className="text-lg font-semibold text-slate-900 dark:text-white">API Keys e Inteligencia Artificial</h2>
          </div>
          <div className="p-6 grid md:grid-cols-2 gap-8">
            <div className="space-y-5">
              <div>
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">Google Gemini API Key (Recomendado)</label>
                <input
                  type="password"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  className="w-full py-2.5 px-3 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-sm"
                  placeholder="AIzaSy..."
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">ElevenLabs API Key (Doblaje)</label>
                <input
                  type="password"
                  value={elevenLabsKey}
                  onChange={(e) => setElevenLabsKey(e.target.value)}
                  className="w-full py-2.5 px-3 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-sm"
                  placeholder="Llave para ElevenLabs"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">HuggingFace Token (Transcripción Pro)</label>
                <input
                  type="password"
                  value={huggingfaceToken}
                  onChange={(e) => setHuggingfaceToken(e.target.value)}
                  className="w-full py-2.5 px-3 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-sm"
                  placeholder="hf_..."
                />
              </div>
            </div>

            <div className="space-y-5">
              <div>
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">Upload-Post API Key (Social)</label>
                <input
                  type="password"
                  value={uploadPostKey}
                  onChange={(e) => setUploadPostKey(e.target.value)}
                  className="w-full py-2.5 px-3 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-sm"
                  placeholder="Tu llave de subida"
                />
              </div>
              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="text-sm font-medium text-slate-700 dark:text-slate-300">Perfil de Usuario Vinculado</label>
                  <button onClick={fetchUserProfiles} className="text-xs text-primary hover:underline">Actualizar perfiles</button>
                </div>
                <select
                  value={uploadUserId}
                  onChange={(e) => setUploadUserId(e.target.value)}
                  className="w-full py-2.5 px-3 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-sm"
                >
                  <option value="">Selecciona un usuario</option>
                  {(userProfiles || []).map((profile) => (
                    <option key={profile.username} value={profile.username}>
                      {profile.username}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>
        </section>

        <section className="rounded-2xl border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 overflow-hidden shadow-sm">
          <div className="px-6 py-4 border-b border-slate-200 dark:border-slate-700 bg-slate-50/80 dark:bg-slate-800/50 flex items-center justify-between">
            <h2 className="text-lg font-semibold text-slate-900 dark:text-white flex items-center gap-2">
              <Sparkles size={18} className="text-primary" />
              Kit de Marca
            </h2>
            <span className="text-xs font-medium px-2 py-1 rounded-md bg-primary/10 text-primary text-white">Configuración Visual</span>
          </div>
          <div className="p-6 grid md:grid-cols-2 gap-8">
            <div className="space-y-5">
              <div>
                <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">Nombre del preset</label>
                <input
                  type="text"
                  value={brandKit.name}
                  onChange={(e) => handleBrandKitFieldChange('name', e.target.value)}
                  className="w-full py-2.5 px-3 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-sm"
                  placeholder="Mi marca"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">Posición</label>
                  <select
                    value={brandKit.subtitle_position}
                    onChange={(e) => handleBrandKitFieldChange('subtitle_position', e.target.value)}
                    className="w-full py-2.5 px-3 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-sm"
                  >
                    <option value="top">Arriba</option>
                    <option value="middle">Centro</option>
                    <option value="bottom">Abajo</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">Tipografía</label>
                  <select
                    value={brandKit.subtitle_font_family}
                    onChange={(e) => handleBrandKitFieldChange('subtitle_font_family', e.target.value)}
                    className="w-full py-2.5 px-3 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 text-sm"
                  >
                    {captionFontOptions.map((font) => (
                      <option key={font.value} value={font.value}>
                        {font.label}
                        {font.available === false ? ' (no disponible)' : ''}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              <div>
                <div className="flex justify-between mb-1">
                  <label className="text-xs font-medium text-slate-500 dark:text-slate-400">Tamaño</label>
                  <span className="text-xs text-slate-500 dark:text-slate-400">{brandKit.subtitle_font_size}px</span>
                </div>
                <input
                  type="range"
                  min="12"
                  max="84"
                  value={brandKit.subtitle_font_size}
                  onChange={(e) => handleBrandKitFieldChange('subtitle_font_size', Number(e.target.value || 40))}
                  className="w-full accent-primary"
                />
              </div>

              <div className="grid grid-cols-3 gap-3">
                <div>
                  <label className="block text-xs text-slate-500 dark:text-slate-400 mb-1">Texto</label>
                  <input
                    type="color"
                    value={brandKit.subtitle_font_color}
                    onChange={(e) => handleBrandKitFieldChange('subtitle_font_color', e.target.value)}
                    className="w-full h-10 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 p-1"
                  />
                </div>
                <div>
                  <label className="block text-xs text-slate-500 dark:text-slate-400 mb-1">Contorno</label>
                  <input
                    type="color"
                    value={brandKit.subtitle_stroke_color}
                    onChange={(e) => handleBrandKitFieldChange('subtitle_stroke_color', e.target.value)}
                    className="w-full h-10 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 p-1"
                  />
                </div>
                <div>
                  <label className="block text-xs text-slate-500 dark:text-slate-400 mb-1">Caja</label>
                  <input
                    type="color"
                    value={brandKit.subtitle_box_color}
                    onChange={(e) => handleBrandKitFieldChange('subtitle_box_color', e.target.value)}
                    className="w-full h-10 rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800 p-1"
                  />
                </div>
              </div>
            </div>

            <div className="relative rounded-xl overflow-hidden aspect-[9/16] md:aspect-video bg-slate-900 flex items-center justify-center">
              <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,#334155_0%,#0f172a_45%,#020617_100%)] opacity-95" />
              <div className="absolute inset-x-6 inset-y-8 border-2 border-dashed border-white/20 rounded-lg pointer-events-none">
                <span className="absolute top-2 left-2 text-[10px] text-white/60 uppercase tracking-wider text-white">Zona segura</span>
              </div>
              <div className="relative z-10 text-center px-6 py-2 rounded-md max-w-[90%]" style={{ backgroundColor: `${brandKit.subtitle_box_color}${Math.round((brandKit.subtitle_box_opacity / 100) * 255).toString(16).padStart(2, '0')}` }}>
                <span
                  className="inline-block"
                  style={{
                    fontFamily: brandKit.subtitle_font_family,
                    fontSize: `${Math.max(18, Math.min(52, Number(brandKit.subtitle_font_size) || 40))}px`,
                    fontWeight: brandKit.subtitle_bold ? 800 : 600,
                    color: brandKit.subtitle_font_color,
                    lineHeight: 1.02,
                    textTransform: 'uppercase',
                    textShadow: `0 2px 0 ${brandKit.subtitle_stroke_color}, 0 -2px 0 ${brandKit.subtitle_stroke_color}, 2px 0 0 ${brandKit.subtitle_stroke_color}, -2px 0 0 ${brandKit.subtitle_stroke_color}`
                  }}
                >
                  Esto es increíble
                </span>
              </div>
            </div>
          </div>
        </section>

        <div className="sticky bottom-4 z-40">
          <div className="rounded-xl border border-slate-200 dark:border-slate-700 bg-white/95 dark:bg-slate-900/95 backdrop-blur p-4 shadow-xl flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
            <div className="text-sm text-slate-500 dark:text-slate-400">
              Los cambios se aplican automáticamente y se usarán en el siguiente clip generado.
            </div>
            <div className="flex items-center gap-2 self-end sm:self-auto">
              <button
                type="button"
                onClick={() => setActiveTab('home')}
                className="px-4 py-2 text-sm font-medium text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white"
              >
                Cerrar
              </button>
              <button
                type="button"
                onClick={() => {
                  if (canSaveApiBaseUrl) handleSaveApiBaseUrl();
                  if (canCheckConnectivity) runConnectivityCheck();
                }}
                disabled={!canSaveApiBaseUrl && !canCheckConnectivity}
                className="inline-flex items-center gap-2 px-5 py-2 rounded-lg bg-primary hover:bg-slate-800 text-white text-sm font-medium transition-colors shadow-lg shadow-primary/20"
              >
                <Check size={14} />
                Guardar cambios
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SettingsView;
