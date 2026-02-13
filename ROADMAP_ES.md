# ROADMAP OpenShorts (ES)

Este documento centraliza la hoja de ruta del producto y el estado real de implementación.

## Estado Actual

### ✅ Implementado
- Ingesta por `YouTube URL` y `Upload` local.
- Detección automática de clips con IA.
- Reframe vertical inteligente con tracking (rostro/persona).
- Subtítulos automáticos + preview + estilos.
- Edición posterior (`Auto Edit`, `Recut`).
- Publicación social y programación (Upload-Post).
- Ranking visible por clip con:
  - `virality_score`
  - `score_band` (`top`, `medium`, `low`)
  - `selection_confidence`
  - `score_reason`
  - `topic_tags`
- Orden y filtros en dashboard:
  - Orden: `Top Score`, `Timeline`, `Safe Bets`
  - Filtros: banda de score + tag
- Batch scheduling configurable:
  - `Top N`
  - `Start in`
  - `Every`
  - Scope: `Visible` / `Global`
- Soporte de ingestión **audio-only** (`mp3/wav/m4a/...`) con canvas visual automático.
- Exportación de paquete para equipos/agencia (`/api/export/pack`):
  - `manifest.json`
  - `copies.csv`
  - `copies_by_platform.csv` (variantes por plataforma)
  - metadata
  - clips
  - srt (si existen)
  - thumbnails automáticos por clip
- `Clip Anything` semántico v1:
  - embeddings para matching semántico (con fallback local)
  - ranking híbrido (`semantic + keyword + virality`)
  - `hybrid_shortlist` de clips existentes
  - `chapters/topics` automáticos para VOD largos
  - filtros de búsqueda por `chapter`, `rango temporal` y `speaker`
- Post-procesado inteligente de clips:
  - `Smart Cut Boundaries` (ajuste de inicio/fin a pausas y límites naturales)
  - deduplicación semántica de clips similares (mantiene el mejor por score/confidence)
- Soporte multi-ratio en pipeline y UI:
  - selector `9:16` / `16:9` en input
  - `aspect_ratio` propagado backend -> `main.py`
  - recut respeta ratio seleccionado
  - metadata de clip incluye `aspect_ratio`
- Preconfiguraciones visuales en input:
  - templates rápidos (`Default`, `Modern`, `Bouncy`, `MrBeast`, `Business`)
  - presets por tipo de contenido (`General`, `Podcast`, `Tutorial`, `Entrevista`)
  - auto-aplican parámetros de pipeline (ratio/modelo/preset/CRF/cantidad de clips)
  - target de duración por perfil (`short`, `balanced`, `long`) conectado a prompt + postprocesado
  - persistencia de la última configuración en `localStorage` (recupera presets/ajustes al recargar)
- Transcript Sync en dashboard:
  - endpoint `/api/transcript/{job_id}` con segmentos normalizados
  - lista de transcript filtrable y clickeable (Play al timestamp)
- UI refresh (base `stitch`) v1:
  - navbar superior + layout de paneles adaptado a los HTML de referencia
  - home con hero/presets y cards visuales manteniendo funcionalidad actual
  - vista processing/results y settings alineadas al nuevo lenguaje visual
- Caption presets sociales en modal de subtítulos:
  - `Bold Center`, `Neon Pop`, `Typewriter`, `Bubble`, `Minimal Clean`
- Clip Studio (modo edición unificado) v1:
  - editor en una sola vista con secciones (`Transcripción`, `Subtítulos`, `Editar subtítulos`, `Editar layout`, `Música`)
  - encadenado de acciones en un solo `Aplicar` (recut + subtítulos + música)
  - endpoint nuevo `POST /api/music` para mezclar música de fondo con ducking opcional
- Resiliencia de jobs:
  - auto-retry configurable en backend para jobs fallidos
  - endpoint de retry manual (`POST /api/retry/{job_id}`)
  - metadata de retry en `/api/status/{job_id}` (attempts/last_error)
- QA de relevancia para `Clip Anything`:
  - endpoint `POST /api/search/clips/eval` con métricas (`pass_rate`, `mrr`, overlap)
  - script local `scripts/eval_clip_search.py` + template `scripts/clip_search_cases.example.json`

### 🟡 Parcial
- Brand kit/template engine:
  - ✅ presets de subtítulos por marca guardados en Settings (Brand Kit v1).
  - ⏳ pendiente: logo, tipografía global, paleta y safe margins aplicados a todo el pipeline.
- `Clip Anything` (semántica v1 lista; pendiente iteración de calidad/relevancia en producción).
- Calendario editorial visual:
  - estado: **depriorizado** (ya existe schedule + timeline + batch; no bloquea roadmap actual).

### ⛔ Pendiente
- Auto B-roll/emojis y packaging avanzado.
- Suite avanzada de audio (noise cleanup/filler removal como flujo integrado end-to-end).

## Sprints

## Sprint 1 (completado)
- `virality_score` en backend + UI.
- orden/filtro por score.
- estabilidad de `clip_index` al reordenar.

## Sprint 2 (completado)
- `selection_confidence`, `score_reason`, `topic_tags`.
- filtros por tag en dashboard.
- soporte audio-only.

## Sprint 3 (completado)
- batch scheduling configurable con presets persistidos.
- export pack para equipos/agencia.

## Sprint 4 (completado)
1. Vista calendario/timeline de publicaciones programadas (en panel de resultados).
2. Queue templates por estrategia:
   - `Growth`
   - `Balanced`
   - `Conservative`
   - `Custom`
3. Descarga de reportes batch en CSV.

## Sprint 5 (completado)
1. `Clip Anything`: query semántica sobre transcript con timestamps.
   - Estado: **completado (v1)**.
2. Agrupación por tópicos/chapters en VOD largos.
   - Estado: **completado (v1)**.
3. Shortlist semántica + shortlist por score.
   - Estado: **completado (v1)**.
   - Mejora: **v1.1** con re-ranking por intención de query y thresholds dinámicos.
   - Mejora: **v1.2** con presets de búsqueda en UI (`Exacta`, `Balanceada`, `Amplia`).
   - Mejora: **v1.3** con transcript sync clickeable en dashboard.
   - Mejora: **v1.4** robustez de procesamiento con auto-retry/manual retry de jobs.
   - Mejora: **v1.5** búsqueda acotada por chapter/rango/speaker para VOD largos.
4. Multi-ratio (9:16 / 16:9) en procesamiento real.
   - Estado: **completado (v1)**.

## Sprint 6 (en curso)
1. Brand Kit v1:
   - ✅ preset de subtítulos por marca
   - ⏳ logo
   - ⏳ tipografía global
   - ⏳ paleta
   - ⏳ safe margins
2. Export pack v2 con thumbnails y variantes por plataforma.
   - Estado: **completado (v1)**.
3. `Clip Anything` v2 (calidad de relevancia):
   - ✅ evaluación offline con queries reales (set de pruebas ES/EN) via endpoint/script.
   - ⏳ ajuste fino de pesos/thresholds por intención.
   - ⏳ mejora de ranking para VOD largos con señales de capítulo + speaker.
4. Hardening de producción:
   - reintentos y cola ya listos ✅, pendiente observabilidad básica (métricas/errores/tiempos).
   - limpieza/retención automática de artefactos y cache semántico.
5. Clip Studio v1:
   - ✅ editor unificado con transcripción/subtítulos/layout/música
   - ✅ botón `Editar` en card para abrir flujo integral
   - ⏳ mejorar presets visuales avanzados tipo “industria” (animaciones, templates premium, keyframes)

## Qué Falta (prioridad real)
1. Cerrar Brand Kit aplicado end-to-end (no solo subtítulos): logo, paleta, safe margins, tipografía consistente.
2. Mejorar calidad de `Clip Anything` en producción (benchmark + tuning).
3. Hardening operativo para cargas largas (observabilidad + housekeeping).
4. Auto B-roll/emojis y suite avanzada de audio (siguiente fase).

## Referencia Técnica

Archivos clave:
- Backend API: `app.py`
- Pipeline IA/video: `main.py`
- Dashboard: `dashboard/src/App.jsx`
- Cards de clips: `dashboard/src/components/ResultCard.jsx`
