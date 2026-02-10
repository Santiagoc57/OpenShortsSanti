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
  - metadata
  - clips
  - srt (si existen)

### 🟡 Parcial
- Calendario editorial visual (hoy hay schedule y batch, pero no vista calendario).
- Brand kit/template engine (hoy hay estilos de subtítulos, no presets de marca completos).
- Multi-ratio como feature de producto (hoy el core está optimizado a vertical).

### ⛔ Pendiente
- `Clip Anything` real (búsqueda semántica por prompt dentro del video).
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

## Sprint 5 (siguiente)
1. `Clip Anything`: query semántica sobre transcript con timestamps.
2. Agrupación por tópicos/chapters en VOD largos.
3. Shortlist semántica + shortlist por score.

## Sprint 6 (siguiente)
1. Brand Kit v1:
   - logo
   - tipografía
   - paleta
   - safe margins
   - preset de subtítulos por marca
2. Export pack v2 con thumbnails y variantes por plataforma.

## Referencia Técnica

Archivos clave:
- Backend API: `app.py`
- Pipeline IA/video: `main.py`
- Dashboard: `dashboard/src/App.jsx`
- Cards de clips: `dashboard/src/components/ResultCard.jsx`
