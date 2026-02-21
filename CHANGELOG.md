# Registro de Cambios

## 2026-02-21

### Corregido: “Regenerar” ahora rota variantes pre-generadas (sin depender siempre de Gemini en vivo)
- Qué cambiamos:
  - En `app.py` (`/api/clip/retitle` y `/api/clip/resocial`) se eliminó la dependencia rígida de `X-Gemini-Key` para poder rotar variantes ya existentes.
  - Ambos endpoints ahora intentan API key en este orden:
    - header `X-Gemini-Key`
    - key guardada en `job.env.GEMINI_API_KEY`
    - `GEMINI_API_KEY` del entorno del backend
  - Si no hay key, igual rotan el pool pre-calculado y/o completan con fallback local del backend.
- Para qué sirve:
  - `Regenerar` responde rápido y estable aun cuando no haya cuota/API key disponible en ese momento.
  - Se mantiene el flujo “5 variantes título + 5 variantes social” sin bloquearse por red/cuota.

### Corregido: sincronización de estado de variantes en UI (evita que se “pierdan” al refrescar/re-render)
- Qué cambiamos:
  - En `dashboard/src/components/ResultCard.jsx`, al regenerar título/social se envía `clip_patch` al estado global.
  - En `dashboard/src/App.jsx` se conecta `ResultCard` con `onClipPatched={handleStudioClipPatched}`.
- Para qué sirve:
  - Los cambios de título/social quedan reflejados en la lista principal, no solo en estado local del card.
  - Evita inconsistencias cuando entra polling o se vuelve a pintar la vista.

### Corregido: preview rápido de subtítulos usa el mismo motor de render del export
- Qué cambiamos:
  - En `app.py` (`/api/clip/fast-preview`) se añadieron parámetros de estilo de subtítulo (`font`, `stroke`, `box`, `karaoke`, offsets, `srt_content`, etc.).
  - El endpoint ahora puede generar ASS + quemar subtítulos en el preview con `generate_styled_ass_from_srt` / `generate_karaoke_ass_from_srt` + `burn_subtitles` (mismo pipeline del export final).
  - Se agregó ajuste de SRT al rango temporal del preview para que el texto corresponda al recorte mostrado.
  - En `dashboard/src/components/ClipStudioModal.jsx` el request de `Preview rápido` ahora envía toda la configuración de subtítulos + SRT editado.
  - Cuando el preview llega con subtítulos ya quemados, la UI oculta la superposición React para evitar doble render.
- Para qué sirve:
  - Reduce la desalineación entre lo que ves en preview y lo que sale en “Aplicar / Exportar”.
  - Evita casos de subtítulos distintos en UI vs video final.

## 2026-02-20

### Ajustado: vista dedicada de Super Trailer (foco + transcript a la derecha)
- Qué cambiamos:
  - En `dashboard/src/App.jsx` se agregó un layout exclusivo para proyectos en modo `Super Trailer`:
    - columna izquierda con foco en el reproductor del trailer y pestañas `Transcript / Social / Ajustes`,
    - columna derecha con `Sincronía de transcript` fija (filtro + salto por segmento).
  - En ese modo se ocultan los bloques de lista/búsqueda de “otros clips” para no mezclar flujos.
  - El botón de cada segmento del transcript ahora busca y reproduce directamente en el reproductor del trailer.
- Para qué sirve:
  - Hace que el modo Super Trailer tenga una experiencia más clara y cercana al diseño que pediste.
  - Prioriza revisión rápida del trailer + navegación por transcript en el panel derecho.

### Nuevo: control de cantidad de segmentos en Super Trailer
- Qué cambiamos:
  - En `dashboard/src/components/MediaInput.jsx` se agregó el campo `Segmentos destacados (Super Trailer)` (rango 2-12, default 6).
  - En `dashboard/src/App.jsx` ahora se envía `trailer_fragments_target` al backend en JSON y `FormData`.
  - En `app.py` (`POST /api/process`) se recibe/valida `trailer_fragments_target` y se pasa a `main.py` con `--trailer-fragments-target`.
  - En `main.py`:
    - se añadió el flag `--trailer-fragments-target`,
    - el prompt del LLM ahora sugiere explícitamente el objetivo de fragmentos,
    - el pipeline normaliza, completa con fallback y limita los `trailer_fragments` al objetivo configurado.
- Para qué sirve:
  - Te permite decidir cuántos momentos destacados debe tener cada Super Trailer desde la UI.
  - Reduce variaciones aleatorias en el número de fragmentos cuando el LLM responde con más o menos momentos de los esperados.

### Corregido: Super Trailer en editor mostraba timeline/subtítulos vacíos o desfasados
- Qué cambiamos:
  - En `main.py` se corrigió `_probe_media_duration_seconds` con fallback a `ffmpeg -i` cuando no existe `ffprobe` (caso común en local/macOS).
  - En `main.py` (modo `--trailer-only`) ahora se construye un transcript sintético del trailer (`transcript_segments`) alineado al montaje de fragmentos y se marca `transcript_timebase: clip`.
  - En `app.py` se añadió reparación automática de trailers persistidos con rango incorrecto (`end=3.0`) para recalcular duración real desde el archivo.
  - En `app.py` (`/api/subtitle` y `/api/subtitle/preview`) ahora se usa transcript por-clip cuando existe (`transcript_segments`) antes de caer al transcript global.
  - En `dashboard/src/components/ClipStudioModal.jsx` el editor prioriza `clip.transcript_segments` para poblar transcripción en la línea de tiempo.
  - En `main.py` y `app.py` se agregaron marcadores de transición para trailer (`transition_points`/`fragment_ranges`) y en `ClipStudioModal` ahora se dibujan en la línea de tiempo.
- Para qué sirve:
  - Evita que el Super Trailer se vea bien pero quede sin escenas/subtítulos útiles en el editor.
  - Alinea mejor subtítulos/timeline con el contenido real del trailer.
  - Hace visible dónde ocurren los crossfades del trailer dentro del timeline.

### Nuevo: selector mosaico de modo de generación (Clips virales vs Super Trailer)
- Que cambiamos:
  - En `dashboard/src/components/MediaInput.jsx` se agregó selector tipo mosaico para elegir entre:
    - `Clips virales`
    - `Super Trailer`
  - El botón final del modal ahora adapta su texto según el modo elegido.
  - En modo `Super Trailer`, el campo de número de clips queda deshabilitado con aviso.
  - En `dashboard/src/App.jsx` se envían `generation_mode` y `build_trailer` al backend.
  - En `app.py` (`/api/process`) se añadieron los parámetros `generation_mode` y `build_trailer`, mapeados a flags de `main.py`.
  - En `main.py` se agregaron:
    - `--trailer-only` para omitir render de clips y generar solo trailer.
    - fallback de `trailer_fragments` cuando el LLM no entrega suficientes fragmentos.
- Para que sirve:
  - Permite un flujo separado y claro para generar solo Super Trailer desde el inicio.
  - Deja la base preparada para añadir más “habilidades” en el mosaico.

### Corregido: en modo Super Trailer ahora aparece video y edición como en clips
- Que cambiamos:
  - En `main.py`, cuando se usa `--trailer-only`, se guarda un clip sintético en `shorts` apuntando al trailer generado.
  - En `app.py` (`run_job`) se dejó de sobreescribir ciegamente `video_url`; ahora respeta la URL real del clip/trailer si existe en disco.
  - En `app.py` (`_materialize_result_from_metadata`) se propaga `latest_trailer_url` al resultado y se crea clip sintético al recuperar jobs si no hay clips.
- Para que sirve:
  - Evita que el proyecto quede “completado” sin video visible.
  - Habilita el mismo flujo de edición/subtítulos/transcripción sobre el Super Trailer.

### Corregido: estabilidad de render en Super Trailer (freeze visual/audio desfasado)
- Que cambiamos:
  - En `main.py` (`build_super_trailer`) se hizo robusto el pipeline:
    - limpia archivos temporales/salida previa antes de generar,
    - valida retorno y tamaño real de cada fragmento extraído,
    - corrige duración fallback cuando `ffprobe` no devuelve duración válida,
    - evita considerar éxito solo por “archivo existe”.
  - Se reemplazó el comando shell por llamada estructurada a `ffmpeg` y se forzó salida CFR (`-r 30 -vsync cfr -shortest`) para mayor compatibilidad de reproducción.
  - Si el montaje con `xfade` falla, ahora aplica fallback automático a concatenación simple (hard-cut), para no romper el trailer.
- Para que sirve:
  - Reduce casos donde el video se queda congelado mientras el audio continúa.
  - Evita reutilizar archivos corruptos de intentos anteriores.

### Corregido: Super Trailer fallaba por import legacy de PySceneDetect en autocrop
- Que cambiamos:
  - En `autocrop.py` se actualizó `detect_scenes` para soportar ambas APIs:
    - `scenedetect.detect_scenes` (si existe)
    - `scenedetect.detect` (fallback)
  - También se agregó fallback de `ContentDetector` desde `scenedetect.detectors`.
  - En `main.py`, cuando falla el análisis de escenas en `process_video_to_vertical`, ahora intenta copiar el video de entrada al output y retornar éxito si el archivo quedó generado.
- Para que sirve:
  - Evita que `Super Trailer` termine en `500` por incompatibilidad de versión de `scenedetect`.
  - Permite que el trailer salga aunque el autoanálisis de escenas falle.

### Corregido: compatibilidad de import con PySceneDetect
- Que cambiamos:
  - En `main.py` se agregó fallback de import para usar `detect_scenes` o `detect` según la versión instalada de `scenedetect`.
  - También se añadió fallback para `ContentDetector` desde `scenedetect.detectors` cuando no está exportado en el módulo raíz.
  - En `app.py` (smart reframe) se aplicó el mismo patrón de compatibilidad.
- Para que sirve:
  - Evita el error `ImportError: cannot import name 'detect_scenes' from 'scenedetect'`.
  - Permite procesar clips aunque cambie la API entre versiones de PySceneDetect.

### Ajustado: “Eliminar todos” ahora borra también en backend (definitivo)
- Que cambiamos:
  - En `app.py` se agregó endpoint `DELETE /api/jobs/all` para limpiar jobs no activos en memoria, SQLite y artefactos en `output/` y `uploads/`.
  - En `dashboard/src/App.jsx`, el botón `Eliminar todos` ahora llama ese endpoint antes de limpiar estado local.
  - También limpia `openshortsProjectsV1` de `localStorage` para evitar repoblado local.
- Para que sirve:
  - Evita que reaparezcan “proyectos recuperados” después de recargar.
  - Hace el borrado realmente persistente para proyectos no activos.

### Ajustado: se removió la funcionalidad de “Highlight reel” en la UI
- Que cambiamos:
  - En `dashboard/src/App.jsx` se eliminó:
    - botón `Generar highlight reel`,
    - selector `Reel ratio`,
    - bloque visual de resultado `Highlight reel`.
  - También se removieron handlers/estados de highlight reel y el auto-disparo al completar procesamiento.
  - En `dashboard/src/components/MediaInput.jsx` se quitó la opción de formato `Highlight reel` del modal de configuración.
- Para que sirve:
  - Simplifica el flujo de generación dejando solo clips y trailer.
  - Evita que aparezca una función que ya no necesitas.

### Ajustado: filtros de proyectos más claros + botón “Eliminar todos”
- Que cambiamos:
  - En `dashboard/src/App.jsx` se mejoró el contraste/legibilidad de los botones de filtro `Todos` y `Favoritos`, con etiqueta + contador separado.
  - Se reforzó el estado activo de esos filtros con color primario + texto blanco y ancho mínimo para que la etiqueta no se pierda visualmente.
  - Se agregó botón `Eliminar todos` en la cabecera de `Mis proyectos`, con confirmación de seguridad.
- Para que sirve:
  - Evita confusión cuando el botón activo no se alcanza a leer.
  - Permite limpiar toda la lista de proyectos en un solo paso.

### Ajustado: Groq removido del frontend (flujo simplificado a Gemini)
- Que cambiamos:
  - En `dashboard/src/components/MediaInput.jsx` se eliminó el selector de proveedor/modelos Groq y la generación ahora envía `llm_provider: gemini` de forma fija.
  - En `dashboard/src/components/KeyInput.jsx` se eliminó la sección de API Key de Groq.
  - En `dashboard/src/App.jsx` se removieron estados/props de Groq y se limpia `groq_key_v1` del `localStorage`.
- Para que sirve:
  - Evita bloqueos por límites de Groq y reduce complejidad operativa.
  - Mantiene una sola ruta estable de generación con Gemini.

### Corregido: reintento automatico en Groq cuando hay rate limit (429) + fallback a Gemini
- Que cambiamos:
  - En `main.py` (`get_viral_clips`) se agrego deteccion de errores `rate_limit_exceeded` de Groq.
  - Si Groq responde `429`, ahora el proceso:
    - extrae el `try again in Xs` del mensaje,
    - espera ese tiempo (+margen),
    - reintenta una vez automaticamente.
  - Si Groq sigue fallando y existe `GEMINI_API_KEY`, hace fallback automatico a Gemini (`gemini-2.5-flash-lite` por defecto, configurable con `GROQ_FALLBACK_GEMINI_MODEL`).
  - En `app.py` (`/api/process`), cuando `llm_provider=groq`, ahora tambien se propaga `X-Gemini-Key` al entorno del proceso para habilitar ese fallback.
- Para que sirve:
  - Evita que el job caiga inmediatamente por picos temporales de TPM en Groq.
  - Mantiene la generacion de clips incluso si Groq esta limitado en ese momento.

### Ajustado: selector Groq solo muestra modelos con límite >=70K TPM
- Que cambiamos:
  - En `dashboard/src/components/MediaInput.jsx` se retiraron del selector Groq los modelos:
    - `llama-3.3-70b-versatile`
    - `llama-3.1-70b-versatile`
    - `llama-3.1-8b-instant`
    - `mixtral-8x7b-32768`
  - Se agregaron:
    - `groq/compound (70K TPM)`
    - `groq/compound-mini (70K TPM)`
  - Se añadió normalización para que, si había un modelo viejo guardado en `localStorage`, se migre automáticamente a uno válido del proveedor actual.
- Para que sirve:
  - Evita errores frecuentes de límite de tokens en Groq al procesar transcripciones largas.
  - Mantiene el selector alineado con tu requisito de mínimo `70K TPM`.

## 2026-02-19

### Corregido: resolución de tipografías más robusta en entornos Colab/headless
- Que cambiamos:
  - En `subtitles.py` se mejoró el mapeo de familias para preferir variantes reales disponibles (ej. `Montserrat Bold`, `Oswald Bold`, `Teko Bold`) y luego fallback a familia base.
  - Se agregó registro de fuentes bundle en fontconfig de usuario (`~/.local/share/fonts/openshorts`) para mejorar detección de fuentes al exportar.
- Para que sirve:
  - Reduce casos donde el render final ignora `Montserrat` u otras fuentes y cae en tipografías no deseadas.
  - Mejora consistencia de estilo entre UI y video exportado.

### Corregido: paneo manual de layout con mayor rango real (preview + render)
- Que cambiamos:
  - En `dashboard/src/components/ClipStudioModal.jsx` se ajustó `LAYOUT_OFFSET_FACTOR` de `0.35` a `1.0`.
  - En `app.py` se ajustó `_LAYOUT_OFFSET_FACTOR` de `0.35` a `1.0` para mantener consistencia con el preview.
- Para que sirve:
  - Permite mover el encuadre manualmente (izquierda/derecha/arriba/abajo) con rango completo útil.
  - Evita la sensación de que sliders/drag “no hacen nada” en algunos clips.

### Corregido: paneo en Cover con offsets guardados ahora aplica zoom mínimo útil
- Que cambiamos:
  - En `dashboard/src/components/ClipStudioModal.jsx` se agregó una protección para subir automáticamente el zoom a `1.06` cuando:
    - el modo es `single`
    - `Auto smart reframe` está apagado
    - `fit` está en `Cover`
    - ya existe offset manual X/Y distinto de cero.
  - Se mantiene la misma regla al iniciar drag o al mover sliders de offset.
  - En `app.py` ya está alineado el render final: si hay offsets en `Cover` con zoom ~`1.0`, aplica zoom implícito `1.06` para que el paneo no se pierda al exportar.
- Para que sirve:
  - Evita que un clip “parezca fijo” lateralmente cuando vuelves a abrirlo con offsets guardados.
  - Garantiza que lo que ves al mover encuadre en la UI se parezca al resultado exportado.

### Corregido: cambio de formato 9:16/16:9 ahora limpia encuadre heredado
- Que cambiamos:
  - En `dashboard/src/components/ClipStudioModal.jsx` se agregó `handleLayoutAspectChange`.
  - Al cambiar formato:
    - se limpian offsets/zoom heredados que distorsionaban la vista (`offset_x/y` a 0, zoom base).
    - en modo `single`, si pasas a `16:9` con `Cover`, se ajusta a `Contain` para evitar recorte agresivo inicial.
    - en `split`, se resetean zoom/offset por panel para arrancar limpio.
- Para que sirve:
  - Evita que al pasar a `16:9` parezca que el video “no cambia” o queda mal encuadrado por arrastre de ajustes previos.
  - Da una base consistente para hacer ajustes finos después del cambio de formato.

### Ajustado: en 16:9 se mantiene orientación natural (sin girar 90°)
- Que cambiamos:
  - Se revirtió la rotación automática en `ClipStudioModal` para evitar que el sujeto se vea acostado.
  - Se revirtió el `transpose=1` en `/api/recut` para que el export conserve orientación natural.
- Para que sirve:
  - El formato `16:9` ahora significa lienzo horizontal, no rotación forzada del contenido.
  - Evita el efecto visual incorrecto reportado por el usuario.

### Corregido: Preview rápido ahora prioriza fuente original del proyecto
- Que cambiamos:
  - En `app.py` (`POST /api/clip/fast-preview`) la resolución de fuente ahora prioriza:
    1) `input_filename` válido
    2) `job.input_path` (video original)
    3) `clip.video_url` como fallback
  - En `dashboard/src/components/ClipStudioModal.jsx`, `handleFastPreview` ya no fuerza `input_filename` del clip actual.
- Para que sirve:
  - Evita que el preview rápido herede un recorte vertical viejo cuando estás ajustando a `16:9`.
  - Muestra una referencia más fiel del material original al cambiar formato.

### Ajustado: al cambiar formato 9:16/16:9 se dispara preview rápido automáticamente
- Que cambiamos:
  - En `dashboard/src/components/ClipStudioModal.jsx`, al pulsar un botón de formato se ejecuta:
    - `handleLayoutAspectChange(...)`
    - `handleFastPreview(...)` con el ratio seleccionado.
  - `handleFastPreview` ahora acepta `targetAspect` explícito para render inmediato del formato elegido.
- Para que sirve:
  - Evita quedarse viendo el video previo mientras cambias de formato.
  - Al pasar a `16:9`, la vista se actualiza enseguida hacia una referencia horizontal.

### Corregido: preview rápido evita archivos vacíos/no reproducibles
- Que cambiamos:
  - En `app.py` (`POST /api/clip/fast-preview`) ahora se prueban múltiples fuentes candidatas de forma robusta.
  - Se ajusta `start` automáticamente según la duración real de cada fuente para evitar renders en tramo inexistente.
  - Se valida la salida generada (dimensiones + duración mínima) antes de devolverla al frontend.
- Para que sirve:
  - Evita la pantalla gris con “El navegador no pudo reproducir este archivo...” al cambiar formato.
  - Mejora la confiabilidad del preview en proyectos antiguos o con rutas mixtas (original/clip).

### Ajustado: aclaración visual en modos Cover/Contain del editor de layout
- Que cambiamos:
  - En `dashboard/src/components/ClipStudioModal.jsx` se renombraron botones:
    - `Cover` -> `Cover (llenar)`
    - `Contain` -> `Contain (completo)`
  - Se agregó texto de ayuda bajo ese selector explicando que:
    - `Contain` puede mostrar barras para respetar proporción.
    - `Cover` llena todo el cuadro recortando excedentes.
- Para que sirve:
  - Evita confusión cuando se edita 16:9 con video vertical y parece “cortado”.
  - Hace explícito qué comportamiento esperar en cada modo.

### Ajustado: Se removió el panel de "Métricas sociales" en resultados
- Que cambiamos:
  - En `dashboard/src/App.jsx` se eliminó la tarjeta visual de `Métricas sociales` (incluyendo botón `Recargar métricas`).
  - También se removieron estados y llamadas automáticas a `/api/social/metrics/{job_id}` que alimentaban ese bloque.
- Para que sirve:
  - Limpia la interfaz en la vista de proyecto.
  - Reduce consultas innecesarias al backend.

### Corregido: render de subtítulos más fiel al preset (fuentes + ASS)
- Que cambiamos:
  - En `subtitles.py` se ajustó el mapeo de fuentes para usar familias canónicas estables (`Montserrat`, `Oswald`, `Teko`) en lugar de variantes tipo `* Bold` que podían disparar fallback inesperado en `libass`.
  - Se corrigió el mapeo de alineación ASS a valores estándar:
    - `top` -> `8`
    - `middle` -> `5`
    - `bottom` -> `2`
  - Al quemar archivos `.ass`, ahora se usa el filtro `ass` de FFmpeg (en vez de `subtitles`) para respetar mejor estilos karaoke/caja/fuente.
- Para que sirve:
  - Reduce casos donde el video exportado salía con tipografía distinta o “rara” respecto al preset elegido en UI.
  - Mejora consistencia entre preview del editor y render final.

### Ajustado: Se removieron textos de score/confianza del panel "Puntaje viral"
- Que cambiamos:
  - En `dashboard/src/components/ResultCard.jsx` se quitaron del panel de `Puntaje viral` los textos:
    - `Puntaje de viralidad`
    - `Confianza del modelo`
  - El panel ahora muestra solo el texto explicativo de viralidad.
- Para que sirve:
  - Simplifica la vista y reduce ruido visual en el tab.

### Corregido: warning de React por estructura HTML inválida en Configuración
- Que cambiamos:
  - En `dashboard/src/App.jsx` se reemplazó un bloque `<p>...</p>` que contenía un `<div>` interno por una estructura válida con contenedor `<div>` y párrafos separados.
- Para que sirve:
  - Elimina el warning `validateDOMNesting(...): <div> cannot appear as a descendant of <p>.`
  - Evita ruido en consola durante desarrollo.

## 2026-02-18

### Ajustado: Se removio la barra visual en "Puntaje viral"
- Que cambiamos:
  - En `dashboard/src/components/ResultCard.jsx` se quitó la barra de progreso (gradiente) del tab `Puntaje viral`.
  - Se mantiene el valor numérico `score/100`, la confianza del modelo y la explicación textual.
- Para que sirve:
  - Reduce ruido visual y libera espacio en la tarjeta del clip.
  - Mantiene la lectura del puntaje sin duplicar información.

### Documentado: bootstrap robusto para Colab (evita errores de ruta/repo)
- Que cambiamos:
  - En `EJECUTAR.md` se agregó una celda recomendada para Colab que:
    - si `/content/OpenShortsSanti/.git` existe: hace `git checkout main` + `git pull --ff-only`.
    - si no existe: hace `git clone` limpio en `/content/OpenShortsSanti`.
  - Se documentó secuencia completa `instalar -> uvicorn -> ngrok -> validación (docs/openapi/health)`.
  - Se añadió troubleshooting específico para:
    - `[Errno 2] No such file or directory: '/content/OpenShortsSanti'`
    - `fatal: not a git repository`
    - `Could not open requirements file: requirements.txt`
- Para que sirve:
  - Evita romper el flujo cuando Colab reinicia runtime o se pierde el directorio del repo.
  - Reduce errores por ejecutar comandos fuera de `/content/OpenShortsSanti`.

### Corregido: Regenerar ahora rota sobre pools precargados (5 títulos + 5 sociales)
- Que cambiamos:
  - Backend ahora usa por defecto `TITLE_VARIANTS_PER_CLIP=5` y `SOCIAL_VARIANTS_PER_CLIP=5`.
  - Se agregó pool de copy social por clip:
    - `social_variants` (lista)
    - `social_variant_index` (posición activa)
  - En finalización de job, además del pool de títulos, se precarga también el pool social para cada clip.
  - `POST /api/clip/retitle` ahora rota en ciclo sobre el pool existente (sin pedir IA en cada click).
  - `POST /api/clip/resocial` ahora rota en ciclo sobre el pool social existente (sin pedir IA en cada click).
- Para que sirve:
  - `Regenerar` se vuelve instantáneo y estable: cambia entre opciones ya preparadas por backend.
  - Reduce fallas por cuota/rate-limit al evitar llamada a Gemini en cada regeneración.
  - Cumple flujo esperado: 1º, 2º, 3º, 4º, 5º y luego vuelve a la primera opción.

### Corregido: Subtítulos desfasados tras recut/layout
- Que cambiamos:
  - En `dashboard/src/components/ClipStudioModal.jsx`, cuando `Aplicar` ejecuta un recorte (`/api/recut`) y cambia el rango `inicio/fin`, ahora se refresca el SRT con `POST /api/subtitle/preview` antes de llamar `POST /api/subtitle`.
  - El `srt_content` que se envía al backend ya no reutiliza el buffer viejo del modal si el rango del clip cambió.
- Para que sirve:
  - Evita que el export final queme subtítulos de un tramo anterior del video.
  - Alinea el texto exportado con el rango real que quedó después de editar layout/tiempos.

### Agregado: Persistencia de jobs y recuperación tras reinicio
- Que cambiamos:
  - Se agregó store SQLite (`output/jobs_state.sqlite3`) para estado de jobs.
  - El backend restaura jobs al iniciar y recupera proyectos desde artifacts (`output/<job_id>` + metadata).
  - Nuevo healthcheck real: `GET /api/status/__healthcheck__` (ya no depende de 404).
  - Nuevo listado rápido: `GET /api/jobs/recent`.
- Para que sirve:
  - Evita el error de “Job not found” al reiniciar Colab/ngrok.
  - Permite reabrir proyectos existentes sin reprocesar todo.

### Agregado: Export pack con variantes de video por plataforma
- Que cambiamos:
  - `POST /api/export/pack` soporta `include_platform_video_variants`.
  - Se generan variantes MP4 para `youtube`, `instagram`, `tiktok` y se incluyen en el zip.
  - Se actualizó manifiesto a `export_version: v3`.
- Para que sirve:
  - Entrega lista para publicación multi-plataforma sin reprocesos manuales.

### Agregado: Métricas de publicación social
- Que cambiamos:
  - `POST /api/social/post` ahora registra eventos (éxito/fallo por clip/plataforma).
  - Nuevo endpoint `GET /api/social/metrics/{job_id}` con agregados por plataforma y eventos recientes.
  - Panel de métricas sociales en resultados del dashboard.
- Para que sirve:
  - Da visibilidad operativa de qué se publicó y qué falló.

### Agregado: Highlight reel con ratio configurable
- Que cambiamos:
  - `POST /api/highlight/reel` acepta `aspect_ratio` (`9:16` o `16:9`).
  - Se normalizan segmentos a resolución de salida consistente antes de concatenar.
  - UI agrega selector de ratio para “Generar highlight reel”.
- Para que sirve:
  - Permite generar reels verticales u horizontales desde el mismo proyecto.

### Agregado: Preview rápido en Clip Studio
- Que cambiamos:
  - Nuevo endpoint `POST /api/clip/fast-preview` para render corto (~3s) con layout actual.
  - Botón `Preview rápido` en header de Clip Studio.
- Para que sirve:
  - Acelera iteraciones visuales antes de aplicar render completo.

### Corregido: Preview rápido con fallback local cuando backend está desactualizado
- Que cambiamos:
  - En `dashboard/src/components/ClipStudioModal.jsx` el botón `Preview rápido` ahora cae a reproducción local (~3.2s) si `/api/clip/fast-preview` responde `404/Not Found` o error de red.
  - Se agregó limpieza de timer al desmontar para evitar estados colgados de reproducción.
- Para que sirve:
  - Evita bloquear edición cuando Colab/backend no tiene el endpoint nuevo.
  - Mantiene una experiencia usable aunque el entorno remoto esté atrasado.

### Corregido: Consistencia de tipografías entre preview y export
- Que cambiamos:
  - Se normaliza la familia tipográfica en frontend antes de aplicar/exportar (`Impact`/`Arial Black` -> `Anton`, etc.).
  - `POST /api/subtitle` ahora resuelve y persiste `caption_font_family` ya saneada (`resolved_font_family`) para que metadata/UI reflejen la fuente real de export.
  - En `subtitles.py` se mejoró detección de familias disponibles (`family + fullname`) y mapeos preferidos en pesos bold (`Montserrat Bold`, `Oswald Bold`, `Teko Bold`).
- Para que sirve:
  - Evita que el MP4 final salga con una fuente distinta a la que se ve en el editor.
  - Reduce fallback impredecible de fuentes en entornos headless (Colab/servidor).

### Corregido: Subtítulos aplicados ya no se pierden al exportar ni al reabrir
- Que cambiamos:
  - En `POST /api/subtitle` ahora también se persiste `video_url` del clip generado en metadata (`*_metadata.json`) y en `job.result`.
  - Se aplicó `_resolve_subtitle_source_filename(...)` también cuando `input_filename` no viene explícito, evitando reutilizar por error un `subtitled_*` previo como base.
  - En frontend se amplió normalización de fuentes para nombres devueltos por backend (`Montserrat Bold`, `Oswald Bold`, `Teko Bold`, etc.) en:
    - `dashboard/src/components/ClipStudioModal.jsx`
    - `dashboard/src/App.jsx`
    - `dashboard/src/components/SubtitleModal.jsx`
  - `dashboard/src/components/ResultCard.jsx` ahora sincroniza `currentVideoUrl/baseVideoUrl` cuando cambia `clip.video_url` en estado global.
- Para que sirve:
  - Lo que aplicas en UI queda como versión “vigente” del clip para `Descargar` y `Exportar paquete`.
  - Evita desalineaciones donde preview mostraba un estilo pero export salía con otra versión antigua.

### Agregado: Regeneración de título por pool (4 opciones + rotación)
- Que cambiamos:
  - Se incorporó pool de variantes por clip en backend:
    - `title_variants` (lista)
    - `title_variant_index` (posición activa)
  - Al terminar un job (`run_job`), el backend pre-genera variantes de título por clip (default 4, configurable con `TITLE_VARIANTS_PER_CLIP`).
  - `POST /api/clip/retitle` ahora rota al siguiente título del pool en cada click de `Regenerar`.
  - Si se agotan variantes, el backend genera más (`TITLE_VARIANTS_TOPUP_COUNT`) y continúa la rotación.
  - Se persisten variantes e índice tanto en metadata (`*_metadata.json`) como en `job.result`.
- Para que sirve:
  - El primer título visible sigue siendo uno solo, pero ya hay alternativas listas.
  - `Regenerar` deja de ser “pedir 1 título nuevo desde cero” y pasa a “siguiente opción IA”, más rápido y consistente.

### Ajustado: Tamaño por defecto de subtítulos en 40
- Que cambiamos:
  - Se unificó el default de `font_size/subtitle_font_size` a `40` en frontend y backend:
    - `dashboard/src/components/ClipStudioModal.jsx`
    - `dashboard/src/components/SubtitleModal.jsx`
    - `dashboard/src/App.jsx` (Brand Kit)
    - `app.py` (`SubtitleRequest`)
    - `subtitles.py` (generadores ASS)
- Para que sirve:
  - Mantiene un baseline visual consistente en presets, edición y render final.

### Corregido: Regenerar título tolera cuota/rate-limit de Gemini
- Que cambiamos:
  - En `dashboard/src/components/ResultCard.jsx` se detecta `quota/rate-limit/429` y se aplica título local de fallback en vez de mostrar error técnico largo.
  - Se añadió generador local `buildFallbackTitleLocal` en `dashboard/src/geminiTitle.js`.
  - En backend se removió el bloqueo duro por API key faltante en `/api/clip/retitle` para permitir fallback local.
- Para que sirve:
  - El botón `Regenerar` sigue funcionando incluso sin cuota de Gemini o sin key configurada.
  - Mejora UX con mensajes cortos y resultado útil inmediato.

### Corregido: Highlight reel más tolerante (incluye caso de 1 segmento)
- Que cambiamos:
  - En `app.py` el armado de highlight reel ahora permite `max_segments >= 1` y no falla si solo hay 1 momento utilizable.
  - Se agregan mensajes de warning cuando el reel final queda con un solo segmento.
  - En `dashboard/src/App.jsx` se ajustó la validación para permitir generar reel con `>= 1` clip disponible.
  - Al abrir un proyecto guardado, se conserva `processingMedia.aspectRatio` según `project.ratio` (incluyendo modo `highlight`) para que no se pierda el flujo automático.
- Para que sirve:
  - Reduce falsos errores de “highlight no funciona” en proyectos con poco material elegible.
  - Evita que el modo highlight se desactive al reabrir proyectos.

### Ajustado: Se removió botón de recorte rápido (tijeras) en card de clip
- Que cambiamos:
  - En `dashboard/src/components/ResultCard.jsx` se quitó el botón ícono de tijeras del footer de acciones del clip.
- Para que sirve:
  - Simplifica la interfaz y elimina una acción que ya no se quiere exponer en esa vista.


## 2026-02-17

### Documentacion: guia de ejecucion y diagnostico de conexion
- Que cambiamos:
  - Se actualizo `EJECUTAR.md` con flujo claro para correr en local (`./start.sh`) y en modo remoto (Colab/ngrok).
  - Se documentaron verificaciones reales de conectividad (`/docs` y `/api/status/__healthcheck__`).
  - Se agrego troubleshooting para errores frecuentes: `curl (7)`, `ERR_NGROK_725`, CORS y URL vieja.
  - Se incluyo el caso clave de `localStorage` (`openshorts_api_base_url`) que puede pisar `.env.local`.
- Para que sirve:
  - Reduce tiempo de diagnostico cuando API/ngrok aparecen en rojo.
  - Evita falsos errores de CORS por tunel vencido o configuracion cacheada en el navegador.


## 2026-02-16

### Agregado: Highlight Reel compuesto (multi-momento viral)
- Que cambiamos:
  - Se agrego `POST /api/highlight/reel` en `app.py`.
  - El backend ahora arma un reel unico con varios momentos top por `virality_score`.
  - Se aplica diversidad temporal para evitar tomar solo momentos solapados o demasiado cercanos.
  - Se ajusta la duracion de cada momento para cumplir una duracion objetivo del reel final.
  - Se renderizan segmentos intermedios y luego se concatenan en un MP4 final.
  - El resultado se persiste en metadata y en `job.result`:
    - `highlight_reels`
    - `latest_highlight_reel`
  - En `dashboard/src/App.jsx` se agrego:
    - Boton `Generar highlight reel`.
    - Preview/reproduccion del reel generado.
    - Atajos de `Abrir video` y `Descargar`.
- Para que sirve:
  - Convierte momentos aislados en una pieza teaser con narrativa de varios hooks.
  - Ayuda a invitar a la audiencia a ver el podcast/video completo.

### Agregado: Botón para regenerar título del clip en Clip Studio
- Que cambiamos:
  - Se agrego `POST /api/clip/retitle` en `app.py`.
  - El endpoint genera una nueva variante de título con IA (Gemini) y fallback local si no hay IA disponible.
  - La respuesta actualiza y persiste:
    - `video_title_for_youtube_short`
    - `title`
  - Se guarda en metadata del proyecto y tambien en `job.result` para mantener estado consistente.
  - En `dashboard/src/components/ClipStudioModal.jsx` se agrego el botón `Regenerar título` junto al título del preview.
  - En `dashboard/src/App.jsx` se agregó `onClipPatched` para actualizar el clip en UI sin cerrar el editor.
- Para que sirve:
  - Permite iterar rápido el título cuando la primera propuesta no convence.
  - Evita reprocesar todo el proyecto solo para ajustar copy del encabezado.

### Corregido: Aplicar subtítulos ya no mezcla renders anteriores
- Que cambiamos:
  - En `POST /api/subtitle` se resolvio el origen real del video cuando llega un `input_filename` que ya era `subtitled_*`.
  - Ahora el backend remueve prefijos `subtitled_` encadenados (si existe el archivo base) antes de quemar subtítulos de nuevo.
  - El archivo de salida de subtítulos ahora siempre se genera con sufijo único (`timestamp + id corto`) para evitar confusión de caché/versión.
- Para que sirve:
  - Evita que al re-aplicar subtítulos se encimen con subtítulos viejos.
  - Reduce casos donde parece que “aplicó otro subtítulo” por estar viendo una versión anterior.

### Corregido: Descarga de clip en entornos remotos (Colab/ngrok)
- Que cambiamos:
  - En `dashboard/src/components/ResultCard.jsx` la descarga ahora usa `apiFetch` para URLs remotas de video en lugar de `fetch` directo.
  - Se unifico el flujo de descarga para ambos botones (`Descargar` principal y el del modal de recorte).
  - Se agrego deteccion de respuestas HTML invalidas (p. ej. pagina intermedia del proxy) y fallback automatico para abrir el archivo.
  - Se mejoro el nombre del archivo descargado a partir del titulo del clip.
- Para que sirve:
  - Mejora la compatibilidad cuando el backend corre en Colab/ngrok.
  - Evita descargas que no arrancan o respuestas incorrectas al bajar el clip final.

### Corregido: Contraste de textos en panel de resultados (tema claro)
- Que cambiamos:
  - Se ajustaron colores de badges (`Fuente`, `Objetivo`, `Ratio`, `Costo`) en `dashboard/src/App.jsx`.
  - Se actualizaron labels y controles (`select/input`) del bloque de filtros para usar variantes claras/oscura con mayor legibilidad.
- Para que sirve:
  - Evita textos casi invisibles en fondo claro.
  - Mantiene lectura consistente en ambos temas (light/dark).

### Ajustado: Footer de acciones en card de clip
- Que cambiamos:
  - En `dashboard/src/components/ResultCard.jsx` se reordeno el footer para dejar iconos de edicion a la izquierda y boton `Descargar` a la derecha.
  - Se forzo color blanco en texto e icono de `Descargar` para evitar contraste bajo por estilos heredados.
- Para que sirve:
  - Alinea el flujo visual de acciones como solicitaste.
  - Mejora legibilidad del CTA principal de descarga.

### Corregido: Generar highlight reel con fallback de endpoint
- Que cambiamos:
  - En backend (`app.py`) se agrego alias `POST /api/highlight-reel` ademas de `POST /api/highlight/reel`.
  - En frontend (`dashboard/src/App.jsx`) el flujo de generacion ahora intenta ambos endpoints en orden.
  - Se mejoro el log de error para casos `404 Not Found`, indicando que el backend activo puede estar desactualizado (comun en Colab/ngrok).
- Para que sirve:
  - Evita fallos silenciosos por desajuste de ruta entre versiones de frontend/backend.
  - Da diagnostico claro cuando hace falta reiniciar el servidor remoto.

### Ajustado: Progreso de procesamiento sin cajones por paso
- Que cambiamos:
  - En `dashboard/src/App.jsx` se removio la fila de tarjetas/cajones verdes de etapas.
  - El timeline ahora trabaja con 5 pasos compactos:
    - Enviando enlace/Subiendo video
    - Procesando video
    - Buscando mejores momentos
    - Generando clips
    - Finalizando
  - El estado de paso se muestra en el mismo lugar del porcentaje con formato dinamico:
    - `Paso X de 5: ... (Y%)`
    - Al completar: `Paso 5 de 5 completados al 100%`
- Para que sirve:
  - Ahorra espacio vertical en la cabecera del procesamiento.
  - Hace mas clara la lectura del avance por etapa en un solo indicador.

### Corregido: Legibilidad de chips de búsqueda semántica
- Que cambiamos:
  - En `dashboard/src/App.jsx` se ajustaron colores de contraste para los chips:
    - `semántica`
    - `intención`
    - `modo`
  - Se mejoro el color de textos auxiliares (`palabras clave`, `frases`, `alcance`) en tema claro, manteniendo variantes dark.
- Para que sirve:
  - Evita que los badges se vean lavados o casi invisibles en fondo claro.
  - Mejora lectura rápida del estado del buscador híbrido.

### Corregido: Contraste en botones de acciones de clips
- Que cambiamos:
  - En `dashboard/src/App.jsx` se ajustaron colores de botones del bloque de acciones:
    - `Encolar`
    - `Exportar paquete`
    - `Generar highlight reel`
  - En tema claro ahora usan fondos y texto con mayor contraste, manteniendo variantes dark.
- Para que sirve:
  - Evita que los textos de botones se pierdan en fondo claro.
  - Mejora visibilidad y clicabilidad de acciones principales.

### Ajustado: Header de tarjeta de clip sin badges de score/confianza
- Que cambiamos:
  - En `dashboard/src/components/ResultCard.jsx` se removieron los badges superiores:
    - `Puntaje X/100`
    - `Confianza X%`
  - Se mantiene el rango de tiempo del clip en ese bloque.
- Para que sirve:
  - Limpia ruido visual en la cabecera del card.
  - Deja solo contexto operativo (rango) y evita duplicar métricas.

### Ajustado: Remoción global de métricas de score/confianza en cards
- Que cambiamos:
  - En `dashboard/src/components/ResultCard.jsx` se quitaron todas las superficies restantes de `Puntaje/Confianza`:
    - bloque lateral de `Virality`
    - pestaña `Puntaje viral`
    - contenido interno con barra de viralidad y confianza del modelo
- Para que sirve:
  - Unifica la decisión de no mostrar métricas de scoring en ninguna tarjeta.
  - Simplifica la UI para enfocarla en edición/publicación.

### Agregado: Layout multiple tipo Split (2 personas)
- Que cambiamos:
  - Se agrego `layout_mode` con opcion `split` en `POST /api/recut`.
  - En Split se renderizan dos paneles del mismo clip:
    - `9:16`: panel superior + panel inferior.
    - `16:9`: panel izquierdo + panel derecho.
  - Se agregaron controles independientes por panel en Clip Studio:
    - `Zoom A/B`
    - `Offset X/Y A/B`
  - Se agrego preview Split en tiempo real con dos videos sincronizados en el editor.
  - Se persisten en metadata los nuevos campos:
    - `layout_mode`
    - `layout_split_zoom_a`, `layout_split_offset_a_x`, `layout_split_offset_a_y`
    - `layout_split_zoom_b`, `layout_split_offset_b_x`, `layout_split_offset_b_y`
- Para que sirve:
  - Permite mostrar dos interlocutores en pantalla al mismo tiempo (estilo podcast/entrevista).
  - Facilita separar visualmente cada hablante sin salir de Clip Studio.

### Agregado: Contexto visual en preview de edicion
- Que cambiamos:
  - Se agrego una cabecera informativa arriba del preview en Clip Studio (estilo ficha).
  - La cabecera muestra el nombre del clip en edicion y el contexto social (plataformas + copy corto).
  - Se agrego badge de `Virality` con score normalizado (0.0 - 10.0) cuando existe.
  - Se removio el prefijo de redes en la sublinea (`YouTube / TikTok / Instagram |`) para dejar solo el texto.
  - Se removio el contenedor visual de la cabecera para dejar la info en estilo limpio (sin recuadro).
  - Se removio el bloque visual de `Virality` en la cabecera.
  - Se aumento el tamano tipografico del titulo y del resumen social para mejorar legibilidad.
  - Se redujo ~30% el tamano del titulo y del resumen social por ajuste visual fino.
  - Se usan fallbacks (titulo general, nombre de archivo o numero de clip) si faltan datos.
- Para que sirve:
  - Da contexto inmediato sobre que video se esta editando.
  - Evita confusiones cuando se trabaja varios clips seguidos.

### Corregido: Recorte falso al mover encuadre en "Editar layout"
- Que cambiamos:
  - El preview de layout dejo de mover el elemento `<video>` con `translate`.
  - Ahora el paneo manual usa `object-position` (encuadre interno) y `scale` para zoom.
  - Se mantuvo la sensibilidad/offset efectiva para no romper valores guardados.
- Para que sirve:
  - Evita que aparezcan franjas vacias o zonas "cortadas" al arrastrar un poco.
  - El movimiento del encuadre aprovecha mejor el ancho/alto real disponible del video.

### Agregado: Control anti-corte abrupto de transcripcion
- Que cambiamos:
  - Se agregaron controles de `Empezar antes (s)` y `Terminar después (s)` en Layout de Clip Studio.
  - Se agregaron atajos rapidos `+0.2s al inicio` y `+0.2s al final`.
  - El recut ahora aplica ese pre/post-roll al enviar `start/end` al backend.
  - El backend devuelve `start` y `end` finales en la respuesta de `/api/recut` para mantener el estado del clip sincronizado en frontend.
- Para que sirve:
  - Evita que el clip corte palabras justo al inicio o al final.
  - Permite afinar rapidamente contexto al hablar, sin editar manualmente todo el rango.

### Agregado: Auto Smart Reframe en Clip Studio
- Que cambiamos:
  - Se agrego el modo `auto_smart_reframe` al endpoint `POST /api/recut` en `app.py`.
  - Se implemento un pipeline de reencuadre inteligente por escena:
    - Deteccion de escenas con `scenedetect`.
    - Deteccion de personas con YOLO (`ultralytics`) + fallback de rostro.
    - Estrategia por escena: `TRACK` (recorte al sujeto) o `LETTERBOX` (preserva toma completa).
    - Render frame a frame hacia FFmpeg con audio del clip.
  - Se guardan nuevas banderas en metadata del clip:
    - `layout_auto_smart`
    - `layout_smart_summary` (resumen de escenas y estrategia aplicada)
- Para que sirve:
  - Reduce trabajo manual de layout.
  - Mantiene al hablante principal centrado automaticamente.
  - Evita recortes agresivos en escenas grupales o abiertas.

### Actualizado: UI de Layout para controlar Smart Reframe
- Que cambiamos:
  - Se agrego el toggle `Auto smart reframe (beta)` en `dashboard/src/components/ClipStudioModal.jsx`.
  - El frontend envia el flag al backend en recut (`auto_smart_reframe`).
  - El estado aplicado se persiste con `layout_auto_smart`.
  - Se desactiva el pan manual cuando smart mode esta activo.
- Para que sirve:
  - Permite control directo desde UI sin pasos extra.
  - Evita conflictos entre encuadre automatico y manual.

### Corregido: Espaciado de palabras en preview de karaoke
- Que cambiamos:
  - Se ajusto el espaciado para usar margen entre palabras en lugar de espacios finales en `inline-block`.
- Para que sirve:
  - Evita que las palabras se vean pegadas en el preview karaoke.

### Corregido: Falso error de reproduccion en preview de video
- Que cambiamos:
  - Se evito mostrar warnings falsos de `"El navegador no pudo reproducir..."` cuando el fallback directo si carga.
  - Se limpia el error en `onCanPlay` y `onLoadedData`.
  - `onError` ahora ignora casos donde ya hay frame decodificado valido.
- Para que sirve:
  - Elimina alertas confusas en la vista previa de Clip Studio.

### Agregado: Highlight reel semantico independiente de clips existentes
- Que cambiamos:
  - En `app.py` se agrego `source_mode` a `HighlightReelRequest` (default `semantic`).
  - El endpoint `POST /api/highlight/reel` ahora intenta primero descubrir momentos desde transcript/timeline completo (modo semantico), en vez de depender solo de `shorts` ya generados.
  - Se agrego resolucion de fuente para highlight:
    - prioriza `job.input_path`
    - si no existe, busca video maestro en `output/<job_id>` excluyendo artefactos generados (`*_clip_*`, `highlight_reel_*`, `reclip_*`, `subtitled_*`, `temp_*`, etc.).
  - Se agrego fallback automatico al modo legacy por clips cuando no hay fuente o no hay transcript suficiente.
  - En `process_endpoint` se fuerza `--keep-original` para jobs por URL, de modo que el backend conserve el video fuente necesario para recortes semanticos.
  - En `dashboard/src/App.jsx`, `handleGenerateHighlightReel` ahora envia `source_mode: "semantic"` al backend.
- Para que sirve:
  - Permite que el highlight reel encuentre momentos nuevos fuera de los clips ya existentes.
  - Reduce errores por dependencia de un set limitado de clips.
  - Mantiene compatibilidad con jobs viejos gracias al fallback.

### Corregido: Error `name 'unicodedata' is not defined` en ejecucion
- Que cambiamos:
  - Se agrego `import unicodedata` en `app.py`.
- Para que sirve:
  - Evita caida del backend al normalizar fingerprints de titulos durante la generacion/regeneracion.

### Actualizado: Highlight reel sin limite rigido de duracion
- Que cambiamos:
  - `POST /api/highlight/reel` ahora interpreta:
    - `target_duration <= 0` como duracion automatica libre (usa duracion de fuente/timeline).
    - `max_segments <= 0` como seleccion automatica amplia de segmentos.
  - Se removieron topes internos cortos (antes 12 segmentos / 240s), ampliando limites operativos para reels largos.
  - En frontend (`dashboard/src/App.jsx`) el payload de highlight ahora envia:
    - `target_duration: 0`
    - `max_segments: 0`
  - El boton de highlight ya no bloquea por cantidad de clips en UI (solo por estado/job).
- Para que sirve:
  - Permite generar highlight reels largos sin tope fijo de tiempo.
  - Reduce la necesidad de reintentar manualmente para “sumar” duracion.

### Restaurado: Métricas de viralidad en tarjeta de clip
- Que cambiamos:
  - En `dashboard/src/components/ResultCard.jsx` se restauraron:
    - badges superiores de `Puntaje` y `Confianza`
    - bloque lateral `Virality` (escala 0.0-10.0)
    - pestaña `Puntaje viral` con barra, confianza y `score_reason`
- Para que sirve:
  - Recupera visibilidad del scoring para decidir rápido qué clips priorizar.
  - Alinea la UI con el flujo anterior que el usuario venía usando.

### Ajustado: Orden visual entre "Regenerar" y bloque de Virality
- Que cambiamos:
  - En `dashboard/src/components/ResultCard.jsx` se invirtió el orden del header derecho para mostrar:
    - botón `Regenerar` primero
    - bloque `Virality` a la derecha
- Para que sirve:
  - Evita lectura visual rara en cards angostos y mantiene jerarquía de acción > métrica.
