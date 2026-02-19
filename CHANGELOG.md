# Registro de Cambios

## 2026-02-18

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
