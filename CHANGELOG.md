# Registro de Cambios

## 2026-02-16

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
