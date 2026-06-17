# OpenShorts.app 🚀🎬

OpenShorts is an all-in-one open-source solution to automate the creation and distribution of viral vertical content. It transforms long YouTube videos or local files into high-potential short clips optimized for **TikTok**, **Instagram Reels**, and **YouTube Shorts**.

![OpenShorts Demo](https://github.com/kamilstanuch/Autocrop-vertical/blob/main/churchil_queen_vertical_short.gif?raw=true)

### 📺 Video Tutorial: How it works
[![OpenShorts Tutorial](https://img.youtube.com/vi/xlyjD1qCaX0/maxresdefault.jpg)](https://www.youtube.com/watch?v=xlyjD1qCaX0 "Click to watch the video on YouTube")

*Click the image above to watch the full walkthrough.*

---

## ✨ Key Features

OpenShorts leverages state-of-the-art AI to handle the entire content lifecycle:

1.  **🧠 Viral Moment Detection:**
    *   **Faster-Whisper**: High-speed, CPU-optimized transcription and word-level timestamps.
    *   **Google Gemini 2.0 Flash**: Advanced AI analysis to identify the 3-15 most viral moments based on hooks and engagement potential.
    *   **Automatic Copywriting**: Generates SEO-optimized titles and descriptions for all platforms.

2.  **✂️ Smart AI Cropping & Tracking (New V2 Engine):**
    *   **Dual-Mode Strategy**: Automatically detects scene composition to apply the best framing strategy.
        *   **TRACK Mode (Single Subject)**: Uses **MediaPipe Face Detection** + **YOLOv8** fallback for ultra-fast, robust subject tracking. Features a **"Heavy Tripod" stabilization engine** that eliminates jitter and unnatural movements, providing smooth, cinematic reframing. Includes **Speaker Identification** to stick to the active speaker and avoid erratic switching.
        *   **GENERAL Mode (Groups/Landscapes)**: For scenes with multiple people or no clear subject, it automatically switches to a professional **blurred-background layout**, preserving the full width of the original shot while filling the 9:16 vertical space.
    *   **Intelligent Scene Analysis**: Pre-scans every scene to determine the optimal strategy before processing.

3.  **☁️ Automated S3 Backup:**
    *   **Silent Background Upload**: Once clips are generated, they are automatically uploaded to an AWS S3 bucket.
    *   **Seamless Integration**: Operates in the background without affecting processing logs or UI performance.

4.  **📲 Direct Social posting:**
    *   **Upload-Post Integration**: Share your generated clips directly to TikTok, Instagram, and YouTube with a single click.
    *   **Profile Selector**: Manage multiple social accounts easily through the dashboard.

4.  **🎨 Modern Web Dashboard:**
    *   **Real-time Progress**: Watch clips appear as they are generated with a live results feed.
    *   **Log Streaming**: Follow the technical process with real-time log updates.
    *   **Responsive Design**: A premium, dark-mode glassmorphism interface.

---

## 🛠️ Requirements

*   **Docker & Docker Compose**.
*   **Google Gemini API Key** ([Get it for free here](https://aistudio.google.com/app/apikey)).
*   **Upload-Post API Key** (Optional, for direct social posting. **Free tier available, no credit card required**).

### 📲 Social Media Setup (Upload-Post)
To enable direct posting, follow these steps:
1.  **Login/Register**: [app.upload-post.com/login](https://app.upload-post.com/login)
2.  **Create Profile**: Go to [Manage Users](https://app.upload-post.com/manage-users) and create a user profile.
3.  **Connect Accounts**: In the same section, connect your TikTok, Instagram, or YouTube accounts to that profile.
4.  **Get API Key**: Navigate to [API Keys](https://app.upload-post.com/api-keys) and generate your key.
5.  **Use in OpenShorts**: Paste the API Key and select your Profile in the dashboard.
    

### ☁️ AWS S3 Setup (Optional)
To enable automatic backup of your clips to S3:
1. **Environment Variables**: Set the following in your `.env` file or system environment:
    * `AWS_ACCESS_KEY_ID`: Your AWS access key.
    * `AWS_SECRET_ACCESS_KEY`: Your AWS secret key.
    * `AWS_REGION`: (Optional) Defaults to `us-east-1`.
    * `AWS_S3_BUCKET`: (Optional) Defaults to `openshorts.app-clips`.
2. **Bucket**: Clips are uploaded to the specified bucket automatically after generation.
3. **Restore**: The Project Library can list remote job prefixes and restores archived clips on demand when a project is opened.

### 🔒 Safety & Rights
OpenShorts requires users to confirm they own the content or have permission to process it before starting a job.

Optional deployment flag:
* `DISABLE_YOUTUBE_URL=true`: disables YouTube URL ingestion and accepts uploaded files only.

### 🎬 Remotion Render Service (Optional)
This fork includes a separate Remotion render service for modern React-based video renders with animated captions, hooks and effects.

With Docker Compose it starts automatically as `render-service` on port `3100` and writes final MP4 files into the shared `./output` folder. In Clip Studio, use the `Remotion` button to render the current clip with the service. The backend also proxies it through:

* `GET /api/render/health`
* `POST /api/render/remotion`
* `GET /api/render/remotion/{render_id}`

For local non-Docker runs, start it separately:

```bash
cd render-service
npm install
npm run build
OUTPUT_DIR="../output" npm start
```

If the backend runs outside Docker, point it to the service:

```bash
export RENDER_SERVICE_URL="http://localhost:3100"
```

---

## 🚀 Getting Started

The easiest way to run OpenShorts is using Docker Compose.

### 1. Setup
```bash
git clone https://github.com/your-username/OpenShorts.git
cd OpenShorts
```

### 2. Launch the Application
```bash
docker compose up --build
```

### 3. Access the Dashboard
Open your browser and navigate to:
**`http://localhost:5173`**

1.  Enter your **Gemini API Key**.
2.  (Optional) Enter your **Upload-Post API Key** to enable social sharing.
3.  Paste a **YouTube URL** or **Upload a Video**.
4.  Click **"Generate Clips"** and watch the magic happen!

---

## 🏗️ Technical Pipeline

1.  **Ingestion**: Downloads YouTube videos via `yt-dlp` or handles local uploads.
2.  **Transcription**: `faster-whisper` converts audio to text in seconds.
3.  **AI Intelligence**: Gemini reads the transcript and selects periods of high interest.
4.  **Extraction**: FFmpeg precisely cuts the selected segments.
5.  **Reframing**: AI-powered visual tracking crops clips to vertical format.
6.  **Backup**: Automated silent upload of clips and metadata to AWS S3.
7.  **Distribution**: One-click posting via Upload-Post API.

---

## 🧭 Hoja de Ruta

La hoja de ruta se mantiene en un documento separado para evitar mezclarla con la guía de instalación:

- **`ROADMAP_ES.md`** (estado, sprints completados, próximos sprints y prioridades)

### QA rápido de Clip Anything (relevancia)
Puedes evaluar búsquedas semánticas contra casos etiquetados:

```bash
python3 scripts/eval_clip_search.py \
  --api-base http://localhost:8000 \
  --job-id <JOB_ID> \
  --cases scripts/clip_search_cases.example.json \
  --search-mode balanced
```

También puedes enviar directamente `POST /api/search/clips/eval`.

---

## 🧩 Novedades Operativas (Colab/Ngrok)

- **Persistencia de jobs**: estado en SQLite (`output/jobs_state.sqlite3`) para recuperar proyectos tras reinicio.
- **Healthcheck real**: `GET /api/status/__healthcheck__`.
- **Recuperación de proyectos**: `GET /api/jobs/recent`.
- **Métricas sociales**: `GET /api/social/metrics/{job_id}`.
- **Highlight reel configurable**: `POST /api/highlight/reel` con `aspect_ratio` (`9:16` o `16:9`).
- **Preview rápido de edición**: `POST /api/clip/fast-preview`.
- **Render Remotion opcional**: `POST /api/render/remotion` y `GET /api/render/remotion/{render_id}`.

---

## 🔒 Security & Performance

*   **Non-Root Execution**: Containers run as a dedicated `appuser` for security.
*   **Concurrency Control**: Configurable job queue (`MAX_CONCURRENT_JOBS`).
*   **Auto-Cleanup**: Automatic purging of old jobs and temporary files.
*   **File Limits**: Built-in protection against oversized uploads.

---

## 🤝 Contributions

Contributions are welcome! Whether it's adding new AI models or improving the cropping engine, feel free to open a PR.

## 📄 License

MIT License. OpenShorts is yours to use, modify, and scale.

---

# 🧭 Instalación Local (Mac, sin Docker) — Guía rápida (lo que hicimos)

> Esta guía resume los pasos y ajustes que usamos para que funcione en local.

## ✅ Requisitos
- Python 3.9+ (ideal 3.10+)
- Node.js + npm
- `ffmpeg` instalado (ej: `ffmpeg 7.x`)

## 1) Backend (Python)
```bash
cd "/Users/santiagocordoba/Downloads/openshorts-main 2"
python3 -m venv .venv
source .venv/bin/activate
```

### Dependencias (evitar errores de NumPy 2.x)
```bash
pip install "numpy<2" --force-reinstall
```

### (Opcional) Evitar build lento de OpenCV
```bash
pip install opencv-contrib-python==4.10.0.84
```

### Instalar requirements
```bash
pip install -r requirements.txt
```

## 2) Frontend (Dashboard)
```bash
cd "/Users/santiagocordoba/Downloads/openshorts-main 2/dashboard"
npm install
```

## 3) Arrancar servicios
### Backend
```bash
cd "/Users/santiagocordoba/Downloads/openshorts-main 2"
source .venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd "/Users/santiagocordoba/Downloads/openshorts-main 2/dashboard"
npm run dev -- --host 0.0.0.0 --port 5173
```

## 4) Ajuste de Vite proxy (si falla el URL)
En `dashboard/vite.config.js` el proxy debe apuntar al backend local:
```
http://localhost:8000
```
Esto evita `getaddrinfo ENOTFOUND backend`.

---

# 🔑 YouTube: Cookies (si el download falla)
Si falla por 403, usar upload manual o cookies válidas.

### Opción A: Usar cookies exportadas
```bash
export YOUTUBE_COOKIES_FILE="/Users/santiagocordoba/Downloads/openshorts-main 2/www.youtube.com_cookies.txt"
```

### Opción B: Cookies directas del navegador (mejor)
```bash
export YOUTUBE_COOKIES_FROM_BROWSER="chrome"
```

> YouTube puede bloquear igual (403). En ese caso, usa **Upload Video**.

---

# 🎙️ Whisper (transcripción)
Para estabilidad en Mac:
```bash
export WHISPER_BACKEND="openai"
export WHISPER_MODEL="base"   # usa "tiny" si quieres más rápido
export WHISPER_DEVICE="cpu"
```

Si quieres velocidad (menos estable):
```bash
export WHISPER_BACKEND="faster"
export WHISPER_MODEL="tiny"
export WHISPER_COMPUTE_TYPE="int8"
export WHISPER_CPU_THREADS="4"
export WHISPER_NUM_WORKERS="1"
```

---

# ⚡ Opciones nuevas en la UI (velocidad + idioma + clips)
En el formulario de upload ahora puedes:
- Idioma (auto/es/en/…)
- Número de clips
- Whisper backend/modelo
- Word timestamps (ON/OFF)
- Preset/CRF de ffmpeg

Esto ajusta la velocidad y evita traducciones no deseadas.

---

# 📝 Subtítulos (edición + estilos)
Hay un modal de subtítulos con:
- Tipografía, tamaño, color, borde, caja
- Botón “Cargar subtítulos” para corregir tildes antes de quemarlos
- Toggle ON/OFF en la tarjeta del clip

---

# ✂️ Edit Video (recut visual)
Nuevo modal visual con preview + sliders:
- Ajusta inicio/fin con timeline
- Botón “Set to playhead”
- Download dentro del modal

⚠️ **Recut requiere que el video original haya sido UPLOAD** (no URL).

---

# 📼 Preview “archivo dañado”
Se forzó salida `yuv420p` y `faststart` para que los previews funcionen en navegador.

---

# ✅ Notas rápidas
- Si el preview falla, prueba regenerar el clip (ahora sale compatible).
- Si Auto Edit no hace nada: falta GEMINI_API_KEY o falla `/api/edit`.

---

# ☁️ Modo Colab-First (verificado)

Este es el flujo estable cuando el procesamiento corre en Colab y tu Mac solo usa el frontend.

## 1) Frontend en Mac
```bash
cd "/Users/santiagocordoba/GITHUBS/-- 05 Openshorts-main 2/dashboard"
npm run dev -- --host 0.0.0.0 --port 5173
```

Luego en la app:
- Ir a `Configuración`.
- Pegar la URL pública de ngrok en `Backend remoto (Colab / ngrok)`.
- Guardar.

> `set-colab-api.sh` queda como alternativa por terminal, pero no es obligatorio si ya usas el campo en la UI.

## 2) Backend en Colab
```python
%cd /content
!rm -rf OpenShortsSanti
!git clone https://github.com/Santiagoc57/OpenShortsSanti.git
%cd /content/OpenShortsSanti

!apt-get update -y
!apt-get install -y ffmpeg
!python3 -m pip install -U pip
!python3 -m pip install -r requirements.txt pyngrok
```

```python
import os
from pyngrok import ngrok

os.environ["WHISPER_BACKEND"] = "openai"
os.environ["WHISPER_MODEL"] = "base"
os.environ["WHISPER_DEVICE"] = "cuda"

ngrok.set_auth_token("TU_NGROK_TOKEN")
```

```python
!pkill -f "uvicorn app:app" || true
ngrok.kill()

!nohup python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 > /content/backend.log 2>&1 &
!sleep 4
!tail -n 60 /content/backend.log
```

```python
from pyngrok import ngrok
import requests

t = ngrok.connect(8000, "http")
url = t.public_url
print(url)

headers = {"ngrok-skip-browser-warning": "true"}
print("docs:", requests.get(f"{url}/docs", headers=headers).status_code)
print("openapi:", requests.get(f"{url}/openapi.json", headers=headers).status_code)
print("health:", requests.get(f"{url}/api/status/__healthcheck__?ts=1", headers=headers).status_code)
```

Valores esperados:
- `/docs` -> `200`
- `/openapi.json` -> `200`
- `/api/status/__healthcheck__` -> `200`

## 3) Verificación desde Mac (opcional, recomendado)
```bash
URL="https://tu-subdominio.ngrok-free.dev"
curl -i -H "ngrok-skip-browser-warning: true" "$URL/docs"
curl -i -H "ngrok-skip-browser-warning: true" "$URL/openapi.json"
```

Si recibes `HTTP 421 Received a request for different Host`, estás usando una URL placeholder o distinta al túnel real.

## 4) Errores comunes
- `NetworkError when attempting to fetch resource`: URL de ngrok vencida/caída o backend no corriendo en Colab.
- `CORS Missing Allow Origin`: normalmente no es CORS real; suele ser respuesta de error de ngrok o URL incorrecta.
- `ERR_NGROK_324`: demasiados endpoints abiertos en la sesión. Ejecuta `ngrok.kill()` y crea un solo túnel.

## 5) Seguridad
- No publiques ni commitees tu `ngrok authtoken`.
- Si se expuso en capturas o logs, rótalo desde el dashboard de ngrok y usa uno nuevo.
