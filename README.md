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

## 🧭 Product Roadmap (Opus-like + Differentiation)

This roadmap is based on the current codebase and benchmarked feature sets from Opus/Vizard/Flowjin/Descript-style products.

### Status Snapshot

#### ✅ Already implemented
- Upload + YouTube URL ingestion.
- AI highlight detection with ranked clip proposals.
- Vertical reframing with face/speaker tracking.
- Subtitle generation + preview + style controls.
- Clip post-editing (`Auto Edit`, `Recut`) and social posting.
- Scheduled publishing via Upload-Post.

#### 🟡 Partially implemented
- Virality ranking exists, but no explicit numeric `virality_score` in UI/API.
- Social distribution works, but no unified content calendar UX.
- Video-first flow is complete; audio-only podcast flow is not first-class.

#### ⛔ Not implemented yet
- Search-inside-video (`clip anything` by prompt/topic/moment intent).
- Brand kit/templates per team (logo/fonts/safe margins/presets).
- Auto B-roll + emoji packaging.
- Multi-ratio output strategy (9:16 / 1:1 / 16:9 as a first-class option).
- Agency export pack (clips + copies + hashtags + SRT + thumbnails + publish plan).
- Full “suite” extras (filler-word removal, audio enhancement, translation workflow).

### Phase Plan

#### Phase 1 — Core Competitive (MVP hardening)
1. Add `virality_score` per clip in backend metadata and dashboard card.
2. Add clip sorting/filtering by score (`Top`, `Balanced`, `Safe`).
3. Expose confidence/explanation metadata for clip selection.
4. Add audio-only ingestion mode for podcasts.

#### Phase 2 — Revenue Features (V1 paid)
1. Brand kit and reusable templates.
2. Better scheduling UX (calendar view + per-platform copy preview).
3. Batch actions for teams (approve/schedule multiple clips).

#### Phase 3 — Product Differentiation (V2)
1. `Clip Anything`: natural-language search over transcript + timestamp retrieval.
2. Long-VOD navigation (chapters, topic clusters, no-scrub exploration).
3. Agency export package and optional human-QA workflow.

### Suggested Engineering Track (Execution Order)

1. API schema extension: `virality_score`, `score_reason`, `topic_tags`.
2. Dashboard update: score badges + sorting controls + queue actions.
3. Search layer: semantic index over transcript chunks.
4. Packaging layer: template presets + brand assets + team exports.
5. Post-production extras: audio cleanup + translation + style automation.

### Open-Source / Integration Boosters

- `kaixxx/noScribe`: local whisper + diarization ideas for speaker-aware workflows.
- `tryvinci/vinci-clips`: reference architecture for long-form to short-form pipelines.
- MCP stack in this project:
  - GitHub code search patterns for feature implementation references.
  - Context docs retrieval for FastAPI/video pipeline libraries.
  - Browser automation for E2E validation of clip generation/publishing flows.

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
