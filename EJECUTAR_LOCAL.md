# EJECUTAR LOCALMENTE (MODO 100% LOCAL)

Esta guía ha sido actualizada para Mac (procesador Intel o M1/M2) con los ajustes necesarios para que las librerías de IA funcionen correctamente.

## Requisitos Previos

- **Importante**: No usar Python 3.13 ni 3.14 (son demasiado nuevos para las librerías de IA).
- **Usar Python 3.10** (instalable con `brew install python@3.10`).
- **Limpiar configuración**: Si tienes un archivo `.env.local` dentro de la carpeta `dashboard`, asegúrate de que esté vacío o borra cualquier enlace de `ngrok` que veas ahí.
- **FFmpeg con subtítulos**: ¡Ya lo instalamos! He configurado una versión especial en tu carpeta `.venv` que ya trae todo lo necesario para los subtítulos. No necesitas volver a instalarlo con `brew`.

---

## 1. Configurar y Arrancar el Backend (AI Server)

Abre una terminal y ejecuta paso a paso el proceso de limpieza y arranque:

```bash
# 1. Navegar a la carpeta raíz
cd "/Users/santiagocordoba/GITHUBS/-- 05 Openshorts-main 2"

# 2. Re-crear el entorno virtual forzando Python 3.10
rm -rf .venv
/usr/local/bin/python3.10 -m venv .venv

# 3. Activar el entorno
source .venv/bin/activate

# 4. Arreglar "setuptools" antes de instalar dependencias
pip install setuptools==69.5.1

# 5. Instalar librerias que dan problemas (numba / llvmlite) de forma pre-compilada
pip install llvmlite==0.41.1 numba==0.58.1 opencv-contrib-python==4.10.0.84 --only-binary=:all:

# 6. Instalar el resto de requerimientos
pip install -r requirements.txt

# Si pip queda resolviendo Whisper/WhisperX o falla con pkg_resources:
pip install "setuptools<81" wheel
pip install "numpy<2" llvmlite==0.41.1 numba==0.58.1 --only-binary=:all:
pip install faster-whisper==1.1.0 openai-whisper==20250625 whisperx==3.3.1 nest-asyncio==1.6.0
pip install -r requirements.txt

# 7. Arrancar el servidor
cd "/Users/santiagocordoba/GITHUBS/-- 05 Openshorts-main 2"
source .venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

> **Deja esta ventana abierta**, verás los logs de procesamiento aquí. No cierres la terminal.

---

## 2. Arrancar el Frontend (Panel Web)

Abre una **NUEVA ventana de terminal** (`Cmd + T`) y entra a la subcarpeta de la web:

```bash
# 1. ENTRAR a la subcarpeta dashboard (Crítico)
cd "/Users/santiagocordoba/GITHUBS/-- 05 Openshorts-main 2/dashboard"

# 2. Arrancar Vite
npm run dev -- --host 0.0.0.0 --port 5173

```

---

## 3. Opcional: Arrancar Remotion Render Service (para el botón Remotion)

Solo necesitas esta tercera terminal si vas a usar el botón **Remotion** dentro de Clip Studio.

```bash
cd "/Users/santiagocordoba/GITHUBS/-- 05 Openshorts-main 2/remotion"
npm install
npm run build

cd "/Users/santiagocordoba/GITHUBS/-- 05 Openshorts-main 2/render-service"
npm install
npm run build
OUTPUT_DIR="../output" REMOTION_BUNDLE_PATH="../remotion" PORT=3100 npm start
```

El backend local usa por defecto:

```bash
RENDER_SERVICE_URL="http://localhost:3100"
```

Si lo cambias, exporta esa variable antes de arrancar `uvicorn`.

---

## 4. Configuración en la Web

1. Entra a [http://localhost:5173](http://localhost:5173).
2. Haz clic en el **engranaje** (Configuración) arriba a la derecha.
3. El campo **"Backend remoto (Colab / ngrok)"** debe estar **VACÍO** para que la web detecte tu Mac local.
4. Pulsa **Guardar**.

### Si la web no se conecta al Backend
- Borra la memoria del navegador pegando esto en la consola (F12):
  `localStorage.removeItem('openshorts_api_base_url'); location.reload();`
- Verifica que en la terminal del Backend no haya errores en rojo.

---

## 5. Manifests locales por job

Cada procesamiento genera:

```text
output/<job_id>/job_manifest.json
```

Ese archivo guarda comando, estado, eventos, artefactos generados y entorno relevante sin exponer claves.

También puedes verlo por API:

```bash
curl http://localhost:8000/api/projects/manifest/<job_id>
```
