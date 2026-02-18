# EJECUTAR (Modo Colab-First)

Este flujo asume:
- Backend corriendo en Colab (con ngrok).
- Tu Mac solo corre el frontend.

## 1) Frontend en tu Mac

```bash
cd "/Users/santiagocordoba/GITHUBS/-- 05 Openshorts-main 2"
cd dashboard
npm run dev -- --host 0.0.0.0 --port 5173
```

Abre:

```text
http://localhost:5173
```

En la app:
- Ve a `Configuracion`.
- Pega tu URL de Colab/ngrok en `Backend remoto (Colab / ngrok)`.
- Pulsa `Guardar`.

Alternativa por terminal (opcional):

```bash
cd "/Users/santiagocordoba/GITHUBS/-- 05 Openshorts-main 2"
./set-colab-api.sh https://TU-NGROK.ngrok-free.app
```

## 2) Backend en Colab

Ejemplo minimo en Colab:

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

ngrok.set_auth_token("TU_TOKEN")
```

```python
!nohup python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 > /content/backend.log 2>&1 &
!sleep 5
!tail -n 40 /content/backend.log
```

```python
from pyngrok import ngrok
import requests

ngrok.kill()
tunnel = ngrok.connect(8000, "http")
public_url = tunnel.public_url
print(public_url)

print("docs:", requests.get(f"{public_url}/docs").status_code)
print("health:", requests.get(f"{public_url}/api/status/__healthcheck__?ts=1").status_code)
```

Usa ese `public_url` en la UI (`Configuracion`) o en `./set-colab-api.sh`.

## 3) Cuando cambia la URL ngrok

Cada vez que Colab reinicia runtime o recrea tunel, la URL puede cambiar.

Actualiza asi (recomendado):

1. En la app, abre `Configuracion`.
2. Pega la nueva URL `https://NUEVA-URL.ngrok-free.app`.
3. Pulsa `Guardar`.

Alternativa por terminal:

```bash
cd "/Users/santiagocordoba/GITHUBS/-- 05 Openshorts-main 2"
./set-colab-api.sh https://NUEVA-URL.ngrok-free.app
```

Nota:
- Si cambias la URL desde la UI, no hace falta reiniciar Vite.
- Si cambias la URL con `set-colab-api.sh`, si conviene reiniciar Vite.

## 4) No usar en este modo

- `./start-ngrok.sh` en tu Mac.
- `./start.sh` para procesamiento pesado.
- `python -m uvicorn ...` local para este flujo.

## 5) Troubleshooting rapido

### NetworkError / Failed to fetch / CORS

Normalmente significa tunel caido, URL vieja o ngrok devolviendo error HTML.

Valida desde tu Mac:

```bash
curl -i https://TU-NGROK.ngrok-free.app/docs
curl -i "https://TU-NGROK.ngrok-free.app/api/status/__healthcheck__?ts=$(date +%s)"
```

Si falla aqui, el problema esta en Colab/ngrok, no en Vite.

### Frontend sigue usando URL vieja

La app prioriza `localStorage` (`openshorts_api_base_url`) sobre `.env.local`.

En consola del navegador:

```js
localStorage.removeItem('openshorts_api_base_url');
location.reload();
```

Luego vuelve a guardar la URL correcta en Configuracion.

### `ERR_NGROK_324` (max endpoints)

Tienes demasiados tuneles activos en esa sesion.

En Colab:

```python
from pyngrok import ngrok
ngrok.kill()
```

Y crea solo un tunel con `ngrok.connect(8000, "http")`.
