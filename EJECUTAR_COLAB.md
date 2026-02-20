# EJECUTAR CON COLAB (MODO COLAB-FIRST)

Esta guía asume que todo el peso de la Inteligencia Artificial y FFmpeg (Backend) se va a procesar en un servidor externo gratuito proveído por **Google Colab** + la herramienta **Ngrok**.

Tu computadora (Mac) únicamente va a encender la interfaz gráfica web (Frontend) mediante Node.js, ahorrando memoria y procesador local.

---

## 1. Configurar y Encender el Backend Remoto (Colab)

Ve a tu libreta o crea un cuaderno nuevo en Google Colab con entorno **T4 GPU** o superior.

Ejecuta el bloque principal para descargar e instalar el código allí:

```python
%cd /content
!if [ -d /content/OpenShortsSanti/.git ]; then \
  cd /content/OpenShortsSanti && git checkout main && git pull --ff-only origin main; \
else \
  rm -rf /content/OpenShortsSanti && git clone https://github.com/Santiagoc57/OpenShortsSanti.git /content/OpenShortsSanti; \
fi
%cd /content/OpenShortsSanti

!python3 -m pip install -U pip
!pip install pyngrok
!apt-get update -y && apt-get install -y ffmpeg
!python3 -m pip install -r requirements.txt
```

Luego, usa tu Token de Ngrok para enlazar tu nube al mundo real:

```python
import os
from pyngrok import ngrok

# Configuraciones para acelerar Whisper en Colab 
os.environ["WHISPER_BACKEND"] = "openai"
os.environ["WHISPER_MODEL"] = "base"
os.environ["WHISPER_DEVICE"] = "cuda"

# Pega aquí tu token real de ngrok
ngrok.set_auth_token("TU_TOKEN_DE_NGROK_AQUI")
```

Enciende el servidor internamente:

```python
!pkill -f "uvicorn app:app" || true
!nohup python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 > /content/backend.log 2>&1 &
!sleep 5
!tail -n 40 /content/backend.log
```

Por último, levanta el puente y obtén tu **URL PÚBLICA**:

```python
from pyngrok import ngrok

ngrok.kill()
tunnel = ngrok.connect(8000, "http")
public_url = tunnel.public_url

print("Copia exactamente esta URL para pegarla en tu Web de Mac:\n")
print(public_url)
print("\n")
```

Copia esa línea con `https://xxxxxx.ngrok-free.app` que te arroja Colab.

---

## 2. Arrancar el Frontend en tu Mac

En tu Mac abre una ventana de terminal, navega a la carpeta y enciende la web:

```bash
# 1. Navegar a la carpeta frontend 
cd "/Users/santiagocordoba/GITHUBS/-- 05 Openshorts-main 2/dashboard"

# 2. Encender 
npm run dev -- --host 0.0.0.0 --port 5173
```

---

## 3. Enlazar la Web con tu Colab (Paso Final)

1. Abre en el navegador de tu Mac: [http://localhost:5173](http://localhost:5173)
2. Ve a **Configuración** (ícono del engranaje en la esquina superior derecha).
3. Busca el campo llamado **"Backend remoto (Colab / ngrok)"**.
4. Pega ahí la URL pública de ngrok que copiaste en el paso de Colab. No incluyas barras diagonales ni textos extra, solo `https://...`
5. Pulsa **Guardar**.
   
Tu Mac ahora enviará todas sus operaciones exigentes al servidor remoto de Colab.

---

### Solución de Problemas Frecuentes

- **Colab se reinicia y arroja un error 404/NetworkError**: Cada vez que tu sesión de Colab expira, la URL pública cambia. Debes correr todos los comandos nuevamente, ir a la configuración en tu Mac, y pegar la nueva URL que te arrojó Ngrok hoy.
- **`ERR_NGROK_324`**: Has creado muchos túneles o se te bugeó Ngrok. Corre solo la instrucción `ngrok.kill()` en tu cuaderno para limpiarlos, y activa tu código de `ngrok.connect` una vez más.
- **`NetworkError` con URL correcta**: Si pegaste bien la URL de Colab, pero sigue fallando, es probable que la celda de Colab que dice `uvicorn app:app...` haya fallado por falta de dependencias u otro error en la nube. Revisa los logs de Colab.
