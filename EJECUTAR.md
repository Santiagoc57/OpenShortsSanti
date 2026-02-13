# EJECUTAR (Modo Colab-First)

Este proyecto se usará **siempre conectado a Colab** para procesar videos (tu Mac solo levanta el frontend).

## Flujo recomendado

1. Ve a la raíz del proyecto:

```bash
cd "/Users/santiagocordoba/GITHUBS/openshorts-main 2"
```

2. Configura la URL pública del backend de Colab:

```bash
./set-colab-api.sh https://62cb-34-168-226-133.ngrok-free.app
```

3. Levanta el frontend (Vite):

```bash
cd "/Users/santiagocordoba/GITHUBS/openshorts-main 2/dashboard"
npm run dev -- --host 0.0.0.0 --port 5173

```

4. Abre la app en:

```text
http://localhost:5173
```

---

## ¿Debo cambiar la URL de ngrok cada vez?

**Sí.**

Cada vez que Colab se reinicia o recrea el túnel, la URL `https://...ngrok-free.app` puede cambiar.

Cuando cambie:

1. Ejecuta de nuevo:

```bash
cd "/Users/santiagocordoba/GITHUBS/openshorts-main 2"
./set-colab-api.sh https://NUEVA-URL.ngrok-free.app
```

2. Reinicia Vite (Ctrl+C y volver a correr):

```bash
cd dashboard
npm run dev -- --host 0.0.0.0 --port 5173
```

---

## No usar en este modo

- `./start-ngrok.sh` (no hace falta si el backend corre en Colab).
- Backend local (`./start.sh`) para procesamiento pesado.

---

## Troubleshooting rápido

- Error `NetworkError when attempting to fetch resource`:
  - URL de Colab/ngrok caída o vieja.
  - Reconfigura con `./set-colab-api.sh <nueva_url>` y reinicia Vite.

- Error `ECONNREFUSED` en consola de Vite:
  - Frontend está intentando `localhost:8000`.
  - Vuelve a correr `./set-colab-api.sh <url_colab>` y reinicia Vite.
