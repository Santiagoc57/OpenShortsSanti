#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
BACKEND_HOST="${BACKEND_HOST:-0.0.0.0}"
FRONTEND_HOST="${FRONTEND_HOST:-0.0.0.0}"

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
  echo
  echo "Deteniendo servicios..."
  if [[ -n "${BACKEND_PID}" ]] && kill -0 "${BACKEND_PID}" 2>/dev/null; then
    kill "${BACKEND_PID}" 2>/dev/null || true
  fi
  if [[ -n "${FRONTEND_PID}" ]] && kill -0 "${FRONTEND_PID}" 2>/dev/null; then
    kill "${FRONTEND_PID}" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

if [[ ! -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  echo "No se encontró ${ROOT_DIR}/.venv/bin/python"
  echo "Crea el virtualenv e instala dependencias primero."
  exit 1
fi

if [[ ! -d "${ROOT_DIR}/dashboard/node_modules" ]]; then
  echo "Instalando dependencias del frontend..."
  (
    cd "${ROOT_DIR}/dashboard"
    npm install
  )
fi

echo "Iniciando backend en http://localhost:${BACKEND_PORT} ..."
(
  cd "${ROOT_DIR}"
  exec .venv/bin/python -m uvicorn app:app --host "${BACKEND_HOST}" --port "${BACKEND_PORT}"
) &
BACKEND_PID=$!

echo "Iniciando frontend en http://localhost:${FRONTEND_PORT} ..."
(
  cd "${ROOT_DIR}/dashboard"
  exec npm run dev -- --host "${FRONTEND_HOST}" --port "${FRONTEND_PORT}"
) &
FRONTEND_PID=$!

echo "Servicios activos. Presiona Ctrl+C para detener ambos."

while true; do
  if ! kill -0 "${BACKEND_PID}" 2>/dev/null; then
    echo "Backend finalizó."
    exit 1
  fi
  if ! kill -0 "${FRONTEND_PID}" 2>/dev/null; then
    echo "Frontend finalizó."
    exit 1
  fi
  sleep 2
done
