#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
BACKEND_HOST="${BACKEND_HOST:-0.0.0.0}"
FRONTEND_HOST="${FRONTEND_HOST:-0.0.0.0}"

START_FRONTEND="${START_FRONTEND:-1}"      # 1 = levantar Vite, 0 = solo backend
EXPOSE_FRONTEND="${EXPOSE_FRONTEND:-1}"    # 1 = crear túnel frontend, 0 = solo túnel backend

NGROK_BIN="${NGROK_BIN:-ngrok}"
NGROK_TIMEOUT_SECONDS="${NGROK_TIMEOUT_SECONDS:-25}"
NGROK_BACKEND_ARGS="${NGROK_BACKEND_ARGS:-}"
NGROK_FRONTEND_ARGS="${NGROK_FRONTEND_ARGS:-}"

BACKEND_PID=""
FRONTEND_PID=""
NGROK_BACKEND_PID=""
NGROK_FRONTEND_PID=""
NGROK_BACKEND_LOG=""
NGROK_FRONTEND_LOG=""
BACKEND_NGROK_URL=""
FRONTEND_NGROK_URL=""

cleanup() {
  echo
  echo "Deteniendo servicios..."
  if [[ -n "${NGROK_FRONTEND_PID}" ]] && kill -0 "${NGROK_FRONTEND_PID}" 2>/dev/null; then
    kill "${NGROK_FRONTEND_PID}" 2>/dev/null || true
  fi
  if [[ -n "${NGROK_BACKEND_PID}" ]] && kill -0 "${NGROK_BACKEND_PID}" 2>/dev/null; then
    kill "${NGROK_BACKEND_PID}" 2>/dev/null || true
  fi
  if [[ -n "${FRONTEND_PID}" ]] && kill -0 "${FRONTEND_PID}" 2>/dev/null; then
    kill "${FRONTEND_PID}" 2>/dev/null || true
  fi
  if [[ -n "${BACKEND_PID}" ]] && kill -0 "${BACKEND_PID}" 2>/dev/null; then
    kill "${BACKEND_PID}" 2>/dev/null || true
  fi
  [[ -n "${NGROK_BACKEND_LOG}" && -f "${NGROK_BACKEND_LOG}" ]] && rm -f "${NGROK_BACKEND_LOG}"
  [[ -n "${NGROK_FRONTEND_LOG}" && -f "${NGROK_FRONTEND_LOG}" ]] && rm -f "${NGROK_FRONTEND_LOG}"
}

trap cleanup EXIT INT TERM

if [[ ! -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  echo "No se encontró ${ROOT_DIR}/.venv/bin/python"
  echo "Crea el virtualenv e instala dependencias primero."
  exit 1
fi

if ! command -v "${NGROK_BIN}" >/dev/null 2>&1; then
  echo "No se encontró '${NGROK_BIN}' en PATH."
  echo "Instala ngrok y autentícalo primero:"
  echo "  brew install ngrok/ngrok/ngrok"
  echo "  ngrok config add-authtoken <TU_TOKEN>"
  exit 1
fi

if [[ "${START_FRONTEND}" == "1" && ! -d "${ROOT_DIR}/dashboard/node_modules" ]]; then
  echo "Instalando dependencias del frontend..."
  (
    cd "${ROOT_DIR}/dashboard"
    npm install
  )
fi

wait_for_ngrok_url() {
  local log_file="$1"
  local timeout="$2"
  local url=""
  local pattern='https://[a-zA-Z0-9.-]+\.(ngrok-free\.app|ngrok\.app|ngrok\.io)'
  local has_rg="0"
  if command -v rg >/dev/null 2>&1; then
    has_rg="1"
  fi

  for _ in $(seq 1 "${timeout}"); do
    if [[ -f "${log_file}" ]]; then
      if [[ "${has_rg}" == "1" ]]; then
        url="$(rg -o "${pattern}" "${log_file}" -N | head -n1 || true)"
      else
        url="$(grep -Eo "${pattern}" "${log_file}" | head -n1 || true)"
      fi
      if [[ -n "${url}" ]]; then
        echo "${url}"
        return 0
      fi
    fi
    sleep 1
  done

  return 1
}

echo "Iniciando backend en http://localhost:${BACKEND_PORT} ..."
(
  cd "${ROOT_DIR}"
  exec .venv/bin/python -m uvicorn app:app --host "${BACKEND_HOST}" --port "${BACKEND_PORT}"
) &
BACKEND_PID=$!

if [[ "${START_FRONTEND}" == "1" ]]; then
  echo "Iniciando frontend en http://localhost:${FRONTEND_PORT} ..."
  (
    cd "${ROOT_DIR}/dashboard"
    exec npm run dev -- --host "${FRONTEND_HOST}" --port "${FRONTEND_PORT}"
  ) &
  FRONTEND_PID=$!
fi

echo "Abriendo túnel ngrok para backend..."
NGROK_BACKEND_LOG="$(mktemp)"
(
  cd "${ROOT_DIR}"
  if [[ -n "${NGROK_BACKEND_ARGS}" ]]; then
    # shellcheck disable=SC2086
    exec "${NGROK_BIN}" http "http://localhost:${BACKEND_PORT}" --log=stdout --log-format=logfmt ${NGROK_BACKEND_ARGS}
  else
    exec "${NGROK_BIN}" http "http://localhost:${BACKEND_PORT}" --log=stdout --log-format=logfmt
  fi
) >"${NGROK_BACKEND_LOG}" 2>&1 &
NGROK_BACKEND_PID=$!

if ! BACKEND_NGROK_URL="$(wait_for_ngrok_url "${NGROK_BACKEND_LOG}" "${NGROK_TIMEOUT_SECONDS}")"; then
  echo "No se pudo obtener la URL de túnel para backend dentro del timeout."
  echo "Logs ngrok backend:"
  tail -n 80 "${NGROK_BACKEND_LOG}" || true
  exit 1
fi

if [[ "${EXPOSE_FRONTEND}" == "1" && "${START_FRONTEND}" == "1" ]]; then
  echo "Abriendo túnel ngrok para frontend..."
  NGROK_FRONTEND_LOG="$(mktemp)"
  (
    cd "${ROOT_DIR}"
    if [[ -n "${NGROK_FRONTEND_ARGS}" ]]; then
      # shellcheck disable=SC2086
      exec "${NGROK_BIN}" http "http://localhost:${FRONTEND_PORT}" --log=stdout --log-format=logfmt ${NGROK_FRONTEND_ARGS}
    else
      exec "${NGROK_BIN}" http "http://localhost:${FRONTEND_PORT}" --log=stdout --log-format=logfmt
    fi
  ) >"${NGROK_FRONTEND_LOG}" 2>&1 &
  NGROK_FRONTEND_PID=$!

  if ! FRONTEND_NGROK_URL="$(wait_for_ngrok_url "${NGROK_FRONTEND_LOG}" "${NGROK_TIMEOUT_SECONDS}")"; then
    echo "No se pudo obtener la URL de túnel para frontend dentro del timeout."
    echo "Logs ngrok frontend:"
    tail -n 80 "${NGROK_FRONTEND_LOG}" || true
    exit 1
  fi
fi

echo
echo "Servicios activos:"
echo "  Backend local : http://localhost:${BACKEND_PORT}"
if [[ "${START_FRONTEND}" == "1" ]]; then
  echo "  Frontend local: http://localhost:${FRONTEND_PORT}"
fi
echo "  Backend ngrok : ${BACKEND_NGROK_URL}"
if [[ -n "${FRONTEND_NGROK_URL}" ]]; then
  echo "  Frontend ngrok: ${FRONTEND_NGROK_URL}"
fi
echo
echo "Para clientes remotos que llamen API directo, usa:"
echo "  ${BACKEND_NGROK_URL}"
echo
echo "Presiona Ctrl+C para detener backend, frontend y túneles."

while true; do
  if ! kill -0 "${BACKEND_PID}" 2>/dev/null; then
    echo "Backend finalizó."
    exit 1
  fi
  if [[ "${START_FRONTEND}" == "1" ]] && ! kill -0 "${FRONTEND_PID}" 2>/dev/null; then
    echo "Frontend finalizó."
    exit 1
  fi
  if ! kill -0 "${NGROK_BACKEND_PID}" 2>/dev/null; then
    echo "Túnel ngrok de backend finalizó."
    exit 1
  fi
  if [[ -n "${NGROK_FRONTEND_PID}" ]] && ! kill -0 "${NGROK_FRONTEND_PID}" 2>/dev/null; then
    echo "Túnel ngrok de frontend finalizó."
    exit 1
  fi
  sleep 2
done
