#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DASHBOARD_ENV="${ROOT_DIR}/dashboard/.env.local"

if [[ $# -lt 1 ]]; then
  echo "Uso:"
  echo "  ./set-colab-api.sh https://<tu-ngrok-colab>.ngrok-free.app"
  exit 1
fi

API_URL="$1"
if [[ ! "${API_URL}" =~ ^https:// ]]; then
  echo "Error: la URL debe empezar con https://"
  exit 1
fi

cat > "${DASHBOARD_ENV}" <<EOF
VITE_API_URL=${API_URL}
VITE_PROXY_TARGET=http://localhost:8000
EOF

echo "OK: actualizado ${DASHBOARD_ENV}"
echo "VITE_API_URL=${API_URL}"
echo
echo "Reinicia Vite:"
echo "  cd dashboard && npm run dev -- --host 0.0.0.0 --port 5173"
