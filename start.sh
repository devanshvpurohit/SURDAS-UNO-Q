#!/usr/bin/env bash
# ==============================================================================
# SURDAS UNO Q — Single-Command System Launcher
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Locate Python Interpreter (Prioritize virtual environments)
PYTHON_BIN=""
if [ -x "$SCRIPT_DIR/venv/bin/python3" ]; then
  PYTHON_BIN="$SCRIPT_DIR/venv/bin/python3"
elif [ -x "/opt/surdas/venv/bin/python3" ]; then
  PYTHON_BIN="/opt/surdas/venv/bin/python3"
elif command -v python3 &>/dev/null; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "❌ Error: python3 is not installed."
  exit 1
fi

# 2. Check if Caregiver Dashboard has been compiled
if [ ! -d "$SCRIPT_DIR/caregiver_dashboard/dist" ]; then
  if command -v npm &>/dev/null; then
    echo "📦 Pre-building Caregiver Dashboard for port 8000 hosting..."
    (cd "$SCRIPT_DIR/caregiver_dashboard" && npm run build)
  fi
fi

# 3. Launch System Entrypoint
exec "$PYTHON_BIN" "$SCRIPT_DIR/start.py" "$@"
