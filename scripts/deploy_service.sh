#!/usr/bin/env bash
# ==============================================================================
# Deploy & Start SURDAS Systemd Service on Arduino UNO Q
# ==============================================================================
set -euo pipefail

if [ "$EUID" -ne 0 ]; then
  echo "⚠️  Please run with sudo: sudo ./scripts/deploy_service.sh"
  exit 1
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_SRC="$PROJECT_DIR/deploy/surdas.service"
SERVICE_DST="/etc/systemd/system/surdas.service"

echo "=========================================================="
echo "  🚀 Installing SURDAS Systemd Service"
echo "  Working Directory: $PROJECT_DIR"
echo "=========================================================="

if [ ! -f "$SERVICE_SRC" ]; then
  echo "❌ Error: $SERVICE_SRC not found."
  exit 1
fi

# Locate python interpreter
PYTHON_BIN="$PROJECT_DIR/venv/bin/python3"
if [ ! -x "$PYTHON_BIN" ]; then
  if command -v python3 &>/dev/null; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "❌ Error: python3 not found."
    exit 1
  fi
fi

# Generate customized systemd unit with the correct paths
sed -e "s|WorkingDirectory=.*|WorkingDirectory=$PROJECT_DIR|g" \
    -e "s|ExecStart=.*|ExecStart=$PYTHON_BIN start.py --headless|g" \
    "$SERVICE_SRC" > "$SERVICE_DST"

echo "📋 Installed systemd service unit to $SERVICE_DST"

echo "🔄 Reloading systemd daemon..."
systemctl daemon-reload

echo "⚡ Enabling surdas.service to start on boot..."
systemctl enable surdas.service

echo "🚀 Starting surdas.service..."
systemctl restart surdas.service

echo "=========================================================="
echo "  ✅ SURDAS Service Successfully Deployed!"
echo ""
echo "Useful commands:"
echo "  • Check service status:  sudo systemctl status surdas"
echo "  • View live logs:        sudo journalctl -u surdas -f"
echo "  • Stop service:          sudo systemctl stop surdas"
echo "  • Restart service:       sudo systemctl restart surdas"
echo "=========================================================="
