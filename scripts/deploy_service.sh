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
echo "=========================================================="

if [ ! -f "$SERVICE_SRC" ]; then
  echo "❌ Error: $SERVICE_SRC not found."
  exit 1
fi

echo "📋 Copying systemd service unit to $SERVICE_DST..."
cp "$SERVICE_SRC" "$SERVICE_DST"

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
