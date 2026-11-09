#!/usr/bin/env bash
# ==============================================================================
# Flash SURDAS Safety & Sensor Firmware to Arduino UNO Q STM32U585 MCU
# ==============================================================================
set -euo pipefail

SKETCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../sketch" && pwd)"
PORT="${1:-/dev/ttyRPMSG0}"

echo "=========================================================="
echo "  ⚡ Compiling & Uploading STM32 Firmware"
echo "  Target Port: $PORT"
echo "=========================================================="

if ! command -v arduino-cli &>/dev/null; then
  echo "❌ Error: arduino-cli is not installed. Run sudo ./scripts/install_uno_q.sh first."
  exit 1
fi

echo "📦 Compiling sketch..."
arduino-cli compile \
  --config-file "$SKETCH_DIR/sketch.yaml" \
  "$SKETCH_DIR/uno_q_firmware"

if [ -e "$PORT" ]; then
  echo "🚀 Uploading firmware to $PORT..."
  arduino-cli upload \
    -p "$PORT" \
    --config-file "$SKETCH_DIR/sketch.yaml" \
    "$SKETCH_DIR/uno_q_firmware"
  echo "✅ Firmware successfully flashed to STM32 MCU!"
else
  echo "⚠️ Target port $PORT not found."
  echo "Available ports:"
  arduino-cli board list || true
  echo "To flash manually: arduino-cli upload -p <port> --config-file $SKETCH_DIR/sketch.yaml $SKETCH_DIR/uno_q_firmware"
fi
