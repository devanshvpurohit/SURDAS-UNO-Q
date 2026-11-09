#!/usr/bin/env bash
# ==============================================================================
# SURDAS Assistive Vision & Navigation System
# Target Hardware: Arduino UNO Q (Qualcomm QRB2210 Linux MPU + STM32U585 MCU)
# Automated Provisioning & Installation Script
# ==============================================================================
set -euo pipefail

echo "=========================================================="
echo "  🚀 Starting SURDAS Installation for Arduino UNO Q"
echo "=========================================================="

# 1. Check Root Privileges for System Package Installation
if [ "$EUID" -ne 0 ]; then
  echo "⚠️  Please run with sudo or as root: sudo ./scripts/install_uno_q.sh"
  exit 1
fi

INSTALL_DIR="${SURDAS_INSTALL_DIR:-/opt/surdas}"
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "📂 Source directory: $CURRENT_DIR"
echo "📂 Target install directory: $INSTALL_DIR"

# 2. Update Debian Package Index & Install Base Tools
echo "📦 Updating system packages..."
apt-get update -y
apt-get install -y --no-install-recommends \
  build-essential \
  python3-dev \
  python3-venv \
  python3-pip \
  v4l-utils \
  libgl1 \
  libglib2.0-0 \
  libasound2-dev \
  portaudio19-dev \
  curl \
  git \
  rsync

# 3. Synchronize Files to Installation Directory
if [ "$CURRENT_DIR" != "$INSTALL_DIR" ]; then
  echo "🚚 Deploying codebase to $INSTALL_DIR..."
  mkdir -p "$INSTALL_DIR"
  rsync -a --exclude='.git' --exclude='node_modules' --exclude='.pytest_cache' "$CURRENT_DIR/" "$INSTALL_DIR/"
fi

cd "$INSTALL_DIR"

# 4. Configure Device Group Permissions
echo "🔐 Configuring hardware device permissions (dialout, video, audio)..."
TARGET_USER="${SUDO_USER:-root}"
usermod -a -G dialout,video,audio "$TARGET_USER" || true

# 5. Create Python Virtual Environment
echo "🐍 Setting up Python 3 virtual environment in $INSTALL_DIR/venv..."
if [ ! -d "$INSTALL_DIR/venv" ]; then
  python3 -m venv "$INSTALL_DIR/venv"
fi

"$INSTALL_DIR/venv/bin/pip" install --upgrade pip setuptools wheel
echo "📦 Installing Python dependencies from requirements.txt..."
"$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt"

# 6. Build Caregiver Dashboard (if Node.js is present)
if command -v npm &>/dev/null; then
  echo "⚛️ Building Caregiver Dashboard..."
  (cd "$INSTALL_DIR/caregiver_dashboard" && npm ci && npm run build)
elif [ -d "$INSTALL_DIR/caregiver_dashboard/dist" ]; then
  echo "✅ Pre-built Caregiver Dashboard found at caregiver_dashboard/dist"
else
  echo "⚠️ Node.js / npm not detected. Installing NodeJS for dashboard build..."
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y nodejs
  (cd "$INSTALL_DIR/caregiver_dashboard" && npm install && npm run build)
fi

# 7. Install & Configure Arduino CLI for STM32 MCU
if ! command -v arduino-cli &>/dev/null; then
  echo "🔧 Installing Arduino CLI..."
  curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | BINDIR=/usr/local/bin sh
fi

echo "⚙️ Initializing Arduino CLI configuration..."
arduino-cli config init --overwrite || true
arduino-cli core update-index || true

echo "=========================================================="
echo "  ✅ SURDAS Installation Complete on Arduino UNO Q!"
echo ""
echo "Next steps:"
echo "  1. Flash STM32 MCU:   sudo ./scripts/flash_firmware.sh"
echo "  2. Deploy service:    sudo ./scripts/deploy_service.sh"
echo "  3. Run manually:      $INSTALL_DIR/venv/bin/python3 surdas_brain.py --headless"
echo "  4. Dashboard URL:     http://<arduino-uno-q-ip>:8000"
echo "=========================================================="
