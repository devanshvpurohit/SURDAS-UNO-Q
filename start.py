#!/usr/bin/env python3
"""
start.py — Single-command startup entrypoint for SURDAS on Arduino UNO Q.

Usage:
    python3 start.py                 # Auto-detects hardware, starts system & dashboard
    python3 start.py --sim           # Force full simulation mode (mock camera & sensors)
    python3 start.py --headless      # Run in headless mode without local preview window
    python3 start.py --gui           # Run with desktop OpenCV GUI window
"""

import sys
import os
import socket
import argparse
import subprocess
import signal

def get_local_ip() -> str:
    """Detect LAN IP address of this device."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        # Does not actually connect or send packets
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def print_banner(ip: str, port: int = 8000, headless: bool = True):
    print("=" * 64)
    print("      🚀 SURDAS UNO Q — Assistive Vision & Navigation")
    print("=" * 64)
    print(f"  • Platform:        Arduino UNO Q (Debian Linux + STM32U585)")
    print(f"  • Caregiver UI:    http://{ip}:{port}/")
    print(f"  • Localhost UI:    http://localhost:{port}/")
    print(f"  • Video Stream:    http://{ip}:{port}/video_feed")
    print(f"  • Telemetry WS:    ws://{ip}:{port}/ws")
    print(f"  • Mode:            {'Headless (Embedded)' if headless else 'Desktop GUI'}")
    print("=" * 64)
    print("  Press Ctrl+C to stop the system.")
    print("=" * 64)
    print()


def main():
    parser = argparse.ArgumentParser(description="Start SURDAS UNO Q System")
    parser.add_argument("--sim", action="store_true", help="Force simulation mode (mock camera + mock sensors)")
    parser.add_argument("--gui", action="store_true", help="Open local desktop GUI window (disables headless)")
    parser.add_argument("--headless", action="store_true", default=None, help="Run headless (default on embedded)")
    parser.add_argument("--camera", type=str, default=None, help="Explicit camera device path or index")
    parser.add_argument("--no-voice", action="store_true", help="Disable microphone voice input")

    args, unknown = parser.parse_known_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)

    # Determine execution arguments
    cmd = [sys.executable, os.path.join(project_root, "surdas_brain.py")]

    is_headless = True
    if args.gui:
        is_headless = False
        cmd.append("--gui")
    else:
        cmd.append("--headless")

    if args.sim:
        cmd.extend(["--mock-camera", "--mock-sensors"])

    if args.camera:
        cmd.extend(["--camera", args.camera])

    if args.no_voice:
        cmd.append("--no-voice")

    if unknown:
        cmd.extend(unknown)

    local_ip = get_local_ip()
    print_banner(local_ip, port=8000, headless=is_headless)

    # Launch surdas_brain
    try:
        proc = subprocess.Popen(cmd)

        def handle_sig(sig, frame):
            print("\n[START] Shutting down SURDAS gracefully...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            sys.exit(0)

        signal.signal(signal.SIGINT, handle_sig)
        signal.signal(signal.SIGTERM, handle_sig)

        proc.wait()
    except KeyboardInterrupt:
        print("\n[START] Stopping SURDAS...")
        proc.terminate()
        proc.wait()


if __name__ == "__main__":
    main()
