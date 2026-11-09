"""Guards the two startup/audio latency fixes in voice/tts.py.

1. A missing reference WAV must not trigger the ~8s `from TTS.api import TTS`.
2. A broken Coqui install is paid for once, not on every get_xtts() call.
3. macOS speech must reuse one `say` process, not spawn one per utterance.

Runs standalone (`python3 tests/test_tts_latency.py`) and under pytest.
Uses stdlib unittest.mock so it does not depend on a working pytest install.
"""
import os
import sys
import time
import types
import threading
from unittest.mock import patch

os.environ.setdefault("OFFLINE_MODE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import voice.tts as tts


def test_missing_reference_wav_skips_coqui_import():
    """No speaker reference -> never import Coqui (it costs seconds and fails)."""
    tripped = []

    def guard(name, *a, **k):
        if name == "TTS.api":
            tripped.append(name)
        return real_import(name, *a, **k)

    real_import = __builtins__ if isinstance(__builtins__, dict) else __builtins__.__import__

    with patch.object(tts, "_REFERENCE_WAV", tts.Path("/nonexistent/ref.wav")), \
         patch.object(tts, "_xtts_failed", False), \
         patch.object(tts, "_xtts_model", None), \
         patch("builtins.__import__", guard):
        assert tts.get_xtts() is None
    assert not tripped, f"Coqui imported despite missing reference wav: {tripped}"


def test_coqui_failure_is_sticky():
    """A broken Coqui install is paid for once, not on every get_xtts() call."""
    calls = []

    class FakeTTS:
        def __init__(self, *a, **k):
            calls.append(1)
            raise RuntimeError("broken install")

    fake_api = types.ModuleType("TTS.api")
    fake_api.TTS = FakeTTS
    fake_pkg = types.ModuleType("TTS")

    with patch.dict(sys.modules, {"TTS": fake_pkg, "TTS.api": fake_api}), \
         patch.object(tts, "_XTTS_ENABLED", True), \
         patch.object(tts, "_xtts_failed", False), \
         patch.object(tts, "_xtts_model", None), \
         patch.object(tts, "_REFERENCE_WAV", tts.Path(__file__)):  # exists
        assert tts.get_xtts() is None
        assert tts._xtts_failed is True
        assert len(calls) == 1, f"expected one init attempt, got {len(calls)}"

        # Second call must short-circuit without re-entering the slow import.
        assert tts.get_xtts() is None
        assert len(calls) == 1, f"re-entered broken init {len(calls)} times"


def test_macos_say_process_is_reused():
    """One long-lived `say -f -` process, not a spawn per utterance."""
    if sys.platform != "darwin":
        print("  (skipped, not macOS)")
        return

    eng = tts.VoiceEngine.__new__(tts.VoiceEngine)
    eng._proc = None
    eng._proc_lock = threading.Lock()
    eng._audio_stop_event = threading.Event()

    first = eng._ensure_say()
    second = eng._ensure_say()
    assert first is second, "say process was respawned; per-utterance spawn costs ~1.6s"

    started = time.perf_counter()
    eng._say_macos("path is clear", "en")
    elapsed = time.perf_counter() - started
    # 3 words at 200wpm ~= 0.9s + pad. Guards the completion sleep, not the audio.
    assert elapsed < 2.0, f"utterance blocked for {elapsed:.2f}s"

    first.kill()
    eng._proc = None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("all tts latency checks passed")
