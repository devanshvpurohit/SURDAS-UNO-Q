"""
test_wakeword.py — Wake-word model loading

Guards the regression where OFFLINE_MODE=1 (set by surdas_brain.py) made
voice/wakeword.py skip openWakeWord init, leaving the trained surdas.onnx
model permanently unloaded and the device on transcription-only matching.
"""
import importlib
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL = Path(__file__).parent.parent / "models" / "surdas.onnx"


def test_wakeword_model_loads_under_offline_mode(monkeypatch):
    """OFFLINE_MODE must not disable the local model — it needs no network."""
    monkeypatch.setenv("OFFLINE_MODE", "1")
    import voice.wakeword as ww

    importlib.reload(ww)
    detector = ww.WakeWordDetector()

    assert detector.oww_model is not None, (
        "openWakeWord model not loaded — OFFLINE_MODE regression?"
    )
    assert detector._surdas_model_key == "surdas"


@pytest.mark.skipif(not MODEL.exists(), reason="models/surdas.onnx not present")
def test_wakeword_predicts_on_silence():
    """A loaded model must run without raising and report a low score."""
    import numpy as np

    import voice.wakeword as ww

    detector = ww.WakeWordDetector()
    assert detector.oww_model is not None
    assert detector.process_audio(np.zeros(16000, dtype=np.int16)) is False


def test_wakeword_threshold_is_configurable(monkeypatch):
    """Threshold must be tunable without editing source on the device."""
    monkeypatch.setenv("WAKE_THRESHOLD", "0.42")
    import voice.wakeword as ww

    importlib.reload(ww)
    assert ww.WakeWordDetector().threshold == pytest.approx(0.42)


def test_wakeword_text_fallback_still_works():
    """With no model, transcription matching must still extract the command."""
    import voice.wakeword as ww

    detector = ww.WakeWordDetector.__new__(ww.WakeWordDetector)
    detector.oww_model = None
    has_wake, command = detector.check_transcription("Surdas, what do you see?")
    assert has_wake is True
    # trailing punctuation is stripped by check_transcription; irrelevant to
    # routing, which substring-matches phrases
    assert command == "what do you see"
