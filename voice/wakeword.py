import os
import pathlib
import numpy as np

from voice.i18n_phrases import WAKE_WORDS, match as _i18n_match

# Path to our custom trained surdas.onnx wake word model (trained in this repo)
_SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SURDAS_ONNX = os.path.join(_SCRIPT_DIR, "models", "surdas.onnx")


class WakeWordDetector:
    """
    Wake word detector supporting openWakeWord with fallback to bilingual
    text-based keyword matching.

    Priority:
      1. models/surdas.onnx — our trained custom wake word model
      2. Caller-supplied model_path
      3. Built-in openWakeWord cache (alexa / hey_jarvis etc.)
      4. Text-based bilingual matching (Hey Surdas / सुरदास)

    Offline-first: no network access at runtime.
    """

    def __init__(self, model_path=None):
        self.oww_model = None
        self._surdas_model_key: str = "surdas"
        # ponytail: fixed threshold is tuned by ear on one mic. Ceiling: a noisy
        # street needs a per-device value. Upgrade: WAKE_THRESHOLD env var.
        self.threshold = float(os.environ.get("WAKE_THRESHOLD", 0.6))
        self._predict_failed = False
        self._init_openwakeword(model_path)

    def _init_openwakeword(self, model_path):
        # NOTE: openWakeWord loads a local .onnx and never touches the network, so
        # it stays enabled regardless of OFFLINE_MODE. A missing model falls
        # through to bilingual text matching below.
        try:
            from openwakeword.model import Model

            # Priority 1: Our custom trained surdas.onnx
            chosen_path = None
            if os.path.exists(_SURDAS_ONNX):
                chosen_path = _SURDAS_ONNX
                print(f"[WAKEWORD] Found custom surdas.onnx at {_SURDAS_ONNX}")

            # Priority 2: Caller-supplied path
            if chosen_path is None and model_path and os.path.exists(model_path):
                chosen_path = model_path

            if chosen_path:
                self.oww_model = Model(wakeword_models=[chosen_path])
                self._surdas_model_key = pathlib.Path(chosen_path).stem
                print(
                    f"[WAKEWORD] \u2705 Loaded custom openWakeWord model "
                    f"'{self._surdas_model_key}' from: {chosen_path}"
                )
                return

            # Priority 3: Built-in OWW cache
            self.oww_model = Model()
            self._surdas_model_key = "surdas"
            print("[WAKEWORD] \u2705 openWakeWord loaded from local cache.")

        except Exception as e:
            self.oww_model = None
            print(
                f"[WAKEWORD] openWakeWord not available ({e}). "
                "Using bilingual text-based wake-word matching (Hey Surdas / \u0938\u0941\u0930\u0926\u093e\u0938)."
            )

    def process_audio(self, audio_chunk: np.ndarray) -> bool:
        """
        Process raw 16kHz audio chunk through openWakeWord if available.
        Returns True if a wake word is detected.

        Uses WAKE_THRESHOLD (default 0.6) for our custom surdas.onnx to reduce
        false positives, and 0.5 for built-in OWW models.
        """
        if self.oww_model is not None:
            try:
                if audio_chunk.dtype == np.float32:
                    audio_int16 = (audio_chunk * 32767).astype(np.int16)
                else:
                    audio_int16 = audio_chunk

                prediction = self.oww_model.predict(audio_int16)
                for model_name, score in prediction.items():
                    # Stricter threshold for our custom model to reduce false positives
                    threshold = self.threshold if model_name == self._surdas_model_key else 0.5
                    if score > threshold:
                        print(f"[WAKEWORD] ✅ Wake word '{model_name}' triggered (score: {score:.3f} >= {threshold})")
                        return True
            except Exception as e:
                if not self._predict_failed:
                    self._predict_failed = True
                    print(f"[WAKEWORD] predict() failed, wake word degraded: {e}")
        return False

    def check_transcription(self, text: str) -> tuple[bool, str]:
        """
        Checks if transcribed text (English or Hindi) contains any wake word.
        Returns (is_wake_word_present, remaining_command).

        Supports two modes:
        1. Same-utterance: "Surdas, navigate to door" → command extracted
        2. Traditional wake: "Hey Surdas" → activates, waits for next utterance

        Works on both Latin and Devanagari scripts via i18n_phrases.match().
        """
        if not text or not text.strip():
            return False, ""

        text_stripped = text.strip()
        text_lower = text_stripped.lower()

        # Priority 1: Check for same-utterance wake words at the START of phrase
        # This enables: "Surdas, what do you see?" in a single utterance
        same_utterance_wakes = [
            "surdas", "सुरदास", "सूरदास", "soordas", "sura das"
        ]
        
        for wake in same_utterance_wakes:
            # Check if text starts with wake word (case-insensitive for English)
            if wake in ["surdas", "soordas", "sura das"]:
                if text_lower.startswith(wake):
                    # Extract command after wake word
                    remaining = text_stripped[len(wake):].strip(" ,.?!")
                    if remaining:
                        # Found command in same utterance
                        print(f"[WAKEWORD] Same-utterance wake: '{wake}' → command: '{remaining}'")
                        return True, remaining
            else:
                # Hindi wake word - case-sensitive
                if text_stripped.startswith(wake):
                    remaining = text_stripped[len(wake):].strip(" ,.?!।")
                    if remaining:
                        print(f"[WAKEWORD] Same-utterance wake: '{wake}' → command: '{remaining}'")
                        return True, remaining

        # Priority 2: Check for traditional wake phrases (with "Hey", etc.)
        if _i18n_match(WAKE_WORDS, text):
            # Strip the matched wake phrase from the rest of the text.
            remaining = text_stripped

            # Try to find and remove the matched English phrase
            for phrase in WAKE_WORDS["en"]:
                if phrase in text_lower:
                    idx = text_lower.find(phrase)
                    remaining = text_stripped[idx + len(phrase):].strip(" ,.?!")
                    
                    # If there's a command after "Hey Surdas", treat as same-utterance
                    if remaining:
                        print(f"[WAKEWORD] Traditional wake with command: '{phrase}' → '{remaining}'")
                        return True, remaining
                    else:
                        # Just wake word, no command (traditional activation)
                        print(f"[WAKEWORD] Traditional wake-only: '{phrase}'")
                        return True, ""

            # Try to find and remove the matched Hindi phrase
            for phrase in WAKE_WORDS["hi"]:
                if phrase in text:
                    idx = text.find(phrase)
                    remaining = text_stripped[idx + len(phrase):].strip(" ,.?!।")
                    
                    if remaining:
                        print(f"[WAKEWORD] Traditional wake with command: '{phrase}' → '{remaining}'")
                        return True, remaining
                    else:
                        print(f"[WAKEWORD] Traditional wake-only: '{phrase}'")
                        return True, ""

            # Match found but couldn't strip — return as wake-only
            return True, ""

        return False, text_stripped
