import numpy as np
try:
    import torch
except ImportError:
    torch = None
import os

_OFFLINE_MODE = os.environ.get("OFFLINE_MODE", "0") == "1"


class VoiceActivityDetector:
    """
    Voice Activity Detector using Silero VAD with fallback to RMS energy detection.

    Offline-first: if the Silero cache directory is not present on disk,
    we skip the remote torch.hub.load() call entirely and use the
    energy-based RMS fallback.  The remote fetch path lives ONLY in
    setup_offline_models.py — never here.
    """

    # torch.hub typically expands the repo to one of these names.
    _SILERO_CACHE_CANDIDATES = [
        "snakers4_silero-vad_master",
        "snakers4_silero-vad_main",
        "snakers4_silero-vad_latest",
    ]

    def __init__(self, sample_rate: int = 16000, threshold: float = 0.25):
        self.sample_rate  = sample_rate
        self.threshold    = threshold
        self.silero_model = None
        self._init_silero()

    def _find_local_cache(self) -> str | None:
        """Return the path to the cached Silero repo dir, or None."""
        hub_base = os.path.expanduser("~/.cache/torch/hub")
        for candidate in self._SILERO_CACHE_CANDIDATES:
            path = os.path.join(hub_base, candidate)
            if os.path.isdir(path):
                return path
        return None

    def _init_silero(self):
        if torch is None:
            return
        if _OFFLINE_MODE:
            # Strict offline — only attempt if cache is confirmed present.
            cached_dir = self._find_local_cache()
            if cached_dir is None:
                print(
                    "[VAD] OFFLINE_MODE: Silero cache not found. "
                    "Using energy-based VAD. Run setup_offline_models.py while online."
                )
                return
        else:
            cached_dir = self._find_local_cache()

        if cached_dir is not None:
            try:
                model, _ = torch.hub.load(
                    repo_or_dir=cached_dir,
                    model="silero_vad",
                    source="local",
                    trust_repo=True,
                )
                self.silero_model = model
                print("[VAD] ✅ Silero VAD initialized from local cache (offline).")
                return
            except Exception as e:
                print(f"[VAD] Local Silero load failed ({e}). Using energy-based VAD.")
                return

        # No local cache and not in strict OFFLINE_MODE.
        # Do NOT attempt remote fetch here — that belongs in setup_offline_models.py.
        print(
            "[VAD] Silero cache not found. Using energy-based VAD. "
            "Run setup_offline_models.py while online to enable neural VAD."
        )

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        Determines if an audio chunk (16kHz float32 or int16) contains speech.
        """
        if len(audio_chunk) == 0:
            return False

        # Convert to float32 normalized [-1.0, 1.0] if int16
        if audio_chunk.dtype == np.int16:
            audio_float = audio_chunk.astype(np.float32) / 32768.0
        else:
            audio_float = audio_chunk.astype(np.float32)

        if self.silero_model is not None:
            try:
                tensor = torch.from_numpy(audio_float)
                if tensor.ndim == 1:
                    tensor = tensor.unsqueeze(0)
                # Silero VAD requires input chunk length >= 512 samples at 16kHz (>= 31.25ms)
                if tensor.shape[1] < 512:
                    tensor = torch.nn.functional.pad(tensor, (0, 512 - tensor.shape[1]))
                with torch.no_grad():
                    speech_prob = self.silero_model(tensor, self.sample_rate).item()
                return speech_prob > self.threshold
            except Exception:
                pass

        # Fallback: RMS Energy detection (lower threshold for normal speech)
        rms = np.sqrt(np.mean(audio_float ** 2))
        return rms > 0.008
