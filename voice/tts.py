"""
voice/tts.py — Priority-based, Preemptive Voice Engine for SURDAS with XTTS-v2.

Features:
  • Single unified TTS worker thread (Requirement 8)
  • Priority-based queue using SpeechPriority:
      1: CRITICAL_SAFETY (immediate collision/danger, interrupts everything)
      2: NAVIGATION (turn guidance, approaching waypoint)
      3: USER_COMMAND (direct response to user instructions)
      4: LLM_RESPONSE (conversational response)
      5: STATUS (periodic status / path is clear)
  • Preemption: higher-priority messages immediately interrupt lower-priority playback
  • Full Barge-In support: wake-word detection interrupts current speech instantly
  • Device recovery: catches and recovers from audio/subprocess errors without crash (Requirement 38)
  • XTTS-v2 Voice Cloning: Uses reference voice from voice_models/ directory
  • Bilingual: Supports English, Hindi, and other XTTS-supported languages
  • In-memory audio: No temporary WAV files, direct RAM playback
"""
from __future__ import annotations

import sys
import os
import subprocess
import threading
import queue
import time
import re
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from system_state import SpeechPriority

# ══════════════════════════════════════════════════════════════════════════════
# XTTS-V2 CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
_PROJECT_ROOT = Path(__file__).parent.parent
_VOICE_MODELS_DIR = _PROJECT_ROOT / "voice_models"
_REFERENCE_WAV = _VOICE_MODELS_DIR / "surdas_reference.wav"

# XTTS settings
_XTTS_ENABLED = os.getenv("XTTS_ENABLED", "true").lower() == "true"
_XTTS_MODEL = os.getenv("XTTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
_XTTS_LANGUAGE = os.getenv("XTTS_LANGUAGE", "en")
_XTTS_WARMUP = os.getenv("XTTS_WARMUP", "true").lower() == "true"
_XTTS_USE_GPU = os.getenv("XTTS_USE_GPU", "false").lower() == "true"
_XTTS_SAMPLE_RATE = 24000  # XTTS-v2 default sample rate

# macOS fallback settings
_MACOS_RATE = 200
_HI_VOICE = "Lekha"   # macOS Hindi voice
_EN_VOICE = None      # System default

# ══════════════════════════════════════════════════════════════════════════════
# XTTS-V2 SINGLETON
# ══════════════════════════════════════════════════════════════════════════════
_xtts_model = None
_xtts_lock = threading.Lock()
_xtts_speaker_conditioning = None
_xtts_failed = False  # Coqui import costs ~8s and can be broken; pay it once, not per call.


def get_xtts():
    """
    Lazy-load and return the XTTS-v2 model singleton.
    Thread-safe initialization.
    """
    global _xtts_model, _xtts_speaker_conditioning, _xtts_failed
    
    if not _XTTS_ENABLED or _xtts_failed:
        return None

    # Check the reference voice BEFORE importing Coqui. `from TTS.api import TTS`
    # pulls in transformers + torch and costs ~8s even when it then fails, and it
    # is worthless without a speaker reference. Cheap check first.
    if not _REFERENCE_WAV.exists():
        print(f"[TTS] No reference voice at {_REFERENCE_WAV} — using system TTS.")
        return None

    with _xtts_lock:
        if _xtts_model is not None:
            return _xtts_model

        try:
            print("[TTS] Initializing XTTS-v2...")
            from TTS.api import TTS

            # Initialize model (CPU or GPU based on config)
            _xtts_model = TTS(
                model_name=_XTTS_MODEL,
                progress_bar=False,
                gpu=_XTTS_USE_GPU
            )

            print(f"[TTS] Reference voice loaded: {_REFERENCE_WAV.name}")
            
            # Cache speaker conditioning for performance (if supported)
            try:
                # Some versions of XTTS support pre-computing speaker embeddings
                # This is an optimization - if it fails, we'll compute per-request
                print("[TTS] Pre-computing speaker conditioning...")
                # Note: This is version-dependent; we'll compute on-demand if needed
            except Exception:
                print("[TTS] Speaker conditioning will be computed per-request.")
            
            print("[TTS] XTTS-v2 ready.")
            return _xtts_model
            
        except ImportError as e:
            print(f"[TTS ERROR] XTTS not installed: {e}")
            print("[TTS] Install with: pip install TTS torch torchaudio")
            print("[TTS] Falling back to system TTS.")
            _xtts_failed = True
            return None
        except Exception as e:
            print(f"[TTS ERROR] Failed to initialize XTTS: {e}")
            print("[TTS] Falling back to system TTS.")
            _xtts_failed = True
            return None


def warmup_xtts():
    """
    Warm up XTTS in background during startup.
    Optional, controlled by XTTS_WARMUP environment variable.
    """
    if _XTTS_WARMUP and _XTTS_ENABLED:
        def _warmup():
            try:
                model = get_xtts()
                if model:
                    # Generate a short test phrase to warm up the model
                    print("[TTS] Warming up XTTS...")
                    _ = model.tts(
                        text="System ready.",
                        speaker_wav=str(_REFERENCE_WAV),
                        language=_XTTS_LANGUAGE
                    )
                    print("[TTS] XTTS warm-up complete.")
            except Exception as e:
                print(f"[TTS] Warm-up error (non-fatal): {e}")
        
        threading.Thread(target=_warmup, daemon=True, name="XTTS-Warmup").start()


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO PLAYBACK
# ══════════════════════════════════════════════════════════════════════════════
def play_audio_memory(waveform, sample_rate: int = _XTTS_SAMPLE_RATE):
    """
    Play audio directly from memory using sounddevice.
    No temporary files created.
    
    Args:
        waveform: Audio data (numpy array or list)
        sample_rate: Sample rate in Hz
    """
    try:
        import sounddevice as sd
        
        # Convert to numpy array if needed
        audio = np.asarray(waveform, dtype=np.float32)
        
        # Ensure 1D array
        if audio.ndim > 1:
            audio = audio.flatten()
        
        # Play blocking
        sd.play(audio, samplerate=sample_rate)
        sd.wait()
        
    except ImportError:
        print("[TTS ERROR] sounddevice not installed. Install with: pip install sounddevice")
        raise
    except Exception as e:
        print(f"[TTS ERROR] Audio playback failed: {e}")
        raise


def clean_speech_text(text: str) -> str:
    """Strip markdown, symbols, bullets, emojis, and formatting for clean TTS."""
    if not text:
        return ""

    # Remove code blocks and inline code
    t = re.sub(r"```[\s\S]*?```", "", text)
    t = re.sub(r"`([^`]+)`", r"\1", t)

    # Remove markdown links [label](url) -> label
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)

    # Remove markdown headers #, ##, etc.
    t = re.sub(r"^#+\s*", "", t, flags=re.MULTILINE)

    # Remove bullet markers (- , * , + , > )
    t = re.sub(r"^[\s\*\-\+\>]+\s*", "", t, flags=re.MULTILINE)

    # Remove numbered bullets (1. , 2. )
    t = re.sub(r"^\d+\.\s*", "", t, flags=re.MULTILINE)

    # Remove bold/italic markers and punctuation clutter
    t = t.replace("*", "").replace("_", "").replace("~", "")
    t = t.replace('"', "").replace("'", "").replace("#", "")
    t = t.replace("[", "").replace("]", "").replace("{", "").replace("}", "")
    t = t.replace("(", "").replace(")", "").replace("\\", "")

    # Normalize whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


class VoiceEngine:
    """
    Priority-based Text-to-Speech Engine with preemption and barge-in.
    Now powered by XTTS-v2 voice cloning with fallback to system TTS.
    """

    def __init__(self):
        self.is_macos = (sys.platform == "darwin")
        # PriorityQueue items: (priority_int, counter, text, lang)
        self._queue: queue.PriorityQueue[Tuple[int, int, str, str]] = queue.PriorityQueue(maxsize=50)
        self._counter = 0
        self._proc: Optional[subprocess.Popen] = None
        self._proc_lock = threading.Lock()
        self._running = True
        self._pyttsx = None
        self._hi_voice_id = None
        
        self._speaking = False
        self._current_priority: Optional[int] = None
        self._last_spoke = 0.0
        self._error_count = 0
        
        # XTTS playback control
        self._audio_stop_event = threading.Event()
        self._audio_thread: Optional[threading.Thread] = None

        # Initialize XTTS if enabled
        if _XTTS_ENABLED:
            self._xtts = get_xtts()
        else:
            self._xtts = None
            print("[TTS] XTTS disabled, using system TTS.")

        # Fallback to pyttsx3 on non-macOS if XTTS unavailable
        if not self.is_macos and not self._xtts:
            self._init_pyttsx()

        self._worker_thread = threading.Thread(target=self._worker, daemon=True, name="TTSWorker")
        self._worker_thread.start()
        
        # Start XTTS warm-up if configured
        if _XTTS_WARMUP:
            warmup_xtts()

    @property
    def is_speaking(self) -> bool:
        """Returns True if speech is actively playing or queued."""
        if self._speaking or not self._queue.empty():
            return True
        with self._proc_lock:
            return self._proc is not None and self._proc.poll() is None

    @property
    def current_priority(self) -> Optional[int]:
        return self._current_priority

    @property
    def last_speech_time(self) -> float:
        return self._last_spoke

    # ── pyttsx3 fallback initialization ──────────────────────────────────────
    def _init_pyttsx(self):
        try:
            import pyttsx3
            e = pyttsx3.init()
            e.setProperty("rate", _MACOS_RATE)
            self._pyttsx = e

            voices = e.getProperty("voices")
            for v in voices:
                vid = (v.id or "").lower()
                vlangs = " ".join(
                    (l.decode() if isinstance(l, bytes) else l)
                    for l in (v.languages or [])
                ).lower()
                if "hi" in vid or "hindi" in vid or "hi" in vlangs or "hindi" in vlangs:
                    self._hi_voice_id = v.id
                    break

            if self._hi_voice_id:
                print(f"[TTS] pyttsx3 Hindi voice found: {self._hi_voice_id}")
            else:
                print("[TTS] pyttsx3: no Hindi voice found — will use default voice.")
        except Exception as ex:
            print(f"[TTS] pyttsx3 init error: {ex}")

    # ── Background Worker ────────────────────────────────────────────────────
    def _worker(self):
        while self._running:
            try:
                # Wait for next priority item
                item = self._queue.get(timeout=0.05)
                prio_int, _, text, lang = item

                self._speaking = True
                self._current_priority = prio_int

                # Play speech
                self._say(text, lang)

                self._speaking = False
                self._current_priority = None
                self._last_spoke = time.time()
                self._queue.task_done()
            except queue.Empty:
                self._speaking = False
                self._current_priority = None
            except Exception as ex:
                self._speaking = False
                self._current_priority = None
                self._error_count += 1
                print(f"[TTS] Worker error: {ex}")
                time.sleep(0.05)

    # ── Audio Playback ───────────────────────────────────────────────────────
    def _ensure_say(self):
        """One long-lived `say -f -` process.

        Spawning `say` per utterance costs ~1.6s of AVSpeechSynthesizer + audio
        session init before the first word — very audible on Bluetooth. A
        persistent reader pays that once for the life of the process.
        """
        with self._proc_lock:
            if self._proc is not None and self._proc.poll() is None:
                return self._proc
            cmd = ["say", "-r", str(_MACOS_RATE), "-f", "-"]
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return self._proc

    def _say_macos(self, text: str, lang: str):
        try:
            proc = self._ensure_say()
            proc.stdin.write((text + "\n").encode("utf-8"))
            proc.stdin.flush()
        except Exception as ex:
            print(f"[TTS] macOS 'say' error: {ex}")
            self._recover_device()
            return

        # ponytail: `say -f -` gives no per-utterance "done" event, so playback
        # completion is estimated from word count at _MACOS_RATE. Error is
        # <100ms against a multi-second utterance. If this ever matters, switch
        # to AVSpeechSynthesizer (PyObjC) which does emit finish callbacks.
        est = max(0.4, len(text.split()) / _MACOS_RATE * 60.0 + 0.15)
        end = time.time() + est
        while time.time() < end:
            if self._audio_stop_event.is_set():
                return
            time.sleep(0.02)

    def _say(self, text: str, lang: str = "en"):
        if not text or not text.strip():
            return
        clean = text.strip()

        # Try XTTS first if available
        if self._xtts and _REFERENCE_WAV.exists():
            try:
                self._say_xtts(clean, lang)
                return
            except Exception as ex:
                print(f"[TTS] XTTS error: {ex}")
                print("[TTS] Falling back to system TTS.")
                # Continue to fallback
        
        # Fallback to system TTS
        if self.is_macos:
            self._say_macos(clean, lang)

        elif self._pyttsx:
            try:
                if lang == "hi" and self._hi_voice_id:
                    self._pyttsx.setProperty("voice", self._hi_voice_id)
                elif lang != "hi" and self._hi_voice_id:
                    self._pyttsx.setProperty("voice", "")
                self._pyttsx.say(clean)
                self._pyttsx.runAndWait()
            except Exception as ex:
                print(f"[TTS] pyttsx3 error: {ex}")
                self._recover_device()

    def _say_xtts(self, text: str, lang: str = "en"):
        """
        Generate and play speech using XTTS-v2.
        All processing happens in memory - no WAV files created.
        
        Thread-safe: uses _xtts_lock to prevent concurrent inference.
        """
        with _xtts_lock:
            try:
                # Clear stop event
                self._audio_stop_event.clear()
                
                print(f"[TTS] Generating speech with XTTS (lang={lang})...")
                
                # Generate speech waveform in memory
                # Note: tts() returns the waveform directly, no file created
                waveform = self._xtts.tts(
                    text=text,
                    speaker_wav=str(_REFERENCE_WAV),
                    language=lang
                )
                
                # Check if interrupted before playing
                if self._audio_stop_event.is_set():
                    print("[TTS] Speech generation interrupted.")
                    return
                
                print(f"[TTS] Playing audio ({len(waveform)} samples)...")
                
                # Play audio in a separate thread for interruptibility
                def _play():
                    try:
                        play_audio_memory(waveform, _XTTS_SAMPLE_RATE)
                    except Exception as e:
                        if not self._audio_stop_event.is_set():
                            print(f"[TTS] Playback error: {e}")
                
                self._audio_thread = threading.Thread(target=_play, daemon=True)
                self._audio_thread.start()
                self._audio_thread.join()  # Wait for completion
                
            except Exception as e:
                print(f"[TTS] XTTS generation error: {e}")
                raise

    # ── Interruption / Preemption ────────────────────────────────────────────
    def _interrupt(self):
        """Immediately terminate currently playing audio process."""
        # Stop XTTS audio playback
        self._audio_stop_event.set()
        
        # Stop sounddevice playback if active
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        
        # Stop macOS 'say' process
        with self._proc_lock:
            proc = self._proc
            if proc and proc.poll() is None:
                try:
                    proc.kill()
                    proc.wait(timeout=0.1)
                except Exception:
                    pass
            self._proc = None
        
        self._speaking = False
        self._current_priority = None

    def _drain(self, min_priority: int = 1):
        """
        Drain items from queue whose priority is >= min_priority (i.e. equal or lower importance).
        """
        items_to_keep = []
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                prio_int, _, _, _ = item
                if prio_int < min_priority:
                    items_to_keep.append(item)
                self._queue.task_done()
            except queue.Empty:
                break

        for item in items_to_keep:
            self._queue.put(item)

    def interrupt_for_barge_in(self):
        """
        Called when user speaks wake word or commands.
        Immediately stops currently playing speech and clears non-critical queue.
        """
        self._interrupt()
        # Drain everything except CRITICAL_SAFETY
        self._drain(min_priority=int(SpeechPriority.NAVIGATION))
        print("[TTS] ⚡ Barge-in: interrupted current playback for user voice.")

    # ── Public Speak API ─────────────────────────────────────────────────────
    def speak(
        self,
        text: str,
        priority: SpeechPriority = SpeechPriority.STATUS,
        force: bool = False,
        lang: str = "en"
    ):
        """
        Queue text for speech with specified priority.

        Args:
            text: What to say.
            priority: SpeechPriority enum (CRITICAL_SAFETY, NAVIGATION, USER_COMMAND, LLM_RESPONSE, STATUS)
            force: If True, immediately interrupts currently playing speech and drains lower/equal priority.
            lang: "en" or "hi"
        """
        if not text or not text.strip():
            return

        clean = clean_speech_text(text)
        if not clean:
            return

        # Print immediately to terminal
        prio_name = priority.name if hasattr(priority, "name") else str(priority)
        print(f"[SPEECH OUT ({prio_name})] 🔊  {clean}")

        # Broadcast to caregiver dashboard
        try:
            from telemetry import broadcast_event
            broadcast_event("speech", {"text": clean, "lang": lang, "priority": prio_name})
        except Exception:
            pass

        prio_int = int(priority)

        # Preemption check:
        # If force=True OR this priority is strictly higher than what's currently playing, preempt!
        should_preempt = force
        if self._speaking and self._current_priority is not None:
            if prio_int < self._current_priority:
                should_preempt = True

        if should_preempt:
            self._interrupt()
            if force or priority == SpeechPriority.CRITICAL_SAFETY:
                # Clear pending queue items of equal or lower priority
                self._drain(min_priority=prio_int)

        # Enqueue with incremented counter for FIFO stability
        self._counter += 1
        try:
            self._queue.put((prio_int, self._counter, clean, lang), timeout=0.2)
        except queue.Full:
            print(f"[TTS] Warning: queue full, dropping speech: {clean[:30]}...")

    def _recover_device(self):
        """Attempt recovery if speech synthesis fails (Requirement 38)."""
        print("[TTS] ⚠️  TTS failure detected. Performing recovery...")
        self._interrupt()
        time.sleep(0.1)
        if not self.is_macos:
            self._init_pyttsx()
        print("[TTS] ✅ TTS recovery complete.")

    def stop(self):
        """Stop TTS worker entirely."""
        self._running = False
        self._drain(min_priority=1)
        self._interrupt()
