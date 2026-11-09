"""
assistant.py — Bilingual Voice Assistant Loop with TWS Tuning & Conversational Follow-up

Features:
  • Universal Mic & TWS Earbud Support (AirPods, boAt, Galaxy Buds, Sony, etc.)
  • Automatic hardware sample-rate discovery & native resampling to 16kHz
  • Adaptive Software AGC (Automatic Gain Control) to boost quiet TWS microphones
  • Conversational Follow-up Mode: 20-second active window after any query
  • Acoustic Echo Cancellation: ignores speaker output while AI speaks
  • Clean audio buffer flushing between utterances
  • Bilingual: STT returns (text, lang); lang stored and passed to router/TTS
  • Language-switch commands ("speak in hindi" / "अंग्रेज़ी में बोलो")
"""
import numpy as np
import threading
import time
import queue
import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from surdas_brain import SurdasBrain

from voice.vad import VoiceActivityDetector
from voice.wakeword import WakeWordDetector
from voice.stt import SpeechToText
from voice.llm import LocalLLM
from voice.command_router import CommandRouter
from voice.audio_device import find_best_input_device, AudioStreamProcessor
import voice.i18n_phrases as _p
from voice.i18n_phrases import match as _match

_OFFLINE_MODE = os.environ.get("OFFLINE_MODE", "0") == "1"


class VoiceAssistant:
    """Continuous mic listener: VAD → STT (faster-whisper) → CommandRouter.
    
    CRITICAL FIX: Wake-word detection must work during TTS playback to prevent
    the system from becoming unresponsive when frequent safety announcements occur.
    """

    SILENCE_MS           = 600    # ms of quiet before utterance is considered done
    MIN_CLIP_SEC         = 0.35   # ignore clips shorter than this
    FOLLOW_UP_WINDOW_SEC = 20.0   # seconds to keep conversation open without wake word
    
    # Wake-word detection during TTS
    TTS_BARGE_IN_BUFFER_SEC = 1.5  # Buffer audio during TTS for wake-word detection
    TTS_BARGE_IN_CHUNK_LIMIT = 50  # Max chunks to buffer (50 * 30ms = 1.5s)

    def __init__(
        self,
        brain: "SurdasBrain",
        sample_rate: int = 16000,
        mic_device: Optional[str] = None,
        mic_gain: float = 1.0,
    ):
        self.brain       = brain
        self.sample_rate = sample_rate
        self.chunk_ms    = 30
        self.chunk_size  = int(sample_rate * self.chunk_ms / 1000)
        self.mic_device  = mic_device
        self.mic_gain    = mic_gain

        self.vad      = VoiceActivityDetector(sample_rate=sample_rate)
        self.wakeword = WakeWordDetector()
        self.stt      = SpeechToText()

        # Reuse brain's LLM instance — one Ollama connection shared
        self.llm    = getattr(brain, "llm", None) or LocalLLM()
        self.router = CommandRouter(brain, self.llm)

        self._running               = False
        self._audio_q               = queue.Queue()
        self._processing            = False
        self._is_recording         = False
        self._last_interaction_time = 0.0
        self._processor             = None
        
        # Wake-word detection during TTS: buffer recent audio for barge-in
        self._tts_audio_buffer      = []  # Circular buffer for wake-word detection during TTS
        self._tts_buffer_max        = self.TTS_BARGE_IN_CHUNK_LIMIT
        
        # Voice health monitoring
        self._wake_listener_active  = False
        self._last_audio_chunk_time = 0.0
        self._mic_error_count       = 0
        self._last_wake_detection   = 0.0
        self._warned_bargein_stt    = False

        # Bilingual language state
        self._last_detected_lang = "en"   # updated from STT each utterance
        self._forced_lang: Optional[str] = None  # set by explicit lang-switch command

        # Warn once if voice models aren't loaded (offline mode, missing cache)
        if _OFFLINE_MODE and self.stt.model is None:
            self.brain.voice.speak(
                "Voice models not fully cached, run setup script while online.",
                lang="en",
            )

    @property
    def is_recording(self) -> bool:
        """True while the user is actively speaking / audio is being buffered."""
        return self._is_recording

    @property
    def is_active(self) -> bool:
        """True while user is speaking, or assistant is transcribing / routing command."""
        return self._is_recording or self._processing
    
    @property
    def wake_listener_healthy(self) -> bool:
        """Returns True if wake-word listener is actively processing audio."""
        # Check if we've received audio in the last 2 seconds
        return self._wake_listener_active and (time.time() - self._last_audio_chunk_time < 2.0)
    
    def get_health_status(self) -> dict:
        """Returns diagnostic information about voice system health."""
        return {
            "wake_listener_active": self._wake_listener_active,
            "is_recording": self._is_recording,
            "is_processing": self._processing,
            "last_audio_chunk": time.time() - self._last_audio_chunk_time if self._last_audio_chunk_time > 0 else None,
            "last_wake_detection": time.time() - self._last_wake_detection if self._last_wake_detection > 0 else None,
            "mic_error_count": self._mic_error_count,
            "healthy": self.wake_listener_healthy
        }

    # ── Active language ──────────────────────────────────────────────────────


    @property
    def active_lang(self) -> str:
        """The language to use for the current response."""
        return self._forced_lang if self._forced_lang else self._last_detected_lang

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        self._wake_listener_active = True
        threading.Thread(target=self._mic_loop, daemon=True, name="MicLoop").start()
        print("[VOICE] 🎤 Mic listener active — say 'Hey Surdas' / 'सुरदास' or ask any question.")
        print("[VOICE] 🔊 Neural wake word active on every chunk (instant barge-in).")

    def stop(self):
        self._running = False
        self._wake_listener_active = False

    # ── Sounddevice callback (runs in audio thread) ──────────────────────────

    def _audio_callback(self, indata, frames, time_info, status):
        if self._running and self._processor is not None:
            processed = self._processor.process_chunk(indata)
            self._audio_q.put(processed)

    def _flush_audio_queue(self):
        """Discard any accumulated audio chunks (e.g. from during TTS playback)."""
        while not self._audio_q.empty():
            try:
                self._audio_q.get_nowait()
            except queue.Empty:
                break

    # ── Main mic loop ────────────────────────────────────────────────────────

    def _mic_loop(self):
        try:
            import sounddevice as sd
        except ImportError:
            print("[VOICE] sounddevice not installed — voice input disabled.")
            return

        dev_idx, dev_name, native_rate, is_tws = find_best_input_device(self.mic_device)

        effective_gain = self.mic_gain if self.mic_gain != 1.0 else (1.8 if is_tws else 1.5)
        tws_tag = " [🎧 TWS / Bluetooth Headset Mode]" if is_tws else ""
        print(f"[VOICE] 🎙️ Mic: {dev_name} (Index {dev_idx} | {native_rate}Hz | Gain {effective_gain:.1f}x){tws_tag}")

        self._processor = AudioStreamProcessor(
            input_rate=native_rate,
            target_rate=self.sample_rate,
            gain_boost=effective_gain,
        )
        native_chunk_size = int(native_rate * self.chunk_ms / 1000)

        try:
            stream = sd.InputStream(
                device=dev_idx,
                samplerate=native_rate,
                channels=1,
                dtype="float32",
                blocksize=native_chunk_size,
                callback=self._audio_callback,
            )
        except Exception as e:
            print(f"[VOICE] Falling back to default system mic due to ({e})...")
            try:
                self._processor = AudioStreamProcessor(
                    input_rate=self.sample_rate,
                    target_rate=self.sample_rate,
                    gain_boost=effective_gain,
                )
                stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="float32",
                    blocksize=self.chunk_size,
                    callback=self._audio_callback,
                )
            except Exception as e2:
                print(f"[VOICE] Could not open microphone ({e2}). Check macOS microphone permission.")
                return

        max_silence = int(self.SILENCE_MS / self.chunk_ms)

        with stream:
            recording      = False
            self._is_recording = False
            speech_buf     = []
            silence_chunks = 0
            was_speaking   = False

            while self._running:
                try:
                    chunk = self._audio_q.get(timeout=0.1)
                    self._last_audio_chunk_time = time.time()  # Health monitoring
                except queue.Empty:
                    continue
                except Exception as e:
                    self._mic_error_count += 1
                    print(f"[VOICE] Audio queue error: {e}")
                    if self._mic_error_count > 10:
                        print("[VOICE] ⚠️  Multiple audio errors detected. Attempting recovery...")
                        time.sleep(0.5)
                        self._mic_error_count = 0
                    continue

                # ── ENHANCED Barge-in / Echo-Suppression Logic ─────────────────────
                # CRITICAL FIX: During TTS playback we NOW buffer audio and perform
                # FULL wake-word detection (both neural + transcription) to allow
                # user to interrupt at any time, even during frequent safety announcements.
                
                is_ai_speaking = (
                    self.brain.voice.is_speaking
                    or (time.time() - self.brain.voice.last_speech_time < 0.35)
                )

                # Neural wake word, one call per chunk, idle or speaking. While idle
                # we only record the trigger: the command that follows rides the
                # normal record → transcribe path below, so nothing is dropped
                # mid-sentence.
                wake_hit = self.wakeword.process_audio(chunk)
                if wake_hit and not is_ai_speaking:
                    self._last_wake_detection = time.time()
                    print("\n[VOICE] 👂 Wake word heard.", flush=True)
                    continue
                
                # Get current TTS priority to determine echo suppression strategy
                tts_priority = self.brain.voice.current_priority
                is_critical_tts = (tts_priority is not None and tts_priority <= 2)  # CRITICAL_SAFETY or NAVIGATION

                if is_ai_speaking:
                    # STRATEGY 1: Neural wake-word detection (fast, low latency)
                    if wake_hit:
                        print("\n[VOICE] ⚡ Barge-in detected (neural)! Interrupting TTS...")
                        self._last_wake_detection = time.time()
                        self.brain.voice.interrupt_for_barge_in()
                        was_speaking = False
                        recording = False
                        self._is_recording = False
                        speech_buf = []
                        silence_chunks = 0
                        self._tts_audio_buffer = []  # Clear buffer
                        self._flush_audio_queue()
                        # Announce ready for command
                        self.brain.voice.speak("Yes?", priority=__import__("system_state").SpeechPriority.USER_COMMAND, force=True)
                        continue
                    
                    # STRATEGY 2: Buffer audio for transcription-based wake-word detection
                    # This enables "Hey Surdas" to work even if openWakeWord model isn't detecting
                    # Only buffer during non-critical TTS to preserve safety announcements
                    if not is_critical_tts:
                        # Add to circular buffer
                        self._tts_audio_buffer.append(chunk)
                        if len(self._tts_audio_buffer) > self._tts_buffer_max:
                            self._tts_audio_buffer.pop(0)  # Remove oldest
                        
                        # Check if we have enough audio to transcribe (~1.5 seconds)
                        if len(self._tts_audio_buffer) >= 40:  # 40 chunks * 30ms = 1.2s
                            # Check if recent audio contains speech (VAD)
                            recent_chunks = self._tts_audio_buffer[-10:]  # Last 300ms
                            has_recent_speech = any(self.vad.is_speech(c) for c in recent_chunks)
                            
                            if has_recent_speech:
                                # Perform quick transcription check for wake word
                                try:
                                    buffer_audio = np.concatenate(self._tts_audio_buffer)
                                    text, _ = self.stt.transcribe(buffer_audio, sample_rate=self.sample_rate)
                                    
                                    if text and text.strip():
                                        # Check for wake word in transcription
                                        has_wake, _ = self.wakeword.check_transcription(text)
                                        
                                        if has_wake:
                                            print(f"\n[VOICE] ⚡ Barge-in detected (transcription): \"{text}\"! Interrupting TTS...")
                                            self._last_wake_detection = time.time()
                                            self.brain.voice.interrupt_for_barge_in()
                                            was_speaking = False
                                            recording = False
                                            self._is_recording = False
                                            speech_buf = []
                                            silence_chunks = 0
                                            self._tts_audio_buffer = []
                                            self._flush_audio_queue()
                                            self.brain.voice.speak("Yes?", priority=__import__("system_state").SpeechPriority.USER_COMMAND, force=True)
                                            continue
                                except Exception as e:
                                    # Transcription failed - not critical, but never
                                    # silent: this path is the only wake word left
                                    # during priority TTS, so a fault here is total.
                                    if not self._warned_bargein_stt:
                                        self._warned_bargein_stt = True
                                        print(f"[VOICE] Barge-in transcription failed: {e}")
                    
                    # Normal echo suppression: discard this chunk from recording
                    # but wake-word detection above still happened
                    was_speaking = True
                    recording = False
                    self._is_recording = False
                    speech_buf = []
                    silence_chunks = 0
                    continue
                
                # Clear TTS buffer when not speaking
                if self._tts_audio_buffer:
                    self._tts_audio_buffer = []

                if was_speaking:
                    was_speaking = False
                    self._flush_audio_queue()
                    continue

                is_speech = self.vad.is_speech(chunk)

                if is_speech:
                    if not recording:
                        recording          = True
                        self._is_recording = True
                        speech_buf         = []
                        silence_chunks     = 0
                        print("\n[VOICE] 🔴 Listening…", end="", flush=True)
                    speech_buf.append(chunk)

                elif recording:
                    speech_buf.append(chunk)
                    silence_chunks += 1

                    if silence_chunks >= max_silence:
                        recording          = False
                        self._is_recording = False
                        clip               = np.concatenate(speech_buf)
                        speech_buf         = []
                        silence_chunks     = 0
                        print()   # newline after 🔴 Listening…

                        if len(clip) >= self.sample_rate * self.MIN_CLIP_SEC:
                            threading.Thread(
                                target=self._process_clip,
                                args=(clip,),
                                daemon=True,
                            ).start()

    # ── Utterance processing ─────────────────────────────────────────────────

    def _process_clip(self, audio: np.ndarray):
        self._processing = True
        try:
            t0 = time.time()
            print("[VOICE] ⏳ Transcribing…", end="", flush=True)
            text, lang = self.stt.transcribe(audio, sample_rate=self.sample_rate)
            dt = time.time() - t0
            print(f" ({dt*1000:.0f} ms) [lang={lang}]")

            if not text or not text.strip():
                return

            # Update detected language (only if not forced by user)
            if self._forced_lang is None:
                self._last_detected_lang = lang

            print(f"[VOICE] 💬 Heard: \"{text}\"")
            self._route(text, lang)

        except Exception as e:
            print(f"[VOICE] Processing error: {e}")
        finally:
            self._processing = False

    def _route(self, text: str, lang: str):
        """Check for language-switch, wake word, or active conversation window, then route."""

        # ── Language-switch commands (handled before wake word) ───────────────
        if _match(_p.LANG_SWITCH_HINDI, text):
            self._forced_lang = "hi"
            self.brain.voice.speak("ठीक है, अब हिंदी में बात करते हैं।", lang="hi", force=True)
            self._last_interaction_time = time.time()
            return

        if _match(_p.LANG_SWITCH_ENGLISH, text):
            self._forced_lang = "en"
            self.brain.voice.speak("Okay, switching to English.", lang="en", force=True)
            self._last_interaction_time = time.time()
            return

        effective_lang = self.active_lang

        # ── Wake word check ────────────────────────────────────────────────────
        has_wake, command = self.wakeword.check_transcription(text)

        if has_wake:
            self._last_wake_detection = time.time()  # Health monitoring
            print(f"[VOICE] ✨ Wake word matched! Command: \"{command}\" [lang={effective_lang}]")
            self._last_interaction_time = time.time()
            if command and len(command.strip()) > 1:
                self.router.route_command(command, lang=effective_lang)
            else:
                if effective_lang == "hi":
                    self.brain.voice.speak("हाँ?", lang="hi", force=True)
                else:
                    self.brain.voice.speak("Yes?", force=True)
            return

        # ── Direct-command fast path triggers (EN + HI via i18n) ──────────────
        lower = text.lower().strip()

        # Build a quick set of EN trigger substrings (kept for conversational window)
        direct_en_triggers = [
            "turn on light", "turn off light", "turn on torch", "turn off torch",
            "read text", "read this", "navigation mode", "what do you see",
            "stop", "describe", "quiet", "silence",
            "open ", "close ", "launch ", "quit ",
            "volume up", "volume down", "mute", "screenshot", "lock screen",
            "what time", "what's the time", "whats the time", "time now", "the time",
            "what date", "what's the date", "whats the date", "what day", "today's date",
            "list models", "available models", "switch to ", "use model",
            "current model", "which model",
            "what is", "what are", "how do", "how to", "how does",
            "tell me", "can you", "who is", "where is", "why is",
            "explain", "define", "help me",
        ]

        # Hindi direct triggers (any match = treat as a direct command)
        hi_direct = any(_match(d, text) for d in [
            _p.TORCH_ON, _p.TORCH_OFF, _p.NAV_MODE, _p.OCR_MODE,
            _p.STOP_SILENCE, _p.TIME_QUERY, _p.DATE_QUERY,
            _p.VOLUME_UP, _p.VOLUME_DOWN, _p.VOLUME_MUTE,
            _p.SCREENSHOT, _p.LOCK_SCREEN, _p.WHAT_DO_YOU_SEE,
            _p.MODEL_LIST, _p.MODEL_CURRENT, _p.APP_LIST,
        ])

        en_direct = any(k in lower for k in direct_en_triggers)
        in_conversation = (time.time() - self._last_interaction_time < self.FOLLOW_UP_WINDOW_SEC)

        if in_conversation or en_direct or hi_direct or lower in ("time", "time now", "date", "today"):
            self._last_interaction_time = time.time()
            if in_conversation and not en_direct and not hi_direct:
                print(f"[VOICE] 💬 Conversational follow-up: \"{text}\"")
            else:
                print(f"[VOICE] ⚡ Direct command: \"{text}\"")
            self.router.route_command(text, lang=effective_lang)
        else:
            print(f"[VOICE] (Ignored: \"{text}\" — say 'Hey Surdas' / 'सुरदास' to activate)")
