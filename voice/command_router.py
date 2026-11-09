"""
command_router.py — SURDAS Voice Command Dispatcher (bilingual EN + HI)

Priority order:
  1. App launch / close commands          → app_launcher (instant, no LLM)
  2. SURDAS hardware / mode commands      → deterministic handlers (instant)
  3. Vision description queries           → fast heuristic (instant)
  4. General knowledge / conversation     → LocalLLM via Ollama (streamed)

All phrase matching uses voice.i18n_phrases.match() so both English and
Hindi commands are handled without any explicit language switch.
"""
import re
import subprocess
from typing import TYPE_CHECKING, Optional

from voice.app_launcher import open_app, close_app, get_app_list
import voice.i18n_phrases as _p
from voice.i18n_phrases import match as _match, HINDI_FILLERS

if TYPE_CHECKING:
    from surdas_brain import SurdasBrain
    from voice.llm import LocalLLM


# ── helpers ──────────────────────────────────────────────────────────────────

def _extract_app_name(text: str, trigger: str) -> str:
    """Pull out the app name that follows a trigger word/phrase."""
    idx = text.find(trigger)
    if idx == -1:
        return ""
    after = text[idx + len(trigger):].strip()
    # Strip English and Hindi filler words
    for filler in HINDI_FILLERS:
        after = after.replace(filler, "").strip()
    return after.strip()


# ── main router ──────────────────────────────────────────────────────────────

class CommandRouter:
    """
    Routes transcribed voice text to the correct handler.
    lang is now threaded through from the STT result so TTS can reply
    in the same language.
    """

    def __init__(self, brain: "SurdasBrain", llm: Optional["LocalLLM"] = None):
        self.brain = brain
        self.llm = llm

    @property
    def indoor_navigator(self):
        return getattr(self.brain, "indoor_navigator", None)

    @property
    def spatial_memory(self):
        return getattr(self.brain, "spatial_memory", None)

    @property
    def navigator(self):
        return getattr(self.brain, "navigator", None)

    # ─────────────────────────────────────────────────────────────────────────
    def route_command(self, raw_text: str, lang: str = "en") -> bool:
        text = raw_text.strip().lower()
        if not text:
            return False

        print(f"[ROUTER] ▶ '{text}'  [lang={lang}]")

        # ── 1. INDOOR NAVIGATION & SPATIAL MEMORY ─────────────────────────────
        if self._handle_indoor_navigation_commands(text, raw_text, lang):
            return True

        # ── 2. OFFLINE NAVIGATION ─────────────────────────────────────────────
        if self._handle_navigation_commands(text, raw_text, lang):
            return True

        # ── 3. APP LAUNCHER ───────────────────────────────────────────────────
        if self._handle_app_commands(text, raw_text, lang):
            return True

        # ── 4. HARDWARE / MODE COMMANDS ───────────────────────────────────────
        if self._handle_hardware_commands(text, lang):
            return True

        # ── 5. FAST VISION DESCRIPTION ────────────────────────────────────────
        if self._handle_vision_query(text, lang):
            return True

        # ── 6. MODEL MANAGEMENT ───────────────────────────────────────────────
        if self._handle_model_commands(text, lang):
            return True

        # ── 7. GENERAL KNOWLEDGE / CONVERSATION (Ollama LLM) ─────────────────
        self._handle_llm_query(raw_text, lang)
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Category 0 — Indoor Navigation & Spatial Memory Commands
    # ─────────────────────────────────────────────────────────────────────────
    def _handle_indoor_navigation_commands(self, text: str, raw_text: str, lang: str) -> bool:
        if not self.indoor_navigator or not self.spatial_memory:
            return False

        # Pause indoor navigation
        if _match(_p.PAUSE_NAVIGATION, text):
            self.indoor_navigator.pause(lang)
            return True

        # Resume indoor navigation
        if _match(_p.RESUME_NAVIGATION, text):
            self.indoor_navigator.resume(lang)
            return True

        # Stop indoor navigation (also check NAV_STOP for compatibility)
        if _match(_p.NAV_STOP, text) and self.indoor_navigator.get_status().state != "IDLE":
            self.indoor_navigator.stop(lang)
            return True

        # Label current room
        if _match(_p.ROOM_LABEL, raw_text):
            room_name = None
            for trigger in _p.ROOM_LABEL["en"]:
                if trigger in text:
                    room_name = _extract_app_name(text, trigger)
                    break
            if not room_name:
                for trigger in _p.ROOM_LABEL["hi"]:
                    if trigger in raw_text:
                        room_name = _extract_app_name(raw_text, trigger)
                        break
            
            if room_name:
                # Save room at current estimated position (0, 0 if no tracking)
                room_id = self.spatial_memory.add_room(room_name, center_x=0.0, center_y=0.0, radius_m=5.0)
                if lang == "hi":
                    self.brain.voice.speak(f"ठीक है, मैंने इस कमरे को {room_name} के रूप में याद कर लिया है।", lang)
                else:
                    self.brain.voice.speak(f"Okay, I've remembered this room as {room_name}.", lang)
                
                # Broadcast telemetry
                try:
                    from telemetry import broadcast_event
                    broadcast_event("room_labeled", {"name": room_name, "room_id": room_id})
                except Exception:
                    pass
                return True

        # Label landmark
        if _match(_p.LANDMARK_LABEL, raw_text):
            landmark_name = None
            for trigger in _p.LANDMARK_LABEL["en"]:
                if trigger in text:
                    landmark_name = _extract_app_name(text, trigger)
                    break
            if not landmark_name:
                for trigger in _p.LANDMARK_LABEL["hi"]:
                    if trigger in raw_text:
                        landmark_name = _extract_app_name(raw_text, trigger)
                        break
            
            if landmark_name:
                landmark_id = self.spatial_memory.add_landmark(
                    landmark_name, x=0.0, y=0.0, z=0.0, description="User-labeled location"
                )
                if lang == "hi":
                    self.brain.voice.speak(f"ठीक है, मैंने इस स्थान को {landmark_name} के रूप में बचा लिया है।", lang)
                else:
                    self.brain.voice.speak(f"Okay, I've saved this location as {landmark_name}.", lang)
                
                try:
                    from telemetry import broadcast_event
                    broadcast_event("landmark_labeled", {"name": landmark_name, "landmark_id": landmark_id})
                except Exception:
                    pass
                return True

        # Navigate to object
        if _match(_p.INDOOR_NAV_TO_OBJECT, raw_text):
            object_name = None
            for trigger in _p.INDOOR_NAV_TO_OBJECT["en"]:
                if trigger in text:
                    object_name = _extract_app_name(text, trigger)
                    break
            if not object_name:
                for trigger in _p.INDOOR_NAV_TO_OBJECT["hi"]:
                    if trigger in raw_text:
                        object_name = _extract_app_name(raw_text, trigger)
                        break
            
            if object_name:
                success = self.indoor_navigator.navigate_to_object(object_name, lang)
                if success:
                    try:
                        from telemetry import broadcast_event
                        broadcast_event("indoor_nav_started", {"destination": object_name, "type": "object"})
                    except Exception:
                        pass
                return True

        # Navigate to landmark
        if _match(_p.INDOOR_NAV_TO_LANDMARK, raw_text) and "room" not in text and "the" not in text:
            landmark_name = None
            for trigger in _p.INDOOR_NAV_TO_LANDMARK["en"]:
                if trigger in text:
                    landmark_name = _extract_app_name(text, trigger)
                    break
            if not landmark_name:
                for trigger in _p.INDOOR_NAV_TO_LANDMARK["hi"]:
                    if trigger in raw_text:
                        landmark_name = _extract_app_name(raw_text, trigger)
                        break
            
            if landmark_name:
                success = self.indoor_navigator.navigate_to_landmark(landmark_name, lang)
                if success:
                    try:
                        from telemetry import broadcast_event
                        broadcast_event("indoor_nav_started", {"destination": landmark_name, "type": "landmark"})
                    except Exception:
                        pass
                return True

        # Where is object
        if _match(_p.WHERE_IS_OBJECT, raw_text):
            object_name = None
            for trigger in _p.WHERE_IS_OBJECT["en"]:
                if trigger in text:
                    object_name = _extract_app_name(text, trigger)
                    break
            if not object_name:
                for trigger in _p.WHERE_IS_OBJECT["hi"]:
                    if trigger in raw_text:
                        object_name = _extract_app_name(raw_text, trigger)
                        break
            
            if object_name:
                obj = self.spatial_memory.find_object_by_label(object_name)
                if obj:
                    # Rough distance calculation
                    dist = (obj.x ** 2 + obj.y ** 2) ** 0.5
                    if lang == "hi":
                        self.brain.voice.speak(f"{object_name} लगभग {int(dist)} मीटर दूर है। मैं आपको वहां ले जा सकता हूं।", lang)
                    else:
                        self.brain.voice.speak(f"The {object_name} is approximately {int(dist)} metres away. I can guide you there.", lang)
                else:
                    if lang == "hi":
                        self.brain.voice.speak(f"मुझे {object_name} याद नहीं है। कृपया इसे पहले दिखाएं।", lang)
                    else:
                        self.brain.voice.speak(f"I don't remember seeing a {object_name}. Please show it to me first.", lang)
                return True

        # What room is this
        if _match(_p.WHAT_ROOM, raw_text):
            room = self.spatial_memory.get_room_at_position(0.0, 0.0)
            if room:
                if lang == "hi":
                    self.brain.voice.speak(f"आप {room.name} में हैं।", lang)
                else:
                    self.brain.voice.speak(f"You are in the {room.name}.", lang)
            else:
                if lang == "hi":
                    self.brain.voice.speak("मुझे नहीं पता यह कौन सा कमरा है। आप इसे लेबल कर सकते हैं।", lang)
                else:
                    self.brain.voice.speak("I don't know which room this is. You can label it.", lang)
            return True

        # List rooms
        if _match(_p.LIST_ROOMS, raw_text):
            rooms = self.spatial_memory.list_rooms()
            if not rooms:
                if lang == "hi":
                    self.brain.voice.speak("मुझे कोई कमरा याद नहीं है। आप कमरों को लेबल कर सकते हैं।", lang)
                else:
                    self.brain.voice.speak("I don't remember any rooms yet. You can label rooms by saying 'this is the bedroom'.", lang)
            else:
                room_names = ", ".join(r.name for r in rooms)
                if lang == "hi":
                    self.brain.voice.speak(f"मुझे ये कमरे याद हैं: {room_names}।", lang)
                else:
                    self.brain.voice.speak(f"I remember these rooms: {room_names}.", lang)
            return True

        # Describe surroundings
        if _match(_p.DESCRIBE_SURROUNDINGS, raw_text):
            description = self.spatial_memory.describe_surroundings(0.0, 0.0, radius_m=3.0)
            self.brain.voice.speak(description, lang)
            return True

        # Clear memory (safety check)
        if _match(_p.CLEAR_MEMORY, raw_text):
            if lang == "hi":
                self.brain.voice.speak("क्या आप वाकई सब भूलना चाहते हैं? यह कार्रवाई वापस नहीं की जा सकती।", lang)
            else:
                self.brain.voice.speak("Are you sure you want me to forget everything? This cannot be undone. Say 'yes forget everything' to confirm.", lang)
            # Note: actual clear would need confirmation - left as exercise
            return True

        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Category 2 — Offline Navigation Commands  
    # ─────────────────────────────────────────────────────────────────────────
    def _handle_navigation_commands(self, text: str, raw_text: str, lang: str) -> bool:
        if not self.navigator:
            return False

        # Stop navigation
        if _match(_p.NAV_STOP, text):
            self.navigator.stop(lang)
            return True

        # Repeat instructions
        if _match(_p.NAV_REPEAT, text):
            self.navigator.repeat_instruction()
            return True

        # Where am I
        if _match(_p.NAV_WHERE_AM_I, text):
            status = self.navigator.get_status()
            loc = getattr(self.navigator, "_current_location", None)
            if not loc:
                from navigation.voice_guidance import NavigationVoiceGuide
                self.brain.voice.speak(NavigationVoiceGuide.gps_unavailable(lang), lang)
                return True
            results = self.navigator._manager.search_place("", loc, limit=1)
            if results:
                from navigation.voice_guidance import NavigationVoiceGuide
                msg = NavigationVoiceGuide.where_am_i(results[0]["name"], results[0]["distance_m"], loc.get("accuracy", 10.0), lang)
                self.brain.voice.speak(msg, lang)
            return True

        # Nearest <category>
        if _match(_p.NAV_NEAREST, raw_text):
            loc = getattr(self.navigator, "_current_location", None)
            from navigation.voice_guidance import NavigationVoiceGuide
            # strip filler
            q = text
            for trigger in _p.NAV_NEAREST["en"] + _p.NAV_NEAREST["hi"]:
                q = q.replace(trigger, "").strip()
            # Try to search
            results = self.navigator._manager.search_place(raw_text, loc, limit=1)
            if results:
                msg = NavigationVoiceGuide.nearest_result(results[0], results[0]["type"] or "place", lang)
                self.brain.voice.speak(msg, lang)
            else:
                self.brain.voice.speak(NavigationVoiceGuide.no_nearby("place", lang), lang)
            return True

        # Navigate to <destination>
        for trigger in _p.NAV_NAVIGATE_TO["en"]:
            if trigger in text:
                dest = _extract_app_name(text, trigger)  # re-use the app name extractor
                if dest:
                    self.navigator.start(dest, lang=lang)
                    return True
        for trigger in _p.NAV_NAVIGATE_TO["hi"]:
            if trigger in raw_text:
                dest = _extract_app_name(raw_text, trigger)
                if dest:
                    self.navigator.start(dest, lang=lang)
                    return True

        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Category 3 — App commands
    # ─────────────────────────────────────────────────────────────────────────
    def _handle_app_commands(self, text: str, raw_text: str, lang: str) -> bool:

        # OPEN triggers — check English triggers in lowercased text and
        # Hindi triggers (matched via i18n_phrases) in original text.
        open_en_triggers = [
            "open ", "launch ", "start ", "run ",
            "open up ", "open the ", "launch the ", "can you open ",
        ]
        open_hi_triggers = _p.APP_OPEN["hi"]

        for trigger in open_en_triggers:
            if trigger in text:
                app_name = _extract_app_name(text, trigger)
                if not app_name:
                    continue
                success, msg = open_app(app_name)
                self.brain.voice.speak(msg, lang=lang)
                return True

        for trigger in open_hi_triggers:
            if trigger in raw_text:
                app_name = _extract_app_name(raw_text, trigger)
                if not app_name:
                    continue
                success, msg = open_app(app_name)
                self.brain.voice.speak(msg, lang=lang)
                return True

        # CLOSE / QUIT triggers
        close_en_triggers = [
            "close ", "quit ", "exit ", "kill ",
            "close the ", "quit the ",
        ]
        close_hi_triggers = _p.APP_CLOSE["hi"]

        for trigger in close_en_triggers:
            if trigger in text:
                app_name = _extract_app_name(text, trigger)
                if not app_name:
                    continue
                success, msg = close_app(app_name)
                self.brain.voice.speak(msg, lang=lang)
                return True

        for trigger in close_hi_triggers:
            if trigger in raw_text:
                app_name = _extract_app_name(raw_text, trigger)
                if not app_name:
                    continue
                success, msg = close_app(app_name)
                self.brain.voice.speak(msg, lang=lang)
                return True

        # App list query
        if _match(_p.APP_LIST, raw_text):
            self.brain.voice.speak(get_app_list(), lang=lang)
            return True

        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Category 4 — Hardware / SURDAS mode commands
    # ─────────────────────────────────────────────────────────────────────────
    def _handle_hardware_commands(self, text: str, lang: str) -> bool:

        # Navigation mode
        if _match(_p.NAV_MODE, text):
            self.brain.mode = "NAV"
            if lang == "hi":
                self.brain.voice.speak("नेविगेशन मोड चालू।", lang="hi")
            else:
                self.brain.voice.speak("Navigation mode active.")
            return True

        # OCR / Read text mode
        if _match(_p.OCR_MODE, text):
            self.brain.mode = "OCR"
            return True

        # Flashlight ON
        if _match(_p.TORCH_ON, text):
            self.brain.toggle_esp32_led(True)
            if lang == "hi":
                self.brain.voice.speak("टॉर्च चालू।", lang="hi")
            else:
                self.brain.voice.speak("Flashlight on.")
            return True

        # Flashlight OFF
        if _match(_p.TORCH_OFF, text):
            self.brain.toggle_esp32_led(False)
            if lang == "hi":
                self.brain.voice.speak("टॉर्च बंद।", lang="hi")
            else:
                self.brain.voice.speak("Flashlight off.")
            return True

        # Stop / silence
        if _match(_p.STOP_SILENCE, text) or text in ("stop", "quiet", "silence", "be quiet", "shut up", "cancel"):
            self.brain.voice.speak("", force=True)
            return True

        # System status
        if _match(_p.STATUS, text):
            self.brain.voice.speak(self.brain.get_status_summary(), lang=lang)
            return True

        # Time
        if _match(_p.TIME_QUERY, text):
            import datetime
            now = datetime.datetime.now().strftime("%I:%M %p").lstrip("0")
            if lang == "hi":
                self.brain.voice.speak(f"अभी {now} बजे हैं।", lang="hi", force=True)
            else:
                self.brain.voice.speak(f"The time is {now}.", force=True)
            return True

        # Date
        if _match(_p.DATE_QUERY, text):
            import datetime
            today = datetime.datetime.now().strftime("%A, %B %d, %Y")
            if lang == "hi":
                self.brain.voice.speak(f"आज {today} है।", lang="hi", force=True)
            else:
                self.brain.voice.speak(f"Today is {today}.", force=True)
            return True

        # Volume up
        if _match(_p.VOLUME_UP, text):
            subprocess.run(["osascript", "-e", "set volume output volume (output volume of (get volume settings) + 10)"])
            if lang == "hi":
                self.brain.voice.speak("आवाज़ बढ़ाई।", lang="hi")
            else:
                self.brain.voice.speak("Volume increased.")
            return True

        # Volume down
        if _match(_p.VOLUME_DOWN, text):
            subprocess.run(["osascript", "-e", "set volume output volume (output volume of (get volume settings) - 10)"])
            if lang == "hi":
                self.brain.voice.speak("आवाज़ कम की।", lang="hi")
            else:
                self.brain.voice.speak("Volume decreased.")
            return True

        # Mute
        if _match(_p.VOLUME_MUTE, text):
            subprocess.run(["osascript", "-e", "set volume with output muted"])
            if lang == "hi":
                self.brain.voice.speak("म्यूट।", lang="hi")
            else:
                self.brain.voice.speak("Muted.")
            return True

        # Screenshot
        if _match(_p.SCREENSHOT, text):
            subprocess.Popen(["screencapture", "-i", "/tmp/surdas_screenshot.png"])
            if lang == "hi":
                self.brain.voice.speak("स्क्रीनशॉट ले रहा हूँ।", lang="hi")
            else:
                self.brain.voice.speak("Taking a screenshot.")
            return True

        # Lock screen
        if _match(_p.LOCK_SCREEN, text):
            subprocess.Popen(["osascript", "-e",
                'tell application "System Events" to keystroke "q" using {command down, control down}'])
            if lang == "hi":
                self.brain.voice.speak("स्क्रीन लॉक हो रही है।", lang="hi")
            else:
                self.brain.voice.speak("Locking your screen.")
            return True
        
        # Voice health diagnostic
        if text in ("voice health", "voice status", "voice diagnostic", "check voice", 
                    "microphone status", "mic status", "wake word status"):
            health = self.brain.get_voice_health()
            
            if not health.get("available"):
                self.brain.voice.speak("Voice system unavailable.", lang=lang)
                return True
            
            # Build status report
            status_parts = []
            
            if health.get("healthy"):
                status_parts.append("Voice system healthy.")
            else:
                status_parts.append("Voice system warning.")
            
            if health.get("wake_listener_active"):
                status_parts.append("Wake listener active.")
            else:
                status_parts.append("Wake listener inactive.")
            
            if health.get("is_recording"):
                status_parts.append("Currently recording.")
            elif health.get("is_processing"):
                status_parts.append("Processing command.")
            
            last_audio = health.get("last_audio_chunk")
            if last_audio is not None and last_audio < 5.0:
                status_parts.append(f"Microphone responding. Last audio {last_audio:.1f} seconds ago.")
            
            last_wake = health.get("last_wake_detection")
            if last_wake is not None and last_wake < 60.0:
                status_parts.append(f"Last wake word detected {int(last_wake)} seconds ago.")
            
            tts_speaking = health.get("tts_speaking")
            if tts_speaking:
                priority = health.get("tts_priority")
                if priority == 1:
                    status_parts.append("TTS: critical safety announcement.")
                elif priority == 2:
                    status_parts.append("TTS: navigation guidance.")
                else:
                    status_parts.append("TTS: speaking.")
            
            error_count = health.get("mic_error_count", 0)
            if error_count > 0:
                status_parts.append(f"{error_count} microphone errors.")
            
            report = " ".join(status_parts)
            self.brain.voice.speak(report, lang=lang, force=True)
            return True

        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Category 5 — Fast vision description (no LLM latency)
    # ─────────────────────────────────────────────────────────────────────────
    def _handle_vision_query(self, text: str, lang: str) -> bool:
        if not _match(_p.WHAT_DO_YOU_SEE, text):
            return False

        ctx = self.brain.get_vision_context()
        objects = ctx.get("detected_objects", [])
        closest = ctx.get("closest_obstacle", None)
        wall = ctx.get("wall_ahead", False)

        if wall:
            if lang == "hi":
                reply = "सावधान! सामने दीवार है।"
            else:
                reply = "Warning! There is a wall or barrier directly ahead."
        elif not objects:
            if lang == "hi":
                reply = "रास्ता साफ है। कोई रुकावट नहीं।"
            else:
                reply = "The path ahead looks clear. No obstacles detected."
        else:
            unique = list(dict.fromkeys(objects))
            obj_str = ", ".join(unique[:4])
            if lang == "hi":
                reply = f"मुझे दिख रहा है: {obj_str}। {closest if closest else 'सावधान रहें।'}"
            else:
                reply = f"I can see: {obj_str}. {closest if closest else 'Use caution.'}"

        self.brain.voice.speak(reply, lang=lang)
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Category 6 — System and Model Status Commands
    # ─────────────────────────────────────────────────────────────────────────
    def _handle_model_commands(self, text: str, lang: str) -> bool:
        if _match(_p.MODEL_CURRENT, text) or _match(_p.MODEL_LIST, text):
            if lang == "hi":
                self.brain.voice.speak("सूरदास अर्डुइनो यूनो क्यू पर योलो वी8एन मॉडल चला रहा है।", lang=lang)
            else:
                self.brain.voice.speak("SURDAS is running YOLOv8n object detection on Arduino UNO Q Linux.", lang=lang)
            return True
        return False

    # ─────────────────────────────────────────────────────────────────────────
    # Category 7 — Deterministic Spoken Fallback (Rule-Based, No LLM)
    # ─────────────────────────────────────────────────────────────────────────
    def _handle_llm_query(self, raw_text: str, lang: str = "en"):
        """Deterministic rule-based reply when no specific intent matched."""
        from system_state import SpeechPriority
        if lang == "hi":
            reply = f"मैंने सुना: '{raw_text}'। सूरदास अर्डुइनो यूनो क्यू पर काम कर रहा है। आप आसपास का वर्णन, कमरा याद करने, या नेविगेशन के लिए कह सकते हैं।"
        else:
            reply = f"I heard '{raw_text}'. I am running in deterministic mode on Arduino UNO Q. You can ask me to describe surroundings, guide you to a destination, or label a room."

        self.brain.voice.speak(reply, lang=lang, priority=SpeechPriority.USER_COMMAND)
