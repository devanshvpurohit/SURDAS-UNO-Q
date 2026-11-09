"""
Unit tests for Rule-Based Deterministic Voice Command Routing.
Ensures commands route deterministically without any Ollama server or LLM dependency.
"""

from unittest.mock import Mock
from voice.command_router import CommandRouter
from spatial_memory import SpatialMemory
from indoor_navigator import IndoorNavigator, NavigationState


def test_voice_command_router_deterministic_rules(tmp_path):
    db_file = tmp_path / "test_voice_spatial.db"
    spatial_mem = SpatialMemory(db_path=str(db_file))
    
    mock_brain = Mock()
    mock_voice = Mock()
    mock_brain.voice = mock_voice
    mock_brain.spatial_memory = spatial_mem
    mock_brain.indoor_navigator = IndoorNavigator(spatial_mem, mock_voice)
    mock_brain.navigator = None
    mock_brain.get_vision_context = Mock(return_value={
        "detected_objects": ["chair", "table"],
        "closest_obstacle": "chair (1.2m)",
        "wall_ahead": False
    })

    # Instantiate router without any LLM instance
    router = CommandRouter(brain=mock_brain, llm=None)

    # 1. Label Room: "remember this room as living room"
    routed = router.route_command("this room is the living room", lang="en")
    assert routed is True
    assert len(spatial_mem.list_rooms()) == 1
    assert "living room" in spatial_mem.list_rooms()[0].name

    # 2. What room is this: "what room is this"
    routed = router.route_command("what room is this", lang="en")
    assert routed is True
    mock_voice.speak.assert_called()

    # 3. Add object to memory & navigate to object: "guide me to the chair"
    spatial_mem.add_object("chair", category="furniture", x=0.0, y=2.0, z=0.0, confidence=0.9)
    routed = router.route_command("take me to the chair", lang="en")
    assert routed is True
    assert mock_brain.indoor_navigator.get_status().state in (NavigationState.NAVIGATING, NavigationState.APPROACHING)

    # 4. Pause navigation: "pause navigation"
    routed = router.route_command("pause navigation", lang="en")
    assert routed is True
    assert mock_brain.indoor_navigator.get_status().state == NavigationState.SAFETY_HOLD

    # 5. Resume navigation: "resume navigation"
    routed = router.route_command("resume navigation", lang="en")
    assert routed is True

    # 6. Stop navigation: "stop navigation"
    routed = router.route_command("stop navigation", lang="en")
    assert routed is True
    assert mock_brain.indoor_navigator.get_status().state == NavigationState.IDLE

    # 7. Vision query: "what do you see"
    routed = router.route_command("what do you see", lang="en")
    assert routed is True

    # 8. Unmatched input produces deterministic guidance reply (never calls Ollama)
    routed = router.route_command("hello tell me something random", lang="en")
    assert routed is True
    last_spoken = mock_voice.speak.call_args[0][0]
    assert "deterministic" in last_spoken.lower() or "surdas" in last_spoken.lower()


def test_bilingual_hindi_routing(tmp_path):
    db_file = tmp_path / "test_hi_spatial.db"
    spatial_mem = SpatialMemory(db_path=str(db_file))
    
    mock_brain = Mock()
    mock_voice = Mock()
    mock_brain.voice = mock_voice
    mock_brain.spatial_memory = spatial_mem
    mock_brain.indoor_navigator = IndoorNavigator(spatial_mem, mock_voice)
    mock_brain.navigator = None
    mock_brain.get_vision_context = Mock(return_value={
        "detected_objects": ["कुर्सी"],
        "closest_obstacle": "कुर्सी",
        "wall_ahead": False
    })

    router = CommandRouter(brain=mock_brain, llm=None)

    # What room is this in Hindi
    routed = router.route_command("यह कौन सा कमरा है", lang="hi")
    assert routed is True
    mock_voice.speak.assert_called()
