"""
Unit tests for Persistent SQLite Spatial Memory.
Tests creating and listing rooms, labeling landmarks and objects,
searching objects, and spatial relationships without LLMs.
"""

from spatial_memory import SpatialMemory, ObjectCategory


def test_spatial_memory_operations(tmp_path):
    db_file = tmp_path / "test_memory.db"
    mem = SpatialMemory(db_path=str(db_file))

    # 1. Add rooms
    room1_id = mem.add_room("living room", center_x=0.0, center_y=0.0, radius_m=5.0)
    room2_id = mem.add_room("kitchen", center_x=6.0, center_y=0.0, radius_m=4.0)

    rooms = mem.list_rooms()
    assert len(rooms) == 2
    room_names = [r.name for r in rooms]
    assert "living room" in room_names
    assert "kitchen" in room_names

    # 2. Add objects
    obj_id = mem.add_object(
        label="sofa",
        category=ObjectCategory.FURNITURE,
        x=1.0, y=2.0, z=0.0,
        confidence=0.92,
        room_id=room1_id
    )
    assert obj_id is not None

    found = mem.find_object_by_label("sofa")
    assert found is not None
    assert found.label == "sofa"
    assert found.confidence == 0.92

    # 3. Add landmark
    lm_id = mem.add_landmark(
        name="front door",
        x=0.0, y=-4.0, z=0.0,
        description="Main apartment entrance"
    )
    landmarks = mem.list_landmarks()
    assert len(landmarks) == 1
    assert landmarks[0].name == "front door"

    # 4. Describe surroundings
    desc = mem.describe_surroundings(0.0, 0.0, radius_m=10.0)
    assert isinstance(desc, str)
    assert "sofa" in desc
