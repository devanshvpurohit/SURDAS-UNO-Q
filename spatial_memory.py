"""
spatial_memory.py — Persistent spatial memory for SURDAS indoor navigation.

Responsibilities:
  • Store room definitions with boundaries and labels
  • Store object positions with labels, categories, and timestamps
  • Support spatial queries: "where is the chair?", "what's in this room?"
  • Track user's labeled locations (e.g., "this is the bedroom entrance")
  • Provide spatial relationships (nearest object, room membership)
  • Persist all data in SQLite for offline operation

Design principles:
  • Thread-safe: all operations use database locks
  • Offline-first: no network access, local SQLite only
  • Simple coordinates: (lat, lon) or (x, y, z) from camera
  • Confidence tracking: each memory has confidence score
  • Temporal awareness: track when objects were last seen
  • Category-aware: group objects by type (furniture, door, obstacle)

Database schema:
  rooms:       id, name, center_x, center_y, radius_m, created_at, updated_at
  objects:     id, label, category, x, y, z, confidence, room_id, last_seen, created_at
  landmarks:   id, name, x, y, z, description, created_at
  spatial_relations: id, object_id, relation_type, target_object_id, distance_m
"""

import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
import json


class ObjectCategory(str, Enum):
    """Object categories for spatial memory."""
    FURNITURE = "furniture"
    DOOR = "door"
    WINDOW = "window"
    OBSTACLE = "obstacle"
    APPLIANCE = "appliance"
    PERSON = "person"
    UNKNOWN = "unknown"


class RelationType(str, Enum):
    """Spatial relationship types."""
    NEAR = "near"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    IN_FRONT_OF = "in_front_of"
    BEHIND = "behind"
    INSIDE = "inside"


@dataclass
class SpatialObject:
    """Represents a remembered object in space."""
    id: Optional[int]
    label: str
    category: ObjectCategory
    x: float
    y: float
    z: float
    confidence: float
    room_id: Optional[int]
    last_seen: float
    created_at: float


@dataclass
class Room:
    """Represents a labeled room or area."""
    id: Optional[int]
    name: str
    center_x: float
    center_y: float
    radius_m: float
    created_at: float
    updated_at: float


@dataclass
class Landmark:
    """Represents a user-labeled navigation landmark."""
    id: Optional[int]
    name: str
    x: float
    y: float
    z: float
    description: str
    created_at: float


class SpatialMemory:
    """
    Persistent spatial memory system for indoor navigation.
    
    Usage:
        memory = SpatialMemory()
        
        # Label a room
        room_id = memory.add_room("bedroom", center_x=0.0, center_y=0.0, radius_m=5.0)
        
        # Remember an object
        obj_id = memory.add_object(
            label="chair",
            category=ObjectCategory.FURNITURE,
            x=1.5, y=2.0, z=0.0,
            confidence=0.9,
            room_id=room_id
        )
        
        # Query objects
        chair = memory.find_object_by_label("chair")
        nearby = memory.get_objects_near(x=1.0, y=1.8, radius_m=2.0)
        
        # Update when seen again
        memory.update_object_last_seen(obj_id)
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize spatial memory.
        
        Args:
            db_path: Path to SQLite database. Defaults to ./spatial_memory.db
        """
        if db_path is None:
            db_path = Path(__file__).parent / "spatial_memory.db"
        
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Create database tables if they don't exist."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS rooms (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        center_x REAL NOT NULL,
                        center_y REAL NOT NULL,
                        radius_m REAL NOT NULL,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS objects (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        label TEXT NOT NULL,
                        category TEXT NOT NULL,
                        x REAL NOT NULL,
                        y REAL NOT NULL,
                        z REAL NOT NULL,
                        confidence REAL NOT NULL,
                        room_id INTEGER,
                        last_seen REAL NOT NULL,
                        created_at REAL NOT NULL,
                        FOREIGN KEY (room_id) REFERENCES rooms(id)
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS landmarks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL UNIQUE,
                        x REAL NOT NULL,
                        y REAL NOT NULL,
                        z REAL NOT NULL,
                        description TEXT,
                        created_at REAL NOT NULL
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS spatial_relations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        object_id INTEGER NOT NULL,
                        relation_type TEXT NOT NULL,
                        target_object_id INTEGER NOT NULL,
                        distance_m REAL NOT NULL,
                        FOREIGN KEY (object_id) REFERENCES objects(id),
                        FOREIGN KEY (target_object_id) REFERENCES objects(id)
                    )
                """)
                
                # Indexes for common queries
                conn.execute("CREATE INDEX IF NOT EXISTS idx_objects_label ON objects(label)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_objects_room ON objects(room_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_objects_category ON objects(category)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_landmarks_name ON landmarks(name)")

                # Ensure observation_count and is_static columns exist (Requirement 21 & 22)
                try:
                    conn.execute("ALTER TABLE objects ADD COLUMN observation_count INTEGER DEFAULT 1")
                except sqlite3.OperationalError:
                    pass
                try:
                    conn.execute("ALTER TABLE objects ADD COLUMN is_static INTEGER DEFAULT 1")
                except sqlite3.OperationalError:
                    pass

                conn.commit()
                print("[SPATIAL_MEMORY] ✅ Database initialized")
            finally:
                conn.close()
    
    # ── Room management ───────────────────────────────────────────────────────
    
    def add_room(
        self,
        name: str,
        center_x: float,
        center_y: float,
        radius_m: float = 5.0
    ) -> int:
        """
        Add or update a room definition.
        
        Returns:
            room_id
        """
        now = time.time()
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                # Check if room exists
                cursor = conn.execute(
                    "SELECT id FROM rooms WHERE name = ?",
                    (name,)
                )
                row = cursor.fetchone()
                
                if row:
                    # Update existing room
                    room_id = row[0]
                    conn.execute(
                        """UPDATE rooms 
                           SET center_x = ?, center_y = ?, radius_m = ?, updated_at = ?
                           WHERE id = ?""",
                        (center_x, center_y, radius_m, now, room_id)
                    )
                else:
                    # Insert new room
                    cursor = conn.execute(
                        """INSERT INTO rooms (name, center_x, center_y, radius_m, created_at, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (name, center_x, center_y, radius_m, now, now)
                    )
                    room_id = cursor.lastrowid
                
                conn.commit()
                print(f"[SPATIAL_MEMORY] Room '{name}' saved (id={room_id})")
                return room_id
            finally:
                conn.close()
    
    def get_room_by_name(self, name: str) -> Optional[Room]:
        """Find a room by name."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute(
                    "SELECT id, name, center_x, center_y, radius_m, created_at, updated_at FROM rooms WHERE name = ?",
                    (name,)
                )
                row = cursor.fetchone()
                if row:
                    return Room(
                        id=row[0],
                        name=row[1],
                        center_x=row[2],
                        center_y=row[3],
                        radius_m=row[4],
                        created_at=row[5],
                        updated_at=row[6]
                    )
                return None
            finally:
                conn.close()
    
    def get_room_at_position(self, x: float, y: float) -> Optional[Room]:
        """Find which room contains the given position."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute(
                    "SELECT id, name, center_x, center_y, radius_m, created_at, updated_at FROM rooms"
                )
                rows = cursor.fetchall()
                
                for row in rows:
                    center_x, center_y, radius_m = row[2], row[3], row[4]
                    dist = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
                    if dist <= radius_m:
                        return Room(
                            id=row[0],
                            name=row[1],
                            center_x=center_x,
                            center_y=center_y,
                            radius_m=radius_m,
                            created_at=row[5],
                            updated_at=row[6]
                        )
                return None
            finally:
                conn.close()
    
    def list_rooms(self) -> List[Room]:
        """Get all rooms."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute(
                    "SELECT id, name, center_x, center_y, radius_m, created_at, updated_at FROM rooms ORDER BY name"
                )
                rows = cursor.fetchall()
                return [
                    Room(
                        id=row[0],
                        name=row[1],
                        center_x=row[2],
                        center_y=row[3],
                        radius_m=row[4],
                        created_at=row[5],
                        updated_at=row[6]
                    )
                    for row in rows
                ]
            finally:
                conn.close()
    
    # ── Object management ─────────────────────────────────────────────────────
    
    def add_object(
        self,
        label: str,
        category: ObjectCategory,
        x: float,
        y: float,
        z: float,
        confidence: float,
        room_id: Optional[int] = None
    ) -> int:
        """
        Add a new object to spatial memory.
        
        Returns:
            object_id
        """
        now = time.time()
        cat_val = category.value if hasattr(category, "value") else str(category)
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute(
                    """INSERT INTO objects (label, category, x, y, z, confidence, room_id, last_seen, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (label, cat_val, x, y, z, confidence, room_id, now, now)
                )
                obj_id = cursor.lastrowid
                conn.commit()
                print(f"[SPATIAL_MEMORY] Object '{label}' added (id={obj_id})")
                return obj_id
            finally:
                conn.close()

    def update_or_add_object(
        self,
        label: str,
        category: ObjectCategory,
        x: float,
        y: float,
        z: float,
        confidence: float,
        room_id: Optional[int] = None,
        association_radius_m: float = 1.5,
    ) -> Tuple[int, bool]:
        """
        Object association & update (Requirement 22):
        If an object with the same label exists within association_radius_m,
        update its position, confidence, last_seen, and observation_count.
        Otherwise, create a new object.
        Returns: (object_id, is_new)
        """
        now = time.time()
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute(
                    "SELECT id, x, y, z, confidence, observation_count FROM objects WHERE label = ?",
                    (label,)
                )
                rows = cursor.fetchall()
                best_match = None
                min_dist = float("inf")
                for row in rows:
                    oid, ox, oy, oz, oconf, ocount = row
                    dist = math.hypot(x - ox, y - oy)
                    if dist <= association_radius_m and dist < min_dist:
                        min_dist = dist
                        best_match = (oid, ox, oy, oz, oconf, ocount if ocount is not None else 1)

                if best_match:
                    oid, ox, oy, oz, oconf, ocount = best_match
                    smooth_x = 0.7 * ox + 0.3 * x
                    smooth_y = 0.7 * oy + 0.3 * y
                    smooth_z = 0.7 * oz + 0.3 * z
                    new_conf = max(oconf, confidence)
                    new_count = ocount + 1
                    conn.execute(
                        """UPDATE objects 
                           SET x = ?, y = ?, z = ?, confidence = ?, last_seen = ?, observation_count = ?
                           WHERE id = ?""",
                        (smooth_x, smooth_y, smooth_z, new_conf, now, new_count, oid)
                    )
                    conn.commit()
                    return oid, False
                else:
                    cat_val = category.value if hasattr(category, "value") else str(category)
                    is_static = 1 if cat_val in ("furniture", "door", "window") else 0
                    cursor = conn.execute(
                        """INSERT INTO objects (label, category, x, y, z, confidence, room_id, last_seen, created_at, observation_count, is_static)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)""",
                        (label, cat_val, x, y, z, confidence, room_id, now, now, is_static)
                    )
                    obj_id = cursor.lastrowid
                    conn.commit()
                    print(f"[SPATIAL_MEMORY] Associated new object '{label}' (id={obj_id})")
                    return obj_id, True
            finally:
                conn.close()
    
    def update_object_position(
        self,
        obj_id: int,
        x: float,
        y: float,
        z: float,
        confidence: float
    ) -> None:
        """Update object position and confidence."""
        now = time.time()
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute(
                    """UPDATE objects 
                       SET x = ?, y = ?, z = ?, confidence = ?, last_seen = ?
                       WHERE id = ?""",
                    (x, y, z, confidence, now, obj_id)
                )
                conn.commit()
            finally:
                conn.close()
    
    def update_object_last_seen(self, obj_id: int) -> None:
        """Update the last_seen timestamp for an object."""
        now = time.time()
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute(
                    "UPDATE objects SET last_seen = ? WHERE id = ?",
                    (now, obj_id)
                )
                conn.commit()
            finally:
                conn.close()
    
    def find_object_by_label(self, label: str) -> Optional[SpatialObject]:
        """Find the most recently seen object with the given label."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute(
                    """SELECT id, label, category, x, y, z, confidence, room_id, last_seen, created_at
                       FROM objects 
                       WHERE label = ?
                       ORDER BY last_seen DESC
                       LIMIT 1""",
                    (label,)
                )
                row = cursor.fetchone()
                if row:
                    return SpatialObject(
                        id=row[0],
                        label=row[1],
                        category=ObjectCategory(row[2]),
                        x=row[3],
                        y=row[4],
                        z=row[5],
                        confidence=row[6],
                        room_id=row[7],
                        last_seen=row[8],
                        created_at=row[9]
                    )
                return None
            finally:
                conn.close()
    
    def get_objects_near(
        self,
        x: float,
        y: float,
        radius_m: float = 2.0,
        category: Optional[ObjectCategory] = None
    ) -> List[SpatialObject]:
        """
        Find objects within radius_m of position (x, y).
        
        Args:
            x, y: Position to search around
            radius_m: Search radius in metres
            category: Optional filter by category
        """
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                if category:
                    cursor = conn.execute(
                        """SELECT id, label, category, x, y, z, confidence, room_id, last_seen, created_at
                           FROM objects 
                           WHERE category = ?""",
                        (category.value,)
                    )
                else:
                    cursor = conn.execute(
                        "SELECT id, label, category, x, y, z, confidence, room_id, last_seen, created_at FROM objects"
                    )
                
                rows = cursor.fetchall()
                results = []
                
                for row in rows:
                    obj_x, obj_y = row[3], row[4]
                    dist = ((x - obj_x) ** 2 + (y - obj_y) ** 2) ** 0.5
                    
                    if dist <= radius_m:
                        results.append(SpatialObject(
                            id=row[0],
                            label=row[1],
                            category=ObjectCategory(row[2]),
                            x=obj_x,
                            y=obj_y,
                            z=row[5],
                            confidence=row[6],
                            room_id=row[7],
                            last_seen=row[8],
                            created_at=row[9]
                        ))
                
                # Sort by distance
                results.sort(key=lambda obj: ((x - obj.x) ** 2 + (y - obj.y) ** 2) ** 0.5)
                return results
            finally:
                conn.close()
    
    def get_objects_in_room(self, room_id: int) -> List[SpatialObject]:
        """Get all objects in a specific room."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute(
                    """SELECT id, label, category, x, y, z, confidence, room_id, last_seen, created_at
                       FROM objects 
                       WHERE room_id = ?
                       ORDER BY last_seen DESC""",
                    (room_id,)
                )
                rows = cursor.fetchall()
                return [
                    SpatialObject(
                        id=row[0],
                        label=row[1],
                        category=ObjectCategory(row[2]),
                        x=row[3],
                        y=row[4],
                        z=row[5],
                        confidence=row[6],
                        room_id=row[7],
                        last_seen=row[8],
                        created_at=row[9]
                    )
                    for row in rows
                ]
            finally:
                conn.close()
    
    def delete_object(self, obj_id: int) -> None:
        """Remove an object from memory."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute("DELETE FROM spatial_relations WHERE object_id = ? OR target_object_id = ?", (obj_id, obj_id))
                conn.execute("DELETE FROM objects WHERE id = ?", (obj_id,))
                conn.commit()
                print(f"[SPATIAL_MEMORY] Object {obj_id} deleted")
            finally:
                conn.close()
    
    # ── Landmark management ───────────────────────────────────────────────────
    
    def add_landmark(
        self,
        name: str,
        x: float,
        y: float,
        z: float,
        description: str = ""
    ) -> int:
        """
        Add or update a named landmark (e.g., "kitchen entrance").
        
        Returns:
            landmark_id
        """
        now = time.time()
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                # Try update first
                conn.execute(
                    """INSERT INTO landmarks (name, x, y, z, description, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ON CONFLICT(name) DO UPDATE SET
                       x = excluded.x, y = excluded.y, z = excluded.z,
                       description = excluded.description""",
                    (name, x, y, z, description, now)
                )
                cursor = conn.execute("SELECT id FROM landmarks WHERE name = ?", (name,))
                landmark_id = cursor.fetchone()[0]
                conn.commit()
                print(f"[SPATIAL_MEMORY] Landmark '{name}' saved (id={landmark_id})")
                return landmark_id
            finally:
                conn.close()
    
    def find_landmark(self, name: str) -> Optional[Landmark]:
        """Find a landmark by name."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute(
                    "SELECT id, name, x, y, z, description, created_at FROM landmarks WHERE name = ?",
                    (name,)
                )
                row = cursor.fetchone()
                if row:
                    return Landmark(
                        id=row[0],
                        name=row[1],
                        x=row[2],
                        y=row[3],
                        z=row[4],
                        description=row[5],
                        created_at=row[6]
                    )
                return None
            finally:
                conn.close()
    
    def list_landmarks(self) -> List[Landmark]:
        """Get all landmarks."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute(
                    "SELECT id, name, x, y, z, description, created_at FROM landmarks ORDER BY name"
                )
                rows = cursor.fetchall()
                return [
                    Landmark(
                        id=row[0],
                        name=row[1],
                        x=row[2],
                        y=row[3],
                        z=row[4],
                        description=row[5],
                        created_at=row[6]
                    )
                    for row in rows
                ]
            finally:
                conn.close()
    
    # ── Spatial queries ───────────────────────────────────────────────────────
    
    def get_nearest_object(
        self,
        x: float,
        y: float,
        category: Optional[ObjectCategory] = None,
        exclude_stale_seconds: float = 300.0
    ) -> Optional[Tuple[SpatialObject, float]]:
        """
        Find the nearest object to position (x, y).
        
        Returns:
            (object, distance_m) or None
        """
        now = time.time()
        objects = self.get_objects_near(x, y, radius_m=100.0, category=category)
        
        # Filter stale objects
        objects = [
            obj for obj in objects
            if (now - obj.last_seen) < exclude_stale_seconds
        ]
        
        if not objects:
            return None
        
        # Already sorted by distance
        nearest = objects[0]
        dist = ((x - nearest.x) ** 2 + (y - nearest.y) ** 2) ** 0.5
        return nearest, dist
    
    def describe_surroundings(
        self,
        x: float,
        y: float,
        radius_m: float = 3.0
    ) -> str:
        """
        Generate a natural language description of nearby objects.
        
        Example: "You are near a chair and a table. The door is 2 metres ahead."
        """
        objects = self.get_objects_near(x, y, radius_m=radius_m)
        
        if not objects:
            return "No known objects nearby."
        
        # Group by category
        by_category: Dict[str, List[str]] = {}
        for obj in objects[:5]:  # Limit to 5 nearest
            cat = obj.category.value
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(obj.label)
        
        parts = []
        for category, labels in by_category.items():
            if len(labels) == 1:
                parts.append(f"a {labels[0]}")
            else:
                parts.append(f"{len(labels)} {category} items")
        
        if len(parts) == 0:
            return "No known objects nearby."
        elif len(parts) == 1:
            return f"You are near {parts[0]}."
        else:
            return f"You are near {', '.join(parts[:-1])} and {parts[-1]}."
    
    # ── Statistics and maintenance ────────────────────────────────────────────
    
    def get_stats(self) -> Dict[str, int]:
        """Get memory statistics."""
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute("SELECT COUNT(*) FROM rooms")
                room_count = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM objects")
                object_count = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM landmarks")
                landmark_count = cursor.fetchone()[0]
                
                return {
                    "rooms": room_count,
                    "objects": object_count,
                    "landmarks": landmark_count
                }
            finally:
                conn.close()
    
    def clear_stale_objects(self, max_age_seconds: float = 3600.0) -> int:
        """
        Remove objects that haven't been seen in max_age_seconds.
        
        Returns:
            Number of objects deleted
        """
        now = time.time()
        threshold = now - max_age_seconds
        
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute(
                    "SELECT id FROM objects WHERE last_seen < ?",
                    (threshold,)
                )
                stale_ids = [row[0] for row in cursor.fetchall()]
                
                for obj_id in stale_ids:
                    conn.execute("DELETE FROM spatial_relations WHERE object_id = ? OR target_object_id = ?", (obj_id, obj_id))
                
                conn.execute("DELETE FROM objects WHERE last_seen < ?", (threshold,))
                conn.commit()
                
                if stale_ids:
                    print(f"[SPATIAL_MEMORY] Cleared {len(stale_ids)} stale objects")
                
                return len(stale_ids)
            finally:
                conn.close()
    
    def export_to_json(self) -> str:
        """Export all spatial memory to JSON string."""
        data = {
            "rooms": [
                {
                    "id": r.id,
                    "name": r.name,
                    "center_x": r.center_x,
                    "center_y": r.center_y,
                    "radius_m": r.radius_m,
                    "created_at": r.created_at,
                    "updated_at": r.updated_at
                }
                for r in self.list_rooms()
            ],
            "objects": [],
            "landmarks": [
                {
                    "id": lm.id,
                    "name": lm.name,
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "description": lm.description,
                    "created_at": lm.created_at
                }
                for lm in self.list_landmarks()
            ]
        }
        
        # Get all objects
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute(
                    "SELECT id, label, category, x, y, z, confidence, room_id, last_seen, created_at FROM objects"
                )
                rows = cursor.fetchall()
                data["objects"] = [
                    {
                        "id": row[0],
                        "label": row[1],
                        "category": row[2],
                        "x": row[3],
                        "y": row[4],
                        "z": row[5],
                        "confidence": row[6],
                        "room_id": row[7],
                        "last_seen": row[8],
                        "created_at": row[9]
                    }
                    for row in rows
                ]
            finally:
                conn.close()
        
        return json.dumps(data, indent=2)
