# SPDX-License-Identifier: MPL-2.0
"""
ECHO — Edge-AI Spatial Memory
------------------------------
Runs entirely on the Arduino UNO Q using Arduino App Lab's Bricks:
  - VideoObjectDetection brick: on-device COCO object detection from the camera feed
  - WebUI brick: serves the Live Feed + "Ask ECHO" query interface

Pipeline:
  camera -> object detection -> zone lookup (Desk / Drawer) ->
  confirm/debounce logic -> log event to SQLite -> answer queries via /ask
"""

import sqlite3
import threading
import time
from datetime import datetime, UTC

from arduino.app_utils import App, Logger
from arduino.app_bricks.web_ui import WebUI
from arduino.app_bricks.video_objectdetection import VideoObjectDetection
from arduino.app_peripherals.camera import Camera

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

DB_PATH = "events.db"
logger = Logger("ECHO_Detect")

# Start the camera and expose its live feed to the WebUI at /camera
camera = Camera(resolution=(640, 480), fps=15)
ui = WebUI()
camera.start()
ui.expose_camera("/camera", camera)

# VideoObjectDetection brick: runs a built-in COCO-pretrained model on-device.
# confidence=0.65   -> minimum detection confidence to consider a hit
# debounce_sec=0.3  -> brick-level debounce to reduce flicker between frames
detection_stream = VideoObjectDetection(camera=camera, confidence=0.65, debounce_sec=0.3)

# ---------------------------------------------------------------------------
# Zones
# ---------------------------------------------------------------------------
# Each zone is a bounding box in pixel coordinates (x1, y1, x2, y2) over the
# 640x480 camera frame. An object's center point is checked against these
# boxes to decide which zone it currently belongs to.
ZONES = {
    "Desk":   (0, 0, 640, 302),
    "Drawer": (0, 302, 640, 480),
}

# Only these COCO class labels are tracked; everything else detected is ignored.
TRACKED_OBJECTS = {"keyboard", "mouse", "bottle", "book", "cell phone", "remote"}

# ---------------------------------------------------------------------------
# Timing thresholds (tuned to reduce false triggers from momentary occlusion
# or single-frame misdetections from the COCO model)
# ---------------------------------------------------------------------------
CONFIRM_SECONDS = 0.5          # how long an object must be seen in a zone before "appeared" fires
MISSING_SECONDS = 3.0          # how long an object must be absent before "disappeared" fires
MONITOR_INTERVAL_SECONDS = 0.3  # how often the background thread checks for disappearances

# ---------------------------------------------------------------------------
# In-memory state (protected by state_lock since detection callbacks and the
# disappearance-monitor thread both read/write these dicts concurrently)
# ---------------------------------------------------------------------------
state_lock = threading.Lock()
confirmed_zone = {}      # {object: zone}                      -> currently confirmed location
pending_seen = {}        # {object: {"zone": zone, "first_seen": ts}}  -> awaiting confirmation
last_seen_time = {}      # {object: unix_timestamp}             -> last frame the object was seen


def get_zone(bbox):
    """Return the zone name whose box contains the bounding box's center point,
    or None if the object isn't inside any defined zone."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    for zone_name, (zx1, zy1, zx2, zy2) in ZONES.items():
        if zx1 <= cx < zx2 and zy1 <= cy < zy2:
            return zone_name
    return None


def init_db():
    """Create the events table if it doesn't already exist. This table is
    ECHO's entire persistent memory — no images or video frames are ever
    stored, only these structured rows."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            object TEXT,
            event_type TEXT,
            zone TEXT,
            confidence REAL,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_event(obj, event_type, zone, confidence):
    """Insert one event (an object appearing in or disappearing from a zone)
    into the database and print a matching log line, e.g.:
    [APPEARED] keyboard in Desk (confidence 0.87)"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO events (object, event_type, zone, confidence, timestamp) VALUES (?, ?, ?, ?, ?)",
        (obj, event_type, zone, confidence, datetime.now(UTC).isoformat())
    )
    conn.commit()
    conn.close()
    logger.info(f"[{event_type.upper()}] {obj} in {zone} (confidence {confidence:.2f})")


def send_detections_to_ui(detections: dict):
    """Callback fired on every detection frame from the VideoObjectDetection
    brick. Applies zone lookup + confirm-debounce logic before logging an
    'appeared' event, so a single flickery frame doesn't trigger a false log."""
    now = time.time()

    with state_lock:
        for label, instances in detections.items():
            if label not in TRACKED_OBJECTS:
                continue

            # Only consider the highest-confidence detection of this label this frame
            best = max(instances, key=lambda d: d['confidence'])
            zone = get_zone(best['bounding_box_xyxy'])
            if zone is None:
                continue  # detected, but outside any defined zone

            last_seen_time[label] = now

            if label in confirmed_zone:
                # Object was already confirmed somewhere — check if it moved zones
                if confirmed_zone[label] != zone:
                    confirmed_zone[label] = zone
                    log_event(label, "appeared", zone, best['confidence'])
                pending_seen.pop(label, None)
            else:
                # Object isn't confirmed yet — require CONFIRM_SECONDS of
                # consistent presence in the same zone before logging it
                prev = pending_seen.get(label)
                if prev and prev["zone"] == zone:
                    if now - prev["first_seen"] >= CONFIRM_SECONDS:
                        confirmed_zone[label] = zone
                        log_event(label, "appeared", zone, best['confidence'])
                        pending_seen.pop(label, None)
                else:
                    # First time seeing it (or it changed zone before confirming) — start the timer
                    pending_seen[label] = {"zone": zone, "first_seen": now}


def disappearance_monitor():
    """Background thread: periodically checks confirmed objects and logs a
    'disappeared' event once an object hasn't been seen for MISSING_SECONDS.
    Runs independently of the detection callback so disappearances are caught
    even if the object simply stops being detected (rather than moving zones)."""
    while True:
        time.sleep(MONITOR_INTERVAL_SECONDS)
        now = time.time()
        with state_lock:
            for label in list(confirmed_zone.keys()):
                last_seen = last_seen_time.get(label, now)
                if now - last_seen >= MISSING_SECONDS:
                    last_zone = confirmed_zone.pop(label)
                    last_seen_time.pop(label, None)
                    pending_seen.pop(label, None)
                    log_event(label, "disappeared", last_zone, 0.0)


# Wire the detection brick's callback and start the background monitor thread
detection_stream.on_detect_all(send_detections_to_ui)

monitor_thread = threading.Thread(target=disappearance_monitor, daemon=True)
monitor_thread.start()


def answer_question(request: dict):
    """Handler for POST /ask. Takes a natural-language question, matches it
    against known object names logged so far, then returns that object's
    current visibility plus its first-seen and last-seen zone/timestamp."""
    question = (request.get("question") or "").lower()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT object FROM events")
    known_objects = [row[0] for row in cur.fetchall()]

    # Simple keyword match: look for the object's full name or any of its
    # individual words inside the question text
    target = None
    for obj in known_objects:
        obj_words = obj.lower().split()
        if obj.lower() in question or any(word in question for word in obj_words):
            target = obj
            break

    if target is None:
        conn.close()
        return {
            "found": False,
            "message": f"No record found. Known objects so far: {', '.join(known_objects) if known_objects else 'none yet'}."
        }

    # Earliest "appeared" event ever logged for this object
    cur.execute(
        "SELECT timestamp, zone FROM events WHERE object = ? AND event_type = 'appeared' ORDER BY timestamp ASC LIMIT 1",
        (target,)
    )
    first_row = cur.fetchone()

    # Most recent event of any type for this object
    cur.execute(
        "SELECT timestamp, zone FROM events WHERE object = ? ORDER BY timestamp DESC LIMIT 1",
        (target,)
    )
    last_row = cur.fetchone()

    conn.close()

    with state_lock:
        currently_visible = target in confirmed_zone
        current_zone = confirmed_zone.get(target)

    return {
        "found": True,
        "object": target,
        "currently_visible": currently_visible,
        "current_zone": current_zone,
        "first_seen": first_row[0] if first_row else None,
        "first_seen_zone": first_row[1] if first_row else None,
        "last_seen": last_row[0] if last_row else None,
        "last_seen_zone": last_row[1] if last_row else None,
    }


# Expose the query endpoint to the WebUI's "Ask ECHO" panel
ui.expose_api("POST", "/ask", answer_question)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
init_db()
App.run()
