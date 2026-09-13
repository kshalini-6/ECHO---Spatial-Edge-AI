# ECHO — Spatial Edge AI

Edge AI is making it possible for everyday spaces to become quietly aware — watching, remembering, and responding without ever sending anything to the cloud. ECHO is one small example of that idea in practice: running entirely on the Arduino UNO Q with Arduino App Lab, it watches a couple of zones and tracks objects moving between them using on-device AI. Ask it where something is, and it tells you the last zone it was seen in — no cloud, no recorded video, everything running locally on the board.
## How It Works

- **VideoObjectDetection brick** — runs a built-in COCO-pretrained model on the camera feed, detecting objects on-device.
- **Zone tracking** — checks each detected object's position against two defined zones (Desk, Drawer) using confirm/missing timers to filter out false triggers.
- **Event logging** — every appearance/disappearance is written to a local SQLite database (`events.db`) as a structured event: object, event type, zone, confidence, timestamp. No images or video frames are ever stored.
- **WebUI brick** — serves a live camera feed and an "Ask ECHO" panel where you can type a question like *"Where's my charger?"* and get back the object's current visibility and last known zone.

## Files

- `main.py` — core application logic: detection callback, zone lookup, event logging, disappearance monitor, and the `/ask` query endpoint.
- `index.html` — WebUI frontend: live feed display and the "Ask ECHO" query interface.
- `app.yaml` — App Lab manifest declaring the app and its bricks (Web UI + Video Object Detection).

## Hardware Used

- Arduino UNO Q
- Arduino USB-C Hub
- USB webcam (HD 1080p)

## Limitations

Currently limited to two zones (Desk, Drawer) due to on-device processing constraints. The COCO-pretrained model can occasionally register inconsistent detections; confirm/missing timers help reduce false positives but don't eliminate them entirely.

## Built For

Arduino UNO Q & App Lab Challenge — Home Automation category.
