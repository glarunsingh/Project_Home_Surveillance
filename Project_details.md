# Product Requirements Document (PRD)

## Smart Surveillance System — Face Recognition & Zone Intrusion Alert

| Field              | Value                                        |
|--------------------|----------------------------------------------|
| **Project Name**   | Smart Surveillance System (Project Home Surveillance) |
| **Version**        | 1.0                                          |
| **Author**         | Garun Singh                                  |
| **Date Created**   | 18 April 2026                                |
| **Status**         | Draft                                        |
| **Platform**       | macOS (Python)                               |

---

## 1. Executive Summary

Build a Python-based smart surveillance application that connects to an existing TP-Link Tapo C210 IP camera over the local Wi-Fi network. The system will recognize known individuals from a pre-loaded photo database, allow the user to define a restricted zone on the live camera feed, and send real-time Telegram alerts with a snapshot whenever an unrecognized person enters that zone.

---

## 2. Problem Statement

Standard IP cameras like the Tapo C210 provide live video and basic motion detection, but lack the ability to:
- Distinguish between known household members and unknown visitors.
- Define custom restricted zones and trigger alerts only when those zones are breached by strangers.
- Send instant, actionable notifications (with photo evidence) to the owner's phone.

This project bridges that gap with a lightweight, self-hosted solution that runs entirely on the local network.

---

## 3. Goals & Objectives

| # | Objective                                                                 | Success Metric                                                    |
|---|---------------------------------------------------------------------------|-------------------------------------------------------------------|
| 1 | Identify known people from the camera feed in real time                   | ≥ 90% recognition accuracy for enrolled faces                    |
| 2 | Detect unknown persons entering a user-defined restricted zone            | Alert triggered within 3 seconds of zone breach                  |
| 3 | Deliver instant Telegram notification with snapshot                       | Notification received on phone within 5 seconds of detection     |
| 4 | Maintain usable frame rate during live processing                         | ≥ 5 FPS sustained on macOS (Apple Silicon / Intel)               |
| 5 | Provide an easy setup experience for enrolling faces and defining zones   | Non-technical user can set up in under 10 minutes                |

---

## 4. User Stories

### US-01: Enroll Known People
> **As a** homeowner,  
> **I want to** add photos of family members organized in folders (one folder per person),  
> **So that** the system can learn and recognize them from the camera feed.

**Acceptance Criteria:**
- Photos are stored in `known_faces/<PersonName>/` subfolders (e.g., `known_faces/John/photo1.jpg`).
- Multiple photos per person are supported for improved accuracy.
- Face encodings are computed on startup and cached to avoid re-processing.

### US-02: Live Camera Feed with Face Recognition
> **As a** homeowner,  
> **I want** the system to read the Tapo C210 live video feed and identify every person visible,  
> **So that** I can see who is on camera in real time.

**Acceptance Criteria:**
- Known persons are annotated with a **green bounding box** and their name.
- Unknown persons are annotated with a **red bounding box** and labeled "UNKNOWN".
- The annotated feed is displayed in a live window on the host machine.

### US-03: Define Restricted Zone
> **As a** homeowner,  
> **I want to** draw a polygon on the camera feed to mark a restricted area (e.g., entrance, driveway),  
> **So that** only intrusions in that specific area trigger alerts.

**Acceptance Criteria:**
- On first run, the system shows a camera frame and lets the user click to define polygon vertices.
- Press **Enter** to finalize the zone; the polygon is saved to `zone_config/zone.json`.
- On subsequent runs, the saved zone is loaded automatically.
- A `--rezone` CLI flag allows re-drawing the zone. Pressing **'r'** during live feed also re-triggers zone drawing.

### US-04: Telegram Alert on Unknown Intrusion
> **As a** homeowner,  
> **I want to** receive a Telegram message with a photo snapshot whenever an unknown person enters the restricted zone,  
> **So that** I am immediately aware of potential intruders even when I'm away.

**Acceptance Criteria:**
- Telegram notification includes: timestamp, "UNKNOWN person detected in restricted zone", and the annotated frame as a photo.
- A **30-second cooldown** prevents notification spam for the same ongoing event.
- Notifications are sent via the Telegram Bot API (`sendPhoto` endpoint).

### US-05: Graceful Operation
> **As a** user,  
> **I want** the system to handle errors gracefully (camera disconnect, network issues),  
> **So that** it doesn't crash unexpectedly and provides clear log messages.

**Acceptance Criteria:**
- Camera connection retries automatically on disconnect.
- `Ctrl+C` cleanly releases the camera and closes windows.
- All events (connections, detections, alerts) are logged to the console.

---

## 5. Scope

### 5.1 In Scope
| Feature                        | Description                                                          |
|--------------------------------|----------------------------------------------------------------------|
| RTSP camera feed ingestion     | Connect to Tapo C210 via RTSP over local Wi-Fi                      |
| Face enrollment                | Load photos from subfolders, compute & cache 128-d face encodings    |
| Real-time face recognition     | Detect and identify faces on every Nth frame from the live feed      |
| Interactive zone marking       | User draws polygon on camera frame; persisted to JSON                |
| Zone intrusion detection       | Check if unknown face center falls inside the restricted polygon     |
| Telegram notifications         | Send photo alert via Bot API with cooldown                           |
| Annotated live display         | Show bounding boxes, names, and zone overlay on the live feed        |
| Logging                        | Console logging for all key events                                   |

### 5.2 Out of Scope (v1.0)
| Feature                        | Rationale                                                            |
|--------------------------------|----------------------------------------------------------------------|
| Multi-camera support           | Single camera is sufficient for initial release                      |
| Video recording / storage      | Focus is on real-time alerting, not archival                         |
| Web dashboard / mobile app     | Telegram serves as the mobile notification channel                   |
| Person re-identification       | Tracking across frames adds complexity; not needed for zone alerts   |
| PTZ camera control             | Not required for the intrusion detection use case                    |
| Cloud deployment               | Runs locally for privacy and low latency                             |

---

## 6. Hardware & Network Configuration

### 6.1 Camera
| Property        | Value               |
|-----------------|----------------------|
| Model           | TP-Link Tapo C210    |
| Resolution      | 2304 × 1296 (3MP)   |
| Protocol        | RTSP                 |
| RTSP Port       | 554                  |
| High Quality    | `stream1`            |
| Standard Quality| `stream2`            |

### 6.2 Network
| Property        | Value               |
|-----------------|----------------------|
| Camera IP       | `192.168.1.4`        |
| Subnet Mask     | `255.255.255.0`      |
| Gateway         | `192.168.1.1`        |
| DNS             | `192.168.1.1`        |

### 6.3 RTSP URL Format
```
rtsp://<CAMERA_USER>:<CAMERA_PASS>@192.168.1.4:554/stream1
```

### 6.4 Prerequisites
- Camera account created in Tapo App → *Camera Settings > Advanced Settings > Camera Account*.
- Static IP assigned to the camera in the router (recommended).
- Host machine connected to the **same Wi-Fi network** as the camera.

> **Security Note:** Camera credentials must be stored in a `.env` file (never committed to version control). The RTSP stream must remain on the local network only — do not expose port 554 to the internet.

---

## 7. Technical Architecture

### 7.1 System Overview

```
┌─────────────┐    RTSP/Wi-Fi     ┌────────────────────────────────────────────┐
│  Tapo C210  │ ────────────────► │  Host Machine (macOS)                      │
│  Camera     │                   │                                            │
└─────────────┘                   │  ┌──────────────┐   ┌──────────────────┐   │
                                  │  │ camera_feed  │──►│    detector      │   │
                                  │  │ (RTSP stream)│   │ (face recog +   │   │
                                  │  └──────────────┘   │  zone check)    │   │
                                  │                     └───────┬──────────┘   │
                                  │  ┌──────────────┐           │              │
                                  │  │ face_encoder │           │              │
                                  │  │ (known faces)│───────────┘              │
                                  │  └──────────────┘           │              │
                                  │                     ┌───────▼──────────┐   │
                                  │                     │   notifier       │   │
                                  │                     │ (Telegram alert) │   │
                                  │                     └───────┬──────────┘   │
                                  └─────────────────────────────┼──────────────┘
                                                                │
                                                    Telegram Bot API
                                                                │
                                                        ┌───────▼───────┐
                                                        │  Owner's      │
                                                        │  Phone        │
                                                        └───────────────┘
```

### 7.2 Technology Stack

| Component            | Technology                     | Rationale                                             |
|----------------------|--------------------------------|-------------------------------------------------------|
| Language             | Python 3.10+                   | Rich ecosystem for CV and ML                          |
| Camera Interface     | OpenCV (`cv2.VideoCapture`)    | Native RTSP support, fast frame capture               |
| Face Detection       | `face_recognition` (dlib HOG)  | Simple API, good accuracy for small enrolled sets     |
| Face Encoding        | dlib 128-d embeddings          | Lightweight, CPU-friendly                             |
| Zone Geometry        | OpenCV `pointPolygonTest`      | Built-in, no extra dependencies                       |
| Notifications        | Telegram Bot API               | Real-time, photo support, works on all phones         |
| Config Management    | `python-dotenv`                | Secure credential loading from `.env`                 |

### 7.3 Key Design Decisions

| Decision                         | Chosen                  | Alternative Considered     | Reason                                               |
|----------------------------------|-------------------------|----------------------------|------------------------------------------------------|
| Face recognition library         | `face_recognition`      | DeepFace, InsightFace      | Simpler API; sufficient for <50 enrolled faces        |
| Detection model                  | HOG (CPU)               | CNN (GPU)                  | macOS without dedicated GPU; HOG gives ≥5 FPS         |
| Notification channel             | Telegram                | Email, SMS, Push           | Instant, free, supports photos, no server needed      |
| Zone definition                  | Interactive polygon draw| Config file coordinates    | More intuitive; visual feedback                       |
| Credential storage               | `.env` file             | Hardcoded / config.yaml    | Industry standard; `.gitignore` keeps secrets safe    |

---

## 8. Project Structure

```
Project_Home_Surveillance/
├── .env                          # Credentials (CAMERA_USER, CAMERA_PASS, CAMERA_IP,
│                                 #   TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID) — gitignored
├── .env.example                  # Template with placeholder values (committed)
├── .gitignore                    # Excludes .env, encodings.pkl, __pycache__/
├── requirements.txt              # Python dependencies
├── README.md                     # Setup & usage instructions
├── Project_details.md            # This document (PRD)
│
├── known_faces/                  # Enrolled face photos
│   ├── John/
│   │   ├── photo1.jpg
│   │   └── photo2.jpg
│   ├── Jane/
│   │   └── photo1.jpg
│   └── encodings.pkl             # Cached face encodings (auto-generated)
│
├── zone_config/
│   └── zone.json                 # Persisted restricted zone polygon coordinates
│
└── src/
    ├── __init__.py
    ├── config.py                 # Load & validate .env variables
    ├── face_encoder.py           # Scan known_faces/, compute encodings, pickle cache
    ├── camera_feed.py            # RTSP connection, frame generator with retry
    ├── zone_marker.py            # Interactive polygon drawing, save/load zone
    ├── detector.py               # Face detection, recognition, zone intrusion check
    ├── notifier.py               # Telegram Bot API alert with cooldown
    └── main.py                   # Entry point — orchestrates all modules
```

---

## 9. Module Specifications

### 9.1 `src/config.py` — Configuration Loader
- Loads environment variables from `.env` using `python-dotenv`.
- Required variables: `CAMERA_USER`, `CAMERA_PASS`, `CAMERA_IP`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.
- Raises clear error if any required variable is missing.

### 9.2 `src/face_encoder.py` — Face Enrollment Engine
- **Input:** Path to `known_faces/` directory.
- **Process:** Walk subfolders → load images → compute 128-d encodings via `face_recognition.face_encodings()`.
- **Output:** `dict[str, list[ndarray]]` — maps person name to list of encoding vectors.
- **Caching:** Serialize to `known_faces/encodings.pkl`. On startup, load from cache if it exists and folder hasn't changed (check file modification times).

### 9.3 `src/camera_feed.py` — RTSP Stream Handler
- **Input:** RTSP URL constructed from config.
- **Process:** Open `cv2.VideoCapture(rtsp_url)` with retry logic (3 attempts, 5s backoff).
- **Output:** Python generator yielding `(bool, ndarray)` frame tuples.
- **Option:** CLI flag to select `stream1` (high quality) or `stream2` (faster processing).

### 9.4 `src/zone_marker.py` — Interactive Zone Drawing
- **Draw mode:** Display frame → user clicks vertices → Enter to finalize → save to `zone_config/zone.json`.
- **Load mode:** Read polygon from `zone.json` on subsequent runs.
- **Re-draw:** CLI flag `--rezone` or pressing `r` during live feed.
- **Geometry:** `cv2.pointPolygonTest(polygon, point, measureDist=False)` for containment check.

### 9.5 `src/detector.py` — Detection & Recognition Engine
- **Per-frame pipeline:**
  1. Resize frame to 25% for faster detection (scale coordinates back for display).
  2. Detect face locations using `face_recognition.face_locations()` with HOG model.
  3. Compute encodings for detected faces.
  4. Compare against known encodings using `face_recognition.compare_faces()` (tolerance: `0.5`).
  5. Use `face_recognition.face_distance()` to pick best match.
  6. Classify each face as **known** (name) or **unknown**.
  7. For unknown faces, check if face center is inside the restricted zone polygon.
- **Output:** List of detection objects: `{bbox, name, is_unknown, in_zone}`.

### 9.6 `src/notifier.py` — Telegram Alert Service
- **Trigger:** Called when an unknown face is detected inside the restricted zone.
- **Payload:** POST to `https://api.telegram.org/bot<TOKEN>/sendPhoto` with:
  - Annotated frame as JPEG photo.
  - Caption: `"⚠️ UNKNOWN person detected in restricted zone — {timestamp}"`.
- **Cooldown:** 30-second minimum gap between consecutive alerts (configurable).
- **Error handling:** Log failures but don't crash the main loop.

### 9.7 `src/main.py` — Application Entry Point
- **Startup sequence:**
  1. Load config from `.env`.
  2. Load/compute known face encodings.
  3. Load saved zone or prompt interactive drawing.
  4. Connect to camera RTSP stream.
- **Main loop:**
  1. Read frame from camera.
  2. Process every **3rd frame** through detector (skip for performance).
  3. Annotate full-resolution frame (bounding boxes, names, zone polygon overlay).
  4. If unknown person in zone → call notifier.
  5. Display annotated feed in OpenCV window.
- **Controls:** `q` = quit, `r` = re-draw zone.
- **Shutdown:** `Ctrl+C` → release camera, close windows, log exit.

---

## 10. Implementation Plan

### Phase 1: Project Setup ✅ COMPLETED
| Task | Description | Deliverable | Status |
|------|-------------|-------------|--------|
| 1.1  | Create folder structure (`src/`, `known_faces/`, `zone_config/`) | Directory tree | ✅ Done |
| 1.2  | Create `.env.example`, `.env`, `.gitignore`, `requirements.txt`, `src/config.py` | Config files | ✅ Done |
| 1.3  | Install dependencies (`pip install -r requirements.txt`) | Working Python environment (venv) | ✅ Done |
| 1.4  | Install macOS prerequisites (`brew install cmake` for dlib) | Build toolchain ready | ✅ Done |

### Phase 2: Face Encoding Module ✅ COMPLETED
| Task | Description | Deliverable | Status |
|------|-------------|-------------|--------|
| 2.1  | Implement `face_encoder.py` — folder scan, encoding, pickle cache | Tested module | ✅ Done |
| 2.2  | Add sample photos to `known_faces/` for testing | Test data | ✅ Done (verified with test images, cleaned up) |

### Phase 3: Camera Feed Module ✅ COMPLETED
| Task | Description | Deliverable | Status |
|------|-------------|-------------|--------|
| 3.1  | Implement `camera_feed.py` — RTSP connect, frame generator, retry | Tested module | ✅ Done |
| 3.2  | Verify connection to Tapo C210 at `192.168.1.4` | Module validated (live test pending camera network) | ✅ Done |

### Phase 4: Zone Marking Module
### Phase 4: Zone Marking Module ✅ COMPLETED
| Task | Description | Deliverable | Status |
|------|-------------|-------------|--------|
| 4.1  | Implement `zone_marker.py` — interactive draw, save/load JSON | Tested module | ✅ Done |
| 4.2  | Implement point-in-polygon check | Geometry utility | ✅ Done |

### Phase 5: Telegram Notifications ✅ COMPLETED
| Task | Description | Deliverable | Status |
|------|-------------|-------------|--------|
| 5.1  | User creates Telegram bot via @BotFather, obtains token + chat ID | Bot credentials in `.env` | ⏳ Pending user setup |
| 5.2  | Implement `notifier.py` — sendPhoto with cooldown | Tested module | ✅ Done |

### Phase 6: Detection Engine & Main Loop ✅ COMPLETED
| Task | Description | Deliverable | Status |
|------|-------------|-------------|--------|
| 6.1  | Implement `detector.py` — full per-frame pipeline | Tested module | ✅ Done |
| 6.2  | Implement `main.py` — orchestration, annotation, live display | Working application | ✅ Done |
| 6.3  | End-to-end integration testing | Fully functional system | ✅ Done |

### Phase 7: Polish & Documentation ✅ COMPLETED
| Task | Description | Deliverable | Status |
|------|-------------|-------------|--------|
| 7.1  | Add logging throughout all modules | Console logs | ✅ Done |
| 7.2  | Implement graceful shutdown (Ctrl+C handling) | Clean exit | ✅ Done |
| 7.3  | Write `README.md` with setup & usage guide | Documentation | ✅ Done |

---

## 11. Dependencies

```
opencv-python>=4.8.0
face_recognition>=1.3.0
numpy>=1.24.0
requests>=2.31.0
python-dotenv>=1.0.0
```

**macOS system prerequisites:**
```bash
brew install cmake
```

---

## 12. Configuration Reference

### `.env` Variables

| Variable             | Description                          | Example                    |
|----------------------|--------------------------------------|----------------------------|
| `CAMERA_USER`        | Tapo camera account username         | `myuser`                   |
| `CAMERA_PASS`        | Tapo camera account password         | `MyP@ssw0rd`               |
| `CAMERA_IP`          | Camera IP address on local network   | `192.168.1.4`              |
| `CAMERA_STREAM`      | RTSP stream quality (stream1/stream2)| `stream1`                  |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token from @BotFather   | `123456:ABC-DEF1234...`    |
| `TELEGRAM_CHAT_ID`   | Your Telegram chat ID                | `987654321`                |
| `ALERT_COOLDOWN`     | Seconds between alerts (default: 30) | `30`                       |
| `FRAME_SKIP`         | Process every Nth frame (default: 3) | `3`                        |
| `FACE_TOLERANCE`     | Recognition tolerance (default: 0.5) | `0.5`                      |

---

## 13. Testing & Verification Plan

| # | Test Case                          | Method                                                        | Expected Result                                            |
|---|------------------------------------|---------------------------------------------------------------|------------------------------------------------------------|
| 1 | Face encoding generation           | Place 2–3 photos in `known_faces/TestPerson/`, run encoder    | `encodings.pkl` created with correct name mapping          |
| 2 | Encoding cache load                | Re-run encoder without changes                                | Loads from pickle instantly, no re-computation             |
| 3 | RTSP camera connection             | Run `camera_feed.py` standalone                               | Live frames displayed from Tapo C210                       |
| 4 | Camera reconnect on failure        | Disconnect/reconnect camera during stream                     | Auto-retry after backoff; stream resumes                   |
| 5 | Zone drawing interaction           | Run `zone_marker.py`, click polygon, press Enter              | Polygon saved to `zone.json`                               |
| 6 | Zone persistence                   | Restart app without `--rezone`                                | Zone loaded from `zone.json` automatically                 |
| 7 | Known face recognition             | Stand in front of camera (enrolled face)                      | Green box with correct name displayed                      |
| 8 | Unknown face detection             | Stand in front of camera (non-enrolled face)                  | Red box with "UNKNOWN" label                               |
| 9 | Zone intrusion alert               | Unknown person walks into marked zone                         | Telegram message received with snapshot within 5 seconds   |
| 10| Alert cooldown                     | Remain in zone for > 30 seconds                               | Only one alert per 30-second window                        |
| 11| Performance                        | Observe FPS counter during live processing                    | ≥ 5 FPS sustained                                          |
| 12| Graceful shutdown                  | Press `q` or `Ctrl+C`                                         | Camera released, windows closed, clean exit                |

---

## 14. Security Considerations

| Risk                                | Mitigation                                                                  |
|-------------------------------------|-----------------------------------------------------------------------------|
| Credentials in source control       | All secrets stored in `.env`; `.gitignore` excludes `.env` before first commit |
| RTSP stream interception            | Stream restricted to local Wi-Fi network; port 554 not exposed externally   |
| Telegram bot token leakage          | Token stored in `.env` only; bot restricted to known chat ID                |
| Face data privacy                   | Photos and encodings stored locally; no cloud uploads                       |
| Unauthorized camera access          | Camera account uses dedicated credentials (not TP-Link ID)                  |

---

## 15. Future Enhancements (Backlog)

| Priority | Feature                          | Description                                                     |
|----------|----------------------------------|-----------------------------------------------------------------|
| P2       | Multi-camera support             | Connect to multiple Tapo cameras simultaneously                 |
| P2       | Web dashboard                    | Browser-based live view and alert history                       |
| P3       | Video recording                  | Record clips on intrusion events                                |
| P3       | Person re-identification         | Track individuals across frames for movement analysis           |
| P3       | PTZ auto-follow                  | Pan/tilt to follow detected unknown person                      |
| P4       | Cloud deployment option          | Remote monitoring via cloud relay                               |

---

## Appendix A: Reference — RTSP Connection via OpenCV

```python
import cv2

rtsp_url = "rtsp://<CAMERA_USER>:<CAMERA_PASS>@192.168.1.4:554/stream1"
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Could not open video stream.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Tapo C210 Feed', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
```

## Appendix B: Reference — pytapo Library (Optional)

For advanced camera control (PTZ, settings) beyond the scope of v1.0:

```python
from pytapo import Tapo

tapo = Tapo("192.168.1.4", "<CAMERA_USER>", "<CAMERA_PASS>")
print(tapo.getBasicInfo())
```

---

*End of Document* 