# Smart Surveillance System

A Python-based smart surveillance system using the **Tapo C210** IP camera with real-time **face recognition** and **restricted zone intrusion detection**. Alerts are sent to Telegram with annotated photos.

## Features

- **Live RTSP Camera Feed** — Connects to Tapo C210 via RTSP with auto-retry
- **Face Recognition** — Identifies known persons using 128-dimensional face encodings (dlib)
- **Restricted Zone Detection** — Interactive polygon drawing to define no-go areas
- **Telegram Alerts** — Sends annotated photos on unknown face or zone intrusion (with cooldown)
- **Configurable** — Frame skip, face tolerance, alert cooldown via `.env`
- **Graceful Shutdown** — Clean exit with Ctrl+C or 'q' key

## Architecture

```
src/
├── config.py         # Loads & validates .env configuration
├── face_encoder.py   # Computes & caches 128-d face encodings
├── camera_feed.py    # RTSP connection with retry logic
├── zone_marker.py    # Interactive zone drawing, save/load, point-in-polygon
├── detector.py       # Per-frame face detection + recognition + zone check
├── notifier.py       # Telegram Bot API (sendPhoto with cooldown)
└── main.py           # Orchestrator — ties all modules together
```

## Prerequisites

- **Python 3.10+**
- **Tapo C210** camera on the same network
- **cmake** (required for dlib/face_recognition compilation)
- **Telegram Bot** token from [@BotFather](https://t.me/BotFather)

### macOS

```bash
brew install cmake
```

### Ubuntu/Debian

```bash
sudo apt-get install cmake build-essential
```

## Setup

### 1. Clone & Create Virtual Environment

```bash
git clone https://github.com/glarunsingh/Project_Home_Surveillance.git
cd Project_Home_Surveillance
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note (Python 3.14+):** If `face_recognition` fails to install, you may need:
> ```bash
> pip install git+https://github.com/ageitgey/face_recognition_models
> ```

### 3. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your values:

| Variable | Description | Example |
|---|---|---|
| `CAMERA_USER` | Tapo camera username | `admin` |
| `CAMERA_PASS` | Tapo camera password | `yourpassword` |
| `CAMERA_IP` | Camera IP address | `192.168.1.4` |
| `CAMERA_STREAM` | RTSP stream path | `stream1` (HD) or `stream2` (SD) |
| `TELEGRAM_BOT_TOKEN` | Bot token from @BotFather | `123456:ABC-DEF...` |
| `TELEGRAM_CHAT_ID` | Your Telegram user/chat ID | `1234567890` |
| `ALERT_COOLDOWN` | Seconds between alerts | `30` |
| `FRAME_SKIP` | Process every Nth frame | `3` |
| `FACE_TOLERANCE` | Face match threshold (lower = stricter) | `0.5` |

### 4. Add Known Faces

Create a subfolder per person inside `known_faces/`:

```
known_faces/
├── John/
│   ├── photo1.jpg
│   └── photo2.jpg
└── Jane/
    └── photo1.jpg
```

- Use clear, front-facing photos
- Multiple photos per person improves accuracy
- Encodings are cached to `known_faces/encodings.pkl` (auto-regenerated when images change)

### 5. Set Up Restricted Zone (Optional)

```bash
python3 -m src.zone_marker --rezone
```

Click on the camera frame to define a polygon. Press **Enter** to save, **Esc** to cancel.

## Usage

### Run Surveillance

```bash
python3 -m src.main
```

### CLI Options

| Flag | Description |
|---|---|
| *(none)* | Normal mode — loads existing zone config |
| `--rezone` | Redraw the restricted zone before starting |
| `--no-zone` | Run without zone intrusion detection |

### Keyboard Controls (Live Window)

| Key | Action |
|---|---|
| `q` | Quit surveillance |
| `Ctrl+C` | Graceful shutdown |

### Test Individual Modules

```bash
# Test camera connection
python3 -m src.camera_feed

# Test Telegram bot
python3 -m src.notifier

# Test zone drawing
python3 -m src.zone_marker --rezone
```

## How It Works

1. Camera feed streams frames via RTSP
2. Every Nth frame runs through the detection pipeline:
   - Detect face locations (HOG model)
   - Compute 128-d face encodings
   - Match against known face database
   - Check if face centers fall inside restricted zone
3. Annotated frame displayed with bounding boxes:
   - 🟢 **Green** — Known person, safe area
   - 🔴 **Red** — Unknown person
   - 🟠 **Orange** — Known person in restricted zone
4. Alerts sent to Telegram with annotated snapshot (rate-limited by cooldown)

## Project Structure

```
Project_Home_Surveillance/
├── .env                  # Your credentials (git-ignored)
├── .env.example          # Template
├── .gitignore
├── requirements.txt
├── Project_details.md    # Full PRD document
├── README.md             # This file
├── known_faces/          # Face images (subfolders per person)
│   └── encodings.pkl     # Auto-generated cache
├── zone_config/
│   └── zone.json         # Saved zone polygon
└── src/
    ├── __init__.py
    ├── config.py
    ├── face_encoder.py
    ├── camera_feed.py
    ├── zone_marker.py
    ├── detector.py
    ├── notifier.py
    └── main.py
```

## License

This project is for personal/educational use.
