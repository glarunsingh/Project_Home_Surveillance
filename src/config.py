import os
import sys
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

REQUIRED_VARS = [
    "CAMERA_USER",
    "CAMERA_PASS",
    "CAMERA_IP",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
]


def get_config():
    """Load and validate configuration from .env file."""
    missing = [v for v in REQUIRED_VARS if not os.getenv(v)]
    if missing:
        logger.error("Missing required environment variables: %s", ", ".join(missing))
        logger.error("Copy .env.example to .env and fill in your values.")
        sys.exit(1)

    return {
        "camera_user": os.getenv("CAMERA_USER"),
        "camera_pass": os.getenv("CAMERA_PASS"),
        "camera_ip": os.getenv("CAMERA_IP"),
        "camera_stream": os.getenv("CAMERA_STREAM", "stream1"),
        "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN"),
        "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID"),
        "alert_cooldown": int(os.getenv("ALERT_COOLDOWN", "30")),
        "frame_skip": int(os.getenv("FRAME_SKIP", "3")),
        "face_tolerance": float(os.getenv("FACE_TOLERANCE", "0.5")),
        "min_face_size": int(os.getenv("MIN_FACE_SIZE", "40")),
        "persistence_seconds": float(os.getenv("PERSISTENCE_SECONDS", "5")),
        "gone_timeout": float(os.getenv("GONE_TIMEOUT", "8")),
    }


def get_rtsp_url(config):
    """Build RTSP URL from config, URL-encoding the password for special chars."""
    from urllib.parse import quote
    user = quote(config['camera_user'], safe='')
    password = quote(config['camera_pass'], safe='')
    return (
        f"rtsp://{user}:{password}"
        f"@{config['camera_ip']}:554/{config['camera_stream']}"
    )
