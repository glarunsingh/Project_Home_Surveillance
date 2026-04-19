"""
Smart Surveillance System — Main Entry Point

Uses persistence-based face tracking: only alerts on faces that remain
in the camera's view for a configurable duration (default 5 seconds).
This naturally filters out road traffic, passersby, and other transient
detections — no zone/ROI drawing needed.

Usage:
    python3 -m src.main              # Run surveillance
"""

import sys
import signal
import time
import logging
import cv2

from src.config import get_config, get_rtsp_url
from src.face_encoder import load_known_faces
from src.camera_feed import connect, LatestFrameGrabber, latest_frame_generator, release
from src.detector import process_frame, build_recognition_data, annotate_frame
from src.face_tracker import FaceTracker
from src.notifier import TelegramNotifier

logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
_running = True


def _signal_handler(sig, frame):
    """Handle Ctrl+C for graceful shutdown."""
    global _running
    logger.info("Shutdown signal received. Stopping...")
    _running = False


def main():
    global _running

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Register signal handler
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    logger.info("=" * 60)
    logger.info("  SMART SURVEILLANCE SYSTEM")
    logger.info("  Persistence-based face tracking")
    logger.info("=" * 60)

    # --- 1. Load config ---
    config = get_config()
    logger.info("Configuration loaded.")

    # --- 2. Load known faces ---
    known_names, encodings_by_person = load_known_faces("known_faces")
    person_count = len(encodings_by_person)
    encoding_count = sum(len(v) for v in encodings_by_person.values())
    if person_count > 0:
        logger.info("Loaded %d encoding(s) for %d person(s).", encoding_count, person_count)
    else:
        logger.warning("No known faces loaded. All faces will be marked UNKNOWN.")
        logger.warning("Add face images to known_faces/<PersonName>/ and restart.")

    # Pre-compute flat recognition arrays (avoids rebuilding per frame)
    all_known_names, all_known_encodings = build_recognition_data(encodings_by_person)

    # --- 3. Setup Telegram notifier ---
    notifier = TelegramNotifier(
        bot_token=config["telegram_bot_token"],
        chat_id=config["telegram_chat_id"],
        cooldown_seconds=config["alert_cooldown"],
    )
    bot_name = notifier.test_connection()
    if bot_name:
        logger.info("Telegram bot online: @%s", bot_name)
    else:
        logger.warning("Telegram bot connection failed. Alerts will not be sent.")

    # --- 4. Connect to camera ---
    rtsp_url = get_rtsp_url(config)
    cap = connect(rtsp_url)
    if cap is None:
        logger.error("Could not connect to camera. Exiting.")
        sys.exit(1)

    # --- 5. Setup face persistence tracker ---
    persistence_secs = config.get("persistence_seconds", 5.0)
    gone_timeout = config.get("gone_timeout", 8.0)
    tracker = FaceTracker(
        persistence_seconds=persistence_secs,
        match_tolerance=config["face_tolerance"] + 0.05,  # slightly looser for cross-frame matching
        gone_timeout=gone_timeout,
    )
    logger.info("Face tracker: alert after %.1fs persistence, forget after %.1fs absence.",
                persistence_secs, gone_timeout)

    # Start threaded frame grabber for real-time feed
    grabber = LatestFrameGrabber(cap)

    # --- 6. Main surveillance loop ---
    logger.info("-" * 60)
    logger.info("Surveillance active. Press 'q' to quit.")
    logger.info("-" * 60)

    if bot_name:
        notifier.send_text("✅ Surveillance system started.")

    frame_count = 0
    fps_counter = 0
    fps_timer = time.time()
    current_fps = 0.0
    frame_skip = config["frame_skip"]
    tolerance = config["face_tolerance"]
    min_face_size = config["min_face_size"]

    while _running:
        for frame in latest_frame_generator(grabber):
            if not _running:
                break

            frame_count += 1

            # Compensate for camera pan/tilt on EVERY frame (fast ~1 ms).
            # Keeps tracked positions aligned with the actual frame so
            # spatial matching survives camera motion.
            tracker.compensate_camera_motion(frame)

            # Skip frames for performance — but still draw tracked face boxes
            if frame_count % frame_skip != 0:
                if tracker.active_tracks:
                    annotate_frame(frame, tracker.active_tracks,
                                   tracker.persistence_seconds)
                cv2.imshow("Surveillance", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    _running = False
                    break
                continue

            # Run full detection + persistence tracking pipeline
            result = process_frame(
                frame,
                all_known_names=all_known_names,
                all_known_encodings=all_known_encodings,
                face_tracker=tracker,
                tolerance=tolerance,
                min_face_size=min_face_size,
            )

            # Send alerts (only for persistent unknown faces)
            for alert in result["alerts"]:
                caption = alert["message"]
                sent = notifier.send_photo(result["frame"], caption=caption,
                                           alert_type=alert["type"])
                if sent:
                    logger.info("ALERT SENT: %s", alert["message"])
                logger.warning("ALERT: %s", alert["message"])

            # FPS overlay (sliding 1-second window)
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                current_fps = fps_counter / (time.time() - fps_timer)
                fps_counter = 0
                fps_timer = time.time()
            cv2.putText(result["frame"], f"FPS: {current_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Face count + tracking overlay
            face_count = len(result["faces"])
            tracking_count = len(tracker.active_tracks)
            if tracking_count > 0:
                cv2.putText(result["frame"], f"Tracking: {tracking_count}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            if face_count > 0:
                cv2.putText(result["frame"], f"Faces: {face_count}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Surveillance", result["frame"])

            if cv2.waitKey(1) & 0xFF == ord("q"):
                _running = False
                break

        # If user quit, don't attempt reconnection
        if not _running:
            break

        # Camera disconnected — attempt reconnection
        logger.warning("Camera connection lost. Attempting reconnection...")
        release(cap, grabber)
        tracker.clear()  # Reset tracker since camera feed was interrupted
        if bot_name:
            notifier.send_text("⚠️ Camera disconnected. Reconnecting...")
        time.sleep(2)
        cap = connect(rtsp_url)
        if cap is None:
            logger.error("Reconnection failed after retries. Exiting.")
            if bot_name:
                notifier.send_text("🔴 Camera disconnected. Reconnection failed.")
            break
        grabber = LatestFrameGrabber(cap)
        logger.info("Camera reconnected successfully.")
        if bot_name:
            notifier.send_text("🔄 Camera reconnected.")

    # --- 7. Cleanup ---
    logger.info("Shutting down...")
    release(cap, grabber)
    cv2.destroyAllWindows()

    if bot_name:
        notifier.send_text("🔴 Surveillance system stopped.")

    logger.info("System stopped cleanly.")


if __name__ == "__main__":
    main()
