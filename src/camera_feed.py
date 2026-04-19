import cv2
import time
import logging
import threading

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


def connect(rtsp_url, retries=MAX_RETRIES):
    """
    Open an RTSP video stream with retry logic and low-latency settings.

    Returns a cv2.VideoCapture object or None if all retries fail.
    """
    for attempt in range(1, retries + 1):
        logger.info("Connecting to camera (attempt %d/%d)...", attempt, retries)

        # Use default backend (CAP_FFMPEG may not support RTSP on all builds)
        cap = cv2.VideoCapture(rtsp_url)

        if cap.isOpened():
            # Minimize internal buffer
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Read one test frame to confirm the stream is live
            ret, _ = cap.read()
            if ret:
                logger.info("Camera connected successfully.")
                return cap
            else:
                logger.warning("Camera opened but failed to read frame.")
                cap.release()
        else:
            logger.warning("Failed to open camera stream.")

        if attempt < retries:
            logger.info("Retrying in %d seconds...", RETRY_DELAY)
            time.sleep(RETRY_DELAY)

    logger.error("Could not connect to camera after %d attempts.", retries)
    return None


class LatestFrameGrabber:
    """
    Threaded frame grabber that always provides the latest camera frame.

    A background thread continuously reads from the VideoCapture, discarding
    old frames. The main thread always gets the most recent frame via read().
    This eliminates the 1-2 minute delay caused by OpenCV's internal buffer.
    """

    def __init__(self, cap):
        self._cap = cap
        self._frame = None
        self._ret = False
        self._lock = threading.Lock()
        self._stopped = False
        self._fail_count = 0
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()
        logger.info("Threaded frame grabber started.")

    def _grab_loop(self):
        """Continuously read frames, keeping only the latest."""
        while not self._stopped:
            ret, frame = self._cap.read()
            if not ret:
                self._fail_count += 1
                if self._fail_count <= 3 or self._fail_count % 100 == 0:
                    logger.warning("Frame grab failed (%d consecutive failures).", self._fail_count)
                time.sleep(0.05)
                continue
            self._fail_count = 0
            with self._lock:
                self._ret = ret
                self._frame = frame

    def read(self):
        """Return the most recent frame (thread-safe)."""
        with self._lock:
            return self._ret, self._frame.copy() if self._frame is not None else None

    def is_healthy(self, max_failures=150):
        """Check if the grabber is still receiving frames (~7.5s at 0.05s sleep)."""
        return not self._stopped and self._fail_count < max_failures

    def stop(self):
        """Stop the background thread."""
        self._stopped = True
        self._thread.join(timeout=3)
        logger.info("Threaded frame grabber stopped.")


def latest_frame_generator(grabber):
    """
    Generator that yields the latest frame from a LatestFrameGrabber.

    Unlike frame_generator(), this never falls behind — every frame
    yielded is the most recent one available from the camera.
    """
    while True:
        if not grabber.is_healthy():
            logger.warning("Frame grabber unhealthy — stopping generator.")
            return
        ret, frame = grabber.read()
        if not ret or frame is None:
            # Brief pause before retry — the grabber thread is still running
            time.sleep(0.01)
            continue
        yield frame


def frame_generator(cap):
    """
    Generator that yields frames from a VideoCapture object.

    NOTE: This reads sequentially from the buffer and WILL fall behind
    on slow processing. Prefer LatestFrameGrabber for live surveillance.

    Yields:
        numpy.ndarray: BGR frame from the camera.

    Stops when the stream ends or an error occurs.
    """
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to read frame from camera.")
            break
        yield frame


def release(cap, grabber=None):
    """Release camera resources and stop the frame grabber."""
    if grabber is not None:
        grabber.stop()
    if cap is not None:
        cap.release()
        logger.info("Camera released.")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.config import get_config, get_rtsp_url

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = get_config()
    rtsp_url = get_rtsp_url(config)
    logger.info("RTSP URL: rtsp://%s:****@%s:554/%s",
                config["camera_user"], config["camera_ip"], config["camera_stream"])

    cap = connect(rtsp_url)
    if cap is None:
        logger.error("Exiting — could not connect to camera.")
        sys.exit(1)

    print("\nCamera feed is live (threaded grabber). Press 'q' to quit.\n")

    grabber = LatestFrameGrabber(cap)
    fps_start = time.time()
    frame_count = 0

    for frame in latest_frame_generator(grabber):
        frame_count += 1

        # Calculate and display FPS every 30 frames
        if frame_count % 30 == 0:
            elapsed = time.time() - fps_start
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Tapo C210 Live Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    release(cap, grabber)
    cv2.destroyAllWindows()
    print(f"Displayed {frame_count} frames.")
