import os
import json
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

ZONE_FILE = "zone_config/zone.json"
ROI_FILE = "zone_config/roi.json"


def _mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks to add polygon vertices."""
    points = param["points"]
    frame = param["frame"]
    drawing_frame = param["drawing_frame"]
    window_name = param["window_name"]

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        logger.info("  Point %d added: (%d, %d)", len(points), x, y)

        # Redraw the frame with all points and lines
        drawing_frame[:] = frame[:]
        _draw_polygon_on_frame(drawing_frame, points, closed=False)
        cv2.imshow(window_name, drawing_frame)


def _draw_polygon_on_frame(frame, points, closed=True):
    """Draw polygon points and lines on a frame."""
    if not points:
        return

    pts = np.array(points, dtype=np.int32)

    # Draw lines between consecutive points
    if len(points) > 1:
        cv2.polylines(frame, [pts], isClosed=closed, color=(0, 0, 255), thickness=2)

    # Draw circles at each vertex
    for i, (px, py) in enumerate(points):
        cv2.circle(frame, (px, py), 6, (0, 255, 0), -1)
        cv2.putText(frame, str(i + 1), (px + 10, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # If closed, fill with transparent overlay
    if closed and len(points) >= 3:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 0, 255, 50))
        cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)


def draw_zone(frame, title="DRAW RESTRICTED ZONE", hint="Click to add points | 'u' = undo | Enter = done | Esc = cancel"):
    """
    Display a frame and let the user click to define a polygon zone.

    Controls:
        - Left click: Add a vertex
        - 'u': Undo last point
        - Enter: Finalize the polygon (minimum 3 points)
        - 'c': Cancel and return None
        - Esc: Cancel and return None

    Returns:
        list of (x, y) tuples defining the polygon, or None if cancelled.
    """
    drawing_frame = frame.copy()
    window_name = title
    param = {"points": [], "frame": frame.copy(), "drawing_frame": drawing_frame, "window_name": window_name}
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, _mouse_callback, param)

    # Show instructions on the frame
    instructions = [title, hint]
    for i, text in enumerate(instructions):
        cv2.putText(drawing_frame, text, (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow(window_name, drawing_frame)

    print("\n--- ZONE DRAWING MODE ---")
    print("Click on the video to add polygon vertices.")
    print("Press 'u' to undo last point.")
    print("Press Enter when done (minimum 3 points).")
    print("Press Esc or 'c' to cancel.\n")

    while True:
        key = cv2.waitKey(50) & 0xFF

        if key == 27 or key == ord("c"):  # Esc or 'c' = cancel
            logger.info("Zone drawing cancelled.")
            cv2.destroyWindow(window_name)
            return None

        elif key == 13 or key == 10:  # Enter = finalize
            if len(param["points"]) >= 3:
                logger.info("Zone finalized with %d points.", len(param["points"]))
                cv2.destroyWindow(window_name)
                return param["points"]
            else:
                print("Need at least 3 points. Keep clicking.")

        elif key == ord("u"):  # Undo last point
            if param["points"]:
                removed = param["points"].pop()
                logger.info("  Undid point: (%d, %d)", removed[0], removed[1])
                # Redraw
                drawing_frame[:] = frame[:]
                _draw_polygon_on_frame(drawing_frame, param["points"], closed=False)
                # Re-add instructions
                for i, text in enumerate(instructions):
                    cv2.putText(drawing_frame, text, (10, 30 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow(window_name, drawing_frame)


def save_zone(points):
    """Save polygon points to zone_config/zone.json."""
    os.makedirs(os.path.dirname(ZONE_FILE), exist_ok=True)
    data = {"zone_points": [list(p) for p in points]}
    with open(ZONE_FILE, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Zone saved to %s (%d points)", ZONE_FILE, len(points))


def load_zone():
    """
    Load polygon points from zone_config/zone.json.

    Returns:
        list of (x, y) tuples, or None if file doesn't exist.
    """
    if not os.path.exists(ZONE_FILE):
        return None
    with open(ZONE_FILE, "r") as f:
        data = json.load(f)
    points = [tuple(p) for p in data["zone_points"]]
    logger.info("Zone loaded from %s (%d points)", ZONE_FILE, len(points))
    return points


def save_roi(points):
    """Save monitoring ROI polygon to zone_config/roi.json."""
    os.makedirs(os.path.dirname(ROI_FILE), exist_ok=True)
    data = {"roi_points": [list(p) for p in points]}
    with open(ROI_FILE, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("ROI saved to %s (%d points)", ROI_FILE, len(points))


def load_roi():
    """
    Load monitoring ROI polygon from zone_config/roi.json.

    Returns:
        list of (x, y) tuples, or None if file doesn't exist.
    """
    if not os.path.exists(ROI_FILE):
        return None
    with open(ROI_FILE, "r") as f:
        data = json.load(f)
    points = [tuple(p) for p in data["roi_points"]]
    logger.info("ROI loaded from %s (%d points)", ROI_FILE, len(points))
    return points


def is_point_in_zone(point, zone_points):
    """
    Check if a point is inside the zone polygon.

    Args:
        point: (x, y) tuple to test.
        zone_points: list of (x, y) tuples defining the polygon.

    Returns:
        True if point is inside the polygon.
    """
    if not zone_points or len(zone_points) < 3:
        return False
    contour = np.array(zone_points, dtype=np.int32)
    result = cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False)
    return result >= 0  # >= 0 means inside or on edge


def draw_zone_overlay(frame, zone_points):
    """Draw the zone polygon as a semi-transparent overlay on a frame."""
    if not zone_points or len(zone_points) < 3:
        return
    _draw_polygon_on_frame(frame, zone_points, closed=True)


# --- Standalone test ---
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.config import get_config, get_rtsp_url
    from src.camera_feed import connect, release

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Check if --rezone flag is passed
    rezone = "--rezone" in sys.argv

    # Try loading existing zone
    zone = None
    if not rezone:
        zone = load_zone()

    if zone:
        print(f"Loaded existing zone with {len(zone)} points.")
        print("Run with --rezone to redraw.")
    else:
        # Connect to camera and grab a frame for drawing
        config = get_config()
        rtsp_url = get_rtsp_url(config)
        cap = connect(rtsp_url)
        if cap is None:
            print("Could not connect to camera.")
            sys.exit(1)

        ret, frame = cap.read()
        release(cap)

        if not ret:
            print("Could not read frame from camera.")
            sys.exit(1)

        zone = draw_zone(frame)
        if zone:
            save_zone(zone)
            print(f"\nZone saved with {len(zone)} points: {zone}")
        else:
            print("\nZone drawing cancelled.")
            sys.exit(0)

    # Test point-in-zone
    print(f"\nZone polygon: {zone}")
    test_point = zone[0]  # first vertex should be inside/on edge
    print(f"Test point {test_point} in zone: {is_point_in_zone(test_point, zone)}")
