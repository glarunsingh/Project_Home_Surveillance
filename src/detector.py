import logging
import cv2
import numpy as np
import face_recognition

logger = logging.getLogger(__name__)

# Minimum number of facial landmarks required to consider a detection valid.
# A real face has 68 landmarks. We check for key groups: eyes, nose, mouth.
REQUIRED_LANDMARK_GROUPS = {"left_eye", "right_eye", "nose_bridge", "top_lip"}

# Tracking status colors
COLOR_TRACKING = (0, 200, 255)   # Orange — face being tracked, not yet persistent
COLOR_KNOWN = (0, 255, 0)        # Green — known person (persistent)
COLOR_UNKNOWN = (0, 0, 255)      # Red — unknown person (persistent, alerted)


def _is_valid_face(rgb_frame, face_location):
    """
    Validate a detected face using facial landmark analysis.
    Rejects false positives (bikes, objects) that lack real facial features.

    Args:
        rgb_frame: RGB frame.
        face_location: (top, right, bottom, left) tuple.

    Returns:
        True if the detection has real facial landmarks.
    """
    landmarks_list = face_recognition.face_landmarks(rgb_frame, [face_location])
    if not landmarks_list:
        return False

    landmarks = landmarks_list[0]
    # Check that key facial feature groups are present
    found_groups = set(landmarks.keys())
    missing = REQUIRED_LANDMARK_GROUPS - found_groups
    if missing:
        logger.debug("Rejected face — missing landmarks: %s", missing)
        return False

    return True


def _face_size(face_location):
    """Return (width, height) of a face bounding box."""
    top, right, bottom, left = face_location
    return (right - left, bottom - top)


def detect_faces(frame, model="hog", min_face_size=40, scale=0.5):
    """
    Detect face locations and compute encodings.

    Uses frame downscaling for faster detection, then computes encodings
    on the full-resolution frame for accuracy.

    Args:
        frame: BGR OpenCV frame.
        model: 'hog' (fast, CPU) or 'cnn' (accurate, GPU).
        min_face_size: Minimum width/height in pixels to accept a face.
        scale: Downscale factor for detection (0.5 = half resolution). 1.0 = no scaling.

    Returns:
        (face_locations, face_encodings) — only validated faces.
    """
    rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Downscale for faster face detection
    if scale < 1.0:
        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        raw_locations = face_recognition.face_locations(rgb_small, model=model)
        # Scale locations back to original resolution
        inv = 1.0 / scale
        raw_locations = [
            (int(t * inv), int(r * inv), int(b * inv), int(l * inv))
            for t, r, b, l in raw_locations
        ]
    else:
        raw_locations = face_recognition.face_locations(rgb_full, model=model)

    if not raw_locations:
        return [], []

    # Filter by size (cheap check)
    valid_locations = []
    for loc in raw_locations:
        w, h = _face_size(loc)
        if w < min_face_size or h < min_face_size:
            logger.debug("Rejected face — too small: %dx%d (min %d)", w, h, min_face_size)
            continue
        valid_locations.append(loc)

    if not valid_locations:
        return [], []

    # Compute encodings on full resolution for accuracy
    encodings = face_recognition.face_encodings(rgb_full, valid_locations)
    logger.info("Detected %d face(s) out of %d candidate(s)",
                len(valid_locations), len(raw_locations))
    return valid_locations, encodings


def build_recognition_data(encodings_by_person):
    """
    Pre-compute flat arrays for face recognition. Call once at startup.

    Args:
        encodings_by_person: Dict mapping name -> list of 128-d encoding arrays.

    Returns:
        (all_names, all_encodings) — flat lists for efficient per-frame matching.
    """
    all_encodings = []
    all_names = []
    for name, encs in encodings_by_person.items():
        for enc in encs:
            all_encodings.append(enc)
            all_names.append(name)
    return all_names, all_encodings


def recognize_faces(face_encodings, all_known_names, all_known_encodings, tolerance=0.5):
    """
    Match detected face encodings against known faces.

    Args:
        face_encodings: List of 128-d encodings from the current frame.
        all_known_names: Pre-computed flat name list (from build_recognition_data).
        all_known_encodings: Pre-computed flat encoding list (from build_recognition_data).
        tolerance: Distance threshold (lower = stricter).

    Returns:
        List of name strings (one per face). "Unknown" if no match.
    """
    if not all_known_names or not all_known_encodings:
        return ["Unknown"] * len(face_encodings)

    results = []
    for face_enc in face_encodings:
        distances = face_recognition.face_distance(all_known_encodings, face_enc)
        if len(distances) == 0:
            results.append("Unknown")
            continue

        best_idx = np.argmin(distances)
        if distances[best_idx] <= tolerance:
            results.append(all_known_names[best_idx])
        else:
            results.append("Unknown")

    return results


def annotate_frame(frame, active_tracks, persistence_seconds):
    """
    Draw bounding boxes with smooth color transitions.

    Color flow:
        - Orange: New face, identity not yet confirmed
        - Green: Confirmed known person
        - Red: Confirmed unknown person
    Extra for unknowns not yet alerted:
        - Progress bar showing time to alert threshold

    Args:
        frame: BGR frame to annotate (modified in-place).
        active_tracks: List of TrackedFace objects from FaceTracker.
        persistence_seconds: Threshold in seconds (for progress display).

    Returns:
        The annotated frame.
    """
    for track in active_tracks:
        top, right, bottom, left = track.last_location
        duration = track.duration

        # Color flow: orange → green/red once confirmed
        if not track.confirmed:
            color = COLOR_TRACKING
            label = "Detecting..."
        elif track.name == "Unknown":
            color = COLOR_UNKNOWN
            label = "UNKNOWN"
        else:
            color = COLOR_KNOWN
            label = track.name

        # Add timer to label once confirmed
        if track.confirmed and duration >= 1.0:
            label += f" ({duration:.0f}s)"

        # Bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Progress bar for unknowns not yet alerted (shows time-to-alert)
        if track.confirmed and track.name == "Unknown" and not track.alerted \
                and duration < persistence_seconds:
            bar_width = right - left
            if bar_width > 0:
                progress = min(duration / persistence_seconds, 1.0)
                filled = int(bar_width * progress)
                cv2.rectangle(frame, (left, bottom + 2), (left + filled, bottom + 6),
                              COLOR_UNKNOWN, -1)
                cv2.rectangle(frame, (left, bottom + 2), (right, bottom + 6),
                              (100, 100, 100), 1)

        # Name label background
        label_y = top - 10 if top - 10 > 10 else bottom + 20
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (left, label_y - h - 4), (left + w + 4, label_y + 4), color, -1)
        cv2.putText(frame, label, (left + 2, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return frame


def process_frame(frame, all_known_names, all_known_encodings, face_tracker,
                  tolerance=0.5, model="hog", min_face_size=40, scale=0.5):
    """
    Full per-frame detection + persistence tracking pipeline.

    Detects faces, recognizes them, feeds results to the FaceTracker,
    and only generates alerts for faces that persist beyond the threshold.

    Args:
        frame: BGR frame from camera.
        all_known_names: Pre-computed flat name list (from build_recognition_data).
        all_known_encodings: Pre-computed flat encoding list (from build_recognition_data).
        face_tracker: FaceTracker instance for persistence tracking.
        tolerance: Face match tolerance.
        model: Detection model ('hog' or 'cnn').
        min_face_size: Minimum face pixel size to accept.
        scale: Downscale factor for detection (0.5 = half resolution).

    Returns:
        dict with keys:
            - 'frame': Annotated frame.
            - 'faces': List of dicts with 'name', 'location', 'duration'.
            - 'alerts': List of alert dicts (only for persistent unknown faces).
    """
    face_locations, face_encodings = detect_faces(frame, model=model,
                                                   min_face_size=min_face_size,
                                                   scale=scale)
    names = recognize_faces(face_encodings, all_known_names, all_known_encodings, tolerance)

    # Update persistence tracker — returns only NEW threshold crossings
    new_alerts_tracks = face_tracker.update(face_locations, face_encodings, names)

    # Annotate frame with all active tracks
    annotated = annotate_frame(frame, face_tracker.active_tracks,
                               face_tracker.persistence_seconds)

    # Build face list from active tracks
    faces = []
    for track in face_tracker.active_tracks:
        faces.append({
            "name": track.name,
            "location": track.last_location,
            "duration": track.duration,
        })

    # Generate alerts only for persistent unknowns
    alerts = []
    for track in new_alerts_tracks:
        if track.name == "Unknown":
            alerts.append({
                "type": "unknown_persistent",
                "message": f"🚨 Unknown person lingering for {track.duration:.0f}s!",
                "face": {"name": track.name, "location": track.last_location,
                          "duration": track.duration},
            })
        else:
            # Known person persisted — just log, no alert
            logger.info("Known person confirmed: %s (present %.1fs)",
                        track.name, track.duration)

    return {
        "frame": annotated,
        "faces": faces,
        "alerts": alerts,
    }
