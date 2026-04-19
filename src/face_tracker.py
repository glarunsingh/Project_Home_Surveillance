"""
Face Persistence Tracker — Only alert on faces that linger.

Tracks detected faces across frames using encoding similarity and spatial
proximity.  A face must persist for a configurable duration before triggering
an alert.

Matching strategy (three passes):
    1. Encoding distance — confirmed tracks get a wider tolerance.
    2. Spatial proximity (center-distance) — confirmed tracks matched purely
       by position; no encoding check.  Handles head turns / lighting.
    3. Anti-duplicate — any remaining detection that overlaps a confirmed
       track is absorbed rather than spawning a new track.
"""

import time
import logging
import cv2
import numpy as np
import face_recognition

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _center_dist(loc_a, loc_b):
    """Euclidean distance between the centres of two (top, right, bottom, left) boxes."""
    cy_a = (loc_a[0] + loc_a[2]) / 2.0
    cx_a = (loc_a[3] + loc_a[1]) / 2.0
    cy_b = (loc_b[0] + loc_b[2]) / 2.0
    cx_b = (loc_b[3] + loc_b[1]) / 2.0
    return ((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) ** 0.5


def _box_size(loc):
    """Average of width and height of a (top, right, bottom, left) box."""
    w = abs(loc[1] - loc[3])
    h = abs(loc[2] - loc[0])
    return (w + h) / 2.0


class TrackedFace:
    """A single face being tracked across frames."""

    __slots__ = ("encoding", "name", "first_seen", "last_seen",
                 "detection_count", "alerted", "last_location", "confirmed")

    CONFIRM_THRESHOLD = 2

    def __init__(self, encoding, location, name="Unknown"):
        self.encoding = encoding
        self.name = name
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.detection_count = 1
        self.alerted = False
        self.last_location = location
        self.confirmed = False

    @property
    def duration(self):
        return self.last_seen - self.first_seen

    def update(self, encoding, location, name="Unknown"):
        """Update this tracked face with a new detection."""
        self.encoding = 0.7 * self.encoding + 0.3 * encoding

        # Smooth the box position — responsive enough to follow the face,
        # stable enough to avoid jitter.
        w = 0.7 if self.confirmed else 0.5
        nw = 1.0 - w
        old = self.last_location
        self.last_location = (
            int(w * old[0] + nw * location[0]),
            int(w * old[1] + nw * location[1]),
            int(w * old[2] + nw * location[2]),
            int(w * old[3] + nw * location[3]),
        )

        self.last_seen = time.time()
        self.detection_count += 1

        # Name locking: once a known name is assigned it never reverts.
        if self.name == "Unknown" and name != "Unknown":
            self.name = name

        if self.detection_count >= self.CONFIRM_THRESHOLD:
            self.confirmed = True


class FaceTracker:
    """
    Track faces across frames and alert only on persistent presences.

    Args:
        persistence_seconds: Duration face must be present to trigger alert.
        match_tolerance: Base encoding distance for same-person matching.
        gone_timeout: Seconds without detection before removing a track.
    """

    def __init__(self, persistence_seconds=5.0, match_tolerance=0.55,
                 gone_timeout=8.0):
        self.persistence_seconds = persistence_seconds
        self.match_tolerance = match_tolerance
        self.gone_timeout = gone_timeout
        self._tracked = []
        self._prev_gray_small = None   # for camera motion estimation

    # ------------------------------------------------------------------
    def compensate_camera_motion(self, frame):
        """
        Estimate global camera pan/tilt and shift all tracked positions
        so spatial matching works even when the camera is physically moving.

        Uses phase correlation on downscaled grayscale frames (~1 ms).
        Call once per frame (including skipped frames) BEFORE update().
        """
        h, w = frame.shape[:2]
        small_w, small_h = max(w // 4, 1), max(h // 4, 1)
        small_bgr = cv2.resize(frame, (small_w, small_h))
        small_gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)

        if self._prev_gray_small is not None and self._tracked:
            try:
                shift, response = cv2.phaseCorrelate(
                    np.float64(self._prev_gray_small),
                    np.float64(small_gray),
                )
                # Scale shift back to original resolution
                dx = shift[0] * (w / small_w)
                dy = shift[1] * (h / small_h)

                # Only compensate meaningful, reliable motion (skip noise)
                magnitude = max(abs(dx), abs(dy))
                if response > 0.05 and 3 < magnitude < 500:
                    for t in self._tracked:
                        top, right, bottom, left = t.last_location
                        t.last_location = (
                            int(top + dy), int(right + dx),
                            int(bottom + dy), int(left + dx),
                        )
                    logger.debug(
                        "Camera motion compensated: dx=%.1f dy=%.1f (resp=%.2f)",
                        dx, dy, response,
                    )
            except Exception:
                pass

        self._prev_gray_small = small_gray

    # ------------------------------------------------------------------
    def update(self, face_locations, face_encodings, names):
        """
        Match detections to existing tracks (three-pass).

        Returns list of TrackedFace objects that just crossed the
        persistence threshold (need alerting).  Each returned only once.
        """
        now = time.time()
        matched_tracked = set()
        matched_detected = set()

        # ==============================================================
        # Pass 1 — Encoding-based matching
        #   Confirmed tracks get +0.15 wider tolerance.
        # ==============================================================
        if self._tracked and face_encodings:
            tracked_encodings = [t.encoding for t in self._tracked]
            for det_idx, det_enc in enumerate(face_encodings):
                distances = face_recognition.face_distance(
                    tracked_encodings, det_enc
                )
                for best_idx in np.argsort(distances):
                    best_idx = int(best_idx)
                    if best_idx in matched_tracked:
                        continue
                    thr = self.match_tolerance
                    if self._tracked[best_idx].confirmed:
                        thr += 0.15
                    if distances[best_idx] <= thr:
                        self._tracked[best_idx].update(
                            det_enc, face_locations[det_idx], names[det_idx]
                        )
                        matched_tracked.add(best_idx)
                        matched_detected.add(det_idx)
                        break

        # ==============================================================
        # Pass 2 — Spatial proximity for confirmed tracks
        #   If the face centre is within 1.5× the tracked box size,
        #   it IS the same person — NO encoding check.  This handles
        #   head turns, expressions, and lighting changes.
        # ==============================================================
        SPATIAL_DIST_FACTOR = 1.5  # multiples of box size

        for det_idx in range(len(face_encodings)):
            if det_idx in matched_detected:
                continue
            best_dist = float("inf")
            best_track_idx = -1
            for track_idx, track in enumerate(self._tracked):
                if track_idx in matched_tracked:
                    continue
                if not track.confirmed:
                    continue
                max_dist = _box_size(track.last_location) * SPATIAL_DIST_FACTOR
                dist = _center_dist(face_locations[det_idx], track.last_location)
                if dist < max_dist and dist < best_dist:
                    best_dist = dist
                    best_track_idx = track_idx
            if best_track_idx >= 0:
                self._tracked[best_track_idx].update(
                    face_encodings[det_idx], face_locations[det_idx],
                    names[det_idx]
                )
                matched_tracked.add(best_track_idx)
                matched_detected.add(det_idx)
                logger.debug("Spatial match: %s (dist=%.0fpx)",
                             self._tracked[best_track_idx].name, best_dist)

        # ==============================================================
        # Pass 3 — Anti-duplicate: never spawn a new track on top of a
        #   confirmed track.  Use a generous 2× radius.
        # ==============================================================
        ABSORB_DIST_FACTOR = 2.0

        for det_idx in range(len(face_encodings)):
            if det_idx in matched_detected:
                continue
            for track_idx, track in enumerate(self._tracked):
                if track_idx in matched_tracked:
                    continue
                if not track.confirmed:
                    continue
                max_dist = _box_size(track.last_location) * ABSORB_DIST_FACTOR
                dist = _center_dist(face_locations[det_idx], track.last_location)
                if dist < max_dist:
                    track.update(
                        face_encodings[det_idx], face_locations[det_idx],
                        names[det_idx]
                    )
                    matched_tracked.add(track_idx)
                    matched_detected.add(det_idx)
                    logger.debug("Absorbed duplicate into %s (dist=%.0fpx)",
                                 track.name, dist)
                    break

        # ==============================================================
        # Create new tracks for genuinely new faces
        # ==============================================================
        for det_idx in range(len(face_encodings)):
            if det_idx not in matched_detected:
                new_track = TrackedFace(
                    face_encodings[det_idx], face_locations[det_idx],
                    names[det_idx]
                )
                self._tracked.append(new_track)
                logger.info("New face — tracking started (%s, total: %d)",
                            names[det_idx], len(self._tracked))

        # Progress logging
        for t in self._tracked:
            if not t.alerted and t.duration > 0.5:
                logger.info("Tracking: %s — %.1fs / %.1fs",
                            t.name, t.duration, self.persistence_seconds)

        # ==============================================================
        # Stale removal — confirmed tracks survive 2× longer
        # ==============================================================
        def _is_stale(t):
            timeout = self.gone_timeout * (2.0 if t.confirmed else 1.0)
            return (now - t.last_seen) >= timeout

        stale = [t for t in self._tracked if _is_stale(t)]
        self._tracked = [t for t in self._tracked if not _is_stale(t)]
        for t in stale:
            logger.info("Face lost — %s tracked %.1fs", t.name, t.duration)
        if stale:
            logger.info("Removed %d stale track(s) (remaining: %d)",
                        len(stale), len(self._tracked))

        # ==============================================================
        # Persistence alerts
        # ==============================================================
        new_alerts = []
        for t in self._tracked:
            if not t.alerted and t.duration >= self.persistence_seconds:
                t.alerted = True
                new_alerts.append(t)
                logger.info("Persisted %.1fs — alerting: %s", t.duration, t.name)

        return new_alerts

    @property
    def active_tracks(self):
        return list(self._tracked)

    def clear(self):
        self._tracked.clear()
        self._prev_gray_small = None
