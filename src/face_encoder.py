import os
import json
import logging
import numpy as np
import face_recognition

logger = logging.getLogger(__name__)

ENCODINGS_NPZ = "known_faces/encodings.npz"
NAMES_JSON = "known_faces/names.json"


def _get_folder_mtime(known_faces_dir):
    """Get the latest modification time across all files in known_faces/."""
    latest = 0
    for root, dirs, files in os.walk(known_faces_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                mtime = os.path.getmtime(os.path.join(root, f))
                if mtime > latest:
                    latest = mtime
    return latest


def _cache_is_valid(known_faces_dir):
    """Check if the cache files exist and are newer than all face images."""
    if not os.path.exists(ENCODINGS_NPZ) or not os.path.exists(NAMES_JSON):
        return False
    cache_mtime = min(os.path.getmtime(ENCODINGS_NPZ), os.path.getmtime(NAMES_JSON))
    folder_mtime = _get_folder_mtime(known_faces_dir)
    return cache_mtime > folder_mtime


def load_known_faces(known_faces_dir="known_faces"):
    """
    Scan known_faces/<PersonName>/ subfolders, compute 128-d face encodings,
    and return a dict mapping name -> list of encoding arrays.

    Results are cached to encodings.pkl for fast subsequent loads.
    """
    # Try loading from cache first
    if _cache_is_valid(known_faces_dir):
        logger.info("Loading face encodings from cache...")
        with open(NAMES_JSON, "r") as f:
            meta = json.load(f)
        npz_data = np.load(ENCODINGS_NPZ, allow_pickle=False)
        known_names = meta["names"]
        encodings_by_person = {}
        idx = 0
        for name, count in meta["person_counts"].items():
            encodings_by_person[name] = [npz_data["encodings"][i] for i in range(idx, idx + count)]
            idx += count
        total = sum(len(v) for v in encodings_by_person.values())
        logger.info(
            "Loaded %d encoding(s) for %d person(s) from cache",
            total,
            len(encodings_by_person),
        )
        return known_names, encodings_by_person

    # Compute fresh encodings
    logger.info("Computing face encodings from: %s", known_faces_dir)
    known_names = []  # flat list: ["John", "John", "Jane", ...]
    known_encodings = []  # flat list of 128-d arrays
    encodings_by_person = {}  # {"John": [enc1, enc2], "Jane": [enc1]}

    if not os.path.isdir(known_faces_dir):
        logger.warning("Known faces directory not found: %s", known_faces_dir)
        return known_names, encodings_by_person

    for person_name in sorted(os.listdir(known_faces_dir)):
        person_dir = os.path.join(known_faces_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        person_encodings = []
        for img_file in sorted(os.listdir(person_dir)):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                continue

            img_path = os.path.join(person_dir, img_file)
            logger.info("  Processing: %s", img_path)

            image = face_recognition.load_image_file(img_path)
            face_encs = face_recognition.face_encodings(image)

            if len(face_encs) == 0:
                logger.warning("  No face found in %s — skipping", img_path)
                continue
            if len(face_encs) > 1:
                logger.warning(
                    "  Multiple faces in %s — using first face only", img_path
                )

            encoding = face_encs[0]
            known_names.append(person_name)
            known_encodings.append(encoding)
            person_encodings.append(encoding)

        encodings_by_person[person_name] = person_encodings
        logger.info(
            "  %s: %d encoding(s) computed", person_name, len(person_encodings)
        )

    # Save cache (numpy + JSON — no pickle for security)
    cache_dir = os.path.dirname(ENCODINGS_NPZ)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(ENCODINGS_NPZ, encodings=np.array(known_encodings))
    meta = {
        "names": known_names,
        "person_counts": {name: len(encs) for name, encs in encodings_by_person.items()},
    }
    with open(NAMES_JSON, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(
        "Cached %d encoding(s) for %d person(s)",
        len(known_names),
        len(encodings_by_person),
    )

    return known_names, encodings_by_person


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    names, encodings = load_known_faces()
    if not encodings:
        print("\nNo faces found. Add photos to known_faces/<PersonName>/ subfolders.")
    else:
        print(f"\nEnrolled {len(encodings)} person(s):")
        for name, encs in encodings.items():
            print(f"  {name}: {len(encs)} photo(s)")
