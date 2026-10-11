"""
preprocessor.py
---------------
Unified preprocessing pipeline for:
  1. Text instructions (normalisation, tokenization, object keyword extraction)
  2. Camera frames (OpenCV decoding, letterbox resizing, lens quality assessment)
  3. Radar point clouds (nuScenes .pcd loading, coordinate transform, kinematic feature extraction)
"""

import re
import string
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
import pandas as pd

from backend.utils.logger import get_logger

logger = get_logger(__name__)

# ==============================================================================
# 1. TEXT PREPROCESSING
# ==============================================================================

_STOPWORDS = {
    "a", "an", "the", "to", "of", "in", "on", "at", "for", "and", "is",
    "are", "was", "were", "be", "please", "just", "up", "over",
}

import unicodedata

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

_CYRILLIC_TO_LATIN = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "і": "i", "х": "x", "у": "y",
    "А": "A", "Е": "E", "О": "O", "Р": "P", "С": "C", "І": "I", "Х": "X", "У": "Y",
})
_ZERO_WIDTH_CHARS = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\ufeff]")

DRIVING_OBJECT_VOCABULARY = {
    "car", "van", "truck", "bus", "suv", "sedan", "taxi", "bicycle",
    "bike", "cyclist", "pedestrian", "person", "man", "woman", "child",
    "motorcycle", "motorcyclist", "sign", "stop sign", "traffic light",
    "crosswalk", "pedestrian crossing", "trash can", "storefront",
    "coffee shop", "parking spot", "intersection", "curb",
}


def clean_text(text: str) -> str:
    """Lowercase, strip zero-width chars, normalize homoglyphs, strip punctuation and collapse whitespace."""
    if not isinstance(text, str):
        return ""
    # Strip zero-width and invisible formatting characters
    text = _ZERO_WIDTH_CHARS.sub("", text)
    # Translate Cyrillic homoglyphs and apply NFKD decomposition
    text = text.translate(_CYRILLIC_TO_LATIN)
    text = unicodedata.normalize("NFKD", text)
    text = text.lower().strip()
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text: str, remove_stopwords: bool = False) -> List[str]:
    """Split cleaned text into tokens, optionally dropping stopwords."""
    cleaned = clean_text(text)
    tokens = cleaned.split(" ") if cleaned else []
    if remove_stopwords:
        tokens = [t for t in tokens if t and t not in _STOPWORDS]
    return [t for t in tokens if t]


def extract_object_keywords(text: str) -> List[str]:
    """Extract driving object keywords from instruction for cross-modal verification."""
    tokens = tokenize(text)
    found = set()
    joined = " ".join(tokens)
    for phrase in sorted(DRIVING_OBJECT_VOCABULARY, key=len, reverse=True):
        if phrase in joined:
            found.add(phrase)
    return sorted(found)


# ==============================================================================
# 2. CAMERA IMAGE PREPROCESSING
# ==============================================================================

TARGET_IMAGE_SIZE: Tuple[int, int] = (640, 640)


def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode raw uploaded bytes into a BGR OpenCV image array."""
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("cv2.imdecode returned None -- unsupported or corrupted image format")
        return img
    except Exception as exc:
        logger.error(f"Failed to decode uploaded image: {exc}")
        raise


def resize_for_inference(image: np.ndarray, size: Tuple[int, int] = TARGET_IMAGE_SIZE) -> np.ndarray:
    """Resize while preserving aspect ratio via letterboxing."""
    h, w = image.shape[:2]
    target_w, target_h = size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def basic_quality_check(image: np.ndarray) -> Dict[str, Any]:
    """Check optical blur and exposure independent of model predictions."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean_brightness = float(np.mean(gray))
    return {
        "sharpness": round(float(laplacian_var), 1),
        "mean_brightness": round(mean_brightness, 1),
        "is_blurry": bool(laplacian_var < 60.0),
        "is_over_or_under_exposed": bool(mean_brightness < 25 or mean_brightness > 230),
    }


# ==============================================================================
# 3. RADAR POINT-CLOUD PREPROCESSING
# ==============================================================================

RADAR_COLUMNS = [
    "track_id",
    "object_class",
    "range_m",
    "azimuth_deg",
    "relative_speed_mps",
    "rcs_db",
    "timestamp",
]


def empty_radar_dataframe() -> pd.DataFrame:
    """Return an empty DataFrame using the project's radar schema."""
    return pd.DataFrame(columns=RADAR_COLUMNS)


def load_nuscenes_radar(path: Path) -> pd.DataFrame:
    """
    Load a nuScenes radar .pcd file and extract kinematic features:
    [range_m, azimuth_deg, relative_speed_mps, rcs_db, track_id].
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Radar file not found: {path}")

    if path.suffix.lower() != ".pcd":
        raise ValueError(f"Expected a nuScenes .pcd file: {path}")

    try:
        from nuscenes.utils.data_classes import RadarPointCloud

        radar = RadarPointCloud.from_file(str(path))
        points = radar.points

        if points.size == 0:
            return empty_radar_dataframe()

        # nuScenes radar point layout:
        # 0: x, 1: y, 2: z, 4: id, 5: rcs, 8: vx_comp, 9: vy_comp
        x = points[0]
        y = points[1]

        range_m = np.sqrt(x ** 2 + y ** 2)
        azimuth_deg = np.degrees(np.arctan2(y, x))
        vx_comp = points[8]
        vy_comp = points[9]

        safe_range = np.where(range_m > 1e-6, range_m, 1.0)
        relative_speed_mps = (x * vx_comp + y * vy_comp) / safe_range
        rcs_db = points[5]
        track_id = points[4].astype(int)

        dataframe = pd.DataFrame({
            "track_id": track_id,
            "object_class": "unknown",
            "range_m": range_m,
            "azimuth_deg": azimuth_deg,
            "relative_speed_mps": relative_speed_mps,
            "rcs_db": rcs_db,
            "timestamp": 0.0,
        })

        logger.info(f"Loaded nuScenes radar: {path.name} ({len(dataframe)} points)")
        return dataframe

    except Exception as exc:
        logger.error(f"Failed to load nuScenes radar {path}: {exc}")
        raise


def get_radar_files(radar_directory: Path) -> List[Path]:
    """Return all nuScenes RADAR_FRONT files in a directory."""
    radar_directory = Path(radar_directory)
    if not radar_directory.exists():
        logger.warning(f"RADAR_FRONT directory not found: {radar_directory}")
        return []
    return sorted(radar_directory.glob("*.pcd"))
