"""
image_preprocessor.py - Backward-compatible alias for unified preprocessor.py.
"""
from backend.preprocessing.preprocessor import (
    load_image_from_bytes,
    resize_for_inference,
    basic_quality_check,
    TARGET_IMAGE_SIZE as TARGET_SIZE,
)
