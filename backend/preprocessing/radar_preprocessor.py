"""
radar_preprocessor.py - Backward-compatible alias for unified preprocessor.py.
"""
from backend.preprocessing.preprocessor import (
    load_nuscenes_radar,
    get_radar_files,
    empty_radar_dataframe,
    RADAR_COLUMNS,
)
