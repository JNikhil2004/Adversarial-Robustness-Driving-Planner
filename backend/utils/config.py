"""
Central configuration for the Adversarial Robustness Driving Planner backend.

All important thresholds and decision weights are kept here so the
robustness framework remains auditable and easy to tune.
"""

from pathlib import Path


# ===========================================================================
# PATHS
# ===========================================================================

BASE_DIR = Path(__file__).resolve().parents[2]

BACKEND_DIR = BASE_DIR / "backend"
FRONTEND_DIR = BASE_DIR / "frontend"
DATASETS_DIR = BASE_DIR / "datasets"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = BACKEND_DIR / "models"

TALK2CAR_DIR = DATASETS_DIR / "talk2car"
NUSCENES_DIR = DATASETS_DIR / "nuscenes"

PROMPT_MODEL_PATH = MODELS_DIR / "prompt_svm_pipeline.joblib"
DEEP_PROMPT_MODEL_PATH = MODELS_DIR / "deep_prompt_model.pt"
RADAR_AUTOENCODER_PATH = MODELS_DIR / "radar_autoencoder.pt"
DRIVING_PLANNER_PATH = MODELS_DIR / "driving_planner.pt"

TRANSFORMER_PROMPT_PATH = MODELS_DIR / "transformer_prompt.pt"
SEMANTIC_GROUNDABILITY_PATH = MODELS_DIR / "semantic_groundability.pt"
RADAR_TEMPORAL_AE_PATH = MODELS_DIR / "radar_temporal_ae.pt"
VL_ALIGNMENT_PATH = MODELS_DIR / "vl_alignment.pt"
MULTIMODAL_FUSER_PATH = MODELS_DIR / "multimodal_fuser.pt"

YOLO_WEIGHTS = str(
    MODELS_DIR / "yolov8n.pt"
)

for _path in (OUTPUTS_DIR, MODELS_DIR):
    _path.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# PROMPT ROBUSTNESS
# ===========================================================================

PROMPT_LABELS = [
    "trusted",
    "suspicious",
    "high_risk",
]


# ===========================================================================
# RADAR VALIDATION
# ===========================================================================

# Broad physical plausibility limits.
# Radar attacks should manipulate values relative to the actual radar data,
# rather than repeatedly assigning fixed constants.

RADAR_MAX_RANGE_M = 100.0
RADAR_MIN_RANGE_M = 0.3

RADAR_MAX_SPEED_MPS = 45.0

# Temporal consistency limits.
RADAR_MAX_RANGE_JUMP_M = 25.0
RADAR_MAX_SPEED_JUMP_MPS = 20.0

# Radar cross-section limits.
RADAR_MIN_RCS_DB = -25.0
RADAR_MAX_RCS_DB = 40.0


# ===========================================================================
# AMBIGUITY DETECTOR
# ===========================================================================

VAGUE_REFERENTS = {
    "there",
    "here",
    "that",
    "it",
    "this",
    "somewhere",
    "anywhere",
    "thing",
    "stuff",
    "wherever",
    "whatever",
    "someplace",
}

VAGUE_QUANTIFIERS = {
    "maybe",
    "probably",
    "i guess",
    "sort of",
    "kind of",
    "whatever you think",
}

MIN_INSTRUCTION_TOKENS = 3


# ===========================================================================
# CAMERA / SCENE UNDERSTANDING
# ===========================================================================

YOLO_CONFIDENCE_THRESHOLD = 0.35

COCO_TO_DRIVING_CLASSES = {
    "person": "pedestrian",
    "bicycle": "cyclist",
    "car": "car",
    "motorcycle": "motorcyclist",
    "bus": "bus",
    "truck": "truck",
    "traffic light": "traffic_light",
    "stop sign": "stop_sign",
}


# ===========================================================================
# DECISION ENGINE WEIGHTS
# ===========================================================================

"""
All five channels participate in the final safety score.

Prompt       -> instruction safety
Ambiguity    -> instruction clarity
Camera       -> visual scene reliability
Radar        -> radar-data reliability
Consistency  -> cross-modal agreement

Weights must sum to 1.0.
"""

DECISION_WEIGHTS = {
    "prompt": 0.25,
    "ambiguity": 0.15,
    "camera": 0.25,
    "radar": 0.20,
    "consistency": 0.15,
}


# ===========================================================================
# FINAL SAFETY THRESHOLDS
# ===========================================================================

# >= 75 -> Safe to Execute
# 45-74 -> Require Human Confirmation
# < 45  -> Execution Blocked

SAFETY_SCORE_SAFE_THRESHOLD = 75.0
SAFETY_SCORE_CONFIRM_THRESHOLD = 45.0


# ===========================================================================
# RISK LEVEL THRESHOLDS
# ===========================================================================

RISK_LEVEL_LOW_THRESHOLD = 75.0
RISK_LEVEL_MEDIUM_THRESHOLD = 45.0


# ===========================================================================
# HARD SAFETY CONDITIONS
# ===========================================================================

"""
These conditions override the weighted average.

A high-risk passenger instruction must never become safe simply because
camera/radar happen to have high scores.
"""

HARD_BLOCK_PROMPT_LABELS = {
    "high_risk",
}


# ===========================================================================
# LOW-SCORE CHANNEL THRESHOLDS
# ===========================================================================

PROMPT_LOW_SCORE = 45.0
AMBIGUITY_LOW_SCORE = 45.0
CAMERA_LOW_SCORE = 45.0
RADAR_LOW_SCORE = 45.0
CONSISTENCY_LOW_SCORE = 45.0


# ===========================================================================
# MULTI-CHANNEL DEGRADATION
# ===========================================================================

"""
If multiple independent channels are simultaneously degraded, the
decision engine records this as an additional safety concern.

This does NOT artificially subtract a fixed number of points.
The actual channel scores already determine the weighted safety score.
"""

MULTI_CHANNEL_DEGRADATION_LIMIT = 2
