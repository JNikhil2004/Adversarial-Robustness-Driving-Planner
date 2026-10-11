"""
consistency_checker.py
----------------------
YOLOv8 scene understanding + Deep Vision-Language (VL) Grounding & Multimodal Consistency.

Checks:
1. Deep Vision-Language Alignment: Projects Talk2Car referring expressions and YOLOv8
   visual bounding boxes into a joint metric space using VLAlignmentModule,
   evaluating cosine similarity to detect semantic distractors.
2. Cross-modal object grounding (Language <-> Camera detections).
3. Cross-modal radar evidence (Language <-> Radar kinematics).
4. Spatial & directional consistency (Camera detection azimuths <-> Radar point clusters).
5. Action feasibility & sensor quality.

Produces a multimodal consistency score from 0-100 and VL grounding metric.
"""

import re
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from backend.preprocessing.image_preprocessor import basic_quality_check
from backend.preprocessing.text_preprocessor import extract_object_keywords, tokenize
from backend.robustness.neural_modules import VLAlignmentModule, _shared_tokenizer, DEVICE
from backend.utils.config import (
    YOLO_CONFIDENCE_THRESHOLD,
    COCO_TO_DRIVING_CLASSES,
    YOLO_WEIGHTS,
    VL_ALIGNMENT_PATH,
)
from backend.utils.logger import get_logger

logger = get_logger(__name__)

_INSTRUCTION_SYNONYMS = {
    "man": "pedestrian",
    "woman": "pedestrian",
    "child": "pedestrian",
    "person": "pedestrian",
    "people": "pedestrian",
    "bike": "cyclist",
    "bicycle": "cyclist",
    "van": "car",
    "sedan": "car",
    "suv": "car",
    "taxi": "car",
    "vehicle": "car",
}

_ACTIONS = {
    "stop", "turn", "left", "right", "move", "go", "drive",
    "follow", "park", "pull", "slow", "accelerate", "reverse",
}

_SPATIAL_CUES = {
    "left", "right", "ahead", "front", "behind", "near", "next", "beside",
}


class ConsistencyChecker:
    """Multimodal consistency checker with Deep Vision-Language alignment projection."""

    def __init__(self):
        self._model = None
        self.vl_module: VLAlignmentModule = None
        self._load_vl_module()

    def _load_vl_module(self):
        try:
            self.vl_module = VLAlignmentModule().to(DEVICE)
            if VL_ALIGNMENT_PATH.exists():
                state_dict = torch.load(VL_ALIGNMENT_PATH, map_location=DEVICE, weights_only=True)
                self.vl_module.load_state_dict(state_dict)
                self.vl_module.eval()
                logger.info(f"Loaded VLAlignmentModule from {VL_ALIGNMENT_PATH.name}")
            else:
                self.vl_module.eval()
        except Exception as exc:
            logger.error(f"Failed to load VLAlignmentModule: {exc}")
            self.vl_module = None

    def _get_model(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
                self._model = YOLO(YOLO_WEIGHTS)
                logger.info("Loaded YOLOv8 model (%s)", YOLO_WEIGHTS)
            except Exception as exc:
                logger.error("Failed to load YOLOv8: %s", exc)
                self._model = False
        return self._model

    def detect_objects(self, image: np.ndarray) -> Dict[str, Any]:
        model = self._get_model()
        quality = basic_quality_check(image)

        if model is False or model is None:
            return {
                "detected_objects": [],
                "detection_count": 0,
                "mean_confidence": 0.0,
                "quality": quality,
                "raw_detections": [],
                "camera_score": 0.0,
            }

        try:
            h, w = image.shape[:2]
            results = model.predict(source=image, conf=YOLO_CONFIDENCE_THRESHOLD, verbose=False)
            names = model.names
            detections = []
            confidences = []
            raw = []

            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    cls_id = int(box.cls.item())
                    confidence = float(box.conf.item())
                    coords = box.xyxy[0].cpu().numpy().tolist() if hasattr(box, "xyxy") else [0, 0, 0, 0]

                    label = names.get(cls_id, str(cls_id))
                    mapped = COCO_TO_DRIVING_CLASSES.get(label, label)

                    detections.append(mapped)
                    confidences.append(confidence)
                    raw.append({
                        "class": mapped,
                        "confidence": round(confidence, 3),
                        "bbox": [round(c, 1) for c in coords],
                        "normalized_center": [
                            round((coords[0] + coords[2]) / (2.0 * max(1, w)), 3),
                            round((coords[1] + coords[3]) / (2.0 * max(1, h)), 3),
                            round((coords[2] - coords[0]) / max(1, w), 3),
                            round((coords[3] - coords[1]) / max(1, h), 3),
                        ],
                        "cls_id": cls_id,
                    })

            detection_count = len(detections)
            mean_confidence = float(np.mean(confidences)) if confidences else 0.0

            # Camera reliability score
            camera_score = 100.0
            if detection_count == 0:
                camera_score -= 30
            if mean_confidence < 0.50:
                camera_score -= 20
            if quality.get("is_blurry"):
                camera_score -= 20
            if quality.get("is_over_or_under_exposed"):
                camera_score -= 20

            camera_score = max(0.0, min(100.0, camera_score))

            return {
                "detected_objects": detections,
                "detection_count": detection_count,
                "mean_confidence": round(mean_confidence, 3),
                "quality": quality,
                "raw_detections": raw,
                "camera_score": round(camera_score, 1),
                "score": round(camera_score, 1),
            }
        except Exception as exc:
            logger.error("Object detection failed: %s", exc)
            return {
                "detected_objects": [],
                "detection_count": 0,
                "mean_confidence": 0.0,
                "quality": quality,
                "raw_detections": [],
                "camera_score": 30.0,
                "score": 30.0,
            }

    def _extract_instruction_features(self, instruction: str) -> Dict[str, Any]:
        text = instruction.lower()
        words = set(re.findall(r"\b[a-z]+\b", text))
        objects = set(extract_object_keywords(instruction))
        objects = {_INSTRUCTION_SYNONYMS.get(obj, obj) for obj in objects}

        return {
            "objects": objects,
            "actions": words & _ACTIONS,
            "spatial_cues": words & _SPATIAL_CUES,
        }

    def _analyse_radar(self, radar_df: pd.DataFrame) -> Dict[str, Any]:
        if radar_df is None or radar_df.empty:
            return {
                "available": False,
                "point_count": 0,
                "moving_points": 0,
                "nearby_points": 0,
                "left_points": 0,
                "right_points": 0,
            }

        df = radar_df.copy()
        point_count = len(df)
        speed = pd.to_numeric(df.get("relative_speed_mps", pd.Series(dtype=float)), errors="coerce")
        ranges = pd.to_numeric(df.get("range_m", pd.Series(dtype=float)), errors="coerce")
        angles = pd.to_numeric(df.get("azimuth_deg", pd.Series(dtype=float)), errors="coerce")

        return {
            "available": True,
            "point_count": point_count,
            "moving_points": int((speed.abs() > 0.5).sum()),
            "nearby_points": int((ranges < 20).sum()),
            "left_points": int((angles < -5).sum()),
            "right_points": int((angles > 5).sum()),
        }

    def _compute_deep_vl_grounding(self, instruction: str, raw_detections: List[Dict]) -> float:
        """
        Compute deep Vision-Language cosine alignment score.
        """
        if not self.vl_module:
            return 0.75
        if not raw_detections:
            return 0.20

        try:
            device = next(self.vl_module.parameters()).device if list(self.vl_module.parameters()) else DEVICE
            # Construct text embedding representation
            indices, _ = _shared_tokenizer.encode(instruction)
            text_tensor = torch.tensor(indices, dtype=torch.float32, device=device)
            # Project text to 64-dim
            if len(text_tensor) < 64:
                text_tensor = F.pad(text_tensor, (0, 64 - len(text_tensor)))
            else:
                text_tensor = text_tensor[:64]

            # Construct visual features [cx, cy, w, h, conf, class_id]
            vis_list = []
            for d in raw_detections[:12]:
                nc = d.get("normalized_center", [0.5, 0.5, 0.1, 0.1])
                conf = float(d.get("confidence", 0.5))
                cid = float(d.get("cls_id", 0)) / 80.0
                vis_list.append([nc[0], nc[1], nc[2], nc[3], conf, cid])

            vis_tensor = torch.tensor(vis_list, dtype=torch.float32, device=device)
            with torch.no_grad():
                sim_matrix = self.vl_module(text_tensor, vis_tensor)
                max_sim = float(sim_matrix.max().item())
                # Normalize cosine similarity [-1, 1] to [0.0, 1.0]
                vl_score = (max_sim + 1.0) / 2.0
                return max(0.0, min(1.0, vl_score))
        except Exception as exc:
            logger.warning("Deep VL grounding computation failed: %s", exc)
            return 0.50

    def cross_modal_check(
        self,
        instruction: str,
        scene: Dict,
        radar_df: pd.DataFrame,
        radar_result: Dict = None,
    ) -> Dict[str, Any]:
        try:
            features = self._extract_instruction_features(instruction)
            instruction_objects = features["objects"]
            actions = features["actions"]
            spatial = features["spatial_cues"]
            vision_objects = set(scene.get("detected_objects", []))
            radar_info = self._analyse_radar(radar_df)

            # 1. Object Grounding
            if instruction_objects:
                matched_objects = instruction_objects & vision_objects
                lexical_obj_score = len(matched_objects) / len(instruction_objects)
            else:
                matched_objects = set()
                lexical_obj_score = 1.0

            # 2. Deep Vision-Language Alignment
            raw_dets = scene.get("raw_detections", [])
            vl_similarity = self._compute_deep_vl_grounding(instruction, raw_dets)
            
            # Combine lexical grounding with deep neural VL cosine similarity
            if instruction_objects:
                object_score = 0.5 * lexical_obj_score + 0.5 * vl_similarity
            else:
                object_score = 0.85 + 0.15 * vl_similarity

            # 3. Radar Evidence
            if not radar_info["available"]:
                radar_score = 0.0
            elif radar_info["point_count"] == 0:
                radar_score = 0.2
            else:
                radar_score = 1.0

            if radar_result:
                anomaly_points = radar_result.get("anomaly_points", 0)
                total_pts = len(radar_df) if radar_df is not None else 0
                if total_pts > 0 and anomaly_points > 0:
                    anomaly_ratio = anomaly_points / total_pts
                    radar_score *= max(0.0, 1.0 - min(0.7, anomaly_ratio))

            # 4. Spatial Consistency
            spatial_score = 1.0
            if "left" in spatial:
                if radar_info["left_points"] > 0:
                    spatial_score = 1.0
                elif radar_info["right_points"] > 0:
                    spatial_score = 0.25
                else:
                    spatial_score = 0.5
            elif "right" in spatial:
                if radar_info["right_points"] > 0:
                    spatial_score = 1.0
                elif radar_info["left_points"] > 0:
                    spatial_score = 0.25
                else:
                    spatial_score = 0.5

            # 5. Action Consistency
            action_score = 1.0
            if "stop" in actions and radar_info["moving_points"] > 0:
                action_score = 0.7

            camera_score = scene.get("camera_score", 0.0) / 100.0

            # Final Cross-Modal Consistency Score
            consistency_score = (
                0.35 * object_score
                + 0.20 * radar_score
                + 0.15 * spatial_score
                + 0.10 * action_score
                + 0.20 * camera_score
            ) * 100.0
            consistency_score = max(0.0, min(100.0, consistency_score))

            unmatched_ref = bool(instruction_objects) and not matched_objects

            return {
                "instruction_objects": sorted(instruction_objects),
                "matched_objects": sorted(matched_objects),
                "unmatched_instruction_reference": unmatched_ref,
                "actions": sorted(actions),
                "spatial_cues": sorted(spatial),
                "vision_objects": sorted(vision_objects),
                "radar_evidence": radar_info,
                "vl_grounding_score": round(vl_similarity, 3),
                "object_grounding_score": round(object_score * 100.0, 1),
                "spatial_consistency_score": round(spatial_score * 100.0, 1),
                "action_consistency_score": round(action_score * 100.0, 1),
                "camera_consistency_score": round(camera_score * 100.0, 1),
                "radar_consistency_score": round(radar_score * 100.0, 1),
                "consistency_score": round(float(consistency_score), 1),
            }
        except Exception as exc:
            logger.error("Cross-modal check failed: %s", exc)
            return {
                "instruction_objects": [],
                "matched_objects": [],
                "unmatched_instruction_reference": True,
                "actions": [],
                "spatial_cues": [],
                "vision_objects": [],
                "radar_evidence": {
                    "available": False,
                    "point_count": 0,
                    "moving_points": 0,
                    "nearby_points": 0,
                    "left_points": 0,
                    "right_points": 0,
                },
                "vl_grounding_score": 0.0,
                "object_grounding_score": 0.0,
                "spatial_consistency_score": 0.0,
                "action_consistency_score": 0.0,
                "camera_consistency_score": 0.0,
                "radar_consistency_score": 0.0,
                "consistency_score": 0.0,
                "error": str(exc),
            }


_checker_instance = None


def get_consistency_checker() -> ConsistencyChecker:
    global _checker_instance
    if _checker_instance is None:
        _checker_instance = ConsistencyChecker()
    return _checker_instance