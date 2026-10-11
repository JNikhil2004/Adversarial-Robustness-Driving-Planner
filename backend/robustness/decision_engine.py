"""
decision_engine.py
------------------
Combines prompt, ambiguity, camera, radar and cross-modal checks into
a single safety decision.

Outputs:
    - Overall Safety Score
    - Risk Level
    - Driving Recommendation
    - Per-channel robustness scores
    - Explanation trail
"""

from typing import Dict, List

from backend.utils.config import (
    DECISION_WEIGHTS,
    SAFETY_SCORE_SAFE_THRESHOLD,
    SAFETY_SCORE_CONFIRM_THRESHOLD,
    RISK_LEVEL_LOW_THRESHOLD,
    RISK_LEVEL_MEDIUM_THRESHOLD,
    HARD_BLOCK_PROMPT_LABELS,
    MULTI_CHANNEL_DEGRADATION_LIMIT,
    PROMPT_LOW_SCORE,
    AMBIGUITY_LOW_SCORE,
    CAMERA_LOW_SCORE,
    RADAR_LOW_SCORE,
    CONSISTENCY_LOW_SCORE,
)

from backend.robustness.neural_modules import MultimodalSafetyFuser, DEVICE
from backend.robustness.driving_planner import get_driving_planner
from backend.utils.config import MULTIMODAL_FUSER_PATH
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class DecisionEngine:
    """Fuse all robustness channels into one safety decision with deep neural fusion and trajectory mitigation."""

    def __init__(self):
        self.fuser: MultimodalSafetyFuser = None
        self._load_fuser()

    def _load_fuser(self):
        try:
            import torch
            self.fuser = MultimodalSafetyFuser().to(DEVICE)
            if MULTIMODAL_FUSER_PATH.exists():
                state_dict = torch.load(MULTIMODAL_FUSER_PATH, map_location=DEVICE, weights_only=True)
                self.fuser.load_state_dict(state_dict)
                self.fuser.eval()
                logger.info(f"Loaded MultimodalSafetyFuser from {MULTIMODAL_FUSER_PATH.name}")
            else:
                self.fuser.eval()
        except Exception as exc:
            logger.error(f"Failed to load MultimodalSafetyFuser: {exc}")
            self.fuser = None

    def decide(
        self,
        prompt_result: Dict,
        ambiguity_result: Dict,
        radar_result: Dict,
        scene_result: Dict,
        cross_modal_result: Dict,
        prompt_attack_applied: str = None,
        camera_attack_applied: str = None,
        radar_attack_applied: str = None,
        instruction: str = "",
    ) -> Dict:

        try:
            import torch

            # 1. CAMERA SCORE
            camera_score = self._scene_understanding_score(scene_result)
            scene_notes = self._scene_notes(scene_result)

            # 2. CHANNEL SCORES
            prompt_score = float(prompt_result.get("score", 0.0))
            ambiguity_score = float(ambiguity_result.get("score", 0.0))
            radar_score = float(radar_result.get("score", 0.0))
            consistency_score = float(cross_modal_result.get("consistency_score", 0.0))

            # 3. CLASSICAL WEIGHTED MULTI-MODAL SAFETY SCORE
            weighted_safety_score = (
                DECISION_WEIGHTS["prompt"] * prompt_score
                + DECISION_WEIGHTS["ambiguity"] * ambiguity_score
                + DECISION_WEIGHTS["camera"] * camera_score
                + DECISION_WEIGHTS["radar"] * radar_score
                + DECISION_WEIGHTS["consistency"] * consistency_score
            )

            # 4. DEEP NEURAL FUSION WITH EPISTEMIC UNCERTAINTY
            epistemic_uncertainty = 0.05
            fused_score = weighted_safety_score
            if self.fuser is not None:
                lbl_map = {
                    "trusted": [0.95, 0.05, 0.0],
                    "suspicious": [0.1, 0.85, 0.05],
                    "high_risk": [0.0, 0.1, 0.9],
                }
                p_probs = lbl_map.get(prompt_result.get("label", "trusted"), [0.8, 0.1, 0.1])
                amb_score_norm = ambiguity_score / 100.0
                is_amb = 1.0 if ambiguity_result.get("is_ambiguous", False) else 0.0
                det_cnt_norm = min(1.0, scene_result.get("detection_count", 0) / 10.0)
                mean_conf = float(scene_result.get("mean_confidence", 0.0))
                cam_ok = 0.0 if scene_result.get("quality", {}).get("is_blurry") else 1.0
                recon_mse = float(radar_result.get("reconstruction_mse", 0.25))
                recon_mse_norm = min(2.0, recon_mse) / 2.0
                radar_anom_ratio = float(radar_result.get("anomaly_ratio", 0.0))
                track_cnt_norm = min(1.0, radar_result.get("total_tracks", 1) / 10.0)
                vl_grounding = float(cross_modal_result.get("vl_grounding_score", 0.85))

                feat_vec = p_probs + [
                    amb_score_norm, is_amb,
                    det_cnt_norm, mean_conf, cam_ok,
                    recon_mse_norm, radar_anom_ratio, track_cnt_norm,
                    vl_grounding,
                ]
                in_tensor = torch.tensor([feat_vec[:12]], dtype=torch.float32, device=DEVICE)
                neural_score, _, uncertainty = self.fuser.predict_with_uncertainty(in_tensor, n_mc_samples=8)
                epistemic_uncertainty = uncertainty
                # Blend weighted physical rule score with neural latent fuser score
                fused_score = 0.60 * weighted_safety_score + 0.40 * neural_score

            overall_safety_score = round(max(0.0, min(100.0, fused_score)), 1)

            # 5. FIND DEGRADED CHANNELS
            degraded_channels = self._degraded_channels(
                prompt_score=prompt_score,
                ambiguity_score=ambiguity_score,
                camera_score=camera_score,
                radar_score=radar_score,
                consistency_score=consistency_score,
            )

            # 6. RISK LEVEL & HARD SAFETY OVERRIDES
            is_prompt_high_risk = prompt_result.get("label") in HARD_BLOCK_PROMPT_LABELS
            is_severe_radar_spoof = radar_result.get("score", 100.0) < 30.0

            if is_prompt_high_risk or is_severe_radar_spoof:
                overall_safety_score = min(overall_safety_score, 18.0)
                risk_level = "High"
            elif overall_safety_score >= RISK_LEVEL_LOW_THRESHOLD:
                risk_level = "Low"
            elif overall_safety_score >= RISK_LEVEL_MEDIUM_THRESHOLD:
                risk_level = "Medium"
            else:
                risk_level = "High"

            # 7. RECOMMENDATION & EXPLANATION
            recommendation, explanation = self._recommend(
                overall_score=overall_safety_score,
                prompt_result=prompt_result,
                ambiguity_result=ambiguity_result,
                radar_result=radar_result,
                scene_result=scene_result,
                cross_modal_result=cross_modal_result,
                degraded_channels=degraded_channels,
            )
            explanation.extend(scene_notes)

            overall_robustness = round(
                (prompt_score + ambiguity_score + camera_score + radar_score + consistency_score) / 5.0,
                1,
            )

            # 8. ATTACK INFORMATION
            explanation.extend(
                self._attack_notes(
                    prompt_attack_applied,
                    camera_attack_applied,
                    radar_attack_applied,
                )
            )

            # 9. TRAJECTORY SYNTHESIS (NOMINAL / UNGUARDED VS MITIGATED FAIL-SAFE)
            planner = get_driving_planner()
            eval_instruction = instruction or prompt_result.get("instruction", "")
            guarded_trajectory = planner.plan_trajectory(
                instruction=eval_instruction,
                prompt_assessment=prompt_result,
                camera_scene=scene_result,
                radar_result=radar_result,
                cross_modal_result=cross_modal_result,
                apply_defense=True,
            )
            unguarded_trajectory = planner.plan_trajectory(
                instruction=eval_instruction,
                prompt_assessment=prompt_result,
                camera_scene=scene_result,
                radar_result=radar_result,
                cross_modal_result=cross_modal_result,
                apply_defense=False,
            )

            return {
                "overall_safety_score": overall_safety_score,
                "risk_level": risk_level,
                "recommendation": recommendation,
                "epistemic_uncertainty": round(epistemic_uncertainty, 3),
                "explanation": explanation,
                "scene_understanding_score": round(camera_score, 1),
                "degraded_channels": degraded_channels,
                "trajectories": {
                    "guarded": guarded_trajectory,
                    "unguarded": unguarded_trajectory,
                },
                "robustness": {
                    "prompt_robustness": round(prompt_score, 1),
                    "ambiguity_robustness": round(ambiguity_score, 1),
                    "camera_robustness": round(camera_score, 1),
                    "radar_robustness": round(radar_score, 1),
                    "consistency_robustness": round(consistency_score, 1),
                    "overall_robustness": overall_robustness,
                },
            }

        except Exception as exc:

            logger.error(
                f"Decision engine failure: {exc}"
            )

            return {
                "overall_safety_score": 0.0,
                "risk_level": "High",
                "recommendation": "Execution Blocked",

                "explanation": [
                    f"Decision engine encountered an internal error: {exc}"
                ],

                "scene_understanding_score": 0.0,

                "degraded_channels": [],

                "robustness": {
                    "prompt_robustness": 0.0,
                    "ambiguity_robustness": 0.0,
                    "camera_robustness": 0.0,
                    "radar_robustness": 0.0,
                    "consistency_robustness": 0.0,
                    "overall_robustness": 0.0,
                },
            }

    # ======================================================================
    # CAMERA SCORING
    # ======================================================================

    @staticmethod
    def _scene_understanding_score(
        scene_result: Dict,
    ) -> float:

        detection_count = scene_result.get(
            "detection_count",
            0,
        )

        mean_confidence = scene_result.get(
            "mean_confidence",
            0.0,
        )

        quality = scene_result.get(
            "quality",
            {},
        )

        # No detections means the scene is poorly understood.
        if detection_count == 0:

            score = 40.0

        else:

            score = (
                60.0
                + min(
                    30.0,
                    detection_count * 5.0,
                )
                + mean_confidence * 10.0
            )

        # Penalize degraded image quality.
        if quality.get("is_blurry"):

            score -= 20.0

        if quality.get("is_over_or_under_exposed"):

            score -= 20.0

        return max(
            0.0,
            min(
                100.0,
                score
            )
        )

    # ======================================================================
    # CAMERA EXPLANATION
    # ======================================================================

    @staticmethod
    def _scene_notes(
        scene_result: Dict,
    ) -> List[str]:

        notes: List[str] = []

        quality = scene_result.get(
            "quality",
            {},
        )

        if quality.get("is_blurry"):

            notes.append(
                "Camera frame quality check flagged "
                "the image as blurry; detections may be unreliable."
            )

        if quality.get(
            "is_over_or_under_exposed"
        ):

            notes.append(
                "Camera frame is over/under-exposed; "
                "detections may be unreliable."
            )

        if scene_result.get(
            "detection_count",
            0,
        ) == 0:

            notes.append(
                "YOLOv8 found no recognizable objects "
                "in the camera frame."
            )

        return notes

    # ======================================================================
    # DEGRADED CHANNEL DETECTION
    # ======================================================================

    @staticmethod
    def _degraded_channels(
        prompt_score: float,
        ambiguity_score: float,
        camera_score: float,
        radar_score: float,
        consistency_score: float,
    ) -> List[str]:

        degraded: List[str] = []

        if prompt_score < PROMPT_LOW_SCORE:
            degraded.append("prompt")

        if ambiguity_score < AMBIGUITY_LOW_SCORE:
            degraded.append("ambiguity")

        if camera_score < CAMERA_LOW_SCORE:
            degraded.append("camera")

        if radar_score < RADAR_LOW_SCORE:
            degraded.append("radar")

        if consistency_score < CONSISTENCY_LOW_SCORE:
            degraded.append("consistency")

        return degraded

    # ======================================================================
    # RECOMMENDATION
    # ======================================================================

    @staticmethod
    def _recommend(
        overall_score: float,
        prompt_result: Dict,
        ambiguity_result: Dict,
        radar_result: Dict,
        scene_result: Dict,
        cross_modal_result: Dict,
        degraded_channels: List[str],
    ):

        explanation: List[str] = []

        # ==============================================================
        # PROMPT HARD BLOCK
        # ==============================================================

        hard_block = (
            prompt_result.get("label")
            in HARD_BLOCK_PROMPT_LABELS
        )

        if hard_block:

            explanation.append(
                "Passenger instruction was classified as "
                "HIGH RISK by the prompt detector "
                f"(confidence "
                f"{prompt_result.get('confidence', 0):.0%}). "
                f"Trigger terms: "
                f"{prompt_result.get('trigger_terms', [])}."
            )

        # ==============================================================
        # AMBIGUITY
        # ==============================================================

        if ambiguity_result.get(
            "is_ambiguous",
            False,
        ):

            reasons = ambiguity_result.get(
                "ambiguity_reasons",
                [],
            )

            explanation.append(
                "Instruction clarity check flagged ambiguity: "
                + (
                    "; ".join(reasons)
                    if reasons
                    else "instruction is insufficiently specific."
                )
            )

        # ==============================================================
        # RADAR
        # ==============================================================

        anomalous_tracks = radar_result.get(
            "anomalous_tracks",
            0,
        )

        if anomalous_tracks > 0:

            anomalies = radar_result.get(
                "anomalies",
                [],
            )

            explanation.append(
                "Radar validator found "
                f"{anomalous_tracks} anomalous track(s): "
                + (
                    "; ".join(anomalies[:3])
                    if anomalies
                    else "radar data contains suspicious measurements."
                )
            )

        # ==============================================================
        # CAMERA
        # ==============================================================

        quality = scene_result.get(
            "quality",
            {},
        )

        if quality.get("is_blurry"):

            explanation.append(
                "Camera reliability was reduced because "
                "the image is blurry."
            )

        if quality.get(
            "is_over_or_under_exposed"
        ):

            explanation.append(
                "Camera reliability was reduced because "
                "of exposure degradation."
            )

        detection_count = scene_result.get(
            "detection_count",
            0,
        )

        if detection_count == 0:

            explanation.append(
                "Camera scene understanding produced "
                "no recognizable object detections."
            )

        # ==============================================================
        # CROSS-MODAL CONSISTENCY
        # ==============================================================

        if cross_modal_result.get(
            "unmatched_instruction_reference",
            False,
        ):

            explanation.append(
                "Cross-modal verification could not ground "
                "the instruction's referenced object in the "
                "available camera or radar observations."
            )

        # ==============================================================
        # MULTI-CHANNEL DEGRADATION
        # ==============================================================

        if len(degraded_channels) >= (
            MULTI_CHANNEL_DEGRADATION_LIMIT
        ):

            explanation.append(
                "Multiple independent robustness channels "
                "are degraded: "
                + ", ".join(degraded_channels)
                + "."
            )

        # ==============================================================
        # NO PROBLEMS
        # ==============================================================

        if not explanation:

            explanation.append(
                "All robustness checks passed with no "
                "significant anomalies detected."
            )

        # ==============================================================
        # FINAL RECOMMENDATION
        # ==============================================================

        # A high-risk instruction is always blocked.
        if hard_block:

            recommendation = "Execution Blocked"

        # Very low overall safety.
        elif overall_score < SAFETY_SCORE_CONFIRM_THRESHOLD:

            recommendation = "Execution Blocked"

        # Intermediate safety.
        elif overall_score < SAFETY_SCORE_SAFE_THRESHOLD:

            recommendation = "Require Human Confirmation"

        # High safety.
        else:

            recommendation = "Safe to Execute"

        return recommendation, explanation

    # ======================================================================
    # ATTACK NOTES
    # ======================================================================

    @staticmethod
    def _attack_notes(
        prompt_attack,
        camera_attack,
        radar_attack,
    ) -> List[str]:

        notes: List[str] = []

        if prompt_attack and prompt_attack != "none":

            notes.append(
                f"Prompt attack simulation active: "
                f"'{prompt_attack}'."
            )

        if camera_attack and camera_attack != "none":

            notes.append(
                f"Camera attack simulation active: "
                f"'{camera_attack}'."
            )

        if radar_attack and radar_attack != "none":

            notes.append(
                f"Radar attack simulation active: "
                f"'{radar_attack}'."
            )

        return notes


# ===========================================================================
# SINGLETON
# ===========================================================================

_engine_instance = None


def get_decision_engine() -> DecisionEngine:

    global _engine_instance

    if _engine_instance is None:
        _engine_instance = DecisionEngine()

    return _engine_instance