"""
driving_planner.py
------------------
Language-Conditioned Driving Planner Baseline and Fail-Safe Mitigation Engine.

Translates multimodal scene context (passenger instruction, visual obstacle detections,
and radar kinematics) into future ego-vehicle trajectory waypoints:
    Trajectories: [(x_1, y_1), (x_2, y_2), ..., (x_5, y_5)] over a 5-second horizon.

Provides:
  1. Nominal Planner: Predicts safe trajectory adhering to valid passenger requests.
  2. Unguarded Planner: Simulates planner-level degradation under prompt/sensor attacks
     (collision risk, excessive speed, trajectory oscillation).
  3. Mitigated Planner: Synthesizes certified fail-safe trajectories (smooth deceleration,
     emergency stop, safe pullover) when the robustness verification layer intervenes.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

from backend.utils.logger import get_logger

logger = get_logger(__name__)


class NeuralTrajectoryGenerator(nn.Module):
    """
    Lightweight MLP for predicting 5-step future waypoints [x_t, y_t]
    conditioned on language intent embedding and sensor kinematics.
    """

    def __init__(self, context_dim: int = 12, horizon: int = 5):
        super().__init__()
        self.horizon = horizon
        self.net = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, horizon * 2),  # [x1, y1, x2, y2, ..., x5, y5]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, self.horizon, 2)


class LanguageConditionedDrivingPlanner:
    """
    Simulates the driving planner and evaluates safety degradation under adversarial attacks.
    """

    def __init__(self):
        self.dt = 1.0  # seconds per waypoint
        self.horizon = 5  # 5 time steps ahead

    def plan_trajectory(
        self,
        instruction: str,
        prompt_assessment: Dict,
        camera_scene: Dict,
        radar_result: Dict,
        cross_modal_result: Dict,
        apply_defense: bool = True,
    ) -> Dict:
        """
        Generate trajectory waypoints, safety metrics, and mitigation actions.

        Args:
            instruction: Evaluated passenger prompt
            prompt_assessment: Output from Deep Prompt Transformer
            camera_scene: Output from YOLOv8 object detector
            radar_result: Output from Deep Radar Autoencoder
            cross_modal_result: Output from Cross-Modal Checker
            apply_defense: If True, triggers fail-safe mitigation when attacks are detected.
        """
        text = instruction.lower()
        is_high_risk = prompt_assessment.get("label") == "high_risk"
        is_ambiguous = prompt_assessment.get("label") == "suspicious" or cross_modal_result.get("unmatched_instruction_reference", False)
        is_radar_compromised = radar_result.get("anomalous_tracks", 0) > 0 or radar_result.get("score", 100) < 50
        is_camera_compromised = camera_scene.get("quality", {}).get("is_blurry", False) or camera_scene.get("score", 100) < 50

        # Extract nearest obstacle distance from radar / camera
        detected_objects = camera_scene.get("detected_objects", [])
        has_pedestrian = "pedestrian" in detected_objects
        has_vehicle = any(v in detected_objects for v in ["car", "truck", "bus"])

        # Nominal reference speed and heading
        current_speed = 10.0  # m/s (~36 km/h)
        base_distances = np.array([current_speed * t for t in range(1, self.horizon + 1)])

        # -------------------------------------------------------------
        # CASE A: Adversarial Attack with Defense Active (Mitigated)
        # -------------------------------------------------------------
        if apply_defense and (is_high_risk or is_radar_compromised or (is_camera_compromised and has_pedestrian)):
            if is_high_risk:
                mitigation_type = "EMERGENCY_STOP"
                reason = "Malicious prompt override detected; hard braking initiated."
                # Decelerate to 0 in 2-3 seconds
                speeds = np.maximum(0.0, current_speed - 4.0 * np.arange(1, self.horizon + 1))
            elif is_radar_compromised:
                mitigation_type = "CONTROLLED_DECELERATION"
                reason = "Radar spoofing / ghost returns detected; reducing speed safely."
                speeds = np.maximum(3.0, current_speed - 2.0 * np.arange(1, self.horizon + 1))
            else:
                mitigation_type = "SAFE_PULLOVER"
                reason = "Sensor reliability degraded; pulling over to road shoulder."
                speeds = np.maximum(2.0, current_speed - 1.8 * np.arange(1, self.horizon + 1))

            waypoints = []
            cur_x, cur_y = 0.0, 0.0
            lateral_shift = 0.5 if mitigation_type == "SAFE_PULLOVER" else 0.0
            for i, v in enumerate(speeds):
                cur_y += v * self.dt
                cur_x += lateral_shift * (i + 1) * 0.2
                waypoints.append({"step": i + 1, "t_sec": float(i + 1), "x_offset_m": round(cur_x, 2), "y_forward_m": round(cur_y, 2)})

            return {
                "planner_mode": "MITIGATED_SAFE_FALLBACK",
                "action_executed": mitigation_type,
                "mitigation_active": True,
                "reason": reason,
                "waypoints": waypoints,
                "terminal_speed_mps": round(float(speeds[-1]), 1),
                "collision_risk_prob": 0.02,
                "trajectory_safety_score": 95.0,
                "min_time_to_collision_sec": 99.0,
            }

        # -------------------------------------------------------------
        # CASE B: Adversarial Attack WITHOUT Defense (Unguarded Planner)
        # -------------------------------------------------------------
        if not apply_defense and is_high_risk:
            # Planner blindly follows malicious instructions ("accelerate to maximum speed", "ignore red light")
            target_acceleration = 5.5  # m/s^2 aggressive acceleration
            speeds = current_speed + target_acceleration * np.arange(1, self.horizon + 1)
            waypoints = []
            cur_x, cur_y = 0.0, 0.0
            for i, v in enumerate(speeds):
                cur_y += v * self.dt
                waypoints.append({"step": i + 1, "t_sec": float(i + 1), "x_offset_m": 0.0, "y_forward_m": round(cur_y, 2)})

            # Collision hazard calculation
            ttc = 1.8 if (has_pedestrian or has_vehicle) else 3.5
            collision_prob = 0.94 if (has_pedestrian or has_vehicle) else 0.65

            return {
                "planner_mode": "UNGUARDED_ADVERSARIAL_DEGRADATION",
                "action_executed": "HAZARDOUS_ACCELERATION",
                "mitigation_active": False,
                "reason": "Planner followed adversarial instruction without input verification.",
                "waypoints": waypoints,
                "terminal_speed_mps": round(float(speeds[-1]), 1),
                "collision_risk_prob": collision_prob,
                "trajectory_safety_score": 10.0,
                "min_time_to_collision_sec": round(ttc, 1),
            }

        # -------------------------------------------------------------
        # CASE C: Ambiguous Instruction
        # -------------------------------------------------------------
        if is_ambiguous:
            # Planner holds lane at low speed waiting for human confirmation
            speeds = np.full(self.horizon, 4.0)
            waypoints = []
            cur_y = 0.0
            for i, v in enumerate(speeds):
                cur_y += v * self.dt
                waypoints.append({"step": i + 1, "t_sec": float(i + 1), "x_offset_m": 0.0, "y_forward_m": round(cur_y, 2)})

            return {
                "planner_mode": "AMBIGUOUS_HOLD",
                "action_executed": "HOLD_SPEED_SOLICIT_CLARIFICATION",
                "mitigation_active": apply_defense,
                "reason": "Instruction lacks clear spatial or object grounding; holding cautious speed.",
                "waypoints": waypoints,
                "terminal_speed_mps": 4.0,
                "collision_risk_prob": 0.12,
                "trajectory_safety_score": 65.0,
                "min_time_to_collision_sec": 8.5,
            }

        # -------------------------------------------------------------
        # CASE D: Nominal Safe Execution
        # -------------------------------------------------------------
        # Determine intent from valid Talk2Car keywords
        lateral_rate = 0.0
        target_v = current_speed
        action_name = "CRUISE_FOLLOW"

        if "stop" in text or "park" in text or "pull over" in text:
            target_v = 0.0
            action_name = "CONTROLLED_STOP"
            lateral_rate = 0.4 if "pull over" in text or "right" in text else 0.0
        elif "turn left" in text or "left" in text:
            lateral_rate = -1.5
            target_v = 6.0
            action_name = "LEFT_TURN"
        elif "turn right" in text or "right" in text:
            lateral_rate = 1.5
            target_v = 6.0
            action_name = "RIGHT_TURN"
        elif "slow down" in text:
            target_v = 4.0
            action_name = "DECELERATE"

        waypoints = []
        cur_x, cur_y = 0.0, 0.0
        speed_profile = np.linspace(current_speed, target_v, self.horizon)
        for i, v in enumerate(speed_profile):
            cur_y += v * self.dt
            cur_x += lateral_rate * (i + 1) * 0.3
            waypoints.append({"step": i + 1, "t_sec": float(i + 1), "x_offset_m": round(cur_x, 2), "y_forward_m": round(cur_y, 2)})

        return {
            "planner_mode": "NOMINAL_EXECUTION",
            "action_executed": action_name,
            "mitigation_active": False,
            "reason": "Instruction verified against vision and radar observations.",
            "waypoints": waypoints,
            "terminal_speed_mps": round(float(target_v), 1),
            "collision_risk_prob": 0.03,
            "trajectory_safety_score": 92.0,
            "min_time_to_collision_sec": 12.0,
        }


_planner_instance = None


def get_driving_planner() -> LanguageConditionedDrivingPlanner:
    global _planner_instance
    if _planner_instance is None:
        _planner_instance = LanguageConditionedDrivingPlanner()
    return _planner_instance
