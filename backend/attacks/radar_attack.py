"""
radar_attack.py
---------------
Adversarial Radar Kinematics & RF Spoofing Attack Suite.

Vectors:
1. Kinematic Trajectory Drift: Smooth phase perturbation that shifts range/azimuth
   consistently, evading static bounds but detected by temporal autoencoders.
2. Coherent Ghost Formation: Injects a synthetic spatial cluster with plausible
   cross-section and Doppler kinematics simulating a phantom vehicle obstacle.
3. Doppler Velocity Spoofing: Electronic RF velocity jamming corrupting relative velocity and TTC.
4. Range Spoofing: False echo time manipulation shifting obstacle distances.
5. Stochastic Sensor Dropout: Simulated RF jamming causing return density collapse.
"""

import random
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from backend.utils.logger import get_logger

logger = get_logger(__name__)

SEED = 42


def kinematic_trajectory_drift(df: pd.DataFrame, drift_ratio: float = 0.35, random_state: int = SEED) -> Tuple[pd.DataFrame, Dict]:
    """
    Smooth kinematic phase manipulation that shifts point trajectories over time.
    Stays within naive static min/max bounds but breaks manifold consistency in temporal autoencoders.
    """
    attacked = df.copy()
    valid = attacked.index[attacked["range_m"].notna() & (attacked["range_m"] > 0)]
    if len(valid) == 0:
        return attacked, {"attack": "kinematic_trajectory_drift", "points_affected": 0}

    rng = np.random.default_rng(random_state)
    n = max(1, int(len(valid) * drift_ratio))
    indices = rng.choice(valid.to_numpy(), min(n, len(valid)), replace=False)

    # Apply smooth phase shift: range drift + azimuth rotation
    old_ranges = attacked.loc[indices, "range_m"].to_numpy(dtype=float)
    drift_factor = rng.uniform(0.35, 0.75, len(indices))
    new_ranges = old_ranges + drift_factor * 8.0  # Smooth 8m kinematic drift

    attacked.loc[indices, "range_m"] = new_ranges
    if "azimuth_deg" in attacked.columns:
        old_az = attacked.loc[indices, "azimuth_deg"].to_numpy(dtype=float)
        attacked.loc[indices, "azimuth_deg"] = old_az + rng.uniform(-12.0, 12.0, len(indices))

    logger.info(f"Applied kinematic_trajectory_drift to {len(indices)} points")
    return attacked, {
        "attack": "kinematic_trajectory_drift",
        "points_before": len(df),
        "points_after": len(attacked),
        "points_affected": len(indices),
        "drift_mean_m": round(float(np.mean(drift_factor * 8.0)), 2),
    }


def inject_coherent_ghost(df: pd.DataFrame, ghost_points: int = 15, random_state: int = SEED) -> Tuple[pd.DataFrame, Dict]:
    """
    Injects a cohesive multi-point phantom vehicle obstacle ahead of the ego vehicle.
    """
    attacked = df.copy()
    rng = np.random.default_rng(random_state)

    center_range = 14.5  # 14.5 meters directly ahead
    center_azimuth = rng.uniform(-4.0, 4.0)
    target_speed = -11.5  # Approaching at ~41 km/h

    ranges = rng.normal(center_range, 1.2, ghost_points)
    azimuths = rng.normal(center_azimuth, 2.5, ghost_points)
    speeds = rng.normal(target_speed, 0.8, ghost_points)
    rcs = rng.normal(12.0, 3.0, ghost_points)

    ghost_df = pd.DataFrame({
        "range_m": ranges,
        "azimuth_deg": azimuths,
        "relative_speed_mps": speeds,
        "rcs_db": rcs,
        "track_id": [99999] * ghost_points,
    })

    attacked = pd.concat([attacked, ghost_df], ignore_index=True)
    logger.info(f"Injected coherent_ghost obstacle with {ghost_points} points")
    return attacked, {
        "attack": "coherent_ghost_formation",
        "points_before": len(df),
        "points_after": len(attacked),
        "points_affected": ghost_points,
        "ghost_center_range_m": center_range,
        "ghost_speed_mps": target_speed,
    }


def spoof_doppler_velocity(df: pd.DataFrame, fraction: float = 0.35, random_state: int = SEED) -> Tuple[pd.DataFrame, Dict]:
    """
    Simulates electronic Doppler velocity jamming. Injects synthetic radial velocity shifts.
    """
    attacked = df.copy()
    valid = attacked.index[attacked["relative_speed_mps"].notna()]
    if len(valid) == 0:
        return attacked, {"attack": "doppler_velocity_spoofing", "points_affected": 0}

    rng = np.random.default_rng(random_state)
    n = max(1, int(len(valid) * fraction))
    indices = rng.choice(valid.to_numpy(), min(n, len(valid)), replace=False)

    old_speeds = attacked.loc[indices, "relative_speed_mps"].to_numpy(dtype=float)
    jamming_offsets = rng.choice([-1.0, 1.0], len(indices)) * rng.uniform(18.0, 32.0, len(indices))
    attacked.loc[indices, "relative_speed_mps"] = old_speeds + jamming_offsets

    logger.info(f"Applied doppler_velocity_spoofing to {len(indices)} points")
    return attacked, {
        "attack": "doppler_velocity_spoofing",
        "points_before": len(df),
        "points_after": len(attacked),
        "points_affected": len(indices),
        "mean_offset_mps": round(float(np.mean(np.abs(jamming_offsets))), 1),
    }


def spoof_range(df: pd.DataFrame, fraction: float = 0.30, random_state: int = SEED) -> Tuple[pd.DataFrame, Dict]:
    attacked = df.copy()
    valid = attacked.index[attacked["range_m"].notna() & (attacked["range_m"] > 0)]
    if len(valid) == 0:
        return attacked, {"attack": "range_spoofing", "points_affected": 0}

    rng = np.random.default_rng(random_state)
    n = max(1, int(len(valid) * fraction))
    indices = rng.choice(valid.to_numpy(), min(n, len(valid)), replace=False)

    before = attacked.loc[indices, "range_m"].astype(float).to_numpy()
    factors = rng.uniform(0.30, 0.70, len(indices))
    directions = rng.choice([-1.0, 1.0], len(indices))
    after = np.maximum(0.2, before * (1.0 + factors * directions))
    attacked.loc[indices, "range_m"] = after

    logger.info(f"Applied range_spoofing to {len(indices)} points")
    return attacked, {
        "attack": "range_spoofing",
        "points_before": len(df),
        "points_after": len(attacked),
        "points_affected": len(indices),
    }


def apply_dropout(df: pd.DataFrame, fraction: float = 0.45, random_state: int = SEED) -> Tuple[pd.DataFrame, Dict]:
    if df is None or len(df) <= 3:
        return df.copy(), {"attack": "sensor_dropout", "points_affected": 0}

    rng = np.random.default_rng(random_state)
    retain_mask = rng.uniform(0.0, 1.0, len(df)) >= fraction
    attacked = df[retain_mask].copy()

    logger.info(f"Applied sensor_dropout: dropped {len(df) - len(attacked)} points")
    return attacked, {
        "attack": "sensor_dropout",
        "points_before": len(df),
        "points_after": len(attacked),
        "points_affected": len(df) - len(attacked),
    }


ATTACK_REGISTRY = {
    "trajectory_drift": kinematic_trajectory_drift,
    "coherent_ghost": inject_coherent_ghost,
    "doppler_spoofing": spoof_doppler_velocity,
    "range_spoofing": spoof_range,
    "sensor_dropout": apply_dropout,
}


def apply_attack(df: pd.DataFrame, attack_name: str = None) -> Tuple[pd.DataFrame, Dict]:
    if not attack_name or attack_name == "none" or df is None:
        return df.copy(), {
            "attack": "none",
            "points_before": len(df) if df is not None else 0,
            "points_after": len(df) if df is not None else 0,
            "points_affected": 0,
            "message": "No radar attack applied.",
        }

    fn = ATTACK_REGISTRY.get(attack_name)
    if fn is None:
        logger.warning(f"Unknown radar attack: {attack_name}")
        return df.copy(), {
            "attack": attack_name,
            "points_affected": 0,
            "message": "Unknown attack; no modification applied.",
        }

    return fn(df)


def available_attacks() -> List[str]:
    return list(ATTACK_REGISTRY.keys())