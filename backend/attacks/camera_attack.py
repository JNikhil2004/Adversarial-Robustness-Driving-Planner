"""
camera_attack.py
------------------
Adversarial Camera & Perceptual Robustness Attack Suite.

Vectors:
1. L_infty High-Frequency Adversarial Noise: Gradient-like high-frequency signed perturbation (FGSM/PGD style).
2. Projected Adversarial Patch: Textured physical-style patch designed to suppress YOLO detections.
3. Atmospheric Fog & Scattering: Optical Koschmieder atmospheric attenuation.
4. Intense Sunlight Glare: Direct solar lens flare over-saturating dynamic range.
5. Motion Streaks: Directional camera vibration & high-velocity vehicle motion streaks.
6. Targeted Semantic Occlusion: Masks the central driving corridor and road obstacles.
"""

from typing import List
import cv2
import numpy as np

from backend.utils.logger import get_logger

logger = get_logger(__name__)


def add_linf_adversarial_noise(image: np.ndarray, epsilon: float = 24.0) -> np.ndarray:
    """
    Simulates an L_infty bounded high-frequency adversarial perturbation.
    Targets edge-gradients and high-frequency spatial components.
    """
    h, w, c = image.shape
    # Generate high-frequency signed gradient proxy
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_norm = np.sign(lap)[:, :, np.newaxis]
    noise = np.random.choice([-1.0, 1.0], size=(h, w, c)).astype(np.float32)
    delta = epsilon * (0.6 * lap_norm + 0.4 * noise)
    delta = np.clip(delta, -epsilon, epsilon)
    attacked = np.clip(image.astype(np.float32) + delta, 0, 255).astype(np.uint8)
    return attacked


def apply_projected_adversarial_patch(image: np.ndarray, patch_ratio: float = 0.22) -> np.ndarray:
    """
    Simulates a printed adversarial patch placed in the scene.
    Uses high-contrast chromatic textures that disrupt feature pyramid activations in YOLO.
    """
    h, w = image.shape[:2]
    patch_h, patch_w = max(40, int(h * patch_ratio)), max(40, int(w * patch_ratio))
    top = int(h * 0.45)
    left = int(w * 0.38)

    attacked = image.copy()
    # Synthesize colorful adversarial texture pattern
    grid = np.zeros((patch_h, patch_w, 3), dtype=np.uint8)
    step = 8
    for i in range(0, patch_h, step):
        for j in range(0, patch_w, step):
            color = [
                int((i * 25 + j * 17) % 256),
                int((255 - (i * 13 + j * 29) % 256)),
                int(((i + j) * 37) % 256),
            ]
            grid[i : i + step, j : j + step] = color

    attacked[top : top + patch_h, left : left + patch_w] = grid
    return attacked


def apply_atmospheric_fog(image: np.ndarray, density: float = 0.70) -> np.ndarray:
    """
    Simulates dense optical fog using Koschmieder's atmospheric scattering equation.
    Attenuates contrast exponentially with synthetic depth.
    """
    h, w = image.shape[:2]
    # Depth proxy: bottom of image is close (depth ~ 0), top/horizon is far (depth ~ 1)
    y_coords = np.linspace(0.9, 0.1, h)[:, np.newaxis]
    depth_map = np.repeat(y_coords, w, axis=1)
    transmission = np.exp(-density * (1.0 - depth_map))[:, :, np.newaxis]
    atmospheric_light = 230.0  # Bright white/gray sky haze

    foggy = image.astype(np.float32) * transmission + atmospheric_light * (1.0 - transmission)
    return np.clip(foggy, 0, 255).astype(np.uint8)


def apply_lens_glare(image: np.ndarray) -> np.ndarray:
    """
    Simulates intense solar lens flare blinding the vehicle forward camera.
    """
    h, w = image.shape[:2]
    glare_mask = np.zeros((h, w), dtype=np.float32)
    center = (int(w * 0.65), int(h * 0.25))
    radius = int(min(h, w) * 0.65)
    cv2.circle(glare_mask, center, radius, 1.0, -1)
    glare_mask = cv2.GaussianBlur(glare_mask, (151, 151), 0)

    glare_3c = glare_mask[:, :, np.newaxis]
    glare_color = np.array([255.0, 245.0, 220.0], dtype=np.float32)
    attacked = image.astype(np.float32) * (1.0 - 0.75 * glare_3c) + glare_color * (0.75 * glare_3c)
    return np.clip(attacked, 0, 255).astype(np.uint8)


def apply_motion_streaks(image: np.ndarray, kernel_size: int = 25) -> np.ndarray:
    """
    Simulates high-velocity chassis vibration and motion blur.
    """
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[(kernel_size - 1) // 2, :] = np.ones(kernel_size) / kernel_size
    return cv2.filter2D(image, -1, kernel)


def apply_targeted_occlusion(image: np.ndarray) -> np.ndarray:
    """
    Targeted black-box occlusion suppressing the primary visual road corridor.
    """
    h, w = image.shape[:2]
    top = int(h * 0.40)
    bottom = int(h * 0.85)
    left = int(w * 0.25)
    right = int(w * 0.75)

    attacked = image.copy()
    attacked[top:bottom, left:right] = 12
    return attacked


ATTACK_REGISTRY = {
    "adversarial_noise": add_linf_adversarial_noise,
    "adversarial_patch": apply_projected_adversarial_patch,
    "atmospheric_fog": apply_atmospheric_fog,
    "lens_glare": apply_lens_glare,
    "motion_streaks": apply_motion_streaks,
    "targeted_occlusion": apply_targeted_occlusion,
}


def apply_attack(image: np.ndarray, attack_name: str) -> np.ndarray:
    if image is None or not attack_name or attack_name == "none":
        return image
    fn = ATTACK_REGISTRY.get(attack_name)
    if fn is None:
        logger.warning(f"Unknown camera attack '{attack_name}', skipping")
        return image
    logger.info(f"Applied camera attack: {attack_name}")
    return fn(image)


def available_attacks() -> List[str]:
    return list(ATTACK_REGISTRY.keys())
