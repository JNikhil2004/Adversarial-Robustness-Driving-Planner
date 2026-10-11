from pathlib import Path
import random
import shutil

from nuscenes.nuscenes import NuScenes


# Your actual NuScenes directory
NUSCENES_DIR = Path(
    r"C:\Adversarial-Robustness-Driving-Planner(1)\datasets\nuscenes"
)

# Save the selected examples here
OUTPUT_DIR = Path(
    r"C:\Adversarial-Robustness-Driving-Planner(1)\test_inputs\random_samples"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load NuScenes
nusc = NuScenes(
    version="v1.0-mini",
    dataroot=str(NUSCENES_DIR),
    verbose=False
)


# Pick 5 different NuScenes samples
samples = random.sample(nusc.sample, 5)


for i, sample in enumerate(samples, 1):

    camera_token = sample["data"]["CAM_FRONT"]
    radar_token = sample["data"]["RADAR_FRONT"]

    camera_path = Path(
        nusc.get_sample_data_path(camera_token)
    )

    radar_path = Path(
        nusc.get_sample_data_path(radar_token)
    )

    # Folder for this test case
    test_folder = OUTPUT_DIR / f"test_{i}"
    test_folder.mkdir(parents=True, exist_ok=True)

    # Copy camera
    shutil.copy2(
        camera_path,
        test_folder / f"camera_{i}.jpg"
    )

    # Copy radar
    shutil.copy2(
        radar_path,
        test_folder / f"radar_{i}.pcd"
    )

    print(f"\nTest {i}")
    print("Camera:", camera_path)
    print("Radar :", radar_path)


print("\n================================")
print("5 test cases saved successfully")
print("Location:")
print(OUTPUT_DIR)
print("================================")