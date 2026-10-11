# Datasets

This framework is designed against two public datasets:

| Folder | Dataset | Used for |
|---|---|---|
| `talk2car/` | [Talk2Car](https://talk2car.github.io/) | Natural-language passenger instructions paired with driving scenes |
| `nuscenes_mini/` | [nuScenes-mini](https://www.nuscenes.org/) | Camera images + Radar point returns (LiDAR is intentionally **not** used) |

## Why sample files ship here

The real Talk2Car / nuScenes-mini archives are multi-GB downloads that require registration and are not
reachable from this offline build environment. To keep the project runnable out of the box, small
**schema-accurate** sample files are included:

- `talk2car/sample_instructions.json` – 30 hand-written instructions in the exact Talk2Car command schema
  (`command`, `action`, `referred_object`), spanning trusted, ambiguous and adversarial/high-risk examples.
- `nuscenes_mini/sample_radar.csv` – radar returns in the nuScenes radar schema
  (`range_m`, `azimuth_deg`, `relative_speed_mps`, `rcs_db`) for a short multi-sweep scene.
- `nuscenes_mini/sample_camera/` – a couple of placeholder scene frames for the CV pipeline demo.

## Plugging in the real datasets

`backend/preprocessing/dataset_loader.py` already supports both formats. To use the full datasets:

1. Download Talk2Car JSON splits into `talk2car/` (keep the original filenames, e.g. `train.json`, `val.json`).
2. Download nuScenes-mini (`v1.0-mini`) into `nuscenes_mini/` following the official folder layout
   (`samples/CAM_FRONT`, `samples/RADAR_FRONT`, ...).
3. Restart the backend — `dataset_loader.py` auto-detects the richer schema and the sample files are ignored.

No code changes are required to switch from sample data to the full datasets.
