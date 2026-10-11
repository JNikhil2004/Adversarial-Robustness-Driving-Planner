"""
run_benchmark.py
----------------
Standalone executable CLI runner for the Adversarial Robustness Benchmark
and Planner Safety Degradation Evaluation.

Runs:
  1. Prompt Robustness Comparison (Clean vs 4 Adversarial Attacks)
  2. Radar Robustness Comparison (Clean vs 4 Radar Sensor Perturbations)
  3. Camera Robustness Comparison (Clean vs 4 Visual Perturbations)
  4. Driving Planner Safety Degradation (Unguarded vs Guarded Planner on 15 Scenarios)

Generates:
  - outputs/failure_analysis_report.json
  - outputs/failure_analysis_report.md
  - Formatted terminal evaluation summary tables
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from backend.evaluation.benchmark import (
    run_prompt_robustness_comparison,
    run_radar_robustness_comparison,
    run_camera_robustness_comparison,
    run_planner_safety_benchmark,
)
from backend.utils.logger import get_logger

logger = get_logger(__name__)


def print_divider(title: str):
    print("\n" + "=" * 76)
    print(f"  {title.upper()}")
    print("=" * 76)


def main():
    print_divider("Language-Conditioned Driving Planner Robustness Benchmark")
    print("Executing Deep Neural Verification & Planner Safety Evaluation...\n")

    # 1. PROMPT BENCHMARK
    print_divider("1. Prompt Robustness Benchmark (Deep Transformer)")
    prompt_results = run_prompt_robustness_comparison()
    print(f"{'Attack':<24} | {'Accuracy':<10} | {'Mean Score':<12} | {'Acc Drop (%)':<12}")
    print("-" * 66)
    for name, res in prompt_results.items():
        acc = f"{res['metrics']['accuracy'] * 100:.1f}%"
        score = f"{res['score_summary']['mean']:.1f}/100"
        drop = f"{res.get('accuracy_drop', 0.0):+.1f}%"
        print(f"{name:<24} | {acc:<10} | {score:<12} | {drop:<12}")

    # 2. RADAR BENCHMARK
    print_divider("2. Radar Robustness Benchmark (Deep Autoencoder)")
    radar_results = run_radar_robustness_comparison()
    print(f"{'Attack':<20} | {'Health Score':<14} | {'Score Drop':<12} | {'Anomalies Flagged':<18}")
    print("-" * 72)
    clean_r = radar_results["clean"]["clean_result"]
    print(f"{'Clean Baseline':<20} | {clean_r['score']:<14.1f} | {'0.0':<12} | {len(clean_r.get('anomalies', [])):<18}")
    for name, res in radar_results.items():
        if name == "clean":
            continue
        att_res = res["attacked_result"]
        print(f"{name:<20} | {att_res['score']:<14.1f} | {res['score_drop']:<12.1f} | {res['anomalies_detected']:<18}")

    # 3. CAMERA BENCHMARK
    print_divider("3. Camera Robustness Benchmark (YOLOv8 + Quality)")
    camera_results = run_camera_robustness_comparison()
    print(f"{'Attack':<20} | {'Camera Score':<14} | {'Confidence Drop':<18} | {'Detections Lost':<16}")
    print("-" * 74)
    clean_c = camera_results["clean"]["clean_result"]
    print(f"{'Clean Baseline':<20} | {clean_c['camera_score']:<14.1f} | {'0.0%':<18} | {'0':<16}")
    for name, res in camera_results.items():
        if name == "clean":
            continue
        att_c = res["attacked_result"]
        print(f"{name:<20} | {att_c['camera_score']:<14.1f} | {res['confidence_drop_pct']:<17.1f}% | {res['detections_lost']:<16}")

    # 4. PLANNER SAFETY BENCHMARK
    print_divider("4. Planner Safety Degradation & Fail-Safe Mitigation (Deliverable 4)")
    planner_results = run_planner_safety_benchmark()
    summary = planner_results["summary"]
    print(f"Scenarios Evaluated              : {summary['total_planner_scenarios_evaluated']}")
    print(f"Unguarded Planner High-Risk Cases : {summary['unguarded_planner_high_collision_cases']}")
    print(f"Guarded Planner High-Risk Cases   : {summary['guarded_planner_high_collision_cases']}")
    print(f"Collision Hazard Prevention Rate  : {summary['collision_prevention_rate_pct']}%")
    print(f"Mean Trajectory Drift (ADE)       : {summary['mean_trajectory_drift_m']} meters")
    print(f"\nPersisted Failure Reports:")
    print(f"  -> JSON    : {planner_results['saved_report_paths']['json']}")
    print(f"  -> Markdown: {planner_results['saved_report_paths']['markdown']}")
    print("=" * 76)
    print("BENCHMARK EXECUTION COMPLETED SUCCESSFULLY.\n")


if __name__ == "__main__":
    main()
