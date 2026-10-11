import React from "react";
import {
  ShieldCheck,
  ShieldAlert,
  ShieldX,
  MessageSquare,
  Camera,
  Radio,
  GitCompare,
  Activity,
  CheckCircle2,
  AlertTriangle,
} from "lucide-react";
import { scoreColor, scoreTextClass, RISK_BADGE_CLASS } from "../lib/format";

function TrajectoryMap({ unguardedWaypoints, guardedWaypoints }) {
  const allY = [
    ...(unguardedWaypoints || []).map((w) => w.y_forward_m || 0),
    ...(guardedWaypoints || []).map((w) => w.y_forward_m || 0),
    35,
  ];
  const maxY = Math.max(...allY, 35);

  const toSvgX = (x) => 200 + (x || 0) * 22;
  const toSvgY = (y) => 155 - ((y || 0) / maxY) * 135;

  const buildPath = (pts) => {
    if (!pts || pts.length === 0) return "";
    let d = `M 200 155`;
    pts.forEach((p) => {
      d += ` L ${toSvgX(p.x_offset_m)} ${toSvgY(p.y_forward_m)}`;
    });
    return d;
  };

  const unguardedPath = buildPath(unguardedWaypoints);
  const guardedPath = buildPath(guardedWaypoints);

  return (
    <div className="rounded-xl border border-surface-border bg-slate-950/80 p-3.5 mb-4 overflow-hidden">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-2">
        <span className="font-mono text-[10px] text-slate-400 uppercase tracking-wider">
          Overhead 5-Second Waypoint Path Map
        </span>
        <div className="flex items-center gap-4 text-[10px] font-mono">
          <span className="flex items-center gap-1.5 text-rose-400">
            <span className="w-2.5 h-0.5 bg-rose-500 rounded" />
            Unguarded Trajectory
          </span>
          <span className="flex items-center gap-1.5 text-emerald-400">
            <span className="w-2.5 h-0.5 bg-emerald-500 rounded" />
            Guarded (AegisDrive Fail-Safe)
          </span>
        </div>
      </div>

      <svg viewBox="0 0 400 175" className="w-full h-36">
        {/* Road Corridor */}
        <rect x="120" y="8" width="160" height="155" rx="6" fill="#0b1120" stroke="#1e293b" strokeWidth="1" />
        {/* Road Edge Boundaries */}
        <line x1="120" y1="12" x2="120" y2="160" stroke="#475569" strokeWidth="1.5" />
        <line x1="280" y1="12" x2="280" y2="160" stroke="#475569" strokeWidth="1.5" />
        {/* Lane Center Dashed Line */}
        <line x1="200" y1="12" x2="200" y2="160" stroke="#334155" strokeDasharray="5 5" strokeWidth="1.5" />

        {/* Distance Grid Labels */}
        <text x="125" y="24" fill="#64748b" fontSize="8" fontFamily="monospace">+{Math.round(maxY)}m</text>
        <text x="125" y="88" fill="#64748b" fontSize="8" fontFamily="monospace">+{Math.round(maxY / 2)}m</text>

        {/* Unguarded Trajectory Path & Dots */}
        {unguardedPath && (
          <>
            <path d={unguardedPath} fill="none" stroke="#f43f5e" strokeWidth="2.5" strokeDasharray="4 3" opacity="0.85" />
            {(unguardedWaypoints || []).map((w, idx) => (
              <circle
                key={`u-${idx}`}
                cx={toSvgX(w.x_offset_m)}
                cy={toSvgY(w.y_forward_m)}
                r="3.5"
                fill="#f43f5e"
              >
                <title>{`Unguarded Step ${w.step}: +${w.y_forward_m}m forward, ${w.x_offset_m}m lateral`}</title>
              </circle>
            ))}
          </>
        )}

        {/* Guarded Trajectory Path & Dots */}
        {guardedPath && (
          <>
            <path d={guardedPath} fill="none" stroke="#10b981" strokeWidth="2.5" opacity="0.95" />
            {(guardedWaypoints || []).map((w, idx) => (
              <circle
                key={`g-${idx}`}
                cx={toSvgX(w.x_offset_m)}
                cy={toSvgY(w.y_forward_m)}
                r="4"
                fill="#10b981"
                stroke="#064e3b"
                strokeWidth="1.5"
              >
                <title>{`Guarded Step ${w.step}: +${w.y_forward_m}m forward, ${w.x_offset_m}m lateral`}</title>
              </circle>
            ))}
          </>
        )}

        {/* Ego Vehicle Icon */}
        <rect x="194" y="146" width="12" height="18" rx="2.5" fill="#38bdf8" stroke="#0284c7" strokeWidth="1" />
        <circle cx="200" cy="155" r="1.5" fill="#ffffff" />
        <text x="200" y="171" textAnchor="middle" fill="#94a3b8" fontSize="8" fontFamily="monospace font-bold">EGO</text>
      </svg>
    </div>
  );
}

export default function ResultsSection({ result }) {
  if (!result) return null;

  const score = result.overall_safety_score ?? 0;
  const risk = result.risk_level ?? "High";
  const recommendation = result.recommendation ?? "Execution Blocked";
  const uncertainty = result.epistemic_uncertainty ?? 0.05;

  const isSafe = recommendation === "Safe to Execute";
  const isCaution = recommendation === "Require Human Confirmation";
  const isBlocked = recommendation === "Execution Blocked";

  const promptRisk = result.prompt_risk || {};
  const clarity = result.instruction_clarity || {};
  const scene = result.scene_understanding || {};
  const radar = result.radar_motion || {};
  const crossModal = result.cross_modal || {};
  const trajectories = result.trajectories || {};

  return (
    <section className="space-y-6 animate-fadeIn">
      {/* 1. EXECUTIVE VERDICT HERO BANNER */}
      <div
        className={`enterprise-card border-2 ${
          isSafe
            ? "border-emerald-500/30 bg-emerald-500/[0.04]"
            : isCaution
            ? "border-amber-500/30 bg-amber-500/[0.04]"
            : "border-rose-500/30 bg-rose-500/[0.04]"
        }`}
      >
        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          {/* Verdict Status */}
          <div className="flex items-center gap-4 text-center md:text-left">
            <div
              className={`w-14 h-14 rounded-2xl flex items-center justify-center flex-shrink-0 ${
                isSafe
                  ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30"
                  : isCaution
                  ? "bg-amber-500/15 text-amber-400 border border-amber-500/30"
                  : "bg-rose-500/15 text-rose-400 border border-rose-500/30"
              }`}
            >
              {isSafe ? (
                <ShieldCheck className="w-8 h-8" />
              ) : isCaution ? (
                <ShieldAlert className="w-8 h-8" />
              ) : (
                <ShieldX className="w-8 h-8" />
              )}
            </div>

            <div>
              <div className="flex flex-wrap items-center justify-center md:justify-start gap-2 mb-1.5">
                <span className="font-mono text-xs text-slate-400 uppercase tracking-wider">
                  Planner Decision Verdict
                </span>
                <span
                  className={`font-mono text-[11px] font-semibold px-2.5 py-0.5 rounded-full border ${
                    RISK_BADGE_CLASS[risk] || RISK_BADGE_CLASS.High
                  }`}
                >
                  {risk} Risk Level
                </span>
                <span className="font-mono text-[11px] px-2 py-0.5 rounded-full bg-surface-subtle border border-surface-border text-slate-400">
                  Uncertainty: ±{(uncertainty * 100).toFixed(1)}%
                </span>
              </div>
              <h2
                className={`text-2xl font-bold tracking-tight ${
                  isSafe
                    ? "text-emerald-400"
                    : isCaution
                    ? "text-amber-400"
                    : "text-rose-400"
                }`}
              >
                {recommendation}
              </h2>
              <p className="text-xs text-slate-300 mt-1 max-w-xl">
                {isSafe
                  ? "All sensor channels verified nominal and cross-modally consistent. Trajectory approved."
                  : isCaution
                  ? "Semantic ambiguity or moderate sensor inconsistency detected. Human confirmation required."
                  : "Safety policy override active. Certified emergency deceleration and fail-safe stop engaged."}
              </p>
            </div>
          </div>

          {/* Safety Gauge */}
          <div className="flex items-center gap-4 bg-surface-subtle border border-surface-border rounded-xl px-5 py-3 flex-shrink-0">
            <div className="text-right">
              <div className="font-mono text-[10px] text-slate-400 uppercase tracking-wider">
                Overall Safety Score
              </div>
              <div className={`text-3xl font-extrabold tracking-tight ${scoreTextClass(score)}`}>
                {score}
                <span className="text-xs text-slate-500 font-normal"> / 100</span>
              </div>
              <div className="text-[10px] font-mono text-slate-500">
                Safe Threshold: ≥ 75.0
              </div>
            </div>

            {/* Circular Gauge Ring */}
            <div className="relative w-14 h-14 flex items-center justify-center">
              <svg className="w-14 h-14 transform -rotate-90">
                <circle
                  cx="28"
                  cy="28"
                  r="23"
                  stroke="#1E2B42"
                  strokeWidth="4"
                  fill="transparent"
                />
                <circle
                  cx="28"
                  cy="28"
                  r="23"
                  stroke={scoreColor(score)}
                  strokeWidth="4"
                  strokeDasharray={144.51}
                  strokeDashoffset={144.51 - (144.51 * Math.min(100, Math.max(0, score))) / 100}
                  strokeLinecap="round"
                  className="transition-all duration-700 ease-out"
                  fill="transparent"
                />
              </svg>
              <span className={`absolute text-xs font-mono font-bold ${scoreTextClass(score)}`}>
                {Math.round(score)}%
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* 2. MULTI-MODAL DIAGNOSTICS (4 CARDS GRID) */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* CARD 1: PASSENGER LANGUAGE */}
        <div className="enterprise-card">
          <div className="flex items-center justify-between mb-3.5">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-xl bg-brand-500/10 border border-brand-500/25 flex items-center justify-center text-brand-400">
                <MessageSquare className="w-4 h-4" />
              </div>
              <div>
                <h3 className="text-sm font-semibold text-white">
                  1. Passenger Language Channel
                </h3>
                <p className="text-xs text-slate-400">
                  Transformer Risk + Semantic Groundability
                </p>
              </div>
            </div>
            <div className="text-right">
              <span className={`text-base font-bold ${scoreTextClass(promptRisk.score || 0)}`}>
                {promptRisk.score || 0}
              </span>
              <span className="text-[10px] text-slate-500"> / 100</span>
            </div>
          </div>

          <div className="space-y-2.5 text-xs">
            <div className="flex items-center justify-between p-2.5 rounded-xl bg-surface-subtle border border-surface-border">
              <span className="text-slate-400">Classification:</span>
              <span
                className={`font-mono text-xs font-semibold px-2 py-0.5 rounded-lg ${
                  promptRisk.label === "trusted"
                    ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/25"
                    : promptRisk.label === "suspicious"
                    ? "bg-amber-500/15 text-amber-400 border border-amber-500/25"
                    : "bg-rose-500/15 text-rose-400 border border-rose-500/25"
                }`}
              >
                {promptRisk.label?.toUpperCase() || "UNKNOWN"}
              </span>
            </div>

            <div className="flex items-center justify-between p-2.5 rounded-xl bg-surface-subtle border border-surface-border">
              <span className="text-slate-400">Ambiguity Status:</span>
              <span className="font-mono text-xs">
                {clarity.is_ambiguous ? (
                  <span className="text-amber-400 font-medium">Ambiguous Referent</span>
                ) : (
                  <span className="text-emerald-400 font-medium">Grounded & Clear</span>
                )}
              </span>
            </div>

            {/* Token Attention Saliency Chips */}
            {promptRisk.token_saliency && promptRisk.token_saliency.length > 0 && (
              <div className="p-3 rounded-xl bg-surface-subtle border border-surface-border">
                <div className="text-[10px] font-mono text-slate-400 uppercase tracking-wider mb-2">
                  Token Attention Saliency (Transformer Weights):
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {promptRisk.token_saliency.map((tok, i) => (
                    <span
                      key={i}
                      className={`text-[11px] font-mono px-2 py-0.5 rounded-lg transition-all ${
                        tok.is_high_risk || tok.saliency > 0.8
                          ? "bg-rose-500/20 text-rose-300 border border-rose-500/40 font-semibold"
                          : tok.saliency > 0.5
                          ? "bg-amber-500/15 text-amber-300 border border-amber-500/30"
                          : "bg-surface-raised border border-surface-border text-slate-300"
                      }`}
                      title={`Saliency: ${tok.saliency?.toFixed(2)}`}
                    >
                      {tok.token}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Trigger Words Flagged */}
            {promptRisk.trigger_terms && promptRisk.trigger_terms.length > 0 && (
              <div className="p-2.5 rounded-xl bg-rose-500/10 border border-rose-500/25 text-rose-300 text-xs">
                <span className="font-semibold">Detected Injection Triggers: </span>
                {promptRisk.trigger_terms.join(", ")}
              </div>
            )}
          </div>
        </div>

        {/* CARD 2: CAMERA & VISION */}
        <div className="enterprise-card">
          <div className="flex items-center justify-between mb-3.5">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-xl bg-blue-500/10 border border-blue-500/25 flex items-center justify-center text-blue-400">
                <Camera className="w-4 h-4" />
              </div>
              <div>
                <h3 className="text-sm font-semibold text-white">
                  2. Camera & Vision Channel
                </h3>
                <p className="text-xs text-slate-400">
                  YOLOv8 Detection + Optical Quality
                </p>
              </div>
            </div>
            <div className="text-right">
              <span className={`text-base font-bold ${scoreTextClass(scene.score || 0)}`}>
                {scene.score || 0}
              </span>
              <span className="text-[10px] text-slate-500"> / 100</span>
            </div>
          </div>

          <div className="space-y-2.5 text-xs">
            <div className="flex items-center justify-between p-2.5 rounded-xl bg-surface-subtle border border-surface-border">
              <span className="text-slate-400">Detected Obstacles:</span>
              <span className="font-mono text-xs font-semibold text-white">
                {scene.detection_count || 0} object(s) ({scene.mean_confidence ? `${Math.round(scene.mean_confidence * 100)}% conf` : "N/A"})
              </span>
            </div>

            {/* Object Classes Found */}
            {scene.detected_objects && scene.detected_objects.length > 0 && (
              <div className="flex flex-wrap gap-1.5 p-2.5 rounded-xl bg-surface-subtle border border-surface-border">
                {scene.detected_objects.map((cls, i) => (
                  <span
                    key={i}
                    className="font-mono text-[10px] px-2 py-0.5 rounded-md bg-blue-500/10 border border-blue-500/25 text-blue-300 capitalize"
                  >
                    {cls}
                  </span>
                ))}
              </div>
            )}

            {/* Optical Lens Quality */}
            <div className="flex items-center justify-between p-2.5 rounded-xl bg-surface-subtle border border-surface-border">
              <span className="text-slate-400">Lens Sharpness:</span>
              <span className="font-mono text-xs">
                {scene.quality?.is_blurry ? (
                  <span className="text-rose-400 font-semibold">Blur Anomaly ({scene.quality?.sharpness})</span>
                ) : (
                  <span className="text-emerald-400 font-medium">Optimal ({scene.quality?.sharpness || "Clear"})</span>
                )}
              </span>
            </div>

            {/* Side-by-Side Image Preview (Original vs Attacked) */}
            {scene.attacked_image && (
              <div className="pt-2 border-t border-surface-border">
                <div className="text-[10px] font-mono text-slate-400 uppercase tracking-wider mb-2 flex items-center justify-between">
                  <span>Attack Visualization:</span>
                  <span className="text-rose-400 font-semibold uppercase">{scene.attack_applied}</span>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div className="rounded-lg overflow-hidden border border-surface-border">
                    <img
                      src={`data:image/jpeg;base64,${scene.original_image}`}
                      alt="Original frame"
                      className="w-full h-20 object-cover"
                    />
                    <div className="text-[10px] text-center py-0.5 bg-black/70 text-slate-400 font-mono">
                      Original
                    </div>
                  </div>
                  <div className="rounded-lg overflow-hidden border border-rose-500/40">
                    <img
                      src={`data:image/jpeg;base64,${scene.attacked_image}`}
                      alt="Attacked frame"
                      className="w-full h-20 object-cover"
                    />
                    <div className="text-[10px] text-center py-0.5 bg-rose-500/20 text-rose-300 font-mono font-medium">
                      Perturbed
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* CARD 3: RADAR KINEMATICS */}
        <div className="enterprise-card">
          <div className="flex items-center justify-between mb-3.5">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-xl bg-purple-500/10 border border-purple-500/25 flex items-center justify-center text-purple-400">
                <Radio className="w-4 h-4" />
              </div>
              <div>
                <h3 className="text-sm font-semibold text-white">
                  3. Radar Kinematics Channel
                </h3>
                <p className="text-xs text-slate-400">
                  Temporal Autoencoder Sequence Anomaly
                </p>
              </div>
            </div>
            <div className="text-right">
              <span className={`text-base font-bold ${scoreTextClass(radar.score || 0)}`}>
                {radar.score || 0}
              </span>
              <span className="text-[10px] text-slate-500"> / 100</span>
            </div>
          </div>

          <div className="space-y-2.5 text-xs">
            <div className="flex items-center justify-between p-2.5 rounded-xl bg-surface-subtle border border-surface-border">
              <span className="text-slate-400">Total Valid Tracks:</span>
              <span className="font-mono text-xs font-semibold text-white">
                {radar.total_tracks || 0} tracks ({radar.anomalous_tracks || 0} flagged)
              </span>
            </div>

            <div className="flex items-center justify-between p-2.5 rounded-xl bg-surface-subtle border border-surface-border">
              <span className="text-slate-400">Reconstruction Error:</span>
              <span className="font-mono text-xs">
                <span
                  className={
                    radar.reconstruction_mse > radar.reconstruction_threshold
                      ? "text-rose-400 font-semibold"
                      : "text-emerald-400 font-medium"
                  }
                >
                  MSE {radar.reconstruction_mse?.toFixed(3) || "0.000"}
                </span>
                <span className="text-slate-500"> (Threshold: {radar.reconstruction_threshold})</span>
              </span>
            </div>

            {/* Radar Anomalies */}
            {radar.anomalies && radar.anomalies.length > 0 ? (
              <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/25 text-rose-300 text-xs space-y-1">
                <div className="font-semibold">Flagged Kinematic Anomalies:</div>
                {radar.anomalies.slice(0, 2).map((anom, i) => (
                  <div key={i} className="text-[11px] text-rose-200">
                    • {anom}
                  </div>
                ))}
              </div>
            ) : (
              <div className="p-2.5 rounded-xl bg-emerald-500/10 border border-emerald-500/25 text-emerald-300 text-xs">
                ✓ Nominal Doppler velocities and physical bounds verified.
              </div>
            )}
          </div>
        </div>

        {/* CARD 4: CROSS-MODAL CONSISTENCY */}
        <div className="enterprise-card">
          <div className="flex items-center justify-between mb-3.5">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-xl bg-emerald-500/10 border border-emerald-500/25 flex items-center justify-center text-emerald-400">
                <GitCompare className="w-4 h-4" />
              </div>
              <div>
                <h3 className="text-sm font-semibold text-white">
                  4. Cross-Modal Consistency & Grounding
                </h3>
                <p className="text-xs text-slate-400">
                  Vision-Language Metric Alignment (64-D)
                </p>
              </div>
            </div>
            <div className="text-right">
              <span className={`text-base font-bold ${scoreTextClass(crossModal.consistency_score || 0)}`}>
                {crossModal.consistency_score || 0}
              </span>
              <span className="text-[10px] text-slate-500"> / 100</span>
            </div>
          </div>

          <div className="space-y-2.5 text-xs">
            <div className="flex items-center justify-between p-2.5 rounded-xl bg-surface-subtle border border-surface-border">
              <span className="text-slate-400">VL Metric Grounding:</span>
              <span className="font-mono text-xs font-semibold text-emerald-400">
                {Math.round((crossModal.vl_grounding_score || 0.85) * 100)}% Match
              </span>
            </div>

            <div className="flex items-center justify-between p-2.5 rounded-xl bg-surface-subtle border border-surface-border">
              <span className="text-slate-400">Instruction Target Grounded:</span>
              <span className="font-mono text-xs">
                {crossModal.unmatched_instruction_reference ? (
                  <span className="text-rose-400 font-semibold">Unmatched Referent (Distractor)</span>
                ) : (
                  <span className="text-emerald-400 font-medium">Verified in Scene</span>
                )}
              </span>
            </div>

            {crossModal.matched_objects && crossModal.matched_objects.length > 0 && (
              <div className="p-2.5 rounded-xl bg-surface-subtle border border-surface-border text-xs">
                <span className="text-slate-400">Grounded Targets: </span>
                <span className="text-slate-200 font-mono font-medium">
                  {crossModal.matched_objects.join(", ")}
                </span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* 3. TRAJECTORY MITIGATION CARD */}
      {trajectories.guarded && (
        <div className="enterprise-card">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-xl bg-brand-500/10 border border-brand-500/25 flex items-center justify-center text-brand-400">
                <Activity className="w-4 h-4" />
              </div>
              <div>
                <h3 className="text-sm font-semibold text-white">
                  Driving Planner Trajectory Defense (Fail-Safe Synthesis)
                </h3>
                <p className="text-xs text-slate-400">
                  Comparison between naive unguarded planner vs. AegisDrive mitigated trajectory
                </p>
              </div>
            </div>
          </div>

          {/* Overhead Waypoint 2D Visualizer */}
          <TrajectoryMap
            unguardedWaypoints={trajectories.unguarded?.waypoints}
            guardedWaypoints={trajectories.guarded?.waypoints}
          />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* UNGUARDED PLANNER */}
            <div className="p-4 rounded-xl border border-rose-500/25 bg-rose-500/[0.04]">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold text-rose-300">
                  Unguarded Planner (Baseline)
                </span>
                <span className="font-mono text-[10px] px-2 py-0.5 rounded-full bg-rose-500/15 border border-rose-500/30 text-rose-300 font-medium">
                  {trajectories.unguarded?.action_executed || trajectories.unguarded?.hazard_status || "HAZARDOUS"}
                </span>
              </div>
              <p className="text-xs text-slate-300 mb-3">
                {trajectories.unguarded?.reason || trajectories.unguarded?.action_summary || "Executes passenger command directly without sensor verification."}
              </p>
              <div className="grid grid-cols-2 gap-2 text-xs font-mono p-2.5 rounded-lg bg-surface-subtle border border-rose-500/20">
                <div>
                  <span className="text-slate-400">Target Speed:</span>{" "}
                  <span className="text-rose-300 font-semibold">{trajectories.unguarded?.terminal_speed_mps ?? trajectories.unguarded?.target_speed_mps ?? 22.5} m/s</span>
                </div>
                <div>
                  <span className="text-slate-400">Collision Risk:</span>{" "}
                  <span className="text-rose-300 font-semibold">
                    {trajectories.unguarded?.collision_risk_prob != null
                      ? `${Math.round(trajectories.unguarded.collision_risk_prob * 100)}%`
                      : "High"}
                  </span>
                </div>
                <div>
                  <span className="text-slate-400">Safety Score:</span>{" "}
                  <span className="text-rose-300 font-semibold">{trajectories.unguarded?.trajectory_safety_score ?? 10.0}/100</span>
                </div>
                <div>
                  <span className="text-slate-400">Min TTC:</span>{" "}
                  <span className="text-rose-300 font-semibold">{trajectories.unguarded?.min_time_to_collision_sec != null ? `${trajectories.unguarded.min_time_to_collision_sec}s` : "1.8s"}</span>
                </div>
              </div>
            </div>

            {/* GUARDED PLANNER */}
            <div className="p-4 rounded-xl border border-emerald-500/25 bg-emerald-500/[0.04]">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold text-emerald-300">
                  Guarded Planner (AegisDrive Defense)
                </span>
                <span className="font-mono text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/15 border border-emerald-500/30 text-emerald-300 font-medium">
                  {trajectories.guarded?.action_executed || trajectories.guarded?.hazard_status || "MITIGATED"}
                </span>
              </div>
              <p className="text-xs text-slate-300 mb-3">
                {trajectories.guarded?.reason || trajectories.guarded?.action_summary || "Intercepts adversarial command and executes certified fail-safe deceleration."}
              </p>
              <div className="grid grid-cols-2 gap-2 text-xs font-mono p-2.5 rounded-lg bg-surface-subtle border border-emerald-500/20">
                <div>
                  <span className="text-slate-400">Safe Speed:</span>{" "}
                  <span className="text-emerald-300 font-semibold">{trajectories.guarded?.terminal_speed_mps ?? trajectories.guarded?.target_speed_mps ?? 0.0} m/s</span>
                </div>
                <div>
                  <span className="text-slate-400">Collision Risk:</span>{" "}
                  <span className="text-emerald-300 font-semibold">
                    {trajectories.guarded?.collision_risk_prob != null
                      ? `${Math.round(trajectories.guarded.collision_risk_prob * 100)}%`
                      : "Low"}
                  </span>
                </div>
                <div>
                  <span className="text-slate-400">Safety Score:</span>{" "}
                  <span className="text-emerald-300 font-semibold">{trajectories.guarded?.trajectory_safety_score ?? 95.0}/100</span>
                </div>
                <div>
                  <span className="text-slate-400">Min TTC:</span>{" "}
                  <span className="text-emerald-300 font-semibold">
                    {trajectories.guarded?.min_time_to_collision_sec != null
                      ? trajectories.guarded.min_time_to_collision_sec >= 90
                        ? "Safe (Clear)"
                        : `${trajectories.guarded.min_time_to_collision_sec}s`
                      : "Safe"}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
