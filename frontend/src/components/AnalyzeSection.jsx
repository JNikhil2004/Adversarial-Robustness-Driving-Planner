import React, { useEffect, useState } from "react";
import { CheckCircle2, Loader2 } from "lucide-react";

export default function AnalyzeSection() {
  const [activeStage, setActiveStage] = useState(0);

  const stages = [
    {
      title: "Passenger Instruction Safety & Ambiguity",
      model: "Transformer Attention + Semantic Groundability Net",
      detail: "Scanning prompt injection patterns, safety bypasses, and deictic vagueness...",
    },
    {
      title: "Exterior Camera Scene Understanding",
      model: "Ultralytics YOLOv8 + Optical Laplacians",
      detail: "Detecting obstacle entities and verifying exposure and frame sharpness...",
    },
    {
      title: "Millimeter-Wave Radar Kinematics",
      model: "Bidirectional LSTM Sequence Autoencoder",
      detail: "Reconstructing 77 GHz returns and calculating sequence anomaly MSE...",
    },
    {
      title: "Cross-Modal Vision-Language Alignment",
      model: "Metric Space 64-D Latent Projection",
      detail: "Correlating instruction entities with visual bounding boxes and radar clusters...",
    },
    {
      title: "Decision Engine & Trajectory Mitigation",
      model: "Multimodal Safety Fuser + Fail-Safe Planner",
      detail: "Synthesizing safe trajectory and calculating epistemic uncertainty...",
    },
  ];

  useEffect(() => {
    const timer = setInterval(() => {
      setActiveStage((prev) => (prev < stages.length - 1 ? prev + 1 : prev));
    }, 650);
    return () => clearInterval(timer);
  }, [stages.length]);

  return (
    <div className="enterprise-card p-8 text-center max-w-xl mx-auto my-12 animate-fadeIn shadow-lg">
      <div className="w-12 h-12 rounded-2xl bg-brand-500/10 border border-brand-500/30 mx-auto flex items-center justify-center text-brand-400 mb-4">
        <Loader2 className="w-6 h-6 animate-spin text-brand-400" />
      </div>

      <h2 className="text-lg font-semibold text-white tracking-tight mb-1">
        Analyzing Multi-Modal Robustness
      </h2>
      <p className="text-xs text-slate-400 mb-6">
        Verifying sensor authentications and synthesizing fail-safe trajectories
      </p>

      {/* Verification Stages */}
      <div className="space-y-2 text-left">
        {stages.map((stage, idx) => {
          const isDone = idx < activeStage;
          const isCurrent = idx === activeStage;

          return (
            <div
              key={stage.title}
              className={`p-3 rounded-xl border transition-all duration-200 ${
                isCurrent
                  ? "bg-brand-600/10 border-brand-500/40 shadow-sm"
                  : isDone
                  ? "bg-surface-subtle/80 border-surface-border text-slate-200"
                  : "bg-surface-subtle/30 border-surface-border/40 opacity-40 text-slate-500"
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  {isDone ? (
                    <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                  ) : isCurrent ? (
                    <Loader2 className="w-4 h-4 text-brand-400 animate-spin flex-shrink-0" />
                  ) : (
                    <div className="w-4 h-4 rounded-full border border-surface-border flex-shrink-0" />
                  )}
                  <span
                    className={`text-xs font-medium ${
                      isCurrent
                        ? "text-brand-400 font-semibold"
                        : isDone
                        ? "text-slate-200"
                        : "text-slate-500"
                    }`}
                  >
                    {stage.title}
                  </span>
                </div>
                <span className="font-mono text-[10px] text-slate-500 hidden sm:inline">
                  {stage.model}
                </span>
              </div>
              {isCurrent && (
                <div className="text-[11px] text-slate-400 mt-1 ml-6.5 animate-fadeIn">
                  {stage.detail}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
