import React, { useState } from "react";
import {
  FileText,
  Copy,
  Printer,
  RotateCcw,
  Check,
  Compass,
  ListOrdered,
} from "lucide-react";

export default function ExplanationSection({ result, onReset }) {
  const [copied, setCopied] = useState(false);

  if (!result) return null;

  const recommendation = result.recommendation || "Execution Blocked";
  const explanation = result.explanation || [];
  const score = result.overall_safety_score ?? 0;
  const risk = result.risk_level ?? "High";

  const handleCopy = () => {
    const summaryText = `[AegisDrive Audit Report]
Verdict: ${recommendation}
Overall Safety Score: ${score}/100 (${risk} Risk)
Original Instruction: "${result.instruction_original || ""}"
Evaluated Instruction: "${result.instruction_evaluated || ""}"

Reasoning Trail:
${explanation.map((line, idx) => `${idx + 1}. ${line}`).join("\n")}
`;
    navigator.clipboard.writeText(summaryText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handlePrint = () => {
    window.print();
  };

  return (
    <section className="space-y-6 animate-fadeIn pb-12">
      <div className="enterprise-card">
        {/* Title */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-xl bg-brand-500/10 border border-brand-500/25 flex items-center justify-center text-brand-400">
              <Compass className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-white">
                Driving Planner Action & Decision Reasoning
              </h3>
              <p className="text-xs text-slate-400">
                Transparent safety audit explaining why the driving planner accepted or overrode the command
              </p>
            </div>
          </div>

          {/* Action buttons */}
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={handleCopy}
              className="btn-secondary"
            >
              {copied ? (
                <>
                  <Check className="w-3.5 h-3.5 text-emerald-400 mr-1.5" />
                  <span className="text-emerald-400">Copied</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5 text-slate-400 mr-1.5" />
                  <span>Copy Summary</span>
                </>
              )}
            </button>
            <button
              type="button"
              onClick={handlePrint}
              className="btn-secondary hidden sm:inline-flex"
            >
              <Printer className="w-3.5 h-3.5 text-slate-400 mr-1.5" />
              <span>Print Report</span>
            </button>
          </div>
        </div>

        {/* Driving Action Directive */}
        <div className="p-4 rounded-xl bg-surface-subtle border border-surface-border mb-4">
          <div className="text-[10px] font-mono text-brand-400 uppercase tracking-wider mb-1 font-semibold">
            Autonomous Vehicle Driving Directive:
          </div>
          <div className="text-sm font-medium text-white">
            {recommendation === "Safe to Execute" && (
              <span className="text-emerald-300">
                PROCEED: Execute nominal driving trajectory. All sensor modalities verified authentic and cross-modally grounded.
              </span>
            )}
            {recommendation === "Require Human Confirmation" && (
              <span className="text-amber-300">
                HOLD & CLARIFY: Maintain lane position and prompt passenger for spoken confirmation before initiating maneuver.
              </span>
            )}
            {recommendation === "Execution Blocked" && (
              <span className="text-rose-300">
                SAFETY INTERVENTION: Override passenger command immediately. Execute certified emergency deceleration (a ≤ -3.5 m/s²) and bring vehicle to a controlled stop.
              </span>
            )}
          </div>
        </div>

        {/* Reasoning Trail List */}
        <div>
          <div className="flex items-center gap-2 text-xs font-mono text-slate-400 uppercase tracking-wider mb-2.5">
            <ListOrdered className="w-3.5 h-3.5 text-brand-400" />
            <span>Audited Decision Justifications:</span>
          </div>

          <div className="space-y-2">
            {explanation.length > 0 ? (
              explanation.map((item, idx) => (
                <div
                  key={idx}
                  className="flex items-start gap-3 p-3 rounded-xl bg-surface-subtle/60 border border-surface-border text-xs text-slate-200"
                >
                  <span className="w-5 h-5 rounded-lg bg-surface-raised border border-surface-border text-slate-400 font-mono text-[10px] font-bold flex items-center justify-center flex-shrink-0 mt-0.5">
                    {idx + 1}
                  </span>
                  <span className="leading-relaxed">{item}</span>
                </div>
              ))
            ) : (
              <div className="text-xs text-slate-400 p-3">
                No special warnings generated. All channels nominal.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Quick Reset / Test Another Scenario */}
      <div className="text-center pt-2">
        <button
          type="button"
          onClick={onReset}
          className="btn-secondary px-5 py-2.5 text-xs gap-2 font-medium"
        >
          <RotateCcw className="w-4 h-4 text-brand-400" />
          <span>Test Another Scenario</span>
        </button>
      </div>
    </section>
  );
}
