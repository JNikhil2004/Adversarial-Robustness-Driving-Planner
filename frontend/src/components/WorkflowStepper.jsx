import React from "react";
import { Edit3, Cpu, CheckCircle2, FileText } from "lucide-react";

export default function WorkflowStepper({ currentStep, hasResults, onStepClick }) {
  const steps = [
    {
      id: "input",
      num: 1,
      label: "Input",
      subtext: "Scenario & Attacks",
      icon: Edit3,
    },
    {
      id: "analyze",
      num: 2,
      label: "Analyze",
      subtext: "Multi-Modal Engine",
      icon: Cpu,
    },
    {
      id: "results",
      num: 3,
      label: "Results",
      subtext: "Score & Diagnostics",
      icon: CheckCircle2,
    },
    {
      id: "explanation",
      num: 4,
      label: "Explanation",
      subtext: "Directive & Audit",
      icon: FileText,
    },
  ];

  return (
    <nav className="w-full bg-surface-subtle/80 border border-surface-border rounded-2xl p-1.5 backdrop-blur-sm shadow-subtle mb-6">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-1.5">
        {steps.map((step) => {
          const isActive = currentStep === step.id;
          const isDone =
            hasResults &&
            (step.id === "input" ||
              step.id === "analyze" ||
              (step.id === "results" && currentStep === "explanation"));

          const isClickable =
            step.id === "input" ||
            (hasResults && (step.id === "results" || step.id === "explanation"));

          return (
            <button
              key={step.id}
              type="button"
              onClick={() => isClickable && onStepClick && onStepClick(step.id)}
              disabled={!isClickable}
              className={`flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-left transition-all duration-150 ${
                isActive
                  ? "bg-surface-raised border border-brand-500/40 text-white shadow-sm"
                  : isDone
                  ? "bg-transparent text-slate-300 hover:bg-surface-raised/50"
                  : "bg-transparent text-slate-500 opacity-50 cursor-default"
              }`}
            >
              <div
                className={`w-6 h-6 rounded-lg flex items-center justify-center font-mono text-xs font-semibold transition-colors ${
                  isActive
                    ? "bg-brand-600 text-white"
                    : isDone
                    ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/25"
                    : "bg-surface-border text-slate-500"
                }`}
              >
                {isDone && !isActive ? "✓" : step.num}
              </div>
              <div className="min-w-0">
                <div className={`text-xs font-medium ${isActive ? "text-white" : "text-slate-300"}`}>
                  {step.label}
                </div>
                <div className="text-[10px] text-slate-400 truncate">
                  {step.subtext}
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </nav>
  );
}
