import React from "react";
import { ShieldCheck, Sparkles, RotateCcw, CheckCircle } from "lucide-react";

export default function TopHeader({ onLoadSample, onReset, isEvaluating, hasResults }) {
  return (
    <header className="border-b border-surface-border bg-canvas/85 backdrop-blur-md sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        {/* Brand & Title */}
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-brand-600/15 border border-brand-500/30 flex items-center justify-center text-brand-400">
            <ShieldCheck className="w-5 h-5 text-brand-400" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-semibold tracking-tight text-white font-sans">
                AegisDrive
              </h1>
              <span className="font-mono text-[10px] px-2 py-0.5 rounded-full bg-surface-raised border border-surface-border text-slate-400 font-medium">
                Research Studio
              </span>
            </div>
            <p className="text-xs text-slate-400 hidden sm:block">
              Adversarial Robustness Verification for Autonomous Driving Planners
            </p>
          </div>
        </div>

        {/* Status Pill & Action Buttons */}
        <div className="flex items-center gap-3">
          <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface-subtle border border-surface-border text-xs text-slate-400">
            <span className="w-2 h-2 rounded-full bg-emerald-400" />
            <span>nuScenes & Talk2Car Pipeline</span>
          </div>

          <button
            type="button"
            onClick={onLoadSample}
            disabled={isEvaluating}
            className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-brand-600 hover:bg-brand-500 text-white text-xs font-medium transition-all shadow-sm disabled:opacity-50"
            title="Automatically load sample camera frame, radar point-cloud, and instruction presets"
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Load Sample Scenario</span>
          </button>

          {hasResults && (
            <button
              type="button"
              onClick={onReset}
              disabled={isEvaluating}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-surface-raised hover:bg-surface-hover border border-surface-border text-slate-300 text-xs font-medium transition-all"
              title="Reset evaluation"
            >
              <RotateCcw className="w-3.5 h-3.5 text-slate-400" />
              <span>Reset</span>
            </button>
          )}
        </div>
      </div>
    </header>
  );
}
