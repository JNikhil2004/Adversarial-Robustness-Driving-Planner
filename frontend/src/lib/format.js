/**
 * format.js -- clean presentation helpers for enterprise dashboard.
 */

export function scoreColor(score) {
  if (score >= 75) return "#10B981"; // emerald-500
  if (score >= 45) return "#F59E0B"; // amber-500
  return "#F43F5E"; // rose-500
}

export function scoreTextClass(score) {
  if (score >= 75) return "text-emerald-400";
  if (score >= 45) return "text-amber-400";
  return "text-rose-400";
}

export const RISK_BADGE_CLASS = {
  Low: "text-emerald-400 border-emerald-500/25 bg-emerald-500/10",
  Medium: "text-amber-400 border-amber-500/25 bg-amber-500/10",
  High: "text-rose-400 border-rose-500/25 bg-rose-500/10",
};

export const RECOMMENDATION_BADGE_CLASS = {
  "Safe to Execute": "text-emerald-400 border-emerald-500/25 bg-emerald-500/10",
  "Require Human Confirmation": "text-amber-400 border-amber-500/25 bg-amber-500/10",
  "Execution Blocked": "text-rose-400 border-rose-500/25 bg-rose-500/10",
};

export const PROMPT_LABEL_CLASS = {
  trusted: "text-emerald-400 border-emerald-500/25 bg-emerald-500/10",
  suspicious: "text-amber-400 border-amber-500/25 bg-amber-500/10",
  high_risk: "text-rose-400 border-rose-500/25 bg-rose-500/10",
};

export function titleCase(s) {
  return (s || "").replace(/\w\S*/g, (t) => t.charAt(0).toUpperCase() + t.slice(1));
}

export function humanizeSlug(s) {
  return titleCase((s || "").replaceAll("_", " "));
}
