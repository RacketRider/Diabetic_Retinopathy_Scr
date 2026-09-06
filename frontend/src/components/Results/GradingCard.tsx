import type { GradingResult, ConfidenceResult } from '@/types';
import { SEVERITY_LABELS } from '@/types';
import { formatPercent, severityColorClass } from '@/utils/formatters';

interface GradingCardProps {
  grading: GradingResult;
  confidence: ConfidenceResult;
}

export function GradingCard({ grading, confidence }: GradingCardProps) {
  const colorClass = severityColorClass(grading.level);
  const label = SEVERITY_LABELS[grading.label] || grading.label;

  return (
    <div className={`rounded-xl border-2 p-5 ${colorClass}`}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-medium uppercase tracking-wide opacity-70">DR Severity</p>
          <p className="text-2xl font-bold mt-1">Level {grading.level} — {label}</p>
          <p className="text-sm mt-1 opacity-80">{grading.description}</p>
        </div>
        <div className="text-right">
          <p className="text-3xl font-bold">{formatPercent(confidence.referable_probability)}</p>
          <p className="text-xs opacity-70">referable prob.</p>
        </div>
      </div>

      {grading.is_referable && (
        <div className="mt-4 flex items-center gap-2 bg-white/50 rounded-lg px-3 py-2">
          <span className="text-lg">⚠️</span>
          <div>
            <p className="text-sm font-semibold">REFERABLE — Specialist review recommended</p>
            <p className="text-xs opacity-70">
              Decision threshold: {confidence.threshold_used} | Score: {formatPercent(confidence.referable_probability)}
            </p>
          </div>
        </div>
      )}

      {!grading.is_referable && (
        <div className="mt-4 flex items-center gap-2 bg-white/50 rounded-lg px-3 py-2">
          <span className="text-lg">✅</span>
          <p className="text-sm font-semibold">Non-referable — Routine follow-up</p>
        </div>
      )}
    </div>
  );
}
