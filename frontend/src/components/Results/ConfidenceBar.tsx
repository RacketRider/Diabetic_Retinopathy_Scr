import type { ConfidenceResult } from '@/types';
import { SEVERITY_COLORS, SEVERITY_LABELS } from '@/types';
import { formatPercent } from '@/utils/formatters';

interface ConfidenceBarProps {
  confidence: ConfidenceResult;
  predictedClass: string;
}

export function ConfidenceBar({ confidence, predictedClass }: ConfidenceBarProps) {
  const entries = Object.entries(confidence.class_probabilities)
    .sort(([, a], [, b]) => b - a);

  return (
    <div className="space-y-2">
      {entries.map(([cls, prob]) => {
        const color = SEVERITY_COLORS[cls] || '#9ca3af';
        const label = SEVERITY_LABELS[cls] || cls;
        const isPredicted = cls === predictedClass;

        return (
          <div key={cls} className="flex items-center gap-3">
            <div className="w-28 text-xs text-right truncate">
              <span className={isPredicted ? 'font-bold' : 'text-gray-500'}>{label}</span>
            </div>
            <div className="flex-1 h-5 bg-gray-100 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{ width: `${Math.max(prob * 100, 0.5)}%`, backgroundColor: color }}
              />
            </div>
            <div className="w-14 text-xs text-right">
              <span className={isPredicted ? 'font-bold' : 'text-gray-500'}>
                {formatPercent(prob)}
              </span>
            </div>
            {isPredicted && <span className="text-xs text-gray-400">← predicted</span>}
          </div>
        );
      })}
    </div>
  );
}
