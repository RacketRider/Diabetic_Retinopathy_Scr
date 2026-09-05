import { Card } from '@/components/common/Card';
import type { GradCAMResponse } from '@/types';
import { SEVERITY_LABELS } from '@/types';

interface StatsPanelProps {
  /** All completed scan results in the current session. */
  scans: GradCAMResponse[];
}

export function StatsPanel({ scans }: StatsPanelProps) {
  if (scans.length === 0) return null;

  const totalScans = scans.length;
  const referableCount = scans.filter((s) => s.grading.is_referable).length;
  const nonReferableCount = totalScans - referableCount;
  const avgProcessingTime =
    scans.reduce((sum, s) => sum + (s.processing_time_ms ?? 0), 0) / totalScans;

  // Grade distribution
  const gradeCounts: Record<string, number> = {};
  for (const scan of scans) {
    const label = scan.grading.label;
    gradeCounts[label] = (gradeCounts[label] || 0) + 1;
  }

  return (
    <Card title="Session Statistics">
      {/* Summary row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
        <StatBox label="Total Scans" value={totalScans} icon="🔬" />
        <StatBox
          label="Referable"
          value={referableCount}
          icon="⚠️"
          color="text-amber-600"
        />
        <StatBox
          label="Normal"
          value={nonReferableCount}
          icon="✅"
          color="text-green-600"
        />
        <StatBox
          label="Avg. Time"
          value={`${Math.round(avgProcessingTime)}ms`}
          icon="⏱"
        />
      </div>

      {/* Grade distribution */}
      {Object.keys(gradeCounts).length > 0 && (
        <div>
          <h4 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">
            Grade Distribution
          </h4>
          <div className="space-y-1.5">
            {Object.entries(gradeCounts)
              .sort(([, a], [, b]) => b - a)
              .map(([label, count]) => {
                const pct = (count / totalScans) * 100;
                return (
                  <div key={label} className="flex items-center gap-2 text-sm">
                    <span className="w-28 text-gray-600 truncate">
                      {SEVERITY_LABELS[label] ?? label}
                    </span>
                    <div className="flex-1 bg-gray-100 rounded-full h-4 overflow-hidden">
                      <div
                        className="h-full rounded-full bg-primary-500 transition-all duration-500"
                        style={{ width: `${pct}%` }}
                      />
                    </div>
                    <span className="w-12 text-right text-xs text-gray-500">
                      {count} ({Math.round(pct)}%)
                    </span>
                  </div>
                );
              })}
          </div>
        </div>
      )}
    </Card>
  );
}

function StatBox({
  label,
  value,
  icon,
  color,
}: {
  label: string;
  value: string | number;
  icon: string;
  color?: string;
}) {
  return (
    <div className="bg-gray-50 rounded-lg p-3 text-center">
      <div className="text-xl">{icon}</div>
      <div className={`text-lg font-bold mt-0.5 ${color ?? 'text-gray-900'}`}>
        {value}
      </div>
      <div className="text-xs text-gray-500">{label}</div>
    </div>
  );
}
