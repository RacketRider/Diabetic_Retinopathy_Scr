import type { AttentionRegion } from '@/types';

interface LesionEvidenceProps {
  regions: AttentionRegion[];
  clinicalEvidence: string;
}

const LESION_ICONS: Record<string, string> = {
  microaneurysm: '🔴',
  exudate: '🟡',
  hemorrhage: '🟠',
  neovascularization: '🔵',
};

export function LesionEvidence({ regions, clinicalEvidence }: LesionEvidenceProps) {
  return (
    <div className="space-y-3">
      {regions.length > 0 && (
        <div className="space-y-1.5">
          {regions.slice(0, 5).map((region, i) => (
            <div key={i} className="flex items-center gap-2 text-sm">
              <span>{LESION_ICONS[region.type] || '⬤'}</span>
              <span className="capitalize">{region.type.replace('_', ' ')}</span>
              <span className="text-gray-400">—</span>
              <span className="text-gray-500">
                confidence {(region.confidence * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>
      )}

      <div className="bg-gray-50 rounded-lg px-4 py-3 text-sm text-gray-700">
        <p className="font-medium text-gray-800 mb-1">Clinical Assessment</p>
        <p>{clinicalEvidence}</p>
      </div>
    </div>
  );
}
