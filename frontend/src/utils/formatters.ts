/**
 * Display formatting utilities.
 */

export function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function formatMs(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

export function severityColorClass(level: number): string {
  switch (level) {
    case 0: return 'text-green-600 bg-green-50 border-green-200';
    case 1: return 'text-lime-600 bg-lime-50 border-lime-200';
    case 2: return 'text-amber-600 bg-amber-50 border-amber-200';
    case 3: return 'text-orange-600 bg-orange-50 border-orange-200';
    case 4: return 'text-red-600 bg-red-50 border-red-200';
    default: return 'text-gray-600 bg-gray-50 border-gray-200';
  }
}

export function severityBgClass(level: number): string {
  switch (level) {
    case 0: return 'bg-green-500';
    case 1: return 'bg-lime-500';
    case 2: return 'bg-amber-500';
    case 3: return 'bg-orange-500';
    case 4: return 'bg-red-500';
    default: return 'bg-gray-500';
  }
}
