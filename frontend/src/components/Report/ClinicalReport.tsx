import { useState } from 'react';
import { downloadReport } from '@/api/client';

interface ClinicalReportProps {
  file: File;
}

export function ClinicalReport({ file }: ClinicalReportProps) {
  const [isDownloading, setIsDownloading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleDownload = async () => {
    setIsDownloading(true);
    setError(null);

    try {
      const blob = await downloadReport(file);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'dr_screening_report.pdf';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err: any) {
      setError(err?.message || 'Failed to generate report');
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <div>
      <button
        onClick={handleDownload}
        disabled={isDownloading}
        className="w-full px-4 py-2.5 bg-gray-800 text-white rounded-lg hover:bg-gray-900
                   text-sm font-medium transition-colors disabled:opacity-50
                   flex items-center justify-center gap-2"
      >
        {isDownloading ? (
          <>
            <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            Generating...
          </>
        ) : (
          '📄 Download Clinical Report (PDF)'
        )}
      </button>
      {error && <p className="text-red-600 text-xs mt-2">{error}</p>}
    </div>
  );
}
