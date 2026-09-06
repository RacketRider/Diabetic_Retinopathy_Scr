import { useCallback, useState, type DragEvent, type ChangeEvent } from 'react';

interface DropZoneProps {
  onFile: (file: File) => void;
  isValidating: boolean;
  error: string | null;
}

/**
 * Sample images for demo — these will be loaded from public/samples/.
 * Until real fundus images are added, clicking a sample triggers a
 * file-picker as a fallback.
 */
const SAMPLE_GRADES = [
  { label: 'No DR', file: 'no_dr.jpg' },
  { label: 'Mild', file: 'mild.jpg' },
  { label: 'Moderate', file: 'moderate.jpg' },
  { label: 'Severe', file: 'severe.jpg' },
  { label: 'PDR', file: 'proliferative.jpg' },
] as const;

export function DropZone({ onFile, isValidating, error }: DropZoneProps) {
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault();
      setIsDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) onFile(file);
    },
    [onFile]
  );

  const handleChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onFile(file);
      e.target.value = ''; // allow re-selecting same file
    },
    [onFile]
  );

  const handleSampleClick = useCallback(
    async (filename: string) => {
      try {
        const resp = await fetch(`/samples/${filename}`);
        if (!resp.ok) throw new Error('Sample not found');
        const blob = await resp.blob();
        const file = new File([blob], filename, { type: blob.type || 'image/jpeg' });
        onFile(file);
      } catch {
        // Samples not yet available — fall back to file picker
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.jpg,.jpeg,.png';
        input.onchange = (e) => {
          const f = (e.target as HTMLInputElement).files?.[0];
          if (f) onFile(f);
        };
        input.click();
      }
    },
    [onFile]
  );

  return (
    <div className="space-y-3">
      <div
        onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
        onDragLeave={() => setIsDragOver(false)}
        onDrop={handleDrop}
        className={`relative border-2 border-dashed rounded-xl p-10 text-center cursor-pointer
          transition-all duration-200 ${
            isDragOver
              ? 'border-primary-500 bg-primary-50 scale-[1.01]'
              : 'border-gray-300 hover:border-primary-400 hover:bg-gray-50'
          } ${isValidating ? 'opacity-60 pointer-events-none' : ''}`}
      >
        <input
          type="file"
          accept=".jpg,.jpeg,.png"
          onChange={handleChange}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          disabled={isValidating}
        />

        <div className="space-y-3">
          <div className="text-4xl">📷</div>
          <div>
            <p className="text-base font-medium text-gray-700">
              {isValidating ? 'Validating image...' : 'Drag & drop a fundus image'}
            </p>
            <p className="text-sm text-gray-400 mt-1">or click to browse</p>
          </div>
          <p className="text-xs text-gray-400">JPEG / PNG • Max 15 MB • Min 256×256</p>
        </div>
      </div>

      {/* Sample images */}
      <div className="text-center">
        <p className="text-xs text-gray-400 mb-2">Or try a sample:</p>
        <div className="flex flex-wrap justify-center gap-2">
          {SAMPLE_GRADES.map((sample) => (
            <button
              key={sample.file}
              onClick={() => handleSampleClick(sample.file)}
              disabled={isValidating}
              className="px-3 py-1.5 text-xs font-medium text-primary-600 bg-primary-50
                hover:bg-primary-100 rounded-lg transition-colors disabled:opacity-50"
            >
              {sample.label} ▸
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg px-4 py-2.5 text-sm text-red-700">
          ❌ {error}
        </div>
      )}
    </div>
  );
}
