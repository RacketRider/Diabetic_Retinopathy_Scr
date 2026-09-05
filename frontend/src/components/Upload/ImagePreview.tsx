interface ImagePreviewProps {
  src: string;
  onClear: () => void;
  onAnalyze: () => void;
  isLoading: boolean;
}

export function ImagePreview({ src, onClear, onAnalyze, isLoading }: ImagePreviewProps) {
  return (
    <div className="space-y-4">
      <div className="relative rounded-xl overflow-hidden border border-gray-200 bg-black">
        <img
          src={src}
          alt="Fundus preview"
          className="w-full max-h-[400px] object-contain"
        />
      </div>

      <div className="flex gap-3">
        <button
          onClick={onClear}
          disabled={isLoading}
          className="flex-1 px-4 py-2.5 border border-gray-300 text-gray-700 rounded-lg
                     hover:bg-gray-50 text-sm font-medium transition-colors
                     disabled:opacity-50 disabled:cursor-not-allowed"
        >
          ← Clear
        </button>
        <button
          onClick={onAnalyze}
          disabled={isLoading}
          className="flex-1 px-4 py-2.5 bg-primary-600 text-white rounded-lg
                     hover:bg-primary-700 text-sm font-medium transition-colors
                     disabled:opacity-50 disabled:cursor-not-allowed
                     flex items-center justify-center gap-2"
        >
          {isLoading ? (
            <>
              <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              Analyzing...
            </>
          ) : (
            '🔍 Analyze for DR'
          )}
        </button>
      </div>
    </div>
  );
}
