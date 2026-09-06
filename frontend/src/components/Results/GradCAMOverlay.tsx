import { useState } from 'react';

interface GradCAMOverlayProps {
  originalSrc: string;
  overlayBase64: string;
}

export function GradCAMOverlay({ originalSrc, overlayBase64 }: GradCAMOverlayProps) {
  const [opacity, setOpacity] = useState(0.6);

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3">
        <div>
          <p className="text-xs text-gray-500 mb-1.5 font-medium">Original</p>
          <div className="rounded-lg overflow-hidden border border-gray-200 bg-black">
            <img src={originalSrc} alt="Original fundus" className="w-full object-contain" />
          </div>
        </div>
        <div>
          <p className="text-xs text-gray-500 mb-1.5 font-medium">Grad-CAM Overlay</p>
          <div className="rounded-lg overflow-hidden border border-gray-200 bg-black relative">
            <img src={originalSrc} alt="Base" className="w-full object-contain" />
            <img
              src={`data:image/png;base64,${overlayBase64}`}
              alt="Grad-CAM overlay"
              className="absolute inset-0 w-full h-full object-contain"
              style={{ opacity }}
            />
          </div>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <span className="text-xs text-gray-500 w-16">Opacity</span>
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={opacity}
          onChange={(e) => setOpacity(Number(e.target.value))}
          className="flex-1 h-1.5 accent-primary-600"
        />
        <span className="text-xs text-gray-500 w-10 text-right">{Math.round(opacity * 100)}%</span>
      </div>
    </div>
  );
}
