import { useState, useCallback } from 'react';
import { validateImage, type ValidationResult } from '@/utils/imageValidation';
import { stripExif } from '@/utils/exifStrip';

export function useImageUpload() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isValidating, setIsValidating] = useState(false);

  const processFile = useCallback(async (rawFile: File) => {
    setError(null);
    setIsValidating(true);

    try {
      // 1. Client-side validation (extension, size, magic bytes, dimensions)
      const validation: ValidationResult = await validateImage(rawFile);
      if (!validation.valid) {
        setError(validation.error || 'Invalid image');
        setIsValidating(false);
        return;
      }

      // 2. Strip EXIF metadata via canvas re-encode
      const cleanFile = await stripExif(rawFile);

      // 3. Create preview URL
      const previewUrl = URL.createObjectURL(cleanFile);

      // Clean up old preview
      if (preview) URL.revokeObjectURL(preview);

      setFile(cleanFile);
      setPreview(previewUrl);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to process image');
    } finally {
      setIsValidating(false);
    }
  }, [preview]);

  const clear = useCallback(() => {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview(null);
    setError(null);
  }, [preview]);

  return { file, preview, error, isValidating, processFile, clear };
}
