/**
 * Client-side image validation.
 * Defense-in-depth: validates BEFORE sending to the backend.
 */

const ALLOWED_TYPES = new Set(['image/jpeg', 'image/png']);
const MAX_FILE_SIZE = 15 * 1024 * 1024; // 15 MB
const MIN_DIMENSION = 256;

// JPEG magic bytes: FF D8 FF
// PNG magic bytes: 89 50 4E 47
const JPEG_MAGIC = [0xff, 0xd8, 0xff];
const PNG_MAGIC = [0x89, 0x50, 0x4e, 0x47];

export interface ValidationResult {
  valid: boolean;
  error?: string;
}

function checkMagicBytes(bytes: Uint8Array): boolean {
  if (bytes.length < 4) return false;

  const isJPEG = JPEG_MAGIC.every((b, i) => bytes[i] === b);
  const isPNG = PNG_MAGIC.every((b, i) => bytes[i] === b);

  return isJPEG || isPNG;
}

export async function validateImage(file: File): Promise<ValidationResult> {
  // 1. MIME type check
  if (!ALLOWED_TYPES.has(file.type)) {
    return { valid: false, error: `Unsupported file type: ${file.type}. Only JPEG and PNG allowed.` };
  }

  // 2. File size check
  if (file.size > MAX_FILE_SIZE) {
    const sizeMB = (file.size / 1024 / 1024).toFixed(1);
    return { valid: false, error: `File too large (${sizeMB} MB). Maximum: 15 MB.` };
  }

  // 3. Magic byte check
  const header = await file.slice(0, 8).arrayBuffer();
  const bytes = new Uint8Array(header);
  if (!checkMagicBytes(bytes)) {
    return { valid: false, error: 'File content does not match a valid image format.' };
  }

  // 4. Dimension check (load into Image element)
  return new Promise((resolve) => {
    const img = new Image();
    const url = URL.createObjectURL(file);

    img.onload = () => {
      URL.revokeObjectURL(url);
      if (img.width < MIN_DIMENSION || img.height < MIN_DIMENSION) {
        resolve({
          valid: false,
          error: `Image too small (${img.width}×${img.height}). Minimum ${MIN_DIMENSION}×${MIN_DIMENSION}.`,
        });
      } else {
        resolve({ valid: true });
      }
    };

    img.onerror = () => {
      URL.revokeObjectURL(url);
      resolve({ valid: false, error: 'Failed to load image. File may be corrupt.' });
    };

    img.src = url;
  });
}
