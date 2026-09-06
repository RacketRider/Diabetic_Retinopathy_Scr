"""Upload validation and security utilities.

Defense-in-depth pipeline:
1. Extension check
2. File size check
3. Magic byte verification (filetype lib — never trust headers)
4. Safe PIL decode with pixel bomb defense
5. Minimum resolution check
6. EXIF stripping via clean copy
"""

import io
from pathlib import Path

import filetype
from fastapi import HTTPException, UploadFile
from PIL import Image

from app.config import (
    ALLOWED_EXTENSIONS,
    ALLOWED_MIMES,
    MAX_FILE_SIZE,
    MAX_IMAGE_PIXELS,
    MIN_DIMENSION,
)

# Set pixel bomb defense globally
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


async def validate_upload(file: UploadFile) -> Image.Image:
    """Validate and sanitize an uploaded fundus image.

    Returns a clean RGB PIL Image with all metadata stripped.
    Raises HTTPException on any validation failure.
    """
    # 1. Extension check
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Only JPEG/PNG allowed.",
        )

    # 2. Size check
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content) // 1024 // 1024} MB). Maximum: {MAX_FILE_SIZE // 1024 // 1024} MB.",
        )

    # 3. Magic byte verification — never trust filename or Content-Type header
    kind = filetype.guess(content)
    if kind is None or kind.mime not in ALLOWED_MIMES:
        detected = kind.mime if kind else "unknown"
        raise HTTPException(
            status_code=400,
            detail=f"File content ({detected}) does not match a valid image format.",
        )

    # 4. Safe decode with pixel bomb defense
    try:
        buf = io.BytesIO(content)
        img = Image.open(buf)
        img.verify()  # validates header integrity without full decode

        buf.seek(0)
        img = Image.open(buf).convert("RGB")  # re-open after verify
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Corrupt or unreadable image file.",
        )

    # 5. Minimum resolution check
    if img.width < MIN_DIMENSION or img.height < MIN_DIMENSION:
        raise HTTPException(
            status_code=400,
            detail=f"Image too small ({img.width}x{img.height}). Minimum {MIN_DIMENSION}x{MIN_DIMENSION} for fundus analysis.",
        )

    # 6. Strip ALL metadata (EXIF, GPS, camera serial, clinic identifiers)
    clean = Image.new(img.mode, img.size)
    clean.putdata(list(img.getdata()))

    return clean
