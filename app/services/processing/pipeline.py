"""
Main document processing pipeline

Orchestrates detection, transformation, and enhancement to convert
document photos into clean, high-resolution scans.
"""
import time
import io
import base64
import numpy as np
import cv2
import img2pdf
from typing import Dict

from app.models.schemas import ScanOptions
from .detect import detect
from .transform import transform
from .enhance import enhance


def decode_base64_image(image_base64: str) -> bytes:
    """
    Decode base64 string to bytes

    Handles data URLs (strips prefix if present)
    """
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]
    return base64.b64decode(image_base64)


def apply_color_mode(image: np.ndarray, mode: str) -> np.ndarray:
    """
    Apply color mode transformation

    Args:
        mode: "color", "grayscale", or "bw" (black and white)
    """
    if mode == "grayscale":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif mode == "bw":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    else:  # color
        return image


def auto_trim_margins(image: np.ndarray, padding: int = 20) -> np.ndarray:
    """Trim obvious uniform margins from image edges."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate variance for each row and column
    row_var = np.var(gray, axis=1)
    col_var = np.var(gray, axis=0)

    # Find where content exists (variance above threshold)
    threshold = np.mean(row_var) * 0.1  # 10% of mean variance

    rows_with_content = np.where(row_var > threshold)[0]
    cols_with_content = np.where(col_var > threshold)[0]

    if len(rows_with_content) == 0 or len(cols_with_content) == 0:
        return image  # No trimming possible

    # Get bounds with padding
    top = max(0, rows_with_content[0] - padding)
    bottom = min(image.shape[0], rows_with_content[-1] + padding)
    left = max(0, cols_with_content[0] - padding)
    right = min(image.shape[1], cols_with_content[-1] + padding)

    # Only crop if we're removing significant margins (>5% per side)
    min_crop = 0.05
    if (top < image.shape[0] * min_crop and
        bottom > image.shape[0] * (1 - min_crop) and
        left < image.shape[1] * min_crop and
        right > image.shape[1] * (1 - min_crop)):
        return image  # Not enough to trim

    return image[top:bottom, left:right]


def resize_if_needed(image: np.ndarray, max_edge: int = 2000) -> np.ndarray:
    """Resize image if larger than max_edge."""
    h, w = image.shape[:2]
    if max(h, w) <= max_edge:
        return image

    scale = max_edge / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def process_document(image_bytes: bytes, options: ScanOptions) -> Dict:
    """
    Simplified document processing pipeline (MVP)

    Pipeline:
    1. Decode image
    2. Store original
    3. Auto-trim margins (remove obvious dead space)
    4. Enhance image quality (gentle CLAHE only)
    5. Apply color mode
    6. Resize if needed (max 2000px)
    7. Generate PDF

    No document detection/cropping - just enhance what the user photographed.

    Returns:
        Dict with original_bytes, processed_bytes, pdf_bytes, metadata
    """
    start_time = time.time()

    # Decode image from bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Failed to decode image")

    # Store original as JPEG (quality 85 for smaller file size)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
    _, original_jpg = cv2.imencode('.jpg', image, encode_params)
    original_bytes = original_jpg.tobytes()

    # Initialize tracking variables - detection skipped for MVP
    document_detected = False
    detection_method = "skipped"
    detection_confidence = 0.0
    processed = image.copy()

    # DETECTION & TRANSFORM SKIPPED FOR MVP
    # Detection was failing on dark surfaces and cropping incorrectly
    # For MVP, just enhance the full image as photographed

    try:
        # Step 1: Auto-trim margins (remove obvious dead space)
        processed = auto_trim_margins(processed)
        print(f"After trim: {processed.shape[1]}x{processed.shape[0]}")
    except Exception as e:
        # Trim failed - use original
        print(f"Trim error: {e}")
        pass

    try:
        # Step 2: Enhance image quality (gentle CLAHE only)
        processed = enhance(processed)
        print(f"After enhance: {processed.shape[1]}x{processed.shape[0]}")
    except Exception as e:
        # Enhancement failed - use previous result
        print(f"Enhancement error: {e}")
        pass

    try:
        # Step 3: Apply color mode
        processed = apply_color_mode(processed, options.colorMode)
    except Exception as e:
        # Color mode failed - use previous result
        print(f"Color mode error: {e}")
        pass

    try:
        # Step 4: Resize if needed (max 2000px on longest edge)
        original_size = f"{processed.shape[1]}x{processed.shape[0]}"
        processed = resize_if_needed(processed, max_edge=2000)
        if processed.shape[1] != image.shape[1] or processed.shape[0] != image.shape[0]:
            print(f"Resized: {original_size} -> {processed.shape[1]}x{processed.shape[0]}")
    except Exception as e:
        # Resize failed - use previous result
        print(f"Resize error: {e}")
        pass

    # Encode processed image as JPEG (quality 85 for smaller file size)
    encode_params_final = [cv2.IMWRITE_JPEG_QUALITY, 85]
    _, processed_jpg = cv2.imencode('.jpg', processed, encode_params_final)
    processed_bytes = processed_jpg.tobytes()

    # Generate PDF from processed image
    try:
        pdf_bytes = img2pdf.convert(processed_bytes)
    except Exception as e:
        print(f"PDF generation error: {e}")
        pdf_bytes = b""

    # Calculate processing time
    processing_time_ms = int((time.time() - start_time) * 1000)

    # Get output dimensions
    height, width = processed.shape[:2]

    return {
        "original_bytes": original_bytes,
        "processed_bytes": processed_bytes,
        "pdf_bytes": pdf_bytes,
        "width": width,
        "height": height,
        "document_detected": document_detected,
        "processing_time_ms": processing_time_ms,
        "detection_method": detection_method,
        "detection_confidence": detection_confidence,
    }
