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


def process_document(image_bytes: bytes, options: ScanOptions) -> Dict:
    """
    Main document processing pipeline

    Pipeline:
    1. Decode image
    2. Store original
    3. Detect document corners
    4. Transform perspective (if detected)
    5. Enhance image quality
    6. Apply color mode
    7. Generate PDF

    Returns:
        Dict with original_bytes, processed_bytes, pdf_bytes, metadata
    """
    start_time = time.time()

    # Decode image from bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Failed to decode image")

    # Store original as JPEG (quality 92)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 92]
    _, original_jpg = cv2.imencode('.jpg', image, encode_params)
    original_bytes = original_jpg.tobytes()

    # Initialize tracking variables
    document_detected = False
    detection_method = "none"
    detection_confidence = 0.0
    processed = image.copy()

    try:
        # Step 1: Detect document corners
        detection_result = detect(image, debug=False)
        detection_method = detection_result.method
        detection_confidence = detection_result.confidence

        # Step 2: Transform perspective if detected with sufficient confidence
        # Lower threshold to 0.4 to catch more valid detections
        if detection_confidence >= 0.4 and detection_result.mode != "fallback":
            document_detected = True

            transformed = transform(image, detection_result.corners)
            if transformed is not None:
                processed = transformed
                print(f"Document transformed: confidence={detection_confidence:.3f}, method={detection_method}")
            else:
                print(f"Transform failed: confidence={detection_confidence:.3f}, falling back to original")
        else:
            print(f"Detection too low: confidence={detection_confidence:.3f}, mode={detection_result.mode}")

    except Exception as e:
        # Detection failed - use original image
        print(f"Detection error: {e}")
        pass

    try:
        # Step 3: Enhance image quality
        processed = enhance(processed)
    except Exception as e:
        # Enhancement failed - use previous result
        print(f"Enhancement error: {e}")
        pass

    try:
        # Step 4: Apply color mode
        processed = apply_color_mode(processed, options.colorMode)
    except Exception as e:
        # Color mode failed - use previous result
        print(f"Color mode error: {e}")
        pass

    # Encode processed image as JPEG (quality 95 for final output)
    encode_params_final = [cv2.IMWRITE_JPEG_QUALITY, 95]
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
