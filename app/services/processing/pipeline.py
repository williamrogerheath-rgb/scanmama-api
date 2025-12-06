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
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.models.schemas import ScanOptions
from .detect_ml import detect
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


def auto_trim_margins(image: np.ndarray, padding: int = 30) -> np.ndarray:
    """Trim margins that are nearly uniform color."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # For each edge, check if rows/cols are nearly uniform
    # A row/col is "uniform" if its std dev is very low

    def find_content_start(values_2d, axis):
        """Find where content starts from the beginning."""
        std_per_line = np.std(values_2d, axis=axis)
        # Uniform = std dev < 15 (out of 255)
        uniform_threshold = 15
        for i, std in enumerate(std_per_line):
            if std > uniform_threshold:
                return i
        return 0

    def find_content_end(values_2d, axis):
        """Find where content ends."""
        std_per_line = np.std(values_2d, axis=axis)
        uniform_threshold = 15
        for i in range(len(std_per_line) - 1, -1, -1):
            if std_per_line[i] > uniform_threshold:
                return i + 1
        return len(std_per_line)

    # Find bounds
    top = find_content_start(gray, axis=1)  # std of each row
    bottom = find_content_end(gray, axis=1)
    left = find_content_start(gray, axis=0)  # std of each column
    right = find_content_end(gray, axis=0)

    # Apply padding
    top = max(0, top - padding)
    bottom = min(h, bottom + padding)
    left = max(0, left - padding)
    right = min(w, right + padding)

    # Only crop if we're actually removing something meaningful (>3% of image)
    if top < h * 0.03 and bottom > h * 0.97 and left < w * 0.03 and right > w * 0.97:
        print(f"Auto-trim: no significant margins found")
        return image

    print(f"Auto-trim: cropping from ({left},{top}) to ({right},{bottom}) - original {w}x{h}")
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
    Document processing pipeline with ML detection

    Pipeline:
    1. Decode image
    2. Store original
    3. ML detection (DocAligner)
    4. Perspective transform (if detected) OR auto-trim (fallback)
    5. Enhance image quality (AFTER cropping to document only)
    6. Apply color mode
    7. Resize if needed (max 2000px)
    8. Generate PDF

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

    # Step 1: ML Document Detection
    try:
        detection_result = detect(image)
        document_detected = detection_result.confidence > 0.5 and detection_result.mode != "fallback"
        detection_method = detection_result.method
        detection_confidence = detection_result.confidence

        print(f"Detection result: detected={document_detected}, confidence={detection_confidence:.3f}, method={detection_method}")
    except Exception as e:
        print(f"Detection error: {e}")
        document_detected = False
        detection_method = "error"
        detection_confidence = 0.0
        detection_result = None

    # Step 2: Transform or Auto-trim
    processed = image.copy()

    if document_detected and detection_result is not None:
        # Try perspective transformation
        try:
            transformed = transform(image, detection_result.corners)
            if transformed is not None:
                processed = transformed
                print(f"After transform: {processed.shape[1]}x{processed.shape[0]}")
            else:
                print("Transform returned None, falling back to auto-trim")
                processed = auto_trim_margins(processed)
                print(f"After trim (fallback): {processed.shape[1]}x{processed.shape[0]}")
        except Exception as e:
            print(f"Transform error: {e}, falling back to auto-trim")
            processed = auto_trim_margins(processed)
            print(f"After trim (fallback): {processed.shape[1]}x{processed.shape[0]}")
    else:
        # No document detected - use auto-trim fallback
        print("No document detected, using auto-trim fallback")
        try:
            processed = auto_trim_margins(processed)
            print(f"After trim: {processed.shape[1]}x{processed.shape[0]}")
        except Exception as e:
            print(f"Trim error: {e}")
            pass

    try:
        # Step 3: Enhance image quality (AFTER cropping to document only)
        processed = enhance(processed)
        print(f"After enhance: {processed.shape[1]}x{processed.shape[0]}")
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

    try:
        # Step 5: Resize if needed (max 2000px on longest edge)
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


def process_documents_batch(images_bytes: List[bytes], options: ScanOptions, max_workers: int = 4) -> List[Dict]:
    """
    Process multiple documents in parallel using ThreadPoolExecutor

    Args:
        images_bytes: List of image byte arrays to process
        options: Scan options to apply to all images
        max_workers: Maximum number of parallel workers (default: 4)

    Returns:
        List of results from process_document, in same order as input
    """
    if not images_bytes:
        return []

    # If only one image, process directly without threading overhead
    if len(images_bytes) == 1:
        return [process_document(images_bytes[0], options)]

    print(f"Processing {len(images_bytes)} images in parallel with {max_workers} workers")
    start_time = time.time()

    results = [None] * len(images_bytes)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks with their index
        future_to_index = {
            executor.submit(process_document, img_bytes, options): idx
            for idx, img_bytes in enumerate(images_bytes)
        }

        # Collect results as they complete
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                result = future.result()
                results[idx] = result
                print(f"  Image {idx + 1}/{len(images_bytes)} completed")
            except Exception as e:
                print(f"  Image {idx + 1}/{len(images_bytes)} failed: {e}")
                # Return error result
                results[idx] = {
                    "error": str(e),
                    "processing_time_ms": 0,
                    "document_detected": False,
                }

    total_time = time.time() - start_time
    print(f"Batch processing completed in {total_time:.2f}s ({total_time/len(images_bytes):.2f}s per image)")

    return results
