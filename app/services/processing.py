"""
ScanMama Document Processing Pipeline
Production-quality edge detection, perspective correction, and enhancement

Key features:
- Multi-strategy edge detection (works on light AND dark backgrounds)
- Proper contour validation (aspect ratio, convexity, area)
- Premium enhancement (CLAHE, bilateral filter, unsharp masking)
- Confidence scoring with graceful fallbacks
- High-quality output (preserves text sharpness)
"""
import base64
import io
import time
import numpy as np
import cv2
import img2pdf
from PIL import Image
from dataclasses import dataclass
from typing import Optional, Tuple, List

from app.models.schemas import ScanOptions


@dataclass
class DetectionResult:
    """Result of document detection with confidence score"""
    contour: Optional[np.ndarray]
    confidence: float  # 0.0 to 1.0
    method: str  # Which detection method succeeded


def decode_base64_image(image_base64: str) -> bytes:
    """Decode base64 string to bytes"""
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]
    return base64.b64decode(image_base64)


# =============================================================================
# CONTOUR VALIDATION
# =============================================================================

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order points: top-left, top-right, bottom-right, bottom-left
    Uses a robust algorithm that handles edge cases
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    
    # Sum and diff method
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    
    rect[0] = pts[np.argmin(s)]      # Top-left: smallest sum
    rect[2] = pts[np.argmax(s)]      # Bottom-right: largest sum
    rect[1] = pts[np.argmin(diff)]   # Top-right: smallest diff
    rect[3] = pts[np.argmax(diff)]   # Bottom-left: largest diff
    
    return rect


def validate_quadrilateral(contour: np.ndarray, image_shape: Tuple[int, int]) -> Tuple[bool, float]:
    """
    Validate that a contour represents a reasonable document
    Returns: (is_valid, confidence_score)
    """
    if contour is None or len(contour) != 4:
        return False, 0.0
    
    height, width = image_shape[:2]
    image_area = height * width
    
    pts = contour.reshape(4, 2).astype(np.float32)
    ordered = order_points(pts)
    
    # 1. Check area (document should be 15-80% of image)
    contour_area = cv2.contourArea(ordered)
    area_ratio = contour_area / image_area
    if area_ratio < 0.15 or area_ratio > 0.80:
        return False, 0.0
    
    # 2. Check aspect ratio (should be reasonable for a document)
    (tl, tr, br, bl) = ordered
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    
    avg_width = (width_top + width_bottom) / 2
    avg_height = (height_left + height_right) / 2
    
    if avg_width == 0 or avg_height == 0:
        return False, 0.0
    
    aspect_ratio = max(avg_width, avg_height) / min(avg_width, avg_height)
    if aspect_ratio > 5.0:  # Too extreme (5:1 max)
        return False, 0.0
    
    # 3. Check that sides are roughly parallel (document, not random shape)
    width_diff = abs(width_top - width_bottom) / max(width_top, width_bottom)
    height_diff = abs(height_left - height_right) / max(height_left, height_right)
    
    if width_diff > 0.5 or height_diff > 0.5:  # Sides too different
        return False, 0.3
    
    # 4. Check convexity
    hull = cv2.convexHull(ordered)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return False, 0.0
    
    convexity = contour_area / hull_area
    if convexity < 0.8:  # Not convex enough
        return False, 0.3
    
    # Calculate confidence based on quality metrics
    confidence = 0.5
    confidence += min(0.2, area_ratio * 0.3)  # Larger = more confident
    confidence += 0.15 * (1 - width_diff)     # More parallel = more confident
    confidence += 0.15 * (1 - height_diff)
    confidence = min(1.0, confidence)
    
    return True, confidence


def find_quad_from_contour(contour: np.ndarray, min_area: float) -> Optional[np.ndarray]:
    """
    Try to extract a quadrilateral from a contour using multiple epsilon values
    Accepts 4-6 point approximations (uses minAreaRect for 5-6 points to handle noisy edges)
    """
    area = cv2.contourArea(contour)
    if area < min_area:
        return None
    
    peri = cv2.arcLength(contour, True)
    
    # Try different approximation accuracies
    for eps in [0.02, 0.03, 0.04, 0.05, 0.06]:
        approx = cv2.approxPolyDP(contour, eps * peri, True)
        if 4 <= len(approx) <= 6:
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)
            else:
                # More than 4 points - find the minimum area bounding rectangle
                rect = cv2.minAreaRect(approx)
                quad = cv2.boxPoints(rect)
                return np.array(quad, dtype=np.float32)
    
    # Try convex hull if direct approximation fails
    hull = cv2.convexHull(contour)
    hull_peri = cv2.arcLength(hull, True)

    for eps in [0.02, 0.03, 0.04, 0.05]:
        approx = cv2.approxPolyDP(hull, eps * hull_peri, True)
        if 4 <= len(approx) <= 6:
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)
            else:
                # More than 4 points - find the minimum area bounding rectangle
                rect = cv2.minAreaRect(approx)
                quad = cv2.boxPoints(rect)
                return np.array(quad, dtype=np.float32)

    return None


# =============================================================================
# EDGE DETECTION STRATEGIES
# =============================================================================

def detect_with_adaptive_threshold(gray: np.ndarray, min_area: float) -> Optional[np.ndarray]:
    """Strategy 1: Adaptive threshold - good for varied lighting"""
    # Use bilateral filter to smooth while preserving edges
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Clean up with morphology
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    for contour in contours:
        quad = find_quad_from_contour(contour, min_area)
        if quad is not None:
            return quad
    
    return None


def detect_with_canny_multi(gray: np.ndarray, min_area: float) -> Optional[np.ndarray]:
    """Strategy 2: Canny with multiple threshold combinations"""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Try different Canny threshold combinations
    threshold_pairs = [
        (30, 100),   # Low contrast
        (50, 150),   # Medium contrast
        (75, 200),   # High contrast
        (20, 80),    # Very low contrast (light backgrounds)
    ]
    
    for low, high in threshold_pairs:
        edges = cv2.Canny(blurred, low, high)
        
        # Dilate to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        for contour in contours:
            quad = find_quad_from_contour(contour, min_area)
            if quad is not None:
                return quad
    
    return None


def detect_with_otsu(gray: np.ndarray, min_area: float) -> Optional[np.ndarray]:
    """Strategy 3: Otsu threshold - good for bimodal histograms"""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Clean up
    kernel = np.ones((7, 7), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    for contour in contours:
        quad = find_quad_from_contour(contour, min_area)
        if quad is not None:
            return quad
    
    return None


def detect_with_inverted(gray: np.ndarray, min_area: float) -> Optional[np.ndarray]:
    """Strategy 4: Inverted image - for light documents on light backgrounds"""
    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (5, 5), 0)
    
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    for contour in contours:
        quad = find_quad_from_contour(contour, min_area)
        if quad is not None:
            return quad
    
    return None


def detect_with_saturation(image: np.ndarray, min_area: float) -> Optional[np.ndarray]:
    """Strategy 5: Saturation channel - good for colored documents/backgrounds"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    
    # Enhance saturation contrast
    sat = cv2.normalize(sat, None, 0, 255, cv2.NORM_MINMAX)
    
    _, thresh = cv2.threshold(sat, 30, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    for contour in contours:
        quad = find_quad_from_contour(contour, min_area)
        if quad is not None:
            return quad
    
    return None


def detect_with_lab(image: np.ndarray, min_area: float) -> Optional[np.ndarray]:
    """Strategy 6: LAB color space - better color separation"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    
    # CLAHE on L channel for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(l_channel)
    
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    for contour in contours:
        quad = find_quad_from_contour(contour, min_area)
        if quad is not None:
            return quad
    
    return None


def detect_with_hough_lines(gray: np.ndarray, min_area: float, 
                           width: int, height: int) -> Optional[np.ndarray]:
    """Strategy 7: Hough lines - find document by detecting edges as lines"""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Detect lines
    min_line_length = min(width, height) // 5
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/180, 
        threshold=50, 
        minLineLength=min_line_length, 
        maxLineGap=20
    )
    
    if lines is None or len(lines) < 4:
        return None
    
    # Classify lines as horizontal or vertical
    horizontal = []
    vertical = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        
        if angle < 30 or angle > 150:  # Horizontal
            horizontal.append((y1 + y2) // 2)
        elif 60 < angle < 120:  # Vertical
            vertical.append((x1 + x2) // 2)
    
    if len(horizontal) < 2 or len(vertical) < 2:
        return None
    
    # Find extreme lines
    horizontal = sorted(horizontal)
    vertical = sorted(vertical)
    
    top = horizontal[0]
    bottom = horizontal[-1]
    left = vertical[0]
    right = vertical[-1]
    
    # Check if rectangle is large enough
    rect_area = (right - left) * (bottom - top)
    if rect_area < min_area:
        return None
    
    return np.array([
        [left, top],
        [right, top],
        [right, bottom],
        [left, bottom]
    ], dtype=np.float32)


def detect_with_gradient_magnitude(gray: np.ndarray, min_area: float) -> Optional[np.ndarray]:
    """Strategy 8: Gradient magnitude - robust edge detection"""
    # Sobel gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    max_val = magnitude.max()
    if max_val == 0:
        return None  # No gradients found (uniform image)
    magnitude = np.uint8(255 * magnitude / max_val)
    
    # Threshold
    _, thresh = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
    
    # Clean up
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    for contour in contours:
        quad = find_quad_from_contour(contour, min_area)
        if quad is not None:
            return quad
    
    return None


# =============================================================================
# MAIN DETECTION FUNCTION
# =============================================================================

def find_document_contour(image: np.ndarray) -> DetectionResult:
    """
    Find document using multiple strategies with validation
    Returns DetectionResult with contour, confidence, and method used
    """
    height, width = image.shape[:2]
    image_area = height * width
    min_area = image_area * 0.15  # Document must be at least 15% of image
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Define detection strategies in order of reliability
    strategies = [
        ("adaptive_threshold", lambda: detect_with_adaptive_threshold(gray, min_area)),
        ("canny_multi", lambda: detect_with_canny_multi(gray, min_area)),
        ("lab_color", lambda: detect_with_lab(image, min_area)),
        ("otsu", lambda: detect_with_otsu(gray, min_area)),
        ("inverted", lambda: detect_with_inverted(gray, min_area)),
        ("saturation", lambda: detect_with_saturation(image, min_area)),
        ("gradient", lambda: detect_with_gradient_magnitude(gray, min_area)),
        ("hough_lines", lambda: detect_with_hough_lines(gray, min_area, width, height)),
    ]
    
    best_result = DetectionResult(None, 0.0, "none")
    
    for method_name, detect_func in strategies:
        try:
            contour = detect_func()
            if contour is not None:
                is_valid, confidence = validate_quadrilateral(contour, image.shape)
                if is_valid and confidence > best_result.confidence:
                    best_result = DetectionResult(contour, confidence, method_name)

                    # If we found a high-confidence result, use it immediately
                    if confidence >= 0.8:
                        return best_result
        except Exception as e:
            # Log but continue with other strategies
            print(f"Detection strategy {method_name} failed: {e}")
            continue

    return best_result


# =============================================================================
# PERSPECTIVE TRANSFORM
# =============================================================================

def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Apply perspective transformation to get top-down view of document
    Uses high-quality interpolation for sharp output
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Compute dimensions of the new image
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = max(int(width_top), int(width_bottom))
    
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = max(int(height_left), int(height_right))
    
    # Ensure minimum dimensions
    max_width = max(max_width, 100)
    max_height = max(max_height, 100)
    
    # Destination points
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)
    
    # Compute and apply perspective transform with high quality interpolation
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(
        image, M, (max_width, max_height),
        flags=cv2.INTER_CUBIC,  # High-quality interpolation
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return warped


# =============================================================================
# IMAGE ENHANCEMENT
# =============================================================================

def apply_clahe(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Apply CLAHE for contrast enhancement while preserving local detail"""
    if len(image.shape) == 2:
        # Grayscale
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        return clahe.apply(image)
    else:
        # Color - apply to L channel in LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def unsharp_mask(image: np.ndarray, sigma: float = 1.0, strength: float = 1.0) -> np.ndarray:
    """
    Apply unsharp masking for sharpening without harsh artifacts
    This preserves text readability much better than kernel-based sharpening
    """
    # Create blurred version
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    
    # Unsharp mask: original + strength * (original - blurred)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    
    return sharpened


def enhance_image(image: np.ndarray, quality: str) -> np.ndarray:
    """
    Enhance image based on quality setting
    Uses professional-grade techniques that preserve text sharpness
    """
    if quality == "draft":
        # Minimal processing - just light contrast boost
        return cv2.convertScaleAbs(image, alpha=1.05, beta=5)
    
    elif quality == "high":
        # Premium enhancement pipeline
        
        # 1. Denoise while preserving edges (bilateral filter)
        denoised = cv2.bilateralFilter(image, 9, 50, 50)
        
        # 2. CLAHE for contrast (preserves local detail)
        contrasted = apply_clahe(denoised, clip_limit=2.0)
        
        # 3. Gentle unsharp masking for text clarity
        sharpened = unsharp_mask(contrasted, sigma=1.0, strength=0.5)
        
        # 4. Final brightness/contrast adjustment
        enhanced = cv2.convertScaleAbs(sharpened, alpha=1.1, beta=5)
        
        return enhanced
    
    else:  # standard
        # Balanced enhancement
        
        # 1. Light denoising
        denoised = cv2.bilateralFilter(image, 5, 30, 30)
        
        # 2. Moderate CLAHE
        contrasted = apply_clahe(denoised, clip_limit=1.5)
        
        # 3. Light sharpening
        sharpened = unsharp_mask(contrasted, sigma=1.0, strength=0.3)
        
        return sharpened


def apply_color_mode(image: np.ndarray, color_mode: str) -> np.ndarray:
    """Apply color mode transformation"""
    if color_mode == "grayscale":
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    elif color_mode == "bw":
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Use adaptive threshold for clean black/white
        return cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            15, 8  # Larger block size for cleaner result
        )
    
    else:  # color
        return image


# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

def process_document(image_bytes: bytes, options: ScanOptions) -> dict:
    """
    Full document processing pipeline

    Pipeline:
    1. Decode and validate image
    2. Detect document edges (multi-strategy)
    3. Apply perspective correction (if confident)
    4. Enhance image quality
    5. Apply color mode
    6. Generate PDF

    Returns dict with processed files and metadata
    """
    start_time = time.time()
    
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if original_image is None:
        raise ValueError("Invalid image data - could not decode")
    
    original_height, original_width = original_image.shape[:2]
    
    # Downscale for detection if image is very large (performance optimization)
    detection_max = 1500  # Larger than before for better accuracy
    scale_factor = 1.0
    
    if max(original_height, original_width) > detection_max:
        scale_factor = detection_max / max(original_height, original_width)
        detection_width = int(original_width * scale_factor)
        detection_height = int(original_height * scale_factor)
        detection_image = cv2.resize(
            original_image, 
            (detection_width, detection_height), 
            interpolation=cv2.INTER_AREA
        )
    else:
        detection_image = original_image
    
    # Store original with high quality
    _, original_encoded = cv2.imencode(
        '.jpg', original_image, 
        [cv2.IMWRITE_JPEG_QUALITY, 92]  # Higher quality
    )
    original_bytes = original_encoded.tobytes()
    
    # Find document contour
    detection_result = find_document_contour(detection_image)

    document_detected = False
    if detection_result.contour is not None and detection_result.confidence >= 0.5:
        # Scale contour back to original size
        contour = detection_result.contour
        if scale_factor < 1.0:
            contour = contour / scale_factor
        
        # Apply perspective transform
        try:
            warped = four_point_transform(original_image, contour)
            document_detected = True
        except Exception as e:
            print(f"Perspective transform failed: {e}")
            warped = original_image.copy()
    else:
        # Use original if no document found or low confidence
        warped = original_image.copy()
    
    # Apply enhancements
    enhanced = enhance_image(warped, options.quality)
    
    # Apply color mode
    final_image = apply_color_mode(enhanced, options.colorMode)
    
    # Get dimensions
    if len(final_image.shape) == 2:
        height, width = final_image.shape
    else:
        height, width = final_image.shape[:2]
    
    # Convert grayscale back to BGR for consistent output
    if len(final_image.shape) == 2:
        final_image_bgr = cv2.cvtColor(final_image, cv2.COLOR_GRAY2BGR)
    else:
        final_image_bgr = final_image
    
    # Encode processed image with high quality
    _, processed_encoded = cv2.imencode(
        '.jpg', final_image_bgr, 
        [cv2.IMWRITE_JPEG_QUALITY, 92]  # Higher quality
    )
    processed_bytes = processed_encoded.tobytes()
    
    # Generate PDF
    pil_image = Image.fromarray(cv2.cvtColor(final_image_bgr, cv2.COLOR_BGR2RGB))
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='JPEG', quality=92)
    img_byte_arr.seek(0)
    
    pdf_bytes = img2pdf.convert(img_byte_arr.getvalue())
    
    # Calculate processing time
    processing_time_ms = int((time.time() - start_time) * 1000)
    
    return {
        "original_bytes": original_bytes,
        "processed_bytes": processed_bytes,
        "pdf_bytes": pdf_bytes,
        "width": width,
        "height": height,
        "document_detected": document_detected,
        "processing_time_ms": processing_time_ms,
        "detection_method": detection_result.method,
        "detection_confidence": detection_result.confidence,
    }
