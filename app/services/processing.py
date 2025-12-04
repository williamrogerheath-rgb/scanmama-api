"""
OpenCV document processing services
Handles edge detection, perspective correction, and PDF generation
"""
import base64
import io
import numpy as np
import cv2
import img2pdf
from PIL import Image

from app.models.schemas import ScanOptions


def decode_base64_image(image_base64: str) -> bytes:
    """Decode base64 string to bytes"""
    # Remove data URL prefix if present
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]

    return base64.b64decode(image_base64)


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order points in the order: top-left, top-right, bottom-right, bottom-left
    """
    rect = np.zeros((4, 2), dtype="float32")

    # Sum: top-left will have smallest sum, bottom-right will have largest
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Difference: top-right will have smallest difference, bottom-left will have largest
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Apply perspective transformation to get top-down view of document
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Compute height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Destination points for the transform
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Compute perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def enhance_image(image: np.ndarray, quality: str) -> np.ndarray:
    """
    Enhance image based on quality setting
    """
    if quality == "draft":
        # Minimal processing
        return image
    elif quality == "high":
        # Aggressive enhancement
        # Increase contrast
        alpha = 1.5  # Contrast
        beta = 10    # Brightness
        enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        # Sharpen
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)

        return enhanced
    else:  # standard
        # Moderate enhancement
        alpha = 1.2
        beta = 5
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def apply_color_mode(image: np.ndarray, color_mode: str) -> np.ndarray:
    """
    Apply color mode transformation
    """
    if color_mode == "grayscale":
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    elif color_mode == "bw":
        # Convert to grayscale first
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        # Apply adaptive threshold for black and white
        return cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
    else:  # color
        return image


def find_largest_quad_with_hull(edge_image: np.ndarray, min_area: float, padding: int = 0) -> np.ndarray | None:
    """Find largest 4-sided contour, using convex hull if needed"""
    contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        # Try direct approximation first
        peri = cv2.arcLength(contour, True)
        for eps in [0.02, 0.04, 0.06]:
            approx = cv2.approxPolyDP(contour, eps * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)

        # Try convex hull if direct approx fails
        hull = cv2.convexHull(contour)
        peri = cv2.arcLength(hull, True)
        for eps in [0.02, 0.04, 0.06]:
            approx = cv2.approxPolyDP(hull, eps * peri, True)
            if len(approx) == 4:
                return approx.reshape(4, 2).astype(np.float32)

        # Last resort: minAreaRect
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        return box.astype(np.float32)

    return None


def find_document_from_lines(edges: np.ndarray, width: int, height: int, min_area: float) -> np.ndarray | None:
    """Use HoughLines to detect document edges and find corners"""
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=min(width, height)//4, maxLineGap=20)

    if lines is None or len(lines) < 4:
        return None

    # Separate horizontal and vertical lines
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

    # Find extreme lines (document edges)
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

    # Return corners
    return np.array([
        [left, top],
        [right, top],
        [right, bottom],
        [left, bottom]
    ], dtype=np.float32)


def adjust_contour_for_padding(contour: np.ndarray, padding: int) -> np.ndarray:
    """Subtract padding offset from contour coordinates"""
    return contour - padding


def find_document_contour(image: np.ndarray) -> tuple[np.ndarray | None, bool]:
    """Find document using multiple fast strategies"""
    height, width = image.shape[:2]
    image_area = height * width
    min_area = image_area * 0.15

    # Add 5px border (fixes edge detection when doc touches image edge)
    padded = cv2.copyMakeBorder(image, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Strategy 1: Otsu threshold + strong morphology
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((9, 9), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contour = find_largest_quad_with_hull(thresh, min_area, 5)
    if contour is not None:
        return adjust_contour_for_padding(contour, 5), True

    # Strategy 2: Canny with strong dilation to connect edges
    edges = cv2.Canny(blurred, 50, 150)
    dilate_kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, dilate_kernel, iterations=2)

    contour = find_largest_quad_with_hull(edges, min_area, 5)
    if contour is not None:
        return adjust_contour_for_padding(contour, 5), True

    # Strategy 3: HoughLines to find document edges
    edges2 = cv2.Canny(blurred, 30, 100)
    contour = find_document_from_lines(edges2, width + 10, height + 10, min_area)
    if contour is not None:
        return adjust_contour_for_padding(contour, 5), True

    return None, False


async def process_document(image_bytes: bytes, options: ScanOptions) -> dict:
    """
    Full OpenCV document processing pipeline
    Returns dict with processed files and metadata
    """
    import time
    start_time = time.time()

    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if original_image is None:
        raise ValueError("Invalid image data")

    # Get original dimensions
    original_height, original_width = original_image.shape[:2]

    # Create smaller image for fast detection (max 1000px)
    detection_max = 1000
    scale_factor = 1.0

    if max(original_height, original_width) > detection_max:
        scale_factor = detection_max / max(original_height, original_width)
        detection_width = int(original_width * scale_factor)
        detection_height = int(original_height * scale_factor)
        detection_image = cv2.resize(original_image, (detection_width, detection_height), interpolation=cv2.INTER_AREA)
    else:
        detection_image = original_image

    # Store original as bytes (JPEG quality 85)
    _, original_encoded = cv2.imencode('.jpg', original_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    original_bytes = original_encoded.tobytes()

    # Find document contour on smaller image
    doc_contour, document_detected = find_document_contour(detection_image)

    if document_detected and doc_contour is not None:
        # Scale contour back to original size if we downscaled
        if scale_factor < 1.0:
            doc_contour = doc_contour / scale_factor

        # Apply perspective transform on original high-res image
        warped = four_point_transform(original_image, doc_contour)
    else:
        # Use original if no document found
        warped = original_image.copy()

    # Apply enhancements
    enhanced = enhance_image(warped, options.quality)

    # Apply color mode
    final_image = apply_color_mode(enhanced, options.colorMode)

    # Get dimensions of processed image
    if len(final_image.shape) == 2:
        height, width = final_image.shape
    else:
        height, width = final_image.shape[:2]

    # Convert back to BGR if grayscale for consistent output
    if len(final_image.shape) == 2:
        final_image_bgr = cv2.cvtColor(final_image, cv2.COLOR_GRAY2BGR)
    else:
        final_image_bgr = final_image

    # Encode processed image to JPEG (quality 85)
    _, processed_encoded = cv2.imencode('.jpg', final_image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    processed_bytes = processed_encoded.tobytes()

    # Generate PDF
    # img2pdf works best with PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(final_image_bgr, cv2.COLOR_BGR2RGB))
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='JPEG', quality=85)
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
    }
