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


def find_document_contour(image: np.ndarray) -> tuple[np.ndarray | None, bool]:
    """
    Find document contour in image
    Returns: (contour, found)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection
    edges = cv2.Canny(blurred, 75, 200)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # Find the largest 4-point contour (document)
    for contour in contours:
        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # If contour has 4 points, we found the document
        if len(approx) == 4:
            return approx.reshape(4, 2), True

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
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid image data")

    # Resize image if too large (cap at 2000px max dimension)
    original_height, original_width = image.shape[:2]
    max_dimension = 2000

    if max(original_height, original_width) > max_dimension:
        # Calculate new dimensions while maintaining aspect ratio
        if original_height > original_width:
            new_height = max_dimension
            new_width = int(original_width * (max_dimension / original_height))
        else:
            new_width = max_dimension
            new_height = int(original_height * (max_dimension / original_width))

        # Resize using INTER_AREA for best quality when downscaling
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Store original as bytes (JPEG quality 85)
    _, original_encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    original_bytes = original_encoded.tobytes()

    # Find document contour
    doc_contour, document_detected = find_document_contour(image)

    if document_detected and doc_contour is not None:
        # Apply perspective transform
        warped = four_point_transform(image, doc_contour)
    else:
        # Use original if no document found
        warped = image.copy()

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
