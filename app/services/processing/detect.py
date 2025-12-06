"""
Two-mode document detection with confidence scoring

Modes:
- Document: area 20-95%, aspect ratio 0.5-2.0 (standard documents)
- ID Card: area 5-30%, aspect ratio 1.3-2.5 (credit/insurance cards)
"""
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DetectionResult:
    """Result of document detection"""
    corners: Optional[np.ndarray]  # 4 ordered points (tl, tr, br, bl)
    confidence: float  # 0.0 to 1.0
    mode: str  # "document" or "id_card"
    method: str  # Which preprocessing method succeeded


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order points: top-left, top-right, bottom-right, bottom-left"""
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def is_valid_quadrilateral(corners: np.ndarray) -> bool:
    """Check if quadrilateral is convex with roughly parallel sides"""
    if len(corners) != 4:
        return False
    hull = cv2.convexHull(corners, returnPoints=True)
    if len(hull) != 4:
        return False

    pts = order_points(corners)
    vectors = [pts[(i+1)%4] - pts[i] for i in range(4)]
    vectors = [v / (np.linalg.norm(v) + 1e-6) for v in vectors]

    # Opposite sides roughly parallel (relaxed from 0.7 to 0.45 for extreme angles)
    return (abs(np.dot(vectors[0], -vectors[2])) > 0.45 and
            abs(np.dot(vectors[1], -vectors[3])) > 0.45)


def calculate_confidence(
    corners: np.ndarray,
    image_shape: Tuple[int, int],
    area_range: Tuple[float, float],
    aspect_range: Tuple[float, float]
) -> float:
    """
    Multi-factor confidence: 0.35*area + 0.25*aspect + 0.20*convexity + 0.20*edge
    """
    height, width = image_shape[:2]
    image_area = height * width
    pts = order_points(corners)
    contour_area = cv2.contourArea(pts)
    area_ratio = contour_area / image_area

    # Area score
    min_area, max_area = area_range
    if area_ratio < min_area or area_ratio > max_area:
        return 0.0
    area_center = (min_area + max_area) / 2
    area_score = 1.0 - abs(area_ratio - area_center) / (max_area - min_area)

    # Aspect ratio score
    (tl, tr, br, bl) = pts
    avg_width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    avg_height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    if avg_width == 0 or avg_height == 0:
        return 0.0

    aspect_ratio = avg_width / avg_height
    min_aspect, max_aspect = aspect_range
    if aspect_ratio < min_aspect or aspect_ratio > max_aspect:
        aspect_ratio = 1.0 / aspect_ratio
        if aspect_ratio < min_aspect or aspect_ratio > max_aspect:
            return 0.0
    aspect_center = (min_aspect + max_aspect) / 2
    aspect_score = 1.0 - abs(aspect_ratio - aspect_center) / (max_aspect - min_aspect)

    # Convexity score
    hull = cv2.convexHull(pts)
    hull_area = cv2.contourArea(hull)
    convexity_score = contour_area / (hull_area + 1e-6)

    # Edge completeness score
    perimeter = cv2.arcLength(pts, True)
    expected_perimeter = 2 * (avg_width + avg_height)
    edge_score = min(perimeter / (expected_perimeter + 1e-6), 1.0)

    return 0.35 * area_score + 0.25 * aspect_score + 0.20 * convexity_score + 0.20 * edge_score


def find_best_contour(
    edges: np.ndarray,
    image_shape: Tuple[int, int],
    area_range: Tuple[float, float],
    aspect_range: Tuple[float, float],
    debug: bool = False,
    is_binary: bool = False,
    epsilon: float = 0.020
) -> Tuple[Optional[np.ndarray], float]:
    """
    Find best quadrilateral contour in edge-detected or binary image

    Args:
        is_binary: If True, input is binary image (not edge map), look for filled regions
        epsilon: Approximation epsilon for contour simplification (default: 0.020)
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_corners = None
    best_confidence = 0.0

    if debug:
        print(f"  Found {len(contours)} contours (epsilon={epsilon})")
        # Sort by area for debugging
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    else:
        sorted_contours = contours

    for idx, contour in enumerate(sorted_contours):
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon * peri, True)

        if debug and idx < 5:
            height, width = image_shape[:2]
            image_area = height * width
            area_ratio = cv2.contourArea(contour) / image_area
            area_pct = area_ratio * 100
            print(f"  Contour {idx+1}: area={area_pct:.1f}% ({area_ratio:.3f}), points={len(approx)}", end="")

        if len(approx) < 4 or len(approx) > 6:
            if debug and idx < 5:
                print(f" - REJECTED: point count")
            continue

        # Use minAreaRect for 5-6 points to get clean quad
        if len(approx) in [5, 6]:
            rect = cv2.minAreaRect(contour)
            corners = np.intp(cv2.boxPoints(rect))
        else:
            corners = approx.reshape(4, 2)

        if not is_valid_quadrilateral(corners):
            if debug and idx < 5:
                print(f" - REJECTED: not valid quad")
            continue

        confidence = calculate_confidence(corners, image_shape, area_range, aspect_range)

        if debug and idx < 5:
            # Calculate aspect ratio for debug
            pts = order_points(corners)
            (tl, tr, br, bl) = pts
            avg_width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
            avg_height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
            aspect = avg_width / avg_height if avg_height > 0 else 0
            # Calculate convexity
            hull = cv2.convexHull(pts)
            hull_area = cv2.contourArea(hull)
            contour_area = cv2.contourArea(pts)
            convexity = contour_area / (hull_area + 1e-6)

            # Show if confidence is too low
            min_area, max_area = area_range
            min_aspect, max_aspect = aspect_range
            area_ok = min_area <= area_ratio <= max_area
            aspect_ok = (min_aspect <= aspect <= max_aspect) or (min_aspect <= 1.0/aspect <= max_aspect)

            status = "✓ ACCEPTED" if confidence >= 0.5 else "✗ LOW CONF"
            print(f", aspect={aspect:.2f}, convexity={convexity:.2f}, conf={confidence:.3f} - {status}")

            if not area_ok:
                print(f"    └─ Area {area_pct:.1f}% outside range {min_area*100:.0f}-{max_area*100:.0f}%")
            if not aspect_ok:
                print(f"    └─ Aspect {aspect:.2f} outside range {min_aspect:.1f}-{max_aspect:.1f}")

        if confidence > best_confidence:
            best_confidence = confidence
            best_corners = corners

    return best_corners, best_confidence


def detect_mode(
    gray: np.ndarray,
    mode: str,
    area_range: Tuple[float, float],
    aspect_range: Tuple[float, float],
    confidence_threshold: float = 0.5,
    debug: bool = False
) -> Optional[DetectionResult]:
    """Try 4 preprocessing variants with adaptive epsilon fallback"""

    # Add 5px replicated border to prevent edge artifacts (avoids creating artificial edges)
    BORDER_SIZE = 5
    padded = cv2.copyMakeBorder(gray, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE,
                                cv2.BORDER_REPLICATE)

    # Morphological kernel for closing edge gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def morph_close(edges):
        """Apply morphological closing to connect edge gaps"""
        dilated = cv2.dilate(edges, kernel, iterations=1)
        return cv2.erode(dilated, kernel, iterations=1)

    # Preprocessing methods with morphological operations
    methods = [
        ("standard", lambda g: morph_close(cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 50, 150)), False),
        ("low_contrast", lambda g: morph_close(cv2.Canny(cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g), 30, 100)), False),
        ("textured", lambda g: morph_close(cv2.Canny(cv2.bilateralFilter(g, 9, 75, 75), 50, 150)), False),
        ("threshold", lambda g: cv2.threshold(cv2.GaussianBlur(g, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], True),
    ]

    # Try with default epsilon first
    for epsilon in [0.020, 0.030, 0.040]:
        if debug and epsilon > 0.020:
            print(f"  Fallback: trying looser epsilon={epsilon}")

        for method_name, preprocess_func, is_binary in methods:
            if debug:
                if epsilon == 0.020:
                    print(f"  Trying {method_name} preprocessing...")
                else:
                    print(f"    Trying {method_name} with epsilon={epsilon}...")
            processed = preprocess_func(padded)
            corners, confidence = find_best_contour(processed, padded.shape, area_range, aspect_range, debug, is_binary, epsilon)

            if corners is not None and confidence >= confidence_threshold:
                # Adjust corners to remove border offset
                adjusted_corners = corners - BORDER_SIZE
                method_suffix = f"_{epsilon}" if epsilon > 0.020 else ""
                return DetectionResult(
                    corners=order_points(adjusted_corners),
                    confidence=confidence,
                    mode=mode,
                    method=method_name + method_suffix
                )

        # If we found nothing with this epsilon, try next one
        # Only try looser epsilon if initial pass completely failed
        if epsilon == 0.020:
            continue

    return None


def detect(image: np.ndarray, debug: bool = False) -> DetectionResult:
    """
    Two-mode detection: try document and ID card modes
    Returns highest confidence result, prefers ID card on ties
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    h, w = gray.shape[:2]

    # ALWAYS log image info for debugging detection issues
    print(f"\n=== DOCUMENT DETECTION START ===")
    print(f"Image dimensions: {w}x{h} ({w*h} pixels)")
    print(f"Document mode expects: 20-95% area, 0.5-2.0 aspect ratio")
    print(f"ID Card mode expects: 5-30% area, 1.3-2.5 aspect ratio")

    # Try both modes
    print("\n--- Document mode ---")
    doc_result = detect_mode(gray, "document", (0.20, 0.95), (0.5, 2.0), debug=True)
    if doc_result:
        area_pct = cv2.contourArea(doc_result.corners) / (w * h) * 100
        print(f"Document result: confidence={doc_result.confidence:.3f}, area={area_pct:.1f}%, method={doc_result.method}")
    else:
        print("Document result: FAILED - no valid contour found")

    print("\n--- ID Card mode ---")
    id_result = detect_mode(gray, "id_card", (0.05, 0.30), (1.3, 2.5), debug=True)
    if id_result:
        area_pct = cv2.contourArea(id_result.corners) / (w * h) * 100
        print(f"ID Card result: confidence={id_result.confidence:.3f}, area={area_pct:.1f}%, method={id_result.method}")
    else:
        print("ID Card result: FAILED - no valid contour found")

    # Return best (prefer document mode when it has higher confidence, ID card on tie)
    if doc_result is None and id_result is None:
        print(f"\n>>> FALLBACK: Both modes failed, using full image")
        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        return DetectionResult(corners=corners, confidence=0.0, mode="fallback", method="none")

    if doc_result is None:
        print(f"\n>>> SELECTED: ID Card mode (document failed)")
        return id_result
    if id_result is None:
        print(f"\n>>> SELECTED: Document mode (ID card failed)")
        return doc_result

    # Prefer document mode if it has strictly higher confidence
    if doc_result.confidence > id_result.confidence:
        print(f"\n>>> SELECTED: Document mode (conf={doc_result.confidence:.3f} > {id_result.confidence:.3f})")
        return doc_result

    # Otherwise prefer ID card (including ties)
    print(f"\n>>> SELECTED: ID Card mode (conf={id_result.confidence:.3f} >= {doc_result.confidence:.3f})")
    return id_result
