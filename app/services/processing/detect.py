"""
Document detection utilities

Shared utilities for document detection (used by detect_ml.py).
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
    mode: str  # "ml_detection", "fallback", etc.
    method: str  # Which detection method was used


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


def calculate_confidence(
    corners: np.ndarray,
    image_shape: Tuple[int, int],
    area_range: Tuple[float, float],
    aspect_range: Tuple[float, float]
) -> float:
    """
    Calculate detection confidence based on geometric properties.

    Multi-factor scoring: 0.35*area + 0.25*aspect + 0.20*convexity + 0.20*edge

    Args:
        corners: 4 corner points
        image_shape: (height, width) of image
        area_range: (min, max) acceptable area ratio (0.0-1.0)
        aspect_range: (min, max) acceptable aspect ratio

    Returns:
        Confidence score 0.0-1.0
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
