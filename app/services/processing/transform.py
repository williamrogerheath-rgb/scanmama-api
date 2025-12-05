"""
Perspective correction with minimum resolution enforcement

Transforms detected document corners into a rectangular image with
guaranteed minimum resolution of 2000px on the long edge.
"""
import numpy as np
import cv2
from typing import Optional


MIN_OUTPUT_RESOLUTION = 2000  # Minimum pixels on long edge
MAX_ASPECT_RATIO = 5.0  # Reject extreme aspect ratios


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order points: top-left, top-right, bottom-right, bottom-left

    - Top-left: smallest sum (x+y)
    - Bottom-right: largest sum (x+y)
    - Top-right: smallest difference (x-y)
    - Bottom-left: largest difference (x-y)
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()

    rect[0] = pts[np.argmin(s)]      # Top-left
    rect[2] = pts[np.argmax(s)]      # Bottom-right
    rect[1] = pts[np.argmin(diff)]   # Top-right
    rect[3] = pts[np.argmax(diff)]   # Bottom-left

    return rect


def transform(image: np.ndarray, corners: np.ndarray) -> Optional[np.ndarray]:
    """
    Apply perspective correction with minimum resolution enforcement

    Args:
        image: Input image
        corners: 4 corner points (will be ordered automatically)

    Returns:
        Transformed image with minimum 2000px on long edge, or None if invalid
    """
    # Order corners
    rect = order_points(corners)
    (tl, tr, br, bl) = rect

    # Calculate width and height of output rectangle
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    max_width = max(int(width_top), int(width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    max_height = max(int(height_left), int(height_right))

    # Validate dimensions
    if max_width <= 0 or max_height <= 0:
        return None

    # Check aspect ratio
    aspect_ratio = max(max_width / max_height, max_height / max_width)
    if aspect_ratio > MAX_ASPECT_RATIO:
        return None

    # Enforce minimum resolution on long edge
    long_edge = max(max_width, max_height)
    if long_edge < MIN_OUTPUT_RESOLUTION:
        scale_factor = MIN_OUTPUT_RESOLUTION / long_edge
        max_width = int(max_width * scale_factor)
        max_height = int(max_height * scale_factor)

    # Destination points for perspective transform
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    # Compute perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)

    # Apply perspective transformation with high-quality interpolation
    warped = cv2.warpPerspective(
        image,
        M,
        (max_width, max_height),
        flags=cv2.INTER_CUBIC
    )

    return warped
