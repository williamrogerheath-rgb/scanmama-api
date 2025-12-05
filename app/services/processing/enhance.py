"""
Conditional image enhancement with intelligent shadow removal

Analyzes image lighting and applies appropriate enhancement:
- Shadow removal only when needed (based on brightness variation)
- Gentle CLAHE for contrast
- Subtle sharpening for clarity
"""
import numpy as np
import cv2
from typing import Tuple


def analyze_lighting(image: np.ndarray) -> Tuple[float, str]:
    """
    Analyze lighting variation to determine shadow removal strategy

    Divides image into 4x4 grid and measures brightness variation.

    Returns:
        (variation_score, recommendation)
        - variation < 25: "none" (uniform lighting)
        - variation 25-50: "gentle" (moderate shadows)
        - variation > 50: "full" (strong shadows)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    h, w = gray.shape
    grid_h, grid_w = h // 4, w // 4

    # Calculate mean brightness for each 4x4 grid cell
    brightness_values = []
    for i in range(4):
        for j in range(4):
            y1, y2 = i * grid_h, (i + 1) * grid_h
            x1, x2 = j * grid_w, (j + 1) * grid_w
            region = gray[y1:y2, x1:x2]
            brightness_values.append(np.mean(region))

    # Calculate variation
    variation = np.max(brightness_values) - np.min(brightness_values)

    # Determine recommendation
    if variation < 25:
        recommendation = "none"
    elif variation <= 50:
        recommendation = "gentle"
    else:
        recommendation = "full"

    return float(variation), recommendation


def remove_shadows(image: np.ndarray, strength: str = "full") -> np.ndarray:
    """
    Remove shadows using LAB color space normalization

    Args:
        strength: "gentle" (kernel 21) or "full" (kernel 31)
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply median blur to estimate background illumination
    kernel_size = 21 if strength == "gentle" else 31
    background = cv2.medianBlur(l, kernel_size)

    # Normalize L channel by removing background
    l_normalized = cv2.subtract(l, background)
    l_normalized = cv2.add(l_normalized, 128)  # Re-center

    # Merge and convert back
    lab_normalized = cv2.merge([l_normalized, a, b])
    result = cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2BGR)

    return result


def apply_clahe(image: np.ndarray, clip_limit: float = 1.5) -> np.ndarray:
    """
    Apply gentle CLAHE for contrast enhancement

    Uses LAB color space to preserve color while enhancing contrast.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    # Merge and convert back
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    return result


def sharpen(image: np.ndarray, strength: float = 0.3) -> np.ndarray:
    """
    Apply gentle unsharp mask for subtle sharpening

    Args:
        strength: Sharpening intensity (0.0-1.0, default 0.3 for subtle effect)
    """
    # Create Gaussian blur
    blurred = cv2.GaussianBlur(image, (5, 5), 1.0)

    # Unsharp mask: original + strength * (original - blurred)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)

    return sharpened


def enhance(image: np.ndarray) -> np.ndarray:
    """
    Main enhancement pipeline with conditional shadow removal

    1. Analyze lighting variation
    2. Apply shadow removal only if needed (variation >= 25)
    3. Apply gentle CLAHE (always)
    4. Apply subtle sharpening (always)

    Returns:
        Enhanced image
    """
    # Analyze lighting to determine shadow removal strategy
    variation, recommendation = analyze_lighting(image)

    # Apply shadow removal conditionally
    if recommendation != "none":
        image = remove_shadows(image, strength=recommendation)

    # Always apply gentle CLAHE for contrast
    image = apply_clahe(image, clip_limit=1.5)

    # Always apply subtle sharpening
    image = sharpen(image, strength=0.3)

    return image
