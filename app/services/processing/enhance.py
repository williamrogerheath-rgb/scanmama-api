"""
Image enhancement with CLAHE and sharpening

Applies contrast enhancement and sharpening for document clarity:
- Gentle CLAHE for contrast
- Subtle sharpening for clarity
"""
import numpy as np
import cv2


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
    Minimal enhancement - just gentle CLAHE, no shadow removal.
    The aggressive enhancement was destroying image quality.
    """
    print(f"Enhancement: Applying gentle CLAHE only (clipLimit=1.0)")

    # Convert to LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Very gentle CLAHE only
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # Merge and convert back
    lab = cv2.merge([l, a, b])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    print(f"  Enhancement complete")
    return result
