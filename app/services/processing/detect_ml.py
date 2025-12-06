"""
ML-based document detection using DocAligner

Uses deep learning (FastViT_SA24 heatmap model) for robust document corner
detection across various lighting conditions and backgrounds.

Drop-in replacement for traditional CV-based detect.py
"""
import numpy as np
import cv2
import logging
from typing import Optional

# Import DetectionResult and utilities from existing detect module
from .detect import DetectionResult, order_points, calculate_confidence

logger = logging.getLogger(__name__)


class DocAlignerDetector:
    """Singleton wrapper for DocAligner model"""
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(self):
        """Lazy load DocAligner model (downloads weights on first use)"""
        if self._model is None:
            try:
                from docaligner import DocAligner
                print("Loading DocAligner model (weights will auto-download on first run)...")
                self._model = DocAligner()
                print("DocAligner model loaded successfully")
            except ImportError:
                raise ImportError(
                    "DocAligner not installed. Install with: pip install docaligner-docsaid capybara-docsaid"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to load DocAligner model: {e}")
        return self._model


def detect(image: np.ndarray, debug: bool = False) -> DetectionResult:
    """
    ML-based document detection using DocAligner

    Drop-in replacement for traditional detect() function.
    Uses deep learning for robust corner detection.

    Args:
        image: BGR image as numpy array
        debug: Enable debug logging (always logs for consistency with original)

    Returns:
        DetectionResult with corners, confidence, mode, and method
    """
    h, w = image.shape[:2]

    # ALWAYS log image info for consistency with original detect.py
    print(f"\n=== ML DOCUMENT DETECTION START ===")
    print(f"Image dimensions: {w}x{h} ({w*h} pixels)")
    print(f"Using DocAligner (FastViT_SA24 heatmap model)")

    try:
        # Get DocAligner model (lazy loaded singleton)
        detector = DocAlignerDetector()
        model = detector.get_model()

        # Run inference - DocAligner expects BGR image
        print("Running DocAligner inference...")
        polygon = model(image)

        # Check if detection succeeded
        if polygon is None or len(polygon) == 0:
            print("DocAligner returned no detection")
            raise ValueError("No document detected")

        # DocAligner returns (4, 2) array with corners in order: TL, TR, BR, BL
        corners = np.array(polygon, dtype=np.float32)

        if corners.shape != (4, 2):
            # Detailed logging for 3-corner detections
            num_corners = corners.shape[0] if hasattr(corners, 'shape') else len(corners)
            logger.info(f"Partial detection raw data:")
            logger.info(f"  Corners received: {corners}")
            logger.info(f"  Corner shape: {corners.shape if hasattr(corners, 'shape') else len(corners)}")
            logger.info(f"  Image size: {w}x{h}")

            # Log individual corner coordinates if available
            if num_corners >= 3:
                for i, corner in enumerate(corners):
                    logger.info(f"  Corner {i}: x={corner[0]:.1f}, y={corner[1]:.1f}")

            # Check if DocAligner provides any confidence scores
            if hasattr(polygon, 'confidence'):
                logger.info(f"  Confidence scores: {polygon.confidence}")

            print(f"Partial detection: got {num_corners} corners instead of 4")
            print(f"This usually means document is partially visible or occluded")
            raise ValueError(f"Partial detection: {num_corners} corners")

        # Ensure corners are properly ordered (TL, TR, BR, BL)
        corners = order_points(corners)

        # Calculate confidence score using permissive ranges
        # DocAligner doesn't provide confidence, so we calculate it based on
        # geometric properties (area, aspect ratio, convexity, edge completeness)
        # Use wide ranges to support both documents (20-95%) and ID cards (5-30%)
        area_range = (0.05, 0.95)  # Accept 5-95% of image (ID cards to full documents)
        aspect_range = (0.3, 3.0)  # Permissive aspect ratios (portrait/landscape/square)
        confidence = calculate_confidence(corners, (h, w), area_range, aspect_range)

        # Calculate area percentage for logging
        area_pct = cv2.contourArea(corners) / (w * h) * 100

        # Calculate aspect ratio for logging
        (tl, tr, br, bl) = corners
        avg_width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
        avg_height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
        aspect = avg_width / avg_height if avg_height > 0 else 0

        # Determine document type based on area
        if area_pct < 30:
            doc_type = "ID card/small document"
        else:
            doc_type = "full document"

        print(f"Detection successful:")
        print(f"  Type: {doc_type}")
        print(f"  Area: {area_pct:.1f}% of image")
        print(f"  Aspect ratio: {aspect:.2f}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"\n>>> SELECTED: ML detection via DocAligner")

        return DetectionResult(
            corners=corners,
            confidence=confidence,
            mode="ml_detection",
            method="docaligner"
        )

    except Exception as e:
        # Fallback to full image if detection fails
        print(f"ML detection failed: {e}")
        print(f"\n>>> FALLBACK: Using full image")

        corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        return DetectionResult(
            corners=corners,
            confidence=0.0,
            mode="fallback",
            method="none"
        )
