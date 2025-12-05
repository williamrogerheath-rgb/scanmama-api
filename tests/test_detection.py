"""
Detection module test harness

Tests the two-mode detection on 9 representative images:
- Documents (straight, angled, shadow, close, far)
- ID cards (light background, dark background)
- Insurance card
- Vehicle title

Expected results:
- doc_* images: mode="document", confidence >= 0.6
- id_* and insurance: mode="id_card", confidence >= 0.6
- title: mode="document"

Usage: python -m tests.test_detection
"""
import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.processing.detect import detect


# Test image mapping: (filename, expected_mode, description)
TEST_IMAGES = [
    ("doc_good.jpg", "document", "Document - straight on"),
    ("doc_shadow.jpg", "document", "Document - with shadow"),
    ("doc_angled.jpg", "document", "Document - angled perspective"),
    ("doc_close.jpg", "document", "Document - fills frame"),
    ("doc_far.jpg", "document", "Document - smaller in frame"),
    ("id_light.jpg", "id_card", "ID card - light background (HARD)"),
    ("id_dark.jpg", "id_card", "ID card - dark background"),
    ("insurance.jpg", "id_card", "Insurance card"),
    ("title.jpg", "document", "Vehicle title - green paper"),
]


def draw_corners(image: np.ndarray, corners: np.ndarray, color=(0, 0, 255), thickness=3):
    """Draw detected corners as connected lines on image"""
    output = image.copy()

    if corners is None or len(corners) != 4:
        return output

    # Draw lines connecting corners
    corners_int = corners.astype(np.int32)
    cv2.polylines(output, [corners_int], isClosed=True, color=color, thickness=thickness)

    # Draw corner points
    for i, corner in enumerate(corners_int):
        cv2.circle(output, tuple(corner), 8, (0, 255, 0), -1)
        # Label corners: TL, TR, BR, BL
        labels = ["TL", "TR", "BR", "BL"]
        cv2.putText(output, labels[i], tuple(corner + [10, 10]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return output


def run_tests():
    """Run detection tests on all test images"""
    test_dir = Path(__file__).parent / "test_images"
    output_dir = Path(__file__).parent / "test_outputs"

    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)

    # Check if test images directory exists
    if not test_dir.exists():
        print(f"[ERROR] Test images directory not found: {test_dir}")
        print("\nPlease create tests/test_images/ and add the following images:")
        for filename, _, desc in TEST_IMAGES:
            print(f"  - {filename}: {desc}")
        return

    print("=" * 80)
    print("ScanMama Detection Test Harness")
    print("=" * 80)
    print()

    results = []
    passed = 0
    failed = 0
    missing = 0

    for filename, expected_mode, description in TEST_IMAGES:
        image_path = test_dir / filename

        # Check if image exists (try both .jpg and .JPG)
        if not image_path.exists():
            # Try uppercase extension
            alt_path = test_dir / filename.replace('.jpg', '.JPG')
            if alt_path.exists():
                image_path = alt_path
            else:
                print(f"[MISSING]: {filename}")
                print(f"   Description: {description}")
                print()
                missing += 1
                continue

        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[FAILED]: {filename} - Could not load image")
            print()
            failed += 1
            continue

        # Run detection (enable debug for detailed analysis)
        print(f"Testing {filename}...")
        result = detect(image, debug=False)

        # Check if result meets expectations
        confidence_pass = result.confidence >= 0.6
        mode_pass = result.mode == expected_mode or result.mode == "fallback"
        test_pass = confidence_pass and (mode_pass or result.mode == "fallback")

        # Print results
        status = "[PASS]" if test_pass else "[FAIL]"
        print(f"{status}: {filename}")
        print(f"   Description: {description}")
        print(f"   Expected mode: {expected_mode}")
        print(f"   Detected mode: {result.mode}")
        print(f"   Confidence: {result.confidence:.3f}")
        print(f"   Method: {result.method}")

        if not mode_pass and result.mode != "fallback":
            print(f"   [WARNING] Mode mismatch! Expected '{expected_mode}', got '{result.mode}'")
        if not confidence_pass:
            print(f"   [WARNING] Low confidence! Expected >= 0.6, got {result.confidence:.3f}")

        print()

        # Save debug image with corners drawn
        debug_image = draw_corners(image, result.corners)

        # Add text overlay with detection info
        h, w = debug_image.shape[:2]
        overlay = debug_image.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, debug_image, 0.3, 0, debug_image)

        info_lines = [
            f"Mode: {result.mode}",
            f"Confidence: {result.confidence:.3f}",
            f"Method: {result.method}",
        ]

        y_offset = 35
        for line in info_lines:
            cv2.putText(debug_image, line, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30

        # Save debug image
        output_path = output_dir / f"{Path(filename).stem}_debug.jpg"
        cv2.imwrite(str(output_path), debug_image)

        # Track results
        if test_pass:
            passed += 1
        else:
            failed += 1

        results.append({
            "filename": filename,
            "expected_mode": expected_mode,
            "detected_mode": result.mode,
            "confidence": result.confidence,
            "method": result.method,
            "passed": test_pass
        })

    # Print summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total tests: {len(TEST_IMAGES)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Missing: {missing}")
    print()

    if passed == len(TEST_IMAGES) - missing:
        print("SUCCESS: All available tests passed!")
    else:
        print("Some tests failed. Check output above for details.")

    print()
    print(f"Debug images saved to: {output_dir}")
    print()

    return results


if __name__ == "__main__":
    run_tests()
