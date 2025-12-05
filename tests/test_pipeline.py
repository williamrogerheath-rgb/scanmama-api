"""
Full pipeline integration test

Tests the complete document processing pipeline on all test images.
Saves output images and reports results.
"""
import os
import sys
import cv2
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.processing import process_document
from app.models.schemas import ScanOptions


def run_pipeline_test():
    """Run full pipeline on all test images"""
    test_dir = Path(__file__).parent / "test_images"
    output_dir = Path(__file__).parent / "test_outputs"

    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)

    # Test images
    test_images = [
        "doc_good.jpg",
        "doc_shadow.jpg",
        "doc_angled.jpg",
        "doc_close.jpg",
        "doc_far.jpg",
        "id_light.jpg",
        "id_dark.jpg",
        "insurance.jpg",
        "title.jpg",
    ]

    print("=" * 80)
    print("ScanMama Pipeline Integration Test")
    print("=" * 80)
    print()

    results = []
    total_time = 0

    for filename in test_images:
        image_path = test_dir / filename

        # Check if image exists
        if not image_path.exists():
            print(f"[SKIP]: {filename} - File not found")
            continue

        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[ERROR]: {filename} - Could not load image")
            continue

        # Encode to bytes
        _, img_bytes = cv2.imencode('.jpg', image)
        img_bytes = img_bytes.tobytes()

        # Run full pipeline with color mode
        options = ScanOptions(colorMode="color")
        result = process_document(img_bytes, options)

        # Decode processed image for saving
        import numpy as np
        nparr = np.frombuffer(result["processed_bytes"], np.uint8)
        processed_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Save processed output
        output_path = output_dir / f"pipeline_{filename}"
        cv2.imwrite(str(output_path), processed_img)

        # Print results
        print(f"[PROCESSED]: {filename}")
        print(f"   Document detected: {result['document_detected']}")
        print(f"   Confidence: {result['detection_confidence']:.3f}")
        print(f"   Method: {result['detection_method']}")
        print(f"   Processing time: {result['processing_time_ms']}ms")
        print(f"   Output dimensions: {result['width']}x{result['height']}")
        print(f"   Original size: {len(result['original_bytes'])} bytes")
        print(f"   Processed size: {len(result['processed_bytes'])} bytes")
        print(f"   PDF size: {len(result['pdf_bytes'])} bytes")
        print()

        # Track results
        total_time += result['processing_time_ms']
        results.append({
            'filename': filename,
            'detected': result['document_detected'],
            'confidence': result['detection_confidence'],
            'time_ms': result['processing_time_ms'],
            'width': result['width'],
            'height': result['height']
        })

    # Print summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total images processed: {len(results)}")
    print(f"Documents detected: {sum(1 for r in results if r['detected'])}")
    print(f"Average processing time: {total_time // len(results) if results else 0}ms")
    print(f"Total processing time: {total_time}ms")
    print()
    print(f"Output images saved to: {output_dir}")
    print()

    return results


if __name__ == "__main__":
    run_pipeline_test()
