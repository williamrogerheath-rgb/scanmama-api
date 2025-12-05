#!/bin/bash
# Helper script to copy and rename test images
# Place your original IMG_*.JPG files in the same directory as this script,
# then run: bash tests/setup_test_images.sh

SOURCE_DIR="${1:-.}"  # First argument or current directory
DEST_DIR="tests/test_images"

# Create destination directory
mkdir -p "$DEST_DIR"

# Copy and rename images
declare -A IMAGE_MAP=(
    ["IMG_2457.JPG"]="doc_good.jpg"
    ["IMG_2458.JPG"]="doc_shadow.jpg"
    ["IMG_2464.JPG"]="doc_angled.jpg"
    ["IMG_2462.JPG"]="doc_close.jpg"
    ["IMG_2466.jpg"]="doc_far.jpg"
    ["IMG_2459.JPG"]="id_light.jpg"
    ["IMG_2460.JPG"]="id_dark.jpg"
    ["IMG_2461.JPG"]="insurance.jpg"
    ["IMG_2465.jpg"]="title.jpg"
)

echo "Copying test images from: $SOURCE_DIR"
echo "Destination: $DEST_DIR"
echo ""

COPIED=0
MISSING=0

for src in "${!IMAGE_MAP[@]}"; do
    dest="${IMAGE_MAP[$src]}"
    src_path="$SOURCE_DIR/$src"
    dest_path="$DEST_DIR/$dest"

    if [ -f "$src_path" ]; then
        cp "$src_path" "$dest_path"
        echo "✅ Copied: $src -> $dest"
        ((COPIED++))
    else
        echo "❌ Missing: $src"
        ((MISSING++))
    fi
done

echo ""
echo "Summary: $COPIED copied, $MISSING missing"

if [ $COPIED -eq 9 ]; then
    echo "✅ All test images ready! Run: python -m tests.test_detection"
elif [ $COPIED -gt 0 ]; then
    echo "⚠️  Some images are missing. Tests will run for available images."
else
    echo "❌ No images found in $SOURCE_DIR"
    echo "Usage: bash tests/setup_test_images.sh [path/to/images]"
fi
