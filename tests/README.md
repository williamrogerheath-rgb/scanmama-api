# Detection Tests

Test harness for validating the two-mode document detection module.

## Setup

### 1. Place test images in `tests/test_images/`

Copy and rename the 9 test images to `tests/test_images/` with these names:

| Original File | New Name | Description |
|--------------|----------|-------------|
| IMG_2457.JPG | doc_good.jpg | Document - straight on |
| IMG_2458.JPG | doc_shadow.jpg | Document - with shadow |
| IMG_2464.JPG | doc_angled.jpg | Document - angled perspective |
| IMG_2462.JPG | doc_close.jpg | Document - fills frame |
| IMG_2466.jpg | doc_far.jpg | Document - smaller in frame |
| IMG_2459.JPG | id_light.jpg | ID card - light background (HARD) |
| IMG_2460.JPG | id_dark.jpg | ID card - dark background |
| IMG_2461.JPG | insurance.jpg | Insurance card |
| IMG_2465.jpg | title.jpg | Vehicle title - green paper |

### 2. Run the tests

```bash
python -m tests.test_detection
```

## Expected Results

### Document Mode (confidence >= 0.6)
- doc_good.jpg
- doc_shadow.jpg
- doc_angled.jpg
- doc_close.jpg
- doc_far.jpg
- title.jpg

### ID Card Mode (confidence >= 0.6)
- id_light.jpg
- id_dark.jpg
- insurance.jpg

## Output

Debug images with detected corners drawn will be saved to `tests/test_outputs/`:
- Red lines: Detected quadrilateral
- Green dots: Corner points (labeled TL, TR, BR, BL)
- Black overlay: Detection metadata (mode, confidence, method)

## Test Script Features

- Loads each test image
- Runs two-mode detection
- Validates expected mode and confidence threshold
- Prints detailed results for each test
- Saves debug visualization images
- Provides summary statistics
