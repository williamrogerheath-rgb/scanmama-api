# ScanMama API

**Cloud document scanning API for web developers**

Turn camera images into professional PDFs with edge detection, perspective correction, and automatic enhancement. No WASM, no client-side processing, no build configuration.

---

## Features

- **Edge Detection** - Automatically finds document boundaries
- **Perspective Correction** - Straightens skewed documents
- **Image Enhancement** - Adjusts contrast, sharpness, and brightness
- **PDF Generation** - Creates standard PDF files from scans
- **Multiple Quality Levels** - Draft, Standard, High
- **Color Modes** - Color, Grayscale, Black & White
- **Usage Tracking** - Monitors scan limits and overage
- **Secure Storage** - Files stored in Supabase with signed URLs

---

## Tech Stack

- **FastAPI** - Modern, fast web framework
- **OpenCV** - Computer vision processing
- **Supabase** - Database and file storage
- **img2pdf** - PDF generation
- **Python 3.10+** - Required

---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Required variables:
- `SUPABASE_URL` - Your Supabase project URL
- `SUPABASE_SERVICE_KEY` - Your Supabase service role key
- `API_BASE_URL` - Base URL for the API (default: http://localhost:8000)

### 3. Run the Server

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

---

## API Documentation

Once running, visit:
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## API Endpoints

### Health Check

```
GET /health
```

Returns API status and version.

### Scan Document

```
POST /v1/scan
Authorization: Bearer <api_key>

{
  "image": "base64_encoded_image_data",
  "options": {
    "outputFormat": "pdf",
    "quality": "standard",
    "colorMode": "color"
  }
}
```

Process a document scan and return URLs to the processed files.

### Get Scan

```
GET /v1/scan/{scan_id}
Authorization: Bearer <api_key>
```

Retrieve a previously processed scan with fresh signed URLs.

### Validate API Key

```
POST /v1/validate
Authorization: Bearer <api_key>
```

Check if an API key is valid and get usage information without consuming a scan.

### Get Usage

```
GET /v1/usage
Authorization: Bearer <api_key>
```

Get current usage statistics for the authenticated user.

---

## Project Structure

```
scanmama-api/
├── app/
│   ├── config.py          # Configuration and environment variables
│   ├── routers/
│   │   ├── scan.py        # Scan endpoints
│   │   └── validate.py    # Validation endpoints
│   ├── services/
│   │   ├── auth.py        # API key validation
│   │   ├── processing.py  # OpenCV document processing
│   │   └── storage.py     # Supabase Storage operations
│   └── models/
│       └── schemas.py     # Pydantic request/response models
├── main.py                # FastAPI app entry point
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
└── README.md             # This file
```

---

## OpenCV Processing Pipeline

1. **Decode** - Convert base64 to image
2. **Grayscale** - Convert to single channel
3. **Blur** - Reduce noise with Gaussian blur
4. **Edge Detection** - Find edges with Canny
5. **Find Contours** - Locate document boundary
6. **Perspective Transform** - Straighten document if found
7. **Enhancement** - Adjust contrast and sharpness based on quality
8. **Color Mode** - Apply grayscale or black & white if requested
9. **PDF Generation** - Create PDF from processed image

---

## Usage Limits

The API enforces usage limits based on the user's plan:

| Plan | Included Scans | Overage Rate | Rate Limit |
|------|----------------|--------------|------------|
| Trial | 10 (lifetime) | Must upgrade | 10/min |
| Pro | 500/month | $0.05/scan | 100/min |
| Team | 2,000/month | $0.04/scan | 500/min |

---

## Error Handling

The API returns structured errors:

```json
{
  "error": {
    "code": "trial_expired",
    "message": "Your trial has expired. Upgrade to Pro for 500 scans/month.",
    "upgradeUrl": "https://scanmama.com/dashboard/billing",
    "details": {
      "trialScansUsed": 10,
      "trialScansLimit": 10
    }
  }
}
```

Common error codes:
- `invalid_api_key` - API key not found or invalid
- `trial_expired` - Trial plan exhausted
- `invalid_image` - Image data malformed
- `file_too_large` - Image exceeds 10MB limit
- `processing_failed` - OpenCV processing error
- `storage_error` - File upload failed

---

## Development

### Run with Auto-reload

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Run Tests (Coming Soon)

```bash
pytest tests/
```

---

## Production Deployment

### Railway

1. Create new project
2. Add environment variables
3. Deploy from GitHub

### Render

1. Create new Web Service
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add environment variables

### Docker (Coming Soon)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Security

- API keys are hashed with SHA-256 before database lookup
- Service role key used for Supabase operations
- Files stored with signed URLs (1 hour expiry)
- CORS configured (restrict origins in production)
- Input validation on all endpoints
- File size limits enforced (10MB max)

---

## Support

- **Documentation**: https://scanmama.com/docs
- **Issues**: https://github.com/scanmama/api/issues
- **Email**: support@scanmama.com

---

## License

Proprietary - ScanMama by Bacano Systems LLC

---

## Version History

### 0.1.0 (Current)
- Initial release
- Document scanning with OpenCV
- PDF generation
- Supabase integration
- Usage tracking and limits
- Multiple quality levels and color modes
