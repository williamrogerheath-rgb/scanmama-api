"""
Scan endpoints - document processing
"""
import uuid
from fastapi import APIRouter, Header, HTTPException
from supabase import create_client

from app.config import Config
from app.models.schemas import (
    ScanRequest,
    ScanResult,
    ErrorResponse,
    ErrorDetail,
    UsageInfo
)
from app.services.auth import (
    get_api_key_from_header,
    validate_api_key,
    check_scan_allowance,
    calculate_usage_info
)
from app.services.processing import process_document, decode_base64_image
from app.services.storage import upload_scan_files, get_signed_url

router = APIRouter(prefix="/v1", tags=["scan"])


def get_supabase_client():
    """Get Supabase client"""
    return create_client(Config.SUPABASE_URL, Config.SUPABASE_SERVICE_KEY)


@router.post("/scan", response_model=ScanResult)
async def create_scan(
    request: ScanRequest,
    authorization: str = Header(None)
):
    """
    Process document scan
    Requires: Authorization header with API key
    """
    # Validate API key
    api_key = get_api_key_from_header(authorization)
    result = await validate_api_key(api_key)

    if not result:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "code": "invalid_api_key",
                    "message": "Invalid API key",
                }
            }
        )

    api_key_record, user_record = result

    # Check scan allowance
    can_scan, is_overage = check_scan_allowance(user_record)

    if not can_scan:
        # Trial expired
        raise HTTPException(
            status_code=403,
            detail={
                "error": {
                    "code": "trial_expired",
                    "message": "Your trial has expired. Upgrade to Pro for 500 scans/month.",
                    "upgradeUrl": f"{Config.API_BASE_URL}/dashboard/billing",
                    "details": {
                        "trialScansUsed": 10,
                        "trialScansLimit": 10
                    }
                }
            }
        )

    # Decode image
    try:
        image_bytes = decode_base64_image(request.image)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "code": "invalid_image",
                    "message": "Invalid base64 image data",
                }
            }
        )

    # Check file size
    if len(image_bytes) > Config.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "code": "file_too_large",
                    "message": f"Image size exceeds maximum of {Config.MAX_FILE_SIZE / 1024 / 1024}MB",
                }
            }
        )

    # Process document
    try:
        processed = await process_document(image_bytes, request.options)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "processing_failed",
                    "message": f"Failed to process image: {str(e)}",
                }
            }
        )

    # Generate scan ID
    scan_id = f"sc_{uuid.uuid4().hex[:16]}"

    # Upload files to storage
    try:
        storage_result = await upload_scan_files(
            scan_id,
            processed["original_bytes"],
            processed["processed_bytes"],
            processed["pdf_bytes"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "code": "storage_error",
                    "message": f"Failed to upload files: {str(e)}",
                }
            }
        )

    # Record scan in database
    supabase = get_supabase_client()

    try:
        # Insert scan record
        scan_data = {
            "id": scan_id,
            "user_id": user_record["id"],
            "api_key_id": api_key_record["id"],
            "status": "complete",
            "is_overage": is_overage,
            "output_format": request.options.outputFormat,
            "quality": request.options.quality,
            "file_size_bytes": len(processed["pdf_bytes"]),
            "processing_time_ms": processed["processing_time_ms"],
            "pdf_storage_path": storage_result["pdf_path"],
            "image_storage_path": storage_result["processed_path"],
            "original_storage_path": storage_result["original_path"],
        }

        supabase.table("scans").insert(scan_data).execute()

        # Update user scan count
        plan = user_record["plan"]
        if plan == "trial":
            supabase.table("users").update({
                "trial_scans_remaining": user_record["trial_scans_remaining"] - 1
            }).eq("id", user_record["id"]).execute()
        else:
            supabase.table("users").update({
                "current_period_scans": user_record["current_period_scans"] + 1
            }).eq("id", user_record["id"]).execute()

        # Update daily usage aggregate
        from datetime import date
        today = date.today().isoformat()

        supabase.rpc("upsert_daily_usage", {
            "p_user_id": user_record["id"],
            "p_date": today,
            "p_scan_count": 1,
            "p_overage_count": 1 if is_overage else 0
        }).execute()

    except Exception as e:
        print(f"Error recording scan: {e}")
        # Don't fail the request if database update fails
        # The scan was already processed successfully

    # Calculate updated usage info
    # Update user record to reflect new scan
    if plan == "trial":
        user_record["trial_scans_remaining"] -= 1
    else:
        user_record["current_period_scans"] += 1

    usage_info = calculate_usage_info(user_record, is_overage)

    # Return result
    return ScanResult(
        scanId=scan_id,
        status="complete",
        pdfUrl=storage_result["pdf_url"] if request.options.outputFormat == "pdf" else None,
        imageUrl=storage_result["processed_url"],
        originalUrl=storage_result["original_url"],
        width=processed["width"],
        height=processed["height"],
        processingTimeMs=processed["processing_time_ms"],
        documentDetected=processed["document_detected"],
        usage=UsageInfo(**usage_info)
    )


@router.get("/scan/{scan_id}", response_model=ScanResult)
async def get_scan(
    scan_id: str,
    authorization: str = Header(None)
):
    """
    Get scan details by ID
    Requires: Authorization header with API key
    """
    # Validate API key
    api_key = get_api_key_from_header(authorization)
    result = await validate_api_key(api_key)

    if not result:
        raise HTTPException(status_code=401, detail="Invalid API key")

    api_key_record, user_record = result

    # Fetch scan from database
    supabase = get_supabase_client()

    try:
        response = supabase.table("scans") \
            .select("*") \
            .eq("id", scan_id) \
            .eq("user_id", user_record["id"]) \
            .single() \
            .execute()

        if not response.data:
            raise HTTPException(status_code=404, detail="Scan not found")

        scan_data = response.data

    except Exception as e:
        raise HTTPException(status_code=404, detail="Scan not found")

    # Generate fresh signed URLs
    bucket = Config.STORAGE_BUCKET
    pdf_url = await get_signed_url(bucket, scan_data["pdf_storage_path"]) if scan_data.get("pdf_storage_path") else None
    image_url = await get_signed_url(bucket, scan_data["image_storage_path"]) if scan_data.get("image_storage_path") else None
    original_url = await get_signed_url(bucket, scan_data["original_storage_path"]) if scan_data.get("original_storage_path") else None

    # Get current usage info
    usage_info = calculate_usage_info(user_record)

    return ScanResult(
        scanId=scan_data["id"],
        status=scan_data["status"],
        pdfUrl=pdf_url,
        imageUrl=image_url,
        originalUrl=original_url,
        width=0,  # Not stored, would need to parse from image
        height=0,
        processingTimeMs=scan_data.get("processing_time_ms", 0),
        documentDetected=True,  # Assume true for completed scans
        usage=UsageInfo(**usage_info)
    )
