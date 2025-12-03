"""
Validation and usage endpoints
"""
from fastapi import APIRouter, Header, HTTPException

from app.models.schemas import ApiKeyInfo, UsageResponse, UsageInfo
from app.services.auth import (
    get_api_key_from_header,
    validate_api_key,
    get_api_key_info,
    calculate_usage_info
)

router = APIRouter(prefix="/v1", tags=["validate"])


@router.post("/validate", response_model=ApiKeyInfo)
async def validate_key(authorization: str = Header(None)):
    """
    Validate API key without using a scan
    Returns API key information and usage limits
    """
    # Extract API key
    api_key = get_api_key_from_header(authorization)

    # Get API key info
    info = await get_api_key_info(api_key)

    if not info:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "code": "invalid_api_key",
                    "message": "Invalid API key",
                }
            }
        )

    return ApiKeyInfo(**info)


@router.get("/usage", response_model=UsageResponse)
async def get_usage(authorization: str = Header(None)):
    """
    Get current usage statistics for the API key's user
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

    # Calculate usage info
    usage_info = calculate_usage_info(user_record)

    return UsageResponse(usage=UsageInfo(**usage_info))
