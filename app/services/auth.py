"""
Authentication and authorization services
Handles API key validation and usage limits
"""
import hashlib
from typing import Optional, Tuple
from supabase import create_client, Client
from fastapi import HTTPException, Header

from app.config import Config


def get_supabase_client() -> Client:
    """Get Supabase client instance"""
    return create_client(Config.SUPABASE_URL, Config.SUPABASE_SERVICE_KEY)


def get_api_key_from_header(authorization: str) -> str:
    """
    Extract API key from Authorization header
    Expected format: "Bearer sm_live_xxx"
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization header format. Expected: Bearer <api_key>"
        )

    api_key = parts[1]
    if not api_key.startswith(("sm_live_", "sm_test_")):
        raise HTTPException(status_code=401, detail="Invalid API key format")

    return api_key


async def validate_api_key(key: str) -> Optional[Tuple[dict, dict]]:
    """
    Validate API key and return associated records
    Returns: (api_key_record, user_record) or None if invalid
    """
    # Hash the API key
    key_hash = hashlib.sha256(key.encode()).hexdigest()

    supabase = get_supabase_client()

    try:
        # Query api_keys table with user join
        response = supabase.table("api_keys") \
            .select("*, users(*)") \
            .eq("key_hash", key_hash) \
            .eq("is_active", True) \
            .single() \
            .execute()

        if not response.data:
            return None

        api_key_record = {k: v for k, v in response.data.items() if k != "users"}
        user_record = response.data.get("users")

        if not user_record:
            return None

        # Update last_used_at timestamp
        supabase.table("api_keys") \
            .update({"last_used_at": "now()"}) \
            .eq("id", api_key_record["id"]) \
            .execute()

        return api_key_record, user_record

    except Exception as e:
        print(f"Error validating API key: {e}")
        return None


def check_scan_allowance(user: dict) -> Tuple[bool, bool]:
    """
    Check if user can perform a scan
    Returns: (can_scan, is_overage)

    - Trial users: can scan if trial_scans_remaining > 0, no overage
    - Paid users: can always scan, overage if current_period_scans >= monthly_scan_limit
    """
    plan = user.get("plan", "trial")

    if plan == "trial":
        trial_remaining = user.get("trial_scans_remaining", 0)
        can_scan = trial_remaining > 0
        return (can_scan, False)

    # Pro or Team - always allowed, may be overage
    current_period_scans = user.get("current_period_scans", 0)
    monthly_limit = user.get("monthly_scan_limit", 0)
    is_overage = current_period_scans >= monthly_limit

    return (True, is_overage)


def get_rate_limit(plan: str) -> int:
    """Get rate limit for a plan (requests per minute)"""
    return Config.RATE_LIMITS.get(plan, Config.RATE_LIMITS["trial"])


def calculate_usage_info(user: dict, is_overage: bool = False) -> dict:
    """
    Calculate usage information for response
    """
    plan = user.get("plan", "trial")

    if plan == "trial":
        trial_remaining = user.get("trial_scans_remaining", 0)
        trial_total = 10
        trial_used = trial_total - trial_remaining

        return {
            "plan": plan,
            "scansUsedThisPeriod": trial_used,
            "scansIncluded": trial_total,
            "trialRemaining": trial_remaining,
            "overageScans": 0,
            "isOverage": False,
        }

    # Pro or Team
    current_period_scans = user.get("current_period_scans", 0)
    monthly_limit = user.get("monthly_scan_limit", 0)
    overage_scans = max(0, current_period_scans - monthly_limit)

    return {
        "plan": plan,
        "scansUsedThisPeriod": current_period_scans,
        "scansIncluded": monthly_limit,
        "trialRemaining": None,
        "overageScans": overage_scans,
        "isOverage": is_overage,
    }


async def get_api_key_info(key: str) -> Optional[dict]:
    """
    Get detailed API key information without using a scan
    """
    result = await validate_api_key(key)
    if not result:
        return None

    api_key_record, user_record = result
    plan = user_record.get("plan", "trial")

    usage_info = calculate_usage_info(user_record)

    return {
        "valid": True,
        "plan": plan,
        "scansIncluded": usage_info["scansIncluded"],
        "scansUsedThisPeriod": usage_info["scansUsedThisPeriod"],
        "trialRemaining": usage_info["trialRemaining"],
        "overageScans": usage_info["overageScans"],
        "overageRateCents": user_record.get("overage_rate_cents"),
        "rateLimitPerMinute": get_rate_limit(plan),
    }
