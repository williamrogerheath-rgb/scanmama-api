"""
File storage services using Supabase Storage
"""
import asyncio
from typing import Optional
from supabase import create_client

from app.config import Config


def get_supabase_client():
    """Get Supabase client instance"""
    return create_client(Config.SUPABASE_URL, Config.SUPABASE_SERVICE_KEY)


async def upload_file(
    bucket: str,
    path: str,
    data: bytes,
    content_type: str = "application/octet-stream"
) -> str:
    """
    Upload file to Supabase Storage
    Returns: file path
    """
    supabase = get_supabase_client()

    try:
        response = supabase.storage.from_(bucket).upload(
            path=path,
            file=data,
            file_options={"content-type": content_type}
        )

        return path

    except Exception as e:
        print(f"Error uploading file to {bucket}/{path}: {e}")
        raise


async def get_signed_url(bucket: str, path: str, expires_in: int = 3600) -> Optional[str]:
    """
    Generate signed URL for file access
    Returns: signed URL or None if error
    """
    supabase = get_supabase_client()

    try:
        response = supabase.storage.from_(bucket).create_signed_url(
            path=path,
            expires_in=expires_in
        )

        if isinstance(response, dict) and "signedURL" in response:
            return response["signedURL"]

        return None

    except Exception as e:
        print(f"Error generating signed URL for {bucket}/{path}: {e}")
        return None


async def delete_file(bucket: str, path: str) -> bool:
    """
    Delete file from Supabase Storage
    Returns: True if successful, False otherwise
    """
    supabase = get_supabase_client()

    try:
        supabase.storage.from_(bucket).remove([path])
        return True

    except Exception as e:
        print(f"Error deleting file {bucket}/{path}: {e}")
        return False


async def upload_scan_files(
    scan_id: str,
    original_bytes: bytes,
    processed_bytes: bytes,
    pdf_bytes: bytes
) -> dict:
    """
    Upload all scan-related files and generate signed URLs
    Returns: dict with paths and URLs
    """
    bucket = Config.STORAGE_BUCKET

    # Define file paths (using JPEG for images)
    original_path = f"{scan_id}/original.jpg"
    processed_path = f"{scan_id}/processed.jpg"
    pdf_path = f"{scan_id}/document.pdf"

    # Upload files in parallel using asyncio.gather()
    await asyncio.gather(
        upload_file(bucket, original_path, original_bytes, "image/jpeg"),
        upload_file(bucket, processed_path, processed_bytes, "image/jpeg"),
        upload_file(bucket, pdf_path, pdf_bytes, "application/pdf")
    )

    # Generate signed URLs in parallel
    original_url, processed_url, pdf_url = await asyncio.gather(
        get_signed_url(bucket, original_path, Config.SIGNED_URL_EXPIRY),
        get_signed_url(bucket, processed_path, Config.SIGNED_URL_EXPIRY),
        get_signed_url(bucket, pdf_path, Config.SIGNED_URL_EXPIRY)
    )

    return {
        "original_path": original_path,
        "processed_path": processed_path,
        "pdf_path": pdf_path,
        "original_url": original_url,
        "processed_url": processed_url,
        "pdf_url": pdf_url,
    }
