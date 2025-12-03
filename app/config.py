"""
Configuration management for ScanMama API
Loads environment variables and validates required settings
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration"""

    # Supabase settings (required)
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

    # API settings
    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

    # Storage settings
    STORAGE_BUCKET = "scans"
    SIGNED_URL_EXPIRY = 3600  # 1 hour

    # Rate limiting (requests per minute)
    RATE_LIMITS = {
        "trial": 10,
        "pro": 100,
        "team": 500,
    }

    # File size limits
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

    @classmethod
    def validate(cls):
        """Validate required configuration"""
        if not cls.SUPABASE_URL:
            raise ValueError("SUPABASE_URL environment variable is required")
        if not cls.SUPABASE_SERVICE_KEY:
            raise ValueError("SUPABASE_SERVICE_KEY environment variable is required")


# Validate configuration on import
Config.validate()
