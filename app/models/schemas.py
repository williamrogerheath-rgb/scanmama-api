"""
Pydantic models for request/response validation
"""
from typing import Optional, Literal
from pydantic import BaseModel, Field


class ScanOptions(BaseModel):
    """Options for document scanning"""
    outputFormat: Literal["pdf", "png", "jpeg"] = Field(default="pdf")
    quality: Literal["draft", "standard", "high"] = Field(default="standard")
    colorMode: Literal["color", "grayscale", "bw"] = Field(default="color")


class ScanRequest(BaseModel):
    """Request body for POST /v1/scan"""
    image: str = Field(..., description="Base64 encoded image data")
    options: Optional[ScanOptions] = Field(default_factory=ScanOptions)


class MultiScanRequest(BaseModel):
    """Request body for POST /v1/scan/multi"""
    images: list[str] = Field(..., description="Array of base64 encoded image data", min_length=1, max_length=50)
    options: Optional[ScanOptions] = Field(default_factory=ScanOptions)


class UsageInfo(BaseModel):
    """Usage information for a user"""
    plan: str
    scansUsedThisPeriod: int
    scansIncluded: Optional[int] = None
    trialRemaining: Optional[int] = None
    overageScans: int
    isOverage: bool


class ScanResult(BaseModel):
    """Response body for successful scan"""
    scanId: str
    status: str
    pdfUrl: Optional[str] = None
    imageUrl: Optional[str] = None
    originalUrl: Optional[str] = None
    width: int
    height: int
    processingTimeMs: int
    documentDetected: bool
    usage: UsageInfo


class MultiScanResult(BaseModel):
    """Response body for successful multi-page scan"""
    scanId: str
    status: str
    pdfUrl: str
    pageCount: int
    processingTimeMs: int
    usage: UsageInfo


class ApiKeyInfo(BaseModel):
    """API key validation response"""
    valid: bool
    plan: str
    scansIncluded: Optional[int] = None
    scansUsedThisPeriod: int
    trialRemaining: Optional[int] = None
    overageScans: int
    overageRateCents: Optional[int] = None
    rateLimitPerMinute: int


class ErrorDetail(BaseModel):
    """Error detail object"""
    code: str
    message: str
    upgradeUrl: Optional[str] = None
    details: Optional[dict] = None


class ErrorResponse(BaseModel):
    """Error response body"""
    error: ErrorDetail


class UsageResponse(BaseModel):
    """Usage statistics response"""
    usage: UsageInfo


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str


class RootResponse(BaseModel):
    """Root endpoint response"""
    name: str
    docs: str
