"""
ScanMama API - Main application entry point
Cloud document scanning API for web developers
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import scan, validate
from app.models.schemas import HealthResponse, RootResponse

# Create FastAPI app
app = FastAPI(
    title="ScanMama API",
    description="Cloud document scanning API for web apps. No WASM, no build errors.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware - allow all origins for now
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(scan.router)
app.include_router(validate.router)


@app.get("/", response_model=RootResponse)
async def root():
    """Root endpoint - API information"""
    return RootResponse(
        name="ScanMama API",
        docs="/docs"
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="ok",
        version="0.1.0"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
