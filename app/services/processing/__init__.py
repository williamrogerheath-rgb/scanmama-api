"""ScanMama Document Processing Pipeline v2"""
from .pipeline import process_document, decode_base64_image
from .detect_ml import detect
from .detect import DetectionResult

__all__ = ['process_document', 'decode_base64_image', 'detect', 'DetectionResult']
