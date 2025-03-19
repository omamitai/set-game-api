"""
API routes for the SET Game Detector API.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks, status
from fastapi.responses import JSONResponse
from typing import Dict, Any, List
import logging
import time
import json
import uuid
import os
from pydantic import BaseModel

from app.set_detector import SetGameDetector, CardDetectionError, CardClassificationError, ImageProcessingError
from app.config import get_settings, Settings

# Initialize router
router = APIRouter()

# Define response models
class SetCardFeatures(BaseModel):
    number: str
    color: str
    shape: str
    shading: str

class SetCard(BaseModel):
    position: int
    features: SetCardFeatures

class SetFound(BaseModel):
    set_id: int
    cards: List[SetCard]

class DetectionResponse(BaseModel):
    image: str
    sets_found: List[SetFound]
    all_cards: List[SetCard]
    
class HealthResponse(BaseModel):
    status: str
    version: str

# Initialize the SetGameDetector
detector = SetGameDetector()

@router.get("/health", response_model=HealthResponse)
async def health_check(settings: Settings = Depends(get_settings)):
    """
    Health check endpoint to verify API is operational.
    """
    return {
        "status": "ok",
        "version": settings.APP_VERSION
    }

@router.post("/detect", response_model=DetectionResponse, status_code=status.HTTP_200_OK)
async def detect_sets(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings)
):
    """
    Detect SET cards and find valid SETs in the uploaded image.
    
    Args:
        file: The image file to process
        
    Returns:
        DetectionResponse: Results of SET detection including annotated image
    """
    # Check file size
    if file.size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size allowed is {settings.MAX_UPLOAD_SIZE / (1024 * 1024)} MB."
        )
    
    # Check content type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only image files are supported."
        )
    
    try:
        start_time = time.time()
        
        # Read the uploaded image
        contents = await file.read()
        
        # Process the image
        results = detector.process_image(contents)
        
        # Log processing time
        processing_time = time.time() - start_time
        logging.info(f"Image processed in {processing_time:.2f} seconds")
        
        return results
    
    except CardDetectionError as e:
        logging.error(f"Card detection error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not detect cards in the image: {str(e)}"
        )
    
    except CardClassificationError as e:
        logging.error(f"Card classification error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to classify cards: {str(e)}"
        )
    
    except ImageProcessingError as e:
        logging.error(f"Image processing error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image processing failed: {str(e)}"
        )
    
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )
