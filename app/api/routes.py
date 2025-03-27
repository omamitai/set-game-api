"""
API routes for the SET Game Detector.
"""
import logging
from fastapi import APIRouter, File, UploadFile, HTTPException, status, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

# Import the main detector class and specific exceptions
from app.set_detector import (
    SetGameDetector,
    CardDetectionError,
    CardClassificationError,
    GeminiAPIError,
    ImageProcessingError
)
from app.config import get_settings, Settings # Import Settings for dependency

logger = logging.getLogger(__name__)

router = APIRouter()

# --- Pydantic Models for API Response ---
# Matches the structure returned by SetGameDetector.process_image and expected by frontend

class FeatureDetail(BaseModel):
    number: Optional[str] = None
    color: Optional[str] = None
    shape: Optional[str] = None
    shading: Optional[str] = None

class CardDetail(BaseModel):
    position: int
    features: Optional[FeatureDetail] = None # Features might be None if classification failed

class SetDetail(BaseModel):
    set_id: int
    cards: List[CardDetail]

class DetectionResponse(BaseModel):
    image: Optional[str] = None # Base64 encoded annotated image, or original on error
    sets_found: List[SetDetail] = Field(default_factory=list)
    all_cards: List[CardDetail] = Field(default_factory=list)
    error: Optional[str] = None # Error message if processing failed


# --- Dependency Injection ---
# Instantiate the detector once when the application starts
# This avoids reloading models/config on every request.
set_game_detector_instance = SetGameDetector()

def get_detector() -> SetGameDetector:
    return set_game_detector_instance

def get_max_upload_size(settings: Settings = Depends(get_settings)) -> int:
    return settings.MAX_UPLOAD_SIZE

# --- API Endpoint ---

@router.post("/detect", response_model=DetectionResponse, status_code=status.HTTP_200_OK)
async def detect_sets_in_image(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, etc.) containing SET cards."),
    detector: SetGameDetector = Depends(get_detector),
    max_upload_size: int = Depends(get_max_upload_size)
):
    """
    Processes an uploaded image to detect SET cards, classify their features,
    find valid SETs, and return an annotated image along with the results.
    """
    # --- File Validation ---
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type: {file.content_type}. Please upload an image (JPEG, PNG, etc.).",
        )

    # Read file contents
    contents = await file.read()

    # Check file size (optional but good practice)
    if len(contents) > max_upload_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds the limit of {max_upload_size / (1024*1024):.1f} MB.",
        )
    await file.close() # Close the file handle

    logger.info(f"Received image file: {file.filename}, size: {len(contents)} bytes, type: {file.content_type}")

    # --- Process Image ---
    try:
        # Call the detector's main processing method
        result_dict = detector.process_image(contents)

        # If the detector returned an error in the dictionary, return 200 OK but include the error message
        if result_dict.get("error"):
             logger.warning(f"Processing completed with non-critical error: {result_dict['error']}")
             # We still return 200 OK, the frontend should check the 'error' field
             return DetectionResponse(**result_dict)

        # Return the successful result
        return DetectionResponse(**result_dict)

    # --- Handle Specific Expected Exceptions ---
    except ImageProcessingError as e:
        logger.error(f"Image processing error for file {file.filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to process image: {e}. Ensure it's a valid image file.",
        )
    except CardDetectionError as e:
         logger.error(f"Card detection error for file {file.filename}: {e}", exc_info=True)
         # Use 503 Service Unavailable if it's likely an external service (Roboflow) issue
         raise HTTPException(
             status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
             detail=f"Failed during card detection: {e}",
         )
    except (CardClassificationError, GeminiAPIError) as e:
        logger.error(f"Card classification error for file {file.filename}: {e}", exc_info=True)
        # Use 503 Service Unavailable as it relies on the external Gemini API
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed during card classification: {e}",
        )
    # --- Handle Generic Exceptions (Should be caught by main.py handler, but as fallback) ---
    except Exception as e:
        logger.critical(f"Unexpected error processing file {file.filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected internal server error occurred: {str(e)}",
        )
