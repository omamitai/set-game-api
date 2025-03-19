import os
import io
import uuid
import base64
import uvicorn
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import cv2
import numpy as np

# Import your SET detector code (assuming it's in a file named set_detector.py)
# We're importing only what we need and will adapt it slightly
from set_detector import Config, SetGameDetector, ImageProcessingError

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# Initialize FastAPI app
app = FastAPI(
    title="SET Game Detector API",
    description="API for detecting SET card games from images",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define response models
class SetFound(BaseModel):
    positions: List[int]
    cards: List[Dict[str, str]]

class DetectionResponse(BaseModel):
    success: bool
    message: str
    sets_found: Optional[List[SetFound]] = None
    annotated_image: Optional[str] = None  # Base64 encoded image
    error: Optional[str] = None

# Initialize SET detector with API keys from environment variables
def get_detector():
    config = Config()
    # Override config with environment variables (more secure than hardcoding)
    config.ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", config.ROBOFLOW_API_KEY)
    config.CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", config.CLAUDE_API_KEY)
    return SetGameDetector(config)

@app.get("/")
async def root():
    return {"message": "SET Game Detector API is running! Upload an image to /detect to analyze a SET game."}

@app.post("/detect", response_model=DetectionResponse)
async def detect_sets(file: UploadFile = File(...)):
    try:
        # Validate file
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file is not an image")
            
        # Read the image file
        contents = await file.read()
        image_array = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Generate a unique filename
        filename = f"{uuid.uuid4()}.jpg"
        temp_path = os.path.join("/tmp", filename)
        
        # Save the image temporarily
        cv2.imwrite(temp_path, image)
        
        # Process the image with the SET detector
        detector = get_detector()
        annotated_image, sets_found, card_features = detector.process_image(temp_path)
        
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        if annotated_image is None:
            return DetectionResponse(
                success=False,
                message="Failed to process image",
                error="No cards detected or processing error"
            )
        
        # Prepare the response
        # Convert annotated image to base64
        _, img_encoded = cv2.imencode('.jpg', annotated_image)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        
        # Format sets with card details
        formatted_sets = []
        for set_positions in sets_found:
            cards_in_set = []
            for pos in set_positions:
                pos_str = str(pos)
                if pos_str in card_features:
                    cards_in_set.append(card_features[pos_str])
            
            formatted_sets.append(SetFound(
                positions=set_positions,
                cards=cards_in_set
            ))
        
        return DetectionResponse(
            success=True,
            message=f"Successfully processed image. Found {len(sets_found)} sets.",
            sets_found=formatted_sets,
            annotated_image=img_base64
        )
        
    except HTTPException as e:
        raise e
    except ImageProcessingError as e:
        logging.error(f"Image processing error: {str(e)}")
        return DetectionResponse(
            success=False,
            message="Image processing error",
            error=str(e)
        )
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)
        return DetectionResponse(
            success=False,
            message="An unexpected error occurred",
            error=str(e)
        )

# Run the server if executed directly
if __name__ == "__main__":
    # Get port from environment variable for deployment platforms
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
