#!/usr/bin/env python3
"""
SET Game Detector - Production Backend API
For Render.com deployment
"""

# Imports
import os
import requests
import json
import cv2
import numpy as np
import base64
from itertools import combinations
import io
import traceback
import time
import sys
from collections import defaultdict
from inference_sdk import InferenceHTTPClient
import re  # For JSON parsing from Claude
import logging  # For logging
import uuid
from typing import List, Dict, Any, Optional, Tuple
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn
from pydantic import BaseModel
import tempfile
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# --- Custom Middleware ---
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        logging.info(f"[{request_id}] {request.method} {request.url.path}")
        start_time = time.time()
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            logging.info(f"[{request_id}] Completed: {response.status_code} ({process_time:.3f}s)")
            return response
        except Exception as e:
            process_time = time.time() - start_time
            logging.error(f"[{request_id}] Error: {str(e)} ({process_time:.3f}s)", exc_info=True)
            raise

# --- FastAPI App Setup ---
app = FastAPI(
    title="SET Game Detector API",
    description="API for detecting SET card games from images",
    version="1.0.0"
)

# Add middlewares
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],  # Expose all headers to the browser
    max_age=86400,  # Cache preflight requests for 24 hours
)
app.add_middleware(GZipMiddleware, minimum_size=1000)  # Compress responses
app.add_middleware(RequestLoggingMiddleware)  # Log all requests

# --- Configuration ---
class Config:
    """Configuration class to centralize parameters and credentials."""
    # API Keys and URLs
    ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
    CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "")
    ROBOFLOW_API_URL = "https://detect.roboflow.com"
    CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
    CLAUDE_MODEL = "claude-3-7-sonnet-20250219"  # Claude model to use

    # SET Game parameters
    FEATURE_TYPES = {
        'number': ['1', '2', '3'],
        'color': ['red', 'green', 'purple'],
        'shape': ['oval', 'diamond', 'squiggle'],
        'shading': ['solid', 'striped', 'outline']
    }

    # Visualization parameters
    COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
              (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
              (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)]
    BOX_THICKNESS = 3
    FONT_SCALE = 0.9
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    OFFSET_STEP = 4

    # Output directories and paths
    RESULTS_DIR = os.environ.get("RESULTS_DIR", "/tmp/set_results")
    
    # Timeouts
    ROBOFLOW_TIMEOUT = 30  # seconds
    CLAUDE_TIMEOUT = 30  # seconds
    PROCESS_TIMEOUT = 60  # seconds


# --- Custom Exceptions ---
class CardDetectionError(Exception):
    """Exception raised when card detection fails."""
    pass

class CardClassificationError(Exception):
    """Exception raised when card classification fails."""
    pass

class ImageProcessingError(Exception):
    """Exception raised during image processing."""
    pass

class ClaudeAPIError(Exception):
    """Exception raised when Claude API call fails."""
    pass


# --- Image Processing ---
class ImageProcessor:
    """Handles image loading and saving."""
    def __init__(self, config: Config):
        self.config = config

    def load_image_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Loads an image from bytes."""
        try:
            image_np = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            if image is None:
                raise ImageProcessingError("Failed to decode image bytes. Ensure it's a valid image file.")
            logging.debug("Image loaded from bytes successfully.")
            return image
        except Exception as e:
            logging.error(f"Error loading image from bytes: {e}", exc_info=True)
            raise ImageProcessingError(f"Failed to load image: {str(e)}")

    def save_image_to_path(self, image: np.ndarray, path: str) -> bool:
        """Saves an image to the specified path."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            return cv2.imwrite(path, image)
        except Exception as e:
            logging.error(f"Error saving image to {path}: {e}", exc_info=True)
            return False

    def encode_image_to_base64(self, image: np.ndarray) -> str:
        """Converts an image to base64 string."""
        try:
            _, img_encoded = cv2.imencode('.jpg', image)
            return base64.b64encode(img_encoded).decode('utf-8')
        except Exception as e:
            logging.error(f"Error encoding image to base64: {e}", exc_info=True)
            raise ImageProcessingError(f"Failed to encode image to base64: {str(e)}")


# --- Card Detector ---
class CardDetector:
    """Detects cards in an image using Roboflow."""
    def __init__(self, config: Config, image_processor: ImageProcessor):
        self.config = config
        self.image_processor = image_processor
        if not self.config.ROBOFLOW_API_KEY:
            logging.warning("Roboflow API key not set!")
        self.roboflow_client = InferenceHTTPClient(
            api_url=self.config.ROBOFLOW_API_URL,
            api_key=self.config.ROBOFLOW_API_KEY
        )

    def detect_cards(self, image: np.ndarray) -> tuple[list, dict]:
        """Detects cards in the image and assigns positions."""
        try:
            # Save image to a temporary file
            temp_filename = f"{uuid.uuid4()}.jpg"
            temp_path = os.path.join("/tmp", temp_filename)
            
            if not self.image_processor.save_image_to_path(image, temp_path):
                raise ImageProcessingError("Failed to save temporary image for card detection.")
            
            # Log that we're about to call Roboflow
            logging.info(f"Calling Roboflow API to detect cards in {temp_path}")
            
            # Set a timeout for the Roboflow API call
            start_time = time.time()
            result = self.roboflow_client.run_workflow(
                workspace_name="tel-aviv",
                workflow_id="custom-workflow",
                images={"image": temp_path},
                use_cache=True
            )
            logging.info(f"Roboflow API call completed in {time.time() - start_time:.2f} seconds")
            
            # Clean up temporary file
            try:
                os.remove(temp_path)
            except Exception as e:
                logging.warning(f"Failed to remove temporary file {temp_path}: {e}")
            
            predictions = result[0]["predictions"]["predictions"]
            image_dimensions = result[0]["predictions"]["image"]
            card_predictions = [pred for pred in predictions if pred["class"] == "card"]

            if not card_predictions:
                raise CardDetectionError("No cards detected in the image by Roboflow.")

            cards_with_positions = self._assign_card_positions(card_predictions, image_dimensions)
            logging.info(f"Detected {len(cards_with_positions)} cards using Roboflow.")
            return cards_with_positions, image_dimensions

        except CardDetectionError as e:
            logging.error(f"Card detection failed: {e}", exc_info=True)
            raise
        except Exception as e:
            logging.error(f"Unexpected error during card detection: {e}", exc_info=True)
            raise CardDetectionError(f"Card detection process failed due to an unexpected error: {e}")

    def _assign_card_positions(self, card_predictions: list, image_dimensions: dict) -> list:
        """
        Assigns position numbers to detected cards based on their location,
        with fewer assumptions about card layout.
        """
        cards = [{
            "x": pred["x"], "y": pred["y"], 
            "width": pred["width"], "height": pred["height"],
            "detection_id": pred["detection_id"]
        } for pred in card_predictions]
        
        # Calculate average card height to use as a threshold for vertical grouping
        avg_card_height = sum(card["height"] for card in cards) / len(cards)
        vertical_threshold = avg_card_height * 0.5
        
        # Group cards by approximate rows (cards with similar y-coordinates)
        rows = []
        cards_sorted_by_y = sorted(cards, key=lambda c: c["y"])
        
        current_row = [cards_sorted_by_y[0]]
        for card in cards_sorted_by_y[1:]:
            # If this card is close enough vertically to the first card in current row, add it
            if abs(card["y"] - current_row[0]["y"]) <= vertical_threshold:
                current_row.append(card)
            else:
                # Otherwise, start a new row
                rows.append(current_row)
                current_row = [card]
        
        # Add the last row if not empty
        if current_row:
            rows.append(current_row)
        
        # Sort each row by x-coordinate
        for row in rows:
            row.sort(key=lambda c: c["x"])
        
        # Assign sequential positions
        cards_with_positions = []
        position = 1
        for row in rows:
            for card in row:
                cards_with_positions.append({
                    "position": position,
                    "box": {
                        "x1": int(card["x"] - card["width"] / 2), 
                        "y1": int(card["y"] - card["height"] / 2),
                        "x2": int(card["x"] + card["width"] / 2), 
                        "y2": int(card["y"] + card["height"] / 2)
                    },
                    "center": {"x": int(card["x"]), "y": int(card["y"])},
                    "detection_id": card["detection_id"]
                })
                position += 1
        
        return cards_with_positions


# --- Card Classifier ---
class CardClassifier:
    """Classifies card features using Claude API."""
    def __init__(self, config: Config, image_processor: ImageProcessor):
        self.config = config
        self.image_processor = image_processor

    def classify_cards(self, image: np.ndarray, cards_with_positions: list) -> dict:
        """Classifies cards using Claude API."""
        if not self.config.CLAUDE_API_KEY:
            raise ValueError("Claude API key is required for card classification. Please set CLAUDE_API_KEY in environment variables.")
        return self._classify_cards_with_claude(image, cards_with_positions)

    def _prepare_labeled_image_for_claude(self, image: np.ndarray, cards_with_positions: list) -> np.ndarray:
        """Creates a labeled version of the image with position numbers for Claude."""
        labeled_image = image.copy()
        for card in cards_with_positions:
            position = card["position"]
            center_x, center_y = card["center"]["x"], card["center"]["y"]
            
            # Draw a circle with a position number for clear identification
            cv2.circle(labeled_image, (center_x, center_y), 20, (0, 0, 0), -1)
            cv2.putText(labeled_image, str(position), (center_x - 7, center_y + 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        return labeled_image

    def _classify_cards_with_claude(self, image: np.ndarray, cards_with_positions: list) -> dict:
        """Classifies cards using Claude API with improved position labeling."""
        try:
            # Create labeled image with position numbers for Claude
            labeled_image = self._prepare_labeled_image_for_claude(image, cards_with_positions)
            img_base64 = self.image_processor.encode_image_to_base64(labeled_image)
            
            headers = {
                "x-api-key": self.config.CLAUDE_API_KEY,
                "content-type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            # Create detailed position information for the prompt
            position_details = []
            for card in cards_with_positions:
                pos = card["position"]
                center = card["center"]
                position_details.append(f"Card {pos}: at coordinates ({center['x']}, {center['y']})")
            
            position_info = "\n".join(position_details)
            
            # Enhanced prompt with explicit position information
            prompt = (
                f"Analyze the SET card game image. Each card has a NUMBER label on it (white numbers in black circles).\n"
                f"There are {len(cards_with_positions)} cards visible with the following positions:\n"
                f"{position_info}\n\n"
                f"For each numbered card, identify these 4 features:\n"
                f"1. Number (1, 2, or 3 shapes)\n"
                f"2. Color (Red, Green, or Purple)\n"
                f"3. Shape (Oval, Diamond, or Squiggle)\n"
                f"4. Shading (Solid, Striped, or Outline)\n\n"
                f"Return a JSON object with card positions as keys:\n"
                f"{{'1': {{'number': '1|2|3', 'color': 'red|green|purple', 'shape': 'oval|diamond|squiggle', 'shading': 'solid|striped|outline'}}, ...}}\n"
                f"Return only JSON, no explanations. Make sure to use the white numbers in black circles as position references."
            )
            
            data = {
                "model": self.config.CLAUDE_MODEL,
                "max_tokens": 1000,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_base64}},
                        {"type": "text", "text": prompt}
                    ]
                }]
            }

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logging.info(f"Calling Claude API (attempt {attempt + 1}/{max_retries})")
                    start_time = time.time()
                    response = requests.post(
                        self.config.CLAUDE_API_URL, 
                        headers=headers, 
                        json=data, 
                        timeout=self.config.CLAUDE_TIMEOUT
                    )
                    logging.info(f"Claude API call completed in {time.time() - start_time:.2f} seconds")
                    response.raise_for_status()
                    break
                except requests.exceptions.Timeout:
                    if attempt == max_retries - 1:
                        raise ClaudeAPIError(f"Claude API request timed out after {self.config.CLAUDE_TIMEOUT} seconds")
                    wait_time = min(2 ** attempt, 10)
                    logging.warning(f"Claude API timeout (attempt {attempt + 1}/{max_retries}). Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                except requests.exceptions.RequestException as e:
                    if attempt == max_retries - 1:
                        raise ClaudeAPIError(f"Claude API request failed after {max_retries} retries: {e}")
                    wait_time = min(2 ** attempt, 10)
                    logging.warning(f"Request error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)

            response_json = response.json()
            content = response_json["content"][0]["text"]

            # Try to extract JSON from potential code blocks
            json_match = re.search(r'```json\n([\s\S]*?)\n```', content)
            classification_json = json_match.group(1) if json_match else content

            try:
                classification_results = json.loads(classification_json)
                logging.info("Card classification successful using Claude API.")
                return classification_results
            except json.JSONDecodeError:
                logging.error("Failed to parse Claude's JSON response. Raw response:\n%s", content)
                raise CardClassificationError("Failed to parse Claude's response as JSON. Please check Claude API response format.")

        except requests.exceptions.HTTPError as e:
            logging.error(f"Claude API HTTP error: {e}. Response: {getattr(e.response, 'text', 'No response text')}", exc_info=True)
            raise ClaudeAPIError(f"Claude API HTTP error: {e}. Status Code: {getattr(e.response, 'status_code', 'Unknown')}")
        except ClaudeAPIError as e:
            logging.error(f"Claude API error after retries: {e}", exc_info=True)
            raise
        except Exception as e:
            logging.error(f"Unexpected error during card classification with Claude: {e}", exc_info=True)
            raise CardClassificationError(f"Card classification with Claude failed due to an unexpected error: {e}")


# --- SET Game Logic ---
class SetGameLogic:
    """Implements the logic to find SETs in a set of cards."""
    def __init__(self, config: Config):
        self.config = config

    def is_set(self, cards: list[dict]) -> bool:
        """Checks if a group of three cards forms a valid SET."""
        for feature in self.config.FEATURE_TYPES.keys():
            values = [card[feature] for card in cards]
            if len(set(values)) == 2:
                return False
        return True

    def find_sets(self, card_features: dict) -> list[list[int]]:
        """Finds all valid SETs from the given card features."""
        sets_found = []
        positions = list(card_features.keys())
        for combo in combinations(positions, 3):
            try:
                cards = [card_features[pos] for pos in combo]
                if self.is_set(cards):
                    sets_found.append(list(map(int, combo)))
            except KeyError:
                logging.warning(f"Card position missing in features: {combo}. Skipping combination.")
                continue
        logging.info(f"Found {len(sets_found)} sets.")
        return sets_found


# --- Visualizer ---
class Visualizer:
    """Handles visualization of detected sets on the image."""
    def __init__(self, config: Config, image_processor: ImageProcessor):
        self.config = config
        self.image_processor = image_processor

    def draw_sets_on_image(self, image: np.ndarray, sets_found: list[list[int]], cards_with_positions: list) -> np.ndarray:
        """Draws bounding boxes, labels, and position numbers for detected sets on the image."""
        annotated_image = image.copy()
        
        # First, draw position numbers on all cards for clarity and reference
        for card in cards_with_positions:
            position = card["position"]
            center_x, center_y = card["center"]["x"], card["center"]["y"]
            
            # Draw a circle with the position number
            cv2.circle(annotated_image, (center_x, center_y), 15, (255, 255, 255), -1)
            cv2.circle(annotated_image, (center_x, center_y), 15, (0, 0, 0), 1)
            cv2.putText(annotated_image, str(position), (center_x - 5, center_y + 5),
                      self.config.FONT, 0.5, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        
        if not sets_found:
            logging.info("No sets found to draw.")
            return annotated_image

        card_position_map = {card["position"]: card for card in cards_with_positions}
        card_set_membership = defaultdict(list)
        for set_idx, set_positions in enumerate(sets_found):
            for pos in set_positions:
                card_set_membership[pos].append(set_idx)

        for pos, set_indices in card_set_membership.items():
            if pos not in card_position_map:
                logging.warning(f"Card position {pos} not found in card position map during visualization.")
                continue
            card = card_position_map[pos]
            box = card["box"]

            for i, set_idx in enumerate(set_indices):
                color = self.config.COLORS[set_idx % len(self.config.COLORS)]
                offset = i * self.config.OFFSET_STEP
                x1, y1 = box["x1"] - offset, box["y1"] - offset
                x2, y2 = box["x2"] + offset, box["y2"] + offset
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, self.config.BOX_THICKNESS)
                if pos == sets_found[set_idx][0]:  # Label only once per set
                    cv2.putText(annotated_image, f"Set {set_idx + 1}", (x1, y1 - 10),
                              self.config.FONT, self.config.FONT_SCALE, color, self.config.BOX_THICKNESS, lineType=cv2.LINE_AA)
        
        logging.info("Sets and position numbers drawn on image.")
        return annotated_image


# --- Main Detector Class ---
class SetGameDetector:
    """Orchestrates the SET game detection process."""
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.image_processor = ImageProcessor(self.config)
        self.card_detector = CardDetector(self.config, self.image_processor)
        self.card_classifier = CardClassifier(self.config, self.image_processor)
        self.set_logic = SetGameLogic(self.config)
        self.visualizer = Visualizer(self.config, self.image_processor)

    def process_image(self, image_bytes: bytes) -> tuple[np.ndarray, list[list[int]], dict]:
        """Processes an image to detect SETs."""
        request_id = str(uuid.uuid4())[:8]
        logging.info(f"[{request_id}] Starting image processing")
        try:
            image = self.image_processor.load_image_from_bytes(image_bytes)

            logging.info(f"[{request_id}] Detecting cards...")
            cards_with_positions, image_dimensions = self.card_detector.detect_cards(image)
            logging.info(f"[{request_id}] Detected {len(cards_with_positions)} cards.")

            logging.info(f"[{request_id}] Classifying cards...")
            card_features = self.card_classifier.classify_cards(image, cards_with_positions)

            logging.info(f"[{request_id}] Finding sets...")
            sets_found = self.set_logic.find_sets(card_features)

            logging.info(f"[{request_id}] Drawing sets on image...")
            annotated_image = self.visualizer.draw_sets_on_image(image, sets_found, cards_with_positions)

            logging.info(f"[{request_id}] Image processing completed. Found {len(sets_found)} sets.")
            return annotated_image, sets_found, card_features

        except Exception as e:
            logging.error(f"[{request_id}] Error during image processing: {e}", exc_info=True)
            raise


# --- FastAPI Routes ---
class ProcessResponse(BaseModel):
    """Model for the response of the process endpoint."""
    image_base64: str
    sets_found: List[List[int]]
    card_features: Dict[str, Dict[str, str]]


@app.on_event("startup")
async def startup_event():
    """Runs on application startup to ensure everything is properly initialized."""
    try:
        logging.info("Starting SET Detector API...")
        
        # Verify environment variables and create necessary directories
        config = Config()
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        
        # Check API key availability without exposing the actual keys
        if not config.ROBOFLOW_API_KEY:
            logging.warning("⚠️ Roboflow API key not configured. Card detection will fail.")
        else:
            logging.info("✓ Roboflow API key configured.")
            
        if not config.CLAUDE_API_KEY:
            logging.warning("⚠️ Claude API key not configured. Card classification will fail.")
        else:
            logging.info("✓ Claude API key configured.")
        
        # Attempt to initialize OpenCV to ensure it's working
        try:
            blank_image = np.zeros((10, 10, 3), np.uint8)
            encoded = cv2.imencode('.jpg', blank_image)[1]
            logging.info("✓ OpenCV initialized successfully.")
        except Exception as cv_error:
            logging.error(f"OpenCV initialization failed: {cv_error}")
            
        logging.info("SET Detector API startup complete.")
    except Exception as e:
        logging.error(f"Error during startup: {e}", exc_info=True)
        # We don't want to prevent the app from starting, so just log the error


@app.get("/api/health")
async def health_check():
    """Enhanced health check endpoint."""
    try:
        config = Config()
        return {
            "status": "healthy",
            "version": "1.0.0",
            "timestamp": time.time(),
            "environment": os.environ.get("ENVIRONMENT", "production"),
            "roboflow_key_configured": bool(config.ROBOFLOW_API_KEY),
            "claude_key_configured": bool(config.CLAUDE_API_KEY)
        }
    except Exception as e:
        logging.error(f"Health check failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.options("/api/health")
async def health_preflight():
    """Preflight response for health check."""
    return PlainTextResponse("")


@app.options("/api/process")
async def process_preflight():
    """Preflight response for process endpoint."""
    return PlainTextResponse("")


@app.get("/api/test-cors")
async def test_cors():
    """Test endpoint to verify CORS is working."""
    return {"cors_test": "success"}


@app.options("/api/test-cors")
async def test_cors_preflight():
    """Preflight response for CORS test."""
    return PlainTextResponse("")


@app.get("/api/ping")
async def ping():
    """Simple ping endpoint for quick availability checks."""
    return {"ping": "pong"}


@app.get("/api/debug-info")
async def debug_info():
    """Return diagnostic information about the API environment."""
    try:
        import platform
        import socket
        
        # Get basic system info without exposing sensitive details
        system_info = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "hostname": socket.gethostname()
        }
        
        # Get environment variables (excluding sensitive values)
        env_vars = {}
        for key in os.environ:
            if "KEY" not in key.upper() and "SECRET" not in key.upper() and "PASSWORD" not in key.upper():
                env_vars[key] = os.environ[key] if len(os.environ[key]) < 50 else f"{os.environ[key][:25]}...truncated"
        
        return {
            "status": "ok",
            "system_info": system_info,
            "environment": env_vars,
        }
    except Exception as e:
        logging.error(f"Debug info retrieval failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Failed to retrieve debug info: {str(e)}"}
        )


@app.post("/api/process", response_model=ProcessResponse)
async def process_image(file: UploadFile = File(...)):
    """Process an image of a SET game and return the results."""
    request_id = str(uuid.uuid4())
    logging.info(f"[{request_id}] Processing request with file: {file.filename}, size: {file.size if hasattr(file, 'size') else 'unknown'}")
    
    # Validate file type
    content_type = file.content_type or ""
    if not content_type.startswith("image/"):
        logging.warning(f"[{request_id}] Invalid file type: {content_type}")
        raise HTTPException(status_code=400, detail="File must be an image (JPEG, PNG, etc.)")
    
    try:
        # Read image bytes
        try:
            image_bytes = await file.read()
            if not image_bytes:
                raise ValueError("Empty file uploaded")
            logging.info(f"[{request_id}] Successfully read {len(image_bytes)} bytes from uploaded file")
        except Exception as read_error:
            logging.error(f"[{request_id}] Error reading uploaded file: {read_error}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {str(read_error)}")
        
        # Check API keys before processing
        config = Config()
        if not config.ROBOFLOW_API_KEY:
            logging.error(f"[{request_id}] Roboflow API key not configured")
            raise HTTPException(status_code=503, 
                               detail="Roboflow API key is not configured. Please set the ROBOFLOW_API_KEY environment variable.")
        
        if not config.CLAUDE_API_KEY:
            logging.error(f"[{request_id}] Claude API key not configured")
            raise HTTPException(status_code=503, 
                               detail="Claude API key is not configured. Please set the CLAUDE_API_KEY environment variable.")
        
        # Initialize detector
        detector = SetGameDetector(config)
        
        # Process image with timeout handling
        logging.info(f"[{request_id}] Starting image processing pipeline")
        try:
            # We'll set a reasonable timeout for the processing
            start_time = time.time()
            annotated_image, sets_found, card_features = detector.process_image(image_bytes)
            processing_time = time.time() - start_time
            logging.info(f"[{request_id}] Processing completed in {processing_time:.2f} seconds")
        except Exception as process_error:
            logging.error(f"[{request_id}] Error during image processing: {process_error}", exc_info=True)
            if "timed out" in str(process_error).lower():
                raise HTTPException(status_code=504, detail="Processing timed out. The image may be too complex or the server is under heavy load.")
            else:
                raise
        
        # Encode image for response
        try:
            image_base64 = detector.image_processor.encode_image_to_base64(annotated_image)
            logging.info(f"[{request_id}] Successfully encoded result image to base64")
        except Exception as encode_error:
            logging.error(f"[{request_id}] Error encoding result image: {encode_error}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to encode result image: {str(encode_error)}")
        
        # Return results
        logging.info(f"[{request_id}] Request completed successfully. Found {len(sets_found)} sets and {len(card_features)} cards.")
        return ProcessResponse(
            image_base64=image_base64,
            sets_found=sets_found,
            card_features=card_features
        )
    
    except CardDetectionError as e:
        logging.error(f"[{request_id}] Card detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Card detection failed: {str(e)}")
    except CardClassificationError as e:
        logging.error(f"[{request_id}] Card classification failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Card classification failed: {str(e)}")
    except ClaudeAPIError as e:
        logging.error(f"[{request_id}] Claude API error: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=f"Claude API error: {str(e)}")
    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
    except Exception as e:
        error_detail = traceback.format_exc()
        logging.error(f"[{request_id}] Unexpected error during processing: {e}\n{error_detail}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


# --- Run the application ---
if __name__ == "__main__":
    # Get port from environment variable or default to 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
