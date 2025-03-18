# SET Game Detector - Production Backend API
# For Render.com deployment

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
from collections import defaultdict
from inference_sdk import InferenceHTTPClient
import re  # For JSON parsing from Claude
import logging  # For logging
import uuid
from typing import List, Dict, Any, Optional, Tuple
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import tempfile

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(levelname)s - %(message)s')

# --- FastAPI App Setup ---
app = FastAPI(
    title="SET Game Detector API",
    description="API for detecting SET card games from images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
            
            result = self.roboflow_client.run_workflow(
                workspace_name="tel-aviv",
                workflow_id="custom-workflow",
                images={"image": temp_path},
                use_cache=True
            )
            
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
        """Assigns position numbers to detected cards based on their location."""
        cards = [{
            "x": pred["x"], "y": pred["y"], "width": pred["width"], "height": pred["height"], "detection_id": pred["detection_id"]
        } for pred in card_predictions]

        # Adjust number of rows based on total cards - more flexible algorithm
        total_cards = len(cards)
        
        # Estimate number of rows based on typical SET layouts
        if total_cards <= 4:
            n_rows = 1
        elif total_cards <= 9:
            n_rows = 3
        elif total_cards <= 16:
            n_rows = 4
        else:
            n_rows = 5
            
        row_height = image_dimensions["height"] / n_rows
        rows = defaultdict(list)

        for card in cards:
            row_index = int(card["y"] / row_height)
            rows[row_index].append(card)

        for row_index in rows:
            rows[row_index].sort(key=lambda c: c["x"])

        cards_with_positions = []
        position = 1
        for row_index in sorted(rows.keys()):
            for card in rows[row_index]:
                cards_with_positions.append({
                    "position": position,
                    "box": {
                        "x1": int(card["x"] - card["width"] / 2), "y1": int(card["y"] - card["height"] / 2),
                        "x2": int(card["x"] + card["width"] / 2), "y2": int(card["y"] + card["height"] / 2)
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

    def _classify_cards_with_claude(self, image: np.ndarray, cards_with_positions: list) -> dict:
        """Classifies cards using Claude API."""
        try:
            img_base64 = self.image_processor.encode_image_to_base64(image)
            headers = {
                "x-api-key": self.config.CLAUDE_API_KEY,
                "content-type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            # Update prompt to handle any number of cards
            total_cards = len(cards_with_positions)
            prompt = (
                f"Analyze the SET card game image. There are {total_cards} cards visible in a grid layout. "
                f"Cards are numbered from top-left to bottom-right, in rows. "
                f"For each card, identify: 1. Number (1, 2, or 3 shapes), 2. Color (Red, Green, or Purple), "
                f"3. Shape (Oval, Diamond, or Squiggle), 4. Shading (Solid, Striped, or Outline). "
                f"Return a JSON object with card positions as keys: "
                f"{{'1': {{'number': '1|2|3', 'color': 'red|green|purple', 'shape': 'oval|diamond|squiggle', 'shading': 'solid|striped|outline'}}, ...}} "
                f"Return only JSON, no explanations."
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
                    response = requests.post(self.config.CLAUDE_API_URL, headers=headers, json=data, timeout=30)
                    response.raise_for_status()
                    break
                except requests.exceptions.RequestException as e:
                    if attempt == max_retries - 1:
                        raise ClaudeAPIError(f"Claude API request failed after {max_retries} retries: {e}")
                    wait_time = min(2 ** attempt, 10)
                    logging.warning(f"Request error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time} seconds...", exc_info=True)
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
            logging.error(f"Claude API HTTP error: {e}. Response: {e.response.text}", exc_info=True)
            raise ClaudeAPIError(f"Claude API HTTP error: {e}. Status Code: {e.response.status_code}")
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
        """Draws bounding boxes and labels for detected sets on the image."""
        annotated_image = image.copy()
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
        logging.info("Sets drawn on image.")
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
        logging.info("Starting image processing")
        try:
            image = self.image_processor.load_image_from_bytes(image_bytes)

            logging.info("Detecting cards...")
            cards_with_positions, image_dimensions = self.card_detector.detect_cards(image)
            logging.info(f"Detected {len(cards_with_positions)} cards.")

            logging.info("Classifying cards...")
            card_features = self.card_classifier.classify_cards(image, cards_with_positions)

            logging.info("Finding sets...")
            sets_found = self.set_logic.find_sets(card_features)

            logging.info("Drawing sets on image...")
            annotated_image = self.visualizer.draw_sets_on_image(image, sets_found, cards_with_positions)

            logging.info(f"Image processing completed. Found {len(sets_found)} sets.")
            return annotated_image, sets_found, card_features

        except Exception as e:
            logging.error(f"Error during image processing: {e}", exc_info=True)
            raise


# --- FastAPI Routes ---
class ProcessResponse(BaseModel):
    """Model for the response of the process endpoint."""
    image_base64: str
    sets_found: List[List[int]]
    card_features: Dict[str, Dict[str, str]]


@app.post("/api/process", response_model=ProcessResponse)
async def process_image(file: UploadFile = File(...)):
    """Process an image of a SET game and return the results."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Initialize detector
        detector = SetGameDetector()
        
        # Process image
        annotated_image, sets_found, card_features = detector.process_image(image_bytes)
        
        # Encode image for response
        image_base64 = detector.image_processor.encode_image_to_base64(annotated_image)
        
        # Return results
        return ProcessResponse(
            image_base64=image_base64,
            sets_found=sets_found,
            card_features=card_features
        )
    
    except CardDetectionError as e:
        raise HTTPException(status_code=400, detail=f"Card detection failed: {str(e)}")
    except CardClassificationError as e:
        raise HTTPException(status_code=400, detail=f"Card classification failed: {str(e)}")
    except ClaudeAPIError as e:
        raise HTTPException(status_code=503, detail=f"Claude API error: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error during processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


# --- Run the application ---
if __name__ == "__main__":
    # Get port from environment variable or default to 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
