# SET Game Detector - API Backend Version
# Modified from the original Colab version

# Imports
import os
import requests
import json
import cv2
import numpy as np
import base64
from itertools import combinations
from PIL import Image
import io
import traceback
import time
from collections import defaultdict
from roboflow import InferenceHTTPClient
import re  # For JSON parsing from Claude
import logging # For logging

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# --- Configuration ---
class Config:
    """Configuration class to centralize parameters and credentials."""
    # API Keys and URLs
    ROBOFLOW_API_KEY = ""  # Default - should be overridden by env var
    CLAUDE_API_KEY = ""  # Default - should be overridden by env var
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
    RESULTS_DIR = "/tmp/set_results"  # Changed to use /tmp for cloud deployments
    TEMP_IMAGE_PATH = "/tmp/temp_image.jpg"  # Changed to use /tmp for cloud deployments


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

    def load_image(self, image_path: str) -> np.ndarray:
        """Loads an image from the given path."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            raise ImageProcessingError(f"Failed to load or decode image: {image_path}. Ensure it's a valid image file.")
        logging.debug(f"Image loaded from: {image_path}")
        return image

    def save_image(self, image: np.ndarray, path: str) -> bool:
        """Saves an image to the specified path."""
        try:
            # Make sure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            return cv2.imwrite(path, image)
        except Exception as e:
            logging.error(f"Error saving image to {path}: {e}", exc_info=True)
            return False


# --- Card Detector ---
class CardDetector:
    """Detects cards in an image using Roboflow."""
    def __init__(self, config: Config, image_processor: ImageProcessor):
        self.config = config
        self.image_processor = image_processor
        self.roboflow_client = InferenceHTTPClient(
            api_url=self.config.ROBOFLOW_API_URL,
            api_key=self.config.ROBOFLOW_API_KEY
        )

    def detect_cards(self, image: np.ndarray) -> tuple[list, dict]:
        """Detects cards in the image and assigns positions."""
        try:
            temp_path = self.config.TEMP_IMAGE_PATH
            if not self.image_processor.save_image(image, temp_path):
                raise ImageProcessingError("Failed to save temporary image for card detection.")
            result = self.roboflow_client.run_workflow(
                workspace_name="tel-aviv",
                workflow_id="custom-workflow",
                images={"image": temp_path},
                use_cache=True
            )
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

        n_rows = 4 if len(cards) > 9 else 3
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
        """Classifies cards using Claude API. Raises error if API key is missing."""
        if not self.config.CLAUDE_API_KEY:
            raise ValueError("Claude API key is required for card classification. Please set CLAUDE_API_KEY in the environment variables.")
        return self._classify_cards_with_claude(image, cards_with_positions)

    def _classify_cards_with_claude(self, image: np.ndarray, cards_with_positions: list) -> dict:
        """Classifies cards using Claude API."""
        try:
            _, img_encoded = cv2.imencode('.jpg', image)
            img_base64 = base64.b64encode(img_encoded).decode('utf-8')
            headers = {
                "x-api-key": self.config.CLAUDE_API_KEY,
                "content-type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            data = {
                "model": self.config.CLAUDE_MODEL,
                "max_tokens": 1000,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_base64}},
                        {"type": "text", "text": (
                            "Analyze the SET card game image. Classify each card by position (1-12, top-left to bottom-right, rows)."
                            "For each card, identify: 1. Number (1, 2, or 3 shapes), 2. Color (Red, Green, or Purple), "
                            "3. Shape (Oval, Diamond, or Squiggle), 4. Shading (Solid, Striped, or Outline)."
                            "Return a JSON object with card positions as keys: "
                            "{'1': {'number': '1|2|3', 'color': 'red|green|purple', 'shape': 'oval|diamond|squiggle', 'shading': 'solid|striped|outline'}, ...}"
                            "Return only JSON, no explanations."
                        )}
                    ]
                }]
            }

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.post(self.config.CLAUDE_API_URL, headers=headers, json=data, timeout=30)
                    response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                    break  # If successful, break retry loop
                except requests.exceptions.RequestException as e:
                    if attempt == max_retries - 1:
                        raise ClaudeAPIError(f"Claude API request failed after {max_retries} retries: {e}")
                    wait_time = min(2 ** attempt, 10)
                    logging.warning(f"Request error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time} seconds...", exc_info=True)
                    time.sleep(wait_time)

            response_json = response.json()
            content = response_json["content"][0]["text"]

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

    def process_image(self, image_path: str) -> tuple[np.ndarray, list[list[int]], dict]:
        """Processes an image to detect SETs."""
        logging.info(f"Starting image processing for: {image_path}")
        try:
            image = self.image_processor.load_image(image_path)

            logging.info("Detecting cards...")
            cards_with_positions, image_dimensions = self.card_detector.detect_cards(image)
            logging.info(f"Detected {len(cards_with_positions)} cards.")

            logging.info("Classifying cards...")
            card_features = self.card_classifier.classify_cards(image, cards_with_positions)

            logging.info("Finding sets...")
            sets_found = self.set_logic.find_sets(card_features)

            logging.info("Drawing sets on image...")
            annotated_image = self.visualizer.draw_sets_on_image(image, sets_found, cards_with_positions)

            logging.info(f"Image processing completed for: {image_path}. Found {len(sets_found)} sets.")
            return annotated_image, sets_found, card_features

        except FileNotFoundError as e:
            logging.error(f"File not found error: {e}")
            return None, [], {}
        except CardDetectionError as e:
            logging.error(f"Card detection error: {e}")
            return None, [], {}
        except CardClassificationError as e:
            logging.error(f"Card classification error: {e}")
            return None, [], {}
        except ClaudeAPIError as e:
            logging.error(f"Claude API error: {e}")
            return None, [], {}
        except ValueError as e:
            logging.error(f"Configuration error: {e}")
            return None, [], {}
        except ImageProcessingError as e:
            logging.error(f"Image processing error: {e}")
            return None, [], {}
        except Exception as e:
            logging.error(f"Unexpected error during image processing: {e}", exc_info=True)
            traceback.print_exc()
            return None, [], {}
