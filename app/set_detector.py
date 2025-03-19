"""
SET Game Detector - Backend Implementation
This module detects SET cards in images and identifies valid SETs
"""
# --- Import Required Libraries ---
import cv2
import numpy as np
import requests
import json
import base64
from itertools import combinations
import io
import traceback
import time
import re
import os
import uuid
import logging
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
from roboflow import Roboflow

from app.config import get_settings

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
class Config:
    """Configuration class to centralize parameters and credentials."""
    
    def __init__(self, roboflow_api_key=None, claude_api_key=None):
        settings = get_settings()
        
        # API Keys and URLs
        self.ROBOFLOW_API_KEY = roboflow_api_key or settings.ROBOFLOW_API_KEY
        self.CLAUDE_API_KEY = claude_api_key or settings.CLAUDE_API_KEY
        self.ROBOFLOW_API_URL = settings.ROBOFLOW_API_URL
        self.CLAUDE_API_URL = settings.CLAUDE_API_URL
        self.CLAUDE_MODEL = settings.CLAUDE_MODEL

        # SET Game parameters
        self.FEATURE_TYPES = {
            'number': ['1', '2', '3'],
            'color': ['red', 'green', 'purple'],
            'shape': ['oval', 'diamond', 'squiggle'],
            'shading': ['solid', 'striped', 'outline']
        }

        # Visualization parameters
        self.COLORS = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
        ]
        self.BOX_THICKNESS = 5
        self.FONT_SCALE = 0.9
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.OFFSET_STEP = 6


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
        """
        Loads an image from bytes.
        
        Args:
            image_bytes: Raw bytes of the image
            
        Returns:
            OpenCV image as numpy array
            
        Raises:
            ImageProcessingError: If image loading fails
        """
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

    def encode_image_to_base64(self, image: np.ndarray) -> str:
        """
        Converts an image to base64 string.
        
        Args:
            image: OpenCV image as numpy array
            
        Returns:
            Base64 encoded string of the image
            
        Raises:
            ImageProcessingError: If encoding fails
        """
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
        
        # Initialize Roboflow SDK
        self.rf = Roboflow(api_key=self.config.ROBOFLOW_API_KEY)
        self.project = self.rf.workspace("tel-aviv").project("custom-workflow")

    def detect_cards(self, image: np.ndarray) -> tuple[list, dict]:
        """
        Detects cards in the image and assigns positions.
        
        Args:
            image: OpenCV image as numpy array
            
        Returns:
            Tuple containing:
                - List of cards with their positions and bounding boxes
                - Dictionary with image dimensions
                
        Raises:
            CardDetectionError: If card detection fails
        """
        try:
            # Save image to a temporary file
            temp_filename = f"{uuid.uuid4()}.jpg"
            cv2.imwrite(temp_filename, image)

            # Get image dimensions
            height, width = image.shape[:2]
            image_dimensions = {"width": width, "height": height}

            # Call Roboflow API
            model = self.project.version("1").model
            result = model.predict(temp_filename, confidence=40, overlap=30)
            
            # Clean up temporary file
            try:
                os.remove(temp_filename)
            except Exception as e:
                logging.warning(f"Failed to remove temporary file {temp_filename}: {e}")

            # Process predictions
            predictions = result.json()
            card_predictions = [pred for pred in predictions['predictions'] if pred["class"] == "card"]

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
            raise CardDetectionError(f"Card detection process failed: {e}")

    def _assign_card_positions(self, card_predictions: list, image_dimensions: dict) -> list:
        """
        Assigns position numbers to detected cards based on their location.
        Cards are grouped into rows and numbered left-to-right, top-to-bottom.
        
        Args:
            card_predictions: List of card predictions from Roboflow
            image_dimensions: Dictionary with image width and height
            
        Returns:
            List of cards with assigned positions
        """
        # Extract card information
        cards = [{
            "x": pred["x"], 
            "y": pred["y"],
            "width": pred["width"], 
            "height": pred["height"],
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

        # Sort each row by x-coordinate (left to right)
        for row in rows:
            row.sort(key=lambda c: c["x"])

        # Assign sequential positions
        cards_with_positions = []
        position = 1
        
        for row in rows:
            for card in row:
                # Calculate bounding box
                x1 = int(card["x"] - card["width"] / 2)
                y1 = int(card["y"] - card["height"] / 2)
                x2 = int(card["x"] + card["width"] / 2)
                y2 = int(card["y"] + card["height"] / 2)
                
                cards_with_positions.append({
                    "position": position,
                    "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
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
        """
        Classifies the SET card features using Claude API.
        
        Args:
            image: OpenCV image as numpy array
            cards_with_positions: List of cards with their positions
            
        Returns:
            Dictionary with card features keyed by position
            
        Raises:
            CardClassificationError: If classification fails
        """
        if not self.config.CLAUDE_API_KEY:
            raise CardClassificationError("Claude API key is required for card classification")
        
        return self._classify_cards_with_claude(image, cards_with_positions)

    def _prepare_labeled_image_for_claude(self, image: np.ndarray, cards_with_positions: list) -> np.ndarray:
        """
        Creates a labeled version of the image with position numbers for Claude.
        
        Args:
            image: OpenCV image as numpy array
            cards_with_positions: List of cards with their positions
            
        Returns:
            Image with position labels
        """
        labeled_image = image.copy()
        
        # Add position labels to each card
        for card in cards_with_positions:
            position = card["position"]
            center_x, center_y = card["center"]["x"], card["center"]["y"]

            # Draw a circle with position number for clear identification
            cv2.circle(labeled_image, (center_x, center_y), 20, (0, 0, 0), -1)
            cv2.putText(
                labeled_image, 
                str(position), 
                (center_x - 7, center_y + 7),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (255, 255, 255), 
                2, 
                cv2.LINE_AA
            )

        return labeled_image

    def _classify_cards_with_claude(self, image: np.ndarray, cards_with_positions: list) -> dict:
        """
        Classifies cards using Claude API with position labeling.
        
        Args:
            image: OpenCV image as numpy array
            cards_with_positions: List of cards with their positions
            
        Returns:
            Dictionary with card features keyed by position
            
        Raises:
            CardClassificationError: If classification fails
            ClaudeAPIError: If API call fails
        """
        try:
            # Create labeled image with position numbers for Claude
            labeled_image = self._prepare_labeled_image_for_claude(image, cards_with_positions)

            # Encode image for API request
            img_base64 = self.image_processor.encode_image_to_base64(labeled_image)

            # Set up request headers
            headers = {
                "x-api-key": self.config.CLAUDE_API_KEY,
                "content-type": "application/json",
                "anthropic-version": "2023-06-01"
            }

            # Create detailed position information for the prompt
            position_details = [
                f"Card {card['position']}: at coordinates ({card['center']['x']}, {card['center']['y']})"
                for card in cards_with_positions
            ]
            position_info = "\n".join(position_details)

            # Create prompt for Claude
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
                f"{{'1': {{'number': '1|2|3', 'color': 'red|green|purple', 'shape': 'oval|diamond|squiggle', "
                f"'shading': 'solid|striped|outline'}}, ...}}\n"
                f"Return only JSON, no explanations. Use the white numbers in black circles as position references."
            )

            # Prepare request data
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

            # Send request to Claude API with retries
            logging.info("Sending request to Claude API...")
            max_retries = 3
            
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        self.config.CLAUDE_API_URL, 
                        headers=headers, 
                        json=data, 
                        timeout=30
                    )
                    response.raise_for_status()
                    break
                except requests.exceptions.RequestException as e:
                    if attempt == max_retries - 1:
                        raise ClaudeAPIError(f"Claude API request failed after {max_retries} retries: {e}")
                    
                    wait_time = min(2 ** attempt, 10)
                    logging.warning(
                        f"Request error (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {wait_time} seconds..."
                    )
                    time.sleep(wait_time)

            # Process response
            response_json = response.json()
            content = response_json["content"][0]["text"]
            logging.info("Received response from Claude API.")

            # Try to extract JSON from potential code blocks
            json_match = re.search(r'```json\n([\s\S]*?)\n```', content)
            classification_json = json_match.group(1) if json_match else content

            try:
                classification_results = json.loads(classification_json)
                logging.info("Card classification successful using Claude API.")
                return classification_results
            except json.JSONDecodeError:
                logging.error("Failed to parse Claude's JSON response. Raw response:\n%s", content)
                raise CardClassificationError(
                    "Failed to parse Claude's response as JSON. Please check response format."
                )

        except requests.exceptions.HTTPError as e:
            logging.error(f"Claude API HTTP error: {e}. Response: {e.response.text}")
            raise ClaudeAPIError(f"Claude API HTTP error: {e}. Status Code: {e.response.status_code}")
        except ClaudeAPIError as e:
            logging.error(f"Claude API error after retries: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during card classification with Claude: {e}")
            raise CardClassificationError(f"Card classification with Claude failed: {e}")


# --- SET Game Logic ---
class SetGameLogic:
    """Implements the logic to find SETs in a set of cards."""
    
    def __init__(self, config: Config):
        self.config = config

    def is_set(self, cards: list[dict]) -> bool:
        """
        Checks if a group of three cards forms a valid SET.
        A valid SET requires that for each feature, all cards have either
        all the same value or all different values.
        
        Args:
            cards: List of three card feature dictionaries
            
        Returns:
            True if the cards form a valid SET, False otherwise
        """
        for feature in self.config.FEATURE_TYPES.keys():
            values = [card[feature] for card in cards]
            # If there are exactly 2 distinct values, it's not a SET
            if len(set(values)) == 2:
                return False
        return True

    def find_sets(self, card_features: dict) -> list[list[int]]:
        """
        Finds all valid SETs from the given card features.
        
        Args:
            card_features: Dictionary of card features keyed by position
            
        Returns:
            List of lists, where each inner list contains positions of cards forming a SET
        """
        sets_found = []
        positions = list(card_features.keys())
        
        # Try all combinations of 3 cards
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

    def draw_sets_on_image(self, image: np.ndarray, sets_found: list[list[int]], 
                          cards_with_positions: list) -> np.ndarray:
        """
        Draws bounding boxes, labels, and position numbers for detected sets on the image.
        
        Args:
            image: OpenCV image as numpy array
            sets_found: List of SETs found (each SET is a list of card positions)
            cards_with_positions: List of cards with their positions and bounding boxes
            
        Returns:
            Annotated image with SETs highlighted
        """
        annotated_image = image.copy()

        # First, draw position numbers on all cards for clarity
        for card in cards_with_positions:
            position = card["position"]
            center_x, center_y = card["center"]["x"], card["center"]["y"]

            # Draw a circle with the position number
            cv2.circle(annotated_image, (center_x, center_y), 15, (255, 255, 255), -1)
            cv2.circle(annotated_image, (center_x, center_y), 15, (0, 0, 0), 1)
            cv2.putText(
                annotated_image, 
                str(position), 
                (center_x - 5, center_y + 5),
                self.config.FONT, 
                0.5, 
                (0, 0, 0), 
                1, 
                lineType=cv2.LINE_AA
            )

        if not sets_found:
            logging.info("No sets found to draw.")
            return annotated_image

        # Create a mapping from position to card data for easier access
        card_position_map = {card["position"]: card for card in cards_with_positions}
        
        # For each card, track which sets it belongs to
        card_set_membership = defaultdict(list)
        for set_idx, set_positions in enumerate(sets_found):
            for pos in set_positions:
                card_set_membership[pos].append(set_idx)

        # Draw each card's SET memberships
        for pos, set_indices in card_set_membership.items():
            if pos not in card_position_map:
                logging.warning(f"Card position {pos} not found in position map during visualization.")
                continue
                
            card = card_position_map[pos]
            box = card["box"]

            # Draw a different colored box for each SET this card belongs to
            for i, set_idx in enumerate(set_indices):
                color = self.config.COLORS[set_idx % len(self.config.COLORS)]
                offset = i * self.config.OFFSET_STEP
                
                x1, y1 = box["x1"] - offset, box["y1"] - offset
                x2, y2 = box["x2"] + offset, box["y2"] + offset
                
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, self.config.BOX_THICKNESS)
                
                # Label only the first card in each SET
                if pos == sets_found[set_idx][0]:
                    cv2.putText(
                        annotated_image, 
                        f"Set {set_idx + 1}", 
                        (x1, y1 - 10),
                        self.config.FONT, 
                        self.config.FONT_SCALE, 
                        color, 
                        self.config.BOX_THICKNESS, 
                        lineType=cv2.LINE_AA
                    )

        logging.info("Sets and position numbers drawn on image.")
        return annotated_image


# --- Main Detector Class ---
class SetGameDetector:
    """Orchestrates the SET game detection process."""
    
    def __init__(self, roboflow_api_key=None, claude_api_key=None):
        self.config = Config(roboflow_api_key, claude_api_key)
        self.image_processor = ImageProcessor(self.config)
        self.card_detector = CardDetector(self.config, self.image_processor)
        self.card_classifier = CardClassifier(self.config, self.image_processor)
        self.set_logic = SetGameLogic(self.config)
        self.visualizer = Visualizer(self.config, self.image_processor)

    def process_image(self, image_bytes: bytes) -> dict:
        """
        Processes an image to detect SET cards and find valid SETs.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Dictionary containing:
                - image: Base64 encoded annotated image
                - sets_found: List of SETs found with card features
                - all_cards: List of all detected cards with features
                
        Raises:
            Various exceptions for different types of failures
        """
        logging.info("Starting image processing")
        try:
            # Load and process image
            image = self.image_processor.load_image_from_bytes(image_bytes)

            # Detect cards
            logging.info("Detecting cards...")
            cards_with_positions, image_dimensions = self.card_detector.detect_cards(image)
            logging.info(f"Detected {len(cards_with_positions)} cards.")

            # Classify cards
            logging.info("Classifying cards...")
            card_features = self.card_classifier.classify_cards(image, cards_with_positions)

            # Find sets
            logging.info("Finding sets...")
            sets_found = self.set_logic.find_sets(card_features)

            # Draw sets on image
            logging.info("Drawing sets on image...")
            annotated_image = self.visualizer.draw_sets_on_image(image, sets_found, cards_with_positions)

            # Convert image to base64 for return
            annotated_image_base64 = self.image_processor.encode_image_to_base64(annotated_image)
            
            # Construct results
            results = {
                "image": annotated_image_base64,
                "sets_found": [
                    {
                        "set_id": i + 1,
                        "cards": [
                            {
                                "position": pos,
                                "features": card_features[str(pos)]
                            } for pos in set_positions
                        ]
                    } for i, set_positions in enumerate(sets_found)
                ],
                "all_cards": [
                    {
                        "position": int(pos),
                        "features": features
                    } for pos, features in card_features.items()
                ]
            }

            logging.info(f"Image processing completed. Found {len(sets_found)} sets.")
            return results

        except Exception as e:
            logging.error(f"Error during image processing: {e}")
            raise
