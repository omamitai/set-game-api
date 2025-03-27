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
from inference_sdk import InferenceHTTPClient

# Attempt to import get_settings, handle potential import error gracefully
try:
    from app.config import get_settings
except ImportError:
    logging.warning("Could not import 'get_settings' from 'app.config'. Using environment variables directly or defaults.")
    # Define a dummy get_settings if it doesn't exist to avoid NameError
    class DummySettings:
        ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        ROBOFLOW_API_URL = os.getenv("ROBOFLOW_API_URL", "https://infer.roboflow.com")
        # Default Gemini URL Template - Requires model and key
        GEMINI_API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite") # Changed default to 2.0 flash lite

    def get_settings():
        return DummySettings()

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
class Config:
    """Configuration class to centralize parameters and credentials."""

    def __init__(self, roboflow_api_key=None, gemini_api_key=None):
        settings = get_settings()

        # API Keys and URLs
        self.ROBOFLOW_API_KEY = roboflow_api_key or settings.ROBOFLOW_API_KEY
        self.GEMINI_API_KEY = gemini_api_key or settings.GEMINI_API_KEY
        self.ROBOFLOW_API_URL = settings.ROBOFLOW_API_URL
        # Gemini URL requires formatting with model and key
        self.GEMINI_API_URL_TEMPLATE = settings.GEMINI_API_URL_TEMPLATE
        # Using gemini-1.5-flash as it's generally available and strong for multimodal
        self.GEMINI_MODEL = settings.GEMINI_MODEL

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
        self.BOX_THICKNESS = 8
        self.FONT_SCALE = 0.9
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.OFFSET_STEP = 10


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

class GeminiAPIError(Exception):
    """Exception raised when Gemini API call fails."""
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
            # Encode to JPEG for potentially smaller size
            success, img_encoded = cv2.imencode('.jpg', image)
            if not success:
                raise ImageProcessingError("Failed to encode image to JPEG format.")
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
            logging.warning("Roboflow API key not set! Card detection may fail.")
            # Allow initialization but expect errors later if key is needed and missing
            self.roboflow_client = None
        else:
            try:
                self.roboflow_client = InferenceHTTPClient(
                    api_url=self.config.ROBOFLOW_API_URL,
                    api_key=self.config.ROBOFLOW_API_KEY
                )
            except Exception as e:
                logging.error(f"Failed to initialize Roboflow client: {e}", exc_info=True)
                self.roboflow_client = None # Ensure client is None if init fails

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
        if not self.roboflow_client:
             raise CardDetectionError("Roboflow client not initialized. Check API key and configuration.")

        temp_filename = None # Initialize to None
        try:
            # Save image to a temporary file
            temp_filename = f"{uuid.uuid4()}.jpg"
            success = cv2.imwrite(temp_filename, image)
            if not success:
                raise ImageProcessingError(f"Failed to write temporary image file: {temp_filename}")

            logging.info("Calling Roboflow API for card detection...")
            # Call Roboflow API - Adjust parameters if needed based on your workflow
            # Assuming a simple object detection workflow for "card" class
            result = self.roboflow_client.infer(temp_filename, model_id="set-cards-detection/1") # Example model_id, replace with yours
            # Or use run_workflow if applicable:
            # result = self.roboflow_client.run_workflow(...)

            # Process predictions
            # The structure of 'result' depends on the Roboflow endpoint used (infer vs workflow)
            # Adapt the parsing logic based on the actual response structure
            if isinstance(result, list) and len(result) > 0 and "predictions" in result[0]: # Workflow structure?
                 predictions_data = result[0]["predictions"]
                 predictions = predictions_data.get("predictions", []) # Get the list part
                 image_dimensions = predictions_data.get("image", {"width": image.shape[1], "height": image.shape[0]})

            elif isinstance(result, dict) and "predictions" in result: # Infer structure?
                predictions = result["predictions"]
                image_dimensions = result.get("image", {"width": image.shape[1], "height": image.shape[0]})
            else:
                logging.error(f"Unexpected Roboflow response structure: {result}")
                raise CardDetectionError("Could not parse card predictions from Roboflow response.")


            card_predictions = [pred for pred in predictions if pred["class"] == "card"] # Adjust class name if needed

            if not card_predictions:
                # It's possible no cards are present, log as warning instead of error?
                logging.warning("No cards detected in the image by Roboflow.")
                # Return empty list and image dimensions
                return [], image_dimensions
                # Or raise CardDetectionError("No cards detected in the image by Roboflow.") if it's always an error

            cards_with_positions = self._assign_card_positions(card_predictions, image_dimensions)
            logging.info(f"Detected {len(cards_with_positions)} cards using Roboflow.")
            return cards_with_positions, image_dimensions

        except CardDetectionError as e:
            logging.error(f"Card detection failed: {e}", exc_info=True)
            raise
        except Exception as e:
            logging.error(f"Unexpected error during card detection: {e}", exc_info=True)
            # Log the specific Roboflow error if available (e.g., from requests.HTTPError)
            if hasattr(e, 'response') and e.response is not None:
                 logging.error(f"Roboflow API response: {e.response.text}")
            raise CardDetectionError(f"Card detection process failed: {e}")
        finally:
            # Clean up temporary file if it was created
            if temp_filename and os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except Exception as e_rem:
                    logging.warning(f"Failed to remove temporary file {temp_filename}: {e_rem}")


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
        if not card_predictions:
            return []

        # Extract card information
        cards = []
        for pred in card_predictions:
             # Ensure necessary keys exist
             if all(k in pred for k in ["x", "y", "width", "height"]):
                 cards.append({
                     "x": pred["x"],
                     "y": pred["y"],
                     "width": pred["width"],
                     "height": pred["height"],
                     "detection_id": pred.get("detection_id", uuid.uuid4().hex) # Use provided ID or generate one
                 })
             else:
                 logging.warning(f"Skipping prediction due to missing keys: {pred}")
        
        if not cards:
            logging.warning("No valid card predictions found after filtering.")
            return []


        # Calculate average card height to use as a threshold for vertical grouping
        avg_card_height = sum(card["height"] for card in cards) / len(cards)
        vertical_threshold = avg_card_height * 0.5 # Adjust multiplier if needed (e.g., 0.6)

        # Group cards by approximate rows (cards with similar y-coordinates)
        rows = []
        # Sort primarily by Y, then by X as a secondary sort for stability
        cards_sorted_by_y = sorted(cards, key=lambda c: (c["y"], c["x"]))

        if not cards_sorted_by_y: return [] # Should not happen if cards list was not empty

        current_row = [cards_sorted_by_y[0]]
        last_y_in_row = cards_sorted_by_y[0]["y"]

        for card in cards_sorted_by_y[1:]:
            # Check proximity to the *average* Y of the current row or the last card added
            # Using the Y of the first card in the row might be more stable for tilted rows
            # if abs(card["y"] - current_row[0]["y"]) <= vertical_threshold:
            # Alternative: check proximity to the last card added to the row
            if abs(card["y"] - last_y_in_row) <= vertical_threshold:
                current_row.append(card)
                # Update the reference Y, perhaps using the average Y of the growing row
                last_y_in_row = sum(c['y'] for c in current_row) / len(current_row)
            else:
                # Finalize the previous row (sort by x) and start a new one
                current_row.sort(key=lambda c: c["x"])
                rows.append(current_row)
                current_row = [card]
                last_y_in_row = card["y"]

        # Add the last row if not empty
        if current_row:
            current_row.sort(key=lambda c: c["x"])
            rows.append(current_row)

        # Assign sequential positions
        cards_with_positions = []
        position = 1

        img_width = image_dimensions.get("width", 0)
        img_height = image_dimensions.get("height", 0)

        for row in rows:
            for card in row:
                # Calculate bounding box, ensuring coordinates are within image bounds
                x_center, y_center = card["x"], card["y"]
                width, height = card["width"], card["height"]

                x1 = max(0, int(x_center - width / 2))
                y1 = max(0, int(y_center - height / 2))
                # Ensure x2 and y2 do not exceed image dimensions if available
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                if img_width > 0: x2 = min(img_width -1 , x2)
                if img_height > 0: y2 = min(img_height -1, y2)


                cards_with_positions.append({
                    "position": position,
                    "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "center": {"x": int(x_center), "y": int(y_center)},
                    "detection_id": card["detection_id"]
                })
                position += 1

        return cards_with_positions


# --- Card Classifier ---
class CardClassifier:
    """Classifies card features using Google Gemini API."""

    def __init__(self, config: Config, image_processor: ImageProcessor):
        self.config = config
        self.image_processor = image_processor

    def classify_cards(self, image: np.ndarray, cards_with_positions: list) -> dict:
        """
        Classifies the SET card features using Gemini API.

        Args:
            image: OpenCV image as numpy array
            cards_with_positions: List of cards with their positions

        Returns:
            Dictionary with card features keyed by position (as strings)

        Raises:
            CardClassificationError: If classification fails
        """
        if not self.config.GEMINI_API_KEY:
            raise CardClassificationError("Gemini API key is required for card classification")
        if not cards_with_positions:
             logging.warning("No cards provided for classification.")
             return {}

        return self._classify_cards_with_gemini(image, cards_with_positions)

    def _prepare_labeled_image_for_llm(self, image: np.ndarray, cards_with_positions: list) -> np.ndarray:
        """
        Creates a labeled version of the image with position numbers for the LLM.

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
            # Make circle/text size relative to card size or image size? Fixed for now.
            radius = 20
            font_scale = 0.7
            thickness = 2
            text_offset_x = int(radius * 0.35) # Adjust for centering
            text_offset_y = int(radius * 0.35)

            cv2.circle(labeled_image, (center_x, center_y), radius, (0, 0, 0), -1) # Black circle
            cv2.putText(
                labeled_image,
                str(position),
                (center_x - text_offset_x - (len(str(position))-1)*5 , center_y + text_offset_y), # Adjust x pos based on number of digits
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255), # White text
                thickness,
                cv2.LINE_AA
            )

        return labeled_image

    def _classify_cards_with_gemini(self, image: np.ndarray, cards_with_positions: list) -> dict:
        """
        Classifies cards using Google Gemini API with position labeling.

        Args:
            image: OpenCV image as numpy array
            cards_with_positions: List of cards with their positions

        Returns:
            Dictionary with card features keyed by position (as strings)

        Raises:
            CardClassificationError: If classification fails
            GeminiAPIError: If API call fails
        """
        try:
            # Create labeled image with position numbers for Gemini
            labeled_image = self._prepare_labeled_image_for_llm(image, cards_with_positions)

            # Encode image for API request
            img_base64 = self.image_processor.encode_image_to_base64(labeled_image)

            # Construct API URL
            api_url = self.config.GEMINI_API_URL_TEMPLATE.format(
                model=self.config.GEMINI_MODEL,
                api_key=self.config.GEMINI_API_KEY
            )

            # Set up request headers
            headers = {
                "Content-Type": "application/json",
            }

            # Create detailed position information for the prompt
            position_details = [
                f"Card {card['position']}: located near the black circle containing the white number '{card['position']}'"
                for card in cards_with_positions
            ]
            position_info = "\n".join(position_details)

            # Create prompt for Gemini (similar to Claude's but emphasizing the labels)
            # Gemini 1.5 Flash is good with JSON, let's be explicit.
            prompt = (
                f"You are analyzing an image of cards from the game SET. Each relevant card in the image has been marked with a black circle containing a white number.\n"
                f"There are {len(cards_with_positions)} marked cards visible with the following position numbers:\n"
                f"{position_info}\n\n"
                f"Your task is to identify the 4 features for EACH numbered card. The features are:\n"
                f"1. Number: The count of shapes on the card (1, 2, or 3).\n"
                f"2. Color: The color of the shapes (red, green, or purple).\n"
                f"3. Shape: The type of shape (oval, diamond, or squiggle).\n"
                f"4. Shading: The filling of the shape (solid, striped, or outline).\n\n"
                f"Carefully examine the card associated with each white number in a black circle.\n"
                f"Provide your response STRICTLY as a single JSON object. Do not include any text before or after the JSON object. Do not use markdown formatting (like ```json).\n"
                f"The JSON object should have the card position numbers (as strings) as keys. Each value should be another JSON object containing the four features exactly as specified below:\n"
                f"{{\n"
                f'  "1": {{"number": "1|2|3", "color": "red|green|purple", "shape": "oval|diamond|squiggle", "shading": "solid|striped|outline"}},\n'
                f'  "2": {{...}},\n'
                f'  "{len(cards_with_positions)}": {{...}}\n'
                f"}}"
            )

            # Prepare request data according to Gemini API format
            data = {
                "contents": [{
                    "parts": [
                        {"text": prompt}, # Text part first
                        { # Image part
                            "inline_data": {
                                "mime_type": "image/jpeg", # Match the encoding format
                                "data": img_base64
                            }
                        }
                    ]
                }],
                # Optional: Add generation config if needed (e.g., temperature, max output tokens)
                # "generationConfig": {
                #     "maxOutputTokens": 1500,
                #     "temperature": 0.2 # Lower temp for more deterministic JSON output
                # }
            }


            # Send request to Gemini API with retries
            logging.info(f"Sending request to Gemini API ({self.config.GEMINI_MODEL})...")
            max_retries = 3
            response = None # Initialize response to None

            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        api_url,
                        headers=headers,
                        json=data,
                        timeout=60 # Increased timeout for potentially larger payload/processing
                    )
                    # Check for HTTP errors (4xx, 5xx)
                    response.raise_for_status()
                    # If successful, break the loop
                    break
                except requests.exceptions.Timeout:
                     logging.warning(f"Gemini API request timed out (attempt {attempt + 1}/{max_retries}). Retrying...")
                     if attempt == max_retries - 1:
                         raise GeminiAPIError(f"Gemini API request failed after {max_retries} retries due to timeout.")
                     time.sleep(min(2 ** attempt, 10)) # Exponential backoff
                except requests.exceptions.RequestException as e:
                    logging.warning(f"Gemini API request error (attempt {attempt + 1}/{max_retries}): {e}. Retrying...")
                    # Check if it's an HTTP error to potentially log response body
                    if isinstance(e, requests.exceptions.HTTPError) and e.response is not None:
                         logging.warning(f"Gemini API response (error): {e.response.text}")
                    if attempt == max_retries - 1:
                        raise GeminiAPIError(f"Gemini API request failed after {max_retries} retries: {e}")
                    time.sleep(min(2 ** attempt, 10)) # Exponential backoff

            # If loop finished without success (e.g., all retries failed)
            if response is None or not response.ok:
                 # This case should ideally be caught by the exceptions above, but as a safeguard:
                 err_msg = f"Gemini API request failed with status {response.status_code}" if response else "Gemini API request failed after retries"
                 if response is not None: err_msg += f" Response: {response.text}"
                 raise GeminiAPIError(err_msg)


            # Process successful response
            response_json = response.json()
            logging.info("Received response from Gemini API.")

            # Extract content - Handle potential errors like blocked content
            if "candidates" not in response_json or not response_json["candidates"]:
                 # Check for prompt feedback for reasons (e.g., safety)
                 feedback = response_json.get("promptFeedback")
                 if feedback and feedback.get("blockReason"):
                     reason = feedback["blockReason"]
                     details = feedback.get("safetyRatings", [])
                     logging.error(f"Gemini API blocked the prompt/response. Reason: {reason}. Details: {details}")
                     raise GeminiAPIError(f"Gemini API request blocked due to: {reason}")
                 else:
                     logging.error(f"Invalid Gemini response: 'candidates' field missing or empty. Response: {response_json}")
                     raise GeminiAPIError("Gemini API response missing 'candidates'.")


            try:
                # Assuming the first candidate is the one we want
                candidate = response_json["candidates"][0]
                if "content" not in candidate or "parts" not in candidate["content"] or not candidate["content"]["parts"]:
                     raise GeminiAPIError("Gemini response candidate missing content parts.")

                # Get the text part
                generated_text = candidate["content"]["parts"][0].get("text", "")
                if not generated_text:
                    raise GeminiAPIError("Gemini response text part is empty.")

                logging.debug(f"Raw Gemini response text:\n{generated_text}")

                # Clean potential markdown code blocks (though we asked not to use them)
                json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', generated_text, re.IGNORECASE)
                if json_match:
                    classification_json_str = json_match.group(1).strip()
                    logging.info("Extracted JSON from markdown code block.")
                else:
                    # Assume the whole text is JSON, strip potential leading/trailing whitespace/newlines
                    classification_json_str = generated_text.strip()

                # Attempt to parse the JSON
                classification_results = json.loads(classification_json_str)

                # --- Data Validation (Crucial Step) ---
                validated_results = {}
                expected_keys = set(self.config.FEATURE_TYPES.keys())
                valid_positions = {str(card['position']) for card in cards_with_positions}

                for pos, features in classification_results.items():
                    if pos not in valid_positions:
                        logging.warning(f"Gemini returned classification for unexpected position '{pos}'. Ignoring.")
                        continue
                    if not isinstance(features, dict):
                        logging.warning(f"Features for position '{pos}' is not a dictionary: {features}. Skipping.")
                        continue

                    if set(features.keys()) != expected_keys:
                         logging.warning(f"Features for position '{pos}' have incorrect keys: {features.keys()}. Expected: {expected_keys}. Skipping.")
                         continue

                    # Optional: Validate feature values against allowed lists
                    is_valid = True
                    for feature_name, allowed_values in self.config.FEATURE_TYPES.items():
                        value = features.get(feature_name)
                        if value not in allowed_values:
                            logging.warning(f"Invalid value '{value}' for feature '{feature_name}' at position '{pos}'. Allowed: {allowed_values}. Skipping card.")
                            is_valid = False
                            break # Stop checking features for this card
                    
                    if is_valid:
                        validated_results[pos] = features
                    # -----------------------------------------

                if not validated_results:
                     raise CardClassificationError("Gemini response parsed, but no valid card classifications found after validation.")

                logging.info(f"Card classification successful using Gemini API. Validated {len(validated_results)} cards.")
                return validated_results # Return validated results

            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse Gemini's JSON response. Error: {e}. Raw response text:\n{generated_text}", exc_info=True)
                raise CardClassificationError(
                    f"Failed to parse Gemini's response as JSON. Check response format. Raw text: {generated_text[:500]}..." # Log snippet
                )
            except KeyError as e:
                 logging.error(f"KeyError accessing Gemini response structure: {e}. Response: {response_json}", exc_info=True)
                 raise GeminiAPIError(f"Invalid Gemini response structure (KeyError: {e}).")
            except IndexError as e:
                 logging.error(f"IndexError accessing Gemini response structure: {e}. Response: {response_json}", exc_info=True)
                 raise GeminiAPIError(f"Invalid Gemini response structure (IndexError: {e}).")


        except requests.exceptions.HTTPError as e:
            # This catches the raise_for_status() errors
            error_body = e.response.text if e.response else "No response body"
            logging.error(f"Gemini API HTTP error: {e}. Status Code: {e.response.status_code}. Response: {error_body}", exc_info=True)
            raise GeminiAPIError(f"Gemini API HTTP error: {e}. Status Code: {e.response.status_code}. Body: {error_body[:500]}...")
        except GeminiAPIError as e:
            # Re-raise specific API errors
            logging.error(f"Gemini API error: {e}", exc_info=True)
            raise
        except ImageProcessingError as e:
             # Handle errors from image encoding
             logging.error(f"Image processing error during classification: {e}", exc_info=True)
             raise CardClassificationError(f"Image processing failed before Gemini call: {e}")
        except Exception as e:
            # Catch any other unexpected errors
            logging.error(f"Unexpected error during card classification with Gemini: {e}", exc_info=True)
            # Include traceback in the log
            logging.error(traceback.format_exc())
            raise CardClassificationError(f"Unexpected error during Gemini classification: {e}")


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
        if len(cards) != 3:
            return False # Only groups of 3 can form a set

        for feature in self.config.FEATURE_TYPES.keys():
            try:
                values = [card[feature] for card in cards]
                # If there are exactly 2 distinct values, it's not a SET
                if len(set(values)) == 2:
                    return False
            except KeyError:
                logging.warning(f"Feature '{feature}' missing in one of the cards during SET check. Cards: {cards}")
                return False # Cannot be a set if features are missing
        return True # If loop completes, all features satisfy the condition

    def find_sets(self, card_features: dict) -> list[list[int]]:
        """
        Finds all valid SETs from the given card features.

        Args:
            card_features: Dictionary of card features keyed by position (string keys)

        Returns:
            List of lists, where each inner list contains integer positions of cards forming a SET
        """
        sets_found = []
        # Ensure positions are strings for lookup, but store ints in the result
        positions = list(card_features.keys()) # These should be strings '1', '2', etc.

        if len(positions) < 3:
             logging.info("Not enough cards classified to find a set.")
             return []

        # Try all combinations of 3 cards
        for combo_pos_str in combinations(positions, 3):
            try:
                # Retrieve features using string keys
                cards_in_combo = [card_features[pos_str] for pos_str in combo_pos_str]
                if self.is_set(cards_in_combo):
                    # Convert string positions to integers for the final result list
                    sets_found.append(list(map(int, combo_pos_str)))
            except KeyError as e:
                # This might happen if card_features dict is inconsistent, though validation should prevent it
                logging.warning(f"Card position key missing during set finding: {e}. Combo: {combo_pos_str}. Skipping.")
                continue
            except Exception as e:
                 logging.error(f"Unexpected error checking combination {combo_pos_str}: {e}", exc_info=True)
                 continue # Skip this combination

        logging.info(f"Found {len(sets_found)} sets from {len(positions)} classified cards.")
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
            sets_found: List of SETs found (each SET is a list of integer card positions)
            cards_with_positions: List of cards with their positions and bounding boxes

        Returns:
            Annotated image with SETs highlighted
        """
        annotated_image = image.copy()

        # Create a mapping from position (int) to card data for easier access
        # Ensure keys are integers for matching with sets_found
        card_position_map = {card["position"]: card for card in cards_with_positions}


        # First, draw position numbers on all cards for clarity (use the same style as the LLM prep)
        for card in cards_with_positions:
             position = card["position"]
             if position not in card_position_map: continue # Should exist, but safety check
             
             center_x, center_y = card_position_map[position]["center"]["x"], card_position_map[position]["center"]["y"]
             
             radius = 15 # Slightly smaller than the LLM prep version? Or keep consistent?
             font_scale = 0.5
             thickness = 1
             text_offset_x = int(radius * 0.5) # Adjust for centering
             text_offset_y = int(radius * 0.3)

             # White circle, Black border
             cv2.circle(annotated_image, (center_x, center_y), radius, (255, 255, 255), -1)
             cv2.circle(annotated_image, (center_x, center_y), radius, (0, 0, 0), thickness)
             # Black text
             cv2.putText(
                 annotated_image,
                 str(position),
                 (center_x - text_offset_x - (len(str(position))-1)*3 , center_y + text_offset_y), # Adjust x pos based on num digits
                 self.config.FONT,
                 font_scale,
                 (0, 0, 0), # Black text
                 thickness,
                 lineType=cv2.LINE_AA
             )


        if not sets_found:
            logging.info("No sets found to draw.")
            # Return image with only position numbers drawn
            return annotated_image

        # For each card, track which sets it belongs to (using integer positions)
        card_set_membership = defaultdict(list)
        for set_idx, set_positions in enumerate(sets_found):
            for pos_int in set_positions: # Positions here are integers
                card_set_membership[pos_int].append(set_idx)

        # Draw each card's SET memberships
        for pos_int, set_indices in card_set_membership.items():
            if pos_int not in card_position_map:
                logging.warning(f"Card position {pos_int} not found in position map during visualization.")
                continue

            card = card_position_map[pos_int]
            box = card["box"]

            # Draw a different colored box for each SET this card belongs to
            for i, set_idx in enumerate(set_indices):
                color = self.config.COLORS[set_idx % len(self.config.COLORS)]
                # Offset boxes slightly so multiple sets on one card are visible
                offset = i * self.config.OFFSET_STEP

                # Adjust box coordinates based on offset, ensure they stay within image bounds
                x1 = max(0, box["x1"] - offset)
                y1 = max(0, box["y1"] - offset)
                x2 = min(annotated_image.shape[1] - 1, box["x2"] + offset)
                y2 = min(annotated_image.shape[0] - 1, box["y2"] + offset)


                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, self.config.BOX_THICKNESS)

                # Label only the first card in each SET (using the integer position)
                # Check if the current card's integer position is the first in the set list
                if pos_int == sets_found[set_idx][0]:
                    label_text = f"Set {set_idx + 1}"
                    # Calculate text size to position it nicely above the box
                    (text_width, text_height), baseline = cv2.getTextSize(label_text, self.config.FONT, self.config.FONT_SCALE, self.config.BOX_THICKNESS)
                    
                    label_y_pos = y1 - 10 # Position above the (potentially offset) box
                    # Prevent label from going off the top edge
                    label_y_pos = max(text_height + 10, label_y_pos)
                    label_x_pos = x1

                    # Add a background rectangle for the text label for better visibility
                    cv2.rectangle(annotated_image, (label_x_pos, label_y_pos - text_height - baseline), (label_x_pos + text_width, label_y_pos + baseline), color, -1)
                    # Put text on top of the background rectangle
                    cv2.putText(
                        annotated_image,
                        label_text,
                        (label_x_pos, label_y_pos), # Use adjusted y position
                        self.config.FONT,
                        self.config.FONT_SCALE,
                        (255, 255, 255), # White text for contrast on colored background
                        self.config.BOX_THICKNESS // 2, # Thinner text thickness
                        lineType=cv2.LINE_AA
                    )


        logging.info(f"Drew {len(sets_found)} sets and position numbers on image.")
        return annotated_image


# --- Main Detector Class ---
class SetGameDetector:
    """Orchestrates the SET game detection process."""

    def __init__(self, roboflow_api_key=None, gemini_api_key=None):
        """
        Initializes the detector.

        Args:
            roboflow_api_key (Optional[str]): Roboflow API Key. If None, uses config/env.
            gemini_api_key (Optional[str]): Google Gemini API Key. If None, uses config/env.
        """
        self.config = Config(roboflow_api_key=roboflow_api_key, gemini_api_key=gemini_api_key)
        self.image_processor = ImageProcessor(self.config)
        self.card_detector = CardDetector(self.config, self.image_processor)
        self.card_classifier = CardClassifier(self.config, self.image_processor)
        self.set_logic = SetGameLogic(self.config)
        self.visualizer = Visualizer(self.config, self.image_processor)
        
        # Log API key status on initialization
        if not self.config.ROBOFLOW_API_KEY:
             logging.warning("SetGameDetector initialized without Roboflow API key.")
        if not self.config.GEMINI_API_KEY:
             logging.warning("SetGameDetector initialized without Gemini API key.")


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
                - error: Optional error message if processing failed partially or completely

        Raises:
            Specific exceptions (CardDetectionError, CardClassificationError, etc.) on failure.
            The return dict's 'error' field can be used for non-fatal issues.
        """
        logging.info("--- Starting New Image Processing Request ---")
        result = {
            "image": None,
            "sets_found": [],
            "all_cards": [],
            "error": None
        }
        annotated_image = None # Initialize

        try:
            # 1. Load and process image
            image = self.image_processor.load_image_from_bytes(image_bytes)
            logging.info(f"Image loaded successfully. Shape: {image.shape}")
            # Keep original image for drawing later if needed
            original_image = image.copy()

            # 2. Detect cards
            logging.info("Detecting cards using Roboflow...")
            cards_with_positions, image_dimensions = self.card_detector.detect_cards(image)
            logging.info(f"Detected {len(cards_with_positions)} potential cards.")

            if not cards_with_positions:
                 logging.warning("No cards detected. Cannot proceed with classification or set finding.")
                 result["error"] = "No cards were detected in the image."
                 # Still return the original (unannotated) image
                 result["image"] = self.image_processor.encode_image_to_base64(original_image)
                 return result

            # 3. Classify cards
            logging.info("Classifying cards using Gemini...")
            # Use the original image for classification if labeling is done inside classifier
            card_features = self.card_classifier.classify_cards(original_image, cards_with_positions)
            logging.info(f"Successfully classified features for {len(card_features)} cards.")

            if not card_features:
                 logging.warning("Card classification returned no features. Cannot find sets.")
                 result["error"] = "Card classification failed or returned no valid features."
                 # Draw detected positions on the image even if classification failed
                 annotated_image = self.visualizer.draw_sets_on_image(original_image, [], cards_with_positions)
                 result["image"] = self.image_processor.encode_image_to_base64(annotated_image)
                 # Include detected cards info even without features
                 result["all_cards"] = [{"position": c["position"], "features": None} for c in cards_with_positions]
                 return result


            # Update all_cards list with classified features
            # Ensure keys are integers in the final output
            result["all_cards"] = [
                {
                    "position": int(pos), # Convert string pos key back to int
                    "features": features
                } for pos, features in card_features.items()
            ]


            # 4. Find sets
            logging.info("Finding sets...")
            # Pass the dictionary with string keys as received from classifier
            sets_found_indices = self.set_logic.find_sets(card_features) # Returns list of lists of INT positions
            logging.info(f"Found {len(sets_found_indices)} sets.")


            # 5. Format sets_found output
            result["sets_found"] = [
                {
                    "set_id": i + 1,
                    "cards": [
                        {
                            "position": pos_int, # Use integer position
                            "features": card_features.get(str(pos_int)) # Lookup features using string key
                        } for pos_int in set_indices if str(pos_int) in card_features # Check feature exists
                    ]
                } for i, set_indices in enumerate(sets_found_indices)
            ]
            # Filter out sets where feature lookup might have failed (shouldn't happen with checks)
            result["sets_found"] = [s for s in result["sets_found"] if len(s["cards"]) == 3]


            # 6. Draw sets on image
            logging.info("Drawing sets on image...")
            # Use the original image and the found integer indices
            annotated_image = self.visualizer.draw_sets_on_image(original_image, sets_found_indices, cards_with_positions)


            # 7. Convert final image to base64
            result["image"] = self.image_processor.encode_image_to_base64(annotated_image)

            logging.info(f"--- Image processing completed successfully. Found {len(result['sets_found'])} sets. ---")
            return result

        except (CardDetectionError, CardClassificationError, GeminiAPIError, ImageProcessingError) as e:
            logging.error(f"Error during image processing pipeline: {e}", exc_info=True)
            result["error"] = f"Processing failed: {type(e).__name__}: {str(e)}"
            # Try to return the original image if available, otherwise None
            if 'original_image' in locals():
                 try:
                     result["image"] = self.image_processor.encode_image_to_base64(original_image)
                 except Exception as enc_e:
                     logging.error(f"Failed to encode original image on error: {enc_e}")
                     result["image"] = None # Set image to None if encoding fails during error handling
            else:
                 result["image"] = None
            # Return partial results if available, e.g., detected cards before classification failed
            if 'cards_with_positions' in locals() and not result['all_cards']:
                 result['all_cards'] = [{"position": c["position"], "features": None} for c in cards_with_positions]

            return result # Return dict with error message and potentially partial results

        except Exception as e:
            logging.error(f"An unexpected critical error occurred: {e}", exc_info=True)
            logging.error(traceback.format_exc()) # Log full traceback for unexpected errors
            result["error"] = f"An unexpected critical error occurred: {str(e)}"
            result["image"] = None # Cannot guarantee image state on unexpected error
            result["sets_found"] = []
            result["all_cards"] = []
            return result

# --- Example Usage (Optional, for testing) ---
if __name__ == '__main__':
    # This block will only run if the script is executed directly
    # You would need to provide actual API keys and an image file path for this to work

    print("Running SetGameDetector example...")

    # Load dummy keys from environment variables if not set (replace with your actual keys)
    robo_key = os.environ.get("ROBOFLOW_API_KEY")
    gem_key = os.environ.get("GEMINI_API_KEY")

    if not robo_key or not gem_key:
        print("\nWARNING: ROBOFLOW_API_KEY and/or GEMINI_API_KEY environment variables not set.")
        print("Example usage requires valid API keys.")
        # You might want to exit or provide default dummy keys for structure testing
        # sys.exit(1) # Uncomment to exit if keys are required

    # Path to a sample image file (replace with your image)
    image_path = 'path/to/your/set_game_image.jpg' # <--- CHANGE THIS

    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        print("Please update the 'image_path' variable in the script.")
        # sys.exit(1) # Uncomment to exit
    else:
        try:
            # Initialize detector (it will use keys from Config/env if None are passed)
            detector = SetGameDetector(roboflow_api_key=robo_key, gemini_api_key=gem_key)

            # Read image bytes
            with open(image_path, 'rb') as f:
                image_data = f.read()

            # Process the image
            print(f"Processing image: {image_path}")
            start_time = time.time()
            results = detector.process_image(image_data)
            end_time = time.time()
            print(f"Processing took {end_time - start_time:.2f} seconds.")

            # Print results
            if results.get("error"):
                print(f"\nProcessing Error: {results['error']}")

            print(f"\nFound {len(results.get('sets_found', []))} sets.")
            for found_set in results.get('sets_found', []):
                print(f"  Set {found_set['set_id']}: Positions {[card['position'] for card in found_set['cards']]}")
                # Optionally print features
                # for card in found_set['cards']:
                #    print(f"    Card {card['position']}: {card['features']}")


            print(f"\nDetected {len(results.get('all_cards', []))} cards in total.")
            # for card in results.get('all_cards', []):
                 # print(f" Card {card['position']}: {card['features']}") # Can be long

            # Save the annotated image (optional)
            if results.get("image"):
                try:
                    img_bytes = base64.b64decode(results["image"])
                    output_filename = "set_output_annotated.jpg"
                    with open(output_filename, 'wb') as f_out:
                        f_out.write(img_bytes)
                    print(f"\nAnnotated image saved as {output_filename}")
                except Exception as save_e:
                    print(f"\nError saving annotated image: {save_e}")
            else:
                 print("\nNo annotated image was generated (likely due to an error).")


        except Exception as main_e:
            print(f"\nAn error occurred during the example execution: {main_e}")
            print(traceback.format_exc())
