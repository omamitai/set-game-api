"""
SET Game Detector - Backend Implementation
This module detects SET cards in images and identifies valid SETs using Roboflow and Google Gemini.
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
from inference_sdk import InferenceHTTPClient, RoboflowError

# Import get_settings carefully
try:
    from app.config import get_settings, Settings
except ImportError:
    logging.error("Failed to import 'get_settings' from 'app.config'. Ensure app/config.py exists and is importable.")
    # Provide a fallback mechanism or raise a critical error
    raise ImportError("Cannot start without configuration.")


# --- Logging Setup ---
# Basic config might be set in config.py or main.py, ensure logger is named
logger = logging.getLogger(__name__)


# --- Configuration Class (Simplified - reads directly from global settings) ---
class ConfigProvider:
    """Provides access to configuration settings."""
    def __init__(self):
        self._settings = get_settings()

    @property
    def settings(self) -> Settings:
        return self._settings

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

    # No config needed directly if encoding format is fixed
    # def __init__(self, config_provider: ConfigProvider):
    #     self.config_provider = config_provider

    def load_image_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Loads an image from bytes."""
        try:
            image_np = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            if image is None:
                raise ImageProcessingError("Failed to decode image bytes. Ensure it's a valid image format (JPEG, PNG, etc.).")
            logger.debug(f"Image loaded from bytes successfully. Shape: {image.shape}")
            return image
        except Exception as e:
            logger.error(f"Error loading image from bytes: {e}", exc_info=True)
            raise ImageProcessingError(f"Failed to load image: {str(e)}")

    def encode_image_to_base64(self, image: np.ndarray, format: str = '.jpg') -> str:
        """Converts an image to base64 string (default JPEG)."""
        try:
            success, img_encoded = cv2.imencode(format, image)
            if not success:
                 raise ImageProcessingError(f"Failed to encode image to {format} format.")
            return base64.b64encode(img_encoded).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image to base64: {e}", exc_info=True)
            raise ImageProcessingError(f"Failed to encode image to base64: {str(e)}")


# --- Card Detector (Uses Roboflow) ---
class CardDetector:
    """Detects cards in an image using Roboflow."""

    def __init__(self, config_provider: ConfigProvider, image_processor: ImageProcessor):
        self.config_provider = config_provider
        self.image_processor = image_processor
        self.settings = self.config_provider.settings

        if not self.settings.ROBOFLOW_API_KEY:
            logger.warning("Roboflow API key not set! Card detection will likely fail.")
            self.roboflow_client = None
        else:
            try:
                # Use the Roboflow URL from settings
                self.roboflow_client = InferenceHTTPClient(
                    api_url=self.settings.ROBOFLOW_API_URL,
                    api_key=self.settings.ROBOFLOW_API_KEY
                )
                logger.info(f"Roboflow client initialized for URL: {self.settings.ROBOFLOW_API_URL}")
            except Exception as e:
                logger.error(f"Failed to initialize Roboflow client: {e}", exc_info=True)
                self.roboflow_client = None

    def detect_cards(self, image: np.ndarray) -> tuple[list, dict]:
        """Detects cards using the configured Roboflow workflow."""
        if not self.roboflow_client:
             raise CardDetectionError("Roboflow client not initialized. Check API key and config.")

        temp_filename = None
        try:
            # Save image to a temporary file for the SDK
            temp_filename = f"/tmp/{uuid.uuid4()}.jpg" # Use /tmp which usually exists
            success = cv2.imwrite(temp_filename, image)
            if not success:
                raise ImageProcessingError(f"Failed to write temporary image file: {temp_filename}")

            logger.info(f"Calling Roboflow workflow '{self.settings.ROBOFLOW_WORKFLOW_ID}' in workspace '{self.settings.ROBOFLOW_WORKSPACE}'...")

            # Use run_workflow based on original user code structure
            result = self.roboflow_client.run_workflow(
                workspace_name=self.settings.ROBOFLOW_WORKSPACE,
                workflow_id=self.settings.ROBOFLOW_WORKFLOW_ID,
                images={"image": temp_filename},
                use_cache=True # Caching might be useful for repeated identical images
            )

            # Process workflow result (assuming standard structure)
            if not result or not isinstance(result, list) or not result[0].get("predictions"):
                 logger.error(f"Unexpected Roboflow workflow response structure: {result}")
                 raise CardDetectionError("Could not parse card predictions from Roboflow workflow response.")

            predictions_data = result[0]["predictions"]
            # Ensure 'predictions' key exists within the predictions data
            if "predictions" not in predictions_data:
                logger.error(f"Roboflow response missing inner 'predictions' list: {predictions_data}")
                raise CardDetectionError("Roboflow workflow response structure invalid (missing inner predictions list).")

            predictions = predictions_data["predictions"]
            image_dimensions = predictions_data.get("image", {"width": image.shape[1], "height": image.shape[0]})

            # Filter for "card" class (adjust class name if needed)
            card_predictions = [pred for pred in predictions if pred.get("class") == "card"]

            if not card_predictions:
                logger.warning("No 'card' class objects detected in the image by Roboflow.")
                return [], image_dimensions # Return empty list if no cards found

            cards_with_positions = self._assign_card_positions(card_predictions, image_dimensions)
            logger.info(f"Detected and positioned {len(cards_with_positions)} cards using Roboflow.")
            return cards_with_positions, image_dimensions

        except RoboflowError as e:
            logger.error(f"Roboflow API error during card detection: {e}", exc_info=True)
            raise CardDetectionError(f"Roboflow API error: {e}")
        except CardDetectionError as e:
            logger.error(f"Card detection failed: {e}", exc_info=True)
            raise
        except ImageProcessingError as e:
             logger.error(f"Image processing error during detection: {e}", exc_info=True)
             raise # Re-raise specific error
        except Exception as e:
            logger.error(f"Unexpected error during card detection: {e}", exc_info=True)
            raise CardDetectionError(f"Card detection process failed unexpectedly: {e}")
        finally:
            # Clean up temporary file
            if temp_filename and os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                    logger.debug(f"Removed temporary file: {temp_filename}")
                except Exception as e_rem:
                    logger.warning(f"Failed to remove temporary file {temp_filename}: {e_rem}")

    def _assign_card_positions(self, card_predictions: list, image_dimensions: dict) -> list:
        """Assigns position numbers (1-based) based on row/column."""
        if not card_predictions:
            return []

        cards = []
        for pred in card_predictions:
            if all(k in pred for k in ["x", "y", "width", "height"]):
                cards.append({
                    "x": pred["x"], "y": pred["y"],
                    "width": pred["width"], "height": pred["height"],
                    "detection_id": pred.get("detection_id", uuid.uuid4().hex)
                })
            else:
                logger.warning(f"Skipping Roboflow prediction due to missing keys: {pred}")
        if not cards: return []

        avg_card_height = sum(card["height"] for card in cards) / len(cards) if cards else 0
        vertical_threshold = avg_card_height * 0.5

        rows = []
        cards_sorted_by_y = sorted(cards, key=lambda c: (c["y"], c["x"]))
        if not cards_sorted_by_y: return []

        current_row = [cards_sorted_by_y[0]]
        last_y_in_row = cards_sorted_by_y[0]["y"]

        for card in cards_sorted_by_y[1:]:
            # Group by proximity to the first card's Y in the current row
            if abs(card["y"] - current_row[0]["y"]) <= vertical_threshold:
                current_row.append(card)
                # Update reference Y using average Y of the current row for stability
                last_y_in_row = sum(c['y'] for c in current_row) / len(current_row)
            else:
                current_row.sort(key=lambda c: c["x"])
                rows.append(current_row)
                current_row = [card]
                last_y_in_row = card["y"]

        if current_row:
            current_row.sort(key=lambda c: c["x"])
            rows.append(current_row)

        cards_with_positions = []
        position = 1
        img_width = image_dimensions.get("width", 0)
        img_height = image_dimensions.get("height", 0)

        for row in rows:
            for card in row:
                x_center, y_center = card["x"], card["y"]
                width, height = card["width"], card["height"]
                x1 = max(0, int(x_center - width / 2))
                y1 = max(0, int(y_center - height / 2))
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                if img_width > 0: x2 = min(img_width - 1, x2)
                if img_height > 0: y2 = min(img_height - 1, y2)

                cards_with_positions.append({
                    "position": position,
                    "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "center": {"x": int(x_center), "y": int(y_center)},
                    "detection_id": card["detection_id"]
                })
                position += 1
        return cards_with_positions

# --- Card Classifier (Uses Google Gemini) ---
class CardClassifier:
    """Classifies card features using Google Gemini API."""

    def __init__(self, config_provider: ConfigProvider, image_processor: ImageProcessor):
        self.config_provider = config_provider
        self.image_processor = image_processor
        self.settings = self.config_provider.settings

        # Define valid features based on ConfigProvider/Settings
        self.FEATURE_TYPES = {
            'number': ['1', '2', '3'],
            'color': ['red', 'green', 'purple'],
            'shape': ['oval', 'diamond', 'squiggle'],
            'shading': ['solid', 'striped', 'outline']
        }

    def classify_cards(self, image: np.ndarray, cards_with_positions: list) -> dict:
        """Classifies SET card features using Gemini."""
        if not self.settings.GEMINI_API_KEY:
            # This check is crucial and should reference GEMINI_API_KEY
            raise CardClassificationError("Gemini API key is required for card classification.")
        if not cards_with_positions:
             logger.warning("No cards provided to classify.")
             return {}

        return self._classify_cards_with_gemini(image, cards_with_positions)

    def _prepare_labeled_image_for_llm(self, image: np.ndarray, cards_with_positions: list) -> np.ndarray:
        """Creates a labeled version of the image for the LLM."""
        labeled_image = image.copy()
        radius = 20
        font_scale = 0.7
        thickness = 2

        for card in cards_with_positions:
            position = card["position"]
            center_x, center_y = card["center"]["x"], card["center"]["y"]

            text_size, _ = cv2.getTextSize(str(position), cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_offset_x = text_size[0] // 2
            text_offset_y = text_size[1] // 2

            cv2.circle(labeled_image, (center_x, center_y), radius, (0, 0, 0), -1) # Black circle
            cv2.putText(
                labeled_image, str(position),
                (center_x - text_offset_x, center_y + text_offset_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
            )
        return labeled_image

    def _classify_cards_with_gemini(self, image: np.ndarray, cards_with_positions: list) -> dict:
        """Sends request to Gemini API and parses the response."""
        try:
            labeled_image = self._prepare_labeled_image_for_llm(image, cards_with_positions)
            img_base64 = self.image_processor.encode_image_to_base64(labeled_image, format='.jpg') # Use JPEG

            api_url = self.settings.GEMINI_API_URL_TEMPLATE.format(
                model=self.settings.GEMINI_MODEL,
                api_key=self.settings.GEMINI_API_KEY
            )
            headers = {"Content-Type": "application/json"}

            position_details = [
                f"Card {card['position']}: Near the black circle labeled '{card['position']}'."
                for card in cards_with_positions
            ]
            position_info = "\n".join(position_details)

            prompt = (
                f"Analyze the image containing SET cards. Each card of interest is marked with a black circle containing a white number (position label).\n"
                f"There are {len(cards_with_positions)} marked cards:\n{position_info}\n\n"
                f"For EACH numbered card, identify its 4 features:\n"
                f"1. Number: Count of shapes (1, 2, or 3).\n"
                f"2. Color: Shape color (red, green, or purple).\n"
                f"3. Shape: Type of shape (oval, diamond, or squiggle).\n"
                f"4. Shading: Shape filling (solid, striped, or outline).\n\n"
                f"Respond ONLY with a single JSON object. Do not use markdown ```json. Do not add explanations.\n"
                f"The JSON object's keys must be the card position numbers (as strings). Each value must be an object with the 4 features (number, color, shape, shading) using lowercase strings for the values (e.g., 'red', 'oval', 'solid').\n"
                f"Example for card 1: \"1\": {{\"number\": \"1\", \"color\": \"green\", \"shape\": \"diamond\", \"shading\": \"outline\"}}"
                f"Provide this structure for all {len(cards_with_positions)} cards."
            )

            data = {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": img_base64}}
                    ]
                }],
                 "generationConfig": { # Added config for better JSON control
                    "responseMimeType": "application/json", # Request JSON output directly
                    "temperature": 0.1 # Low temperature for deterministic output
                 }
            }

            logger.info(f"Sending request to Gemini API ({self.settings.GEMINI_MODEL})...")
            response = requests.post(api_url, headers=headers, json=data, timeout=90) # Increased timeout
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

            response_json = response.json()
            logger.info("Received response from Gemini API.")

            # --- Parse Gemini Response ---
            if "candidates" not in response_json or not response_json["candidates"]:
                feedback = response_json.get("promptFeedback")
                if feedback and feedback.get("blockReason"):
                    raise GeminiAPIError(f"Gemini API request blocked due to: {feedback['blockReason']}")
                raise GeminiAPIError("Gemini API response missing 'candidates'.")

            candidate = response_json["candidates"][0]
            if "content" not in candidate or "parts" not in candidate["content"] or not candidate["content"]["parts"]:
                raise GeminiAPIError("Gemini response candidate missing content parts.")

            # Since we requested JSON, the text part should contain the JSON string
            generated_text = candidate["content"]["parts"][0].get("text", "")
            if not generated_text:
                raise GeminiAPIError("Gemini response text part is empty.")
            logger.debug(f"Raw Gemini response text (expected JSON):\n{generated_text}")

            try:
                # Parse the JSON string directly
                classification_results = json.loads(generated_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Gemini's JSON response. Error: {e}. Raw text:\n{generated_text}", exc_info=True)
                raise CardClassificationError(f"Failed to parse Gemini's response as JSON. Raw text: {generated_text[:500]}...")

            # --- Validate Parsed Results ---
            validated_results = {}
            expected_features = set(self.FEATURE_TYPES.keys())
            valid_positions = {str(card['position']) for card in cards_with_positions}

            if not isinstance(classification_results, dict):
                 raise CardClassificationError(f"Gemini response was valid JSON but not a dictionary: {type(classification_results)}")

            for pos_str, features in classification_results.items():
                if pos_str not in valid_positions:
                    logger.warning(f"Gemini returned classification for unexpected position '{pos_str}'. Ignoring.")
                    continue
                if not isinstance(features, dict):
                    logger.warning(f"Features for position '{pos_str}' is not a dictionary: {features}. Skipping.")
                    continue
                if set(features.keys()) != expected_features:
                    logger.warning(f"Features for pos '{pos_str}' have incorrect keys: {features.keys()}. Expected: {expected_features}. Skipping.")
                    continue

                # Validate feature values
                is_valid = True
                for feature_name, value in features.items():
                     allowed_values = self.FEATURE_TYPES.get(feature_name)
                     # Check if value is a string and in the allowed list
                     if not isinstance(value, str) or value not in allowed_values:
                         logger.warning(f"Invalid value '{value}' (type: {type(value)}) for feature '{feature_name}' at position '{pos_str}'. Allowed: {allowed_values}. Skipping card.")
                         is_valid = False
                         break
                if is_valid:
                    validated_results[pos_str] = features
            # -----------------------------

            if not validated_results:
                 # This can happen if Gemini returns JSON but it's empty or fails validation
                 logger.error(f"Gemini response parsed as JSON but yielded no valid card classifications after validation. Parsed JSON: {classification_results}")
                 raise CardClassificationError("Gemini response parsed, but no valid card classifications found after validation.")

            logger.info(f"Card classification successful using Gemini. Validated {len(validated_results)} cards.")
            return validated_results

        except requests.exceptions.Timeout:
            logger.error("Gemini API request timed out.", exc_info=True)
            raise GeminiAPIError("Gemini API request timed out.")
        except requests.exceptions.HTTPError as e:
            error_body = e.response.text if e.response else "No response body"
            logger.error(f"Gemini API HTTP error: {e}. Status: {e.response.status_code}. Body: {error_body}", exc_info=True)
            raise GeminiAPIError(f"Gemini API HTTP error: {e.response.status_code}. Check API key and quota. Details: {error_body[:500]}...")
        except requests.exceptions.RequestException as e:
            logger.error(f"Gemini API request failed: {e}", exc_info=True)
            raise GeminiAPIError(f"Gemini API request failed: {e}")
        except (GeminiAPIError, CardClassificationError, ImageProcessingError) as e:
             # Re-raise specific known errors
             logger.error(f"Classification failed: {e}", exc_info=True)
             raise
        except Exception as e:
            logger.error(f"Unexpected error during Gemini classification: {e}", exc_info=True)
            raise CardClassificationError(f"Unexpected error during Gemini classification: {e}")


# --- SET Game Logic ---
class SetGameLogic:
    """Implements the logic to find SETs."""

    def __init__(self, config_provider: ConfigProvider):
        self.config_provider = config_provider
        # Get feature types from config provider settings
        self.FEATURE_TYPES = {
            'number': ['1', '2', '3'],
            'color': ['red', 'green', 'purple'],
            'shape': ['oval', 'diamond', 'squiggle'],
            'shading': ['solid', 'striped', 'outline']
        } # Should ideally come from settings if configurable

    def is_set(self, cards: list[dict]) -> bool:
        """Checks if three cards form a valid SET."""
        if len(cards) != 3: return False
        try:
            for feature in self.FEATURE_TYPES.keys():
                values = {card[feature] for card in cards} # Use set for efficiency
                # A SET requires all same (len=1) or all different (len=3)
                if len(values) == 2:
                    return False
            return True # All features passed
        except KeyError as e:
            logger.warning(f"Feature '{e}' missing in one of the cards during SET check. Cards: {cards}")
            return False
        except Exception as e:
             logger.error(f"Error during is_set check: {e}", exc_info=True)
             return False


    def find_sets(self, card_features: dict) -> list[list[int]]:
        """Finds all valid SETs from classified card features."""
        sets_found = []
        # Positions are string keys from classification ('1', '2', ...)
        positions = list(card_features.keys())
        if len(positions) < 3: return []

        for combo_pos_str in combinations(positions, 3):
            try:
                cards_in_combo = [card_features[pos_str] for pos_str in combo_pos_str]
                if self.is_set(cards_in_combo):
                    # Convert string positions to integers for the final result
                    sets_found.append(sorted(list(map(int, combo_pos_str)))) # Store sorted int positions
            except KeyError as e:
                logger.warning(f"Card position key missing during set finding: {e}. Combo: {combo_pos_str}. Skipping.")
                continue
            except Exception as e:
                 logger.error(f"Unexpected error checking combination {combo_pos_str}: {e}", exc_info=True)
                 continue

        logger.info(f"Found {len(sets_found)} sets from {len(positions)} classified cards.")
        return sets_found


# --- Visualizer ---
class Visualizer:
    """Handles visualization of detected sets on the image."""

    def __init__(self, config_provider: ConfigProvider, image_processor: ImageProcessor):
        self.config_provider = config_provider
        self.image_processor = image_processor
        # Visualization parameters (could also come from config)
        self.COLORS = [ (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                       (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
                       (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128) ]
        self.BOX_THICKNESS = 4 # Reduced thickness
        self.FONT_SCALE = 0.7 # Adjusted scale
        self.FONT = cv2.FONT_HERSHEY_SIMPLEX
        self.OFFSET_STEP = 6 # Reduced offset

    def draw_sets_on_image(self, image: np.ndarray, sets_found: list[list[int]],
                           cards_with_positions: list) -> np.ndarray:
        """Draws bounding boxes and labels for detected sets."""
        annotated_image = image.copy()
        card_position_map = {card["position"]: card for card in cards_with_positions} # Int key

        # Draw position numbers first (using the style for LLM prep)
        radius = 15 # Smaller radius for final viz
        font_scale_pos = 0.5
        thickness_pos = 1
        for card in cards_with_positions:
            pos_int = card["position"]
            center_x, center_y = card["center"]["x"], card["center"]["y"]
            text = str(pos_int)
            text_size, _ = cv2.getTextSize(text, self.FONT, font_scale_pos, thickness_pos)
            text_offset_x = text_size[0] // 2
            text_offset_y = text_size[1] // 2
            cv2.circle(annotated_image, (center_x, center_y), radius, (255, 255, 255), -1) # White circle
            cv2.circle(annotated_image, (center_x, center_y), radius, (0, 0, 0), thickness_pos) # Black border
            cv2.putText(annotated_image, text, (center_x - text_offset_x, center_y + text_offset_y),
                        self.FONT, font_scale_pos, (0, 0, 0), thickness_pos, cv2.LINE_AA)

        if not sets_found:
            logger.info("No sets found to draw.")
            return annotated_image # Return image with just position numbers

        card_set_membership = defaultdict(list)
        for set_idx, set_positions_int in enumerate(sets_found): # Positions are ints
            for pos_int in set_positions_int:
                card_set_membership[pos_int].append(set_idx)

        for pos_int, set_indices in card_set_membership.items():
            if pos_int not in card_position_map:
                logger.warning(f"Card position {pos_int} not found in map during visualization.")
                continue

            card = card_position_map[pos_int]
            box = card["box"]

            for i, set_idx in enumerate(sorted(set_indices)): # Sort indices for consistent offset order
                color = self.COLORS[set_idx % len(self.COLORS)]
                offset = i * self.OFFSET_STEP

                # Calculate offset box, ensuring bounds
                x1 = max(0, box["x1"] - offset)
                y1 = max(0, box["y1"] - offset)
                x2 = min(annotated_image.shape[1] - 1, box["x2"] + offset)
                y2 = min(annotated_image.shape[0] - 1, box["y2"] + offset)

                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, self.BOX_THICKNESS)

                # Label only the first card in each SET (using sorted int positions)
                if pos_int == sets_found[set_idx][0]: # sets_found contains sorted lists
                    label_text = f"Set {set_idx + 1}"
                    (tw, th), baseline = cv2.getTextSize(label_text, self.FONT, self.FONT_SCALE, self.BOX_THICKNESS // 2)
                    label_y = max(y1 - 10, th + 10) # Position above box, ensure space
                    label_x = x1

                    # Text background
                    cv2.rectangle(annotated_image, (label_x, label_y - th - baseline), (label_x + tw, label_y + baseline), color, -1)
                    # Text
                    cv2.putText(annotated_image, label_text, (label_x, label_y), self.FONT,
                                self.FONT_SCALE, (255, 255, 255), self.BOX_THICKNESS // 2, cv2.LINE_AA)

        logger.info(f"Drew {len(sets_found)} sets and position numbers on image.")
        return annotated_image


# --- Main Detector Class (Orchestrator) ---
class SetGameDetector:
    """Orchestrates the SET game detection process."""

    def __init__(self):
        """Initializes all components."""
        self.config_provider = ConfigProvider() # Single source of truth for config
        self.image_processor = ImageProcessor() # Doesn't need config directly
        self.card_detector = CardDetector(self.config_provider, self.image_processor)
        self.card_classifier = CardClassifier(self.config_provider, self.image_processor)
        self.set_logic = SetGameLogic(self.config_provider)
        self.visualizer = Visualizer(self.config_provider, self.image_processor)
        logger.info("SetGameDetector initialized.")

    def process_image(self, image_bytes: bytes) -> dict:
        """Processes an image to detect SET cards and find valid SETs."""
        logger.info("--- Starting New Image Processing Request ---")
        start_time_total = time.time()
        result = { "image": None, "sets_found": [], "all_cards": [], "error": None }
        annotated_image = None
        original_image = None

        try:
            # 1. Load Image
            start_time = time.time()
            original_image = self.image_processor.load_image_from_bytes(image_bytes)
            logger.info(f"Image loaded ({time.time() - start_time:.2f}s).")

            # 2. Detect Cards
            start_time = time.time()
            cards_with_positions, image_dimensions = self.card_detector.detect_cards(original_image)
            logger.info(f"Card detection completed ({time.time() - start_time:.2f}s). Found {len(cards_with_positions)} cards.")
            if not cards_with_positions:
                 result["error"] = "No cards detected by Roboflow."
                 # Return original image (maybe with a "no cards" message?)
                 annotated_image = original_image # No annotations yet
                 result["image"] = self.image_processor.encode_image_to_base64(annotated_image)
                 return result

            # 3. Classify Cards
            start_time = time.time()
            # Pass original image, classifier will add labels internally for LLM
            card_features = self.card_classifier.classify_cards(original_image, cards_with_positions)
            logger.info(f"Card classification completed ({time.time() - start_time:.2f}s). Classified {len(card_features)} cards.")
            if not card_features:
                result["error"] = "Card classification failed or returned no valid features."
                # Draw only positions if classification failed
                annotated_image = self.visualizer.draw_sets_on_image(original_image, [], cards_with_positions)
                result["image"] = self.image_processor.encode_image_to_base64(annotated_image)
                result["all_cards"] = [{"position": c["position"], "features": None} for c in cards_with_positions]
                return result

            # Format all_cards for output (int position key)
            result["all_cards"] = [
                { "position": int(pos_str), "features": features }
                for pos_str, features in card_features.items()
            ]

            # 4. Find Sets
            start_time = time.time()
            # Use features dict with string keys ('1', '2',...)
            sets_found_indices = self.set_logic.find_sets(card_features) # Returns list of lists of INT positions
            logger.info(f"Set finding completed ({time.time() - start_time:.2f}s). Found {len(sets_found_indices)} sets.")

            # Format sets_found for output
            result["sets_found"] = [
                {
                    "set_id": i + 1,
                    "cards": [
                        { "position": pos_int, "features": card_features.get(str(pos_int)) }
                        for pos_int in set_indices if str(pos_int) in card_features # Safety check
                    ]
                } for i, set_indices in enumerate(sets_found_indices)
            ]
            result["sets_found"] = [s for s in result["sets_found"] if len(s["cards"]) == 3] # Ensure valid sets

            # 5. Draw Results
            start_time = time.time()
            # Use original image, int indices list, and original card positions list
            annotated_image = self.visualizer.draw_sets_on_image(original_image, sets_found_indices, cards_with_positions)
            logger.info(f"Drawing completed ({time.time() - start_time:.2f}s).")

            # 6. Encode final image
            result["image"] = self.image_processor.encode_image_to_base64(annotated_image)

            total_time = time.time() - start_time_total
            logger.info(f"--- Image processing successful ({total_time:.2f}s). Found {len(result['sets_found'])} sets. ---")
            return result

        except (CardDetectionError, CardClassificationError, GeminiAPIError, ImageProcessingError) as e:
            logger.error(f"Pipeline error: {type(e).__name__}: {e}", exc_info=True)
            result["error"] = f"Processing failed: {type(e).__name__}: {str(e)}"
            # Try to encode original image if available
            if original_image is not None:
                try:
                    result["image"] = self.image_processor.encode_image_to_base64(original_image)
                except Exception as enc_e:
                    logger.error(f"Failed to encode original image on error: {enc_e}")
            return result # Return dict with error and maybe original image

        except Exception as e:
            logger.critical(f"Unexpected critical error in process_image: {e}", exc_info=True)
            result["error"] = f"An unexpected server error occurred: {str(e)}"
            # Don't return image data on unknown critical errors
            result["image"] = None
            result["sets_found"] = []
            result["all_cards"] = []
            return result
