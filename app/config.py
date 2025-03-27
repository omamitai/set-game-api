"""
Configuration management for the SET Game Detector API.
"""
import os
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List # Import List type hint


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Settings
    APP_NAME: str = "SET Game Detector API"
    APP_VERSION: str = "1.0.1" # Incremented version
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # Service API Keys
    ROBOFLOW_API_KEY: str = os.getenv("ROBOFLOW_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "") # Changed from CLAUDE_API_KEY

    # URLs and Service configuration
    # Roboflow URL (use infer endpoint for SDK)
    ROBOFLOW_API_URL: str = os.getenv("ROBOFLOW_API_URL", "https://infer.roboflow.com")
    ROBOFLOW_WORKSPACE: str = os.getenv("ROBOFLOW_WORKSPACE", "tel-aviv") # Added
    ROBOFLOW_WORKFLOW_ID: str = os.getenv("ROBOFLOW_WORKFLOW_ID", "custom-workflow") # Added

    # Gemini Configuration (Changed from Claude)
    GEMINI_API_URL_TEMPLATE: str = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest") # Using 1.5 flash as default

    # CORS Settings
    # Use environment variable for origins in production, default to wildcard for dev
    CORS_ORIGINS: List[str] = os.getenv("CORS_ORIGINS", "*").split(",")

    # File upload settings
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10 MB

    class Config:
        env_file = ".env" # Load .env file if present (for local dev)
        case_sensitive = True # Environment variables are typically case-sensitive


@lru_cache()
def get_settings() -> Settings:
    """
    Returns cached application settings.

    Using lru_cache to avoid loading the settings multiple times
    for the same instance of the application.
    """
    settings = Settings()
    # Log loaded settings (optional, careful with keys in production logs)
    # logging.info(f"Loaded settings: DEBUG={settings.DEBUG}, CORS_ORIGINS={settings.CORS_ORIGINS}")
    # Basic check for essential keys
    if not settings.ROBOFLOW_API_KEY:
        logging.warning("ROBOFLOW_API_KEY is not set.")
    if not settings.GEMINI_API_KEY:
        logging.warning("GEMINI_API_KEY is not set.")
    return settings

# Initialize logging basic config here to ensure it's set early
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
