"""
Configuration management for the SET Game Detector API.
"""
import os
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Settings
    APP_NAME: str = "SET Game Detector API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Service API Keys
    ROBOFLOW_API_KEY: str = os.getenv("ROBOFLOW_API_KEY", "")
    CLAUDE_API_KEY: str = os.getenv("CLAUDE_API_KEY", "")
    
    # URLs and Service configuration
    ROBOFLOW_API_URL: str = "https://detect.roboflow.com"
    CLAUDE_API_URL: str = "https://api.anthropic.com/v1/messages"
    CLAUDE_MODEL: str = "claude-3-7-sonnet-20250219"
    
    # CORS Settings
    CORS_ORIGINS: list = ["*"]  # In production, specify the exact origins
    
    # File upload settings
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10 MB
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    Returns cached application settings.
    
    Using lru_cache to avoid loading the settings multiple times
    for the same instance of the application.
    """
    return Settings()
