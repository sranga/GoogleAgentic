"""
Configuration

Central configuration for the V-Access system.
For production, use environment variables or a secure configuration service.
"""

import os
from typing import Dict, Any


def get_config() -> Dict[str, Any]:
    """
    Get configuration from environment variables with defaults.

    Returns:
        Configuration dictionary
    """
    return {
        # Model Configuration
        "model": os.getenv("VACCESS_MODEL", "gemini-2.0-flash"),

        # Agent Settings
        "max_candidates": int(os.getenv("VACCESS_MAX_CANDIDATES", "5")),
        "search_radius_km": int(os.getenv("VACCESS_SEARCH_RADIUS_KM", "10")),
        "max_retries": int(os.getenv("VACCESS_MAX_RETRIES", "3")),
        "max_history_length": int(os.getenv("VACCESS_MAX_HISTORY", "8")),

        # Follow-up Settings
        "followup_seconds": int(os.getenv("VACCESS_FOLLOWUP_SECONDS", "86400")),  # 24 hours

        # Security Settings
        "secure_data_dir": os.getenv("VACCESS_SECURE_DATA_DIR", "./secure_data"),
        "encryption_key_file": os.getenv("VACCESS_ENCRYPTION_KEY_FILE", ".encryption_key"),

        # Rate Limiting
        "rate_limit_requests": int(os.getenv("VACCESS_RATE_LIMIT_REQUESTS", "100")),
        "rate_limit_window_seconds": int(os.getenv("VACCESS_RATE_LIMIT_WINDOW", "60")),
        "rate_limit_burst": int(os.getenv("VACCESS_RATE_LIMIT_BURST", "150")),

        # Session Settings
        "session_ttl_seconds": int(os.getenv("VACCESS_SESSION_TTL", "3600")),

        # Tools Configuration
        "tools_enabled": os.getenv("VACCESS_TOOLS_ENABLED", "true").lower() == "true",

        # Logging
        "log_level": os.getenv("VACCESS_LOG_LEVEL", "INFO"),
        "log_json": os.getenv("VACCESS_LOG_JSON", "true").lower() == "true",
    }


# Default configuration instance
CONFIG = get_config()


# Agent-specific configurations
VACCINE_INFO_CONFIG = {
    "model": CONFIG["model"],
    "max_history_length": CONFIG["max_history_length"],
}

CLINIC_FINDER_CONFIG = {
    "model": CONFIG["model"],
    "max_candidates": CONFIG["max_candidates"],
    "search_radius_km": CONFIG["search_radius_km"],
    "tools_enabled": CONFIG["tools_enabled"],
}

APPOINTMENT_CONFIG = {
    "model": CONFIG["model"],
    "max_retries": CONFIG["max_retries"],
}

FOLLOWUP_CONFIG = {
    "model": CONFIG["model"],
    "followup_seconds": CONFIG["followup_seconds"],
}

ANALYTICS_CONFIG = {
    "model": CONFIG["model"],
}


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration values.

    Args:
        config: Configuration dictionary

    Returns:
        True if valid

    Raises:
        ValueError: If configuration is invalid
    """
    required_keys = ["model"]

    for key in required_keys:
        if not config.get(key):
            raise ValueError(f"Missing required configuration: {key}")

    if config.get("max_candidates", 0) < 1:
        raise ValueError("max_candidates must be at least 1")

    if config.get("max_retries", 0) < 1:
        raise ValueError("max_retries must be at least 1")

    if config.get("followup_seconds", 0) < 0:
        raise ValueError("followup_seconds must be non-negative")

    return True


# Validate default configuration on import
validate_config(CONFIG)