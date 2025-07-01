"""
Centralized settings and configuration for tgTrax application.

This module consolidates all configuration values, environment variables,
and constants to follow best practices and avoid magic numbers/strings
scattered throughout the codebase.
"""

import os
from typing import Optional, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==== TELEGRAM API SETTINGS ==== #
TELEGRAM_API_ID_STR: Optional[str] = os.getenv("TELEGRAM_API_ID")
TELEGRAM_API_HASH: Optional[str] = os.getenv("TELEGRAM_API_HASH")
TELEGRAM_PHONE_NUMBER: Optional[str] = os.getenv("TELEGRAM_PHONE_NUMBER")

# Parse API ID to integer with validation
TELEGRAM_API_ID: Optional[int] = None
if TELEGRAM_API_ID_STR:
    try:
        TELEGRAM_API_ID = int(TELEGRAM_API_ID_STR)
    except ValueError:
        # Error will be handled in modules that use this setting
        pass

# ==== DATABASE SETTINGS ==== #
DEFAULT_DB_NAME: str = "tgTrax.db"
DEFAULT_DB_TIMEOUT_SECONDS: float = 15.0

# ==== TRACKING SETTINGS ==== #
USER_STATUS_POLL_INTERVAL_SECONDS: int = int(
    os.getenv("USER_STATUS_POLL_INTERVAL_SECONDS", "60")
)
MINIMUM_ASSUMED_ONLINE_DURATION_SECONDS: int = int(
    os.getenv("MINIMUM_ASSUMED_ONLINE_DURATION_SECONDS", "60")
)

# Maximum number of users to poll in batch before applying delay
MAX_USERS_BEFORE_DELAY: int = 10
INTER_USER_POLL_DELAY_SECONDS: float = 0.5

# ==== CLIENT CONNECTION SETTINGS ==== #
CLIENT_CONNECTION_RETRIES: Optional[int] = None  # Retry indefinitely
CLIENT_RETRY_DELAY_SECONDS: int = 5
CLIENT_DISCONNECT_TIMEOUT_SECONDS: float = 10.0

# ==== ANALYSIS SETTINGS ==== #
DEFAULT_RESAMPLE_PERIOD: str = "1min"
DEFAULT_CORRELATION_THRESHOLD: float = 0.6
DEFAULT_JACCARD_THRESHOLD: float = 0.18

# ==== LOGGING SETTINGS ==== #
DEFAULT_LOG_LEVEL: str = "INFO"
DEFAULT_LOG_PATH: str = "tgTrax.log"

# ==== PROJECT PATHS ==== #
# These are calculated relative to this settings file location
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSIONS_DIR: str = os.path.join(PROJECT_ROOT, "sessions")
DEFAULT_SESSION_NAME: str = os.path.join(SESSIONS_DIR, "correlation_tracker_session")

# Ensure sessions directory exists
if not os.path.exists(SESSIONS_DIR):
    os.makedirs(SESSIONS_DIR)

# ==== DEBUG SETTINGS ==== #
# List of usernames for verbose status logging
TARGET_DEBUG_USERS: List[str] = [
    "metal_vuf",
    "MityaMetelitsa", 
    "kochanovigor",
    "denis_ratnikov",
    "FominaVictoria",
    "sasha_lovelle",
]

# ==== ENVIRONMENT VARIABLE KEYS ==== #
ENV_TARGET_USERS: str = "TARGET_USERS"
ENV_API_ID: str = "TELEGRAM_API_ID"
ENV_API_HASH: str = "TELEGRAM_API_HASH"
ENV_PHONE: str = "TELEGRAM_PHONE_NUMBER"