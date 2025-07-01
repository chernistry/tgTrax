#!/usr/bin/env python3
"""
Main entry point for tgTrax application.

This script provides command-line interface for running the tracker and analyzer
components of tgTrax. It handles configuration file creation, environment setup,
and command routing.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add the project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import settings
from core.tracker import main_tracker_entrypoint
from core.analysis import TemporalAnalyzer
from core.database import SQLiteDatabase
from core.db_manager import get_async_db_manager, close_db_manager
from core.models import TrackingConfig, TelegramConfig
from utils import tui
from utils.logger_adapter import TuiLoggerAdapter

# Initialize logger
logger = TuiLoggerAdapter(tui)


def create_default_config() -> Dict[str, Any]:
    """Create default configuration dictionary."""
    return {
        "telegram_session_name": "tgTrax_session",
        "database_path": settings.DEFAULT_DB_NAME,
        "log_level": settings.DEFAULT_LOG_LEVEL,
        "log_path": settings.DEFAULT_LOG_PATH,
        "scan_user_batch_size": 10,
        "scan_user_delay_seconds": 5
    }


def create_default_env() -> str:
    """Create default .env file content."""
    return """# Telegram API credentials
# Get these from https://my.telegram.org/apps
TELEGRAM_API_ID=your_api_id_here
TELEGRAM_API_HASH=your_api_hash_here

# Optional: Your phone number for authentication
TELEGRAM_PHONE_NUMBER=+1234567890

# Tracking configuration
USER_STATUS_POLL_INTERVAL_SECONDS=60
MINIMUM_ASSUMED_ONLINE_DURATION_SECONDS=60

# Target users to track (comma-separated)
TARGET_USERS=username1,username2,username3
"""


def ensure_config_files() -> None:
    """Ensure configuration files exist with default values."""
    config_file = "tgTrax_config.json"
    env_file = ".env"
    
    # Create config file if it doesn't exist
    if not os.path.exists(config_file):
        tui.tui_print_info(f"Creating default configuration file: {config_file}")
        try:
            with open(config_file, 'w') as f:
                json.dump(create_default_config(), f, indent=4)
            tui.tui_print_success(f"Created {config_file}")
        except Exception as e:
            tui.tui_print_error(f"Failed to create {config_file}: {e}")
    
    # Create .env file if it doesn't exist
    if not os.path.exists(env_file):
        tui.tui_print_info(f"Creating default environment file: {env_file}")
        try:
            with open(env_file, 'w') as f:
                f.write(create_default_env())
            tui.tui_print_success(f"Created {env_file}")
            tui.tui_print_warning(
                f"Please edit {env_file} with your actual Telegram API credentials!"
            )
        except Exception as e:
            tui.tui_print_error(f"Failed to create {env_file}: {e}")


def validate_telegram_config() -> bool:
    """Validate Telegram API configuration."""
    if not settings.TELEGRAM_API_ID or not settings.TELEGRAM_API_HASH:
        tui.tui_print_error(
            "TELEGRAM_API_ID and TELEGRAM_API_HASH must be set in .env file"
        )
        return False
    
    try:
        # Validate using Pydantic model
        TelegramConfig(
            api_id=settings.TELEGRAM_API_ID,
            api_hash=settings.TELEGRAM_API_HASH,
            phone_number=settings.TELEGRAM_PHONE_NUMBER,
            session_name=settings.DEFAULT_SESSION_NAME
        )
        return True
    except Exception as e:
        tui.tui_print_error(f"Invalid Telegram configuration: {e}")
        return False


async def run_tracker(args: argparse.Namespace) -> None:
    """Run the Telegram activity tracker."""
    tui.tui_print_highlight("Starting Telegram Activity Tracker", style="header")
    
    # Validate configuration
    if not validate_telegram_config():
        return
    
    # Get target users
    target_users_env = os.getenv(settings.ENV_TARGET_USERS)
    if not target_users_env:
        tui.tui_print_error(
            f"No target users specified. Set {settings.ENV_TARGET_USERS} in .env file"
        )
        return
    
    # Validate target users using Pydantic
    try:
        target_users_list = [u.strip() for u in target_users_env.split(",") if u.strip()]
        tracking_config = TrackingConfig(
            target_usernames=target_users_list,
            poll_interval_seconds=settings.USER_STATUS_POLL_INTERVAL_SECONDS,
            min_online_duration_seconds=settings.MINIMUM_ASSUMED_ONLINE_DURATION_SECONDS
        )
        tui.tui_print_info(f"Tracking {len(tracking_config.target_usernames)} users")
    except Exception as e:
        tui.tui_print_error(f"Invalid tracking configuration: {e}")
        return
    
    # Initialize async database manager
    try:
        db_manager = await get_async_db_manager()
        tui.tui_print_success("Database manager initialized")
    except Exception as e:
        tui.tui_print_error(f"Failed to initialize database: {e}")
        return
    
    # Run the tracker
    try:
        await main_tracker_entrypoint(target_users_env)
    except KeyboardInterrupt:
        tui.tui_print_info("Tracker interrupted by user")
    except Exception as e:
        logger.error(f"Tracker error: {e}", exc_info=True)
    finally:
        close_db_manager()
        tui.tui_print_success("Tracker shutdown complete")


def run_analyzer(args: argparse.Namespace) -> None:
    """Run the correlation analyzer."""
    tui.tui_print_highlight("Starting Correlation Analyzer", style="header")
    
    # Initialize database
    db_path = os.path.join(settings.PROJECT_ROOT, settings.DEFAULT_DB_NAME)
    if not os.path.exists(db_path):
        tui.tui_print_error(f"Database not found: {db_path}")
        tui.tui_print_info("Run the tracker first to collect data")
        return
    
    try:
        db = SQLiteDatabase(db_path)
        
        if args.demo:
            tui.tui_print_info("Running in demo mode with sample data")
            # TODO: Implement demo data generation
            tui.tui_print_warning("Demo mode not yet implemented")
            return
        
        # Get target users from environment
        target_users_env = os.getenv(settings.ENV_TARGET_USERS)
        if not target_users_env:
            tui.tui_print_error("No target users specified in .env file")
            return
        
        target_users = [u.strip() for u in target_users_env.split(",") if u.strip()]
        
        # Fetch activity data
        tui.tui_print_info(f"Fetching activity data for {len(target_users)} users...")
        activity_data = db.get_all_activity_for_users(target_users)
        
        if not activity_data:
            tui.tui_print_warning("No activity data found in database")
            return
        
        tui.tui_print_info(f"Found {len(activity_data)} activity records")
        
        # TODO: Implement full analysis workflow
        # This would include:
        # 1. Convert to DataFrame
        # 2. Create TemporalAnalyzer
        # 3. Calculate correlations
        # 4. Generate reports
        # 5. Create visualizations
        
        tui.tui_print_warning("Full analysis implementation pending")
        
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
    finally:
        if 'db' in locals():
            db.close()


def main() -> None:
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="tgTrax - Telegram Activity Tracking and Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py tracker          # Start activity tracking
  python main.py analyze          # Run correlation analysis
  python main.py analyze --demo   # Run with demo data
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Tracker command
    tracker_parser = subparsers.add_parser(
        'tracker', 
        help='Start Telegram activity tracker'
    )
    tracker_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # Analyzer command
    analyzer_parser = subparsers.add_parser(
        'analyze',
        help='Run correlation analysis'
    )
    analyzer_parser.add_argument(
        '--demo',
        action='store_true',
        help='Run with demo/sample data'
    )
    analyzer_parser.add_argument(
        '--output', '-o',
        help='Output file for analysis results'
    )
    
    args = parser.parse_args()
    
    # Ensure config files exist
    ensure_config_files()
    
    # Route commands
    if args.command == 'tracker':
        try:
            asyncio.run(run_tracker(args))
        except KeyboardInterrupt:
            tui.tui_print_info("Application interrupted by user")
        except Exception as e:
            logger.error(f"Application error: {e}", exc_info=True)
            sys.exit(1)
    
    elif args.command == 'analyze':
        try:
            run_analyzer(args)
        except KeyboardInterrupt:
            tui.tui_print_info("Analysis interrupted by user")
        except Exception as e:
            logger.error(f"Analysis error: {e}", exc_info=True)
            sys.exit(1)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()