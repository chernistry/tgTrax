#!/usr/bin/env python3
"""
Main entry point for tgTrax - Telegram Activity Analyzer.

This script provides a command-line interface for the tgTrax application,
supporting tracker and analysis operations as referenced by run.sh.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

# Import tgTrax modules
try:
    from core.tracker import CorrelationTracker
    from core.analysis import TemporalAnalyzer, create_activity_gantt_chart
    from core.database import SQLiteDatabase
    from utils import tui
    from utils.logger_adapter import TuiLoggerAdapter
except ImportError as e:
    print(f"Error importing tgTrax modules: {e}")
    print("Ensure you're running from the project root and all dependencies are installed.")
    sys.exit(1)

# Initialize logger
logger = TuiLoggerAdapter(tui)


def create_default_env_file() -> None:
    """Creates a default .env file with template values if it doesn't exist."""
    env_path = Path(".env")
    if env_path.exists():
        tui.tui_print_info("Environment file .env already exists.")
        return
    
    env_template = """# Telegram API Credentials
# Get these from https://my.telegram.org/apps
TELEGRAM_API_ID=your_api_id_here
TELEGRAM_API_HASH=your_api_hash_here

# Optional: Your phone number (for non-interactive login)
# TELEGRAM_PHONE_NUMBER=+1234567890

# Configuration
USER_STATUS_POLL_INTERVAL_SECONDS=60
MINIMUM_ASSUMED_ONLINE_DURATION_SECONDS=60
"""
    
    try:
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write(env_template)
        tui.tui_print_success(f"Created default .env file: {env_path}")
        tui.tui_print_highlight(
            "IMPORTANT: Please edit .env and add your Telegram API credentials "
            "from https://my.telegram.org/apps"
        )
    except Exception as e:
        tui.tui_print_error(f"Failed to create .env file: {e}")


def create_default_config_file() -> None:
    """Creates a default tgTrax_config.json file if it doesn't exist."""
    config_path = Path("tgTrax_config.json")
    if config_path.exists():
        tui.tui_print_info("Configuration file tgTrax_config.json already exists.")
        return
    
    default_config = {
        "telegram_session_name": "tgTrax_session",
        "database_path": "tgTrax.db",
        "log_level": "INFO",
        "log_path": "tgTrax.log",
        "scan_user_batch_size": 10,
        "scan_user_delay_seconds": 5,
        "target_users": [
            "example_user1",
            "example_user2"
        ],
        "analysis": {
            "resample_period": "1min",
            "correlation_threshold": 0.6,
            "jaccard_threshold": 0.18
        }
    }
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)
        tui.tui_print_success(f"Created default config file: {config_path}")
        tui.tui_print_highlight(
            "Please edit tgTrax_config.json to configure target users and settings."
        )
    except Exception as e:
        tui.tui_print_error(f"Failed to create config file: {e}")


def load_config() -> Dict:
    """Loads configuration from tgTrax_config.json."""
    config_path = Path("tgTrax_config.json")
    if not config_path.exists():
        tui.tui_print_warning("Configuration file not found. Creating default...")
        create_default_config_file()
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        tui.tui_print_info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        tui.tui_print_error(f"Failed to load configuration: {e}")
        return {}


def validate_environment() -> bool:
    """Validates that required environment variables are set."""
    load_dotenv()
    
    api_id = os.getenv("TELEGRAM_API_ID")
    api_hash = os.getenv("TELEGRAM_API_HASH")
    
    if not api_id or not api_hash:
        tui.tui_print_error(
            "TELEGRAM_API_ID and TELEGRAM_API_HASH must be set in .env file."
        )
        return False
    
    try:
        int(api_id)
    except ValueError:
        tui.tui_print_error("TELEGRAM_API_ID must be a valid integer.")
        return False
    
    return True


async def run_tracker(target_users: List[str], config: Dict) -> None:
    """Runs the Telegram activity tracker."""
    if not target_users:
        tui.tui_print_error("No target users configured. Please edit tgTrax_config.json")
        return
    
    tui.tui_print_info(f"Starting tracker for {len(target_users)} users: {target_users}")
    
    db_path = config.get("database_path", "tgTrax.db")
    session_name = config.get("telegram_session_name", "tgTrax_session")
    
    tracker = CorrelationTracker(
        target_usernames=target_users,
        db_path=db_path,
        session_name=session_name
    )
    
    try:
        await tracker.start_tracking()
    except KeyboardInterrupt:
        tui.tui_print_info("Received interrupt signal. Stopping tracker...")
        await tracker.stop_tracking()
    except Exception as e:
        tui.tui_print_error(f"Tracker error: {e}")
        logger.exception("Tracker failed with exception")
        await tracker.stop_tracking()


def run_analysis(config: Dict, demo_mode: bool = False) -> None:
    """Runs correlation analysis on collected data."""
    tui.tui_print_info("Starting correlation analysis...")
    
    db_path = config.get("database_path", "tgTrax.db")
    analysis_config = config.get("analysis", {})
    
    if demo_mode:
        tui.tui_print_highlight("Demo mode: Creating sample data for analysis")
        create_demo_data(db_path)
    
    # Initialize database
    db = SQLiteDatabase(db_path)
    
    # Get target users from config
    target_users = config.get("target_users", [])
    if not target_users and not demo_mode:
        tui.tui_print_error("No target users configured. Please edit tgTrax_config.json")
        return
    
    if demo_mode:
        target_users = ["demo_user1", "demo_user2", "demo_user3"]
    
    # Fetch activity data
    tui.tui_print_info(f"Fetching activity data for {len(target_users)} users...")
    activity_records = db.get_all_activity_for_users(target_users)
    
    if not activity_records:
        tui.tui_print_warning("No activity data found. Run tracker first or use --demo mode.")
        return
    
    # Convert to DataFrame
    df_data = []
    for username, timestamp, is_online in activity_records:
        df_data.append({
            'timestamp': pd.to_datetime(timestamp, unit='ms', utc=True),
            'username': username,
            'online': int(is_online)
        })
    
    if not df_data:
        tui.tui_print_warning("No valid activity data to analyze.")
        return
    
    df = pd.DataFrame(df_data)
    
    # Pivot to get users as columns
    activity_df = df.pivot_table(
        index='timestamp',
        columns='username',
        values='online',
        fill_value=0
    )
    
    tui.tui_print_info(f"Activity data shape: {activity_df.shape}")
    
    # Initialize analyzer
    resample_period = analysis_config.get("resample_period", "1min")
    correlation_threshold = analysis_config.get("correlation_threshold", 0.6)
    jaccard_threshold = analysis_config.get("jaccard_threshold", 0.18)
    
    analyzer = TemporalAnalyzer(
        activity_df=activity_df,
        resample_period=resample_period,
        correlation_threshold=correlation_threshold,
        jaccard_threshold=jaccard_threshold
    )
    
    # Perform analysis
    tui.tui_print_info("Calculating correlations...")
    correlation_matrix = analyzer.get_correlation_matrix()
    
    if not correlation_matrix.empty:
        tui.tui_print_info("Correlation Matrix:")
        print(correlation_matrix.round(3))
        print()
    
    # Get significant pairs
    significant_pairs = analyzer.get_significant_pairs()
    if significant_pairs:
        tui.tui_print_success(f"Found {len(significant_pairs)} significant correlations:")
        for (user1, user2), correlation in significant_pairs:
            tui.tui_print_detail(f"  {user1} ↔ {user2}: {correlation:.3f}")
    else:
        tui.tui_print_info("No significant correlations found.")
    
    # Build correlation graph and detect communities
    tui.tui_print_info("Building correlation graph and detecting communities...")
    graph = analyzer.build_correlation_graph()
    communities = analyzer.get_communities(graph)
    
    if communities:
        tui.tui_print_success(f"Detected {len(communities)} communities:")
        for comm_id, users in communities.items():
            tui.tui_print_detail(f"  Community {comm_id}: {', '.join(users)}")
    
    # Generate activity intervals for visualization
    tui.tui_print_info("Generating activity timeline data...")
    activity_intervals = analyzer.get_activity_intervals()
    
    # Summary statistics
    summary = analyzer.get_summary_stats()
    tui.tui_print_info("Analysis Summary:")
    for key, value in summary.items():
        tui.tui_print_detail(f"  {key}: {value}")
    
    # Try to create Gantt chart if plotly is available
    try:
        import plotly
        if activity_intervals:
            tui.tui_print_info("Creating activity Gantt chart...")
            fig = create_activity_gantt_chart(
                activity_intervals, 
                title="User Activity Timeline"
            )
            if fig:
                output_path = "activity_timeline.html"
                fig.write_html(output_path)
                tui.tui_print_success(f"Activity timeline saved to: {output_path}")
    except ImportError:
        tui.tui_print_warning("Plotly not available. Skipping Gantt chart generation.")
    except Exception as e:
        tui.tui_print_warning(f"Could not create Gantt chart: {e}")
    
    tui.tui_print_success("Analysis completed!")


def create_demo_data(db_path: str) -> None:
    """Creates demo activity data for testing analysis."""
    import random
    from datetime import datetime, timedelta
    
    db = SQLiteDatabase(db_path)
    
    # Demo users
    users = ["demo_user1", "demo_user2", "demo_user3"]
    
    # Generate 24 hours of demo data
    start_time = datetime.now() - timedelta(hours=24)
    
    for user in users:
        current_time = start_time
        online = False
        
        while current_time < datetime.now():
            # Random online/offline periods
            if online:
                # Stay online for 10-60 minutes
                duration = timedelta(minutes=random.randint(10, 60))
            else:
                # Stay offline for 5-30 minutes
                duration = timedelta(minutes=random.randint(5, 30))
            
            timestamp_ms = int(current_time.timestamp() * 1000)
            db.insert_activity(user, timestamp_ms, online)
            
            current_time += duration
            online = not online  # Toggle status
    
    tui.tui_print_success("Demo data created successfully!")


def main() -> None:
    """Main entry point for the tgTrax application."""
    parser = argparse.ArgumentParser(
        description="tgTrax - Telegram Activity Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py tracker                    # Start activity tracking
  python main.py analyze                    # Run correlation analysis
  python main.py analyze --demo             # Run analysis with demo data
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Tracker command
    tracker_parser = subparsers.add_parser('tracker', help='Start Telegram activity tracking')
    tracker_parser.add_argument(
        '--users', 
        nargs='+', 
        help='Override target users from config'
    )
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Run correlation analysis')
    analyze_parser.add_argument(
        '--demo', 
        action='store_true', 
        help='Run analysis with demo data'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Print banner
    tui.tui_print_highlight("""
  ┏┳┓     
╋┏┓┃┏┓┏┓┓┏
┗┗┫┻┛ ┗┻┛┗
  ┛       
tgTrax - Telegram Activity Analyzer
""")
    
    # Create default files if needed
    create_default_env_file()
    
    # Load configuration
    config = load_config()
    
    if args.command == 'tracker':
        # Validate environment for tracker
        if not validate_environment():
            tui.tui_print_error("Environment validation failed. Please check your .env file.")
            return
        
        # Get target users
        target_users = args.users or config.get("target_users", [])
        if not target_users:
            tui.tui_print_error(
                "No target users specified. Use --users argument or edit tgTrax_config.json"
            )
            return
        
        # Run tracker
        try:
            asyncio.run(run_tracker(target_users, config))
        except KeyboardInterrupt:
            tui.tui_print_info("Tracker stopped by user.")
    
    elif args.command == 'analyze':
        # Run analysis
        run_analysis(config, demo_mode=args.demo)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()