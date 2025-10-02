import argparse
import asyncio
import sys
import os
from dotenv import load_dotenv
import json # Added for JSON operations
from tgTrax.utils import tui
from telethon import TelegramClient, errors as tg_errors

# Ensure the project root is in sys.path for module resolution
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
SESSIONS_DIR = os.path.join(PROJECT_ROOT, "sessions")
os.makedirs(SESSIONS_DIR, exist_ok=True)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Default values for unreactor (removed module)
DEFAULT_UNREACTOR_ACTION_DELAY_SECONDS = 1.0
DEFAULT_UNREACTOR_SCAN_DELAY_SECONDS = 5.0
DEFAULT_UNREACTOR_MAX_MESSAGES_SCAN = 100
DEFAULT_UNREACTOR_SCAN_DAYS_AGO = 7
DEFAULT_UNREACTOR_MAX_STORIES_SCAN = 50

# --- Default Config Values (used if config file is missing/corrupt) ---
DEFAULT_TG_SESSION_NAME = 'tgTrax_session'
DEFAULT_DB_PATH = 'tgTrax.db' # Relative to project root
DEFAULT_LOG_LEVEL = 'INFO'
DEFAULT_LOG_PATH = 'tgTrax.log' # Relative to project root
DEFAULT_SCAN_USER_BATCH_SIZE = 10
DEFAULT_SCAN_USER_DELAY = 5
DEFAULT_SCRAPE_CHANNEL_LIMIT = 1000
DEFAULT_SCRAPE_OLDER_THAN_DAYS = 30
DEFAULT_DASHBOARD_HOST = '0.0.0.0'
DEFAULT_DASHBOARD_PORT = 8050
DEFAULT_DASHBOARD_DEBUG = True
DEFAULT_REDIS_HOST = 'localhost'
DEFAULT_REDIS_PORT = 6379
DEFAULT_REDIS_DB = 0
DEFAULT_REDIS_CACHE_TTL_SECONDS = 3600 # 1 hour for general cache items


def create_default_config_files_if_missing():
    """
    Checks for .env and tgTrax_config.json in the project root.
    Creates them with default values if they don't exist.
    Uses basic print for feedback as logger might not be configured yet.
    """
    env_path = os.path.join(PROJECT_ROOT, ".env")
    config_json_path = os.path.join(PROJECT_ROOT, "tgTrax_config.json")

    if not os.path.exists(env_path):
        print(f"INFO: .env file not found at {env_path}. Creating default .env file...")
        with open(env_path, "w") as f:
            f.write("# Telegram API Credentials (Required)\\n")
            f.write("TELEGRAM_API_ID=\\\"YOUR_API_ID\\\"\\n")
            f.write("TELEGRAM_API_HASH=\\\"YOUR_API_HASH\\\"\\n")
            f.write("TELEGRAM_PHONE_NUMBER=\\\"YOUR_PHONE_NUMBER_WITH_COUNTRY_CODE\\\"\\n\\n")
            f.write("# Optional: Dashboard Configuration (Uncomment and set if needed)\\n")
            f.write("# FLASK_HOST=\\\"127.0.0.1\\\"\\n")
            f.write("# FLASK_PORT=\\\"5001\\\"\\n\\n")
            f.write("# Optional: Tracker Specific (Uncomment and set if needed for direct tracker.py usage)\\n")
            f.write("# USER_STATUS_POLL_INTERVAL_SECONDS=\\\"60\\\"\\n")
            f.write("# MINIMUM_ASSUMED_ONLINE_DURATION_SECONDS=\\\"60\\\"\\n")
            f.write("# TARGET_USERS=\\\"username1,username2\\\"\\n")
        print(f"INFO: Default .env file created at {env_path}. Please fill in your Telegram API credentials.")
    else:
        print(f"DEBUG: .env file already exists at {env_path}.")

    if not os.path.exists(config_json_path):
        print(f"INFO: tgTrax_config.json not found at {config_json_path}. Creating default config file...")
        default_config_data = {
            "telegram_session_name": DEFAULT_TG_SESSION_NAME,
            "database_path": DEFAULT_DB_PATH,
            "log_level": DEFAULT_LOG_LEVEL,
            "log_path": DEFAULT_LOG_PATH,
            "scan_user_batch_size": DEFAULT_SCAN_USER_BATCH_SIZE,
            "scan_user_delay_seconds": DEFAULT_SCAN_USER_DELAY,
            "scrape_channel_default_limit": DEFAULT_SCRAPE_CHANNEL_LIMIT,
            "scrape_channel_default_older_than_days": DEFAULT_SCRAPE_OLDER_THAN_DAYS,
            "dashboard_host": DEFAULT_DASHBOARD_HOST,
            "dashboard_port": DEFAULT_DASHBOARD_PORT,
            "dashboard_debug": DEFAULT_DASHBOARD_DEBUG,
            "unreactor_default_action_delay_seconds": DEFAULT_UNREACTOR_ACTION_DELAY_SECONDS,
            "unreactor_default_scan_delay_seconds": DEFAULT_UNREACTOR_SCAN_DELAY_SECONDS,
            "unreactor_default_max_messages_per_chat_scan": DEFAULT_UNREACTOR_MAX_MESSAGES_SCAN,
            "unreactor_default_scan_days_ago": DEFAULT_UNREACTOR_SCAN_DAYS_AGO,
            "unreactor_default_max_stories_per_peer_scan": DEFAULT_UNREACTOR_MAX_STORIES_SCAN,
            "redis_host": DEFAULT_REDIS_HOST,
            "redis_port": DEFAULT_REDIS_PORT,
            "redis_db": DEFAULT_REDIS_DB,
            "redis_cache_ttl_seconds": DEFAULT_REDIS_CACHE_TTL_SECONDS
        }
        with open(config_json_path, "w") as f:
            json.dump(default_config_data, f, indent=4)
        print(f"INFO: Default tgTrax_config.json created at {config_json_path}.")
    else:
        print(f"DEBUG: tgTrax_config.json already exists at {config_json_path}.")

# Create config files BEFORE loading .env or initializing the main logger
create_default_config_files_if_missing()

# Load .env file contents into environment variables
load_dotenv(dotenv_path=os.path.join(PROJECT_ROOT, ".env"))

# Now that config files are ensured and .env is loaded, import other modules including logger
from telethon import TelegramClient
from tgTrax.core.auth import get_client, ensure_credentials
from tgTrax.core.tracker import CorrelationTracker as UserTracker
from tgTrax.core.database import SQLiteDatabase as DatabaseManager
# from tgTrax.core.channel_scraper import ChannelScraper  # Removed
# from tgTrax.core.unreactor import unreact_main  # Removed
from tgTrax.web.dashboard import app as dashboard_flask_app
from tgTrax.core.tracker import CorrelationTracker as UserTracker
from tgTrax.core.analysis import TemporalAnalyzer


# --- Application Configuration Loading ---
APP_CONFIG = {}
CONFIG_JSON_PATH = os.path.join(PROJECT_ROOT, "tgTrax_config.json")

try:
    with open(CONFIG_JSON_PATH, 'r') as f:
        APP_CONFIG = json.load(f)
except FileNotFoundError:
    APP_CONFIG = {} # Rely on .get() fallbacks below
except json.JSONDecodeError:
    APP_CONFIG = {} # Rely on .get() fallbacks below

# Helper to resolve paths relative to project root
def get_absolute_path(key_in_config, default_python_value):
    path_to_resolve = APP_CONFIG.get(key_in_config, default_python_value)
    if not os.path.isabs(path_to_resolve):
        return os.path.join(PROJECT_ROOT, path_to_resolve)
    return path_to_resolve

# Use loaded config values, falling back to Python defaults if a key is missing or config is empty/corrupt
TG_SESSION_NAME = APP_CONFIG.get("telegram_session_name", DEFAULT_TG_SESSION_NAME)
# Always use absolute session path under tgTrax/sessions to avoid CWD issues
SESSION_FILE_PATH = os.path.join(SESSIONS_DIR, TG_SESSION_NAME)
# Migrate legacy session file from repo root (telegram/) if present
LEGACY_SESSION = os.path.join(os.path.dirname(PROJECT_ROOT), f"{TG_SESSION_NAME}.session")
try:
    if os.path.exists(LEGACY_SESSION) and not os.path.exists(f"{SESSION_FILE_PATH}.session"):
        import shutil
        shutil.copy2(LEGACY_SESSION, f"{SESSION_FILE_PATH}.session")
        print(f"INFO: Migrated legacy session to {SESSION_FILE_PATH}.session")
except Exception:
    pass
DB_PATH = get_absolute_path("database_path", DEFAULT_DB_PATH)
LOG_LEVEL = APP_CONFIG.get("log_level", DEFAULT_LOG_LEVEL) # Logger util will use this
LOG_PATH = get_absolute_path("log_path", DEFAULT_LOG_PATH) # Logger util will use this

SCAN_USER_BATCH_SIZE = APP_CONFIG.get("scan_user_batch_size", DEFAULT_SCAN_USER_BATCH_SIZE)
SCAN_USER_DELAY = APP_CONFIG.get("scan_user_delay_seconds", DEFAULT_SCAN_USER_DELAY)
SCRAPE_CHANNEL_LIMIT = APP_CONFIG.get("scrape_channel_default_limit", DEFAULT_SCRAPE_CHANNEL_LIMIT)
SCRAPE_OLDER_THAN_DAYS = APP_CONFIG.get("scrape_channel_default_older_than_days", DEFAULT_SCRAPE_OLDER_THAN_DAYS)

# Credentials from .env file (loaded by load_dotenv)
TG_API_ID = os.getenv('TELEGRAM_API_ID')
TG_API_HASH = os.getenv('TELEGRAM_API_HASH')
TG_PHONE_NUMBER = os.getenv('TELEGRAM_PHONE_NUMBER')

# Dashboard config: .env overrides tgTrax_config.json, which overrides Python defaults
DASHBOARD_HOST = os.getenv('FLASK_HOST', APP_CONFIG.get("dashboard_host", DEFAULT_DASHBOARD_HOST))
DASHBOARD_PORT = int(os.getenv('FLASK_PORT', str(APP_CONFIG.get("dashboard_port", DEFAULT_DASHBOARD_PORT))))
DASHBOARD_DEBUG_STR = os.getenv('FLASK_DEBUG', str(APP_CONFIG.get("dashboard_debug", DEFAULT_DASHBOARD_DEBUG)))
DASHBOARD_DEBUG = DASHBOARD_DEBUG_STR.lower() == 'true'

# Redis Config from tgTrax_config.json, with Python defaults
REDIS_HOST = APP_CONFIG.get("redis_host", DEFAULT_REDIS_HOST)
REDIS_PORT = int(APP_CONFIG.get("redis_port", DEFAULT_REDIS_PORT))
REDIS_DB = int(APP_CONFIG.get("redis_db", DEFAULT_REDIS_DB))
REDIS_CACHE_TTL_SECONDS = int(APP_CONFIG.get("redis_cache_ttl_seconds", DEFAULT_REDIS_CACHE_TTL_SECONDS))

# --- End Application Configuration Loading ---

async def run_tracker_mode(args, client: TelegramClient, db_manager: DatabaseManager):
    tui.tui_print_info("Starting user tracker mode...")
    target_users_env = os.getenv("TARGET_USERS")
    if not target_users_env:
        tui.tui_print_error("TARGET_USERS environment variable not set. CorrelationTracker mode cannot start.")
        return
    target_usernames_list = [u.strip() for u in target_users_env.split(',') if u.strip()]
    if not target_usernames_list:
        tui.tui_print_error("No valid target usernames parsed from TARGET_USERS for CorrelationTracker.")
        return
    tracker = UserTracker(target_usernames=target_usernames_list, db_path=DB_PATH)
    await tracker.start_tracking()

def run_dashboard_mode_entry(args): # Synchronous entry for dashboard
    tui.tui_print_info(f"Starting dashboard mode on {DASHBOARD_HOST}:{DASHBOARD_PORT} (Debug: {DASHBOARD_DEBUG})")
    try:
        # Provide DB path to Flask API
        os.environ.setdefault('TGTRAX_DB_PATH', DB_PATH)
        dashboard_flask_app.run(host=DASHBOARD_HOST,
                                port=DASHBOARD_PORT,
                                debug=DASHBOARD_DEBUG,
                                use_reloader=False)
    except Exception as e:
        tui.tui_print_error(f"Failed to start dashboard: {e}")
        pass

def handle_report_mode(args):
    """Generate a TUI report of significant pairs and export JSON if requested."""
    from tgTrax.core.database import SQLiteDatabase
    import sqlite3
    # Load users: TARGET_USERS or discover distinct from DB
    target_users_env = os.getenv("TARGET_USERS")
    if target_users_env:
        user_list = [u.strip() for u in target_users_env.split(',') if u.strip()]
    else:
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT username FROM activity")
            user_list = [r[0] for r in cur.fetchall()]
            conn.close()
        except Exception as e:
            tui.tui_print_error(f"Failed to discover users from DB: {e}")
            user_list = []

    if not user_list:
        tui.tui_print_error("No users to analyze. Set TARGET_USERS or populate DB.")
        return

    tracker = UserTracker(target_usernames=user_list, db_path=DB_PATH)
    df = tracker.get_activity_data()
    if df.empty:
        tui.tui_print_warning("No activity data found. Abort report.")
        return

    analyzer = TemporalAnalyzer(
        df,
        resample_period=args.period,
        correlation_threshold=args.corr_threshold,
        jaccard_threshold=args.jaccard_threshold,
        ewma_alpha=args.ewma_alpha,
        seasonal_adjust=args.seasonal_adjust,
        max_lag_minutes=args.max_lag_minutes,
        corr_method=args.method,
        fdr_alpha=args.fdr_alpha,
    )

    # Collect pairs
    pairs = analyzer.get_crosscorr_significant_pairs(
        q_threshold=args.q_threshold, min_abs_corr=args.min_abs_corr
    )
    jacc = analyzer.get_jaccard_matrix()
    corr = analyzer.get_correlation_matrix()

    # Render TUI table
    from rich.table import Table
    from rich.console import Console
    table = Table(title="Significant Pairs (Cross-corr FDR)")
    table.add_column("#", justify="right")
    table.add_column("User A")
    table.add_column("User B")
    table.add_column("r", justify="right")
    table.add_column("lag(s)", justify="right")
    table.add_column("q", justify="right")
    table.add_column("Jaccard", justify="right")
    table.add_column("Spearman", justify="right")

    rows_out = []
    for idx, (pair, meta) in enumerate(pairs[: args.top], start=1):
        u, v = pair
        r = meta.get("r")
        lag = meta.get("lag_seconds")
        q = meta.get("q")
        jv = None if jacc.empty else float(jacc.at[u, v]) if (u in jacc.index and v in jacc.columns) else None
        cv = None if corr.empty else float(corr.at[u, v]) if (u in corr.index and v in corr.columns) else None
        table.add_row(str(idx), u, v, f"{r:.3f}", f"{lag:.0f}", f"{q:.3f}", f"{jv:.3f}" if jv is not None else "-", f"{cv:.3f}" if cv is not None else "-")
        rows_out.append({"u": u, "v": v, "r": r, "lag_seconds": lag, "q": q, "jaccard": jv, "corr": cv})

    Console().print(table)

    # Optional export
    if args.export:
        try:
            with open(args.export, "w") as f:
                json.dump({"pairs": rows_out}, f, indent=2)
            tui.tui_print_success(f"Report exported to {args.export}")
        except Exception as e:
            tui.tui_print_error(f"Failed to export report: {e}")

async def handle_scrape(args, client: TelegramClient, db_manager: DatabaseManager):
    tui.tui_print_info("Channel scraper functionality removed")
    return

async def handle_unreact(args, client):
    tui.tui_print_info(f"Unreact mode initiated. Dry run: {args.dry_run}, User-provided scan days ago: {args.scan_days_ago}")

    # Get defaults from APP_CONFIG, falling back to Python constants if not in config
    default_action_delay = APP_CONFIG.get("unreactor_default_action_delay_seconds", DEFAULT_UNREACTOR_ACTION_DELAY_SECONDS)
    default_scan_delay = APP_CONFIG.get("unreactor_default_scan_delay_seconds", DEFAULT_UNREACTOR_SCAN_DELAY_SECONDS)
    default_max_messages = APP_CONFIG.get("unreactor_default_max_messages_per_chat_scan", DEFAULT_UNREACTOR_MAX_MESSAGES_SCAN)
    default_max_stories = APP_CONFIG.get("unreactor_default_max_stories_per_peer_scan", DEFAULT_UNREACTOR_MAX_STORIES_SCAN)
    default_scan_days_config = APP_CONFIG.get("unreactor_default_scan_days_ago", DEFAULT_UNREACTOR_SCAN_DAYS_AGO)

    action_delay = args.unreact_action_delay if args.unreact_action_delay is not None else default_action_delay
    scan_delay = args.unreact_scan_delay if args.unreact_scan_delay is not None else default_scan_delay
    max_messages_scan = args.unreact_max_messages_scan if args.unreact_max_messages_scan is not None else default_max_messages
    max_stories_scan = args.unreact_max_stories_scan if args.unreact_max_stories_scan is not None else default_max_stories
    
    # If user provides --scan-days-ago, use that. Otherwise, use the configured default.
    effective_scan_days_ago = args.scan_days_ago if args.scan_days_ago is not None else default_scan_days_config

    tui.tui_print_info(f"Unreact effective params: action_delay={action_delay}, scan_delay={scan_delay}, max_messages_scan={max_messages_scan}, max_stories_scan={max_stories_scan}, scan_days_ago={effective_scan_days_ago}")

    tui.tui_print_info("Unreactor functionality removed")
    return


async def main_async_runner():
    print("DEBUG: main_async_runner started")
    parser = argparse.ArgumentParser(description="TgTrax - Telegram User Activity Tracker and Analyzer.")
    subparsers = parser.add_subparsers(dest="mode", help="Available modes of operation", required=True)

    # Argparse defaults are primarily for help messages; runtime defaults are from config
    default_scrape_limit_arg = APP_CONFIG.get("scrape_channel_default_limit", DEFAULT_SCRAPE_CHANNEL_LIMIT)
    default_scrape_older_than_arg = APP_CONFIG.get("scrape_channel_default_older_than_days", DEFAULT_SCRAPE_OLDER_THAN_DAYS)

    default_unreact_action_delay_arg = APP_CONFIG.get("unreactor_default_action_delay_seconds", DEFAULT_UNREACTOR_ACTION_DELAY_SECONDS)
    default_unreact_scan_delay_arg = APP_CONFIG.get("unreactor_default_scan_delay_seconds", DEFAULT_UNREACTOR_SCAN_DELAY_SECONDS)
    default_unreact_max_messages_arg = APP_CONFIG.get("unreactor_default_max_messages_per_chat_scan", DEFAULT_UNREACTOR_MAX_MESSAGES_SCAN)
    default_unreact_scan_days_arg_help = APP_CONFIG.get("unreactor_default_scan_days_ago", DEFAULT_UNREACTOR_SCAN_DAYS_AGO)
    default_unreact_max_stories_arg = APP_CONFIG.get("unreactor_default_max_stories_per_peer_scan", DEFAULT_UNREACTOR_MAX_STORIES_SCAN)

    # Tracker mode
    tracker_parser = subparsers.add_parser("tracker", help="Run user activity tracking service.")
    tracker_parser.set_defaults(func=run_tracker_mode)

    # Dashboard mode
    dashboard_parser = subparsers.add_parser("dashboard", help="Run the web interface dashboard.")
    dashboard_parser.set_defaults(func=run_dashboard_mode_entry)

    # Report mode (CLI analysis)
    report_parser = subparsers.add_parser("report", help="Generate TUI analysis report with significance and lags.")
    report_parser.add_argument("--period", type=str, default="1min", help="Resample period, e.g. 1min, 5T")
    report_parser.add_argument("--method", type=str, default="spearman", choices=["spearman","pearson"], help="Correlation method")
    report_parser.add_argument("--ewma-alpha", type=float, default=None, help="EWMA smoothing alpha (0,1]")
    report_parser.add_argument("--seasonal-adjust", action="store_true", help="Enable STL seasonal/trend removal")
    report_parser.add_argument("--max-lag-minutes", type=int, default=15, help="Max lag (minutes) for cross-correlation")
    report_parser.add_argument("--fdr-alpha", type=float, default=0.05, help="FDR alpha for q-values")
    report_parser.add_argument("--q-threshold", type=float, default=0.05, help="q-value threshold to accept pair")
    report_parser.add_argument("--min-abs-corr", type=float, default=0.0, help="Minimum absolute cross-corr at best lag")
    report_parser.add_argument("--corr-threshold", type=float, default=0.3, help="Threshold for plain correlation")
    report_parser.add_argument("--jaccard-threshold", type=float, default=0.18, help="Threshold for Jaccard")
    report_parser.add_argument("--top", type=int, default=50, help="Top N pairs to display")
    report_parser.add_argument("--export", type=str, default=None, help="Export report JSON to path")
    report_parser.set_defaults(func=handle_report_mode)

    # Scrape mode
    scrape_parser = subparsers.add_parser("scrape", help="Scrape messages from a specific public channel.")
    scrape_parser.add_argument("channel", type=str, help="Username or ID of the public channel (e.g., 'durovschat').")
    scrape_parser.add_argument("--limit", type=int, default=default_scrape_limit_arg, help=f"Max messages to scrape. Default: {default_scrape_limit_arg}")
    scrape_parser.add_argument("--older-than-days", type=int, default=default_scrape_older_than_arg, help=f"Scrape messages older than N days. 0 for no date limit. Default: {default_scrape_older_than_arg}")
    scrape_parser.set_defaults(func=handle_scrape)

    # Unreact mode
    unreact_parser = subparsers.add_parser("unreact", help="Scan for and remove your reactions.")
    unreact_parser.add_argument("--scan-messages", action="store_true", help="Scan messages for your reactions. If no specific scan/process flag is set, this defaults to active.")
    unreact_parser.add_argument("--scan-stories", action="store_true", help="Scan stories for your reactions. If no specific scan/process flag is set, this defaults to active.")
    unreact_parser.add_argument("--process-only", action="store_true", help="Only process reactions already found in the database (skip scanning). Overrides --scan-messages and --scan-stories.")
    unreact_parser.add_argument("--dry-run", action="store_true", help="Simulate unreacting without making actual changes.")
    unreact_parser.add_argument(
        "--scan-days-ago",
        type=int,
        default=None, # User explicitly sets or it's None. handle_unreact resolves it with config.
        help=f"Scan messages from the last N days. 0 for all history. Default (if not set by user) is from config ({default_unreact_scan_days_arg_help} days). Only applies if message scanning is active."
    )
    unreact_parser.add_argument("--unreact-action-delay", type=int, default=None, help=f"Delay (seconds) between unreact actions. Overrides config. Default from config: {default_unreact_action_delay_arg}s")
    unreact_parser.add_argument("--unreact-scan-delay", type=int, default=None, help=f"Delay (seconds) between scanning dialogs/peers. Overrides config. Default from config: {default_unreact_scan_delay_arg}s")
    unreact_parser.add_argument("--unreact-max-messages-scan", type=int, default=None, help=f"Max messages to scan per chat. Overrides config. Default from config: {default_unreact_max_messages_arg}")
    unreact_parser.add_argument("--unreact-max-stories-scan", type=int, default=None, help=f"Max stories to scan per peer. Overrides config. Default from config: {default_unreact_max_stories_arg}")
    unreact_parser.add_argument("--no-cache", action="store_true", help="Disable Redis caching for this run.")
    unreact_parser.set_defaults(func=handle_unreact)

    # Auth console mode
    auth_parser = subparsers.add_parser("auth", help="Interactive console login to Telegram")
    auth_parser.add_argument("--phone", type=str, help="Phone number with country code")
    auth_parser.add_argument("--code", type=str, help="Login code received from Telegram")
    auth_parser.add_argument("--password", type=str, help="2FA password if enabled")
    auth_parser.add_argument("--session-name", type=str, help="Override session file name")
    auth_parser.set_defaults(func=handle_auth_console)

    # Parse and dispatch
    args = parser.parse_args()
    selected_func = args.func

    client_instance = None
    db_manager_instance = None

    try:
        if asyncio.iscoroutinefunction(selected_func):
            # Special-case: auth console does not require a connected client
            if selected_func is handle_auth_console:
                await selected_func(args)
                return

            # Special-case: tracker manages its own Telethon client; avoid opening a second one
            if selected_func is run_tracker_mode:
                db_manager_instance = DatabaseManager(db_path=DB_PATH)
                await selected_func(args, None, db_manager_instance)  # client arg unused
                return

            ensure_credentials(TG_API_ID, TG_API_HASH, TG_PHONE_NUMBER)
            client_instance = await get_client(SESSION_FILE_PATH, TG_API_ID, TG_API_HASH, TG_PHONE_NUMBER)
            if not client_instance:
                tui.tui_print_error("Failed to initialize Telegram client for async mode.")
                return

            if selected_func is handle_scrape:
                db_manager_instance = DatabaseManager(db_path=DB_PATH)
                await selected_func(args, client_instance, db_manager_instance)
            else:
                await selected_func(args, client_instance)
        else:
            # Synchronous function (e.g., dashboard)
            selected_func(args)
    except Exception as e:
        tui.tui_print_error(f"Critical error during operation in mode '{getattr(args, 'mode', '?')}': {e}")
    finally:
        if db_manager_instance:
            tui.tui_print_info("Closing database connection from main_async_runner...")
            db_manager_instance.close()
        if client_instance and asyncio.iscoroutinefunction(selected_func):
            tui.tui_print_info("Disconnecting Telegram client from main_async_runner...")
            await client_instance.disconnect()
            tui.tui_print_info("Client disconnected.")

async def handle_auth_console(args):
    if not TG_API_ID or not TG_API_HASH:
        tui.tui_print_error("Missing TELEGRAM_API_ID or TELEGRAM_API_HASH in environment.")
        return

    try:
        api_id_int = int(TG_API_ID)
    except ValueError:
        tui.tui_print_error("TELEGRAM_API_ID must be an integer.")
        return

    session_name = args.session_name or SESSION_FILE_PATH
    phone = (args.phone or TG_PHONE_NUMBER or "").strip()
    if not phone:
        phone = input("Enter phone number (+countrycode...): ").strip()

    client = TelegramClient(session_name, api_id_int, TG_API_HASH)
    await client.connect()

    if await client.is_user_authorized():
        tui.tui_print_success("Already authorized. Session file is valid.")
        await client.disconnect()
        return

    tui.tui_print_info(f"Sending code to {phone}...")
    await client.send_code_request(phone)

    code = (args.code or input("Enter the code you received: ")).strip()

    try:
        await client.sign_in(phone=phone, code=code)
    except tg_errors.SessionPasswordNeededError:
        password = args.password or input("2FA password required: ")
        await client.sign_in(password=password)

    if await client.is_user_authorized():
        tui.tui_print_success("Telegram session authorized. You can now run the tracker/API.")
    else:
        tui.tui_print_error("Authorization failed. Please verify the code/password and retry.")

    await client.disconnect()


if __name__ == "__main__":
    try:
        asyncio.run(main_async_runner())
    except KeyboardInterrupt:
        tui.tui_print_warning("Process interrupted by user (KeyboardInterrupt). Exiting...")
        sys.exit(0)
    except Exception as e:
        tui.tui_print_error(f"Unhandled critical exception in __main__: {e}")
        sys.exit(1)
