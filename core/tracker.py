# ==== TELEGRAM ACTIVITY TRACKER MODULE ==== #
"""
Track Telegram user activity and persist events.

Overview:
- Connects to Telegram via Telethon.
- Monitors online/offline status for target usernames.
- Records activity into SQLite using the `SQLiteDatabase` helper.
- Uses real-time events and periodic polling for robustness.
"""

import asyncio
import logging
import datetime
import os
import time
from typing import List, Dict, Optional, Any, Union, Tuple

import pandas as pd
from dotenv import load_dotenv
from telethon import TelegramClient, events, errors
from telethon.tl.types import (
    User,
    UserStatusOnline,
    UserStatusOffline,
    UserStatusRecently,
    UserStatusLastWeek,
    UserStatusLastMonth,
)

from tgTrax.core.database import SQLiteDatabase
 


# --► GLOBAL CONFIGURATION & CONSTANTS
logger = logging.getLogger(__name__)

load_dotenv()

# Telegram API Credentials (from .env)
TELEGRAM_API_ID_STR: Optional[str] = os.getenv("TELEGRAM_API_ID")
TELEGRAM_API_HASH: Optional[str] = os.getenv("TELEGRAM_API_HASH")
TELEGRAM_API_ID: Optional[int] = None

if not TELEGRAM_API_ID_STR or not TELEGRAM_API_HASH:
    logger.error(
        "Critical: TELEGRAM_API_ID or TELEGRAM_API_HASH not found in .env "
        "or environment."
    )
else:
    try:
        TELEGRAM_API_ID = int(TELEGRAM_API_ID_STR)
    except ValueError:
        logger.error("Critical: TELEGRAM_API_ID must be an integer.")

# Project paths
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSIONS_DIR: str = os.path.join(PROJECT_ROOT, "sessions")

if not os.path.exists(SESSIONS_DIR):
    os.makedirs(SESSIONS_DIR)
    logger.info("Created sessions directory: %s", SESSIONS_DIR)

SESSION_BASENAME = os.getenv("TELEGRAM_SESSION_NAME", "tgTrax_session")
SESSION_NAME_DEFAULT: str = os.path.join(SESSIONS_DIR, SESSION_BASENAME)

# Polling and activity configuration (from .env, with defaults)
USER_STATUS_POLL_INTERVAL_SECONDS: int = int(
    os.getenv("USER_STATUS_POLL_INTERVAL_SECONDS", "60")
)
MINIMUM_ASSUMED_ONLINE_DURATION_SECONDS: int = int(
    os.getenv("MINIMUM_ASSUMED_ONLINE_DURATION_SECONDS", "60")
)

# Debugging: List of usernames for verbose status logging
TARGET_DEBUG_USERS: List[str] = [
    "user_one",
    "user_two", 
    "user_three",
    "user_four",
    "user_five",
    "user_six",
]



# ==== CORRELATION TRACKER CLASS ==== #

class CorrelationTracker:
    """
    Tracks Telegram user activity by monitoring online/offline status.

    It connects to Telegram using Telethon, resolves target usernames to IDs,
    listens for real-time status updates, and periodically polls users.
    Activity data is stored in an SQLite database.
    """

    def __init__(
        self,
        target_usernames: List[str],
        db_path: str = "activity.db",
        session_name: str = SESSION_NAME_DEFAULT,
    ) -> None:
        """
        Initialize the tracker with targets, database, and session path.

        Args:
            target_usernames: A list of Telegram @usernames to track.
                              The '@' prefix will be stripped if present.
            db_path: Path to the SQLite database file. If relative, it will be
                     resolved relative to the project root (tgTrax directory).
            session_name: Path to the Telethon session file.
        """
        self.target_usernames: List[str] = [
            name.lstrip("@") for name in target_usernames if name
        ]
        
        # Resolve db_path relative to PROJECT_ROOT if not absolute
        if not os.path.isabs(db_path):
            self.db_path: str = os.path.join(PROJECT_ROOT, db_path)
        else:
            self.db_path: str = db_path
        
        self.db: SQLiteDatabase = SQLiteDatabase(self.db_path)
        logger.info("Database initialized at: %s", self.db_path)

        self.session_name: str = session_name
        self.client: Optional[TelegramClient] = None
        self.user_id_map: Dict[int, str] = {}
        self.polling_task: Optional[asyncio.Task[None]] = None
        
        # Lazily create asyncio.Event when an event loop exists
        self.shutdown_event: Optional[asyncio.Event] = None
        self.is_running: bool = False
        self._handler_registered: bool = False

        if TELEGRAM_API_ID is None or TELEGRAM_API_HASH is None:
            logger.warning(
                "TELEGRAM_API_ID or TELEGRAM_API_HASH is missing/invalid. "
                "Client-dependent features will fail."
            )


    def _initialize_client(self) -> None:
        """
        Initialize the Telethon client if not already present.

        Raises:
            ValueError: If TELEGRAM_API_ID or TELEGRAM_API_HASH is not configured.
        """
        if self.client is not None:
            return

        if TELEGRAM_API_ID is None or TELEGRAM_API_HASH is None:
            err_msg = (
                "Cannot initialize TelegramClient: API_ID or API_HASH is "
                "missing/invalid."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)

        logger.debug("Initializing TelethonClient...")
        self.client = TelegramClient(
            self.session_name,
            TELEGRAM_API_ID,
            TELEGRAM_API_HASH,
            connection_retries=None,
            retry_delay=5,
        )


    async def _ensure_client_connected(self) -> None:
        """
        Ensures the Telethon client is initialized and connected.

        Handles connection and authorization, including interactive prompts
        if the session is new or unathorized.

        Raises:
            ConnectionRefusedError: If client authorization fails.
            errors.PhoneNumberInvalidError: If the phone number is invalid.
            errors.SessionPasswordNeededError: If 2FA password is required but not handled.
            errors.ApiIdInvalidError: If API ID/Hash are invalid.
            errors.RpcError, ConnectionError, TimeoutError, asyncio.TimeoutError: For network issues.
            Exception: For other unexpected errors during connection/auth.
        """
        self._initialize_client() # Ensures self.client is instantiated
        # This assertion is for type narrowing for mypy after _initialize_client call
        assert self.client is not None, "Client should be initialized here"

        if not self.client.is_connected():
            logger.info("Connecting to Telegram with session: %s...", self.session_name)
            try:
                await self.client.connect()
                if not await self.client.is_user_authorized():
                    logger.info(
                        "Client is not authorized. Attempting to sign in..."
                    )
                    logger.info(
                        "If this is the first run or session is invalid, prompts may occur for phone/code/2FA."
                    )
                    # Telethon's client.start() handles interactive login for phone, code, 2FA.
                    await self.client.start()
                    if not await self.client.is_user_authorized():
                        logger.error(
                            "Client authorization failed. Please check credentials/2FA."
                        )
                        raise ConnectionRefusedError("Telegram client authorization failed.")
                    logger.info("Client authorized successfully.")
                
                me: Optional[User] = await self.client.get_me()
                if me:
                    logger.info(
                        "Client connection confirmed. Logged in as: %s",
                        me.username or me.id,
                    )
                else: # Should ideally not happen if authorized
                    logger.warning(
                        "Client connected and authorized, but get_me() returned None."
                    )

            except ( # Specific, potentially recoverable errors first
                errors.PhoneNumberInvalidError,
                errors.SessionPasswordNeededError,
                errors.ApiIdInvalidError,
                errors.UserIsBotError, # If .start() detects bot credentials
            ) as auth_err:
                logger.error(f"Telegram authentication/configuration error: {auth_err}")
                raise
            except (errors.RPCError, ConnectionError, TimeoutError, asyncio.TimeoutError) as net_err:
                logger.error(f"Connection/RPC error during connect or auth: {net_err}")
                raise # Re-raise for outer retry logic
            except Exception as e:
                logger.error(f"Unexpected error during connect or auth: {e}")
                raise


    async def _resolve_target_user_ids(self) -> None:
        """
        Converts target @usernames to user IDs and populates `self.user_id_map`.

        Requires the client to be connected and authorized.
        Handles common errors during entity resolution like non-existent users or flood waits.
        """
        if not self.target_usernames:
            logger.warning("No target usernames provided for tracking.")
            return

        await self._ensure_client_connected()
        assert self.client is not None, "Client must be connected to resolve users."

        resolved_count = 0
        for username_to_resolve in self.target_usernames:
            if self._shutdown_is_set():
                logger.info("Shutdown initiated, stopping user resolution.")
                break
            try:
                logger.debug("Attempting to resolve @%s...", username_to_resolve)
                entity: User = await self.client.get_entity(username_to_resolve) # type: ignore
                # Ensure entity is a User and not Channel, etc. before accessing .username and .id
                if hasattr(entity, 'username') and hasattr(entity, 'id'):
                    self.user_id_map[entity.id] = entity.username # type: ignore
                    logger.info(
                        "Successfully resolved @%s to ID %s",
                        entity.username,  # type: ignore
                        entity.id,
                    )
                    resolved_count += 1
                else:
                    logger.warning(
                        "Resolved entity for @%s is not a standard user type or lacks expected attributes. Skipping.",
                        username_to_resolve,
                    )

            except (errors.UsernameNotOccupiedError, errors.UsernameInvalidError):
                logger.error(f"Username @{username_to_resolve} not found or invalid. Skipping.")
            except ValueError: # Raised by get_entity for various invalid peer types
                logger.error(f"Could not find or process user @{username_to_resolve}. Skipping.")
            except errors.FloodWaitError as e_flood:
                logger.warning(
                    f"Flood wait error resolving @{username_to_resolve}: {e_flood}. "
                    f"Waiting {e_flood.seconds}s."
                )
                try:
                    await asyncio.sleep(e_flood.seconds)
                except asyncio.CancelledError:
                    logger.info("User resolution sleep interrupted by shutdown.")
                    break
            except (errors.RPCError, ConnectionError, TimeoutError, asyncio.TimeoutError) as e_net:
                logger.error(
                    f"Network/RPC error resolving @{username_to_resolve}: {e_net}. "
                    "Will retry later if polling active."
                )
            except TypeError as e_type: # Can indicate API ID/Hash issues with get_entity
                logger.error(
                    f"TypeError during entity resolution for @{username_to_resolve} "
                    f"(check API_ID/HASH): {e_type}"
                )
            except Exception as e_res:
                logger.error(
                    f"Unexpected error resolving @{username_to_resolve}: {e_res}",
                    exc_info=True
                )
        
        if not self.user_id_map:
            logger.warning("Could not resolve any target usernames.")
        else:
            logger.info(
                "Successfully resolved %s/%s users. Tracking: %s",
                resolved_count,
                len(self.target_usernames),
                list(self.user_id_map.values()),
            )


    async def _process_status_update(
        self,
        username: str,
        user_id: int,
        status_obj: Any, # Actually Union[UserStatusOnline, ..., None]
        source: str = "event",
    ) -> None:
        """
        Processes a user status object and records it in the database.

        Handles different Telethon status types (Online, Offline, Recently, etc.)
        and an assumed online duration for offline events with a `was_online` time.

        Args:
            username: The username of the user.
            user_id: The user ID.
            status_obj: The Telethon status object (e.g., UserStatusOnline) or None.
            source: A string indicating the source of the update (e.g., "event", "polling").
        """
        current_event_time: Optional[datetime.datetime] = None
        is_online: bool = False
        assumed_online_timestamp_ms: Optional[int] = None

        if username in TARGET_DEBUG_USERS:
            logger.debug(
                "[%s] Processing status for %s (ID: %s). Raw status: %r",
                source,
                username,
                user_id,
                status_obj,
            )

        if isinstance(status_obj, UserStatusOnline):
            is_online = True
            current_event_time = datetime.datetime.now(datetime.timezone.utc)
            logger.info(
                "[%s] User %s (ID: %s) is ONLINE. Timestamp: %s",
                source,
                username,
                user_id,
                current_event_time.isoformat(),
            )
        elif isinstance(status_obj, UserStatusOffline):
            is_online = False
            if status_obj.was_online:
                # Ensure was_online is timezone-aware (Telethon usually provides UTC)
                current_event_time = status_obj.was_online
                if current_event_time.tzinfo is None:
                    current_event_time = current_event_time.replace(tzinfo=datetime.timezone.utc)
                
                logger.debug(
                    "[%s] User %s (ID: %s) is OFFLINE. Last seen: %s",
                    source,
                    username,
                    user_id,
                    current_event_time.isoformat(),
                )
                if MINIMUM_ASSUMED_ONLINE_DURATION_SECONDS > 0:
                    assumed_online_start_time = current_event_time - datetime.timedelta(
                        seconds=MINIMUM_ASSUMED_ONLINE_DURATION_SECONDS
                    )
                    assumed_online_timestamp_ms = int(
                        assumed_online_start_time.timestamp() * 1000
                    )
                    logger.info(
                        "[%s] %s - inserting assumed online record at %s (due to was_online at %s and MIN_ASSUMED_ONLINE_DUR=%ss)",
                        source,
                        username,
                        assumed_online_start_time.isoformat(),
                        current_event_time.isoformat(),
                        MINIMUM_ASSUMED_ONLINE_DURATION_SECONDS,
                    )
            else:
                current_event_time = datetime.datetime.now(datetime.timezone.utc)
                logger.warning(
                    "[%s] User %s (ID: %s) is OFFLINE but no was_online time. Using current time: %s",
                    source,
                    username,
                    user_id,
                    current_event_time.isoformat(),
                )
        elif isinstance(status_obj, (UserStatusRecently, UserStatusLastWeek, UserStatusLastMonth)):
            is_online = False
            current_event_time = datetime.datetime.now(datetime.timezone.utc)
            logger.debug(
                "[%s] User %s (ID: %s) status is %s. Marking as OFFLINE. Timestamp: %s",
                source,
                username,
                user_id,
                type(status_obj).__name__,
                current_event_time.isoformat(),
            )
        elif status_obj is None:
            is_online = False # Assume offline for None status (e.g. hidden)
            current_event_time = datetime.datetime.now(datetime.timezone.utc)
            logger.warning(
                "[%s] User %s (ID: %s) has None status. Marking as OFFLINE. Timestamp: %s",
                source,
                username,
                user_id,
                current_event_time.isoformat(),
            )
        else:
            is_online = False # Default to offline for unhandled status types
            current_event_time = datetime.datetime.now(datetime.timezone.utc)
            logger.warning(
                "[%s] Unhandled status type for %s (ID: %s): %s. Using current time, assuming OFFLINE.",
                source,
                username,
                user_id,
                type(status_obj).__name__,
            )

        # Ensure current_event_time is set and timezone-aware (should be by now)
        if current_event_time is None:
            logger.error(
                f"[{source}] CRITICAL: current_event_time resolved to None for {username}. "
                f"Defaulting to current UTC."
            )
            current_event_time = datetime.datetime.now(datetime.timezone.utc)
        elif current_event_time.tzinfo is None:
             logger.warning(
                f"[{source}] Timestamp for {username} lacked tzinfo. Assuming UTC. "
                f"Original: {current_event_time}"
            )
             current_event_time = current_event_time.replace(tzinfo=datetime.timezone.utc)

        timestamp_ms: int = int(current_event_time.timestamp() * 1000)

        # Debug log for database preparation
        if username in TARGET_DEBUG_USERS:
            log_msg_db_prep = (
                f"[{source}] DB PREP for {username} (ID: {user_id}): "
                f"Event Time: {current_event_time.isoformat()}, Is Online: {is_online}"
            )
            if assumed_online_timestamp_ms:
                assumed_dt_utc = datetime.datetime.fromtimestamp(
                    assumed_online_timestamp_ms / 1000, tz=datetime.timezone.utc
                )
                log_msg_db_prep += f", Assumed Online TS: {assumed_dt_utc.isoformat()}"
            final_dt_utc = datetime.datetime.fromtimestamp(timestamp_ms / 1000, tz=datetime.timezone.utc)
            log_msg_db_prep += f", Final TS for this record: {final_dt_utc.isoformat()}"
            logger.debug(log_msg_db_prep)

        # Insert assumed online record if applicable
        if assumed_online_timestamp_ms is not None:
            if assumed_online_timestamp_ms >= timestamp_ms:
                # Ensure assumed online is strictly before the offline record
                logger.warning(
                    "[%s] %s - Adjusting assumed online timestamp to be before offline. Original assumed: %s, final: %s",
                    source,
                    username,
                    assumed_online_timestamp_ms,
                    timestamp_ms,
                )
                assumed_online_timestamp_ms = timestamp_ms - 1 
            self.db.insert_activity(username, assumed_online_timestamp_ms, True)
            assumed_log_dt_utc = datetime.datetime.fromtimestamp(
                assumed_online_timestamp_ms / 1000, tz=datetime.timezone.utc
            )
            logger.debug(
                "[%s] DB record for %s (ID: %s): Online (assumed) at %s",
                source,
                username,
                user_id,
                assumed_log_dt_utc.isoformat(),
            )
        
        # Insert the primary status record
        self.db.insert_activity(username, timestamp_ms, is_online)
        final_record_dt_utc = datetime.datetime.fromtimestamp(timestamp_ms / 1000, tz=datetime.timezone.utc)
        logger.debug(
            "[%s] DB record for %s (ID: %s): %s at %s",
            source,
            username,
            user_id,
            "Online" if is_online else "Offline",
            final_record_dt_utc.isoformat(),
        )


    async def _handle_user_update(self, event: events.UserUpdate) -> None:
        """
        Handles user status updates received from Telethon `events.UserUpdate`.

        Args:
            event: The `events.UserUpdate` object from Telethon.
        """
        if not hasattr(event, "user_id") or event.user_id is None:
            logger.debug("Received UserUpdate event without user_id: %r", event)
            return

        user_id: int = event.user_id
        username: Optional[str] = self.user_id_map.get(user_id)

        if username:
            logger.debug(
                "Event received for tracked user: %s (ID: %s), event: %r",
                username,
                user_id,
                event,
            )
            
            status_obj: Any = None # Union[UserStatusOnline, ..., None]
            if hasattr(event, "status") and event.status is not None:
                status_obj = event.status
            elif hasattr(event, "online"): # Fallback for simpler event structures
                status_obj = UserStatusOnline() if event.online else UserStatusOffline(was_online=None)
            
            if status_obj is not None:
                await self._process_status_update(
                    username, user_id, status_obj, source="event_handler"
                )
            else:
                logger.warning(
                    "No status info (event.status/online) in UserUpdate for %s (ID: %s): %r. Cannot process.",
                    username,
                    user_id,
                    event,
                )
        else:
            # This is common for updates about non-tracked users; log once per id to reduce noise
            if not hasattr(self, "_unknown_user_logged"):
                self._unknown_user_logged = set()
            if user_id not in self._unknown_user_logged:
                self._unknown_user_logged.add(user_id)
                logger.debug(
                    "Received update for non-target user id=%s; suppressing further logs for this id",
                    user_id,
                )


    async def _poll_user_statuses(self) -> None:
        """
        Periodically polls the status of all resolved target users.

        This runs as a background asyncio Task and continues until the
        `shutdown_event` is set or `is_running` becomes False.
        Handles client disconnections and retries polling after a delay.
        """
        await asyncio.sleep(5) # Initial delay before first polling cycle
        logger.info(
            "Starting periodic user status polling every %s seconds.",
            USER_STATUS_POLL_INTERVAL_SECONDS,
        )

        while not self._shutdown_is_set() and self.is_running:
            assert self.client is not None, "Client must be initialized for polling."
            if not self.client.is_connected():
                logger.warning("Polling: Client not connected. Waiting for reconnection.")
                try:
                    # Wait for shutdown signal or timeout, tolerant if event is not created yet
                    signaled = await self._shutdown_wait(float(USER_STATUS_POLL_INTERVAL_SECONDS))
                    if signaled:
                        break
                except asyncio.CancelledError:
                    logger.info("Polling task cancelled during client disconnected wait.")
                    break
                continue # Retry connection check

            if not self.user_id_map:
                logger.debug("Polling: No resolved users to poll. Will check again later.")
                try:
                    signaled = await self._shutdown_wait(float(USER_STATUS_POLL_INTERVAL_SECONDS))
                    if signaled:
                        break
                except asyncio.CancelledError:
                    logger.info("Polling task cancelled during no users wait.")
                    break
                continue

            logger.info(
                "Polling run: Checking status for %s users: %s...",
                len(self.user_id_map),
                list(self.user_id_map.values())[:5],
            )
            # Iterate over a copy in case the map is modified (e.g., user deactivated)
            for user_id, username in list(self.user_id_map.items()): 
                if self.shutdown_event.is_set(): break
                try:
                    logger.debug(
                        "Polling: Requesting entity for %s (ID: %s)", username, user_id
                    )
                    entity: User = await self.client.get_entity(user_id) # type: ignore
                    
                    if entity and hasattr(entity, 'status'):
                        logger.debug(
                            "Polling: Entity for %s, status: %s",
                            username,
                            type(entity.status).__name__,
                        )
                        await self._process_status_update(
                            username, user_id, entity.status, source="polling"
                        )
                    elif entity: # Entity found but no status attribute
                        logger.warning(
                            "Polling: Entity for %s lacks 'status' attribute. Entity: %r. Treating as offline.",
                            username,
                            entity,
                        )
                        await self._process_status_update(
                            username, user_id, None, source="polling_no_status_attr"
                        )
                    else: # No entity found
                        logger.warning(
                            "Polling: Could not get entity for %s (ID: %s). Treating as offline.",
                            username,
                            user_id,
                        )
                        await self._process_status_update(
                            username, user_id, None, source="polling_no_entity"
                        )

                except errors.UserDeactivatedBanError:
                    logger.warning(f"Polling: User {username} (ID: {user_id}) is deactivated/banned.")
                    # Optionally remove from self.user_id_map here if desired
                except errors.UserIdInvalidError:
                     logger.warning(f"Polling: User ID {user_id} for {username} is invalid.")
                except errors.FloodWaitError as e_flood_poll:
                    logger.warning(
                        f"Polling: Flood wait for {username}: {e_flood_poll}. Waiting {e_flood_poll.seconds}s."
                    )
                    try: await asyncio.sleep(e_flood_poll.seconds)
                    except asyncio.CancelledError: break # Exit user loop if cancelled
                except (errors.RPCError, ConnectionError, TimeoutError, asyncio.TimeoutError) as e_net_poll:
                    logger.error(f"Polling: Network/RPC error for {username}: {e_net_poll}. Will retry next cycle.")
                except asyncio.CancelledError:
                    logger.info("Polling for user %s cancelled.", username)
                    break # Exit user loop
                except Exception as e_poll_user:
                    logger.error(f"Polling: Unexpected error for {username}: {e_poll_user}", exc_info=True)
                
                # Small delay between polling individual users if many users and short interval
                if len(self.user_id_map) > 10 and USER_STATUS_POLL_INTERVAL_SECONDS < 30:
                     try: await asyncio.sleep(0.5)
                     except asyncio.CancelledError: break # Exit user loop
            
            if self._shutdown_is_set(): break # Exit main while loop

            logger.debug(
                "Polling cycle complete. Waiting %s s for next run.",
                USER_STATUS_POLL_INTERVAL_SECONDS,
            )
            try:
                signaled = await self._shutdown_wait(float(USER_STATUS_POLL_INTERVAL_SECONDS))
                if signaled:
                    break # Shutdown triggered during wait
            except asyncio.TimeoutError:
                pass # Expected timeout, continue polling
            except asyncio.CancelledError:
                logger.info("Polling task cancelled during main sleep.")
                break
        logger.info("Periodic user status polling task finished.")


    def get_activity_data(self) -> pd.DataFrame:
        """
        Retrieves and formats activity data for all tracked users from the database.

        Returns:
            A pandas DataFrame where the index is a UTC-aware DatetimeIndex,
            columns are usernames, and values are boolean (True for online,
            False for offline). Missing users will have columns of NA/False.
        """
        if not self.target_usernames:
            logger.warning("No target users specified for CorrelationTracker. Returning empty DataFrame.")
            return pd.DataFrame()

        logger.debug("Fetching activity data for usernames: %s", self.target_usernames)
        raw_activity: List[Tuple[str, int, bool]] = self.db.get_all_activity_for_users(
            self.target_usernames
        )

        if not raw_activity:
            logger.info("No activity found in DB for users: %s", self.target_usernames)
            # Create an empty DataFrame with correct structure
            empty_df = pd.DataFrame(columns=["timestamp"] + self.target_usernames)
            empty_df["timestamp"] = pd.to_datetime(empty_df["timestamp"], utc=True)
            empty_df = empty_df.set_index("timestamp")
            for col in self.target_usernames:
                empty_df[col] = pd.Series(dtype="boolean")
            return empty_df

        df = pd.DataFrame(raw_activity, columns=["username", "timestamp", "online"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["online"] = df["online"].astype("boolean")

        # Pivot to get timeseries per user
        pivot_df: pd.DataFrame = df.pivot_table(
            index="timestamp", columns="username", values="online", aggfunc="first"
        )
        
        # Ensure all target users are columns, even if they have no data
        for user_col in self.target_usernames:
            if user_col not in pivot_df.columns:
                pivot_df[user_col] = pd.Series(dtype="boolean") 

        return pivot_df


    async def start_tracking(self) -> None:
        """
        Starts the activity tracking process.

        This involves connecting the Telegram client, resolving target user IDs,
        starting the status polling task, and adding an event handler for real-time
        user updates. The method will run until `stop_tracking` is called or an
        uncaught critical error occurs.
        """
        logger.info("Starting Correlation Tracker")
        self.is_running = True
        # Create shutdown event bound to the current running loop if not yet created
        if self.shutdown_event is None:
            self.shutdown_event = asyncio.Event()
        else:
            self.shutdown_event.clear()

        while self.is_running and not self.shutdown_event.is_set():
            try:
                await self._ensure_client_connected()
                logger.info("Telegram client connected and authorized.")
                assert self.client is not None, "Client must be available after _ensure_client_connected"
                
                await self._resolve_target_user_ids()
                
                if not self.user_id_map:
                    logger.warning(
                        "No target users resolved. Polling will not start. "
                        "Event listener will be active."
                    )
                elif self.polling_task is None or self.polling_task.done():
                    logger.info("Starting user status polling task...")
                    self.polling_task = asyncio.create_task(self._poll_user_statuses())
                else:
                    logger.info("Polling task already running or scheduled.")

                if not self._handler_registered:
                    self.client.add_event_handler(self._handle_user_update, events.UserUpdate)
                    self._handler_registered = True
                    logger.info("Event handler added for user status updates.")
                users_being_tracked_str = f"{list(self.user_id_map.values())[:5]}..." if len(self.user_id_map) > 5 else list(self.user_id_map.values())
                logger.info(
                    "Entering main event loop. Client connected: %s. Tracking %s users: %s",
                    self.client.is_connected(),
                    len(self.user_id_map),
                    users_being_tracked_str,
                )
                
                loop_counter = 0
                while self.client.is_connected() and self.is_running and not self._shutdown_is_set():
                    try:
                        await asyncio.sleep(1) # Keep event loop responsive
                        loop_counter = (loop_counter + 1) % 300 # Reset every 5 mins
                        if loop_counter == 0:
                           logger.debug(
                               "Main event loop alive. Client connected. Tracking %s active users.",
                               len(self.user_id_map),
                           )
                    except asyncio.CancelledError:
                        logger.info("Main event loop sleep cancelled.")
                        self.is_running = False # Trigger exit from outer while
                        break
                
                if not self.client.is_connected() and self.is_running and not self._shutdown_is_set():
                    logger.warning("Client disconnected. Attempting to reconnect...")
                    await asyncio.sleep(5) # Brief pause before retry by outer loop

            except ValueError as ve: # From _initialize_client (API ID/Hash missing)
                logger.error(f"Configuration error: {ve}. Tracker cannot start.")
                self.is_running = False
            except ConnectionRefusedError as ce: # From _ensure_client_connected
                logger.error(f"Authorization error: {ce}. Tracker cannot start.")
                self.is_running = False
            except (errors.PhoneNumberInvalidError, errors.SessionPasswordNeededError, errors.ApiIdInvalidError, errors.UserIsBotError) as auth_err:
                logger.error(f"Critical Telegram auth/config error: {auth_err}. Tracker cannot continue.")
                self.is_running = False
            except (errors.RPCError, ConnectionError, TimeoutError, asyncio.TimeoutError) as conn_e:
                logger.error(f"Connection/RPC error in main tracking loop: {conn_e}. Will attempt recovery.")
                if self.client and self.client.is_connected():
                    try: await self.client.disconnect()
                    except Exception as disc_e: logger.error(f"Error during disconnect: {disc_e}")
                try: await asyncio.sleep(15) # Wait before outer loop retries
                except asyncio.CancelledError: self.is_running = False
            except asyncio.CancelledError:
                logger.info("Tracker's start_tracking task was cancelled.")
                self.is_running = False
            except Exception as e_start:
                logger.error(f"Unexpected error in start_tracking: {e_start}", exc_info=True)
                try: await asyncio.sleep(30) # Wait before outer loop retries
                except asyncio.CancelledError: self.is_running = False
            
            if not self.is_running: break # Exit outer while loop if error handlers set is_running=False
        
        logger.info("Tracker main loop and initial setup ended.")
        await self.stop_tracking() # Ensure cleanup if loop exited


    async def stop_tracking(self) -> None:
        """
        Gracefully stops the activity tracker.

        Signals all running tasks to shut down, cancels the polling task,
        disconnects the Telegram client, and closes the database connection.
        """
        logger.info("Initiating shutdown of tracker components...")
        self.is_running = False 
        if self.shutdown_event is not None:
            self.shutdown_event.set()

        if self.polling_task and not self.polling_task.done():
            logger.info("Attempting to cancel polling task...")
            self.polling_task.cancel()
            try:
                await self.polling_task
                logger.info("Polling task finished after cancellation.")
            except asyncio.CancelledError:
                logger.info("Polling task was cancelled successfully.")
            except Exception as e_poll_stop:
                logger.error(f"Error during polling task shutdown: {e_poll_stop}", exc_info=True)
        
        if self.client:
            # Remove event handler before disconnecting to prevent issues during shutdown
            try:
                if self._handler_registered:
                    self.client.remove_event_handler(self._handle_user_update, events.UserUpdate)
                    self._handler_registered = False
            except Exception as e_remove_handler:
                 logger.debug(f"Could not remove event handler (already removed or client issue): {e_remove_handler}")
            
            if self.client.is_connected():
                logger.info("Disconnecting Telegram client...")
                try:
                    await self.client.disconnect()
                    logger.info("Telegram client disconnected.")
                except Exception as e_disc:
                    logger.error(f"Error disconnecting client: {e_disc}", exc_info=True)
                self.client = None # Clear client reference
        
        if self.db:
            logger.info("Closing database connection...")
            self.db.close()
            logger.info("Database connection closed.")
        
        logger.info("Tracker shutdown sequence complete")

    # ---- Internal helpers for shutdown handling ----
    def _shutdown_is_set(self) -> bool:
        try:
            return bool(self.shutdown_event and self.shutdown_event.is_set())
        except Exception:
            return False

    async def _shutdown_wait(self, timeout: float) -> bool:
        """Wait for shutdown_event up to timeout seconds.
        Returns True if signaled, False if timed out or event not yet created.
        """
        if self.shutdown_event is None:
            return False
        try:
            await asyncio.wait_for(self.shutdown_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False
        except Exception:
            return False


# --- Main Entry Point (for script execution) ---

async def main_tracker_entrypoint(usernames_env_var_value: str) -> None:
    """
    Configures and starts the CorrelationTracker based on an environment variable.

    Args:
        usernames_env_var_value: The string value from the TARGET_USERS
                                   environment variable, expected to be a
                                   comma-separated list of usernames.
    """
    if not usernames_env_var_value:
        logger.error("TARGET_USERS environment variable is not set or empty.")
        return
    
    target_users_list: List[str] = [
        u.strip() for u in usernames_env_var_value.split(",") if u.strip()
    ]
    if not target_users_list:
        logger.error("No valid target usernames parsed from TARGET_USERS.")
        return

    # Construct default db_path in project root (e.g., tgTrax/tgTrax.db)
    default_db_name = "tgTrax.db"
    db_path_main = os.path.join(PROJECT_ROOT, default_db_name)
    
    tracker = CorrelationTracker(
        target_usernames=target_users_list, 
        db_path=db_path_main
    )
    
    try:
        await tracker.start_tracking()
    except asyncio.CancelledError:
        logger.info("Main tracker entrypoint received cancellation. Shutting down tracker.")
    except Exception as e_main:
        logger.error(f"Unhandled exception in main_tracker_entrypoint: {e_main}", exc_info=True)
    finally:
        logger.info("Ensuring tracker is stopped from main_tracker_entrypoint finalization.")
        # Check tracker status before attempting to stop if it was never fully started or already stopped.
        if tracker and (tracker.is_running or (tracker.shutdown_event is not None and tracker.shutdown_event.is_set() is False)):
            logger.warning("Tracker was still marked as running or not fully shut down. Forcing stop.")
            await tracker.stop_tracking()
        else:
            logger.info("Tracker already stopped or was not running.")

async def _shutdown_wait(self, timeout: float) -> bool:
        """Wait for shutdown_event up to timeout seconds.
        Returns True if signaled, False if timed out or event not yet created.
        """
        if self.shutdown_event is None:
            # No event yet; just sleep for timeout
            try:
                await asyncio.sleep(timeout)
            except asyncio.CancelledError:
                raise
            return False
        try:
            await asyncio.wait_for(self.shutdown_event.wait(), timeout=float(timeout))
            return True
        except asyncio.TimeoutError:
            return False



# --- Script Execution (`if __name__ == '__main__':`) ---

if __name__ == "__main__":
    logger.info("Running CorrelationTracker module directly (__main__)...")
    
    # Configure TUI logger level if desired for standalone run, e.g., for more verbose debug:
    # import logging as std_logging
    # logger.setLevel(std_logging.DEBUG) 
    
    target_users_from_env: Optional[str] = os.getenv("TARGET_USERS")
    if target_users_from_env:
        # Prefer asyncio.run for Python 3.7+ if loop management is simpler
        # For compatibility or specific loop needs, get_event_loop is used here.
        main_event_loop = asyncio.get_event_loop()
        tracker_task: Optional[asyncio.Task] = None
        try:
            tracker_task = main_event_loop.create_task(main_tracker_entrypoint(target_users_from_env))
            main_event_loop.run_until_complete(tracker_task)
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt in __main__. Initiating graceful shutdown...")
            if tracker_task and not tracker_task.done():
                tracker_task.cancel() # Request cancellation of the main task
                # Allow the task to process cancellation
                main_event_loop.run_until_complete(asyncio.wait([tracker_task], timeout=10.0))
        except Exception as e_script:
            logger.error(f"Unhandled exception during __main__ execution: {e_script}", exc_info=True)
        finally:
            # Additional cleanup for any other tasks if necessary
            # This part primarily ensures the main_tracker_entrypoint (and its tracker) cleans up.
            # If tracker_task was created and might not have cleaned up fully due to abrupt exit:
            if tracker_task and not tracker_task.done():
                logger.info("Ensuring main tracker task is cancelled in __main__ finally block.")
                tracker_task.cancel()
                try:
                    main_event_loop.run_until_complete(tracker_task) # Allow cancellation to complete
                except asyncio.CancelledError:
                    logger.info("Main tracker task successfully cancelled in finally.")
                except Exception as e_final_cancel:
                    logger.error(f"Exception during final cancellation of main task: {e_final_cancel}")
            
            if main_event_loop.is_running():
                 logger.info("Closing __main__ event loop.")
                 main_event_loop.close()
            logger.info("Tracker (__main__ execution finished)")
    else:
        logger.error(
            "TARGET_USERS not set in .env or environment. "
            "Cannot start tracker from __main__."
        ) 
