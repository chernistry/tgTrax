"""Telegram client authentication module for tgTrax.

Handles credential validation and Telegram client initialization, including:
- API ID/Hash verification.
- Phone number validation.
- Interactive login flow with code and 2FA password handling.
- Detection and prevention of bot account usage.
"""

import logging # For TuiLoggerAdapter constants & type hints
import sys # For sys.exit (though not directly used in this refactored version)
import traceback # For detailed exception formatting if needed by logger
from typing import Optional, Union # For Python 3.9+ style Optional and Union

from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError, UserIsBotError
from telethon.tl.types import User # For type hinting `me`

# Assuming tui and TuiLoggerAdapter are correctly importable from tgTrax.utils
from tgTrax.utils import tui
from tgTrax.utils.logger_adapter import TuiLoggerAdapter


# Initialize logger for this module
logger = TuiLoggerAdapter(tui)
logger.setLevel(logging.INFO) # Default log level for auth operations


# ==== CREDENTIAL VALIDATION ==== #

def ensure_credentials(
    api_id: Optional[int],
    api_hash: Optional[str],
    phone_number: Optional[str] = None,
) -> bool:
    """Ensures that essential Telegram API credentials (ID and Hash) are provided.

    Logs a critical error and returns False if API ID or Hash is missing.
    Logs a warning if the phone number is not provided, as interactive login
    will then be necessary.

    Args:
        api_id: The Telegram API ID. Expected to be an integer.
        api_hash: The Telegram API Hash. Expected to be a string.
        phone_number: Optional. The user's phone number. If not provided,
                      a warning is logged as interactive login will be needed.

    Returns:
        True if API ID and Hash are present (even if phone_number is missing),
        False if API ID or API Hash is missing.
    """
    if not api_id or not api_hash:
        logger.critical(
            "TELEGRAM_API_ID and TELEGRAM_API_HASH must be set in your configuration. "
            "Cannot proceed without these core credentials."
        )
        return False

    if not phone_number:
        logger.warning(
            "TELEGRAM_PHONE_NUMBER is not set. Interactive login will be required "
            "if the session is not already authorized."
        )
    return True


# ==== TELEGRAM CLIENT INITIALIZATION ==== #

async def get_client(
    session_name: str,
    api_id: int,
    api_hash: str,
    phone_number: Optional[str] = None,
) -> Optional[TelegramClient]: # Changed from TelegramClient | None for broader compatibility
    """Initializes, connects, and ensures authorization for the Telegram client.

    Handles interactive login flow, including code requests and 2FA passwords.
    Prevents usage with bot accounts, as tgTrax is designed for user accounts.

    Args:
        session_name: The name for the Telethon session file (e.g., "tgTrax_session").
        api_id: The Telegram API ID.
        api_hash: The Telegram API Hash.
        phone_number: Optional. The user's phone number. If provided, it's used
                      for explicit sign-in attempts. If None, `client.start()`
                      will handle interactive prompts for phone, code, and 2FA.

    Returns:
        An authorized `TelegramClient` instance if successful, or `None` if connection
        or authorization fails for any reason.

    Raises:
        UserIsBotError: If the provided credentials belong to a bot account,
                        which is not supported for user-centric operations in tgTrax.
                        This is re-raised to be handled by the caller.
    """
    logger.info(f"Initializing Telegram client with session: '{session_name}'")
    client = TelegramClient(session_name, api_id, api_hash)

    try:
        logger.info("Attempting to connect to Telegram...")
        await client.connect()

        if not client.is_connected():
            logger.error("Failed to connect to Telegram after connect() call.")
            return None
        
        tui.tui_print_success("Successfully connected to Telegram.")

        if not await client.is_user_authorized():
            logger.warning("User is not authorized. Initiating login process...")
            try:
                if phone_number:
                    logger.info(f"Attempting sign-in with phone number: {phone_number}")
                    await client.send_code_request(phone_number)
                    # Prompt styling for better visibility
                    code_prompt = (
                        f"A code was sent to {phone_number}. "
                        "Please enter the code here: "
                    )
                    tui.tui_print_highlight(code_prompt)
                    code = input() # Read input after TUI prompt
                    await client.sign_in(phone=phone_number, code=code)
                else:
                    logger.info(
                        "Phone number not provided. Relying on client.start() "
                        "for interactive login (phone, code, 2FA if needed)."
                    )
                    # `client.start()` will prompt for phone, code, and 2FA password internally.
                    await client.start() 

                # Verify authorization status after sign_in or start attempt
                if not await client.is_user_authorized():
                    logger.error(
                        "Login attempt failed or was incomplete. "
                        "User remains unauthorized."
                    )
                    await client.disconnect()
                    return None
                    tui.tui_print_success("Login successful. User is now authorized.")

            except SessionPasswordNeededError:
                logger.warning("Two-factor authentication (2FA) is required.")
                tui.tui_print_highlight("Please enter your Telegram password (2FA): ")
                password = input() # Read input after TUI prompt
                try:
                    await client.sign_in(password=password)
                    if not await client.is_user_authorized():
                        logger.error(
                            "2FA login failed. Password incorrect or other issue. "
                            "User remains unauthorized."
                        )
                        await client.disconnect()
                        return None
                    tui.tui_print_success("2FA login successful. User authorized.")
                except Exception as e_2fa:
                    logger.error(f"Error during 2FA password entry: {e_2fa}", exc_info=True)
                    await client.disconnect()
                    return None
            except UserIsBotError as e_bot_login:
                logger.critical(
                    "Login attempt with bot credentials failed. "
                    "tgTrax requires a user account. Please use user credentials."
                )
                await client.disconnect()
                raise  # Re-raise to signal critical issue to the caller
            except Exception as e_login:
                logger.error(f"An unexpected error occurred during the login process: {e_login}", exc_info=True)
                await client.disconnect()
                return None
        else:
            # User is already authorized, let's get and log who they are.
            me: Optional[User] = await client.get_me()
            if me:
                if me.bot:
                    logger.critical(
                        f"Logged in as Bot: {me.username or me.id}. "
                        "tgTrax requires a user account, not a bot account."
                    )
                    await client.disconnect()
                    raise UserIsBotError(
                        request=None,  # type: ignore[arg-type]
                        message="Cannot use tgTrax with a bot account."
                    )
                msg = f"User already authorized. Connected as: {me.username or me.first_name or me.id}"
                tui.tui_print_success(msg)
            else:
                 # This case should ideally not happen if client.is_user_authorized() is true.
                 logger.warning(
                    "User is marked as authorized, but get_me() returned None. "
                    "This is an unexpected state."
                 )

        return client

    except ConnectionError as e_conn:
        logger.error(
            f"Telegram connection error: {e_conn}. "
            "Check network connectivity and Telegram service status."
        )
        return None
    except UserIsBotError: # Catch re-raised UserIsBotError from inner blocks
        # Critical error already logged. Ensure client is disconnected and propagate.
        if client and client.is_connected():
            await client.disconnect()
        raise
    except Exception as e_outer:
        logger.error(
            f"An unexpected outer error occurred in get_client: {e_outer}",
            exc_info=True,
        )
        if client and client.is_connected():
            await client.disconnect()
        return None 
