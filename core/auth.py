# ==== TELEGRAM CLIENT AUTHENTICATION MODULE ==== #
"""
Telegram client authentication helpers for tgTrax.

This module:
- Verifies API credentials presence.
- Initializes a Telethon client and ensures authorization.
- Guides interactive flows (code and optional 2FA) without altering behavior.
- Prevents bot-account usage for user-centric features.
"""

import logging
import sys
import traceback
from typing import Optional, Union

from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError, UserIsBotError
from telethon.tl.types import User

 


# --► MODULE INITIALIZATION
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)




# ==== CREDENTIAL VALIDATION ==== #

def ensure_credentials(
    api_id: Optional[int],
    api_hash: Optional[str],
    phone_number: Optional[str] = None,
) -> bool:
    """
    Validate presence of essential Telegram API credentials.

    Preconditions:
    - `api_id` and `api_hash` must be provided to proceed.
    - `phone_number` is optional; absence implies interactive prompts later.

    Args:
        api_id: The Telegram API ID. Expected to be an integer.
        api_hash: The Telegram API Hash. Expected to be a string.
        phone_number: Optional. The user's phone number. If not provided,
                      a warning is logged as interactive login will be needed.

    Returns:
        bool: True if both API ID and Hash are present; False otherwise.
    """
    if not api_id or not api_hash:
        logger.critical(
            "TELEGRAM_API_ID and TELEGRAM_API_HASH must be set in your "
            "configuration. Cannot proceed without these core credentials."
        )
        return False

    if not phone_number:
        logger.warning(
            "TELEGRAM_PHONE_NUMBER is not set. Interactive login will be "
            "required if the session is not already authorized."
        )
    return True


# ==== TELEGRAM CLIENT INITIALIZATION ==== #

async def get_client(
    session_name: str,
    api_id: int,
    api_hash: str,
    phone_number: Optional[str] = None,
) -> Optional[TelegramClient]:
    """
    Initialize, connect, and ensure authorization for a Telethon client.

    Args:
        session_name: The name for the Telethon session file 
                      (e.g., "tgTrax_session").
        api_id: The Telegram API ID.
        api_hash: The Telegram API Hash.
        phone_number: Optional. The user's phone number. If provided, it's used
                      for explicit sign-in attempts. If None, `client.start()`
                      will handle interactive prompts for phone, code, and 2FA.

    Returns:
        Optional[TelegramClient]: Authorized client on success; None otherwise.

    Raises:
        UserIsBotError: If credentials belong to a bot account (re-raised).
    """
    logger.info(f"Initializing Telegram client with session: '{session_name}'")
    client = TelegramClient(session_name, api_id, api_hash)

    try:
        logger.info("Attempting to connect to Telegram...")
        await client.connect()

        if not client.is_connected():
            logger.error("Failed to connect to Telegram after connect() call.")
            return None
        
        logger.info("Successfully connected to Telegram.")

        if not await client.is_user_authorized():
            logger.warning("User is not authorized. Initiating login process...")
            
            try:
                if phone_number:
                    logger.info(
                        f"Attempting sign-in with phone number: {phone_number}"
                    )
                    await client.send_code_request(phone_number)
                    
                    code = input(
                        f"A code was sent to {phone_number}. Please enter the code here: "
                    )
                    await client.sign_in(phone=phone_number, code=code)
                else:
                    logger.info(
                        "Phone number not provided. Relying on client.start() "
                        "for interactive login (phone, code, 2FA if needed)."
                    )
                    await client.start()

                if not await client.is_user_authorized():
                    logger.error(
                        "Login attempt failed or was incomplete. "
                        "User remains unauthorized."
                    )
                    await client.disconnect()
                    return None
                    
                logger.info("Login successful. User is now authorized.")

            except SessionPasswordNeededError:
                logger.warning("Two-factor authentication (2FA) is required.")
                password = input("Please enter your Telegram password (2FA): ")
                
                try:
                    await client.sign_in(password=password)
                    if not await client.is_user_authorized():
                        logger.error(
                            "2FA login failed. Password incorrect or other issue. "
                            "User remains unauthorized."
                        )
                        await client.disconnect()
                        return None
                    logger.info("2FA login successful. User authorized.")
                except Exception as e_2fa:
                    logger.error(
                        f"Error during 2FA password entry: {e_2fa}", 
                        exc_info=True
                    )
                    await client.disconnect()
                    return None
                    
            except UserIsBotError as e_bot_login:
                logger.critical(
                    "Login attempt with bot credentials failed. "
                    "tgTrax requires a user account. Please use user credentials."
                )
                await client.disconnect()
                raise
                
            except Exception as e_login:
                logger.error(
                    f"An unexpected error occurred during the login process: "
                    f"{e_login}", 
                    exc_info=True
                )
                await client.disconnect()
                return None
        else:
            me: Optional[User] = await client.get_me()
            if me:
                if me.bot:
                    logger.critical(
                        f"Logged in as Bot: {me.username or me.id}. "
                        "tgTrax requires a user account, not a bot account."
                    )
                    await client.disconnect()
                    raise UserIsBotError(
                        request=None,
                        message="Cannot use tgTrax with a bot account."
                    )
                    
                msg = (
                    f"User already authorized. Connected as: "
                    f"{me.username or me.first_name or me.id}"
                )
                logger.info(msg)
            else:
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
        
    except UserIsBotError:
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
