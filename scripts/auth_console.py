from __future__ import annotations

import argparse
import asyncio
import os
from typing import Optional

from dotenv import load_dotenv
from telethon import TelegramClient, errors as tg_errors


def load_env(project_root: str) -> None:
    env_path = os.path.join(project_root, ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)


async def do_login(phone: str, code: Optional[str], password: Optional[str], session_name: str) -> int:
    api_id = os.getenv("TELEGRAM_API_ID")
    api_hash = os.getenv("TELEGRAM_API_HASH")
    if not api_id or not api_hash:
        print("ERROR: TELEGRAM_API_ID or TELEGRAM_API_HASH is missing in .env or environment.")
        return 2

    try:
        api_id_int = int(api_id)
    except ValueError:
        print("ERROR: TELEGRAM_API_ID must be an integer.")
        return 2

    client = TelegramClient(session_name, api_id_int, api_hash)
    print("[auth] Connecting (timeout 15s)...")
    try:
        await asyncio.wait_for(client.connect(), timeout=15)
    except asyncio.TimeoutError:
        print("ERROR: Connect timeout. Check network or try again.")
        return 3

    if await client.is_user_authorized():
        me = await client.get_me()
        print(f"[auth] Already authorized as: {getattr(me, 'username', None) or getattr(me, 'first_name', None) or me.id}")
        await client.disconnect()
        return 0

    if not code:
        print(f"[auth] Sending login code to {phone}...")
        try:
            await client.send_code_request(phone)
        except Exception as e:
            print(f"ERROR: Failed to send code: {e}")
            await client.disconnect()
            return 4
        code = input("Enter the code you received: ").strip()

    try:
        await client.sign_in(phone=phone, code=code)
    except tg_errors.SessionPasswordNeededError:
        if not password:
            password = input("2FA password required: ")
        await client.sign_in(password=password)
    except Exception as e:
        print(f"ERROR: sign_in failed: {e}")
        await client.disconnect()
        return 5

    ok = await client.is_user_authorized()
    await client.disconnect()
    if ok:
        print("[auth] Authorization successful. Session is ready.")
        return 0
    print("ERROR: Authorization failed. Please verify the code/password and retry.")
    return 6


def main() -> int:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_env(project_root)

    parser = argparse.ArgumentParser(description="tgTrax console login")
    parser.add_argument("--phone", type=str, help="Phone number with country code, e.g. +1234567890")
    parser.add_argument("--code", type=str, help="Login code (if already received)")
    parser.add_argument("--password", type=str, help="2FA password (if enabled)")
    parser.add_argument("--session-name", type=str, default=os.getenv("TELEGRAM_SESSION_NAME", "tgTrax_session"))
    args = parser.parse_args()

    phone = (args.phone or os.getenv("TELEGRAM_PHONE_NUMBER") or "").strip()
    if not phone:
        phone = input("Enter phone number (+countrycode...): ").strip()

    return asyncio.run(do_login(phone, args.code, args.password, args.session_name))


if __name__ == "__main__":
    raise SystemExit(main())

