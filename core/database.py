"""Manages SQLite database operations for the tgTrax application.

This module provides the `SQLiteDatabase` class, which handles the storage and
retrieval of user activity data (online/offline status with timestamps).
"""

import os
import sqlite3
from typing import Any, List, Optional, Tuple

# Assuming tui is correctly importable from tgTrax.utils
from tgTrax.utils import tui


# ==== SQLITE DATABASE CLASS ==== #

class SQLiteDatabase:
    """Manages connection and operations for an SQLite database storing user activity.

    Attributes:
        db_path (str): The file path to the SQLite database.
        connection (Optional[sqlite3.Connection]): The active SQLite connection object,
            or None if not connected.
    """

    def __init__(self, db_path: str = "tgTrax_activity.db") -> None:
        """Initializes the SQLiteDatabase instance and establishes a connection.

        Creates necessary tables if they don't already exist.

        Args:
            db_path: The file path to the SQLite database. If a relative path
                     is given, it's typically created in the current working
                     directory of script execution.
        """
        self.db_path: str = db_path
        self.connection: Optional[sqlite3.Connection] = None
        try:
            self._connect()
            self._create_tables()
        except sqlite3.Error: # Catch connection/creation errors during init
            # Error already logged by _connect or _create_tables
            # Ensure connection is None if init fails partway
            if self.connection:
                try:
                    self.connection.close()
                except sqlite3.Error:
                    pass # Avoid error during cleanup error
            self.connection = None
            tui.tui_print_error(
                f"SQLiteDatabase initialization failed for {self.db_path}. "
                "Database might be unusable."
            )
            # Depending on desired behavior, could re-raise or handle as non-fatal


    def _connect(self) -> None:
        """Establishes a connection to the SQLite database.

        Sets a connection timeout and configures `check_same_thread=False` for
        broader usability (with appropriate care if using threads).

        Raises:
            sqlite3.Error: If the connection to the database fails, allowing the
                           caller (e.g., __init__) to handle it.
        """
        try:
            # check_same_thread=False is convenient but requires careful management
            # if the same connection is accessed by multiple threads.
            self.connection = sqlite3.connect(
                self.db_path, timeout=15.0, check_same_thread=False
            )
            try:
                # Reduce lock contention
                self.connection.execute("PRAGMA journal_mode=WAL;")
                self.connection.execute("PRAGMA busy_timeout=5000;")
            except sqlite3.Error:
                pass
            # It's good practice to enable foreign key constraints if using them.
            # self.connection.execute("PRAGMA foreign_keys = ON")
            tui.tui_db_event(
                "Connected",
                "SQLite Database",
                f"Path: {self.db_path}, Timeout: 15s",
            )
        except sqlite3.Error as e:
            tui.tui_print_error(
                f"Failed to connect to SQLite database at '{self.db_path}': {e}"
            )
            self.connection = None # Ensure connection is None on failure
            raise # Re-raise for __init__ or other callers to handle


    def _create_tables(self) -> None:
        """Creates the `activity` table in the database if it doesn't exist.

        The `activity` table stores user online status updates with a unique
        constraint on (username, timestamp) to prevent duplicate entries for the
        same event.

        Raises:
            sqlite3.Error: If table creation fails, allowing the caller to handle.
        """
        if not self.connection:
            tui.tui_print_error(
                "Database connection not available. Cannot create tables."
            )
            # Raise an error or handle appropriately, as this is a critical state
            raise sqlite3.OperationalError("No database connection to create tables.")

        create_table_sql = """
            CREATE TABLE IF NOT EXISTS activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                timestamp INTEGER NOT NULL, -- Unix timestamp in milliseconds
                online BOOLEAN NOT NULL,
                UNIQUE(username, timestamp)
            );
        """
        try:
            cursor: sqlite3.Cursor = self.connection.cursor()
            cursor.execute(create_table_sql)
            self.connection.commit()
            tui.tui_db_event("Validated", "Table 'activity'", "Ensured existence")
        except sqlite3.Error as e:
            tui.tui_print_error(f"Error creating/validating 'activity' table: {e}")
            raise # Re-raise for __init__ or other callers to handle


    def insert_activity(
        self, username: str, timestamp: int, online: bool
    ) -> bool:
        """Inserts or updates a user activity record using UPSERT.

        If a record with the same username and timestamp already exists,
        it updates the `online` status. Otherwise, a new record is inserted.

        Args:
            username: The username of the user.
            timestamp: The Unix timestamp in milliseconds for the activity.
            online: A boolean indicating if the user was online (True) or
                    offline (False).

        Returns:
            True if the operation was successful, False otherwise.
        """
        if not self.connection:
            tui.tui_print_error(
                "Database connection not available. Cannot insert activity."
            )
            return False

        insert_sql = """
            INSERT INTO activity (username, timestamp, online)
            VALUES (?, ?, ?)
            ON CONFLICT(username, timestamp) DO UPDATE SET online = excluded.online;
        """
        try:
            cursor: sqlite3.Cursor = self.connection.cursor()
            cursor.execute(insert_sql, (username, timestamp, online))
            self.connection.commit()
            tui.tui_db_event(
                "Inserted/Updated",
                "Activity Record",
                f"User: {username}, Online: {online}, TS: {timestamp}",
            )
            return True
        except sqlite3.Error as e:
            tui.tui_print_error(
                f"Error inserting/updating activity for '{username}': {e}"
            )
            return False


    def get_all_activity_for_users(
        self, usernames: List[str]
    ) -> List[Tuple[str, int, bool]]:
        """Retrieves all activity records for a list of specified usernames.

        Args:
            usernames: A list of usernames for which to fetch activity records.
                       An empty list will result in no records being fetched.

        Returns:
            A list of tuples, where each tuple is (username, timestamp, online),
            ordered by timestamp. Returns an empty list if no connection, no
            usernames are provided, or an error occurs.
        """
        if not self.connection:
            tui.tui_print_error(
                "Database connection not available. Cannot fetch activity."
            )
            return []
        if not usernames:
            tui.tui_print_warning(
                "No usernames provided to get_all_activity_for_users. Returning empty list."
            )
            return []

        try:
            cursor: sqlite3.Cursor = self.connection.cursor()
            # Using f-string for IN clause placeholders is safe as placeholders are generated, not user input.
            placeholders = ", ".join(["?" for _ in usernames])
            query = (
                f"SELECT username, timestamp, online FROM activity "
                f"WHERE username IN ({placeholders}) ORDER BY timestamp ASC"
            )
            cursor.execute(query, usernames)
            # Results are List[Tuple[Any, ...]], explicitly cast during list comprehension.
            results: List[Tuple[Any, ...]] = cursor.fetchall()
            return [
                (str(row[0]), int(row[1]), bool(row[2])) for row in results
            ]
        except sqlite3.Error as e:
            tui.tui_print_error(
                f"Error fetching activity for users ({len(usernames)} users): {e}"
            )
            return []
        except Exception as e_general: # Catch any other unexpected errors
            tui.tui_print_error(
                f"Unexpected error in get_all_activity_for_users: {e_general}"
            )
            return []


    def close(self) -> None:
        """Closes the connection to the SQLite database if it is open.

        Ensures the connection attribute is reset to None after closing.
        """
        if self.connection:
            try:
                self.connection.close()
                tui.tui_db_event("Closed", "SQLite Connection", f"Path: {self.db_path}")
            except sqlite3.Error as e:
                tui.tui_print_error(f"Error closing database connection: {e}")
            finally:
                self.connection = None # Always reset to None


# ==== MODULE TEST BLOCK ==== #

if __name__ == "__main__":
    tui.tui_print_highlight("--- Testing SQLiteDatabase Module ---", style="header")

    # Use a unique name for test DB to avoid conflicts if run multiple times
    test_db_filename = "test_tgTrax_activity.db"
    db = SQLiteDatabase(test_db_filename)

    # Check if DB connection was successful during init
    if not db.connection:
        tui.tui_print_error(
            f"Failed to initialize SQLiteDatabase for testing with {test_db_filename}. Aborting tests."
        )
        # Depending on policy, could sys.exit(1) here
    else:
        tui.tui_print_success(f"Test database initialized: {test_db_filename}")
        try:
            tui.tui_print_info("\n--- Inserting Test Activity Data ---")
            # Test data with clear timestamps
            ts1_user1 = 1678886400000  # User1 online
            ts2_user1 = 1678886460000  # User1 offline
            ts1_user2 = 1678886500000  # User2 online

            db.insert_activity("test_user1", ts1_user1, True)
            db.insert_activity("test_user1", ts2_user1, False)
            db.insert_activity("test_user2", ts1_user2, True)

            # Test UPSERT: update existing record for user1 at ts1_user1 to offline
            tui.tui_print_detail("Testing UPSERT for test_user1 at first timestamp...")
            db.insert_activity("test_user1", ts1_user1, False)

            tui.tui_print_info("\n--- Fetching Activity for 'test_user1' ---")
            activities_user1 = db.get_all_activity_for_users(["test_user1"])
            if activities_user1:
                for act in activities_user1:
                    tui.tui_print_detail(f"  Record: {act}")
            else:
                tui.tui_print_warning("  No activity found for test_user1.")

            tui.tui_print_info("\n--- Fetching Activity for 'test_user1' and 'test_user2' ---")
            activities_all = db.get_all_activity_for_users(["test_user1", "test_user2"])
            if activities_all:
                for act in activities_all:
                    tui.tui_print_detail(f"  Record: {act}")
            else:
                tui.tui_print_warning("  No activity found for combined users.")

            tui.tui_print_info("\n--- Fetching Activity for Non-Existent User ('test_user3') ---")
            activities_none = db.get_all_activity_for_users(["test_user3"])
            if not activities_none:
                tui.tui_print_detail("  No activity found for 'test_user3' (as expected).")
            else:
                tui.tui_print_error("  Unexpected activity found for 'test_user3'!")
                for act in activities_none:
                    tui.tui_print_detail(f"  Record: {act}")

            tui.tui_print_info("\n--- Fetching Activity with Empty User List ---")
            activities_empty_list = db.get_all_activity_for_users([])
            if not activities_empty_list:
                tui.tui_print_detail(
                    "  No activity returned for empty user list (as expected)."
                )
            else:
                tui.tui_print_error("  Unexpectedly received data for empty user list!")

        except Exception as e_test:
            tui.tui_print_error(f"An error occurred during database tests: {e_test}", exc_info=True)
        finally:
            tui.tui_print_info("\n--- Closing Database and Cleaning Up Test File ---")
            db.close() # Ensure connection is closed
            if os.path.exists(test_db_filename):
                try:
                    os.remove(test_db_filename)
                    tui.tui_print_success(f"Test database '{test_db_filename}' removed.")
                except OSError as e_os:
                    tui.tui_print_warning(
                        f"Could not remove test database '{test_db_filename}': {e_os}"
                    )
            else:
                tui.tui_print_warning(
                    f"Test database '{test_db_filename}' was not found for removal (already cleaned up?)"
                )

    tui.tui_print_success("--- SQLiteDatabase Module Tests Completed ---", style="success") 
