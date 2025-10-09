# ==== SQLITE DATABASE OPERATIONS MODULE ==== #
"""
SQLite database helpers for persisting Telegram user activity.

Responsibilities:
- Create and manage the database connection.
- Ensure presence of the `activity` table with expected schema.
- Provide CRUD-like helpers without changing data semantics.
"""

import os
import logging
import sqlite3
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ==== SQLITE DATABASE CLASS ==== #

class SQLiteDatabase:
    """
    Manages connection and operations for an SQLite database storing user activity.

    Attributes:
        db_path (str): The file path to the SQLite database.
        connection (Optional[sqlite3.Connection]): The active SQLite connection 
            object, or None if not connected.
    """

    def __init__(self, db_path: str = "tgTrax_activity.db") -> None:
        """
        Initialize the database and establish a connection.

        Postconditions:
        - Connection is attempted and tables are validated/created.

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
        except sqlite3.Error:
            if self.connection:
                try:
                    self.connection.close()
                except sqlite3.Error:
                    pass
            self.connection = None
            logger.error(
                "SQLiteDatabase initialization failed for %s. Database might be unusable.",
                self.db_path,
            )


    def _connect(self) -> None:
        """
        Establish a connection to the SQLite database.

        Notes:
        - Uses timeout and WAL for better multi-process tolerance.

        Raises:
            sqlite3.Error: If the connection to the database fails, allowing the
                           caller (e.g., __init__) to handle it.
        """
        try:
            self.connection = sqlite3.connect(
                self.db_path, 
                timeout=15.0, 
                check_same_thread=False
            )
            
            try:
                self.connection.execute("PRAGMA journal_mode=WAL;")
                self.connection.execute("PRAGMA busy_timeout=5000;")
            except sqlite3.Error:
                pass
                
            logger.info("SQLite connected. Path=%s, Timeout=15s", self.db_path)
        except sqlite3.Error as e:
            logger.error(
                "Failed to connect to SQLite database at '%s': %s", self.db_path, e
            )
            self.connection = None
            raise


    def _create_tables(self) -> None:
        """
        Create the `activity` table if missing with a unique constraint.

        Raises:
            sqlite3.Error: If table creation fails, allowing the caller to handle.
        """
        if not self.connection:
            logger.error("Database connection not available. Cannot create tables.")
            raise sqlite3.OperationalError(
                "No database connection to create tables."
            )

        create_table_sql = """
            CREATE TABLE IF NOT EXISTS activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                online BOOLEAN NOT NULL,
                UNIQUE(username, timestamp)
            );
        """
        
        try:
            cursor: sqlite3.Cursor = self.connection.cursor()
            cursor.execute(create_table_sql)
            self.connection.commit()
            logger.info("Validated table 'activity' presence.")
        except sqlite3.Error as e:
            logger.error("Error creating/validating 'activity' table: %s", e)
            raise


    def insert_activity(
        self,
        username: str,
        timestamp: int,
        online: bool,
    ) -> bool:
        """
        Inserts or updates a user activity record using UPSERT.

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
            logger.error("Database connection not available. Cannot insert activity.")
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
            
            logger.debug(
                "Upserted activity: user=%s online=%s ts=%s", username, online, timestamp
            )
            return True
        except sqlite3.Error as e:
            logger.error("Error inserting/updating activity for '%s': %s", username, e)
            return False


    def get_all_activity_for_users(
        self,
        usernames: List[str],
    ) -> List[Tuple[str, int, bool]]:
        """
        Retrieves all activity records for a list of specified usernames.

        Args:
            usernames: A list of usernames for which to fetch activity records.
                       An empty list will result in no records being fetched.

        Returns:
            A list of tuples, where each tuple is (username, timestamp, online),
            ordered by timestamp. Returns an empty list if no connection, no
            usernames are provided, or an error occurs.
        """
        if not self.connection:
            logger.error("Database connection not available. Cannot fetch activity.")
            return []
            
        if not usernames:
            logger.warning(
                "No usernames provided to get_all_activity_for_users. Returning empty list."
            )
            return []

        try:
            cursor: sqlite3.Cursor = self.connection.cursor()
            placeholders = ", ".join(["?" for _ in usernames])
            query = (
                f"SELECT username, timestamp, online FROM activity "
                f"WHERE username IN ({placeholders}) ORDER BY timestamp ASC"
            )
            cursor.execute(query, usernames)
            
            results: List[Tuple[Any, ...]] = cursor.fetchall()
            return [
                (str(row[0]), int(row[1]), bool(row[2])) for row in results
            ]
        except sqlite3.Error as e:
            logger.error(
                "Error fetching activity for users (%s users): %s", len(usernames), e
            )
            return []
        except Exception as e_general:
            logger.error("Unexpected error in get_all_activity_for_users: %s", e_general)
            return []


    def close(self) -> None:
        """
        Closes the connection to the SQLite database if it is open.

        Ensures the connection attribute is reset to None after closing.
        """
        if self.connection:
            try:
                self.connection.close()
                logger.info("SQLite connection closed. Path=%s", self.db_path)
            except sqlite3.Error as e:
                logger.error("Error closing database connection: %s", e)
            finally:
                self.connection = None


# (Module self-test block removed.)
