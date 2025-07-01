"""
Advanced database connection manager for tgTrax application.

This module provides connection pooling, session management, and async-safe
database operations to improve performance and reliability.
"""

import asyncio
import sqlite3
import threading
from contextlib import asynccontextmanager, contextmanager
from typing import Optional, AsyncGenerator, Generator, List, Tuple, Any
from concurrent.futures import ThreadPoolExecutor

from tgTrax.core import settings
from tgTrax.core.models import ActivityRecord
from tgTrax.utils import tui


class DatabaseConnectionManager:
    """
    Manages database connections with connection pooling and async support.
    
    This class provides thread-safe database operations and can execute
    blocking database operations in a thread pool for async compatibility.
    """
    
    def __init__(
        self, 
        db_path: str = settings.DEFAULT_DB_NAME,
        max_workers: int = 4,
        timeout_seconds: float = settings.DEFAULT_DB_TIMEOUT_SECONDS
    ):
        """
        Initialize the database connection manager.
        
        Args:
            db_path: Path to the SQLite database file
            max_workers: Maximum number of worker threads for async operations
            timeout_seconds: Database connection timeout
        """
        self.db_path = db_path
        self.timeout_seconds = timeout_seconds
        self.max_workers = max_workers
        self._thread_pool: Optional[ThreadPoolExecutor] = None
        self._lock = threading.Lock()
        self._connections_created = 0
        self._is_initialized = False
        
        # Thread-local storage for connections
        self._local = threading.local()
    
    def _get_connection(self) -> sqlite3.Connection:
        """
        Get a thread-local database connection.
        
        Returns:
            SQLite connection for the current thread
        """
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            with self._lock:
                self._local.connection = sqlite3.connect(
                    self.db_path,
                    timeout=self.timeout_seconds,
                    check_same_thread=False
                )
                # Enable WAL mode for better concurrent access
                self._local.connection.execute("PRAGMA journal_mode=WAL")
                # Enable foreign key constraints
                self._local.connection.execute("PRAGMA foreign_keys=ON")
                self._connections_created += 1
                tui.tui_db_event(
                    "Created", 
                    "Connection", 
                    f"Thread connection #{self._connections_created}"
                )
        
        return self._local.connection
    
    def _init_thread_pool(self) -> None:
        """Initialize the thread pool for async operations."""
        if self._thread_pool is None:
            self._thread_pool = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="db_worker"
            )
            self._is_initialized = True
    
    @contextmanager
    def get_sync_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Context manager for synchronous database operations.
        
        Yields:
            SQLite connection with automatic transaction handling
        """
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    
    async def execute_async(
        self, 
        query: str, 
        params: Optional[Tuple] = None
    ) -> Optional[List[Tuple]]:
        """
        Execute a database query asynchronously.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Query results if it's a SELECT, None for other operations
        """
        self._init_thread_pool()
        
        def _execute():
            with self.get_sync_connection() as conn:
                cursor = conn.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                
                # Return results for SELECT queries
                if query.strip().upper().startswith('SELECT'):
                    return cursor.fetchall()
                return None
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._thread_pool, _execute)
    
    async def insert_activity_async(self, record: ActivityRecord) -> bool:
        """
        Insert an activity record asynchronously.
        
        Args:
            record: Validated activity record
            
        Returns:
            True if successful, False otherwise
        """
        query = """
            INSERT INTO activity (username, timestamp, online)
            VALUES (?, ?, ?)
            ON CONFLICT(username, timestamp) DO UPDATE SET online = excluded.online;
        """
        
        try:
            await self.execute_async(
                query, 
                (record.username, record.timestamp, record.online)
            )
            tui.tui_db_event(
                "Inserted/Updated",
                "Activity Record",
                f"User: {record.username}, Online: {record.online}, Source: {record.source}"
            )
            return True
        except Exception as e:
            tui.tui_print_error(f"Error inserting activity record: {e}")
            return False
    
    async def get_activity_async(
        self, 
        usernames: List[str]
    ) -> List[Tuple[str, int, bool]]:
        """
        Retrieve activity records asynchronously.
        
        Args:
            usernames: List of usernames to fetch
            
        Returns:
            List of activity tuples (username, timestamp, online)
        """
        if not usernames:
            return []
        
        placeholders = ", ".join(["?" for _ in usernames])
        query = f"""
            SELECT username, timestamp, online 
            FROM activity 
            WHERE username IN ({placeholders}) 
            ORDER BY timestamp ASC
        """
        
        try:
            results = await self.execute_async(query, tuple(usernames))
            return [
                (str(row[0]), int(row[1]), bool(row[2])) 
                for row in (results or [])
            ]
        except Exception as e:
            tui.tui_print_error(f"Error fetching activity records: {e}")
            return []
    
    def create_tables_sync(self) -> bool:
        """
        Create database tables synchronously.
        
        Returns:
            True if successful, False otherwise
        """
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
            with self.get_sync_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(create_table_sql)
                tui.tui_db_event("Created/Validated", "Table 'activity'", "Schema ensured")
                return True
        except Exception as e:
            tui.tui_print_error(f"Error creating tables: {e}")
            return False
    
    async def create_tables_async(self) -> bool:
        """
        Create database tables asynchronously.
        
        Returns:
            True if successful, False otherwise
        """
        self._init_thread_pool()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._thread_pool, self.create_tables_sync)
    
    def close_all_connections(self) -> None:
        """Close all database connections and cleanup resources."""
        # Close thread-local connection if exists
        if hasattr(self._local, 'connection') and self._local.connection:
            try:
                self._local.connection.close()
                tui.tui_db_event("Closed", "Thread Connection", "Local connection closed")
            except Exception as e:
                tui.tui_print_error(f"Error closing thread-local connection: {e}")
            finally:
                self._local.connection = None
        
        # Shutdown thread pool
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
            self._thread_pool = None
            tui.tui_db_event("Shutdown", "Thread Pool", "Database workers stopped")
        
        self._is_initialized = False
    
    def get_stats(self) -> dict:
        """
        Get database manager statistics.
        
        Returns:
            Dictionary with connection statistics
        """
        return {
            "connections_created": self._connections_created,
            "thread_pool_active": self._thread_pool is not None,
            "max_workers": self.max_workers,
            "db_path": self.db_path,
            "timeout_seconds": self.timeout_seconds
        }


# Global database manager instance
_db_manager: Optional[DatabaseConnectionManager] = None


def get_db_manager() -> DatabaseConnectionManager:
    """
    Get the global database manager instance.
    
    Returns:
        Singleton database manager instance
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseConnectionManager()
        # Initialize tables on first access
        _db_manager.create_tables_sync()
    return _db_manager


async def get_async_db_manager() -> DatabaseConnectionManager:
    """
    Get the global database manager instance (async version).
    
    Returns:
        Singleton database manager instance with async table initialization
    """
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseConnectionManager()
        # Initialize tables asynchronously on first access
        await _db_manager.create_tables_async()
    return _db_manager


def close_db_manager() -> None:
    """Close the global database manager and cleanup resources."""
    global _db_manager
    if _db_manager:
        _db_manager.close_all_connections()
        _db_manager = None