# ==== CORE PACKAGE INITIALIZATION ==== #
"""
Initializes the 'core' package for the tgTrax application.

This package forms the heart of tgTrax, containing the essential logic for its
operations. Key modules include:

-   `auth.py`: Handles Telegram client authentication and session management.
-   `tracker.py`: Manages activity tracking, data collection from Telegram, and
    scheduling of polling tasks.
-   `database.py`: Provides an interface for data storage and retrieval,
    typically interacting with an SQLite database.
-   `analysis.py`: Contains functions for performing temporal analysis of user
    activity data.

This `__init__.py` file ensures that the `core` directory is treated as a
Python package, allowing its modules to be imported and utilized by other parts
of the application, such as the main tgTrax script or TUI components.
"""

# This comment ensures the file is not empty if no other code is present.
# It explicitly marks the 'core' directory as a Python package.

# This file makes 'core' a package 

# By default, no symbols are exported directly from the core package itself
# when using `from .core import *`. Specific modules should be imported directly.
__all__ = [] 