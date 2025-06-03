# ==== UTILITIES PACKAGE INITIALIZATION ==== #
"""
Initializes the 'utils' package for the tgTrax application.

This package provides utility modules supporting various functionalities such as:
- Text User Interface (TUI) enhancements (via `tui.py`).
- Logger adapter for TUI integration (via `logger_adapter.py`).

This `__init__.py` file marks the `utils` directory as a Python package,
making its modules available for import elsewhere in the tgTrax application.
It can also be used to selectively export symbols from its submodules or
perform package-level initializations if needed in the future.
"""

# No specific package-level objects are exported or initialized by default.
# This file primarily serves to mark the directory as a package.

__all__ = [] # Explicitly define what `from .utils import *` would import 