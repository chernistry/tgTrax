"""
Logger adapter that bridges Python's standard logging module with tgTrax's TUI.

Provides colored console output through Rich library while maintaining
logging.Logger compatible API.
"""

# --- Imports ---
import logging
import traceback # For exc_info handling
from typing import Any, Callable, Dict, Optional, Tuple, Union

# Assuming tui.py provides the TUI functions within the same package
from . import tui


# --- TuiLoggerAdapter Class ---
class TuiLoggerAdapter:
    """
    A logger adapter that directs logging messages to TUI print functions.

    This class provides a subset of the standard `logging.Logger` interface,
    routing calls to appropriate `tui.tui_print_*` methods based on log level.
    It allows existing code using standard logging to output via the TUI.
    This adapter is not a full `logging.Handler` but mimics a `Logger`'s API.
    """

    def __init__(self, tui_module: Any) -> None:
        """
        Initializes the TuiLoggerAdapter.

        Args:
            tui_module: The TUI module (e.g., `tgTrax.utils.tui`)
                        which provides `tui_print_*` functions.
                        Typically, this would be the `tui` module itself.
        """
        self.tui: Any = tui_module
        # Map standard logging levels to TUI print functions
        self._level_to_func: Dict[int, Callable[..., None]] = {
            logging.DEBUG: self.tui.tui_print_debug,
            logging.INFO: self.tui.tui_print_info,
            logging.WARNING: self.tui.tui_print_warning,
            logging.ERROR: self.tui.tui_print_error,
            logging.CRITICAL: self.tui.tui_print_error,  # Map critical to error for TUI
        }
        self.current_level: int = logging.INFO  # Default level

    def _log(
        self,
        level: int,
        msg: Any,
        args: Tuple[Any, ...],
        exc_info: Optional[Union[bool, Any]] = None,
        stack_info: bool = False, # Added to match logging.Logger signature closer
        **kwargs: Any
    ) -> None:
        """
        Internal logging method.

        Args:
            level: The log level.
            msg: The log message or message format string.
            args: Arguments for the message format string.
            exc_info: Exception information to log.
            stack_info: Whether to log stack information (not fully implemented for TUI).
            **kwargs: Additional keyword arguments (mostly ignored for TUI).
        """
        if level < self.current_level:
            return

        func: Callable[..., None] = self._level_to_func.get(level, self.tui.tui_print_info)
        log_msg: str
        if args:
            try:
                log_msg = str(msg) % args
            except TypeError:
                # Fallback for when % formatting fails (e.g. msg not a format string)
                log_msg = f"{str(msg)} {' '.join(map(str, args))}"
        else:
            log_msg = str(msg)
        
        func(log_msg)

        if exc_info:
            # If exc_info is True, or an exception tuple, format it.
            # Standard logger also accepts an exception instance directly.
            exc_text: str = ""
            if isinstance(exc_info, bool):
                # This will use sys.exc_info() internally in traceback.format_exc()
                # if an exception is being handled.
                exc_text = traceback.format_exc()
            elif isinstance(exc_info, tuple):
                exc_text = ''.join(traceback.format_exception(*exc_info))
            elif isinstance(exc_info, BaseException):
                exc_text = ''.join(traceback.format_exception(
                    type(exc_info), exc_info, exc_info.__traceback__
                ))
            else:
                exc_text = "" # Should not happen with typical logger.exception calls
            
            if exc_text.strip() and exc_text.strip() != "NoneType: None":
                # Only print if there's actual traceback info
                self.tui.tui_print_error(f"\n--- Traceback ---:\n{exc_text.strip()}\n-------------------")
        
        # stack_info handling could be added here if needed by TUI
        # if stack_info:
        #     pass 

    def debug(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Logs a message with level DEBUG on this logger."""
        self._log(logging.DEBUG, msg, args, **kwargs)

    def info(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Logs a message with level INFO on this logger."""
        self._log(logging.INFO, msg, args, **kwargs)

    def warning(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Logs a message with level WARNING on this logger."""
        self._log(logging.WARNING, msg, args, **kwargs)

    def error(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Logs a message with level ERROR on this logger."""
        self._log(logging.ERROR, msg, args, **kwargs)

    def critical(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Logs a message with level CRITICAL on this logger."""
        self._log(logging.CRITICAL, msg, args, **kwargs)

    def success(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Non-standard level convenience: prints a success-styled message."""
        if logging.INFO < self.current_level:
            return
        try:
            text = (str(msg) % args) if args else str(msg)
        except Exception:
            text = f"{msg} {' '.join(map(str, args))}"
        self.tui.tui_print_success(text)

    def exception(self, msg: Any, *args: Any, exc_info: bool = True, **kwargs: Any) -> None:
        """
        Convenience method for logging an ERROR with exception information.
        """
        kwargs['exc_info'] = exc_info
        self._log(logging.ERROR, msg, args, **kwargs)

    def isEnabledFor(self, level: int) -> bool:
        """Checks if a message of 'level' would be processed by this logger."""
        return level >= self.current_level

    def setLevel(self, level: Union[int, str]) -> None:
        """
        Sets the logging level of this logger.
        Level must be an int or a str (e.g., "INFO").
        """
        numeric_level: Optional[int] = None
        if isinstance(level, str):
            # Convert string level name to int if necessary
            _lvl = logging.getLevelName(level.upper())
            if isinstance(_lvl, int):
                numeric_level = _lvl
            else:
                self.tui.tui_print_warning(
                    f"(TuiLoggerAdapter: Invalid log level string '{level}'. Keeping current level.)"
                )
                return
        elif isinstance(level, int):
            numeric_level = level
        else:
            self.tui.tui_print_warning(
                f"(TuiLoggerAdapter: Invalid log level type '{type(level).__name__}'. Level must be int or str.)"
            )
            return
            
        if numeric_level is not None:
            self.current_level = numeric_level
            self.tui.tui_print_debug(
                f"(TuiLoggerAdapter: Log level set to {logging.getLevelName(self.current_level)} ({self.current_level}))"
            )

    def hasHandlers(self) -> bool:
        """Checks if this logger has any handlers configured. TUI assumes yes."""
        return True  # Assume TUI is always "handled"

    def addHandler(self, handler: Any) -> None:
        """Adds the specified handler to this logger (no-op for TUI)."""
        self.tui.tui_print_debug(
            "(TuiLoggerAdapter: addHandler called, no-op for TUI)"
        )
        # Pass, as TUI interaction is direct, not via standard handlers

    def removeHandler(self, handler: Any) -> None:
        """Removes the specified handler from this logger (no-op for TUI)."""
        self.tui.tui_print_debug(
            "(TuiLoggerAdapter: removeHandler called, no-op for TUI)"
        )
        # Pass

    # --- Additional methods to mimic logging.Logger if needed ---
    # def getEffectiveLevel(self) -> int:
    #     return self.current_level

    # def getChild(self, suffix: str) -> 'TuiLoggerAdapter':
    #     # For simplicity, child loggers could just return self if not implementing hierarchy
    #     self.tui.tui_print_debug(f"(TuiLoggerAdapter: getChild('{suffix}') called, returning self)")
    #     return self

    # For compatibility, sometimes `logger.getLogger(name)` is used.
    # This adapter is usually instantiated directly, not fetched by name via logging.getLogger.
    # However, if some library code calls getLogger on an instance of this adapter,
    # returning self is a reasonable behavior.
    def getLogger(self, name: str) -> 'TuiLoggerAdapter': # type: ignore
        """Returns a logger with the specified name (returns self for this adapter)."""
        self.tui.tui_print_debug(
            f"(TuiLoggerAdapter: getLogger('{name}') called, returning self)"
        )
        return self 
