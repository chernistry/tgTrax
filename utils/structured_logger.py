"""
Structured logging utilities for tgTrax application.

This module provides structured logging capabilities with proper error surfacing,
context management, and integration with the TUI system.
"""

import logging
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, Optional, Union
from pathlib import Path

from tgTrax.core import settings
from tgTrax.core.models import ErrorInfo
from tgTrax.utils import tui


class StructuredLogger:
    """
    Enhanced logger with structured logging capabilities.
    
    Provides context-aware logging, error surfacing, and integration
    with the TUI system for better user experience.
    """
    
    def __init__(
        self, 
        name: str, 
        level: str = settings.DEFAULT_LOG_LEVEL,
        log_to_file: bool = True
    ):
        """
        Initialize the structured logger.
        
        Args:
            name: Logger name (usually module name)
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_to_file: Whether to also log to a file
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers(log_to_file)
        
        self.context: Dict[str, Any] = {}
        self._error_count = 0
        self._warning_count = 0
    
    def _setup_handlers(self, log_to_file: bool) -> None:
        """Setup logging handlers for console and file output."""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler (optional, TUI handles most console output)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.ERROR)  # Only show errors on console
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_to_file:
            try:
                log_file = Path(settings.DEFAULT_LOG_PATH)
                log_file.parent.mkdir(exist_ok=True)
                
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            except Exception as e:
                tui.tui_print_error(f"Failed to setup file logging: {e}")
    
    def add_context(self, **kwargs: Any) -> None:
        """Add context information to all subsequent log messages."""
        self.context.update(kwargs)
    
    def clear_context(self) -> None:
        """Clear all context information."""
        self.context.clear()
    
    def _format_message(self, message: str, **kwargs: Any) -> str:
        """Format log message with context and additional data."""
        parts = [message]
        
        # Add context if available
        if self.context:
            context_str = " | ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"Context: {context_str}")
        
        # Add additional kwargs
        if kwargs:
            extra_str = " | ".join(f"{k}={v}" for k, v in kwargs.items())
            parts.append(f"Extra: {extra_str}")
        
        return " | ".join(parts)
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message with context."""
        formatted_msg = self._format_message(message, **kwargs)
        self.logger.debug(formatted_msg)
        tui.tui_print_debug(message)
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message with context."""
        formatted_msg = self._format_message(message, **kwargs)
        self.logger.info(formatted_msg)
        tui.tui_print_info(message)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message with context."""
        self._warning_count += 1
        formatted_msg = self._format_message(message, **kwargs)
        self.logger.warning(formatted_msg)
        tui.tui_print_warning(message)
    
    def error(
        self, 
        message: str, 
        error: Optional[Exception] = None,
        exc_info: bool = False,
        **kwargs: Any
    ) -> ErrorInfo:
        """
        Log error message with context and return structured error info.
        
        Args:
            message: Error message
            error: Exception object if available
            exc_info: Whether to include exception traceback
            **kwargs: Additional context
            
        Returns:
            ErrorInfo object with structured error details
        """
        self._error_count += 1
        
        # Create structured error info
        error_info = ErrorInfo(
            error_type=type(error).__name__ if error else "Error",
            message=message,
            context={**self.context, **kwargs},
            timestamp=datetime.utcnow(),
            severity="error"
        )
        
        # Format message for logging
        formatted_msg = self._format_message(message, **kwargs)
        if error:
            formatted_msg += f" | Exception: {error}"
        
        # Log to file with traceback if requested
        if exc_info or error:
            self.logger.error(formatted_msg, exc_info=exc_info or bool(error))
        else:
            self.logger.error(formatted_msg)
        
        # Show in TUI
        tui.tui_print_error(message)
        
        # Show traceback in TUI if available
        if error and hasattr(error, '__traceback__'):
            tb_lines = traceback.format_exception(
                type(error), error, error.__traceback__
            )
            tui.tui_print_debug("".join(tb_lines))
        
        return error_info
    
    def critical(
        self, 
        message: str, 
        error: Optional[Exception] = None,
        **kwargs: Any
    ) -> ErrorInfo:
        """
        Log critical error and return structured error info.
        
        Args:
            message: Critical error message
            error: Exception object if available
            **kwargs: Additional context
            
        Returns:
            ErrorInfo object with structured error details
        """
        error_info = ErrorInfo(
            error_type=type(error).__name__ if error else "CriticalError",
            message=message,
            context={**self.context, **kwargs},
            timestamp=datetime.utcnow(),
            severity="critical"
        )
        
        formatted_msg = self._format_message(message, **kwargs)
        if error:
            formatted_msg += f" | Exception: {error}"
        
        self.logger.critical(formatted_msg, exc_info=bool(error))
        tui.tui_print_error(f"CRITICAL: {message}")
        
        return error_info
    
    def surface_error(
        self, 
        error: Exception, 
        context: str = "",
        reraise: bool = False
    ) -> ErrorInfo:
        """
        Surface an exception with proper logging and error handling.
        
        Args:
            error: The exception to surface
            context: Additional context about where/why the error occurred
            reraise: Whether to re-raise the exception after logging
            
        Returns:
            ErrorInfo object with structured error details
            
        Raises:
            Exception: The original exception if reraise=True
        """
        message = f"{context}: {error}" if context else str(error)
        error_info = self.error(message, error=error, exc_info=True)
        
        if reraise:
            raise error
        
        return error_info
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        return {
            "logger_name": self.name,
            "error_count": self._error_count,
            "warning_count": self._warning_count,
            "context": self.context.copy(),
            "level": self.logger.level,
            "handlers_count": len(self.logger.handlers)
        }


class AsyncLoggerMixin:
    """
    Mixin class to add async-safe logging to other classes.
    
    This mixin ensures that logging operations don't block the async event loop
    by using appropriate async patterns when needed.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger: Optional[StructuredLogger] = None
    
    @property
    def logger(self) -> StructuredLogger:
        """Get or create a structured logger for this instance."""
        if self._logger is None:
            # Use the class name as the logger name
            logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
            self._logger = StructuredLogger(logger_name)
        return self._logger
    
    async def log_async_operation(
        self,
        operation_name: str,
        operation_func,
        *args,
        **kwargs
    ) -> Any:
        """
        Log an async operation with proper error handling.
        
        Args:
            operation_name: Name of the operation for logging
            operation_func: The async function to execute
            *args: Arguments for the operation function
            **kwargs: Keyword arguments for the operation function
            
        Returns:
            Result of the operation function
        """
        self.logger.add_context(operation=operation_name)
        
        try:
            self.logger.debug(f"Starting {operation_name}")
            result = await operation_func(*args, **kwargs)
            self.logger.debug(f"Completed {operation_name}")
            return result
        
        except Exception as e:
            self.logger.surface_error(e, f"Failed {operation_name}", reraise=True)
        
        finally:
            self.logger.clear_context()


# Global logger instance
_global_logger: Optional[StructuredLogger] = None


def get_logger(name: Optional[str] = None) -> StructuredLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (defaults to calling module)
        
    Returns:
        Structured logger instance
    """
    if name is None:
        # Get the calling module name
        import inspect
        frame = inspect.currentframe()
        if frame and frame.f_back:
            name = frame.f_back.f_globals.get('__name__', 'unknown')
        else:
            name = 'unknown'
    
    return StructuredLogger(name)


def get_global_logger() -> StructuredLogger:
    """Get the global application logger."""
    global _global_logger
    if _global_logger is None:
        _global_logger = StructuredLogger("tgTrax.global")
    return _global_logger