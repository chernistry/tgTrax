"""
TUI (Text User Interface) utilities for tgTrax using Rich library.

Provides styled console output, progress bars, tables and panels for better CLI
experience. Includes logger-compatible functions for backward compatibility.
"""

import json
from typing import Any, Dict, List, Optional, Union, Tuple

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.syntax import Syntax
from rich.table import Table
from rich.theme import Theme


# ==== TG TRAX CUSTOM THEME ==== #

custom_theme = Theme({
    "info": "bold bright_white",
    "success": "bold green",
    "warning": "bold yellow",
    "error": "bold red",
    "debug": "grey70",
    "highlight": "bold blue",
    "detail": "grey74",
    "progress": "cyan",
    "code": "white",
    "data": "bright_cyan",
    "db": "blue",
    "api": "magenta",
})

console = Console(theme=custom_theme)


# ==== CORE TUI PRINT FUNCTIONS ==== #

def tui_print_info(message: str, style: str = "info") -> None:
    """Prints an informational message with the specified style.

    Args:
        message: The message string to print.
        style: The Rich style to apply (defaults to "info").
    """
    console.print(f"[{style}]{message}[/]")


def tui_print_success(message: str, style: str = "success") -> None:
    """Prints a success message with the specified style.

    Args:
        message: The message string to print.
        style: The Rich style to apply (defaults to "success").
    """
    console.print(f"[{style}]{message}[/]")


def tui_print_warning(message: str, style: str = "warning") -> None:
    """Prints a warning message with the specified style.

    Args:
        message: The message string to print.
        style: The Rich style to apply (defaults to "warning").
    """
    console.print(f"[{style}]{message}[/]")


def tui_print_error(message: str, style: str = "error") -> None:
    """Prints an error message with the specified style.

    Args:
        message: The message string to print.
        style: The Rich style to apply (defaults to "error").
    """
    console.print(f"[{style}]{message}[/]")


def tui_print_debug(message: str, style: str = "debug") -> None:
    """Prints a debug message with the specified style.

    Args:
        message: The message string to print.
        style: The Rich style to apply (defaults to "debug").
    """
    console.print(f"[{style}]{message}[/]")


def tui_print_highlight(message: str, style: str = "highlight") -> None:
    """Prints a highlighted message with the specified style.

    Args:
        message: The message string to print.
        style: The Rich style to apply (defaults to "highlight").
    """
    console.print(f"[{style}]{message}[/]")


def tui_print_detail(message: str, style: str = "detail") -> None:
    """Prints a detail message with the specified style.

    Args:
        message: The message string to print.
        style: The Rich style to apply (defaults to "detail").
    """
    console.print(f"[{style}]{message}[/]")


def tui_print_code(
    code: str,
    language: str = "python",
    line_numbers: bool = False,
    title: Optional[str] = None,
) -> None:
    """Prints a code block with syntax highlighting, optionally in a panel.

    Args:
        code: The code string to print.
        language: The programming language for syntax highlighting.
        line_numbers: Whether to display line numbers.
        title: Optional title for a panel around the code. If provided,
               the code block will be rendered within a styled panel.
    """
    syntax = Syntax(code, language, theme="monokai", line_numbers=line_numbers)
    if title:
        tui_panel(syntax, title=title, style="code", border_style="code")
    else:
        console.print(syntax)


def tui_print_json(data: Any, title: Optional[str] = None) -> None:
    """Prints JSON data with syntax highlighting, optionally in a panel.

    Handles both stringified JSON and serializable Python objects.

    Args:
        data: The JSON data (can be a string or a serializable Python object).
        title: Optional title for a panel around the JSON data. If provided,
               the JSON will be rendered within a styled panel.
    """
    if isinstance(data, str):
        json_str_data = data
    else:
        try:
            json_str_data = json.dumps(data, indent=2, ensure_ascii=False)
        except TypeError as e:
            tui_print_error(f"Failed to serialize data to JSON: {e}")
            console.print(str(data))  # Fallback to printing raw data
            return

    syntax = Syntax(json_str_data, "json", theme="monokai", line_numbers=False)
    if title:
        tui_panel(syntax, title=title, style="data", border_style="data")
    else:
        console.print(syntax)


def tui_print_table(
    data: List[Dict[str, Any]],
    title: Optional[str] = None,
    style_columns: Optional[Dict[str, str]] = None,
) -> None:
    """Prints data in a formatted table using Rich.

    Auto-styles columns based on common keywords (e.g., "ID", "count")
    and data types.

    Args:
        data: A list of dictionaries, where each dictionary represents a row.
              All dictionaries should ideally have the same keys.
        title: Optional title for the table.
        style_columns: Optional dictionary mapping column names to specific
                       Rich styles to override default styling.
    """
    if not data:  # Handles empty list as well as None
        warning_msg = "No data to display in table"
        if title:
            warning_msg += f" for '{title}'"
        tui_print_warning(warning_msg)
        return

    table = Table(title=title, box=box.ROUNDED, show_lines=True, expand=True)
    headers = list(data[0].keys())

    for key in headers:
        column_style: str = "data"  # Default style
        header_style: str = "bold data"
        justify_rule: str = "left"  # Default justification

        # Apply custom styling if provided
        if style_columns and key in style_columns:
            column_style = style_columns[key]
            header_style = f"bold {style_columns[key]}"
        # Heuristic styling for common patterns
        elif "id" in key.lower():  # Check for "id" or "ID" variants
            column_style = "highlight"
            header_style = "bold highlight"
        elif any(
            num_keyword in key.lower()
            for num_keyword in ["count", "total", "number", "age", "size"]
        ):
            justify_rule = "right"
        # Justify numeric and boolean types to center by default if no other rule applies
        elif isinstance(data[0].get(key), (int, float, bool)):
            justify_rule = "center"

        table.add_column(
            key,
            style=column_style,
            header_style=header_style,
            justify=justify_rule,
            overflow="fold",
        )

    for item in data:
        table.add_row(*[str(item.get(header, "")) for header in headers])

    console.print(table)


# ==== PROCESS STATUS INDICATORS ==== #
# --► For tgTrax specific operations

def tui_starting_process(process_name: str) -> None:
    """Prints a message indicating a process is starting.

    Args:
        process_name: The name of the process that is starting.
    """
    tui_print_info(f"Starting: {process_name}...")


def tui_process_complete(process_name: str, status: str = "Completed") -> None:
    """Prints a message indicating a process has completed successfully.

    Args:
        process_name: The name of the process that completed.
        status: The completion status message (defaults to "Completed").
    """
    tui_print_success(f"{status}: {process_name}")


def tui_process_failed(process_name: str, reason: Optional[str] = None) -> None:
    """Prints a message indicating a process has failed.

    Args:
        process_name: The name of the process that failed.
        reason: Optional string explaining the reason for failure.
    """
    message = f"Failed: {process_name}"
    if reason:
        message += f" - Reason: {reason}"
    tui_print_error(message)


def tui_fetching_data(data_description: str) -> None:
    """Prints a message indicating data is being fetched, styled for API calls.

    Args:
        data_description: Description of the data being fetched (e.g., "user profiles").
    """
    tui_print_info(f"Fetching: {data_description}...", style="api")


def tui_saving_data(data_description: str) -> None:
    """Prints a message indicating data is being saved, styled for DB operations.

    Args:
        data_description: Description of the data being saved (e.g., "session state").
    """
    tui_print_info(f"Saving: {data_description}...", style="db")


def tui_updating_db(update_description: str) -> None:
    """Prints a message indicating a database update is in progress.

    Args:
        update_description: Description of the database update.
    """
    tui_print_info(f"DB Update: {update_description}", style="db")


def tui_waiting(seconds: float, reason: str = "") -> None:
    """Prints a message indicating a waiting period.

    Args:
        seconds: The duration of the wait in seconds.
        reason: Optional reason for the wait.
    """
    message = f"Waiting {seconds:.1f}s"
    if reason:
        message += f" (Reason: {reason})"
    tui_print_detail(message)


def tui_db_event(action: str, entity: str, detail: str = "") -> None:
    """Prints a structured message for database events.

    Args:
        action: The action performed (e.g., "Created", "Updated", "Deleted").
        entity: The entity affected (e.g., "User record", "Activity table").
        detail: Additional details about the event.
    """
    message = f"DB {action}: {entity}"
    if detail:
        message += f" - {detail}"
    console.print(message, style="db")


def tui_progress_bar(
    total_steps: float,
    description: str = "Processing...",
) -> Progress:
    """Creates and returns a pre-configured Rich Progress bar instance.

    The progress bar is set to be transient (disappears on completion).

    Args:
        total_steps: The total number of steps for the progress bar.
        description: A description of the task being tracked.

    Returns:
        A Rich Progress instance, ready to be used in a `with` statement.
    """
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed:.0f}/{task.total:.0f})"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,  # Disappears after completion
    )


def tui_panel(
    content: Union[str, Syntax, Table, Panel],
    title: str = "",
    style: str = "info",
    border_style: Optional[str] = None,
    expand: bool = True,
    padding: Union[int, Tuple[int, int], Tuple[int, int, int, int]] = (1, 2),
) -> None:
    """Prints content within a styled panel using Rich.

    Args:
        content: The content to display (string, Rich Syntax, Table, or another Panel).
        title: Optional title for the panel.
        style: The Rich style for the panel content. Also used for the border
               if `border_style` is not provided.
        border_style: Specific Rich style for the panel's border.
                      Defaults to `style` if None.
        expand: Whether the panel should expand to fill available width.
        padding: Padding within the panel. Can be an int (all sides),
                 a tuple (vertical, horizontal), or a 4-tuple (top, right, bottom, left).
    """
    final_border_style = border_style if border_style is not None else style
    panel_title = f"[{style}]{title}[/]" if title else ""

    console.print(
        Panel(
            content,
            title=panel_title,
            style=style,  # Style for the content area itself
            border_style=final_border_style,
            expand=expand,
            padding=padding,
        )
    )


# ==== LOGGER COMPATIBILITY FUNCTIONS ==== #
# --► For seamless integration with standard logging

def info(message: str, *args: Any, **kwargs: Any) -> None:
    """Logger-compatible info message printout using TUI styles.

    Supports string formatting with `*args`. `**kwargs` are ignored but accepted
    for compatibility.

    Args:
        message: The message string, can be a format string.
        *args: Arguments for message string formatting.
        **kwargs: Arbitrary keyword arguments (ignored).
    """
    if args:
        message = message % args
    tui_print_info(message)


def error(message: str, *args: Any, **kwargs: Any) -> None:
    """Logger-compatible error message printout using TUI styles.

    Supports string formatting with `*args`. `**kwargs` are ignored but accepted
    for compatibility.

    Args:
        message: The message string, can be a format string.
        *args: Arguments for message string formatting.
        **kwargs: Arbitrary keyword arguments (ignored).
    """
    if args:
        message = message % args
    tui_print_error(message)


def warning(message: str, *args: Any, **kwargs: Any) -> None:
    """Logger-compatible warning message printout using TUI styles.

    Supports string formatting with `*args`. `**kwargs` are ignored but accepted
    for compatibility.

    Args:
        message: The message string, can be a format string.
        *args: Arguments for message string formatting.
        **kwargs: Arbitrary keyword arguments (ignored).
    """
    if args:
        message = message % args
    tui_print_warning(message)


def debug(message: str, *args: Any, **kwargs: Any) -> None:
    """Logger-compatible debug message printout using TUI styles.

    Supports string formatting with `*args`. `**kwargs` are ignored but accepted
    for compatibility.

    Args:
        message: The message string, can be a format string.
        *args: Arguments for message string formatting.
        **kwargs: Arbitrary keyword arguments (ignored).
    """
    if args:
        message = message % args
    tui_print_debug(message)


# ==== DEMO / TEST CODE ==== #
# --► Only runs when script is executed directly

if __name__ == '__main__':
    tui_print_info("System initializing...")
    tui_print_success("Tracker started and running.")
    tui_print_warning("Polling interval is set to a low value.")
    tui_print_error("Database connection failed!")
    tui_print_debug("Debug: polling cycle 42, 3 users online.")
    tui_print_highlight("Analysis complete: strong correlation found.")
    tui_print_detail("Detail: using resample period 1min.")

    tui_print_info("--- Process Status Demo ---")
    tui_starting_process("User Status Polling")
    tui_fetching_data("Telegram user statuses")
    tui_updating_db("activity records for 5 users")
    tui_db_event("Inserted", "Activity", "User: Alice, Online: True")
    tui_waiting(0.1, "API rate limit")
    tui_process_complete("User Status Polling")
    tui_process_failed("Data Export", "Network timeout")

    tui_print_info("--- Table Demo ---")
    table_data: List[Dict[str, Any]] = [
        {"ID": 1, "User": "alice@example.com", "Status": "Online", "Score": 0.9876, "Count": 1024},
        {"ID": 2, "User": "bob.the.builder@domain.co.uk", "Status": "Offline", "Score": 0.123, "Count": 50},
        {"ID": 3, "User": "carol_smith", "Status": "Online", "Score": 0.67, "Count": 0},
    ]
    style_overrides = {"User": "bold yellow", "Score": "cyan"}
    tui_print_table(table_data, "User Status Table", style_columns=style_overrides)
    tui_print_table([], title="Empty Data Table") # Test empty table case

    tui_print_info("--- Panel Demo ---")
    panel_content = "Correlation analysis finished. See table above for details."
    tui_panel(
        panel_content,
        title="Analysis Status",
        style="success",
        border_style="green",
    )
    error_panel_content = "This is a [bold red]critical error[/] panel!"
    tui_panel(error_panel_content, title="Critical Error", style="error")

    tui_print_info("--- Progress Bar Demo ---")
    import time # Import time here as it's only used in demo
    TOTAL_DEMO_STEPS = 10
    with tui_progress_bar(
        total_steps=TOTAL_DEMO_STEPS,
        description="Processing Users Demo"
    ) as progress:
        task_id = progress.add_task("Downloading...", total=TOTAL_DEMO_STEPS)
        for i in range(TOTAL_DEMO_STEPS):
            time.sleep(0.05)
            progress.update(
                task_id,
                advance=1,
                description=f"User {i + 1}/{TOTAL_DEMO_STEPS}"
            )
    tui_print_success("User processing demo complete (progress bar finished).")

    tui_print_info("--- Code Block Demo ---")
    sample_code = (
        "def greet(name: str) -> str:\\n"
        "    greeting = f\\\"Hello, {name}!\\\"\\n"
        "    return greeting\\n\\n"
        "print(greet(\\'World\\'))"
    )
    tui_print_code(
        sample_code,
        language="python",
        line_numbers=True,
        title="Sample Python Code"
    )
    json_code_sample = "{\\\"key\\\": \\\"value\\\", \\\"number\\\": 42}"
    tui_print_code(
        json_code_sample,
        language="json",
        title="Simple JSON in Code Block"
    )

    tui_print_info("--- JSON Data Printing Demo ---")
    complex_json_data = {
        "user": "alice",
        "online": True,
        "score": 0.98,
        "details": {"items": [1, 2, 3], "valid": None},
    }
    tui_print_json(complex_json_data, title="Complex Python Dict as JSON")
    tui_print_json("{\\\"raw_string_json\\\": true}", title="Raw String JSON")
    json_array_data = [1, 2, {"a_key": "a_value"}] # Corrected list syntax
    tui_print_json(json_array_data, title="Python List as JSON Array")


    tui_print_info("--- Logger Compatibility Demo ---")
    info("This is an info message via logger compat function: %s", "OK")
    error("This is an error message: %s", "Details about error")
    warning("Warning: %s operation pending.", "Risky")
    debug("Debug data for item %d: %s", 42, {"key": "val"})
