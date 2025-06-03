cat << 'BANNER_EOF'
  ┏┳┓     
╋┏┓┃┏┓┏┓┓┏
┗┗┫┻┛ ┗┻┛┗
  ┛       
BANNER_EOF

#!/bin/bash

# TgTrax Runner Script
# ------------------------------------
# This script helps manage the TgTrax application components.

# Ensure the script is run from the directory where it is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Activate virtual environment if it exists
if [ -d "venv" ]; then
  source venv/bin/activate
fi

# Set PYTHONPATH to include the project's base directory (one level above SCRIPT_DIR)
# This allows main.py (in root/) to correctly import the tgTrax package (e.g., from root/../tgTrax)
PROJECT_BASE_DIR=$(dirname "$SCRIPT_DIR")
export PYTHONPATH="$PROJECT_BASE_DIR:$PYTHONPATH"

# Config files are assumed to be in the current directory (project root, i.e., SCRIPT_DIR)
CONFIG_FILE="tgTrax_config.json"
ENV_FILE=".env"

# Function to check if essential config files exist
check_config() {
    if [ ! -f "$CONFIG_FILE" ] || [ ! -f "$ENV_FILE" ]; then
        echo "Warning: Configuration files ($CONFIG_FILE or $ENV_FILE) not found in $(pwd)."
        echo "main.py should create them with default values if missing when run."
        # Depending on the command, you might want to exit if critical configs are missing.
        # For example, tracker-start might require them more strictly than analyze with --demo.
    fi
}

# --- Configuration ---
PYTHON_CMD="python3" # Using python3 explicitly is good practice
MAIN_SCRIPT="main.py" # Main script in the current directory

REQUIREMENTS_FILE="requirements.txt"

# Log files for background processes (relative to SCRIPT_DIR)
TRACKER_LOG_FILE="tracker.nohup.log"

# PID files for managing background processes (relative to SCRIPT_DIR)
TRACKER_PID_FILE=".tracker.pid"

# --- Helper Functions ---

check_python() {
    if ! command -v "$PYTHON_CMD" &> /dev/null; then
        echo "Error: Python command ('$PYTHON_CMD') not found."
        echo "Please ensure Python 3.9+ is installed and in your PATH, or a virtual environment is active."
        exit 1
    fi
    # Basic version check (optional, customize as needed)
    PY_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [[ "$PY_VERSION" < "3.9" ]]; then # Example check for 3.9+
        echo "Warning: Python version $PY_VERSION detected. Version 3.9 or higher is recommended."
    fi
}

setup_venv() {
    check_python
    if [ -d "venv" ]; then
        echo "Virtual environment 'venv' already exists."
    else
        echo "Creating virtual environment 'venv' with $PYTHON_CMD..."
        "$PYTHON_CMD" -m venv venv
        if [ $? -ne 0 ]; then
            echo "Error: Failed to create virtual environment 'venv'."
            exit 1
        fi
        echo "Virtual environment 'venv' created successfully."
    fi
    
    echo "Activating virtual environment 'venv' for setup..."
    source venv/bin/activate
    
    echo "Installing/updating dependencies from $REQUIREMENTS_FILE using pip..."
    if ! command -v pip &> /dev/null; then
        echo "Error: pip command not found in the activated virtual environment 'venv'."
        echo "Attempting to install pip..."
        "$PYTHON_CMD" -m ensurepip --upgrade
        if ! "$PYTHON_CMD" -m pip --version &>/dev/null; then # Check again
            easy_install_cmd=$(command -v easy_install)
            if [ -n "$easy_install_cmd" ]; then
                echo "Trying to install pip with easy_install..."
                sudo "$easy_install_cmd" pip
            else
                 echo "Error: pip could not be installed automatically. Please install pip manually in the venv."
                 exit 1
            fi
        fi
    fi
    
    if [ -f "$REQUIREMENTS_FILE" ]; then
        pip install -r "$REQUIREMENTS_FILE"
    else
        echo "Error: $REQUIREMENTS_FILE not found in current directory."
        exit 1
    fi

    if [ $? -ne 0 ]; then
        echo "Error: Failed to install dependencies using pip in 'venv'."
        exit 1
    fi
    echo "Dependencies installed successfully."
    echo "Setup complete. Virtual environment 'venv' is configured."
    echo "To activate manually in a new terminal: source venv/bin/activate"
    echo "To deactivate: deactivate"
}


# --- Process Management Functions ---

start_tracker() {
    check_python
    check_config 
    echo "IMPORTANT: Ensure your .env file is configured with TELEGRAM_API_ID and TELEGRAM_API_HASH."
    if [ "$VERBOSE_MODE" = true ]; then
        echo "Starting Telegram Tracker (foreground/verbose)..."
        # Pass any additional arguments $@ to the script
        "$PYTHON_CMD" "$MAIN_SCRIPT" tracker "$@"
    else
        echo "Starting Telegram Tracker (background)..."
        if [ -f "$TRACKER_PID_FILE" ]; then
            local existing_pid
            existing_pid=$(cat "$TRACKER_PID_FILE" 2>/dev/null)
            if ps -p "$existing_pid" > /dev/null 2>&1; then
                echo "Tracker already running (PID: $existing_pid)."
                return
            else
                echo "Stale or invalid PID file found ($TRACKER_PID_FILE for PID \"$existing_pid\"). Removing it."
                rm -f "$TRACKER_PID_FILE"
            fi
        fi

        nohup "$PYTHON_CMD" "$MAIN_SCRIPT" tracker "$@" > "$TRACKER_LOG_FILE" 2>&1 &
        local pid=$!
        echo "$pid" > "$TRACKER_PID_FILE"
        sleep 1

        if ps -p "$pid" > /dev/null 2>&1; then
            echo "Tracker started. PID: $pid. Log: $SCRIPT_DIR/$TRACKER_LOG_FILE"
        else
            echo "Error starting tracker. PID $pid not found after 1s. Check $SCRIPT_DIR/$TRACKER_LOG_FILE."
            if [ -s "$SCRIPT_DIR/$TRACKER_LOG_FILE" ]; then
                echo "--- Recent log entries from $SCRIPT_DIR/$TRACKER_LOG_FILE: ---"
                tail -n 10 "$SCRIPT_DIR/$TRACKER_LOG_FILE"
                echo "-----------------------------------------------------------"
            fi
            rm -f "$TRACKER_PID_FILE"
        fi
    fi
}

stop_tracker() {
    echo "Stopping Telegram Tracker..."
    if [ ! -f "$TRACKER_PID_FILE" ]; then
        echo "Tracker PID file not found. Is the tracker running or was it stopped manually?"
        return
    fi
    local pid
    pid=$(cat "$TRACKER_PID_FILE")
    if ps -p "$pid" > /dev/null 2>&1; then
        kill "$pid"
        sleep 2 # Give it time to shut down
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "Tracker (PID $pid) did not stop gracefully. Sending SIGKILL..."
            kill -9 "$pid"
        else
            echo "Tracker (PID $pid) stopped."
        fi
    else
        echo "Tracker process (PID $pid from PID file) not found."
    fi
    rm -f "$TRACKER_PID_FILE"
}

status_tracker() {
    if [ -f "$TRACKER_PID_FILE" ]; then
        local pid
        pid=$(cat "$TRACKER_PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "Tracker is RUNNING (PID: $pid)."
            echo "Log file: $SCRIPT_DIR/$TRACKER_LOG_FILE"
            if [ "$VERBOSE_MODE" = true ]; then
                echo "--- Last 10 log lines: ---"
                tail -n 10 "$SCRIPT_DIR/$TRACKER_LOG_FILE"
                echo "--------------------------"
            fi
        else
            echo "Tracker is STOPPED (PID $pid from PID file not found)."
            rm -f "$TRACKER_PID_FILE" # Stale PID file
        fi
    else
        echo "Tracker is STOPPED (no PID file)."
    fi
}

run_analyzer() {
    check_python
    check_config
    echo "Running Correlation Analyzer..."
    if [ "$VERBOSE_MODE" = true ]; then
         echo "Command: $PYTHON_CMD $MAIN_SCRIPT analyze $@"
    fi
    # Pass any additional arguments $@ to the script
    "$PYTHON_CMD" "$MAIN_SCRIPT" analyze "$@"
}

# --- Main Script Logic ---

usage() {
    echo "TgTrax Runner Script"
    echo "--------------------"
    echo "Manages TgTrax components: Tracker and Analyzer."
    echo ""
    echo "Usage: ./run.sh [options] <command> [command_args...]"
    echo ""
    echo "Options:"
    echo "  -v, --verbose    Run commands in foreground/verbose mode where applicable."
    echo "  -h, --help       Show this help message."
    echo ""
    echo "Commands:"
    echo "  setup            Create/update Python virtual environment and install dependencies."
    echo "  start            Start the Telegram activity tracker (background by default)."
    echo "                   Any arguments after start will be passed to the tracker."
    echo "  stop             Stop the Telegram activity tracker."
    echo "  status           Check the status of the Telegram activity tracker."
    echo "  logs             Show the last 20 lines of the tracker log (use -f for live)."
    echo "                   Example: ./run.sh logs -n 50"
    echo "                   Example: ./run.sh logs -f"
    echo "  analyze          Run the correlation analyzer. "
    echo "                   Any arguments after analyze will be passed to the analyzer."
    echo "                   Example: ./run.sh analyze"
    echo "                   Example for demo mode: ./run.sh analyze --demo"
    echo ""
    echo "Examples:"
    echo "  ./run.sh setup"
    echo "  ./run.sh start"
    echo "  ./run.sh -v start  # Start tracker in foreground"
    echo "  ./run.sh analyze --demo"
    echo "  ./run.sh status"
    echo "  ./run.sh stop"
    echo ""
    echo "Before first use, run './run.sh setup'."
    echo "Ensure .env and tgTrax_config.json are configured as needed (main.py creates defaults)."
}

VERBOSE_MODE=false

# Parse global options like -v or -h
while [[ $# -gt 0 ]]; do
    case "$1" in
        -v|--verbose)
            VERBOSE_MODE=true
            shift # past argument
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            # If it's not an option, it's the command
            break 
            ;;
    esac
done

COMMAND=$1
if [ -z "$COMMAND" ]; then
    usage
    exit 1
fi
shift # Remove command from arguments, rest are command_args for the specific command

case "$COMMAND" in
    setup)
        setup_venv
        ;;
    start)
        start_tracker "$@" # Pass remaining args to function
        ;;
    stop)
        stop_tracker
        ;;
    status)
        status_tracker
        ;;
    logs)
        if [ -f "$TRACKER_LOG_FILE" ]; then
            echo "Displaying tracker log ($TRACKER_LOG_FILE)... (Ctrl+C to stop if live)"
            tail "$@" "$TRACKER_LOG_FILE" # Pass through args like -n, -f to tail
        else
            echo "Tracker log file ($TRACKER_LOG_FILE) not found."
        fi
        ;;
    analyze)
        run_analyzer "$@" # Pass remaining args to function
        ;;
    *)
        echo "Error: Unknown command '$COMMAND'"
        usage
        exit 1
        ;;
esac

exit 0