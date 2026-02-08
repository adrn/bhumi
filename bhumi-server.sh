#!/bin/bash
# Bhumi server management script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
HOST="${BHUMI_HOST:-0.0.0.0}"
PORT="${BHUMI_PORT:-8000}"
LOG_DIR="${SCRIPT_DIR}/logs"
PID_FILE="${SCRIPT_DIR}/.bhumi.pid"
LOG_FILE="${LOG_DIR}/bhumi.log"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# Check if server is running
is_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            return 0
        else
            rm -f "$PID_FILE"
            return 1
        fi
    fi
    return 1
}

# Start the server
start() {
    if is_running; then
        echo "Server is already running (PID $(cat "$PID_FILE"))"
        return 1
    fi

    echo "Starting Bhumi server on ${HOST}:${PORT}..."

    # Activate virtual environment if it exists
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    fi

    # Start uvicorn in the background
    nohup uvicorn bhumi.app:app \
        --host "$HOST" \
        --port "$PORT" \
        --log-level info \
        >> "$LOG_FILE" 2>&1 &

    PID=$!
    echo "$PID" > "$PID_FILE"

    # Wait a moment and check if it started successfully
    sleep 2
    if is_running; then
        echo "Server started successfully (PID $PID)"
        echo "Logs: $LOG_FILE"
        echo "Access at: http://$(hostname -f):${PORT}"
        return 0
    else
        echo "Failed to start server. Check logs: $LOG_FILE"
        rm -f "$PID_FILE"
        return 1
    fi
}

# Stop the server
stop() {
    if ! is_running; then
        echo "Server is not running"
        return 1
    fi

    PID=$(cat "$PID_FILE")
    echo "Stopping Bhumi server (PID $PID)..."

    kill "$PID"

    # Wait for graceful shutdown (up to 10 seconds)
    for i in {1..10}; do
        if ! ps -p "$PID" > /dev/null 2>&1; then
            echo "Server stopped"
            rm -f "$PID_FILE"
            return 0
        fi
        sleep 1
    done

    # Force kill if still running
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Force killing server..."
        kill -9 "$PID"
        rm -f "$PID_FILE"
    fi
}

# Restart the server
restart() {
    echo "Restarting Bhumi server..."
    stop
    sleep 1
    start
}

# Show server status
status() {
    if is_running; then
        PID=$(cat "$PID_FILE")
        echo "Server is running (PID $PID)"
        echo "Listening on ${HOST}:${PORT}"
        ps -p "$PID" -o pid,etime,cmd
        return 0
    else
        echo "Server is not running"
        return 1
    fi
}

# Show logs
logs() {
    if [ ! -f "$LOG_FILE" ]; then
        echo "No log file found at $LOG_FILE"
        return 1
    fi

    if [ "$1" = "-f" ] || [ "$1" = "--follow" ]; then
        tail -f "$LOG_FILE"
    else
        tail -n 50 "$LOG_FILE"
    fi
}

# Main command dispatcher
case "${1:-}" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    status)
        status
        ;;
    logs)
        shift
        logs "$@"
        ;;
    *)
        cat <<EOF
Bhumi Server Manager

Usage: $0 {start|stop|restart|status|logs}

Commands:
  start      Start the server in the background
  stop       Stop the running server
  restart    Restart the server
  status     Show server status
  logs       Show recent logs (use -f to follow)

Environment variables:
  BHUMI_HOST   Host to bind to (default: 0.0.0.0)
  BHUMI_PORT   Port to listen on (default: 8000)

Examples:
  $0 start
  $0 status
  $0 logs -f
  BHUMI_PORT=9000 $0 start
EOF
        exit 1
        ;;
esac
