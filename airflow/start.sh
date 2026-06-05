#!/usr/bin/env bash
set -e

# Resolve AIRFLOW_HOME to the directory containing this script (absolute path).
# All child processes (scheduler, triggerer, dag-processor, api-server) inherit
# this export, which is what was missing before.
export AIRFLOW_HOME="$(cd "$(dirname "$0")" && pwd)"

PROJECT_ROOT="$(cd "$AIRFLOW_HOME/.." && pwd)"

# macOS fork-safety fix: gunicorn workers SIGSEGV on macOS when forking after
# native libraries (numpy/pandas) have been loaded. This env var disables the
# Objective-C runtime's fork safety check that causes the crash.
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES

# Reduce all internal gunicorn worker counts to 1 so no pre-fork spawning occurs.
# With 1 worker, gunicorn still forks once but doesn't create a pool of workers
# that simultaneously inherit loaded native libs — eliminating the SIGSEGV storm
# that appears after a system clock change (scheduler catching up triggers many
# simultaneous worker spawns, each hitting the fork-safety issue).
export AIRFLOW__API__WORKERS=1
export AIRFLOW__EXECUTION_API__WORKERS=1
export AIRFLOW__SCHEDULER__NUM_PARSING_PROCESSES=1

# Disable the gunicorn-based log-streaming servers (ports 8793/8794).
# These servers are the primary source of SIGSEGV workers on macOS 26 —
# they fork gunicorn workers that immediately crash due to macOS fork-safety.
# Trade-off: task logs won't stream in the Airflow UI; they're still written
# to airflow/logs/ and readable via CLI. Acceptable for local dev.
export AIRFLOW__LOGGING__WORKER_LOG_SERVER_PORT=0
export AIRFLOW__LOGGING__TRIGGER_LOG_SERVER_PORT=0

echo "AIRFLOW_HOME: $AIRFLOW_HOME"
echo "Project root: $PROJECT_ROOT"
echo "---"

cd "$PROJECT_ROOT"
exec "$PROJECT_ROOT/.venv/bin/airflow" standalone
