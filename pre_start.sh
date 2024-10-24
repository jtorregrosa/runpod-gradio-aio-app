#!/bin/bash

# Best practices for safety
#set -euo pipefail
IFS=$'\n\t'

# Directories
WORKSPACE="${WORKSPACE:-/workspace}"
SOURCES="/sources"
APP_DIR="$WORKSPACE/app"

# Check if the "app" directory exists in /workspace
if [ ! -d "$APP_DIR" ]; then
  echo "The 'app' directory does not exist in ${WORKSPACE}. Copying..."
  cp -r "$SOURCES" "$APP_DIR"
  echo "Copy completed."
else
  echo "The 'app' directory already exists in ${WORKSPACE}. No copy performed."
fi


if [[ $RUNPOD_STOP_AUTO ]]
then
    echo "Skipping auto-start of Gradio application"
else
    echo "Started Gradio application through relauncher script"
    cd "${APP_DIR}" || exit
    source /opt/venv/bin/activate
    python /relauncher.py "python $APP_DIR/app.py > /proc/1/fd/1 2>/proc/1/fd/2" &
fi