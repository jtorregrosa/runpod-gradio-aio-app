#!/bin/bash

if [[ $RUNPOD_STOP_AUTO ]]
then
    echo "Skipping auto-start of Gradio application"
else
    echo "Started Gradio application through relauncher script"
    cd /app || exit
    source /opt/venv/bin/activate
    python relauncher.py "python src/app.py" &
fi