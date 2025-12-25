#!/bin/bash
# Ocean Sentry Launcher

echo "🌊 Starting Ocean Sentry AI..."
echo "--------------------------------"

# 1. Navigate to the correct directory
cd /home/marine/ocean_trash_detection/backend

# 2. Run the AI (System Python)
python3 inference_pi.py

# 3. Wait before closing so user can read errors if any
echo "--------------------------------"
echo "Program finished."
read -p "Press ENTER to close this window..."
