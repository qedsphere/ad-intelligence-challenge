#!/bin/bash

set -e

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo -n "Creating virtual environment...    "
    python3 -m venv .venv
    echo "DONE!"
fi

# Activate virtual environment
source .venv/bin/activate

echo -n "Verifying Python dependencies...    "
pip install -r requirements.txt
echo "DONE!"

if [ ! -d "ads" ]; then
    echo -n "Setting up ad dataset...    "
    pip install -q gdown
    FILE_ID="1DDj8l59RyEoE2zeeYAqfg9wHiiZehUfg"
    OUTPUT_FILE="ads.zip"
    gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "${OUTPUT_FILE}"
    unzip -q -o "${OUTPUT_FILE}" -d .

    rm -rf __MACOSX
    rm -rf "${OUTPUT_FILE}"
    echo "DONE!"
fi

echo ""
echo "Setup complete!"

