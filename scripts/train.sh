#!/bin/bash

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Navigate to the src directory
cd ..
cd src

# Run the Python script
python main.py --train
