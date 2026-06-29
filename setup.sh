#!/bin/bash
# MacOS/Linux NNDL Automated Grader Setup Script

echo "==== Starting Setup for NNDL Hate Speech Project ===="
echo "Python Version Required: 3.8+ (Recommend 3.10+)"

# 1. Check if a virtual environment already exists
if [ ! -d ".venv" ]; then
    echo "1. Creating Python Virtual Environment '.venv'..."
    python3 -m venv .venv
else
    echo "1. Virtual Environment '.venv' already exists."
fi

# 2. Activate the virtual environment
echo "2. Activating virtual environment..."
source .venv/bin/activate

# 3. Install Requirements
echo "3. Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Check for model file
mkdir -p models/saved_models
if [ ! -f "models/saved_models/best_bilstm_attention.pth" ]; then
    echo "WARNING: Pytorch model 'best_bilstm_attention.pth' is missing from the models/saved_models folder."
    echo "Please ensure the trained weight file is copy-pasted there before running the server."
else
    echo "4. Verified: Model weights found."
fi

# 5. Start Server
echo "==== Setup Complete! ===="
echo "Starting Flask Server for grading..."
python app/app.py
