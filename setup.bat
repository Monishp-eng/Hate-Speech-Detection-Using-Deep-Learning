@echo off
title NNDL Project Autograder Setup
echo ==== Starting Setup for NNDL Hate Speech Project ====
echo Python Version Required: 3.8+ (Recommend 3.10+)

:: 1. Create Virtual Environment
if not exist ".venv\" (
    echo 1. Creating Python Virtual Environment '.venv'...
    python -m venv .venv
) else (
    echo 1. Virtual Environment '.venv' already exists.
)

:: 2. Activate Virtual Environment
echo 2. Activating virtual environment...
call .venv\Scripts\activate.bat

:: 3. Install Requirements
echo 3. Installing dependencies from requirements.txt...
python -m pip install --upgrade pip
pip install -r requirements.txt

:: 4. Missing Model Warning
if not exist "models\saved_models\best_bilstm_attention.pth" (
    echo.
    echo [WARNING] Pytorch model 'best_bilstm_attention.pth' is missing from 'models\saved_models\'.
    echo Please paste the trained model weights there before running the app.
    echo.
) else (
    echo 4. Verified: Model weights found.
)

:: 5. Boot Up Server
echo ==== Setup Complete! ====
echo Starting Flask Server on Localhost...
python app\app.py
pause
