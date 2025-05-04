@echo off
echo Setting up Conway's Game of Life environment...

:: Check if Python is available
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.7+ first
    pause
    exit /b 1
)

:: Check Python version
python --version
echo.

:: Check if pip is available
where pip >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: pip is not installed or not in PATH
    pause
    exit /b 1
)

:: Option to install with or without CUDA
echo Do you want to install with CUDA support? (Requires NVIDIA GPU)
echo 1. Yes, install with CUDA (recommended for NVIDIA GPUs)
echo 2. No, install CPU-only version
set /p cuda_choice="Enter your choice (1/2): "

if "%cuda_choice%"=="1" (
    echo Installing with CUDA support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else (
    echo Installing CPU-only version...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)

:: Install other requirements
echo Installing additional requirements...
pip install numpy vispy PyQt5

:: Install the package in development mode
echo Installing BB-Life in development mode...
pip install -e .

echo.
echo Installation completed successfully!
echo You can now run the simulation using run.bat
echo.
pause 