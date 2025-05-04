@echo off
setlocal enabledelayedexpansion

echo.
echo +-------------------------------------+
echo ^|      Conway's Game of Life 3D       ^|
echo +-------------------------------------+
echo ^| A GPU-accelerated implementation    ^|
echo ^| with mutation and age-based colors  ^|
echo +-------------------------------------+
echo.

:: Check for Python in PATH
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python not found! Please install Python and add it to your PATH.
    pause
    exit /b 1
)

:: Display Python version
python --version
echo.

:: Run the simulation with default settings
echo Starting simulation...
echo.
echo Press SPACE to start/pause the simulation
echo.

:: Check if CUDA is available and set appropriate device
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
set CUDA_AVAILABLE=0
for /f "tokens=*" %%a in ('python -c "import torch; print(1 if torch.cuda.is_available() else 0)"') do set CUDA_AVAILABLE=%%a

if %CUDA_AVAILABLE%==1 (
    echo CUDA is available, using GPU acceleration
    python main.py --device cuda
) else (
    echo CUDA is not available, using CPU
    python main.py --device cpu
)

if %errorlevel% neq 0 (
    echo.
    echo An error occurred while running the simulation.
    pause
    exit /b 1
)

exit /b 0 