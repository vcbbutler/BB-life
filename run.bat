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

:: Try to find Anaconda and activate the environment
echo Activating Conda environment 'gameoflife'...
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" gameoflife || call "%CONDA_EXE%\..\..\Scripts\activate.bat" gameoflife || call conda activate gameoflife

if "%CONDA_DEFAULT_ENV%" NEQ "gameoflife" (
    echo WARNING: Failed to activate Conda environment 'gameoflife'.
    echo Please ensure Conda is installed and the 'gameoflife' environment exists.
    echo The script will try to proceed with the default Python, but might fail.
    echo You may need to activate it manually: conda activate gameoflife
) else (
    echo Successfully activated Conda environment: %CONDA_DEFAULT_ENV%
)
echo.

:: Check for Python in PATH (now hopefully from the Conda env)
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
    echo Attempting to deactivate Conda environment before exiting...
    call conda deactivate
    pause
    exit /b 1
)

echo Attempting to deactivate Conda environment...
call conda deactivate
exit /b 0 