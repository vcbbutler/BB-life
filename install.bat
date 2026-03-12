@echo off
setlocal enabledelayedexpansion
echo Setting up Conway's Game of Life environment...
echo.

:: Detect Conda: PATH, then Miniconda, then Anaconda
set "CONDA_FOUND="
where conda >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set "CONDA_FOUND=1"
    echo Conda found in PATH.
)
if not defined CONDA_FOUND (
    if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
        set "CONDA_FOUND=1"
        call "%USERPROFILE%\miniconda3\Scripts\activate.bat" base
        echo Conda found: Miniconda3.
    )
)
if not defined CONDA_FOUND (
    if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
        set "CONDA_FOUND=1"
        call "%USERPROFILE%\anaconda3\Scripts\activate.bat" base
        echo Conda found: Anaconda3.
    )
)

if not defined CONDA_FOUND (
    echo Error: Conda is not installed or could not be found.
    echo.
    echo run.bat requires a Conda environment named 'gameoflife'.
    echo Install Miniconda for Windows, then run this script again.
    echo Download: https://docs.conda.io/en/latest/miniconda.html
    echo.
    pause
    exit /b 1
)
echo.

:: Create gameoflife env if it does not exist
conda env list | findstr /C:"gameoflife" >nul 2>nul
if errorlevel 1 (
    echo Creating Conda environment 'gameoflife' with Python 3.10...
    conda create -n gameoflife python=3.10 -y
    if errorlevel 1 (
        echo Failed to create Conda environment.
        pause
        exit /b 1
    )
    echo.
) else (
    echo Conda environment 'gameoflife' already exists.
)

:: Activate gameoflife (use same paths as run.bat)
echo Activating environment 'gameoflife'...
if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\miniconda3\Scripts\activate.bat" gameoflife
) else if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\anaconda3\Scripts\activate.bat" gameoflife
) else (
    call conda activate gameoflife
)

if "%CONDA_DEFAULT_ENV%" NEQ "gameoflife" (
    echo Failed to activate 'gameoflife'. Try: conda activate gameoflife
    pause
    exit /b 1
)
echo Successfully activated: %CONDA_DEFAULT_ENV%
python --version
echo.

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
