@echo off
echo Setting up Game of Life environment...

:: Check if conda is available
where conda >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: Conda is not installed or not in PATH
    echo Please install Miniconda or Anaconda first
    pause
    exit /b 1
)

:: Create and activate conda environment
echo Creating conda environment...
call conda create -n gameoflife python=3.9 -y
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to create conda environment
    pause
    exit /b 1
)

:: Activate environment and install requirements
echo Installing requirements...
call conda activate gameoflife
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to activate conda environment
    pause
    exit /b 1
)

:: Install PyTorch with CUDA support directly from PyTorch channel
call conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to install PyTorch with CUDA
    pause
    exit /b 1
)

:: Install other requirements
pip install numpy>=1.21.0 matplotlib>=3.4.0
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to install additional requirements
    pause
    exit /b 1
)

echo.
echo Installation completed successfully!
echo You can now run the game using run.bat
echo.
pause 