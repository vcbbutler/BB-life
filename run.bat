@echo off
echo Starting Game of Life...

:: Check if conda is available
where conda >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: Conda is not installed or not in PATH
    echo Please install Miniconda or Anaconda first
    pause
    exit /b 1
)

:: Activate environment and run the game
call conda activate gameoflife
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to activate conda environment
    echo Please run install.bat first to set up the environment
    pause
    exit /b 1
)

python game_of_life.py
if %ERRORLEVEL% neq 0 (
    echo Error: Failed to run the game
    pause
    exit /b 1
)

pause 