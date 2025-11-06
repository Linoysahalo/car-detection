@echo off
echo =====================================
echo   launching Car Detection App, pls wait..
echo =====================================
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo couldnt find python, make sure its installed
    pause
    exit /b
)

REM call venv\Scripts\activate

echo Checking dependencies..
pip install -r requirements.txt --quiet

python ui.py

echo.
echo app execution finished
pause
