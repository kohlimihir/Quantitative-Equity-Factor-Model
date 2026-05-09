@echo off
echo.
echo ========================================
echo   Equity Factor Model Dashboard
echo ========================================
echo.

REM Check if data exists
if not exist "data\ensemble_optimize_predictions.csv" (
    echo WARNING: No predictions found!
    echo Please run the model first:
    echo   python run_all.py --config optimized_low_turnover
    echo.
    pause
    exit /b 1
)

echo Starting Streamlit dashboard...
echo Dashboard will open at: http://localhost:8501
echo.
echo Press Ctrl+C to stop
echo.

streamlit run app.py
