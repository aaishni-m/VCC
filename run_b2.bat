@echo off
REM ============================================================
REM  VCC 2026 -- B2 submission generator
REM  Double-click this file from C:\VCC to run it.
REM ============================================================

cd /d "%~dp0"

REM -- Point to the controls folder and the vcc-perturb reference --
REM    If the .part files are in C:\VCC\, the script assembles vcc-perturb.csv
REM    automatically on first run (takes ~10 seconds).
set VCC_CONTROLS_DIR=C:\Users\User\Downloads\controls
set VCC_PERTURB_CSV=C:\VCC\vcc-perturb.csv

REM -- Run the generator (~15-20 min, needs ~6-8 GB free RAM) --
echo.
echo [run_b2] Starting B2 submission generation...
echo [run_b2] Controls dir : %VCC_CONTROLS_DIR%
echo [run_b2] vcc-perturb  : %VCC_PERTURB_CSV%
echo.

python models\generate_b2_submission.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [run_b2] ERROR: generation script exited with code %ERRORLEVEL%
    echo          Check the output above for details.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [run_b2] Generation complete.
echo [run_b2] Output file: %~dp0outputs\b2_submission.h5ad
echo.
echo Next step: run vcc prep to package the submission.
echo   vcc prep outputs\b2_submission.h5ad -g "%VCC_CONTROLS_DIR%\..\gene_names.csv" --perts "%VCC_CONTROLS_DIR%\..\pert_counts.csv" -o outputs\b2_submission.vcc
echo.
pause
