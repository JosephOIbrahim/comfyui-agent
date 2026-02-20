@echo off
REM ============================================================
REM  ComfyUI + Agent Co-Pilot Launcher
REM
REM  Starts ComfyUI, waits for it, then makes the agent available.
REM
REM  Usage:
REM    comfyui_with_agent.bat              Start everything
REM    comfyui_with_agent.bat --no-agent   ComfyUI only
REM    comfyui_with_agent.bat --cli        Agent in CLI chat mode
REM
REM  First-time:
REM    1. Edit the two PATHS below for your machine
REM    2. Double-click this file
REM    3. It will auto-run setup.py if no .env exists
REM ============================================================

REM ┌─────────────────────────────────────────────┐
REM │  EDIT THESE FOR YOUR MACHINE                │
REM └─────────────────────────────────────────────┘
set COMFYUI_BAT=G:\COMFY\ComfyUI\comfyui_zen.bat
set AGENT_DIR=C:\Users\User\comfyui-agent
set COMFYUI_PORT=8188
set MAX_WAIT=90

REM --- Parse arguments ---
set AGENT_MODE=mcp
if "%1"=="--no-agent" goto :start_comfyui_only
if "%1"=="--cli" set AGENT_MODE=cli

title ComfyUI + Agent
echo.
echo  =============================================
echo   ComfyUI + Agent Co-Pilot
echo  =============================================
echo.
echo  ComfyUI:  %COMFYUI_BAT%
echo  Agent:    %AGENT_DIR%
echo  Mode:     %AGENT_MODE%
echo.

REM --- Verify paths ---
if not exist "%COMFYUI_BAT%" (
    echo  [ERROR] ComfyUI script not found: %COMFYUI_BAT%
    echo  Edit COMFYUI_BAT at the top of this file.
    echo.
    pause
    exit /b 1
)
if not exist "%AGENT_DIR%\agent\__init__.py" (
    echo  [ERROR] Agent not found: %AGENT_DIR%
    echo  Edit AGENT_DIR at the top of this file.
    echo.
    pause
    exit /b 1
)

REM --- Auto-setup on first run ---
if not exist "%AGENT_DIR%\.env" (
    echo  No .env found. Running first-time setup...
    echo.
    cd /d "%AGENT_DIR%"
    python scripts\setup.py
    if errorlevel 1 (
        echo  Setup failed. Create .env manually from .env.example
        pause
        exit /b 1
    )
    echo.
)

REM --- Start ComfyUI ---
echo  [1/3] Starting ComfyUI...
start "ComfyUI" cmd /c ""%COMFYUI_BAT%""

REM --- Wait for ComfyUI ---
echo  [2/3] Waiting for ComfyUI (port %COMFYUI_PORT%)...
set /a ATTEMPTS=0
:wait_loop
set /a ATTEMPTS+=1
if %ATTEMPTS% gtr %MAX_WAIT% (
    echo.
    echo  [TIMEOUT] ComfyUI didn't respond in %MAX_WAIT% attempts.
    echo  Check ComfyUI window for errors. If it needs more time
    echo  to load models, increase MAX_WAIT in this script.
    pause
    exit /b 1
)
curl -s -o nul -w "%%{http_code}" http://127.0.0.1:%COMFYUI_PORT%/system_stats 2>nul | findstr "200" >nul 2>&1
if errorlevel 1 (
    set /a MOD=%ATTEMPTS% %% 5
    if %MOD%==0 echo    ... attempt %ATTEMPTS%
    timeout /t 2 /nobreak >nul
    goto :wait_loop
)
echo    Ready!
echo.

REM --- Launch agent ---
if "%AGENT_MODE%"=="cli" (
    echo  [3/3] Starting agent CLI...
    echo  Talk in plain English. Type 'quit' to exit.
    echo  =============================================
    echo.
    cd /d "%AGENT_DIR%"
    agent run
    goto :eof
)

echo  [3/3] All systems go.
echo.
echo  =============================================
echo   ComfyUI: http://127.0.0.1:%COMFYUI_PORT%/
echo  =============================================
echo.
echo  Use the agent via:
echo.
echo    Claude Code (best):   cd %AGENT_DIR% ^& claude
echo    CLI chat:             agent run
echo    Claude Desktop:       see QUICKSTART.md
echo.
echo  =============================================
echo  Press any key to close. ComfyUI stays running.
pause >nul
goto :eof

:start_comfyui_only
echo  Starting ComfyUI only...
start "ComfyUI" cmd /c ""%COMFYUI_BAT%""
echo  Done. Close this window anytime.
pause >nul
