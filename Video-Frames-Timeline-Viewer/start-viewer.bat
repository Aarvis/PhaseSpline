@echo off
setlocal
cd /d "%~dp0"

if not exist node_modules (
  echo Installing FrameLine dependencies...
  call npm install
  if errorlevel 1 exit /b 1
)

echo.
echo FrameLine will be available at http://localhost:3000
echo Keep this window open while using the viewer.
echo Press Ctrl+C to stop it.
echo.
call npm run dev
