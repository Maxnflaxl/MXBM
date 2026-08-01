@echo off
setlocal

rem Offline benchmark: no pool, no wallet, no network. Prints ms/solve and
rem sol/s for this card and exits.
rem
rem If you are reporting a number back to the project, please run it for at
rem least 120 seconds and say what your board power limit was -- a short run
rem overstates sol/s (the solutions/solve figure has not converged yet), and
rem the same card measures about 5 percent apart between sessions on the same
rem binaries. Both effects are documented in docs/performance.md.

cd /d "%~dp0"

rem First argument is the duration in seconds only when it is a number;
rem   benchmark.bat --pl 220
rem passes everything through and keeps the 120 s default.
set "SECS=120"
echo %~1| findstr /r "^[0-9][0-9]*$" >nul && set "SECS=%~1" && shift

rem cmd's %* ignores shift, so rebuild the remaining arguments by hand.
set "ARGS="
:collect
if "%~1"=="" goto run
set ARGS=%ARGS% %1
shift
goto collect

:run
mxbm.exe --benchmark BEAM-III --benchmark-seconds %SECS%%ARGS%
pause
