@echo off

rem ##################################
rem ## Emergency script, run when   ##
rem ## the watchdog sees a GPU stop ##
rem ## producing. User defined      ##
rem ## actions to be inserted here. ##
rem ##################################

rem Wired up with:
rem     mxbm.exe ... --watchdog script --watchdogscript emergency.bat
rem
rem Without --watchdogscript the watchdog's default action is to exit 42, which
rem is the right thing when a supervisor (a rig manager, a restart loop) is
rem there to restart the miner. Use this script instead when you want to react
rem in place -- send yourself a message, log the state.

echo MXBM watchdog: a GPU stopped producing at %DATE% %TIME%
