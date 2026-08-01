@echo off
setlocal

rem #################################
rem ## Begin of user-editable part ##
rem #################################

rem HeroMiners BEAM pool (BeamHash III)
rem   beam.herominers.com auto-routes to the nearest region.
rem   Regions: us. / de. / fr. / br. / sg. / au. / ru. .beam.herominers.com
rem   Port 1130 = standard (start diff 512) / Port 1131 = NiceHash (diff 4096)
set "POOL=beam.herominers.com:1130"

rem WALLET = your BEAM SBBS mining address, optionally .WORKERNAME
rem The default below is the DEVELOPER'S donation address: the script runs
rem without any edit, but until you replace it, everything this rig finds
rem goes to the developer.
rem >>> PASTE YOUR SBBS ADDRESS BELOW <<<
set "WALLET=12cafbe121b5f063d2c63152058575479a2826a41fd4176296dae2e8ad3fc9ffc60.donation"

rem #################################
rem ##  End of user-editable part  ##
rem #################################

cd /d "%~dp0"

rem Anything you add on the command line is passed straight through, so
rem   mine_beam.bat --pl 220
rem runs this pool with a 220 W board cap. See readme.txt.
mxbm.exe --algo BEAM-III --pool "%POOL%" --user "%WALLET%" %*
pause
