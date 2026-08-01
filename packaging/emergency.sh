#!/bin/bash

##################################
## Emergency script, run when   ##
## the watchdog sees a GPU stop ##
## producing. User defined      ##
## actions to be inserted here. ##
##################################

# Wired up with:
#     ./mxbm ... --watchdog script --watchdogscript ./emergency.sh
#
# Without --watchdogscript the watchdog's default action is to exit 42, which is
# the right thing when a supervisor (systemd, a rig manager, a shell loop) is
# there to restart the miner. Use this script instead when you want to react in
# place -- send yourself a message, power-cycle a riser, log the state.

echo "MXBM watchdog: a GPU stopped producing at $(date -Is)"
