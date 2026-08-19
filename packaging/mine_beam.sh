#!/bin/bash

#################################
## Begin of user-editable part ##
#################################

# HeroMiners BEAM pool (BeamHash III)
#   beam.herominers.com auto-routes to the nearest region.
#   Regions: us. / de. / fr. / br. / sg. / au. / ru. .beam.herominers.com
#   Port 1130 = standard (start diff 512) | Port 1131 = NiceHash (diff 4096)
POOL=beam.herominers.com:1130

# WALLET = your BEAM SBBS mining address, optionally .WORKERNAME
# The default below is the DEVELOPER'S donation address: the script runs
# without any edit, but until you replace it, everything this rig finds
# goes to the developer.
# >>> PASTE YOUR SBBS ADDRESS BELOW <<<
WALLET=12cafbe121b5f063d2c63152058575479a2826a41fd4176296dae2e8ad3fc9ffc60.donation

#################################
##  End of user-editable part  ##
#################################

cd "$(dirname "$(readlink -f "$0")")" || exit 1

# Anything you add on the command line is passed straight through, so
#   ./mine_beam.sh --pl auto
# runs this pool with the board cap --tune found. See readme.txt.
./mxbm --algo BEAM-III --pool "$POOL" --user "$WALLET" "$@"
