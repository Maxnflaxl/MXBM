#!/bin/bash

# Offline benchmark: no pool, no wallet, no network. Prints ms/solve and sol/s
# for this card and exits.
#
# If you are reporting a number back to the project, please run it for at least
# 120 seconds and say what your board power limit was -- a short run overstates
# sol/s (the solutions/solve figure has not converged yet), and the same card
# measures ~5 % apart between sessions on the same binaries. Both effects are
# documented in docs/performance.md.

cd "$(dirname "$(readlink -f "$0")")" || exit 1

# First argument is the duration in seconds only when it is a number;
#   ./benchmark.sh --pl 220
# passes everything through and keeps the 120 s default.
SECS=120
case "${1:-}" in ''|*[!0-9]*) ;; *) SECS=$1; shift ;; esac

./mxbm --benchmark BEAM-III --benchmark-seconds "$SECS" "$@"
