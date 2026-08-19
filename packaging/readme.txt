MXBM
====

A BeamHash III miner for Beam. Full docs: https://github.com/maxnflaxl/MXBM


QUICK START
-----------

1. Open mine_beam.sh (Windows: mine_beam.bat) and replace the default BEAM SBBS
   address with your own. The default is the developer's donation address.

2. Run it:

       ./mine_beam.sh            # Linux
       mine_beam.bat             # Windows (or double-click it)

Extra arguments pass through: `./mine_beam.sh --pl auto` adds a board cap.
Run ./mxbm --help for the full option list.

Test the card -- no pool, wallet or network:

       ./mxbm --benchmark BEAM-III --benchmark-seconds 120

Report how your card does:

       ./mxbm --report

Prints hardware, throughput, telemetry and the power curve as one block for the
benchmark-report issue. Needs root (Windows: an Administrator terminal) and
~25 min the first time, then reuses the stored curve. Uploads nothing.


POWER LIMIT: --pl
-----------------

--pl caps the board power in watts. Needs root (Windows: an Administrator
prompt); MXBM restores the previous limit on exit. There is no default cap:
the right one depends on the card and on what you pay for electricity.

--tune measures your card's power/speed curve and recommends a value, storing
it per card (~25 min, needs root, no pool):

       ./mxbm --tune                   # once
       ./mine_beam.sh --pl auto        # uses what it stored

Efficiency has an interior optimum. Past it, capping lower costs efficiency as
well as speed, all the way down to the card's floor -- lower is not better,
which is why it is worth measuring rather than guessing.

--cclk, --mclk, --coff, --moff and --fan work and are restored on exit, but
only --pl has had a hands-on validation pass. Read docs/overclocking.md before
the rest. Core instability shows as CPU-verify rejections, and the healthy
count is zero, so any rejection means the card is computing wrong answers.
Memory instability does not show there at all -- it shows as sol/s falling as
you raise the offset.


CONFIG FILES
------------

    ./mxbm --config mxbm.cfg        flat KEY = VALUE, one rig
    ./mxbm --json --profile RIG1    named profiles from user_config.json,
                                    with failover pool lists

Bare --json reads user_config.json and takes the first profile. --profile
without --json loads nothing. Both samples ship with placeholder addresses.


LOGS
----

    ./mxbm ... --log                writes logs/mxbm_<date>_<time>.log

A log contains the wallet address you mined with -- clear it before sharing.


DEV FEE
-------

MXBM mines for the developer a small fraction of the time, under a separate
worker name. Those solutions are counted separately and never touch your share
counters or your reported speed. --dev-fee raises the fee; it cannot be
lowered. docs/devfee.md documents it and points at the implementation.


VERIFYING THIS DOWNLOAD
-----------------------

    sha256sum -c SHA256SUMS                # Linux
    certutil -hashfile <file> SHA256       # Windows, one file at a time
