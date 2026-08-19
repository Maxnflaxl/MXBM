MXBM -- a from-scratch BeamHash III miner
=========================================

This is the short version, for the archive. The full documentation lives in the
repository: https://github.com/maxnflaxl/MXBM


QUICK START
-----------

1. Open mine_beam.sh (on Windows: mine_beam.bat) and paste your BEAM SBBS
   mining address over the default one. That is the only edit needed -- and
   it matters: the default is the developer's donation address, so the script
   runs unedited, but everything it finds goes to the developer until you put
   your own address in.

2. Run it:

       ./mine_beam.sh            # Linux
       mine_beam.bat             # Windows (or double-click it)

To check the card without a pool or a wallet, with no network involved:

       ./mxbm --benchmark BEAM-III --benchmark-seconds 120

To send us how it performs -- especially if it is slow, or refuses to start --
one command produces the whole report, ready to paste into an issue:

       ./mxbm --report

It benchmarks the card, then measures its power/speed curve (about half an
hour; needs root, or an Administrator terminal on Windows) and prints
hardware, throughput, telemetry and the curve as one block. Nothing is
uploaded. A second run reuses the curve as long as the binary has not changed.

Anything you add on the command line is passed through, so
`./mine_beam.sh --pl 220` runs the same pool with a 220 W board cap.

Run ./mxbm --help for the full option list.


THE ONE SETTING WORTH CHANGING: --pl
------------------------------------

--pl caps the board power in watts. It needs root (on Windows: an Administrator
prompt), and MXBM puts the limit back where it found it when it exits.

On a 4070 Ti SUPER, 210-256 W is the window we can defend with measurements, and
220 W is a reasonable place to start. It is not the default: capping your card is
your decision to make, not ours to make quietly on your behalf.

Two things worth knowing before you tune, both measured rather than assumed --
and the whole sweep was run twice, two days apart, reproducing within ~1 %:

  * MXBM's own efficiency peaks around 200-210 W, right at the edge of the region
    where lolMiner beats us on both speed and efficiency. Below ~210 W you are
    better off with lolMiner today. We would rather say so here than have you
    find out from a spreadsheet. Note also that efficiency gets *worse* below the
    peak, all the way down to the card's 100 W floor -- lower is not better.

  * What MXBM has is the top end: a higher maximum throughput than lolMiner can
    reach at any setting.

The other overclock knobs (--cclk, --mclk, --coff, --moff, --fan) all work and
are all restored on exit, but only --pl has had a full hands-on validation pass.
Treat the rest as yours to validate on your own card, and read docs/overclocking.md
first -- particularly the part about telling memory instability apart from the
~17 % CPU-verify failure rate that is normal and algorithmic.


CONFIGURATION FILES
-------------------

Two formats, both parsed by the shipping binary:

    ./mxbm --config mxbm.cfg        flat KEY = VALUE, one rig
    ./mxbm --json --profile RIG1    named profiles from user_config.json,
                                    with failover pool lists

Bare --json (no filename) reads user_config.json; with --profile omitted, the
FIRST profile in the file is used. --profile without --json loads nothing.
Both samples in this archive have placeholder addresses in
them and will not mine until you replace them.


LOGS
----

    ./mxbm ... --log                writes logs/mxbm_<date>_<time>.log

The logs/ directory in this archive is where those land. Note that the log
contains the wallet address you mined with -- worth clearing before you share a
log or a screenshot.


THE DEV FEE, IN PLAIN LANGUAGE
------------------------------

MXBM mines for the developer for a small fraction of the time. It logs in to the
pool under a separate worker name while it does, and those solutions are counted
separately -- they never touch your own share counters or your reported speed.
You can raise the fee with --dev-fee but not lower it. docs/devfee.md is the
full description, including where in the source it is implemented so you can
check the claim rather than trust it.


VERIFYING THIS DOWNLOAD
-----------------------

    sha256sum -c SHA256SUMS                # Linux
    certutil -hashfile <file> SHA256       # Windows, one file at a time
