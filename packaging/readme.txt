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


THE OPTIONS THAT NEED ROOT
--------------------------

    --pl W        board power limit, watts
    --cclk MHz    lock the core clock          --coff MHz  shift its V/F curve
    --mclk MHz    lock the memory clock        --moff MHz  shift its V/F curve
    --fan PCT     fan target, percent

All need root (Windows: an Administrator prompt), all take a per-GPU list
(`240`, `240,*,260`; `*` skips a card), and all are restored when MXBM exits.
--coff and --moff are offsets and may be negative; the rest are absolute.

--pl is the one to set. It caps the board in watts, and because MXBM runs
pinned at the limit in every kernel, the limit picks the operating point. There
is no default: the right cap depends on the card and on what you pay for power.

--tune measures your card's power/speed curve and recommends a value, storing
it per card (~25 min, needs root, no pool):

       ./mxbm --tune                   # once
       ./mine_beam.sh --pl auto        # uses what it stored

Efficiency has an interior optimum. Past it, capping lower costs efficiency as
well as speed, all the way down to the card's floor -- lower is not better,
which is why it is worth measuring rather than guessing.

Only --pl has had a hands-on validation pass. The clock knobs work and are
restored, but they are yours to validate on your own card:

  - Set too high, they can overheat or hang the card, take down the desktop, or
    leave it in a state only a reboot or a full power cycle clears. Move one
    knob at a time, and compare against what the card reports from a fresh boot.
  - Core instability shows as CPU-verify rejections. The healthy count is zero,
    so any rejection means the card is computing wrong answers.
  - Memory instability does not show there at all -- it shows as sol/s falling
    as you raise the offset.
  - Heat is a maintenance problem too. Clean the fans, and expect thermal pads
    and paste to need attention every few years; see the card's own guidance.

Full guide: https://github.com/maxnflaxl/MXBM/blob/master/docs/overclocking.md

WSL2: the Linux build under WSL2 cannot set power or clocks. The Windows driver
owns them and sudo does not change that, so benchmark figures from there are at
whatever the card was already set to. Use the Windows build in an Administrator
terminal for --pl, --tune and --report.


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
lowered. Full terms:
https://github.com/maxnflaxl/MXBM/blob/master/docs/devfee.md


VERIFYING THIS DOWNLOAD
-----------------------

    sha256sum -c SHA256SUMS                # Linux
    certutil -hashfile <file> SHA256       # Windows, one file at a time
