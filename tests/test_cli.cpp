#include "check.h"
#include "cli/options.h"
using namespace mxbm; using namespace mxbm::cli;
int main() {
    const char* av[] = {"mxbm","--algo","BEAM-III","--pool","beam.2miners.com:5252","--user","addr.rig1"};
    Options o; std::string err;
    check(parse_args((int)(sizeof(av)/sizeof(av[0])),(char**)av,o,err), "parse ok");
    check(o.pools.size()==1, "single pool parsed");
    check(o.pools[0].host=="beam.2miners.com" && o.pools[0].port==5252, "pool split");
    check(o.pools[0].user=="addr.rig1" && o.pools[0].tls==true, "user + tls default on");
    check(o.shortstats==15 && o.longstats==60, "shortstats/longstats defaults");
    check(o.apiport==0 && !o.use_json_config && o.config_path.empty() &&
          o.json_profile.empty() && !o.watchdog_requested && o.devices.empty() &&
          o.solver=="auto",
          "new fields default correctly");
    check(o.seen.pools && o.seen.user && !o.seen.tls && !o.seen.pass &&
          !o.seen.apiport && !o.seen.shortstats && !o.seen.longstats &&
          !o.seen.devices && !o.seen.nocolor && !o.seen.solver,
          "seen flags reflect exactly what was passed");

    // Missing --pool does not fail parse_args: a config file may supply pools
    // instead, so main() enforces "at least one pool" AFTER the config merge.
    // See tests/test_config.cpp for the merge coverage.
    {
        const char* av2[] = {"mxbm","--algo","BEAM-III","--user","addr.rig1"};
        Options o2; std::string err2;
        check(parse_args((int)(sizeof(av2)/sizeof(av2[0])),(char**)av2,o2,err2), "missing --pool now parses ok (config may supply pools)");
        check(o2.pools.empty() && !o2.seen.pools, "no --pool leaves pools empty, seen.pools false");
    }

    {
        const char* av3[] = {"mxbm","--algo","BEAM-III","--pool","beam.2miners.com:5252","--user","addr.rig1","--tls","0"};
        Options o3; std::string err3;
        check(parse_args((int)(sizeof(av3)/sizeof(av3[0])),(char**)av3,o3,err3), "parse ok with --tls 0");
        check(o3.pools[0].tls==false, "--tls 0 disables tls");
        check(o3.seen.tls, "seen.tls true when --tls was passed");
    }

    {
        const char* av4[] = {"mxbm","--algo","ETHASH","--pool","beam.2miners.com:5252","--user","addr.rig1"};
        Options o4; std::string err4;
        check(!parse_args((int)(sizeof(av4)/sizeof(av4[0])),(char**)av4,o4,err4), "unsupported algo rejected");
    }

    {
        const char* av5[] = {"mxbm","--help"};
        Options o5; std::string err5;
        check(!parse_args((int)(sizeof(av5)/sizeof(av5[0])),(char**)av5,o5,err5), "--help returns false");
        check(o5.help_requested, "--help sets help_requested");
        check(!err5.empty(), "--help produces usage text");
    }

    {
        const char* av6[] = {"mxbm","--algo","BEAM-III","--pool","beam.2miners.com:5252","--user","addr.rig1","--frobnicate"};
        Options o6; std::string err6;
        check(!parse_args((int)(sizeof(av6)/sizeof(av6[0])),(char**)av6,o6,err6), "unknown flag rejected");
        check(!err6.empty(), "unknown flag error message non-empty");
    }

    {
        const char* av7[] = {"mxbm","--algo","BEAM-III",
            "--pool","poolA.example.com:1111","--pool","poolB.example.com:2222",
            "--user","userA.rig1","--user","userB.rig2"};
        Options o7; std::string err7;
        check(parse_args((int)(sizeof(av7)/sizeof(av7[0])),(char**)av7,o7,err7), "two pools + two users parse ok");
        check(o7.pools.size()==2, "two pools recorded");
        check(o7.pools[0].host=="poolA.example.com" && o7.pools[0].port==1111 && o7.pools[0].user=="userA.rig1",
              "pool 1 bound to user 1");
        check(o7.pools[1].host=="poolB.example.com" && o7.pools[1].port==2222 && o7.pools[1].user=="userB.rig2",
              "pool 2 bound to user 2");
    }

    {
        const char* av8[] = {"mxbm","--algo","BEAM-III",
            "--pool","poolA.example.com:1111","--pool","poolB.example.com:2222",
            "--user","solo.rig1"};
        Options o8; std::string err8;
        check(parse_args((int)(sizeof(av8)/sizeof(av8[0])),(char**)av8,o8,err8), "two pools + one user parse ok");
        check(o8.pools.size()==2, "two pools recorded");
        check(o8.pools[0].user=="solo.rig1" && o8.pools[1].user=="solo.rig1",
              "single --user applies to all pools");
    }

    {
        const char* av9[] = {"mxbm","--algo","BEAM-III",
            "--user","firstUser","--pool","poolA.example.com:1111",
            "--user","secondUser","--pool","poolB.example.com:2222"};
        Options o9; std::string err9;
        check(parse_args((int)(sizeof(av9)/sizeof(av9[0])),(char**)av9,o9,err9), "interleaved pool/user parse ok");
        check(o9.pools.size()==2 && o9.pools[0].user=="firstUser" && o9.pools[1].user=="secondUser",
              "interleaved --user/--pool still bind by occurrence order");
    }

    {
        const char* av10[] = {"mxbm","--version"};
        Options o10; std::string err10;
        check(!parse_args((int)(sizeof(av10)/sizeof(av10[0])),(char**)av10,o10,err10), "--version returns false");
        check(o10.version_requested, "--version sets version_requested");
        check(err10.empty(), "--version leaves err empty");
    }

    {
        const char* av11[] = {"mxbm","--algo","BEAM-III","--pool","beam.2miners.com:5252","--user","addr.rig1","--apiport","8080"};
        Options o11; std::string err11;
        check(parse_args((int)(sizeof(av11)/sizeof(av11[0])),(char**)av11,o11,err11), "--apiport 8080 parses");
        check(o11.apiport==8080 && o11.seen.apiport, "apiport value + seen flag set");

        const char* av12[] = {"mxbm","--algo","BEAM-III","--pool","beam.2miners.com:5252","--user","addr.rig1","--apiport","99999"};
        Options o12; std::string err12;
        check(!parse_args((int)(sizeof(av12)/sizeof(av12[0])),(char**)av12,o12,err12), "--apiport 99999 rejected");
    }

    {
        const char* av13[] = {"mxbm","--algo","BEAM-III","--pool","beam.2miners.com:5252","--user","addr.rig1","--shortstats","0"};
        Options o13; std::string err13;
        check(!parse_args((int)(sizeof(av13)/sizeof(av13[0])),(char**)av13,o13,err13), "--shortstats 0 rejected");
    }

    {
        const char* av14[] = {"mxbm","--algo","BEAM-III","--pool","beam.2miners.com:5252","--user","addr.rig1","--nocolour"};
        Options o14; std::string err14;
        check(parse_args((int)(sizeof(av14)/sizeof(av14[0])),(char**)av14,o14,err14), "--nocolour parses");
        check(o14.nocolor && o14.seen.nocolor, "--nocolour alias sets nocolor + seen.nocolor");
    }

    {
        const char* av15[] = {"mxbm","--algo","BEAM-III","--pool","beam.2miners.com:5252","--user","addr.rig1","--json"};
        Options o15; std::string err15;
        check(parse_args((int)(sizeof(av15)/sizeof(av15[0])),(char**)av15,o15,err15), "bare --json parses");
        check(o15.use_json_config && o15.config_path=="user_config.json",
              "bare --json defaults to user_config.json");
    }

    {
        const char* av16[] = {"mxbm","--algo","BEAM-III","--pool","beam.2miners.com:5252","--user","addr.rig1",
            "--json","my.json","--profile","rig1"};
        Options o16; std::string err16;
        check(parse_args((int)(sizeof(av16)/sizeof(av16[0])),(char**)av16,o16,err16), "--json my.json --profile rig1 parses");
        check(o16.use_json_config && o16.config_path=="my.json" && o16.json_profile=="rig1",
              "--json PATH + --profile NAME both captured");
    }

    {
        const char* av17[] = {"mxbm","--algo","BEAM-III","--pool","beam.2miners.com:5252","--user","addr.rig1",
            "--config","mxbm.cfg"};
        Options o17; std::string err17;
        check(parse_args((int)(sizeof(av17)/sizeof(av17[0])),(char**)av17,o17,err17), "--config PATH parses");
        check(!o17.use_json_config && o17.config_path=="mxbm.cfg",
              "--config sets config_path, leaves use_json_config false");
    }

    {
        const char* av18[] = {"mxbm","--algo","BEAM-III","--pool","beam.2miners.com:5252","--user","addr.rig1",
            "--devices","0"};
        Options o18; std::string err18;
        check(parse_args((int)(sizeof(av18)/sizeof(av18[0])),(char**)av18,o18,err18), "--devices 0 parses");
        check(o18.devices=="0" && o18.seen.devices, "--devices stores the raw string + sets seen.devices");
    }

    {
        const char* av[] = {"mxbm","--algo","BEAM-III","--pool","pool.example.com:1130","--user","addr123.rig1",
                            "--shortstats","5","--longstats","120","--watchdog"};
        Options o; std::string e;
        check(parse_args(12,(char**)av,o,e), "stats/watchdog flags parse ok");
        check(o.shortstats == 5 && o.seen.shortstats, "--shortstats 5 sets value + seen");
        check(o.longstats == 120 && o.seen.longstats, "--longstats 120 sets value + seen");
        check(o.watchdog_requested, "--watchdog sets watchdog_requested");
        check(!o.seen.apiport && !o.seen.devices, "unrelated seen flags stay false");
    }
    {
        const char* avG[] = {"mxbm","--algo","BEAM-III","--pool","beam.2miners.com:5252","--user","addr.rig1",
                              "--solver","gpu"};
        Options oG; std::string eG;
        check(parse_args((int)(sizeof(avG)/sizeof(avG[0])),(char**)avG,oG,eG), "--solver gpu parses");
        check(oG.solver=="gpu" && oG.seen.solver, "--solver gpu stores value + sets seen.solver");

        const char* avR[] = {"mxbm","--algo","BEAM-III","--pool","beam.2miners.com:5252","--user","addr.rig1",
                              "--solver","ref"};
        Options oR; std::string eR;
        check(parse_args((int)(sizeof(avR)/sizeof(avR[0])),(char**)avR,oR,eR), "--solver ref parses");
        check(oR.solver=="ref" && oR.seen.solver, "--solver ref stores value + sets seen.solver");

        const char* avA[] = {"mxbm","--algo","BEAM-III","--pool","beam.2miners.com:5252","--user","addr.rig1",
                              "--solver","auto"};
        Options oA; std::string eA;
        check(parse_args((int)(sizeof(avA)/sizeof(avA[0])),(char**)avA,oA,eA), "--solver auto parses");
        check(oA.solver=="auto" && oA.seen.solver, "--solver auto stores value + sets seen.solver");

        const char* avB[] = {"mxbm","--algo","BEAM-III","--pool","beam.2miners.com:5252","--user","addr.rig1",
                              "--solver","bogus"};
        Options oB; std::string eB;
        check(!parse_args((int)(sizeof(avB)/sizeof(avB[0])),(char**)avB,oB,eB), "--solver bogus rejected");
        check(!eB.empty(), "--solver bogus error message non-empty");
    }

    {
        const char* av[] = {"mxbm","--algo","BEAM-III","--pool","pool.example.com:1130","--user","addr123.rig1",
                            "--json","--profile","rig1"};
        Options o; std::string e;
        check(parse_args(10,(char**)av,o,e), "bare --json before --profile parses");
        check(o.use_json_config && o.config_path == "user_config.json", "bare --json defaults path");
        check(o.json_profile == "rig1", "--profile survives after bare --json");
    }

    // Parsing accepts any 0..100 value; the raise-only rule lives in main(),
    // which is where the built-in rate is known, and is not covered here.
    {
        const char* av[] = {"mxbm","--algo","BEAM-III","--pool","pool.example.com:1130","--user","addr123.rig1",
                            "--dev-fee","2.5"};
        Options o; std::string e;
        check(parse_args(9,(char**)av,o,e), "--dev-fee 2.5 parses");
        check(o.devfee_pct == 2.5 && o.seen.devfee, "--dev-fee stores the percent + sets seen.devfee");
    }
    {
        const char* av[] = {"mxbm","--algo","BEAM-III","--pool","pool.example.com:1130","--user","addr123.rig1"};
        Options o; std::string e;
        check(parse_args(7,(char**)av,o,e), "no --dev-fee parses");
        check(o.devfee_pct < 0.0 && !o.seen.devfee,
              "absent --dev-fee leaves the built-in rate in force");
    }
    for (const char* bad : {"-1", "101", "abc", "2.5x", ""}) {
        const char* av[] = {"mxbm","--algo","BEAM-III","--pool","pool.example.com:1130","--user","addr123.rig1",
                            "--dev-fee", bad};
        Options o; std::string e;
        check(!parse_args(9,(char**)av,o,e), "--dev-fee rejects an out-of-range or non-numeric value");
        check(!e.empty(), "--dev-fee rejection carries an error message");
    }
    {
        const char* av[] = {"mxbm","--algo","BEAM-III","--pool","pool.example.com:1130","--user","addr123.rig1",
                            "--dev-fee"};
        Options o; std::string e;
        check(!parse_args(8,(char**)av,o,e), "--dev-fee with no value is rejected");
    }

    // -- --log / --logfile / --timeprint / --digits --
    {
        const char* av[] = {"mxbm","--algo","BEAM-III","--pool","p:1130","--user","a.r1"};
        Options o; std::string e;
        check(parse_args(7,(char**)av,o,e), "parse ok with none of the log flags");
        check(!o.log_enabled && o.log_path.empty() && !o.timeprint && o.digits == 2,
              "defaults: no log, no stamp on the short line, two decimals");
    }
    {
        const char* av[] = {"mxbm","--algo","BEAM-III","--log","--timeprint","--pool","p:1130","--user","a.r1"};
        Options o; std::string e;
        check(parse_args(9,(char**)av,o,e), "bare --log/--timeprint parse");
        check(o.log_enabled && o.timeprint, "a bare flag means on");
        check(o.pools.size() == 1 && o.pools[0].host == "p",
              "a bare --log does not swallow the next flag");
    }
    {
        const char* av[] = {"mxbm","--algo","BEAM-III","--pool","p:1130","--user","a.r1",
                            "--log","off","--timeprint","0"};
        Options o; std::string e;
        check(parse_args(11,(char**)av,o,e), "explicit off values parse");
        check(!o.log_enabled && !o.timeprint, "off/0 turn them back off");
    }
    {
        // Naming a file is itself the request to log to it, but that rule is
        // resolved after any config merge (either source may supply either
        // half), so parse_args only records the path and the Seen flag.
        const char* av[] = {"mxbm","--algo","BEAM-III","--pool","p:1130","--user","a.r1",
                            "--logfile","/tmp/x.log"};
        Options o; std::string e;
        check(parse_args(9,(char**)av,o,e), "--logfile parses");
        check(o.log_path == "/tmp/x.log" && o.seen.logfile, "--logfile records the path");
        check(!o.seen.log, "--logfile alone does not claim to have set the switch");
        resolve_implied_options(o);
        check(o.log_enabled, "--logfile implies --log once the sources are merged");
    }
    {
        const char* av[] = {"mxbm","--algo","BEAM-III","--pool","p:1130","--user","a.r1",
                            "--log","off","--logfile","/tmp/x.log"};
        Options o; std::string e;
        check(parse_args(11,(char**)av,o,e), "--log off alongside a --logfile parses");
        resolve_implied_options(o);
        check(!o.log_enabled, "an explicit --log off beats --logfile's implication");
    }
    {
        const char* av[] = {"mxbm","--algo","BEAM-III","--pool","p:1130","--user","a.r1","--digits","4"};
        Options o; std::string e;
        check(parse_args(9,(char**)av,o,e), "--digits parses");
        check(o.digits == 4 && o.seen.digits, "--digits is recorded");
    }
    {
        const char* av[] = {"mxbm","--algo","BEAM-III","--pool","p:1130","--user","a.r1","--digits","9"};
        Options o; std::string e;
        check(!parse_args(9,(char**)av,o,e), "--digits rejects a value outside 0..6");
        check(!e.empty(), "the --digits rejection carries a message");
    }
    {
        const char* av[] = {"mxbm","--algo","BEAM-III","--pool","p:1130","--user","a.r1","--logfile"};
        Options o; std::string e;
        check(!parse_args(8,(char**)av,o,e), "--logfile with no value is rejected");
    }

    // -- --pl / --no-oc-reset --
    // The WATTS are not range-checked here: the legal band is what the driver
    // reports for the actual card (100-366 W on the reference one), which the
    // CLI cannot see. Syntax is checked here, the value at apply time.
    {
        const char* av[] = {"mxbm","--algo","BEAM-III","--pool","p:1130","--user","a.r1"};
        Options o; std::string e;
        check(parse_args(7,(char**)av,o,e), "no --pl parses");
        check(o.power_limit.empty() && !o.seen.power_limit, "--pl defaults to unset (card untouched)");
        check(!o.no_oc_reset && !o.seen.no_oc_reset, "--no-oc-reset defaults to off (restore on exit)");
    }
    {
        const char* av[] = {"mxbm","--algo","BEAM-III","--pool","p:1130","--user","a.r1","--pl","220"};
        Options o; std::string e;
        check(parse_args(9,(char**)av,o,e), "--pl parses");
        check(o.power_limit == "220" && o.seen.power_limit, "--pl is recorded verbatim");
    }
    {
        const char* av[] = {"mxbm","--algo","BEAM-III","--pool","p:1130","--user","a.r1","--pl","240,*,260"};
        Options o; std::string e;
        check(parse_args(9,(char**)av,o,e), "--pl accepts the per-GPU list form");
        check(o.power_limit == "240,*,260", "the whole list is kept for the apply step");
    }
    {
        const char* av[] = {"mxbm","--algo","BEAM-III","--pool","p:1130","--user","a.r1","--pl","abc"};
        Options o; std::string e;
        check(!parse_args(9,(char**)av,o,e), "--pl rejects a non-numeric value");
        check(!e.empty(), "the --pl rejection carries a message");
    }
    {
        const char* av[] = {"mxbm","--algo","BEAM-III","--pool","p:1130","--user","a.r1","--pl"};
        Options o; std::string e;
        check(!parse_args(8,(char**)av,o,e), "--pl with no value is rejected");
    }
    {
        const char* av[] = {"mxbm","--algo","BEAM-III","--pool","p:1130","--user","a.r1","--no-oc-reset"};
        Options o; std::string e;
        check(parse_args(8,(char**)av,o,e), "bare --no-oc-reset parses");
        check(o.no_oc_reset && o.seen.no_oc_reset, "bare --no-oc-reset means on");
    }
    {
        // Optional value, like --log/--timeprint: a bare flag must not swallow
        // the next flag as its argument.
        const char* av[] = {"mxbm","--algo","BEAM-III","--pool","p:1130","--user","a.r1",
                            "--no-oc-reset","--pl","220"};
        Options o; std::string e;
        check(parse_args(10,(char**)av,o,e), "--no-oc-reset does not swallow the next flag");
        check(o.no_oc_reset && o.power_limit == "220", "both flags land");
    }
    {
        const char* av[] = {"mxbm","--algo","BEAM-III","--pool","p:1130","--user","a.r1","--no-oc-reset","0"};
        Options o; std::string e;
        check(parse_args(9,(char**)av,o,e), "--no-oc-reset 0 parses");
        check(!o.no_oc_reset && o.seen.no_oc_reset, "--no-oc-reset 0 means off, but is seen");
    }

    section("the clock and fan knobs parse like --pl, and the offsets take a sign");
    {
        const char* av[] = {"mxbm","--algo","BEAM-III","--pool","p:1130","--user","a.r1",
                            "--cclk","2100","--mclk","10000","--coff","-200","--moff","1500",
                            "--fan","70"};
        Options o; std::string e;
        check(parse_args(17,(char**)av,o,e), "all five knobs parse together");
        check(o.core_clock == "2100" && o.mem_clock == "10000", "the two locks land");
        check(o.core_offset == "-200" && o.mem_offset == "1500",
              "a NEGATIVE core offset survives the parser -- it is an undervolt, not a typo");
        check(o.fan == "70", "and the fan target lands");
        check(o.seen.core_clock && o.seen.fan, "each records that it was seen");
    }
    {
        const char* av[] = {"mxbm","--algo","BEAM-III","--pool","p:1130","--user","a.r1",
                            "--cclk","-2100"};
        Options o; std::string e;
        check(!parse_args(9,(char**)av,o,e),
              "a negative LOCKED clock is rejected -- unlike an offset, it means nothing");
    }
    {
        const char* av[] = {"mxbm","--algo","BEAM-III","--pool","p:1130","--user","a.r1","--fan"};
        Options o; std::string e;
        check(!parse_args(8,(char**)av,o,e), "a knob with no value is an error, not a default");
    }

    section("--devices resolves against the devices actually present");
    {
        std::vector<unsigned> sel; std::string e;
        // "all" and an empty spec both mean everything, so a config file that
        // sets DEVICES = ALL behaves the same as not setting it at all.
        check(resolve_devices("ALL", 3, sel, e) && sel.size() == 3, "ALL selects every device");
        check(resolve_devices("all", 3, sel, e) && sel.size() == 3, "and is case-insensitive");
        check(resolve_devices("", 2, sel, e) && sel.size() == 2, "an empty spec means all too");
        check(resolve_devices("0,2", 3, sel, e) && sel.size() == 2 && sel[0] == 0 && sel[1] == 2,
              "a list selects exactly those, in the order given");
        check(resolve_devices("2,0", 3, sel, e) && sel[0] == 2 && sel[1] == 0,
              "and preserves that order rather than sorting it");
        check(resolve_devices("1,1", 3, sel, e) && sel.size() == 1,
              "a duplicate collapses -- mining one card twice would halve it and look like a fault");
        check(!resolve_devices("3", 3, sel, e), "an index past the end is an error");
        check(e.find("does not exist") != std::string::npos, "and the message says so");
        check(!resolve_devices("0,3", 3, sel, e),
              "a bad entry is rejected even when a good one precedes it");
        check(!resolve_devices("abc", 3, sel, e), "a non-number is an error");
        check(!resolve_devices("0,,1", 3, sel, e), "so is an empty entry");
        check(!resolve_devices("-1", 3, sel, e), "so is a negative index");
        check(!resolve_devices("0", 0, sel, e),
              "asking for a specific device when none was detected is an error, not a no-op");
        check(resolve_devices("ALL", 0, sel, e) && sel.empty(),
              "but ALL of nothing is simply nothing, which the caller reports its own way");
    }

    {
        // --tune and --pl auto. Both are pass-through at parse time: the watts and
        // the store lookup belong to the driver and to main(), which the CLI cannot
        // see -- exactly the --pl precedent.
        const char* t1[] = {"mxbm","--tune"};
        Options ot; std::string et;
        check(parse_args(2,(char**)t1,ot,et), "--tune parses with nothing else");
        check(ot.tune && ot.seen.algo, "--tune is a mode and satisfies --algo itself");
        check(ot.tune_seconds == 60 && ot.tune_caps.empty() && ot.tune_knee == 0.07,
              "tune defaults: 60 s per point, driver-band caps, 0.07 sol/s per W knee");

        const char* t2[] = {"mxbm","--tune","--tune-seconds","30",
                            "--tune-caps","100,160,220","--tune-knee","0.05"};
        Options ot2; std::string et2;
        check(parse_args(8,(char**)t2,ot2,et2), "tune knobs parse");
        check(ot2.tune_seconds == 30 && ot2.tune_caps == "100,160,220"
              && ot2.tune_knee == 0.05, "tune knobs land");

        const char* t3[] = {"mxbm","--tune","--tune-seconds","5"};
        Options ot3; std::string et3;
        check(!parse_args(4,(char**)t3,ot3,et3),
              "a 5 s point is refused: that window measures ramp, not rate");
        const char* t4[] = {"mxbm","--tune","--tune-caps","100,abc"};
        Options ot4; std::string et4;
        check(!parse_args(4,(char**)t4,ot4,et4), "a non-wattage in --tune-caps is an error");

        const char* t5[] = {"mxbm","--algo","BEAM-III","--pool","p:1","--user","u",
                            "--pl","AUTO"};
        Options ot5; std::string et5;
        check(parse_args(9,(char**)t5,ot5,et5), "--pl auto parses (any case)");
        check(ot5.power_limit == "auto" && ot5.seen.power_limit,
              "and is normalized to lowercase for main() to resolve");
        const char* t6[] = {"mxbm","--algo","BEAM-III","--pool","p:1","--user","u",
                            "--pl","autoo"};
        Options ot6; std::string et6;
        check(!parse_args(9,(char**)t6,ot6,et6), "near-miss spellings stay errors");
    }

    // -- short flags --
    {
        const char* s1[] = {"mxbm","-a","BEAM-III","-p","beam.2miners.com:5252","-u","addr.rig1"};
        Options os1; std::string es1;
        check(parse_args(7,(char**)s1,os1,es1), "-a -p -u parse as their long forms");
        check(os1.seen.algo && os1.pools.size()==1 && os1.pools[0].user=="addr.rig1",
              "and land in the same fields");

        const char* s2[] = {"mxbm","-h"};
        Options os2; std::string es2;
        check(!parse_args(2,(char**)s2,os2,es2) && os2.help_requested, "-h is --help");
        const char* s3[] = {"mxbm","-v"};
        Options os3; std::string es3;
        check(!parse_args(2,(char**)s3,os3,es3) && os3.version_requested, "-v is --version");

        const char* s4[] = {"mxbm","-au","x"};
        Options os4; std::string es4;
        check(!parse_args(3,(char**)s4,os4,es4), "bundled short flags are refused");
        const char* s5[] = {"mxbm","-z"};
        Options os5; std::string es5;
        check(!parse_args(2,(char**)s5,os5,es5), "an unknown short flag is an error");
    }

    // -- --coin --
    {
        const char* c1[] = {"mxbm","-c","beam","--pool","p:1","--user","u"};
        Options oc1; std::string ec1;
        check(parse_args(7,(char**)c1,oc1,ec1), "--coin beam parses, any case");
        check(oc1.seen.algo, "and satisfies the algo requirement on its own");
        const char* c2[] = {"mxbm","--coin","ETH","--pool","p:1","--user","u"};
        Options oc2; std::string ec2;
        check(!parse_args(7,(char**)c2,oc2,ec2), "a coin we do not mine is an error");
    }

    // -- --list-algos / --list-coins --
    {
        const char* l1[] = {"mxbm","--list-algos"};
        Options ol1; std::string el1;
        check(parse_args(2,(char**)l1,ol1,el1) && ol1.list_algos && !ol1.list_coins,
              "--list-algos parses without a pool or algo");
        const char* l2[] = {"mxbm","--list-coins"};
        Options ol2; std::string el2;
        check(parse_args(2,(char**)l2,ol2,el2) && ol2.list_coins, "--list-coins parses too");
    }

    // -- --silence / --compactaccept / --devicesbypcie --
    {
        const char* v1[] = {"mxbm","--algo","BEAM-III","--pool","p:1","--user","u",
                            "--silence","2","--compactaccept","--devicesbypcie"};
        Options ov1; std::string ev1;
        check(parse_args(11,(char**)v1,ov1,ev1), "the verbosity flags parse together");
        check(ov1.silence==2 && ov1.seen.silence, "--silence 2 stored");
        check(ov1.compactaccept && ov1.seen.compactaccept, "--compactaccept stored");
        check(ov1.devices_by_pcie && ov1.seen.devices_by_pcie, "--devicesbypcie stored");

        Options od2; std::string ed2;
        const char* d2[] = {"mxbm","--algo","BEAM-III","--pool","p:1","--user","u"};
        check(parse_args(7,(char**)d2,od2,ed2), "parse ok");
        check(od2.silence==0 && !od2.compactaccept && !od2.devices_by_pcie,
              "all three default off");

        for (const char* bad : {"4", "-1", "abc", ""}) {
            const char* av[] = {"mxbm","--algo","BEAM-III","--pool","p:1","--user","u",
                                "--silence", bad};
            Options ob; std::string eb;
            const std::string msg = std::string("--silence rejects '") + bad + "'";
            check(!parse_args(9,(char**)av,ob,eb), msg.c_str());
        }
    }

    // -- --statsformat / --vstats / --hstats --
    {
        const char* f1[] = {"mxbm","--algo","BEAM-III","--pool","p:1","--user","u",
                            "--statsformat","gpuName,speed,power"};
        Options of1; std::string ef1;
        check(parse_args(9,(char**)f1,of1,ef1), "--statsformat parses a field list");
        check(of1.statsformat == "gpuName,speed,power" && of1.seen.statsformat, "and is stored");

        const char* f2[] = {"mxbm","--algo","BEAM-III","--pool","p:1","--user","u",
                            "--statsformat","extended"};
        Options of2; std::string ef2;
        check(!parse_args(9,(char**)f2,of2,ef2), "a preset name is refused at parse time");
        check(ef2.find("no presets") != std::string::npos, "with an error that explains it");

        const char* v1[] = {"mxbm","--algo","BEAM-III","--pool","p:1","--user","u","--vstats"};
        Options ov1; std::string ev1;
        check(parse_args(8,(char**)v1,ov1,ev1) && ov1.vstats && ov1.stats_width == 0,
              "a bare --vstats takes no width");

        const char* h1[] = {"mxbm","--algo","BEAM-III","--pool","p:1","--user","u","--hstats","60"};
        Options oh1; std::string eh1;
        check(parse_args(9,(char**)h1,oh1,eh1) && oh1.hstats && oh1.stats_width == 60,
              "--hstats takes an optional width");

        const char* h2[] = {"mxbm","--algo","BEAM-III","--pool","p:1","--user","u","--hstats"};
        Options oh2; std::string eh2;
        check(parse_args(8,(char**)h2,oh2,eh2) && oh2.hstats && oh2.stats_width == 0,
              "and without one asks the terminal (width 0)");

        // A following flag is a flag, not a width.
        const char* h3[] = {"mxbm","--algo","BEAM-III","--pool","p:1","--user","u",
                            "--hstats","--nocolor"};
        Options oh3; std::string eh3;
        check(parse_args(9,(char**)h3,oh3,eh3) && oh3.hstats && oh3.nocolor && oh3.stats_width == 0,
              "--hstats does not swallow the next flag");

        const char* b[] = {"mxbm","--algo","BEAM-III","--pool","p:1","--user","u",
                           "--vstats","--hstats"};
        Options ob; std::string eb;
        check(!parse_args(9,(char**)b,ob,eb), "--vstats and --hstats together are refused");
    }

    // -- --tstop / --tstart / --tmode --
    {
        const char* t1[] = {"mxbm","--algo","BEAM-III","--pool","p:1","--user","u",
                            "--tstop","85","--tstart","75","--tmode","edge"};
        Options ot1; std::string et1;
        check(parse_args(13,(char**)t1,ot1,et1), "the thermal flags parse together");
        check(ot1.tstop==85 && ot1.tstart==75 && ot1.tmode=="edge" && ot1.seen.tstop,
              "and are stored");

        const char* t2[] = {"mxbm","--algo","BEAM-III","--pool","p:1","--user","u",
                            "--tstop","70","--tstart","75"};
        Options ot2; std::string et2;
        check(!parse_args(11,(char**)t2,ot2,et2),
              "a restart temperature at or above the stop point is refused: it could only flap");

        const char* t3[] = {"mxbm","--algo","BEAM-III","--pool","p:1","--user","u",
                            "--tmode","hotspot"};
        Options ot3; std::string et3;
        check(!parse_args(9,(char**)t3,ot3,et3), "an unknown sensor name is an error");

        const char* t4[] = {"mxbm","--algo","BEAM-III","--pool","p:1","--user","u",
                            "--tstop","999"};
        Options ot4; std::string et4;
        check(!parse_args(9,(char**)t4,ot4,et4), "a temperature no silicon reaches is an error");
    }

    // -- --keepfree --
    {
        const char* k1[] = {"mxbm","--algo","BEAM-III","--pool","p:1","--user","u",
                            "--keepfree","512"};
        Options ok1; std::string ek1;
        check(parse_args(9,(char**)k1,ok1,ek1) && ok1.keepfree_mb == 512 && ok1.seen.keepfree,
              "--keepfree takes megabytes");
        const char* k2[] = {"mxbm","--algo","BEAM-III","--pool","p:1","--user","u",
                            "--keepfree","0"};
        Options ok2; std::string ek2;
        check(parse_args(9,(char**)k2,ok2,ek2) && ok2.keepfree_mb == 0 && ok2.seen.keepfree,
              "0 is a value, not an absence: it means take everything free");
        Options ok3; std::string ek3;
        const char* k3[] = {"mxbm","--algo","BEAM-III","--pool","p:1","--user","u"};
        check(parse_args(7,(char**)k3,ok3,ek3) && ok3.keepfree_mb < 0 && !ok3.seen.keepfree,
              "and not given is negative, so the built-in reserve applies");
        const char* k4[] = {"mxbm","--algo","BEAM-III","--pool","p:1","--user","u",
                            "--keepfree","-5"};
        Options ok4; std::string ek4;
        check(!parse_args(9,(char**)k4,ok4,ek4), "a negative reserve is an error");
    }

    // -- --devices against real identities: PCI addresses and vendor keywords --
    {
        const std::vector<DeviceRef> rig = {
            {"1:0",  "NVIDIA Corporation"},
            {"41:0", "Advanced Micro Devices, Inc."},
            {"c1:0", "NVIDIA Corporation"},
        };
        std::vector<unsigned> sel; std::string e;

        check(resolve_devices("ALL", rig, false, sel, e) && sel.size()==3, "ALL takes every card");
        check(resolve_devices("2,0", rig, false, sel, e) && sel.size()==2 && sel[0]==2 && sel[1]==0,
              "an index list keeps the order typed");

        check(resolve_devices("NVIDIA", rig, false, sel, e) && sel.size()==2 && sel[0]==0 && sel[1]==2,
              "a vendor keyword selects that vendor's cards");
        check(resolve_devices("amd", rig, false, sel, e) && sel.size()==1 && sel[0]==1,
              "'amd' matches 'Advanced Micro Devices, Inc.'");
        check(!resolve_devices("intel", rig, false, sel, e), "a vendor with no cards is an error");

        check(resolve_devices("41:0", rig, true, sel, e) && sel.size()==1 && sel[0]==1,
              "--devicesbypcie selects by address");
        check(resolve_devices("0000:41:00.0", rig, true, sel, e) && sel.size()==1 && sel[0]==1,
              "the long lspci form names the same card");
        check(resolve_devices("41:00", rig, true, sel, e) && sel.size()==1 && sel[0]==1,
              "so does the zero-padded form");
        check(resolve_devices("C1:0,1:0", rig, true, sel, e) && sel.size()==2 && sel[0]==2 && sel[1]==0,
              "addresses are case-insensitive hex and keep the order typed");
        check(!resolve_devices("9:0", rig, true, sel, e), "an address no card has is an error");
        check(!resolve_devices("notanaddress", rig, true, sel, e), "so is a malformed address");

        // Without identities the vendor and address forms have nothing to match.
        std::vector<unsigned> s2; std::string e2;
        check(!resolve_devices("NVIDIA", 2, s2, e2), "vendor keywords need identities to match against");
        check(resolve_devices("1", 2, s2, e2) && s2.size()==1 && s2[0]==1,
              "the count-only form still resolves indices");
    }

    // -- --apihost --
    {
        Options od; std::string ed;
        const char* d1[] = {"mxbm","--algo","BEAM-III","--pool","p:1","--user","u"};
        check(parse_args(7,(char**)d1,od,ed), "parse ok");
        check(od.apihost == "0.0.0.0" && !od.seen.apihost, "--apihost defaults to every interface");

        const char* h1[] = {"mxbm","--algo","BEAM-III","--pool","p:1","--user","u",
                            "--apihost","127.0.0.1"};
        Options oh1; std::string eh1;
        check(parse_args(9,(char**)h1,oh1,eh1), "--apihost 127.0.0.1 parses");
        check(oh1.apihost == "127.0.0.1" && oh1.seen.apihost, "and is stored verbatim");

        const char* bad[][2] = {{"--apihost","300.1.1.1"}, {"--apihost","1.2.3"},
                                {"--apihost","1.2.3.4.5"}, {"--apihost","localhost"},
                                {"--apihost","1.2.3."},    {"--apihost",""}};
        for (const auto& b : bad) {
            const char* av[] = {"mxbm","--algo","BEAM-III","--pool","p:1","--user","u",
                                b[0], b[1]};
            Options ob; std::string eb;
            const std::string msg = std::string("--apihost rejects '") + b[1] + "'";
            check(!parse_args(9,(char**)av,ob,eb), msg.c_str());
        }
    }

    return summary("cli");
}
