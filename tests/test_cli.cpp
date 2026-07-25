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

    // missing --pool no longer fails parse_args itself -- a config file
    // (--json/--config, Task 5) may supply pools instead. parse_args just
    // leaves out.pools empty and seen.pools false; main() enforces "at
    // least one pool, from CLI or config" AFTER the config-file merge. See
    // tests/test_config.cpp for the config-merge coverage.
    {
        const char* av2[] = {"mxbm","--algo","BEAM-III","--user","addr.rig1"};
        Options o2; std::string err2;
        check(parse_args((int)(sizeof(av2)/sizeof(av2[0])),(char**)av2,o2,err2), "missing --pool now parses ok (config may supply pools)");
        check(o2.pools.empty() && !o2.seen.pools, "no --pool leaves pools empty, seen.pools false");
    }

    // --tls 0 -> tls false
    {
        const char* av3[] = {"mxbm","--algo","BEAM-III","--pool","beam.2miners.com:5252","--user","addr.rig1","--tls","0"};
        Options o3; std::string err3;
        check(parse_args((int)(sizeof(av3)/sizeof(av3[0])),(char**)av3,o3,err3), "parse ok with --tls 0");
        check(o3.pools[0].tls==false, "--tls 0 disables tls");
        check(o3.seen.tls, "seen.tls true when --tls was passed");
    }

    // --algo ETHASH -> false (wrong algo rejected)
    {
        const char* av4[] = {"mxbm","--algo","ETHASH","--pool","beam.2miners.com:5252","--user","addr.rig1"};
        Options o4; std::string err4;
        check(!parse_args((int)(sizeof(av4)/sizeof(av4[0])),(char**)av4,o4,err4), "unsupported algo rejected");
    }

    // --help -> false, help_requested true, usage text in err
    {
        const char* av5[] = {"mxbm","--help"};
        Options o5; std::string err5;
        check(!parse_args((int)(sizeof(av5)/sizeof(av5[0])),(char**)av5,o5,err5), "--help returns false");
        check(o5.help_requested, "--help sets help_requested");
        check(!err5.empty(), "--help produces usage text");
    }

    // unknown flag --frobnicate -> false
    {
        const char* av6[] = {"mxbm","--algo","BEAM-III","--pool","beam.2miners.com:5252","--user","addr.rig1","--frobnicate"};
        Options o6; std::string err6;
        check(!parse_args((int)(sizeof(av6)/sizeof(av6[0])),(char**)av6,o6,err6), "unknown flag rejected");
        check(!err6.empty(), "unknown flag error message non-empty");
    }

    // two pools with two users bind positionally
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

    // one user + two pools applies to both
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

    // --pool/--user need not be adjacent -- still bound by occurrence order
    {
        const char* av9[] = {"mxbm","--algo","BEAM-III",
            "--user","firstUser","--pool","poolA.example.com:1111",
            "--user","secondUser","--pool","poolB.example.com:2222"};
        Options o9; std::string err9;
        check(parse_args((int)(sizeof(av9)/sizeof(av9[0])),(char**)av9,o9,err9), "interleaved pool/user parse ok");
        check(o9.pools.size()==2 && o9.pools[0].user=="firstUser" && o9.pools[1].user=="secondUser",
              "interleaved --user/--pool still bind by occurrence order");
    }

    // --version -> false, version_requested, empty err
    {
        const char* av10[] = {"mxbm","--version"};
        Options o10; std::string err10;
        check(!parse_args((int)(sizeof(av10)/sizeof(av10[0])),(char**)av10,o10,err10), "--version returns false");
        check(o10.version_requested, "--version sets version_requested");
        check(err10.empty(), "--version leaves err empty");
    }

    // --apiport 8080 parses; --apiport 99999 (out of range) errors
    {
        const char* av11[] = {"mxbm","--algo","BEAM-III","--pool","beam.2miners.com:5252","--user","addr.rig1","--apiport","8080"};
        Options o11; std::string err11;
        check(parse_args((int)(sizeof(av11)/sizeof(av11[0])),(char**)av11,o11,err11), "--apiport 8080 parses");
        check(o11.apiport==8080 && o11.seen.apiport, "apiport value + seen flag set");

        const char* av12[] = {"mxbm","--algo","BEAM-III","--pool","beam.2miners.com:5252","--user","addr.rig1","--apiport","99999"};
        Options o12; std::string err12;
        check(!parse_args((int)(sizeof(av12)/sizeof(av12[0])),(char**)av12,o12,err12), "--apiport 99999 rejected");
    }

    // --shortstats 0 errors (must be >=1)
    {
        const char* av13[] = {"mxbm","--algo","BEAM-III","--pool","beam.2miners.com:5252","--user","addr.rig1","--shortstats","0"};
        Options o13; std::string err13;
        check(!parse_args((int)(sizeof(av13)/sizeof(av13[0])),(char**)av13,o13,err13), "--shortstats 0 rejected");
    }

    // --nocolour (British spelling) aliases --nocolor
    {
        const char* av14[] = {"mxbm","--algo","BEAM-III","--pool","beam.2miners.com:5252","--user","addr.rig1","--nocolour"};
        Options o14; std::string err14;
        check(parse_args((int)(sizeof(av14)/sizeof(av14[0])),(char**)av14,o14,err14), "--nocolour parses");
        check(o14.nocolor && o14.seen.nocolor, "--nocolour alias sets nocolor + seen.nocolor");
    }

    // bare --json -> use_json_config + default ./user_config.json path
    {
        const char* av15[] = {"mxbm","--algo","BEAM-III","--pool","beam.2miners.com:5252","--user","addr.rig1","--json"};
        Options o15; std::string err15;
        check(parse_args((int)(sizeof(av15)/sizeof(av15[0])),(char**)av15,o15,err15), "bare --json parses");
        check(o15.use_json_config && o15.config_path=="user_config.json",
              "bare --json defaults to user_config.json");
    }

    // --json my.json --profile rig1 captures both
    {
        const char* av16[] = {"mxbm","--algo","BEAM-III","--pool","beam.2miners.com:5252","--user","addr.rig1",
            "--json","my.json","--profile","rig1"};
        Options o16; std::string err16;
        check(parse_args((int)(sizeof(av16)/sizeof(av16[0])),(char**)av16,o16,err16), "--json my.json --profile rig1 parses");
        check(o16.use_json_config && o16.config_path=="my.json" && o16.json_profile=="rig1",
              "--json PATH + --profile NAME both captured");
    }

    // --config PATH captures a flat-config path, leaves use_json_config false
    {
        const char* av17[] = {"mxbm","--algo","BEAM-III","--pool","beam.2miners.com:5252","--user","addr.rig1",
            "--config","mxbm.cfg"};
        Options o17; std::string err17;
        check(parse_args((int)(sizeof(av17)/sizeof(av17[0])),(char**)av17,o17,err17), "--config PATH parses");
        check(!o17.use_json_config && o17.config_path=="mxbm.cfg",
              "--config sets config_path, leaves use_json_config false");
    }

    // --devices is stored verbatim as a string, even a bare "0"
    {
        const char* av18[] = {"mxbm","--algo","BEAM-III","--pool","beam.2miners.com:5252","--user","addr.rig1",
            "--devices","0"};
        Options o18; std::string err18;
        check(parse_args((int)(sizeof(av18)/sizeof(av18[0])),(char**)av18,o18,err18), "--devices 0 parses");
        check(o18.devices=="0" && o18.seen.devices, "--devices stores the raw string + sets seen.devices");
    }

    // positive paths for the stats/watchdog flags (Seen fields feed Task 5's config merge)
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
    // --solver gpu/ref/auto accepted (value stored, seen set); --solver bogus rejected
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

    // bare --json immediately followed by another flag must not swallow it
    {
        const char* av[] = {"mxbm","--algo","BEAM-III","--pool","pool.example.com:1130","--user","addr123.rig1",
                            "--json","--profile","rig1"};
        Options o; std::string e;
        check(parse_args(10,(char**)av,o,e), "bare --json before --profile parses");
        check(o.use_json_config && o.config_path == "user_config.json", "bare --json defaults path");
        check(o.json_profile == "rig1", "--profile survives after bare --json");
    }

    // --dev-fee: a percentage the user opts into paying. Parsing accepts any
    // 0..100 value; the raise-only rule (a value below the built-in rate is
    // refused outright) lives in main(), which is where that rate is known.
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
        // A bare --log/--timeprint must not swallow the flag that follows --
        // the same trap --tls's optional value avoids.
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
        // Naming a file is itself the request to log to it -- but that rule is
        // resolved AFTER any config merge (a config may supply either half), so
        // parse_args only records the path and the Seen flag. See
        // cli::resolve_implied_options.
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
        // ...but an explicit --log off still wins, wherever it appears.
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

    return summary("cli");
}
