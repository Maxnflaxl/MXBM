// The --tune recommendation, pinned as pure arithmetic -- no GPU, no NVML, no
// sweep. run_tune() is the measurement harness around exactly these two functions,
// so a regression here is a wrong `--pl auto` on every rig that trusted a tune.
#include "miner/tune.h"
#include "check.h"
using namespace mxbm;
using namespace mxbm::miner;

int main() {
    section("tune_verdict: the knee stops where a watt stops paying");
    {
        // Shaped like the reference card's measured curve (docs/performance.md,
        // "Power and efficiency"): steep to ~220 W, flat above it. With the 0.07
        // sol/s per W criterion the 220->285 step (0.04/W) is the first that fails.
        std::vector<TunePoint> pts = {
            {100, 99.5, 16.3, 103.9},
            {160, 159.6, 39.0, 51.5},
            {190, 189.8, 48.2, 41.7},
            {220, 219.6, 53.8, 37.3},
            {285, 267.0, 56.4, 33.3},
        };
        const TuneVerdict v = tune_verdict(pts, 0.07);
        check(v.knee_w == 220, "knee at 220 W: the 285 W step returns 0.04 sol/s per W");
        // Efficiency is priced on measured DRAW (267 at the 285 cap), and this
        // curve peaks in the middle, not at the floor: 0.164 at 100 W, 0.254 at
        // 190, 0.245 at 220 -- matching the measured peak-near-200 W shape.
        check(v.eff_w == 190, "efficiency peak mid-band, on measured draw");

        // Order independence: the sweep runs high-to-low, the verdict must not care.
        std::vector<TunePoint> rev(pts.rbegin(), pts.rend());
        const TuneVerdict vr = tune_verdict(rev, 0.07);
        check(vr.knee_w == v.knee_w && vr.eff_w == v.eff_w, "point order is irrelevant");
    }

    section("tune_verdict: edges hold");
    {
        check(tune_verdict({}, 0.07).knee_w == 0, "no points -> no recommendation");
        const TuneVerdict one = tune_verdict({{220, 219.0, 53.8, 37.3}}, 0.07);
        check(one.knee_w == 220 && one.eff_w == 220, "one point -> that point, both roles");
        // A failed point (sol_s <= 0: solver threw, window aborted) is dropped, not
        // treated as a real zero that would poison the marginal on both sides.
        const TuneVerdict drop = tune_verdict({{160, 159.0, 39.0, 51.5},
                                               {190, 0.0, 0.0, 0.0},
                                               {220, 219.0, 53.8, 37.3}}, 0.07);
        check(drop.knee_w == 220, "a failed point vanishes: 160->220 still pays (0.25/W)");
        // The knee walk BREAKS at the first failing step -- the curve is concave, so
        // a later step that happens to pay again (noise) must not resurrect it.
        const TuneVerdict concave = tune_verdict({{100, 0.0, 40.0, 0.0},
                                                  {150, 0.0, 41.0, 0.0},
                                                  {200, 0.0, 46.0, 0.0}}, 0.07);
        check(concave.knee_w == 100, "first failing step ends the climb for good");
        // Draw unknown (no NVML power field): the CAP prices efficiency instead.
        const TuneVerdict nodraw = tune_verdict({{100, 0.0, 16.3, 0.0},
                                                 {220, 0.0, 53.8, 0.0}}, 0.07);
        check(nodraw.eff_w == 220, "cap-priced fallback: 53.8/220 beats 16.3/100");
    }

    section("tune_default_caps: the band is the driver's, not ours");
    {
        const std::vector<unsigned> c = tune_default_caps(100, 285);
        check(c.size() == 6 && c.front() == 100 && c.back() == 285,
              "reference band 100-285: six points, both ends included");
        check(c[1] == 137 && c[2] == 174 && c[3] == 211 && c[4] == 248,
              "evenly spaced between the ends");
        check(tune_default_caps(0, 285).empty(), "no reported minimum -> no candidates");
        check(tune_default_caps(285, 285).empty(), "empty band -> no candidates");
        // A band too narrow to hold six distinct integers collapses; the caller
        // (run_tune) refuses to place a knee on fewer than two points.
        check(tune_default_caps(100, 103).size() >= 2, "narrow band still keeps its ends");
    }

    return summary("tune");
}
