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

    section("tune_refine_caps: pass 2 brackets the coarse knee, bounded");
    {
        // The reference band's coarse grid with its knee mid-list: refine the two
        // intervals touching 248 -- (211,248) and (248,285) -- at 10 W steps.
        const std::vector<TunePoint> coarse = {
            {100, 0, 18, 0}, {137, 0, 30, 0}, {174, 0, 43, 0},
            {211, 0, 54, 0}, {248, 0, 57, 0}, {285, 0, 59, 0},
        };
        const std::vector<unsigned> f = tune_refine_caps(coarse, 248);
        const std::vector<unsigned> want = {221, 231, 241, 251, 261, 271, 281};
        check(f == want, "10 W steps across (211,285), the measured caps excluded");

        // Knee at an end of the grid: only one interval exists to refine.
        check(tune_refine_caps(coarse, 285) == std::vector<unsigned>({258, 268, 278}),
              "knee at the top: refine below it only");
        check(tune_refine_caps(coarse, 100) == std::vector<unsigned>({110, 120, 130}),
              "knee at the floor: refine above it only");

        // The step widens rather than the sweep growing: a 185 W bracket at 10 W
        // steps would be 18 points, so it opens to 25 W and stays at 7.
        const std::vector<TunePoint> wide = {{100, 0, 18, 0}, {285, 0, 59, 0}};
        const std::vector<unsigned> w = tune_refine_caps(wide, 285);
        check(w.size() == 7 && w.front() == 125 && w.back() == 275,
              "an oversized bracket widens the step (25 W) instead of the sweep");

        // Nothing to refine: a lone point, an unknown knee, a bracket under one step.
        check(tune_refine_caps({{220, 0, 54, 0}}, 220).empty(),
              "a single point has no bracket");
        check(tune_refine_caps(coarse, 200).empty(),
              "a knee that is not a measured cap refines nothing");
        check(tune_refine_caps({{100, 0, 18, 0}, {105, 0, 19, 0}, {110, 0, 20, 0}},
                               105).empty(),
              "a bracket narrower than one step is already refined");

        // Pass 4 reuses this function aimed at the best-EFFICIENCY cap, budget 4.
        // The 2026-07-31 first full run's rung series: best efficiency measured at
        // 137 W, but the true record sat at 160, between the 137 and 174 arms. With
        // max_points=4 the step widens to 15 and 160 itself lands on the grid.
        const std::vector<TunePoint> rung = {
            {211, 206.4, 42.35, 0}, {174, 171.9, 41.84, 0},
            {137, 137.2, 34.05, 0}, {100, 99.9, 22.08, 0}};
        check(tune_refine_caps(rung, 137, 4) == std::vector<unsigned>({115, 130, 145, 160}),
              "efficiency refinement lands arms inside the 137-174 gap that hid the record");
    }

    section("tune_rung_pick: the half-rate rung, from the driver's own list");
    {
        // The reference card's list: the big step is 5001, not the neighbouring
        // 10251 bin and not the deep-idle rungs.
        check(tune_rung_pick({10501, 10251, 5001, 810, 405}) == 5001,
              "reference list picks 5001");
        check(tune_rung_pick({10501, 10251, 810, 405}) == 0,
              "no mid rung: 810 is under 20 % of max, nothing worth a pass");
        check(tune_rung_pick({}) == 0, "empty list picks nothing");
        check(tune_rung_pick({8000, 4000}) == 4000,
              "exactly half-rate qualifies (below 70 %, above 20 %)");
    }

    section("tune_rung_below: where the rung stops paying, on this card's shape");
    {
        // The measured coarse shape: rung wins at 100-160, loses from ~174 up.
        const std::vector<TunePoint> stock = {
            {100, 0, 18.6, 0}, {137, 0, 27.0, 0}, {174, 0, 44.0, 0}, {211, 0, 54.0, 0}};
        const std::vector<TunePoint> rung = {
            {100, 0, 21.9, 0}, {137, 0, 31.0, 0}, {174, 0, 43.0, 0}, {211, 0, 44.0, 0}};
        const unsigned below = tune_rung_below(stock, rung);
        check(below > 137 && below < 174,
              "crossover interpolates between the last winning and first losing cap");
        check(tune_rung_below(stock, {{100, 0, 21.9, 0}, {137, 0, 31.0, 0}}) == 137,
              "rung wins at every paired cap: pays at least to the top one measured");
        check(tune_rung_below(stock, {{100, 0, 17.0, 0}, {137, 0, 25.0, 0}}) == 0,
              "rung never wins: 0, and the recommendation stays silent");
        check(tune_rung_below(stock, {{100, 0, 0.0, 0}}) == 0,
              "refused arms (sol_s 0) pair with nothing");
        // Unpaired caps (refine points, refused arms) simply do not vote.
        check(tune_rung_below(stock, {{120, 0, 30.0, 0}}) == 0,
              "a rung cap with no stock partner cannot place a crossover");
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
