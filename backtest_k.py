#!/usr/bin/env python3
# ============================================================================
# backtest_k.py -- the actual k=1.0 vs k=0.5 validation this session's whole
# asof_rate_cache.py build was FOR. Reads every asof_rates/{date}.json
# snapshot (batter/pitcher pre-game rates + real starter matchups + real
# outcomes), reconstructs what project_hit()/project_hr() would have raw-
# projected on each historical date under the CURRENT formula (log5, k=1)
# and the EXPERIMENTAL one (log5_dampened, k=0.5), and checks both against
# what actually happened -- same log_loss/reliability_table machinery
# calibrate.py already uses, imported not re-derived, plus the same
# chronological holdout discipline (never shuffled -- shuffling leaks the
# future) established there and in measure_signals.py.
#
# WHY k=0.5 AT ALL: diagnosed earlier this session -- log5's odds-ratio
# multiplication (odds(batter)/odds(league)) x (odds(pitcher)/odds(league))
# predictably overcompounds when a genuinely above-average hitter faces a
# genuinely below-average pitcher, which is exactly the d9/d10 decile
# overconfidence pattern that reproduced across 4 consecutive live-ledger
# checkpoints. log5_dampened raises each odds-ratio to a power k<1 before
# recombining -- k=1 reduces EXACTLY to the existing log5() (verified below,
# not assumed), and k<1 specifically shrinks compounding in proportion to how
# far each input deviates from league average, leaving near-neutral matchups
# almost untouched.
#
# odds()/log5_dampened() are EXPERIMENTAL and intentionally NOT added to
# mlb_model.py -- they live here only, so nothing about the live board
# changes just by running this script. If k=0.5 survives holdout here, wiring
# it into project_hit()/project_hr() is a separate, deliberate step.
#
# SCOPE (same as agreed before any of this was built): batter/pitcher
# hit-rate and HR-rate only. No platoon splits, no situational context, no
# park factors -- batter_hit_rate_per_pa() is called with pitcher_hand=None
# so its split branch never fires, same as passing no split data at all.
#
# ACTUAL PA, NOT PROJECTED: game_prob() needs a PA count to turn a per-PA
# rate into a per-game probability. This uses each batter's REAL PA that
# game (recorded in the snapshot's outcome block), not expected_pa(lineup
# slot) the way a live pre-game board must. Deliberate: expected_pa() carries
# its own, unrelated error (lineup-slot projection), and conflating that with
# whatever the log5 compounding fix does or doesn't do would make the k
# result impossible to interpret cleanly. This backtest is isolating ONE
# thing -- same discipline as ruling out roster churn before touching d9/d10
# again.
#
# USAGE: python backtest_k.py [--snapshot-dir asof_rates] [--k 0.5]
# Needs calibrate.py and mlb_model.py importable from the same directory.
# ============================================================================

import argparse
import json
import math
import os
import sys

import mlb_model as M
from calibrate import MIN_ACTUAL_PA, log_loss, reliability_table

MARKETS = ("hit", "hr")


# --------------------- experimental formula, isolated here -------------------

def odds(p):
    p = M.clamp01(p)
    return p / (1 - p)


def log5_dampened(a, b, l, k):
    """Reduces EXACTLY to mlb_model.log5(a,b,l) at k=1 -- a generalization,
    not a replacement formula. See module docstring for the compounding
    rationale. Verified against the real log5() below before any real row
    is scored."""
    a, b, l = M.clamp01(a), M.clamp01(b), M.clamp01(l)
    ra, rb = odds(a) / odds(l), odds(b) / odds(l)
    odds_p = odds(l) * (ra ** k) * (rb ** k)
    return odds_p / (1 + odds_p)


def _self_check_k1_matches_log5():
    import random
    random.seed(1)
    max_diff = 0.0
    for _ in range(2000):
        a = random.uniform(.02, .6)
        b = random.uniform(.02, .6)
        l = random.uniform(.05, .4)
        max_diff = max(max_diff, abs(M.log5(a, b, l) - log5_dampened(a, b, l, 1.0)))
    if max_diff > 1e-9:
        raise SystemExit(f"log5_dampened(k=1) does NOT match log5() -- max diff "
                          f"{max_diff}. Refusing to score anything on a formula "
                          f"that isn't verified. Fix before rerunning.")
    print(f"Self-check OK: log5_dampened(k=1) matches log5() (max diff {max_diff:.2e})",
          file=sys.stderr)


# ------------------------------ snapshot loading ------------------------------

def load_snapshots(snapshot_dir):
    dates = sorted(f[:-5] for f in os.listdir(snapshot_dir) if f.endswith(".json"))
    for d in dates:
        try:
            with open(os.path.join(snapshot_dir, d + ".json")) as f:
                yield json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"WARN: unreadable snapshot {d} ({type(e).__name__}) -- skipped",
                  file=sys.stderr)


# --------------------------- one row per real matchup -------------------------

def build_rows(snapshot_dir, k):
    """Every real (date, batter, opposing starter) triple with actual PA >=
    MIN_ACTUAL_PA (same playing-time gate calibrate.py fits on, imported not
    re-derived), scored under both formulas. batter_hit_rate_per_pa/
    batter_hr_rate_per_pa/_pitcher_rate are called EXACTLY as project_hit()/
    project_hr() call them -- reusing the real functions, not reimplementing
    the merge logic a second time."""
    rows = []
    skipped_pa = skipped_no_starter = skipped_no_batter = 0
    for snap in load_snapshots(snapshot_dir):
        league = snap.get("league") or {}
        lh, lhr = league.get("hitRatePerPA"), league.get("hrRatePerPA")
        if lh is None or lhr is None:
            continue  # off day / empty snapshot
        batters, pitchers = snap["batters"], snap["pitchers"]
        for g in snap.get("games", []):
            for b in g["batters"]:
                pa = b["outcome"]["pa"]
                if pa < MIN_ACTUAL_PA:
                    skipped_pa += 1
                    continue
                h = batters.get(str(b["id"]))
                if h is None:
                    skipped_no_batter += 1
                    continue
                opp_id = b.get("oppStarterId")
                p = pitchers.get(str(opp_id)) if opp_id is not None else None
                if p is None:
                    skipped_no_starter += 1
                    # not fatal -- project_hit/project_hr both fall back to
                    # league on a missing pitcher side; keep the row so this
                    # matches live behavior exactly, just note the count.

                # --- HIT: exact same merge as project_hit() lines 326-337 ---
                batter_rate = M.batter_hit_rate_per_pa(h, None, lh)
                pitcher_rate = (M._pitcher_rate(p, "hitRateAllowedPerPA",
                                                M.PITCHER_HIT_PRIOR_BF, lh)
                                if p else None)
                a_h = batter_rate if batter_rate is not None else lh
                b_h = pitcher_rate if pitcher_rate is not None else lh
                raw_hit_orig = M.log5(a_h, b_h, lh)
                raw_hit_damp = log5_dampened(a_h, b_h, lh, k)

                # --- HR: exact same merge as project_hr() lines 399-410 ---
                batter_hr = M.batter_hr_rate_per_pa(h, lhr)
                pitcher_hr = (M._pitcher_rate(p, "hrRateAllowedPerPA",
                                              M.PITCHER_HR_PRIOR_BF, lhr)
                             if p else None)
                a_hr = batter_hr if batter_hr is not None else lhr
                b_hr = pitcher_hr if pitcher_hr is not None else lhr
                raw_hr_orig = M.log5(a_hr, b_hr, lhr)
                raw_hr_damp = log5_dampened(a_hr, b_hr, lhr, k)

                rows.append({
                    "date": snap["date"], "hitterId": b["id"], "actualPA": pa,
                    "hitRawOrig": M.game_prob(raw_hit_orig, pa),
                    "hitRawDamp": M.game_prob(raw_hit_damp, pa),
                    "gotHit": 1 if b["outcome"]["hits"] > 0 else 0,
                    "hrRawOrig": M.game_prob(raw_hr_orig, pa),
                    "hrRawDamp": M.game_prob(raw_hr_damp, pa),
                    "gotHR": 1 if b["outcome"]["hr"] > 0 else 0,
                })
    rows.sort(key=lambda r: (r["date"], r["hitterId"]))
    print(f"Built {len(rows)} rows (skipped {skipped_pa} under {MIN_ACTUAL_PA}-PA gate, "
          f"{skipped_no_batter} with no prior-batter snapshot, {skipped_no_starter} with "
          f"no opposing-starter data [scored on league fallback, not dropped])",
          file=sys.stderr)
    return rows


# --------------------------------- reporting ----------------------------------

def decile_gap(table, idx):
    """observedRate - meanPredicted for bucket idx (0-based; 8=d9, 9=d10)."""
    if idx >= len(table):
        return None
    row = table[idx]
    return round(row["observedRate"] - row["meanPredicted"], 4)


def report_market(market, rows, split_frac=0.7):
    orig_key, damp_key, y_key = f"{market}RawOrig", f"{market}RawDamp", (
        "gotHit" if market == "hit" else "gotHR")
    cut = int(len(rows) * split_frac)
    train, val = rows[:cut], rows[cut:]
    out = {"market": market, "nTotal": len(rows), "nTrain": len(train), "nVal": len(val)}
    for split_name, split in (("train", train), ("val", val)):
        y = [r[y_key] for r in split]
        orig = [r[orig_key] for r in split]
        damp = [r[damp_key] for r in split]
        tbl_orig = reliability_table(orig, y)
        tbl_damp = reliability_table(damp, y)
        out[split_name] = {
            "logLossOrig": round(log_loss(orig, y), 5),
            "logLossDamp": round(log_loss(damp, y), 5),
            "d9GapOrig": decile_gap(tbl_orig, 8), "d9GapDamp": decile_gap(tbl_damp, 8),
            "d10GapOrig": decile_gap(tbl_orig, 9), "d10GapDamp": decile_gap(tbl_damp, 9),
            "reliabilityOrig": tbl_orig, "reliabilityDamp": tbl_damp,
        }
    return out


def print_summary(report, k):
    for m in report["markets"]:
        v = m["val"]
        print(f"\n=== {m['market'].upper()} (n={m['nTotal']}, train={m['nTrain']}, "
              f"val={m['nVal']}) ===")
        print(f"  val log loss   orig={v['logLossOrig']:.5f}   "
              f"k={k} damp={v['logLossDamp']:.5f}   "
              f"({'damp better' if v['logLossDamp'] < v['logLossOrig'] else 'orig better'})")
        print(f"  d9  gap        orig={v['d9GapOrig']}   k={k} damp={v['d9GapDamp']}")
        print(f"  d10 gap        orig={v['d10GapOrig']}   k={k} damp={v['d10GapDamp']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot-dir", default="asof_rates")
    ap.add_argument("--k", type=float, default=0.5)
    ap.add_argument("--out", default="backtest_k_report.json")
    args = ap.parse_args()

    _self_check_k1_matches_log5()

    if not os.path.isdir(args.snapshot_dir):
        raise SystemExit(f"No snapshot dir at {args.snapshot_dir} -- run "
                          f"asof_rate_cache.py --backfill first")

    rows = build_rows(args.snapshot_dir, args.k)
    if len(rows) < 200:
        print(f"WARN: only {len(rows)} rows -- this is a thin sample, treat "
              f"any result as provisional (same bar this codebase applies "
              f"everywhere else, e.g. calibrate.py's MIN_ROWS gate)", file=sys.stderr)

    report = {"k": args.k, "nRows": len(rows),
              "markets": [report_market(mkt, rows) for mkt in MARKETS]}

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print_summary(report, args.k)
    print(f"\nFull report (all deciles, both splits, both markets): {args.out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"BACKTEST FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
