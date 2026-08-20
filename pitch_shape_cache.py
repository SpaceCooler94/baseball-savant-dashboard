#!/usr/bin/env python3
# ============================================================================
# pitch_shape_cache.py -- season-long, per-pitcher per-pitch-type velocity/
# spin/movement baseline, incremental, SAME PATTERN as zone_engine.py and
# umpire_zone.py: season-to-date, updated incrementally so the daily cost is
# one small Statcast pull instead of refetching a season. Structure
# deliberately mirrors those two (empty_cache/load_cache/save_cache/
# update_cache shape, chunk_days=7, stop-on-failure chunking, atomic tempfile
# write) so this codebase has one incremental-cache pattern, not three that
# can drift.
#
# WHY THIS EXISTS: confirmed via a live Savant column-availability check that
# the Pitch Arsenal / Run Value leaderboard build_daily_board.py's
# fetch_pitcher_pitch_mix() calls has NO velocity/spin columns -- only usage/
# outcome stats. recent_form.py's pitch_shape_drift() needs a season BASELINE
# per pitch (avgVelo/avgSpin) to compare pitcher_recent()'s last-N-starts
# window against; that baseline was never real -- enrich_probable() was
# building it from the same arsenal-stats pull that doesn't carry shape data,
# so pitch_decay could never have fired through that path. This cache fixes
# that at the source: same raw Per-Pitch Statcast Search columns
# recent_form.py already pulls for the recent window (release_speed,
# release_spin_rate, release_extension, pfx_x, pfx_z -- confirmed column
# names, already tested in pitcher_recent()), aggregated over the whole
# season instead of the last few starts.
#
# FIRST RUN: python pitch_shape_cache.py --backfill 2026-03-26   (season
# opener, same convention as zone_engine.py / umpire_zone.py --backfill)
# ============================================================================

import datetime
import json
import os
import sys
import tempfile
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
CACHE_PATH = "pitch_shape_cache.json"

# Stable-baseline floor. Deliberately much higher than recent_form.py's
# SHAPE_MIN_N=8 for the last-N-starts window -- that window is meant to
# catch decay AS IT HAPPENS on thin recent data; this cache is the season
# ANCHOR that decay gets measured against, so it needs to actually be
# stable. 50 pitches of a given type is a few starts' worth for a core
# pitch, thin but real for a show-me pitch -- entries below it are omitted
# from the baseline rather than reported on noise.
SEASON_SHAPE_MIN_N = 50

_SHAPE_COLS = {
    "velo": "release_speed",
    "spin": "release_spin_rate",
    "ext": "release_extension",
    "hbreak": "pfx_x",
    "vbreak": "pfx_z",
}


def _col(df, name):
    return name if name in df.columns else None


def _iv(v):
    try:
        f = float(v)
        if f != f:
            return None
        return int(f)
    except (TypeError, ValueError):
        return None


# ------------------------------- cache update --------------------------------
# Structure mirrors zone_engine.py / umpire_zone.py's empty_cache/load_cache/
# save_cache exactly.

def empty_cache(season_start):
    return {"asOf": season_start, "pitchers": {}}


def load_cache():
    try:
        with open(CACHE_PATH) as f:
            c = json.load(f)
        if isinstance(c, dict) and "pitchers" in c and "asOf" in c:
            return c
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return None


def save_cache(cache):
    d = os.path.dirname(os.path.abspath(CACHE_PATH)) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".pitchshape_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(cache, f)
        os.replace(tmp, CACHE_PATH)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def fold_pitch_shape(cache, df):
    """Fold a pitch-level DataFrame into the cache. Pure -- testable with
    synthetic frames, same discipline as zone_engine.fold_batted_balls /
    umpire_zone.fold_umpire_calls. Idempotency is by DATE WINDOW (caller
    only feeds days after asOf), not by row -- also matching both.

    Stores accumulated SUMS + per-metric counts, not running means, so
    folding is always a plain add and never needs to un-average anything --
    same reason umpire_zone stores raw tallies (exp_n/exp_strikes/...)
    instead of rates. Per-metric counts (not one shared count) because a
    pitch missing release_extension shouldn't also throw out its velocity --
    real Statcast rows don't always null out every shape column together.
    """
    need = ["pitcher", "pitch_name"]
    if any(_col(df, c) is None for c in need):
        return 0
    present = {k: _col(df, v) for k, v in _SHAPE_COLS.items()}
    if not any(present.values()):
        return 0

    folded = 0
    for (pid_raw, pitch_raw), pg in df.groupby(["pitcher", "pitch_name"]):
        pid = _iv(pid_raw)
        pitch_display = str(pitch_raw).strip()
        if pid is None or not pitch_display or pitch_display == "nan":
            continue
        key = pitch_display.lower()
        pt = cache["pitchers"].setdefault(str(pid), {})
        entry = pt.setdefault(key, {
            "pitchDisplay": pitch_display, "n": 0,
            "sumVelo": 0.0, "nVelo": 0, "sumSpin": 0.0, "nSpin": 0,
            "sumExt": 0.0, "nExt": 0, "sumH": 0.0, "nH": 0,
            "sumV": 0.0, "nV": 0,
        })
        entry["n"] += int(len(pg))
        for metric, col in present.items():
            if col is None:
                continue
            vals = pg[col].dropna()
            if vals.empty:
                continue
            n_key, sum_key = {
                "velo": ("nVelo", "sumVelo"), "spin": ("nSpin", "sumSpin"),
                "ext": ("nExt", "sumExt"), "hbreak": ("nH", "sumH"),
                "vbreak": ("nV", "sumV"),
            }[metric]
            entry[sum_key] += float(vals.sum())
            entry[n_key] += int(len(vals))
        folded += int(len(pg))
    return folded


def update_cache(backfill_from=None, chunk_days=7):
    """Incremental update -- SAME control flow as zone_engine.update_cache()
    / umpire_zone.update_cache(): pull statcast from cache.asOf+1 to today,
    chunked, folding each chunk and advancing asOf; a failed chunk STOPS the
    advance (never claims coverage it doesn't have) and saves what's
    confirmed so far; the next run resumes from the same point. With
    backfill_from, starts a fresh cache from that date."""
    from pybaseball import statcast
    today = datetime.datetime.now(ET).date()
    if backfill_from:
        cache = empty_cache(backfill_from)
        start = datetime.date.fromisoformat(backfill_from)
    else:
        cache = load_cache()
        if cache is None:
            print("No pitch-shape cache and no --backfill given: starting 45-day "
                  "bootstrap (thin but usable; run --backfill <season opener> for "
                  "full-season read)", file=sys.stderr)
            start = today - datetime.timedelta(days=45)
            cache = empty_cache(start.isoformat())
        else:
            start = datetime.date.fromisoformat(cache["asOf"]) + datetime.timedelta(days=1)
    if start > today:
        print("Pitch-shape cache already current:", cache["asOf"])
        return cache
    total = 0
    cur = start
    while cur <= today:
        end = min(cur + datetime.timedelta(days=chunk_days - 1), today)
        try:
            df = statcast(start_dt=cur.isoformat(), end_dt=end.isoformat())
            if df is not None and len(df):
                total += fold_pitch_shape(cache, df)
        except Exception as e:
            print(f"WARN: statcast chunk {cur}..{end} failed ({type(e).__name__}) "
                  f"-- stopping at asOf={cache['asOf']}", file=sys.stderr)
            save_cache(cache)
            return cache
        cache["asOf"] = end.isoformat()
        cur = end + datetime.timedelta(days=1)
    save_cache(cache)
    print(f"pitch_shape_cache updated through {cache['asOf']}: folded {total} pitches, "
          f"{len(cache['pitchers'])} pitchers", file=sys.stderr)
    return cache


# ------------------------------ reading the cache ------------------------------

def pitcher_shape_baseline(cache, pitcher_id, min_n=SEASON_SHAPE_MIN_N):
    """Season-to-date read for one pitcher -> {pitchDisplayName: {"avgVelo",
    "avgSpin", "avgExtension", "avgHBreak", "avgVBreak", "n"}}, exactly the
    shape recent_form.pitch_shape_drift() expects for its season_shape
    argument (keys matched case-insensitively there already). Gated on the
    SAME philosophy as recent_form.py's own SHAPE_MIN_N check -- one entry,
    one gate on total pitches thrown, not a silently-partial per-metric mix
    that could quietly compare velo from 200 pitches against spin from 3."""
    pt = (cache or {}).get("pitchers", {}).get(str(pitcher_id))
    if not pt:
        return {}
    out = {}
    for key, e in pt.items():
        if e.get("n", 0) < min_n:
            continue
        row = {"n": e["n"]}
        if e.get("nVelo"):
            row["avgVelo"] = round(e["sumVelo"] / e["nVelo"], 1)
        if e.get("nSpin"):
            row["avgSpin"] = round(e["sumSpin"] / e["nSpin"], 0)
        if e.get("nExt"):
            row["avgExtension"] = round(e["sumExt"] / e["nExt"], 2)
        if e.get("nH"):
            row["avgHBreak"] = round(e["sumH"] / e["nH"], 2)
        if e.get("nV"):
            row["avgVBreak"] = round(e["sumV"] / e["nV"], 2)
        out[e.get("pitchDisplay", key)] = row
    return out


if __name__ == "__main__":
    backfill = None
    if len(sys.argv) >= 3 and sys.argv[1] == "--backfill":
        backfill = sys.argv[2]
    try:
        update_cache(backfill_from=backfill)
    except Exception as e:
        print(f"PITCH SHAPE CACHE UPDATE FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
