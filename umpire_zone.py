#!/usr/bin/env python3
# ============================================================================
# umpire_zone.py -- umpire zone-tendency cache, incremental, SAME PATTERN as
# zone_engine.py's zones_cache.json: season-to-date, per-umpire borderline-
# call tallies, updated incrementally so the daily cost is one small Statcast
# pull instead of refetching a season. Structure deliberately mirrors
# zone_engine.py line-for-line (empty_cache/load_cache/save_cache/update_cache
# shape, chunk_days=7, stop-on-failure chunking, atomic tempfile write) so
# this codebase has one incremental-cache pattern, not two that can drift.
#
# STILL NOT WIRED INTO build_daily_board.py OR compute_angles(). Three things
# must be checked against a REAL pull before this goes near the live board --
# unchanged from the standalone version, repeated here because they don't go
# away just because there's now a cache:
#
#   1. COLUMN RELIABILITY. `umpire` is a numeric MLBAM id with a documented
#      history of going null for stretches. umpire_coverage_cached() reads
#      this CUMULATIVELY across the whole cache (not one day's snapshot,
#      which is the point of a season-to-date cache) -- but it only helps if
#      you check it. If cumulative coverage is near zero after a real
#      backfill, this feature is dead regardless of code quality.
#
#   2. ID vs NAME JOIN. situational.home_plate_umpire() returns a NAME.
#      This cache is keyed by numeric MLBAM id. Confirm StatsAPI's officials
#      hydrate actually carries an "id" field before wiring the join --
#      see the standalone version's docstring for the exact check.
#
#   3. ABS CHALLENGES (2026). A human umpire's call can now be overturned by
#      automated review mid-game. If Savant's description/type fields
#      reflect the POST-CHALLENGE result, this cache would silently fold the
#      robot's correction into "the umpire's tendency." Unconfirmed --
#      check for a challenge/review column in a real pull first.
#
# FIRST RUN: python umpire_zone.py --backfill 2026-03-26   (season opener,
# same convention as zone_engine.py --backfill)
# ============================================================================

import datetime
import json
import os
import sys
import tempfile
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
CACHE_PATH = "umpire_cache.json"

# Rulebook plate half-width: 17 inches / 2, in feet.
PLATE_HALF_WIDTH_FT = (17.0 / 12.0) / 2.0   # 0.7083 ft

# Borderline-pitch margin -- see standalone-version docstring for the same
# DOCUMENTED APPROXIMATION honesty note zone_engine.py applies to its own
# heart/shadow boundary. 3 inches is the commonly-cited public-methodology
# band (Umpire Scorecards and similar).
BORDERLINE_MARGIN_FT = 3.0 / 12.0            # 0.25 ft

MIN_BORDERLINE = 40      # per-umpire floor before reporting a rate at all
MIN_COVERAGE = 0.60      # cumulative non-null 'umpire' rate floor
MIN_COVERAGE_SAMPLE = 200  # don't even judge coverage below this many called pitches


def _nv(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _col(df, name):
    return name if name in df.columns else None


def _is_called_ball_or_strike(description):
    d = str(description or "").strip().lower()
    return d in ("called_strike", "ball", "blocked_ball")


def _is_actual_strike(description):
    return str(description or "").strip().lower() == "called_strike"


def _classify_borderline(plate_x, plate_z, sz_top, sz_bot):
    """One pitch's location -> ("expand"|"contract") or None if not
    borderline. "expand" = outside the rulebook zone but close (a strike call
    here is zone EXPANSION); "contract" = inside but close (a ball call here
    is zone CONTRACTION). Pulled out as its own pure function so the
    one-shot path and the incremental-fold path can never quietly diverge."""
    x, z, top, bot = _nv(plate_x), _nv(plate_z), _nv(sz_top), _nv(sz_bot)
    if x is None or z is None or top is None or bot is None or top <= bot:
        return None
    outside_h = x < -PLATE_HALF_WIDTH_FT or x > PLATE_HALF_WIDTH_FT
    outside_v = z > top or z < bot
    in_zone = not outside_h and not outside_v
    if in_zone:
        edge_dist = min(PLATE_HALF_WIDTH_FT - abs(x), z - bot, top - z)
        if edge_dist > BORDERLINE_MARGIN_FT:
            return None
        return "contract"
    else:
        horiz_excess = abs(x) - PLATE_HALF_WIDTH_FT if outside_h else -1e9
        vert_excess = max(z - top, bot - z) if outside_v else -1e9
        excess = max(horiz_excess, vert_excess)
        if excess > BORDERLINE_MARGIN_FT:
            return None
        return "expand"


# ------------------------------- cache update --------------------------------
# Structure mirrors zone_engine.py's empty_cache/load_cache/save_cache exactly.

def empty_cache(season_start):
    return {"asOf": season_start, "totalCalledPitches": 0, "totalWithUmpireId": 0,
            "umpires": {}}


def load_cache():
    try:
        with open(CACHE_PATH) as f:
            c = json.load(f)
        if isinstance(c, dict) and "umpires" in c and "asOf" in c:
            c.setdefault("totalCalledPitches", 0)
            c.setdefault("totalWithUmpireId", 0)
            return c
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return None


def save_cache(cache):
    d = os.path.dirname(os.path.abspath(CACHE_PATH)) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".umpires_", suffix=".json")
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


def fold_umpire_calls(cache, df):
    """Fold a pitch-level DataFrame's called balls/strikes into the cache.
    Pure -- testable with synthetic frames, same discipline as
    zone_engine.fold_batted_balls. Idempotency is by DATE WINDOW (caller only
    feeds days after asOf), not by row -- also matching zone_engine.

    Tracks totalCalledPitches/totalWithUmpireId REGARDLESS of whether a row
    ends up borderline -- coverage has to be measured against every called
    pitch, not just the ones that happened to be close plays, or a real
    outage that only affects certain teams/games would be invisible to it.
    """
    need = ["plate_x", "plate_z", "sz_top", "sz_bot", "description"]
    ump_col = _col(df, "umpire")
    if ump_col is None or any(_col(df, c) is None for c in need):
        return 0
    mask = df["description"].apply(_is_called_ball_or_strike)
    called = df[mask]
    if called.empty:
        return 0

    cache["totalCalledPitches"] += int(len(called))
    cache["totalWithUmpireId"] += int(called[ump_col].notna().sum())

    folded = 0
    for row in called.itertuples(index=False):
        r = row._asdict()
        uid = _nv(r.get(ump_col))
        if uid is None:
            continue
        kind = _classify_borderline(r.get("plate_x"), r.get("plate_z"),
                                    r.get("sz_top"), r.get("sz_bot"))
        if kind is None:
            continue
        is_strike = _is_actual_strike(r.get("description"))
        t = cache["umpires"].setdefault(str(int(uid)),
                                        {"exp_n": 0, "exp_strikes": 0, "con_n": 0, "con_balls": 0})
        if kind == "contract":
            t["con_n"] += 1
            if not is_strike:
                t["con_balls"] += 1
        else:
            t["exp_n"] += 1
            if is_strike:
                t["exp_strikes"] += 1
        folded += 1
    return folded


def update_cache(backfill_from=None, chunk_days=7):
    """Incremental update -- SAME control flow as zone_engine.update_cache():
    pull statcast from cache.asOf+1 to today, chunked, folding each chunk and
    advancing asOf; a failed chunk STOPS the advance (never claims coverage it
    doesn't have) and saves what's confirmed so far; the next run resumes from
    the same point. With backfill_from, starts a fresh cache from that date."""
    from pybaseball import statcast
    today = datetime.datetime.now(ET).date()
    if backfill_from:
        cache = empty_cache(backfill_from)
        start = datetime.date.fromisoformat(backfill_from)
    else:
        cache = load_cache()
        if cache is None:
            print("No umpire cache and no --backfill given: starting 45-day bootstrap "
                  "(thin but usable; run --backfill <season opener> for full-season read)",
                  file=sys.stderr)
            start = today - datetime.timedelta(days=45)
            cache = empty_cache(start.isoformat())
        else:
            start = datetime.date.fromisoformat(cache["asOf"]) + datetime.timedelta(days=1)
    if start > today:
        print("Umpire cache already current:", cache["asOf"])
        return cache
    total = 0
    cur = start
    while cur <= today:
        end = min(cur + datetime.timedelta(days=chunk_days - 1), today)
        try:
            df = statcast(start_dt=cur.isoformat(), end_dt=end.isoformat())
            if df is not None and len(df):
                total += fold_umpire_calls(cache, df)
        except Exception as e:
            print(f"WARN: statcast chunk {cur}..{end} failed ({type(e).__name__}) "
                  f"-- stopping at asOf={cache['asOf']}", file=sys.stderr)
            save_cache(cache)
            return cache
        cache["asOf"] = end.isoformat()
        cur = end + datetime.timedelta(days=1)
    save_cache(cache)
    cov = umpire_coverage_cached(cache)
    print(f"umpire_cache updated through {cache['asOf']}: folded {total} called pitches, "
          f"{len(cache['umpires'])} umpires, cumulative coverage={cov}")
    return cache


# ------------------------------ reading the cache ------------------------------

def umpire_coverage_cached(cache):
    """Cumulative non-null umpire-id rate across the WHOLE cache to date --
    the point of a season-to-date cache over a one-day snapshot: a single bad
    day doesn't swing this, and a real sustained outage becomes visible fast.
    Returns None until there's enough called-pitch volume to judge it at all."""
    total = (cache or {}).get("totalCalledPitches", 0)
    if total < MIN_COVERAGE_SAMPLE:
        return None
    withid = (cache or {}).get("totalWithUmpireId", 0)
    return round(withid / total, 4)


def umpire_zone_score(cache, umpire_id, min_borderline=MIN_BORDERLINE):
    """Season-to-date read for one umpire. None if coverage is too broken to
    trust ANY umpire, or this specific umpire is under the per-umpire floor --
    same two-tier gate as the standalone version, now reading accumulated
    season data instead of a one-shot df."""
    cov = umpire_coverage_cached(cache)
    if cov is not None and cov < MIN_COVERAGE:
        return None
    t = (cache or {}).get("umpires", {}).get(str(umpire_id))
    if not t:
        return None
    total = t["exp_n"] + t["con_n"]
    if total < min_borderline:
        return None
    expansion = round(t["exp_strikes"] / t["exp_n"], 4) if t["exp_n"] else None
    contraction = round(t["con_balls"] / t["con_n"], 4) if t["con_n"] else None
    zone_score = (round(expansion - contraction, 4)
                  if expansion is not None and contraction is not None else None)
    return {"expansionRate": expansion, "expansionN": t["exp_n"],
            "contractionRate": contraction, "contractionN": t["con_n"],
            "zoneScore": zone_score, "totalBorderline": total}


if __name__ == "__main__":
    backfill = None
    if len(sys.argv) >= 3 and sys.argv[1] == "--backfill":
        backfill = sys.argv[2]
    try:
        update_cache(backfill_from=backfill)
    except Exception as e:
        print(f"UMPIRE CACHE UPDATE FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
