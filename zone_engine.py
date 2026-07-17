#!/usr/bin/env python3
# ============================================================================
# zone_engine.py -- zone-matchup layer (DTP component 5).
#
# Maintains zones_cache.json: season-to-date, per-batter, per-Savant-zone
# power aggregates (BBE, barrels, HR, powerScore), updated INCREMENTALLY --
# each run pulls only the days since the cache's asOf date and folds them in,
# so the daily cost is one small Statcast pull instead of refetching a season.
#
# At build time, score_matchup() intersects a batter's STRONG zones with the
# opposing starter's MOST-USED recent locations:
#   strong zone : >= MIN_ZONE_BBE batted balls there AND
#                 powerScore = (barrels + 2*HR) / BBE >= STRONG_SCORE
#   used zone   : >= USED_SHARE of the pitcher's last-3-start pitches land there
#   overlap     : count of zones that are both -> GOOD_ZONES good,
#                 ELITE_ZONES elite.
#
# ZONE SCHEME: Savant zones 1-9 (the 3x3 in-zone grid). Chase zones 11-14 are
# excluded from batter strength -- almost nothing out of the zone is a power
# zone -- but still count in the pitcher's usage denominator, so a pitcher who
# lives off the plate correctly overlaps with nobody. DTP's tool uses a 7-zone
# grid and grades 3/7 good, 4/7 elite; on 9 zones the proportional thresholds
# are 4/9 and 5/9. Same fractions, different grid -- documented so nobody
# "fixes" it later.
#
# FIRST RUN: python zone_engine.py --backfill 2026-03-26   (season opener)
# fetches the whole season in weekly chunks -- run once via workflow_dispatch
# with a raised timeout, then daily runs are incremental. REFERENCE ONLY:
# never touches log5; MODEL_VERSION does not bump.
# ============================================================================

import datetime
import json
import os
import sys
import tempfile
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
CACHE_PATH = "zones_cache.json"

MIN_ZONE_BBE = 8
STRONG_SCORE = 0.15
USED_SHARE = 12.0        # % of recent pitches in a zone to call it "used"
GOOD_ZONES = 4           # of 9 -- proportional to DTP's 3/7
ELITE_ZONES = 5          # of 9 -- proportional to DTP's 4/7
ZONES = [str(z) for z in range(1, 10)]


def _f(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


# ------------------------------- cache update --------------------------------

def empty_cache(season_start):
    return {"asOf": season_start, "batters": {}}


def load_cache():
    try:
        with open(CACHE_PATH) as f:
            c = json.load(f)
        if isinstance(c, dict) and "batters" in c and "asOf" in c:
            return c
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return None


def save_cache(cache):
    d = os.path.dirname(os.path.abspath(CACHE_PATH)) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".zones_", suffix=".json")
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


def fold_batted_balls(cache, df):
    """Fold a pitch-level DataFrame's batted balls into the cache. Pure --
    testable with synthetic frames. Idempotency is by DATE WINDOW (the caller
    only feeds days after asOf), not by row."""
    need = ("batter", "type", "zone")
    if any(c not in df.columns for c in need):
        return 0
    bbe = df[df["type"] == "X"]
    if bbe.empty:
        return 0
    has_lsa = "launch_speed_angle" in bbe.columns
    has_ev = "events" in bbe.columns
    folded = 0
    for (pid, zone), g in bbe.groupby(["batter", "zone"]):
        z = _f(zone)
        if z is None or not (1 <= int(z) <= 9):
            continue
        zkey = str(int(z))
        b = cache["batters"].setdefault(str(int(pid)), {})
        cell = b.setdefault(zkey, {"bbe": 0, "barrels": 0, "hr": 0})
        cell["bbe"] += len(g)
        if has_lsa:
            cell["barrels"] += int((g["launch_speed_angle"] == 6).sum())
        if has_ev:
            cell["hr"] += int((g["events"].astype(str) == "home_run").sum())
        folded += len(g)
    return folded


def update_cache(backfill_from=None, chunk_days=7):
    """Incremental update: pull statcast from cache.asOf+1 to today and fold.
    With backfill_from, start a fresh cache from that date (season opener).
    Chunked weekly so a season backfill doesn't hold one giant request open."""
    from pybaseball import statcast
    today = datetime.datetime.now(ET).date()
    if backfill_from:
        cache = empty_cache(backfill_from)
        start = datetime.date.fromisoformat(backfill_from)
    else:
        cache = load_cache()
        if cache is None:
            print("No cache and no --backfill given: starting 45-day bootstrap "
                  "(thin but usable; run --backfill <season opener> for full-season zones)",
                  file=sys.stderr)
            start = today - datetime.timedelta(days=45)
            cache = empty_cache(start.isoformat())
        else:
            start = datetime.date.fromisoformat(cache["asOf"]) + datetime.timedelta(days=1)
    if start > today:
        print("Cache already current:", cache["asOf"])
        return cache
    total = 0
    cur = start
    while cur <= today:
        end = min(cur + datetime.timedelta(days=chunk_days - 1), today)
        try:
            df = statcast(start_dt=cur.isoformat(), end_dt=end.isoformat())
            if df is not None and len(df):
                total += fold_batted_balls(cache, df)
        except Exception as e:
            # A failed chunk stops the advance so asOf never claims coverage
            # it doesn't have; the next run resumes from the same point.
            print(f"WARN: statcast chunk {cur}..{end} failed ({type(e).__name__}) "
                  f"-- stopping at asOf={cache['asOf']}", file=sys.stderr)
            save_cache(cache)
            return cache
        cache["asOf"] = end.isoformat()
        cur = end + datetime.timedelta(days=1)
    save_cache(cache)
    print(f"zones_cache updated through {cache['asOf']}: folded {total} BBE, "
          f"{len(cache['batters'])} batters")
    return cache


# ------------------------------ matchup scoring ------------------------------

def strong_zones(cache, batter_id):
    """Zones where this batter does real damage."""
    b = (cache or {}).get("batters", {}).get(str(batter_id))
    if not b:
        return set()
    out = set()
    for zkey, cell in b.items():
        bbe = cell.get("bbe", 0)
        if bbe < MIN_ZONE_BBE:
            continue
        score = (cell.get("barrels", 0) + 2 * cell.get("hr", 0)) / bbe
        if score >= STRONG_SCORE:
            out.add(zkey)
    return out


def pitcher_used_zones(df, pitcher_id, game_pks):
    """Zones covering >= USED_SHARE% of the pitcher's pitches across the given
    recent games. Denominator = ALL pitches (chase zones included), so a
    stay-away pitcher's in-zone shares are honestly small."""
    if df is None or any(c not in df.columns for c in ("pitcher", "game_pk", "zone")):
        return set()
    g = df[(df["pitcher"] == pitcher_id) & (df["game_pk"].isin(game_pks))]
    total = len(g)
    if total < 30:
        return set()
    out = set()
    for zone, zg in g.groupby("zone"):
        z = _f(zone)
        if z is None or not (1 <= int(z) <= 9):
            continue
        if len(zg) / total * 100 >= USED_SHARE:
            out.add(str(int(z)))
    return out


def score_matchup(batter_strong, pitcher_used):
    """-> (overlapCount, grade) where grade is 'elite' | 'good' | None."""
    overlap = len(batter_strong & pitcher_used)
    if overlap >= ELITE_ZONES:
        return overlap, "elite"
    if overlap >= GOOD_ZONES:
        return overlap, "good"
    return overlap, None


if __name__ == "__main__":
    backfill = None
    if len(sys.argv) >= 3 and sys.argv[1] == "--backfill":
        backfill = sys.argv[2]
    try:
        update_cache(backfill_from=backfill)
    except Exception as e:
        print(f"ZONE UPDATE FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
