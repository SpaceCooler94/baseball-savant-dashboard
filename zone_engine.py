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
#   strong zone : >= MIN_ZONE_BBE batted balls there AND powerScore
#                 (barrels + 2*HR)/BBE above the batter's OWN zone average
#                 (see STRONG_RATIO), capped at his best MAX_STRONG cells
#   used zone   : >= USED_SHARE of the pitcher's last-3-start pitches land there
#   overlap     : count of zones that are both -> GOOD_ZONES good,
#                 ELITE_ZONES elite (bars set against the chance baseline --
#                 see the note on those constants).
#
# ZONE SCHEME: Savant zones 1-9 (the 3x3 in-zone grid). Chase zones 11-14 are
# excluded from batter strength -- almost nothing out of the zone is a power
# zone. DTP's tool uses a 7-zone grid and grades 3/7 good, 4/7 elite; this
# started as the proportional 4/9 and 5/9, but proportion turned out to be the
# wrong frame once strong zones became relative and capped. The bars are now set
# against how much overlap chance alone produces. Documented so nobody "restores"
# the fractions later.
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

# A zone is "strong" when it beats the batter's OWN zone-average, not an
# absolute bar. v1 used a flat powerScore >= .15, which good power hitters clear
# almost everywhere: on the 2026-08-01 board, 73 of 312 batters came back strong
# in 6+ of 9 zones, and those wide profiles produced 14 of the only 18 graded
# matchups. When a batter is "strong" everywhere the overlap count stops
# measuring a matchup and just re-reads the pitcher's usage -- Goodman and Alonso
# both graded ELITE while flagged in 8 of 9 zones.
#
# Now a zone must clear BOTH: the batter's own average zone powerScore times
# STRONG_RATIO (relative -- "this is where HE does damage"), and STRONG_FLOOR
# (absolute -- keeps a weak hitter's least-bad zone from qualifying). The result
# is capped at MAX_STRONG cells so the widest profile still has to choose.
STRONG_RATIO = 1.25      # x the batter's own mean zone powerScore
STRONG_FLOOR = 0.12      # absolute floor, so "strong" still means something
MAX_STRONG = 4           # keep only his best cells -- a damage zone is a place
# A zone counts as "used" when it takes ABOVE-AVERAGE share of the pitcher's
# IN-ZONE pitches. v1 measured share of ALL pitches (chase zones included in the
# denominator) against a flat 12% bar -- but with 9 in-zone plus 4 chase buckets,
# uniform is ~7.7%, so 12% demanded unusual concentration and almost no starter
# cleared it. Measured on the 2026-08-01 board: 2 of 29 probables had ANY used
# zone, so zoneGrade was null on all 375 rows and the whole layer was dead.
# Uniform across 9 in-zone buckets is 11.1%; the bar below is just above it, so
# "used" now means what it says -- this is where he actually lives.
USED_SHARE = 12.0        # % of the pitcher's IN-ZONE pitches
# Grade bars are set against the CHANCE baseline, not a fraction of the grid.
# Once strong zones are capped at MAX_STRONG, a batter carries ~2-3 damage cells
# and a starter lives in ~4 of 9, so random overlap averages ~0.9. The old 4/9
# and 5/9 bars were derived from DTP's 3/7 and 4/7 on a nine-cell grid -- they
# made sense when a batter could be "strong" in 9 zones, and are unreachable now
# (5 is impossible when the cap is 4). These bars are ~2x and ~3x chance, which
# on a real slate grades roughly 20% good and 3% elite.
GOOD_ZONES = 2           # ~2x the chance baseline
ELITE_ZONES = 3          # ~3x the chance baseline
ZONES = [str(z) for z in range(1, 10)]

# ---------------------------- attack-zone boundaries -------------------------
# Heart / Shadow / Chase / Waste, in the SAME cache, folded from the SAME rows.
# Different partition of the same plate than the 1-9 grid above: those are
# fixed thirds of the rulebook zone; these are Savant's own "how central was
# this pitch" regions, batter-relative via sz_top/sz_bot per row.
#
# HONESTY NOTE, same discipline as the pull% spray-angle formula in
# recent_form.py: MLB has never published Heart's exact numeric boundary.
# FanGraphs (Sarris, "Life Is Easier When You Hit Your Spots") documents
# Shadow precisely -- it straddles the rulebook zone edge by 3.3in on the
# sides and 4in top/bottom -- but Heart itself is only described qualitatively
# ("much bigger than the simple 9-zone center cell," FanGraphs "Looking into
# the Heart Zone"). Rather than invent an unrelated number, HEART_IN below
# mirrors Shadow's margin INWARD from the same rulebook edge -- a documented,
# symmetric, defensible construction, not a claim of byte-parity with
# Savant's proprietary boundary. If MLB ever publishes the exact figure,
# update HEART_IN alone; every zone here reads from these four constants.
PLATE_HALF_WIDTH_FT = 17 / 2 / 12          # rulebook: 17in plate -> 8.5in half-width
SHADOW_OUT_SIDE_FT = 3.3 / 12              # FanGraphs-sourced, see note above
SHADOW_OUT_VERT_FT = 4.0 / 12
HEART_IN_SIDE_FT = SHADOW_OUT_SIDE_FT      # approximation -- see note above
HEART_IN_VERT_FT = SHADOW_OUT_VERT_FT
CHASE_OUT_MULT = 2.0                       # FanGraphs: chase's outer box is
                                            # ~2x the rulebook zone each way
MIN_ATTACK_BBE = 10   # heart-zone contact is a smaller slice of a batter's
                       # total BBE than a 1-9 grid cell; slightly higher floor
                       # than MIN_ZONE_BBE so "n/a" beats a noisy small number


def _f(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def classify_attack_zone(plate_x, plate_z, sz_top, sz_bot):
    """One pitch's (plate_x, plate_z, sz_top, sz_bot) -> 'heart'|'shadow'|
    'chase'|'waste', or None if any coordinate is missing. Boundaries are
    batter-relative (sz_top/sz_bot come from that specific row, exactly like
    Savant's own per-batter zone) using the constants documented above."""
    x, z, top, bot = _f(plate_x), _f(plate_z), _f(sz_top), _f(sz_bot)
    if x is None or z is None or top is None or bot is None or top <= bot:
        return None
    height = top - bot

    heart_x = PLATE_HALF_WIDTH_FT - HEART_IN_SIDE_FT
    heart_top = top - HEART_IN_VERT_FT
    heart_bot = bot + HEART_IN_VERT_FT
    if heart_top > heart_bot and abs(x) <= heart_x and heart_bot <= z <= heart_top:
        return "heart"

    shadow_x = PLATE_HALF_WIDTH_FT + SHADOW_OUT_SIDE_FT
    shadow_top = top + SHADOW_OUT_VERT_FT
    shadow_bot = bot - SHADOW_OUT_VERT_FT
    if abs(x) <= shadow_x and shadow_bot <= z <= shadow_top:
        return "shadow"

    chase_x = PLATE_HALF_WIDTH_FT * CHASE_OUT_MULT
    chase_top = top + height * (CHASE_OUT_MULT - 1)
    chase_bot = bot - height * (CHASE_OUT_MULT - 1)
    if abs(x) <= chase_x and chase_bot <= z <= chase_top:
        return "chase"

    return "waste"


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
    only feeds days after asOf), not by row.

    Folds TWO partitions of the same batted balls in one pass: the existing
    1-9 grid cells (bbe/barrels/hr, unchanged), and now also xSLG into both
    the grid cell AND the attack zone (heart/shadow/chase/waste) that pitch
    fell in. xSLG accumulates as a running (sum, n) pair rather than an
    average, so folding stays purely additive across incremental daily runs --
    the same reason bbe/barrels/hr were already stored as raw counts, not
    rates. A row lacking plate_x/plate_z/sz_top/sz_bot or an xSLG value still
    folds its bbe/barrels/hr normally; only the xSLG and attack-zone pieces
    for that row are skipped.

    RETROFIT NOTE: batted balls folded before this xSLG/attack-zone addition
    shipped have no xslgSum/xslgN/attack data -- those fields start at zero and
    grow from here forward. A fresh --backfill run repopulates them from the
    season; the daily incremental job alone will just take longer to reach a
    useful sample than bbe/barrels/hr did originally.
    """
    need = ("batter", "type", "zone")
    if any(c not in df.columns for c in need):
        return 0
    bbe = df[df["type"] == "X"]
    if bbe.empty:
        return 0
    has_lsa = "launch_speed_angle" in bbe.columns
    has_ev = "events" in bbe.columns
    has_xslg = "estimated_slg_using_speedangle" in bbe.columns
    has_attack = all(c in bbe.columns for c in ("plate_x", "plate_z", "sz_top", "sz_bot"))
    folded = 0
    for (pid, zone), g in bbe.groupby(["batter", "zone"]):
        z = _f(zone)
        if z is None or not (1 <= int(z) <= 9):
            continue
        zkey = str(int(z))
        b = cache["batters"].setdefault(str(int(pid)), {})
        cell = b.setdefault(zkey, {"bbe": 0, "barrels": 0, "hr": 0, "xslgSum": 0.0, "xslgN": 0})
        cell.setdefault("xslgSum", 0.0)
        cell.setdefault("xslgN", 0)
        cell["bbe"] += len(g)
        if has_lsa:
            cell["barrels"] += int((g["launch_speed_angle"] == 6).sum())
        if has_ev:
            cell["hr"] += int((g["events"].astype(str) == "home_run").sum())
        if has_xslg:
            xs = g["estimated_slg_using_speedangle"].dropna()
            cell["xslgSum"] += float(xs.sum())
            cell["xslgN"] += int(len(xs))
        folded += len(g)

    if has_attack:
        attack_cache = cache.setdefault("attackBatters", {})
        # Row-wise classification -- there's no vectorized shortcut here since
        # each row can have a different batter-specific sz_top/sz_bot, exactly
        # like the per-batter zone the rest of this cache already respects.
        for row in bbe.itertuples(index=False):
            r = row._asdict()
            zone_name = classify_attack_zone(r.get("plate_x"), r.get("plate_z"),
                                             r.get("sz_top"), r.get("sz_bot"))
            if zone_name is None:
                continue
            pid = r.get("batter")
            if pid is None:
                continue
            ab = attack_cache.setdefault(str(int(pid)), {})
            cell = ab.setdefault(zone_name, {"bbe": 0, "barrels": 0, "hr": 0,
                                             "xslgSum": 0.0, "xslgN": 0})
            cell["bbe"] += 1
            if has_lsa and _f(r.get("launch_speed_angle")) == 6:
                cell["barrels"] += 1
            if has_ev and str(r.get("events")) == "home_run":
                cell["hr"] += 1
            if has_xslg:
                xv = _f(r.get("estimated_slg_using_speedangle"))
                if xv is not None:
                    cell["xslgSum"] += xv
                    cell["xslgN"] += 1
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

def zone_scores(cache, batter_id):
    """{zone: powerScore} for every zone with enough batted balls to read.
    powerScore = (barrels + 2*HR) / BBE."""
    b = (cache or {}).get("batters", {}).get(str(batter_id))
    if not b:
        return {}
    out = {}
    for zkey, cell in b.items():
        bbe = cell.get("bbe", 0)
        if bbe < MIN_ZONE_BBE:
            continue
        out[zkey] = (cell.get("barrels", 0) + 2 * cell.get("hr", 0)) / bbe
    return out


def strong_zones(cache, batter_id):
    """Zones where this batter does damage RELATIVE TO HIMSELF -- see the
    STRONG_RATIO note above. Returns at most MAX_STRONG zones, his best."""
    scores = zone_scores(cache, batter_id)
    if not scores:
        return set()
    mean = sum(scores.values()) / len(scores)
    bar = max(STRONG_FLOOR, mean * STRONG_RATIO)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return {z for z, s in ranked[:MAX_STRONG] if s >= bar}


def heart_zone_xslg(cache, batter_id):
    """{"xSLG": float|None, "bbe": int} for this batter's HEART-zone contact --
    what he does specifically to pitches down the middle, as distinct from
    strong_zones()'s barrel/HR read on the 1-9 grid. Below MIN_ATTACK_BBE
    returns xSLG=None with the bbe count still shown, so a thin sample reads
    as 'not enough data' rather than a number that happens to be unstable."""
    ab = (cache or {}).get("attackBatters", {}).get(str(batter_id), {})
    cell = ab.get("heart")
    if not cell:
        return {"xSLG": None, "bbe": 0}
    bbe = cell.get("bbe", 0)
    n = cell.get("xslgN", 0)
    if bbe < MIN_ATTACK_BBE or n <= 0:
        return {"xSLG": None, "bbe": bbe}
    return {"xSLG": round(cell["xslgSum"] / n, 3), "bbe": bbe}


def pitcher_used_zones(df, pitcher_id, game_pks):
    """Zones covering >= USED_SHARE% of the pitcher's pitches across the given
    recent games. Denominator = ALL pitches (chase zones included), so a
    stay-away pitcher's in-zone shares are honestly small."""
    if df is None or any(c not in df.columns for c in ("pitcher", "game_pk", "zone")):
        return set()
    g = df[(df["pitcher"] == pitcher_id) & (df["game_pk"].isin(game_pks))]
    if len(g) < 30:
        return set()
    # Denominator is in-zone pitches only -- see USED_SHARE note above.
    in_zone = {}
    for zone, zg in g.groupby("zone"):
        z = _f(zone)
        if z is None or not (1 <= int(z) <= 9):
            continue
        in_zone[str(int(z))] = len(zg)
    total_in = sum(in_zone.values())
    if total_in < 20:
        return set()
    return {z for z, n in in_zone.items() if n / total_in * 100 >= USED_SHARE}


def zone_shares(df, pitcher_id, game_pks):
    """Full 9-cell in-zone distribution for the zone diagram: {zone: pct}.
    Display data -- the strike-zone drawing needs every cell, not just the hot
    ones."""
    if df is None or any(c not in df.columns for c in ("pitcher", "game_pk", "zone")):
        return {}
    g = df[(df["pitcher"] == pitcher_id) & (df["game_pk"].isin(game_pks))]
    if len(g) < 30:
        return {}
    counts = {}
    for zone, zg in g.groupby("zone"):
        z = _f(zone)
        if z is None or not (1 <= int(z) <= 9):
            continue
        counts[str(int(z))] = len(zg)
    total = sum(counts.values())
    if total < 20:
        return {}
    return {z: round(n / total * 100, 1) for z, n in counts.items()}


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
