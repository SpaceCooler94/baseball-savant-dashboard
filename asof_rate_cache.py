#!/usr/bin/env python3
# ============================================================================
# asof_rate_cache.py -- walk-forward, day-by-day reconstruction of what every
# batter's OBP/BB%/HR-rate and every pitcher's rate-allowed actually looked
# like BEFORE each day's games, for the backtest scoped in this session
# (batter/pitcher hit-rate + HR-rate only -- platoon splits and weather
# deliberately excluded, same discipline as everywhere else in this repo:
# don't build what isn't being measured yet).
#
# SAME INCREMENTAL-CACHE SHAPE as zone_engine.py / pitch_shape_cache.py /
# umpire_zone.py: a running cache (asof_rate_cache.json) with an asOf cursor,
# advanced one day at a time, atomic writes, stop-on-failure so a bad run
# never claims coverage it doesn't have. What's different from those three:
# this needs a POINT-IN-TIME snapshot for every date walked, not just the
# current state -- a backtest scoring July 3rd needs the rates as they stood
# on July 3rd, not today's. So each day this also writes one file to
# asof_rates/{date}.json (same archived-artifact idea as boards/{date}.json
# in settle.py) holding the league baseline + every batter/pitcher who
# actually played that day, in the EXACT field shape mlb_model.league_rates/
# batter_hit_rate_per_pa/batter_hr_rate_per_pa/_pitcher_rate already expect --
# a backtest can feed a loaded snapshot straight into those functions with no
# translation layer to drift out of sync.
#
# DATA SOURCE: MLB Stats API box scores (statsapi.mlb.com), same endpoint
# settle.py already uses for outcomes -- game/{gamePk}/boxscore. Real field
# names confirmed against a live 2026 box score before writing any of this
# (batting: atBats/hits/homeRuns/baseOnBalls/hitByPitch/sacFlies/
# plateAppearances; pitching: same shape but hitBatsmen instead of
# hitByPitch, plus battersFaced). Counting stats are accumulated as running
# SUMS (same reason pitch_shape_cache stores sums not means: folding is
# always a plain add, never un-averaging), and OBP/AVG/rate-allowed are
# derived with the exact formulas mlb_model.py and build_daily_board.py's
# get_pitcher_rates() already use (OBP = (H+BB+HBP)/(AB+BB+HBP+SF), pitcher
# hit-rate-allowed = max(0.01, obp_against - bb/bf)) -- one formula, reused,
# not a second copy that can quietly disagree.
#
# ATOMICITY (the one real bug risk here, fixed before it could ship): a day
# with N games where boxscore fetch #k of N fails partway through must NOT
# fold games 1..k-1 into the running cache and then stop -- a retry would
# refetch and re-fold the WHOLE date, double-counting those k-1 games. So a
# day's boxscores are all fetched into a local list FIRST; folding into the
# cache only happens once every game for that date is confirmed in hand. A
# failed day leaves the cache completely untouched and asOf unmoved -- the
# next run retries that date cleanly from zero, matching the stop-on-failure
# discipline zone_engine/pitch_shape_cache/umpire_zone.py already established
# (never claim partial coverage).
#
# DOUBLEHEADER SIMPLIFICATION (documented, not hidden): both games of a
# doubleheader are snapshotted against the SAME pre-day state (before either
# game), not game-2-sees-game-1's-result. Matches the day-granularity the
# backtest itself was scoped at -- intraday sequencing was never part of
# what was asked for.
#
# FIRST RUN: python asof_rate_cache.py --backfill 2026-03-26 --end 2026-07-18
# (season opener through the live ledger's current start, per this session's
# scope). Daily top-up afterward: python asof_rate_cache.py (no flags, resumes
# from cache asOf+1 through today).
# ============================================================================

import datetime
import json
import os
import random
import sys
import tempfile
import time
from zoneinfo import ZoneInfo

import requests

from mlb_model import league_rates  # single source of truth -- do not re-derive

ET = ZoneInfo("America/New_York")
STATS_API = "https://statsapi.mlb.com/api/v1"
CACHE_PATH = "asof_rate_cache.json"
SNAPSHOT_DIR = "asof_rates"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "mlb-daily-board-asof-backtest/1.0 (personal analytics pipeline)"})


def http_json(url, tries=3, timeout=20):
    last = None
    for attempt in range(tries):
        try:
            r = SESSION.get(url, timeout=timeout)
            if r.status_code == 429:
                time.sleep(min(float(r.headers.get("Retry-After", 5)), 30))
                last = requests.HTTPError("429")
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            if attempt < tries - 1:
                time.sleep((2 ** attempt) + random.uniform(0, 0.5))
    raise last


def nv(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


# ------------------------------- cache state --------------------------------
# Structure mirrors zone_engine.py / pitch_shape_cache.py / umpire_zone.py's
# empty_cache/load_cache/save_cache exactly.

def empty_cache(season_start):
    return {"asOf": season_start, "seasonStart": season_start, "batters": {}, "pitchers": {}}


def load_cache():
    try:
        with open(CACHE_PATH) as f:
            c = json.load(f)
        if isinstance(c, dict) and "batters" in c and "pitchers" in c and "asOf" in c:
            return c
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return None


def _atomic_write(path, obj):
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".asof_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def save_cache(cache):
    _atomic_write(CACHE_PATH, cache)


def save_snapshot(snapshot):
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    _atomic_write(os.path.join(SNAPSHOT_DIR, f"{snapshot['date']}.json"), snapshot)


def load_snapshot(date_str):
    """Reader side for the backtest: load_snapshot('2026-07-03') ->
    {"date","league","batters":{id:{...}},"pitchers":{id:{...}}}, or None if
    that date was never processed (off day, or beyond what's been backfilled).
    league is already mlb_model.league_rates()'s exact output shape;
    batters[id]/pitchers[id] are already the exact h/p dict shape
    project_hit()/project_hr() expect -- feed them straight in."""
    path = os.path.join(SNAPSHOT_DIR, f"{date_str}.json")
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# ------------------------------ pure transforms ------------------------------
# Kept side-effect-free so they're unit-testable without a network, same
# discipline as settle.py's parse_boxscore/settle_rows.

def batter_snapshot(entry):
    """Running counts -> the exact fields batter_hit_rate_per_pa/
    batter_hr_rate_per_pa/league_rates read (pa/obp/avg/bbPct/hrRate). None
    entry or 0 PA -> None, same as a rookie's first game: caller (mlb_model's
    shrink()) already falls back cleanly to the league rate on None."""
    if not entry or entry.get("pa", 0) <= 0:
        return None
    pa, ab = entry["pa"], entry["ab"]
    denom = ab + entry["bb"] + entry["hbp"] + entry["sf"]
    obp = (entry["h"] + entry["bb"] + entry["hbp"]) / denom if denom > 0 else None
    avg = (entry["h"] / ab) if ab > 0 else None
    return {
        "pa": pa, "obp": obp, "avg": avg,
        "bbPct": entry["bb"] / pa * 100,
        "hrRate": entry["hr"] / pa * 100,
        "hr": entry["hr"],
    }


def pitcher_snapshot(entry):
    """Same shape as build_daily_board.get_pitcher_rates()'s live-day
    result: hitRateAllowedPerPA = max(0.01, obp_against - bb/bf), the exact
    formula already in production -- reused, not re-derived."""
    if not entry or entry.get("bf", 0) <= 0:
        return None
    bf = entry["bf"]
    denom = entry["ab"] + entry["bb"] + entry["hbp"] + entry["sf"]
    obp_against = (entry["h"] + entry["bb"] + entry["hbp"]) / denom if denom > 0 else None
    bb_frac = entry["bb"] / bf
    hit_rate_allowed = max(0.01, obp_against - bb_frac) if obp_against is not None else None
    return {
        "battersFaced": bf,
        "hitRateAllowedPerPA": hit_rate_allowed,
        "hrRateAllowedPerPA": entry["hr"] / bf,
    }


def fold_boxscore(cache, box, seen_batters, seen_pitchers):
    """One game's boxscore -> (1) records each participant's PRE-GAME
    snapshot into seen_batters/seen_pitchers using the CURRENT cache state,
    the first time they're seen today (a doubleheader's 2nd game must not
    re-snapshot off the 1st game's now-updated totals -- see module docstring),
    then (2) folds this game's counts into the running cache. Mutates cache,
    seen_batters, seen_pitchers in place; returns nothing. Caller (process_day)
    only invokes this after ALL of a date's boxscores are already confirmed
    fetched -- see the atomicity note in the module docstring."""
    for side in ("home", "away"):
        team_block = box.get("teams", {}).get(side) or {}
        tabbr = (team_block.get("team") or {}).get("abbreviation")
        for _, p in (team_block.get("players") or {}).items():
            person = p.get("person") or {}
            pid_raw = person.get("id")
            if pid_raw is None:
                continue
            pid = str(pid_raw)
            name = person.get("fullName")
            stats = p.get("stats") or {}

            bat = stats.get("batting") or {}
            pa = nv(bat.get("plateAppearances"))
            if pa:
                if pid not in seen_batters:
                    snap = batter_snapshot(cache["batters"].get(pid))
                    seen_batters[pid] = {"id": pid_raw, "name": name, "teamAbbr": tabbr,
                                          **(snap or {})}
                e = cache["batters"].setdefault(pid, {"name": name, "teamAbbr": tabbr,
                                                       "pa": 0, "ab": 0, "h": 0, "bb": 0,
                                                       "hbp": 0, "sf": 0, "hr": 0})
                e["name"], e["teamAbbr"] = name, tabbr
                e["pa"] += int(pa)
                e["ab"] += int(nv(bat.get("atBats")) or 0)
                e["h"] += int(nv(bat.get("hits")) or 0)
                e["bb"] += int(nv(bat.get("baseOnBalls")) or 0)
                e["hbp"] += int(nv(bat.get("hitByPitch")) or 0)
                e["sf"] += int(nv(bat.get("sacFlies")) or 0)
                e["hr"] += int(nv(bat.get("homeRuns")) or 0)

            pit = stats.get("pitching") or {}
            bf = nv(pit.get("battersFaced"))
            if bf:
                if pid not in seen_pitchers:
                    snap = pitcher_snapshot(cache["pitchers"].get(pid))
                    seen_pitchers[pid] = {"id": pid_raw, "name": name, "teamAbbr": tabbr,
                                           **(snap or {})}
                e = cache["pitchers"].setdefault(pid, {"name": name, "teamAbbr": tabbr,
                                                        "bf": 0, "ab": 0, "h": 0, "bb": 0,
                                                        "hbp": 0, "sf": 0, "hr": 0})
                e["name"], e["teamAbbr"] = name, tabbr
                e["bf"] += int(bf)
                e["ab"] += int(nv(pit.get("atBats")) or 0)
                e["h"] += int(nv(pit.get("hits")) or 0)
                e["bb"] += int(nv(pit.get("baseOnBalls")) or 0)
                e["hbp"] += int(nv(pit.get("hitBatsmen")) or 0)
                e["sf"] += int(nv(pit.get("sacFlies")) or 0)
                e["hr"] += int(nv(pit.get("homeRuns")) or 0)


# --------------------------------- daily walk ---------------------------------

def _is_final(g):
    return ((g.get("status") or {}).get("abstractGameState")) == "Final"


def get_final_game_pks(date_str):
    data = http_json(f"{STATS_API}/schedule?sportId=1&date={date_str}")
    games = []
    for d in data.get("dates", []):
        games.extend(d.get("games", []))
    return [g["gamePk"] for g in games if _is_final(g) and g.get("gamePk")]


def process_day(cache, date_str):
    """Returns (snapshot_or_None, ok). ok=False means nothing was mutated --
    caller must stop and retry this date whole on the next run (see the
    atomicity note in the module docstring: boxscores are all fetched BEFORE
    any fold happens)."""
    game_pks = get_final_game_pks(date_str)
    if not game_pks:
        return {"date": date_str, "league": league_rates([]), "batters": {}, "pitchers": {}}, True

    boxes = []
    for pk in game_pks:
        try:
            boxes.append(http_json(f"{STATS_API}/game/{pk}/boxscore"))
        except Exception as e:
            print(f"WARN: boxscore fetch failed date={date_str} gamePk={pk} "
                  f"({type(e).__name__}) -- day incomplete, retrying whole date next run",
                  file=sys.stderr)
            return None, False

    seen_batters, seen_pitchers = {}, {}
    for box in boxes:
        fold_boxscore(cache, box, seen_batters, seen_pitchers)
    league = league_rates(list(seen_batters.values()))
    return {"date": date_str, "league": league, "batters": seen_batters,
            "pitchers": seen_pitchers}, True


def update_cache(backfill_from=None, end_date=None, save_every=3):
    today = datetime.datetime.now(ET).date()
    if backfill_from:
        cache = empty_cache(backfill_from)
        start = datetime.date.fromisoformat(backfill_from)
    else:
        cache = load_cache()
        if cache is None:
            print("No asof_rate_cache and no --backfill given -- run "
                  "'python asof_rate_cache.py --backfill 2026-03-26' first",
                  file=sys.stderr)
            sys.exit(1)
        start = datetime.date.fromisoformat(cache["asOf"]) + datetime.timedelta(days=1)

    stop = datetime.date.fromisoformat(end_date) if end_date else today
    if start > stop:
        print(f"asof_rate_cache already current through {cache['asOf']}")
        return cache

    cur = start
    days_done = 0
    while cur <= stop:
        date_str = cur.isoformat()
        try:
            snapshot, ok = process_day(cache, date_str)
        except Exception as e:
            print(f"WARN: day {date_str} failed hard ({type(e).__name__}: {e}) "
                  f"-- stopping at asOf={cache['asOf']}", file=sys.stderr)
            save_cache(cache)
            return cache
        if not ok:
            save_cache(cache)  # cache untouched for this date -- safe to persist as-is
            return cache
        save_snapshot(snapshot)
        cache["asOf"] = date_str
        days_done += 1
        cur += datetime.timedelta(days=1)
        if days_done % save_every == 0:
            save_cache(cache)
            print(f"...checkpoint through {date_str} ({days_done} days done)", file=sys.stderr)

    save_cache(cache)
    print(f"asof_rate_cache updated through {cache['asOf']}: {days_done} day-snapshots "
          f"written to {SNAPSHOT_DIR}/, {len(cache['batters'])} batters / "
          f"{len(cache['pitchers'])} pitchers tracked", file=sys.stderr)
    return cache


if __name__ == "__main__":
    backfill = end = None
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--backfill" and i + 1 < len(args):
            backfill = args[i + 1]; i += 2
        elif args[i] == "--end" and i + 1 < len(args):
            end = args[i + 1]; i += 2
        else:
            i += 1
    try:
        update_cache(backfill_from=backfill, end_date=end)
    except Exception as e:
        print(f"ASOF RATE CACHE UPDATE FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
