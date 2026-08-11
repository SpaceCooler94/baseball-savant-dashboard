#!/usr/bin/env python3
# ============================================================================
# situational.py -- bullpen fatigue, weather, umpire, rest/travel.
#
# The next layer up from "what does the model say": the situational facts a
# sharp bettor actually checks by hand before locking a number -- bullpen
# workload, wind, who's behind the plate, whether a team is playing its third
# game in three time zones. None of it needs a new data source: bullpen
# fatigue reuses the SAME bulk Statcast pull recent_form.py already makes
# (zero new network calls), and wind/temp/umpire ride the SAME schedule call
# build_daily_board.py already fetches, just with two more hydrate keys added.
# Only rest/travel needs one new call, and it's a small JSON schedule range,
# not another Statcast pull.
#
# DESIGN RULES, same as every other module in this pipeline:
#   - Pure functions. The network lives in build_daily_board.py; this module
#     takes data it's handed and returns a dict, so every function here is
#     testable with a plain fixture and no mocking.
#   - Team-code normalization is NOT this module's job. build_daily_board.py
#     owns TEAM_NAME_TO_ABBR/norm_team and always has; a second team-name
#     mapping here would be exactly the kind of duplicated-rule drift that bit
#     mlb_model.py's tier cutoffs. Callers pass already-normalized abbrevs in,
#     and this module returns raw Statcast team codes out (build_daily_board
#     normalizes on the way in AND the way out).
#   - Degrade to None, never guess. StatsAPI's weather/officials hydrate
#     shapes here are based on the documented hydrate vocabulary, not a
#     confirmed live response -- every parser below treats a missing or
#     reshaped field as "no data" rather than raising or inventing a value.
#   - REFERENCE ONLY, except one deliberate exception: temp. mlb_model.py's
#     project_hr has carried a temp-based HR multiplier since v5.0
#     (ctx["weather"]["temp"] -> a +/-6% nudge), but build_daily_board.py has
#     always passed ctx={"weather": {}} -- so that term has been silently
#     dead code for the model's entire life. Wiring in real temp here does not
#     add a new input, it connects one that was already part of the designed
#     formula. Same precedent as the lineup-slot fix: the raw-probability
#     MATH is unchanged, only a previously-null input starts being populated,
#     so MODEL_VERSION does not need to bump for this alone. Wind, bullpen
#     fatigue, rest/travel, and umpire are all genuinely NEW signals and ship
#     as angles/display fields only -- never touching raw_per_pa.
# ============================================================================

import datetime
import re

_WIND_RE = re.compile(r"(\d+(?:\.\d+)?)\s*mph", re.IGNORECASE)


def _nv(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


# ------------------------------ bullpen fatigue -------------------------------

def bullpen_fatigue(df, window_days=3):
    """Bulk Statcast pull (same df recent_form.fetch_statcast() returns) ->
    {raw_team_code: {"reliefPitches": int, "reliefAppearances": int, "games": int}}.

    Team attribution: a pitcher belongs to whichever team is NOT batting --
    inning_topbot=='Top' means the away team is up, so the pitcher is the
    HOME team's; 'Bot' is the reverse. This is the same trick already in the
    codebase, just for pitchers instead of batters: the FIRST pitcher to
    appear for a team in a game (by at_bat_number) is that game's starter,
    and every other pitcher who takes the mound for that team in that game is
    a reliever, by definition -- no separate starter/reliever data needed.

    Window is by CALENDAR DAYS back from the most recent date in the pull, not
    by team-games, so a doubleheader correctly shows as more taxed than a
    team that had an off day.
    """
    need = ["pitcher", "game_pk", "at_bat_number", "inning_topbot", "home_team", "away_team"]
    if any(c not in df.columns for c in need):
        return {}
    has_date = "game_date" in df.columns
    if not has_date:
        return {}
    import numpy as _np
    import pandas as _pd

    dates = _pd.to_datetime(df["game_date"], errors="coerce")
    max_date = dates.max()
    if _pd.isna(max_date):
        return {}
    cutoff = max_date - _pd.Timedelta(days=window_days - 1)
    mask = (dates >= cutoff) & (dates <= max_date)
    recent = df[mask.fillna(False)].copy()
    if recent.empty:
        return {}

    topbot = recent["inning_topbot"].astype(str)
    recent["_team"] = _np.where(topbot == "Top", recent["home_team"], recent["away_team"])
    recent = recent[recent["_team"].notna()]

    out = {}
    for (gpk, team), g in recent.groupby(["game_pk", "_team"]):
        g = g.sort_values("at_bat_number")
        starter = g["pitcher"].iloc[0]
        relief = g[g["pitcher"] != starter]
        bucket = out.setdefault(str(team), {"reliefPitches": 0, "reliefAppearances": 0, "games": 0})
        bucket["reliefPitches"] += int(len(relief))
        bucket["reliefAppearances"] += int(relief["pitcher"].nunique())
        bucket["games"] += 1
    return out


# --------------------------------- weather -------------------------------------

def parse_weather(weather):
    """StatsAPI schedule hydrate=weather block -> {tempF, windMph, windDir,
    condition}. windDir is 'out' (boosts offense), 'in' (suppresses it),
    'cross'/'calm', or None. All keys None-safe -- this hydrate's exact shape
    has not been confirmed against a live response, only against StatsAPI's
    documented hydrate vocabulary, so a reshaped or missing field must degrade
    to None rather than raise."""
    if not isinstance(weather, dict):
        return {"tempF": None, "windMph": None, "windDir": None, "condition": None}
    temp = _nv(weather.get("temp"))
    wind_raw = str(weather.get("wind") or "")
    wind_mph = None
    m = _WIND_RE.search(wind_raw)
    if m:
        wind_mph = float(m.group(1))
    low = wind_raw.lower()
    wind_dir = None
    if "out" in low:
        wind_dir = "out"
    elif "in" in low:
        wind_dir = "in"
    elif wind_mph is not None and wind_mph <= 3:
        wind_dir = "calm"
    elif wind_raw:
        wind_dir = "cross"
    return {"tempF": temp, "windMph": wind_mph, "windDir": wind_dir,
            "condition": weather.get("condition")}


# --------------------------------- umpire --------------------------------------

def home_plate_umpire(officials):
    """StatsAPI schedule hydrate=officials list -> home-plate umpire's name,
    or None. StatsAPI's officialType for the plate umpire is documented as
    'Home Plate'; a crew list with no entry of that type returns None rather
    than guessing which official was behind the plate.

    NAME ONLY. Zone-tendency scoring (how tight/wide a given umpire's zone
    runs) is a genuinely separate feature: it needs its own incremental cache
    joining umpire identity to called-pitch accuracy over many games, the same
    scope of work zone_engine.py's zones_cache.json was for batter/pitcher
    zones. Not built here -- this returns the assignment as context, not a
    graded signal, and should not be read as more than that."""
    if not isinstance(officials, list):
        return None
    for o in officials:
        if not isinstance(o, dict):
            continue
        if str(o.get("officialType", "")).strip().lower() == "home plate":
            person = o.get("official") or {}
            name = person.get("fullName")
            return name if name else None
    return None


# ------------------------------ rest / travel -----------------------------------

def parse_rest_travel(team_abbr, flat_games, target_date):
    """flat_games: [{"date": "YYYY-MM-DD", "hour": int_utc_or_None, "team": abbr}, ...]
    -- already normalized to this pipeline's team codes by the caller
    (build_daily_board.flatten_schedule_range), never by this module.

    Returns {"gamesL3": int, "dayAfterNight": bool, "backToBack": bool}, or
    None if the team has no games in the pulled window -- a legitimate off-day
    result, not an error.

    dayAfterNight is a coarse UTC-hour proxy (night games cluster ~22:00-04:00
    UTC, day games ~16:00-20:00), not a franchise-accurate day/night flag --
    good enough to catch the case that matters (a 10pm getaway game followed by
    a 1pm getaway-day start) without needing venue-local time zones.
    """
    games = [g for g in flat_games if g.get("team") == team_abbr]
    if not games:
        return None
    games.sort(key=lambda g: g.get("date") or "")
    games_l3 = len(games)
    today = [g for g in games if g.get("date") == target_date]
    prior = [g for g in games if (g.get("date") or "") < target_date]

    day_after_night = False
    if today and prior:
        h = prior[-1].get("hour")
        if h is not None:
            day_after_night = h >= 22 or h <= 4

    yday = _date_minus(target_date, 1)
    back_to_back = any(g.get("date") == yday for g in prior)

    return {"gamesL3": games_l3, "dayAfterNight": day_after_night, "backToBack": back_to_back}


def _date_minus(date_str, days):
    d = datetime.date.fromisoformat(date_str) - datetime.timedelta(days=days)
    return d.isoformat()
