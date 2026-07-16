#!/usr/bin/env python3
# ============================================================================
# build_daily_board.py -- Part 2 of the model port (the data layer). REFACTORED.
#
# Runs OFF the phone (on GitHub Actions). Pulls StatsAPI hitter/pitcher data +
# Savant metrics, feeds them into the parity-locked math in mlb_model.py, and
# writes daily_board.json in the exact shape MLB_Daily.js renders.
#
# CHANGES FROM v1 (production-review fixes):
#   1. TIMEZONE: all "today"/season logic pinned to America/New_York. GitHub
#      Actions runs UTC -- the old datetime.now() built TOMORROW's board any
#      time the Action fired after ~8pm ET.
#   2. CALL VOLUME: hitter pool now tries ONE hydrated roster call per team
#      (season stats + vl/vr splits inline) = ~30 requests instead of ~800.
#      Falls back to the proven per-player path (bounded thread pool) if the
#      hydrate shape isn't what we expect. Hydrate path must be confirmed on
#      the first live run -- same honesty rule as before.
#   3. CRASH SAFETY: the pitcher handedness lookup was UNGUARDED -- one failed
#      /people/{id} call after retries killed the entire build. Pitcher stats
#      + hand now come from one guarded hydrated call, cached per pid.
#   4. ATOMIC PUBLISH: board is written to a temp file and os.replace()d into
#      place. The old direct json.dump could leave Render serving truncated
#      JSON if the process died mid-write.
#   5. PUBLISH GATES: refuses to overwrite a good board with a degraded one
#      (min games / min pool thresholds) -- exits 1 so the Action keeps
#      yesterday's file instead of shipping a hollow board.
#   6. OBSERVABLE DEGRADATION: bare `except: pass` blocks replaced with a
#      health counter surfaced in board metadata (dataHealth) and stdout.
#      Additive key -- the thin client ignores unknown fields.
#   7. HTTP: shared Session, backoff with jitter, 429/5xx-aware retries.
#
# mlb_model.py is deliberately NOT touched: it is parity-locked to the JS and
# any "cleanup" there invalidates the parity test. All input hardening happens
# here, at the boundary.
# ============================================================================

import datetime
import json
import math
import os
import random
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from zoneinfo import ZoneInfo

import requests

import mlb_model as M

# ------------------------------ configuration -------------------------------

ET = ZoneInfo("America/New_York")
NOW_ET = datetime.datetime.now(ET)
TODAY = NOW_ET.strftime("%Y-%m-%d")
YEAR = NOW_ET.year

STATS_API = "https://statsapi.mlb.com/api/v1"
USER_AGENT = "mlb-daily-board/1.0 (personal analytics pipeline)"

MIN_PA = 25                 # pool floor, matches the model's MIN_PA
SPLIT_SIT_CODES = "vl,vr"
CAL_MODEL_VERSION = "log5-v5.0"   # calibration gate -- bump alongside the JS
BOARD_SCHEMA_VERSION = 5.1

# Publish gates: don't overwrite a good served board with a hollow one.
MIN_GAMES_TO_PUBLISH = 1
MIN_POOL_TO_PUBLISH = 100   # a normal slate yields ~300-400 qualified hitters

MAX_WORKERS = 6             # bounded concurrency for the per-player fallback
OUTPUT_PATH = "daily_board.json"

# Park factors + team maps, ported verbatim from Daily_Matchups.js constants.
# NOTE (data, not code): OAK=116 reflects Sutter Health Park (Sacramento) --
# revisit each season alongside the rest of this table.
PARK_FACTORS = {
    "COL": 122, "CIN": 108, "BOS": 108, "PHI": 107, "NYY": 106, "TEX": 105,
    "HOU": 104, "ATL": 104, "MIL": 103, "ARI": 103, "WSH": 102, "CHC": 102,
    "NYM": 101, "DET": 101, "MIN": 101, "LAD": 100, "STL": 100, "TOR": 100,
    "CLE": 99, "BAL": 99, "SEA": 99, "PIT": 98, "CHW": 98, "MIA": 98,
    "KCR": 97, "TBR": 97, "LAA": 97, "OAK": 116, "SDP": 94, "SFG": 93,
}

TEAM_MAP = {
    "AZ": "ARI", "ARI": "ARI", "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHC", "CWS": "CHW", "CHW": "CHW", "CIN": "CIN", "CLE": "CLE",
    "COL": "COL", "DET": "DET", "HOU": "HOU", "KC": "KCR", "KCR": "KCR",
    "LAA": "LAA", "LAD": "LAD", "MIA": "MIA", "MIL": "MIL", "MIN": "MIN",
    "NYM": "NYM", "NYY": "NYY", "OAK": "OAK", "ATH": "OAK", "PHI": "PHI",
    "PIT": "PIT", "SD": "SDP", "SDP": "SDP", "SEA": "SEA", "SF": "SFG",
    "SFG": "SFG", "STL": "STL", "TB": "TBR", "TBR": "TBR", "TEX": "TEX",
    "TOR": "TOR", "WSH": "WSH", "WSN": "WSH",
}

TEAM_NAME_TO_ABBR = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Colorado Rockies": "COL",
    "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KCR",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Athletics": "OAK", "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT", "San Diego Padres": "SDP",
    "San Francisco Giants": "SFG", "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TBR", "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}

# --------------------------- health / degradation ---------------------------

HEALTH = {
    "hydratedRosterPath": False,   # did the fast path work?
    "hitterStatMisses": 0,         # hitters skipped for missing season stats
    "splitMisses": 0,              # hitters kept but with null platoon splits
    "pitcherMisses": 0,            # probables we couldn't rate
    "savantExitVeloOk": False,
    "savantExpectedOk": False,
    "savantBatterArsenalOk": False,
    "savantPitcherArsenalOk": False,
    "warnings": [],
}


def warn(msg):
    """Loud, bounded warning: printed for the Action log AND kept in board meta."""
    print(f"WARN: {msg}", file=sys.stderr)
    if len(HEALTH["warnings"]) < 25:
        HEALTH["warnings"].append(msg)


# ------------------------------- HTTP layer ---------------------------------

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})


def http_json(url, tries=3, timeout=15):
    """GET JSON with exponential backoff + jitter. Honors Retry-After on 429."""
    last = None
    for attempt in range(tries):
        try:
            r = SESSION.get(url, timeout=timeout)
            if r.status_code == 429:
                wait = float(r.headers.get("Retry-After", 5))
                time.sleep(min(wait, 30))
                last = requests.HTTPError("429 Too Many Requests")
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            if attempt < tries - 1:
                time.sleep((2 ** attempt) + random.uniform(0, 0.5))
    raise last


def nv(v):
    """Same null-safe numeric coercion as mlb_model._nv / the JS nv()."""
    if v is None or v == "":
        return None
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _norm_name(name):
    """Normalize a player name for joining Savant ('Last, First') to StatsAPI
    ('First Last'). Lowercase, strip, reorder 'Last, First' -> 'first last'."""
    if not name:
        return ""
    n = str(name).strip()
    if "," in n:
        parts = [p.strip() for p in n.split(",", 1)]
        if len(parts) == 2:
            n = parts[1] + " " + parts[0]
    return " ".join(n.lower().split())


def norm_team(abbr):
    if not abbr:
        return None
    return TEAM_MAP.get(str(abbr).upper(), str(abbr).upper())


# --------------------------- batter data (StatsAPI) --------------------------

def _hitter_record(pid, name, tabbr, pos, bat_side, season_stat,
                   vs_l_avg=None, vs_r_avg=None, vs_l_pa=None, vs_r_pa=None):
    """Shape one hitter into the record format mlb_model expects.
    Returns None if the hitter doesn't meet the pool floor."""
    st = season_stat or {}
    pa = nv(st.get("plateAppearances"))
    if pa is None or pa < MIN_PA:
        return None
    bb = nv(st.get("baseOnBalls"))
    hr = nv(st.get("homeRuns"))
    so = nv(st.get("strikeOuts"))
    return {
        "id": pid,
        "name": name,
        "teamAbbr": tabbr,
        "pos": pos,
        "batSide": bat_side,
        "pa": pa,
        "obp": nv(st.get("obp")),
        "avg": nv(st.get("avg")),
        "bbPct": (bb / pa * 100) if bb is not None else None,
        "hr": hr,
        "hrRate": (hr / pa * 100) if hr is not None else None,
        "kPct": (so / pa * 100) if so is not None else None,
        "babip": nv(st.get("babip")),
        "vsLAvg": vs_l_avg,
        "vsRAvg": vs_r_avg,
        "vsLPa": vs_l_pa,
        "vsRPa": vs_r_pa,
        # Recent-form OPS still null (would need per-hitter gameLog).
        "l5Ops": None, "l10Ops": None,
        # Lineup slot unknown until lineups post; null -> whole-lineup PA.
        "orderAvg": None,
    }


def _parse_person_stats(person):
    """Pull (season_stat, vl/vr splits) out of a hydrated person.stats block.
    Returns (season_stat_or_None, vsL_avg, vsR_avg, vsL_pa, vsR_pa)."""
    season_stat = None
    vs_l_avg = vs_r_avg = vs_l_pa = vs_r_pa = None
    for block in person.get("stats") or []:
        btype = ((block.get("type") or {}).get("displayName") or "").lower()
        splits = block.get("splits") or []
        if btype == "season" and splits:
            season_stat = splits[0].get("stat", {})
        elif btype == "statsplits":
            for s in splits:
                code = str((s.get("split") or {}).get("code") or "").lower()
                sst = s.get("stat", {})
                if code == "vl":
                    vs_l_avg = nv(sst.get("avg"))
                    vs_l_pa = nv(sst.get("plateAppearances"))
                elif code == "vr":
                    vs_r_avg = nv(sst.get("avg"))
                    vs_r_pa = nv(sst.get("plateAppearances"))
    return season_stat, vs_l_avg, vs_r_avg, vs_l_pa, vs_r_pa


def _fetch_hitter_slow(pid, name, tabbr, pos, bat_side):
    """Per-player fallback: season stats + vl/vr splits in ONE combined call
    (the old version used two). Best-effort splits: a failed/missing split
    block keeps the hitter with nulls rather than dropping them."""
    try:
        data = http_json(
            f"{STATS_API}/people/{pid}/stats?stats=season,statSplits&group=hitting"
            f"&season={YEAR}&sitCodes={SPLIT_SIT_CODES}"
        )
    except Exception as e:
        HEALTH["hitterStatMisses"] += 1
        warn(f"hitter stats failed pid={pid} ({type(e).__name__})")
        return None
    season_stat = None
    vs_l_avg = vs_r_avg = vs_l_pa = vs_r_pa = None
    for block in data.get("stats") or []:
        btype = ((block.get("type") or {}).get("displayName") or "").lower()
        splits = block.get("splits") or []
        if btype == "season" and splits:
            season_stat = splits[0].get("stat", {})
        elif btype == "statsplits":
            for s in splits:
                code = str((s.get("split") or {}).get("code") or "").lower()
                sst = s.get("stat", {})
                if code == "vl":
                    vs_l_avg = nv(sst.get("avg"))
                    vs_l_pa = nv(sst.get("plateAppearances"))
                elif code == "vr":
                    vs_r_avg = nv(sst.get("avg"))
                    vs_r_pa = nv(sst.get("plateAppearances"))
    if season_stat is None:
        return None
    if vs_l_avg is None and vs_r_avg is None:
        HEALTH["splitMisses"] += 1
    return _hitter_record(pid, name, tabbr, pos, bat_side, season_stat,
                          vs_l_avg, vs_r_avg, vs_l_pa, vs_r_pa)


def build_batter_pool():
    """FAST PATH: one hydrated roster call per team pulls season stats + vl/vr
    splits inline (~30 requests total). If the hydrate shape isn't recognized
    (StatsAPI hydrate grammar must be confirmed on the first live run), we fall
    back to the combined per-player call, bounded-threaded (~400 requests, still
    half the old version's ~800 and off the phone either way)."""
    teams = http_json(f"{STATS_API}/teams?sportId=1&season={YEAR}").get("teams", [])
    pool = []
    slow_queue = []  # (pid, name, tabbr, pos, batSide) needing per-player fetch

    hydrate = (
        "person(stats(group=[hitting],type=[season,statSplits],"
        f"sitCodes=[vl,vr],season={YEAR}))"
    )
    any_hydrated = False

    for t in teams:
        tid = t.get("id")
        tabbr = norm_team(t.get("abbreviation"))
        if not tid:
            continue
        try:
            roster = http_json(
                f"{STATS_API}/teams/{tid}/roster?rosterType=active&season={YEAR}"
                f"&hydrate={hydrate}"
            )
        except Exception as e:
            warn(f"roster fetch failed team={tabbr} ({type(e).__name__})")
            continue
        for entry in roster.get("roster", []):
            person = entry.get("person", {})
            pid = person.get("id")
            pos = entry.get("position", {}).get("abbreviation")
            if not pid or pos == "P":
                continue
            name = person.get("fullName")
            bat_side = (person.get("batSide") or {}).get("code")
            if person.get("stats"):
                any_hydrated = True
                season_stat, la, ra, lpa, rpa = _parse_person_stats(person)
                if season_stat is None:
                    continue
                if la is None and ra is None:
                    HEALTH["splitMisses"] += 1
                rec = _hitter_record(pid, name, tabbr, pos, bat_side,
                                     season_stat, la, ra, lpa, rpa)
                if rec:
                    pool.append(rec)
            else:
                slow_queue.append((pid, name, tabbr, pos, bat_side))

    HEALTH["hydratedRosterPath"] = any_hydrated
    if slow_queue:
        if any_hydrated:
            warn(f"{len(slow_queue)} hitters missing hydrated stats -> per-player fallback")
        else:
            warn("hydrated roster path returned no stats -- full per-player fallback")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(_fetch_hitter_slow, *args) for args in slow_queue]
            for f in as_completed(futures):
                rec = f.result()
                if rec:
                    pool.append(rec)
    return pool


# --------------------------- Savant metrics (bulk) --------------------------

def fetch_savant_metrics():
    """Bulk barrel%/xBA/etc keyed by normalized name. Best-effort per endpoint,
    but every failure is now COUNTED and WARNED instead of silently swallowed.
    Returns {} on total failure so the board still builds on rate stats."""
    metrics = {}
    try:
        from pybaseball import (statcast_batter_exitvelo_barrels,
                                statcast_batter_expected_stats)
    except Exception as e:
        warn(f"pybaseball import failed ({type(e).__name__}) -- Savant metrics skipped")
        return metrics

    # Exit velo / barrels (+ derived FB% from fbld/gb counts)
    try:
        ev = statcast_batter_exitvelo_barrels(YEAR, minBBE=25)
        name_c = find_col(ev, ["last_name, first_name", "player_name", "name"])
        brl_c = find_col(ev, ["brl_percent", "barrel_batted_rate"])
        hh_c = find_col(ev, ["ev95percent", "hard_hit_percent", "hard_hit_rate"])
        ev_c = find_col(ev, ["avg_hit_speed", "exit_velocity_avg"])
        ss_c = find_col(ev, ["anglesweetspotpercent", "sweet_spot_percent"])
        maxev_c = find_col(ev, ["max_hit_speed", "max_exit_velocity"])
        hrdist_c = find_col(ev, ["avg_hr_distance", "avg_distance"])
        la_c = find_col(ev, ["avg_hit_angle", "launch_angle_avg"])
        fbld_c = find_col(ev, ["fbld"])
        gb_c = find_col(ev, ["gb"])
        if name_c:
            for _, row in ev.iterrows():
                k = _norm_name(row.get(name_c))
                if not k:
                    continue
                m = metrics.setdefault(k, {})
                if brl_c: m["barrelPct"] = nv(row.get(brl_c))
                if hh_c: m["hardHitPct"] = nv(row.get(hh_c))
                if ev_c: m["avgEV"] = nv(row.get(ev_c))
                if ss_c: m["sweetSpotPct"] = nv(row.get(ss_c))
                if maxev_c: m["maxEV"] = nv(row.get(maxev_c))
                if hrdist_c: m["hrDistance"] = nv(row.get(hrdist_c))
                if la_c: m["launchAngle"] = nv(row.get(la_c))
                if fbld_c and gb_c:
                    fbld = nv(row.get(fbld_c)); gb = nv(row.get(gb_c))
                    if fbld is not None and gb is not None and (fbld + gb) > 0:
                        m["fbPct"] = round(fbld / (fbld + gb) * 100, 1)
            HEALTH["savantExitVeloOk"] = True
        else:
            warn(f"exitvelo: no name column recognized in {list(ev.columns)[:8]}...")
    except Exception as e:
        warn(f"exitvelo fetch failed ({type(e).__name__}: {e})")

    # Expected stats (xBA, xSLG, xwOBA)
    try:
        xs = statcast_batter_expected_stats(YEAR, minPA=25)
        name_c = find_col(xs, ["last_name, first_name", "player_name", "name"])
        xba_c = find_col(xs, ["est_ba", "xba", "expected_batting_avg"])
        xslg_c = find_col(xs, ["est_slg", "xslg", "expected_slg"])
        xwoba_c = find_col(xs, ["est_woba", "xwoba", "expected_woba"])
        if name_c:
            for _, row in xs.iterrows():
                k = _norm_name(row.get(name_c))
                if not k:
                    continue
                m = metrics.setdefault(k, {})
                if xba_c: m["xBA"] = nv(row.get(xba_c))
                if xslg_c: m["xSLG"] = nv(row.get(xslg_c))
                if xwoba_c: m["xwOBA"] = nv(row.get(xwoba_c))
            HEALTH["savantExpectedOk"] = True
        else:
            warn("expected-stats: no name column recognized")
    except Exception as e:
        warn(f"expected-stats fetch failed ({type(e).__name__}: {e})")

    # Batter vs pitch type (whiff/K/BA per pitch, top 5 by exposure)
    try:
        batter_arsenal = None
        try:
            from pybaseball import statcast_batter_pitch_arsenal as batter_arsenal
        except Exception:
            pass
        if batter_arsenal is not None:
            bdf = batter_arsenal(YEAR, minPA=25)
            bn = find_col(bdf, ["last_name, first_name", "player_name", "name"])
            bp = find_col(bdf, ["pitch_name", "pitch_type", "pitch"])
            bba = find_col(bdf, ["ba"]); bslg = find_col(bdf, ["slg"])
            bwhiff = find_col(bdf, ["whiff_percent"]); bk = find_col(bdf, ["k_percent"])
            busage = find_col(bdf, ["pitch_usage", "pa"])
            if bn and bp:
                for _, row in bdf.iterrows():
                    k = _norm_name(row.get(bn))
                    if not k:
                        continue
                    pitch = str(row.get(bp) or "").strip()
                    if not pitch:
                        continue
                    entry = {"pitch": pitch}
                    if bba: entry["ba"] = nv(row.get(bba))
                    if bslg: entry["slg"] = nv(row.get(bslg))
                    if bwhiff: entry["whiff"] = nv(row.get(bwhiff))
                    if bk: entry["k"] = nv(row.get(bk))
                    if busage: entry["seen"] = nv(row.get(busage))
                    m = metrics.setdefault(k, {})
                    m.setdefault("vsPitch", []).append(entry)
                for k in metrics:
                    vp = metrics[k].get("vsPitch")
                    if vp:
                        metrics[k]["vsPitch"] = sorted(
                            vp, key=lambda x: (x.get("seen") or 0), reverse=True
                        )[:5]
                HEALTH["savantBatterArsenalOk"] = True
    except Exception as e:
        warn(f"batter-arsenal fetch failed ({type(e).__name__}: {e})")

    return metrics


def fetch_pitcher_pitch_mix():
    """Bulk pitch-usage + results-allowed per pitch, keyed by normalized name.
    Reference data only -- does NOT feed the log5 projection."""
    mix = {}
    arsenal_fn = None
    try:
        try:
            from pybaseball import statcast_pitcher_arsenal_stats as arsenal_fn
        except Exception:
            from pybaseball import statcast_pitcher_pitch_arsenal as arsenal_fn
    except Exception as e:
        warn(f"pitcher-arsenal import failed ({type(e).__name__})")
        return mix
    try:
        df = arsenal_fn(YEAR, minPA=50)
        name_c = find_col(df, ["last_name, first_name", "player_name", "name"])
        pitch_c = find_col(df, ["pitch_name", "pitch_type", "pitch"])
        usage_c = find_col(df, ["pitch_usage", "pitch_percent", "usage"])
        ba_c = find_col(df, ["ba"]); slg_c = find_col(df, ["slg"])
        woba_c = find_col(df, ["woba"])
        if name_c and pitch_c and usage_c:
            for _, row in df.iterrows():
                k = _norm_name(row.get(name_c))
                if not k:
                    continue
                pitch = str(row.get(pitch_c) or "").strip()
                usage = nv(row.get(usage_c))
                if not pitch or usage is None:
                    continue
                entry = {"pitch": pitch, "usage": usage}
                if ba_c: entry["ba"] = nv(row.get(ba_c))
                if slg_c: entry["slg"] = nv(row.get(slg_c))
                if woba_c: entry["woba"] = nv(row.get(woba_c))
                mix.setdefault(k, []).append(entry)
            for k in mix:
                mix[k] = sorted(mix[k], key=lambda x: x["usage"], reverse=True)[:5]
            HEALTH["savantPitcherArsenalOk"] = True
        else:
            warn(f"pitcher-arsenal: required columns missing from {list(df.columns)[:8]}...")
    except Exception as e:
        warn(f"pitcher-arsenal fetch failed ({type(e).__name__}: {e})")
    return mix


# --------------------------- schedule + pitchers ----------------------------

_PITCHER_CACHE = {}


def get_pitcher_rates(pid):
    """Season hit/HR-rate-allowed per batter faced + handedness in ONE hydrated
    call (the old version made two, and the hand lookup was UNGUARDED -- a single
    failure there killed the whole build). Cached per pid for doubleheaders.
    Returns None on any failure; the model treats a null pitcher as league-rate
    with a 'partial data' flag."""
    if pid in _PITCHER_CACHE:
        return _PITCHER_CACHE[pid]
    result = None
    try:
        data = http_json(
            f"{STATS_API}/people/{pid}"
            f"?hydrate=stats(group=[pitching],type=[season],season={YEAR})"
        )
        person = (data.get("people") or [{}])[0]
        hand = (person.get("pitchHand") or {}).get("code")
        st = {}
        for block in person.get("stats") or []:
            splits = block.get("splits") or []
            if splits:
                st = splits[0].get("stat", {})
                break
        bf = nv(st.get("battersFaced"))
        if bf:
            obp = nv(st.get("obp"))
            bb = nv(st.get("baseOnBalls"))
            hr = nv(st.get("homeRuns"))
            bb_frac = (bb / bf) if bb is not None else None
            hit_rate_allowed = (
                max(0.01, obp - bb_frac)
                if (obp is not None and bb_frac is not None) else None
            )
            result = {
                "hand": hand,
                "hitRateAllowedPerPA": hit_rate_allowed,
                "hrRateAllowedPerPA": (hr / bf) if hr is not None else None,
                "battersFaced": bf,
                "whip": nv(st.get("whip")),
                "hrPer9": nv(st.get("homeRunsPer9")),
            }
    except Exception as e:
        warn(f"pitcher rates failed pid={pid} ({type(e).__name__})")
    if result is None:
        HEALTH["pitcherMisses"] += 1
    _PITCHER_CACHE[pid] = result
    return result


def get_schedule():
    data = http_json(
        f"{STATS_API}/schedule?sportId=1&date={TODAY}"
        "&hydrate=probablePitcher(note),team,venue,linescore"
    )
    games = []
    for date in data.get("dates", []):
        games.extend(date.get("games", []))
    return games


# ------------------------------- assembly -----------------------------------

def load_calibration():
    """Best-effort read of a committed calibration.json. None -> uncalibrated.
    Version-gated like the JS side."""
    try:
        with open("calibration.json") as f:
            cal = json.load(f)
        if isinstance(cal, dict) and cal.get("modelVersion") == CAL_MODEL_VERSION:
            return cal
        warn(f"calibration.json present but modelVersion != {CAL_MODEL_VERSION} -- ignored")
    except FileNotFoundError:
        pass
    except Exception as e:
        warn(f"calibration.json unreadable ({type(e).__name__}) -- running uncalibrated")
    return None


def project_side(hitters, opp_pitcher, ctx, league, calibration):
    out = []
    for h in hitters:
        hit = M.project_hit(h, opp_pitcher, ctx, league, calibration)
        hr = M.project_hr(h, opp_pitcher, ctx, league, calibration)
        sm = h.get("_savant") or {}
        out.append({
            "hitterId": h["id"], "name": h["name"], "pos": h["pos"],
            "teamAbbr": h["teamAbbr"], "batSide": h.get("batSide"),
            "hitProb": hit["perGame"], "hitProbPerPA": hit["perPA"], "hitTier": hit["tier"],
            "hitDataQuality": hit["dataQuality"], "hitSignals": hit["signals"], "hitRisks": hit["risks"],
            "hitInputs": hit["inputs"],
            "hrProb": hr["perGame"], "hrProbPerPA": hr["perPA"], "hrTier": hr["tier"],
            "hrDataQuality": hr["dataQuality"], "hrSignals": hr["signals"], "hrRisks": hr["risks"],
            "hrInputs": hr["inputs"],
            "expectedPA": hit["expectedPA"], "batOrderAvg": h.get("orderAvg"),
            "lineupUnconfirmed": True,
            "viewScore": (hit["perGame"] + hr["perGame"]),
            "metrics": {
                "avg": h.get("avg"), "obp": h.get("obp"), "hr": h.get("hr"),
                "hrRate": h.get("hrRate"),
                "kPct": h.get("kPct"), "babip": h.get("babip"),
                "barrelPct": sm.get("barrelPct"),
                "hardHitPct": sm.get("hardHitPct"),
                "avgEV": sm.get("avgEV"),
                "pullPct": sm.get("pullPct"),
                "fbPct": sm.get("fbPct"),
                "xBA": sm.get("xBA"),
                "xSLG": sm.get("xSLG"),
                "xwOBA": sm.get("xwOBA"),
                "sweetSpotPct": sm.get("sweetSpotPct"),
                "maxEV": sm.get("maxEV"),
                "hrDistance": sm.get("hrDistance"),
                "launchAngle": sm.get("launchAngle"),
                "vsPitch": sm.get("vsPitch"),
            },
        })
    out.sort(key=lambda x: x["viewScore"], reverse=True)
    return out


def build_board():
    pool = build_batter_pool()
    savant = fetch_savant_metrics()
    pitch_mix = fetch_pitcher_pitch_mix()
    for h in pool:
        h["_savant"] = savant.get(_norm_name(h.get("name")), {})
    league = M.league_rates(pool)
    calibration = load_calibration()
    games = get_schedule()

    by_team = {}
    for h in pool:
        by_team.setdefault(h["teamAbbr"], []).append(h)

    merged = []
    for g in games:
        teams = g.get("teams", {})
        home = teams.get("home", {}).get("team", {})
        away = teams.get("away", {}).get("team", {})
        home_abbr = TEAM_NAME_TO_ABBR.get(home.get("name")) or norm_team(home.get("abbreviation"))
        away_abbr = TEAM_NAME_TO_ABBR.get(away.get("name")) or norm_team(away.get("abbreviation"))
        park = PARK_FACTORS.get(home_abbr, 100)

        home_prob = teams.get("home", {}).get("probablePitcher")
        away_prob = teams.get("away", {}).get("probablePitcher")
        hp = get_pitcher_rates(home_prob["id"]) if home_prob and home_prob.get("id") else None
        ap = get_pitcher_rates(away_prob["id"]) if away_prob and away_prob.get("id") else None
        if hp and home_prob:
            hp = dict(hp)  # cached dict -- don't mutate the shared copy
            hp["name"] = home_prob.get("fullName")
            hp["pitchMix"] = pitch_mix.get(_norm_name(hp["name"]))
        if ap and away_prob:
            ap = dict(ap)
            ap["name"] = away_prob.get("fullName")
            ap["pitchMix"] = pitch_mix.get(_norm_name(ap["name"]))

        ctx = {"park": park, "weather": {}}  # weather omitted in v1; model treats temp=None safely

        home_hitters = project_side(by_team.get(home_abbr, []), ap, ctx, league, calibration)
        away_hitters = project_side(by_team.get(away_abbr, []), hp, ctx, league, calibration)
        all_h = home_hitters + away_hitters

        merged.append({
            "gameId": g.get("gamePk"),
            "gameTime": g.get("gameDate"),
            "venue": (g.get("venue") or {}).get("name"),
            "venueParkFactor": park,
            "weather": {},
            "homeTeam": {"name": home.get("name"), "abbr": home_abbr},
            "awayTeam": {"name": away.get("name"), "abbr": away_abbr},
            "homeProbable": ({"name": hp["name"], "hand": hp.get("hand"), "whip": hp.get("whip"),
                              "hrPer9": hp.get("hrPer9"), "pitchMix": hp.get("pitchMix")}
                             if hp else None),
            "awayProbable": ({"name": ap["name"], "hand": ap.get("hand"), "whip": ap.get("whip"),
                              "hrPer9": ap.get("hrPer9"), "pitchMix": ap.get("pitchMix")}
                             if ap else None),
            "homeMatchups": home_hitters,
            "awayMatchups": away_hitters,
            "topHitTargets": sorted(all_h, key=lambda x: x["hitProb"], reverse=True)[:8],
            "topHrTargets": sorted(all_h, key=lambda x: x["hrProb"], reverse=True)[:8],
            "topOverall": sorted(all_h, key=lambda x: x["viewScore"], reverse=True)[:8],
        })

    return {
        "schemaVersion": BOARD_SCHEMA_VERSION,
        "builtAt": TODAY,
        "builtTs": datetime.datetime.now(ET).isoformat(),
        "gamesCount": len(merged),
        "calibrationApplied": bool(calibration),
        "poolSize": len(pool),
        "leagueRates": {
            "hitRatePerPA": round(league["hitRatePerPA"], 4),
            "hrRatePerPA": round(league["hrRatePerPA"], 4),
        },
        "dataHealth": HEALTH,   # additive key; thin client ignores unknown fields
        "games": merged,
    }


def atomic_write_json(path, obj):
    """Write to a temp file in the same directory, then os.replace() -- readers
    (Render) never see a partially written board."""
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".board_", suffix=".json")
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


def main():
    board = build_board()

    # Publish gates: never overwrite a good served board with a hollow one.
    if board["gamesCount"] < MIN_GAMES_TO_PUBLISH:
        print(f"NOT PUBLISHING: only {board['gamesCount']} games "
              f"(min {MIN_GAMES_TO_PUBLISH}) -- likely off-day or schedule fetch issue",
              file=sys.stderr)
        sys.exit(1)
    if board["poolSize"] < MIN_POOL_TO_PUBLISH:
        print(f"NOT PUBLISHING: pool of {board['poolSize']} hitters "
              f"(min {MIN_POOL_TO_PUBLISH}) -- upstream data degraded",
              file=sys.stderr)
        sys.exit(1)

    atomic_write_json(OUTPUT_PATH, board)
    print(f"Wrote {OUTPUT_PATH}: {board['gamesCount']} games, "
          f"{board['poolSize']} hitters, calibrated={board['calibrationApplied']}, "
          f"hydratedPath={HEALTH['hydratedRosterPath']}, "
          f"warnings={len(HEALTH['warnings'])}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"BUILD FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
