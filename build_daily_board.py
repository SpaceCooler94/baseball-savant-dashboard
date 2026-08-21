#!/usr/bin/env python3
# ============================================================================
# build_daily_board.py -- Part 2 of the model port (the data layer). v2.1
#
# Runs OFF the phone (on GitHub Actions). Pulls StatsAPI hitter/pitcher data +
# Savant metrics, feeds them into the parity-locked math in mlb_model.py, and
# writes daily_board.json in the exact shape MLB_Daily.js renders.
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

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import requests

import mlb_model as M
import recent_form as RF
import zone_engine as Z
import situational as SIT
import pitch_shape_cache as PSC

if not hasattr(M, "MODEL_VERSION"):
    sys.exit("FATAL: mlb_model.py is pre-v5.4 (no MODEL_VERSION). This build "
             "script needs the v5.4+ model -- commit the current mlb_model.py "
             "to the repo alongside build_daily_board.py.")

# ------------------------------ configuration -------------------------------

ET = ZoneInfo("America/New_York")
NOW_ET = datetime.datetime.now(ET)
TODAY = NOW_ET.strftime("%Y-%m-%d")
YEAR = NOW_ET.year

STATS_API = "https://statsapi.mlb.com/api/v1"
USER_AGENT = "mlb-daily-board/1.0 (personal analytics pipeline)"

MIN_PA = 25
SPLIT_SIT_CODES = "vl,vr"
BOARD_SCHEMA_VERSION = 5.8
LEAGUE_K_PER_BF_PCT = 22.0

MIN_GAMES_TO_PUBLISH = 1
MIN_POOL_TO_PUBLISH = 100

MAX_WORKERS = 6
OUTPUT_PATH = "daily_board.json"

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
    "hydratedRosterPath": False,
    "hitterStatMisses": 0,
    "splitMisses": 0,
    "pitcherMisses": 0,
    "savantExitVeloOk": False,
    "savantExpectedOk": False,
    "savantBatterArsenalOk": False,
    "savantPitcherArsenalOk": False,
    "savantPitcherShapeOk": False,
    "recentFormOk": False,
    "batSpeedOk": False,
    "lineupSlotsOk": False,
    "lineupSlotCount": 0,
    "bullpenFatigueOk": False,
    "restTravelOk": False,
    "zoneCacheOk": False,
    "zoneCacheAsOf": None,
    "warnings": [],
}


def warn(msg):
    print(f"WARN: {msg}", file=sys.stderr)
    if len(HEALTH["warnings"]) < 25:
        HEALTH["warnings"].append(msg)


# ------------------------------- HTTP layer ---------------------------------

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})


def http_json(url, tries=3, timeout=15):
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
    st = season_stat or {}
    pa = nv(st.get("plateAppearances"))
    if pa is None or pa < MIN_PA:
        return None
    bb = nv(st.get("baseOnBalls"))
    hr = nv(st.get("homeRuns"))
    so = nv(st.get("strikeOuts"))
    return {
        "id": pid, "name": name, "teamAbbr": tabbr, "pos": pos, "batSide": bat_side,
        "pa": pa, "obp": nv(st.get("obp")), "avg": nv(st.get("avg")),
        "bbPct": (bb / pa * 100) if bb is not None else None,
        "hr": hr, "hrRate": (hr / pa * 100) if hr is not None else None,
        "kPct": (so / pa * 100) if so is not None else None,
        "babip": nv(st.get("babip")),
        "vsLAvg": vs_l_avg, "vsRAvg": vs_r_avg, "vsLPa": vs_l_pa, "vsRPa": vs_r_pa,
        "l5Ops": None, "l10Ops": None, "orderAvg": None,
    }


def _parse_person_stats(person):
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
                    vs_l_avg = nv(sst.get("avg")); vs_l_pa = nv(sst.get("plateAppearances"))
                elif code == "vr":
                    vs_r_avg = nv(sst.get("avg")); vs_r_pa = nv(sst.get("plateAppearances"))
    return season_stat, vs_l_avg, vs_r_avg, vs_l_pa, vs_r_pa


def _fetch_hitter_slow(pid, name, tabbr, pos, bat_side):
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
                    vs_l_avg = nv(sst.get("avg")); vs_l_pa = nv(sst.get("plateAppearances"))
                elif code == "vr":
                    vs_r_avg = nv(sst.get("avg")); vs_r_pa = nv(sst.get("plateAppearances"))
    if season_stat is None:
        return None
    if vs_l_avg is None and vs_r_avg is None:
        HEALTH["splitMisses"] += 1
    return _hitter_record(pid, name, tabbr, pos, bat_side, season_stat,
                          vs_l_avg, vs_r_avg, vs_l_pa, vs_r_pa)


def build_batter_pool():
    teams = http_json(f"{STATS_API}/teams?sportId=1&season={YEAR}").get("teams", [])
    pool = []
    slow_queue = []

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
    metrics = {}
    try:
        from pybaseball import (statcast_batter_exitvelo_barrels,
                                statcast_batter_expected_stats)
    except Exception as e:
        warn(f"pybaseball import failed ({type(e).__name__}) -- Savant metrics skipped")
        return metrics

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
        # Season shape baseline (avgVelo/avgSpin) USED to be extracted here
        # off this same arsenal-stats pull. Removed: confirmed via a live
        # Savant column-availability check that this leaderboard carries no
        # velocity/spin columns at all -- the extraction never found a match
        # on a real pull, HEALTH["savantPitcherShapeOk"] was always False,
        # and this function warned about it every single run for something
        # that was never going to change. That's noise, not a live health
        # check, so it's gone. Real season shape now comes from
        # pitch_shape_cache.py (season-long incremental cache over the raw
        # Per-Pitch Statcast pull, which DOES carry these columns), wired
        # into enrich_probable() directly by pitcher id -- see there.
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
            so = nv(st.get("strikeOuts"))
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
                "kPct": (so / bf * 100) if so is not None else None,
                "whip": nv(st.get("whip")),
                "hrPer9": nv(st.get("homeRunsPer9")),
            }
    except Exception as e:
        warn(f"pitcher rates failed pid={pid} ({type(e).__name__})")
    if result is None:
        HEALTH["pitcherMisses"] += 1
    _PITCHER_CACHE[pid] = result
    return result


def fetch_schedule_range(days_back=3):
    end = NOW_ET.date()
    start = end - datetime.timedelta(days=days_back)
    return http_json(
        f"{STATS_API}/schedule?sportId=1&startDate={start.isoformat()}"
        f"&endDate={end.isoformat()}"
    )


def flatten_schedule_range(sched_range):
    out = []
    if not isinstance(sched_range, dict):
        return out
    for date in sched_range.get("dates", []):
        for g in date.get("games", []):
            game_date = g.get("gameDate") or ""
            try:
                hour = int(game_date[11:13])
            except (ValueError, IndexError):
                hour = None
            for side in ("home", "away"):
                team = (g.get("teams", {}).get(side, {}) or {}).get("team", {}) or {}
                abbr = TEAM_NAME_TO_ABBR.get(team.get("name")) or norm_team(team.get("abbreviation"))
                if abbr:
                    out.append({"date": game_date[:10], "hour": hour, "team": abbr})
    return out


def get_schedule():
    data = http_json(
        f"{STATS_API}/schedule?sportId=1&date={TODAY}"
        "&hydrate=probablePitcher(note),team,venue,linescore,weather,officials"
    )
    games = []
    for date in data.get("dates", []):
        games.extend(date.get("games", []))
    return games


# ------------------------------- assembly -----------------------------------

def load_calibration():
    try:
        with open("calibration.json") as f:
            cal = json.load(f)
        if isinstance(cal, dict) and cal.get("modelVersion") == M.MODEL_VERSION:
            return cal
        warn(f"calibration.json present but modelVersion != {M.MODEL_VERSION} -- ignored")
    except FileNotFoundError:
        pass
    except Exception as e:
        warn(f"calibration.json unreadable ({type(e).__name__}) -- running uncalibrated")
    return None


def pitch_fit(vs_pitch, pitch_mix):
    if not vs_pitch or not pitch_mix:
        return None, None
    by_pitch = {}
    for e in vs_pitch:
        name = str(e.get("pitch") or "").strip().lower()
        if name:
            by_pitch[name] = e
    covered = 0.0
    weighted = 0.0
    for pm in pitch_mix:
        usage = nv(pm.get("usage"))
        if usage is None or usage <= 0:
            continue
        e = by_pitch.get(str(pm.get("pitch") or "").strip().lower())
        ba = nv(e.get("ba")) if e else None
        if ba is not None:
            covered += usage
            weighted += usage * ba
    if covered < 40:
        return None, None
    return round(weighted / covered, 3), round(covered, 1)


def pctl_in(sorted_vals, v):
    if v is None or not sorted_vals:
        return None
    lo, hi = 0, len(sorted_vals)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_vals[mid] < v:
            lo = mid + 1
        else:
            hi = mid
    return round(lo / len(sorted_vals), 3)


def arsenal_overlap_count(vs_pitch, pitch_mix):
    if not vs_pitch or not pitch_mix:
        return 0, 0
    by_pitch = {}
    for v in vs_pitch:
        name = str(v.get("pitch") or "").strip().lower()
        if name:
            by_pitch[name] = v
    core = RF.core_mix(pitch_mix)
    wins = 0
    for m in core:
        key = str(m.get("pitch") or "").strip().lower()
        v = by_pitch.get(key)
        ba = nv(v.get("ba")) if v else None
        if ba is not None and ba >= 0.300:
            wins += 1
    return wins, len(core)


def compute_angles(row, opp):
    angles = []
    def add(key, label, cls):
        angles.append({"key": key, "label": label, "cls": cls})
    m = row.get("metrics") or {}
    xba, avg = m.get("xBA"), m.get("avg")
    if xba is not None and avg is not None:
        gap = xba - avg
        if gap >= 0.020:
            add("xba_pos", "xBA %+d pts vs AVG, positive regression due" % round(gap * 1000), "green")
        elif gap <= -0.020:
            add("xba_neg", "Overperforming xBA by %d pts" % round(-gap * 1000), "orange")
    barrel, hr_rate = m.get("barrelPct"), m.get("hrRate")
    if barrel is not None and hr_rate is not None and barrel >= 12 and hr_rate <= 3.5:
        add("barrels_due", "Barrel rate outruns HR rate, power due", "green")
    hk = m.get("kPct")
    pk = opp.get("kPct") if opp else None
    if hk is not None and pk is not None:
        if pk >= LEAGUE_K_PER_BF_PCT + 5 and hk >= 27:
            add("k_trap", "Strikeout trap: high-K bat vs high-K arm", "red")
        elif pk <= LEAGUE_K_PER_BF_PCT - 4 and hk <= 18:
            add("bip_matchup", "Ball-in-play matchup: low-K bat vs low-K arm", "green")
    fit = row.get("coreFitBA") if row.get("coreFitBA") is not None else row.get("pitchFitBA")
    if fit is not None:
        if fit >= 0.300:
            add("fit_pos", "Hits this arsenal (.%03d core fit)" % round(fit * 1000), "green")
        elif fit <= 0.200:
            add("fit_neg", "Struggles vs this arsenal (.%03d core fit)" % round(fit * 1000), "orange")
    rf = row.get("recentForm")
    if rf and rf.get("profile"):
        p = RF.PROFILES[rf["profile"]]
        add("profile_" + rf["profile"],
            "%s %s (L10: %.0f%% barrel, %d HR + %d near)" % (
                p["emoji"], p["label"], rf["barrelPct"], rf["hr"], rf["nearHr"]),
            "green")

    # Bat speed, slate-relative (see project_side()'s batSpeedPctl). Genuinely
    # new evidence category vs everything else here -- swing mechanics, not a
    # repackaging of any outcome-based signal already tracked. UNPROVEN, same
    # footing as every other reference-only angle until measure_signals.py
    # says otherwise.
    bsq = row.get("batSpeedPctl")
    bs = row.get("batSpeed")
    if bsq is not None and bs:
        if bsq >= 0.85:
            add("bat_speed_hot", "Bat speed top %d%% of slate (%.1f mph L10)"
                % (round((1 - bsq) * 100), bs["avgBatSpeed"]), "green")
        elif bsq <= 0.15:
            add("bat_speed_cold", "Bat speed bottom %d%% of slate (%.1f mph L10)"
                % (round(bsq * 100), bs["avgBatSpeed"]), "orange")

    if opp:
        for d in (opp.get("mixDrift") or [])[:2]:
            add("mix_drift", "SP mix shift: %s %+.0fpp vs season" % (d["pitch"], d["delta"]), "cyan")
        for cpitch in (opp.get("crushedPitches") or [])[:2]:
            bits = []
            if cpitch.get("xSlg") is not None:
                bits.append("xSLG %.3f" % cpitch["xSlg"])
            if cpitch.get("hr"):
                bits.append("%d HR" % cpitch["hr"])
            add("pitch_crushed", "%s getting crushed L%d (%s)" % (
                cpitch["pitch"], (opp.get("recentStarts") or RF.RECENT_STARTS), ", ".join(bits) or "recent"),
                "green")
        # Pitch shape decay -- earlier signal than pitch_crushed's outcome-
        # based read: velocity/spin measured on EVERY pitch of that type, not
        # just the ones that got hit, so this can fire before results catch
        # up. UNVERIFIED (see fetch_pitcher_pitch_mix()'s avgVelo/avgSpin
        # comment) -- correctly produces nothing if the season baseline
        # columns weren't found, same as every other defensive lookup here.
        for decay in (opp.get("shapeDrift") or [])[:2]:
            bits = []
            if "veloDelta" in decay:
                bits.append("%.1f mph" % decay["veloDelta"])
            if "spinDelta" in decay:
                bits.append("%.0f rpm" % decay["spinDelta"])
            add("pitch_decay", "%s losing shape L%d (%s)" % (
                decay["pitch"], (opp.get("recentStarts") or RF.RECENT_STARTS), ", ".join(bits) or "recent"),
                "orange")
    q = row.get("oppHr9Pctl")
    if q is not None and opp and opp.get("hrPer9") is not None:
        if q >= 0.75:
            add("sp_hr9_vuln", "HR-vulnerable arm: %.2f HR/9, top %d%% of slate"
                % (opp["hrPer9"], round((1 - q) * 100)), "green")
        elif q <= 0.25:
            add("sp_hr9_stingy", "Stingy arm: %.2f HR/9, bottom %d%% of slate"
                % (opp["hrPer9"], round(q * 100)), "orange")
    bq = row.get("oppBullpenPctl")
    if bq is not None and bq >= 0.75 and row.get("oppBullpenL3"):
        add("bullpen_taxed", "Opposing bullpen heavily used lately (%d relief pitches, top %d%% of slate)"
            % (row["oppBullpenL3"], round((1 - bq) * 100)), "green")

    vs_pitch = (row.get("metrics") or {}).get("vsPitch")
    pitch_mix = (opp or {}).get("pitchMix")
    wins, core_total = arsenal_overlap_count(vs_pitch, pitch_mix)
    if core_total and wins >= 2:
        add("arsenal_overlap", "Hits %d of his %d core pitches (.300 BA) well"
            % (wins, core_total), "green")

    wm, wd = row.get("windMph"), row.get("windDir")
    if wm is not None and wm >= 8:
        if wd == "out":
            add("wind_out", "Wind blowing out %.0f mph" % wm, "green")
        elif wd == "in":
            add("wind_in", "Wind blowing in %.0f mph" % wm, "orange")

    if row.get("restFlag") == "day_after_night":
        add("day_after_night", "Day game after a night game", "orange")

    hxs = row.get("heartXslg")
    sxs = (row.get("metrics") or {}).get("xSLG")
    if hxs is not None and sxs is not None:
        gap = hxs - sxs
        if gap >= 0.070:
            add("heart_masher", "Heart-zone xSLG runs %.3f above his season number -- "
                "concentrated damage on pitches down the middle" % gap, "green")
        elif gap <= -0.070:
            add("heart_soft", "Heart-zone xSLG runs %.3f BELOW his season number -- "
                "doesn't punish mistakes as much as the overall line suggests" % -gap, "orange")

    zg = row.get("zoneGrade")
    n_strong = len(row.get("zoneStrong") or []) or 1
    if zg == "elite":
        add("zone_elite", "Elite zone overlap -- SP lives in %d of his %d damage zones"
            % (row.get("zoneScore", 0), n_strong), "green")
    elif zg == "good":
        add("zone_good", "Good zone overlap -- SP lives in %d of his %d damage zones"
            % (row.get("zoneScore", 0), n_strong), "green")
    rfit, sfit = row.get("recentFitBA"), row.get("pitchFitBA")
    if rfit is not None and sfit is not None and rfit - sfit >= 0.050:
        add("recent_fit_gap", "Fit improves vs recent mix (.%03d vs .%03d)" % (
            round(rfit * 1000), round(sfit * 1000)), "cyan")
    return angles


def confidence(pa, bf, hit_q, hr_q):
    pa = pa or 0
    bf = bf or 0
    if hit_q == "full" and hr_q == "full" and pa >= 300 and bf >= 200:
        return "high"
    if pa >= 150 and bf >= 100 and "thin" not in (hit_q, hr_q):
        return "med"
    return "low"


def project_side(hitters, opp_pitcher, ctx, league, calibration, extras=None):
    out = []
    for h in hitters:
        hit = M.project_hit(h, opp_pitcher, ctx, league, calibration)
        hr = M.project_hr(h, opp_pitcher, ctx, league, calibration)
        sm = h.get("_savant") or {}
        fit_ba, fit_cov = pitch_fit(sm.get("vsPitch"),
                                    (opp_pitcher or {}).get("pitchMix"))
        core_fit_ba, _ = pitch_fit(sm.get("vsPitch"),
                                   RF.core_mix((opp_pitcher or {}).get("pitchMix")))
        recent_fit_ba, _ = pitch_fit(sm.get("vsPitch"),
                                     RF.core_mix((opp_pitcher or {}).get("recentMix")))
        ex = extras or {}
        rf = (ex.get("rfBatters") or {}).get(h["id"])
        if rf:
            rf = dict(rf)
            if rf.get("profile"):
                rf["profileEmoji"] = RF.PROFILES[rf["profile"]]["emoji"]

        bs = (ex.get("rfBatSpeed") or {}).get(h["id"])
        bat_speed_pctl = pctl_in(ex.get("slateBatSpeed") or [],
                                 bs["avgBatSpeed"] if bs else None)

        opp_team = (opp_pitcher or {}).get("teamAbbr")
        bp = (ex.get("bullpen") or {}).get(opp_team)
        bp_l3 = bp["reliefPitches"] if bp else None
        bp_pctl = pctl_in(ex.get("bullpenPool") or [], bp_l3)

        rest = SIT.parse_rest_travel(h["teamAbbr"], ex.get("flatGames") or [], ex.get("today"))
        rest_flag = None
        if rest:
            if rest["dayAfterNight"]:
                rest_flag = "day_after_night"
            elif rest["backToBack"]:
                rest_flag = "b2b"
        zone_score, zone_grade, strong_list = None, None, None
        zone_cache = ex.get("zoneCache")
        used = set((opp_pitcher or {}).get("_usedZones") or [])
        heart_xslg, heart_bbe = None, 0
        if zone_cache:
            strong = Z.strong_zones(zone_cache, h["id"])
            strong_list = sorted(strong) or None
            if used:
                zone_score, zone_grade = Z.score_matchup(strong, used)
            hz = Z.heart_zone_xslg(zone_cache, h["id"])
            heart_xslg, heart_bbe = hz["xSLG"], hz["bbe"]
        row = {
            "hitterId": h["id"], "name": h["name"], "pos": h["pos"],
            "teamAbbr": h["teamAbbr"], "batSide": h.get("batSide"),
            "hitProb": hit["perGame"], "hitProbPerPA": hit["perPA"], "hitTier": hit["tier"],
            "hitRawPerGame": hit["rawPerGame"],
            "hitDataQuality": hit["dataQuality"], "hitSignals": hit["signals"], "hitRisks": hit["risks"],
            "hitInputs": hit["inputs"],
            "hrProb": hr["perGame"], "hrProbPerPA": hr["perPA"], "hrTier": hr["tier"],
            "hrRawPerGame": hr["rawPerGame"],
            "hrDataQuality": hr["dataQuality"], "hrSignals": hr["signals"], "hrRisks": hr["risks"],
            "hrInputs": hr["inputs"],
            "expectedPA": hit["expectedPA"], "batOrderAvg": h.get("orderAvg"),
            "lineupUnconfirmed": True,
            "slotSource": ("projected" if h.get("_slot") else "default"),
            "slotLast": (h.get("_slot") or {}).get("lastSlot"),
            "slotGames": (h.get("_slot") or {}).get("games"),
            "slotStartPct": (h.get("_slot") or {}).get("startPct"),
            "pitchFitBA": fit_ba,
            "pitchFitCoverage": fit_cov,
            "coreFitBA": core_fit_ba,
            "recentFitBA": recent_fit_ba,
            "recentForm": rf,
            "batSpeed": bs,
            "batSpeedPctl": bat_speed_pctl,
            "zoneScore": zone_score,
            "zoneGrade": zone_grade,
            "zoneStrong": strong_list,
            "heartXslg": heart_xslg,
            "heartBBE": heart_bbe,
            "oppBullpenL3": bp_l3,
            "oppBullpenPctl": bp_pctl,
            "restFlag": rest_flag,
            "gamesL3": rest["gamesL3"] if rest else None,
            "windMph": ctx.get("windMph"),
            "windDir": ctx.get("windDir"),
            "confidence": confidence(h.get("pa"),
                                     (opp_pitcher or {}).get("battersFaced"),
                                     hit["dataQuality"], hr["dataQuality"]),
            "viewScore": 0,
            "metrics": {
                "pa": h.get("pa"),
                "avg": h.get("avg"), "obp": h.get("obp"), "hr": h.get("hr"),
                "hrRate": h.get("hrRate"),
                "kPct": h.get("kPct"), "babip": h.get("babip"),
                "barrelPct": sm.get("barrelPct"),
                "hardHitPct": sm.get("hardHitPct"),
                "avgEV": sm.get("avgEV"),
                "pullPct": (rf.get("pullPct") if rf else None),
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
        }
        row["oppHr9"] = (opp_pitcher or {}).get("hrPer9")
        row["oppHr9Pctl"] = pctl_in(ex.get("slateHr9") or [], row["oppHr9"])
        row["angles"] = compute_angles(row, opp_pitcher)
        out.append(row)
    return out


def apply_view_scores(all_rows):
    def zs(vals):
        n = len(vals)
        if n < 2:
            return [0.0] * n
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        sd = math.sqrt(var)
        if sd < 1e-9:
            return [0.0] * n
        return [(v - mean) / sd for v in vals]

    z_hit = zs([r["hitProb"] for r in all_rows])
    z_hr = zs([r["hrProb"] for r in all_rows])
    for r, zh, zr in zip(all_rows, z_hit, z_hr):
        r["viewScore"] = round(zh + zr, 3)


def fetch_recent_layer():
    rf_batters, rf_pitchers, rf_df, rf_slots, bullpen, rf_batspeed = {}, {}, None, {}, {}, {}
    try:
        rf_df = RF.fetch_statcast()
    except Exception as e:
        warn(f"statcast pull failed ({type(e).__name__}: {e}) -- recent-form layer skipped")
        rf_df = None
    if rf_df is not None:
        for label, fn, sink in (
            ("batter form", RF.batter_form, "batters"),
            ("pitcher recent", RF.pitcher_recent, "pitchers"),
            ("lineup slots", RF.lineup_slots, "slots"),
            ("bullpen fatigue", SIT.bullpen_fatigue, "bullpen"),
            # Same pull, same isolated-failure-domain discipline as the four
            # above (one bad column here can't take the others down) -- see
            # recent_form.bat_speed_form()'s own docstring for why this is
            # NOT restricted to the batter_form() BBE-only rows.
            ("bat speed", RF.bat_speed_form, "batSpeed"),
        ):
            try:
                result = fn(rf_df)
            except Exception as e:
                warn(f"{label} failed ({type(e).__name__}: {e}) -- other recent-form parts unaffected")
                continue
            if not result:
                warn(f"{label} returned no data (no exception) -- check its input assumptions")
            if sink == "batters":
                rf_batters = result
            elif sink == "pitchers":
                rf_pitchers = result
            elif sink == "slots":
                rf_slots = result
            elif sink == "batSpeed":
                rf_batspeed = result
            else:
                bullpen = result
        HEALTH["recentFormOk"] = bool(rf_batters)
        HEALTH["batSpeedOk"] = bool(rf_batspeed)
        HEALTH["lineupSlotsOk"] = bool(rf_slots)
        HEALTH["lineupSlotCount"] = len(rf_slots)
    zone_cache = None
    try:
        zone_cache = Z.load_cache()
        if zone_cache:
            HEALTH["zoneCacheOk"] = True
            HEALTH["zoneCacheAsOf"] = zone_cache.get("asOf")
        else:
            warn("zones_cache.json missing -- run zone_engine.py (see --backfill)")
    except Exception as e:
        warn(f"zone cache load failed ({type(e).__name__})")
    # Same load/health pattern as zone_cache immediately above -- season pitch-
    # shape baseline, replacing the arsenal-stats-based season_shape build
    # that enrich_probable() used to construct (confirmed dead: the arsenal-
    # stats leaderboard fetch_pitcher_pitch_mix() pulls has no velocity/spin
    # columns at all, so avgVelo/avgSpin never populated there -- see that
    # function's now-removed extraction block).
    shape_cache = None
    try:
        shape_cache = PSC.load_cache()
        if shape_cache:
            HEALTH["shapeCacheOk"] = True
            HEALTH["shapeCacheAsOf"] = shape_cache.get("asOf")
        else:
            warn("pitch_shape_cache.json missing -- run pitch_shape_cache.py (see --backfill)")
    except Exception as e:
        warn(f"pitch shape cache load failed ({type(e).__name__})")
    return rf_batters, rf_pitchers, rf_df, zone_cache, rf_slots, bullpen, rf_batspeed, shape_cache


def enrich_probable(pd_dict, pid, rf_pitchers, rf_df, shape_cache=None):
    rec = rf_pitchers.get(pid)
    if not rec:
        return
    pd_dict["recentMix"] = rec["mix"]
    pd_dict["recentStarts"] = rec["starts"]
    drifts, crushed = RF.mix_drift(rec, pd_dict.get("pitchMix"))
    pd_dict["mixDrift"] = drifts
    pd_dict["crushedPitches"] = crushed
    # Season shape baseline now comes from pitch_shape_cache.py, keyed
    # directly by this pitcher's MLBAM id (pid) -- confirmed real, ledger-
    # backed source, replacing the arsenal-stats-based build that used to
    # sit here. That old version read avgVelo/avgSpin off pitchMix entries
    # populated by fetch_pitcher_pitch_mix(), but that leaderboard has no
    # velocity/spin columns at all (confirmed via a live Savant column
    # check), so season_shape was silently empty every single run -- this
    # is a real fix, not a refactor. Keyed by id rather than name also drops
    # a whole class of name-matching risk the old pitchMix join carried.
    season_shape = PSC.pitcher_shape_baseline(shape_cache, pid) if shape_cache else {}
    pd_dict["shapeDrift"] = RF.pitch_shape_drift(rec, season_shape)
    try:
        gpks = rec.get("gamePks") or []
        pd_dict["_usedZones"] = sorted(Z.pitcher_used_zones(rf_df, pid, gpks))
        pd_dict["_zoneShares"] = Z.zone_shares(rf_df, pid, gpks)
    except Exception:
        pd_dict["_usedZones"] = []
        pd_dict["_zoneShares"] = {}


def build_board():
    pool = build_batter_pool()
    savant = fetch_savant_metrics()
    pitch_mix = fetch_pitcher_pitch_mix()
    rf_batters, rf_pitchers, rf_df, zone_cache, rf_slots, bullpen_raw, rf_batspeed, shape_cache = fetch_recent_layer()
    bullpen = {}
    for k, v in (bullpen_raw or {}).items():
        code = norm_team(k)
        if code:
            bullpen[code] = v
    bullpen_pool = sorted(v["reliefPitches"] for v in bullpen.values()) if bullpen else []

    flat_games = []
    try:
        sched_range = fetch_schedule_range()
        flat_games = flatten_schedule_range(sched_range)
    except Exception as e:
        warn(f"schedule-range fetch failed ({type(e).__name__}: {e}) -- rest/travel skipped")
    if not flat_games:
        warn("flatten_schedule_range returned no data -- rest/travel will be null on every row")
    HEALTH["bullpenFatigueOk"] = bool(bullpen)
    HEALTH["restTravelOk"] = bool(flat_games)

    # Slate-relative bat speed pool -- same self-calibrating pattern as
    # slateHr9 below: "fast bat speed" only means something relative to
    # tonight's actual pool, not a fixed league number nobody's confirmed.
    slate_bat_speed = sorted(v["avgBatSpeed"] for v in rf_batspeed.values()
                             if v.get("avgBatSpeed") is not None)

    extras = {"rfBatters": rf_batters, "rfBatSpeed": rf_batspeed, "zoneCache": zone_cache,
              "bullpen": bullpen, "bullpenPool": bullpen_pool,
              "slateBatSpeed": slate_bat_speed, "flatGames": flat_games, "today": TODAY}
    for h in pool:
        h["_savant"] = savant.get(_norm_name(h.get("name")), {})
        sl = rf_slots.get(h["id"])
        if sl:
            h["orderAvg"] = sl["orderAvg"]
            h["_slot"] = sl
    league = M.league_rates(pool)
    calibration = load_calibration()
    games = get_schedule()

    slate_hr9 = []
    for g in games:
        for side in ("home", "away"):
            prob = (g.get("teams", {}).get(side, {}) or {}).get("probablePitcher")
            if prob and prob.get("id"):
                rec = get_pitcher_rates(prob["id"])
                v = nv((rec or {}).get("hrPer9"))
                if v is not None:
                    slate_hr9.append(v)
    slate_hr9.sort()
    extras["slateHr9"] = slate_hr9

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
            hp = dict(hp)
            hp["name"] = home_prob.get("fullName")
            hp["teamAbbr"] = home_abbr
            hp["pitchMix"] = pitch_mix.get(_norm_name(hp["name"]))
            enrich_probable(hp, home_prob.get("id"), rf_pitchers, rf_df, shape_cache)
        if ap and away_prob:
            ap = dict(ap)
            ap["name"] = away_prob.get("fullName")
            ap["teamAbbr"] = away_abbr
            ap["pitchMix"] = pitch_mix.get(_norm_name(ap["name"]))
            enrich_probable(ap, away_prob.get("id"), rf_pitchers, rf_df, shape_cache)

        weather = SIT.parse_weather(g.get("weather"))
        ctx = {"park": park, "weather": {"temp": weather["tempF"]},
               "windMph": weather["windMph"], "windDir": weather["windDir"]}
        umpire = SIT.home_plate_umpire(g.get("officials"))

        home_hitters = project_side(by_team.get(home_abbr, []), ap, ctx, league, calibration, extras)
        away_hitters = project_side(by_team.get(away_abbr, []), hp, ctx, league, calibration, extras)

        def slim_probable(pd):
            if not pd:
                return None
            return {"name": pd["name"], "hand": pd.get("hand"), "whip": pd.get("whip"),
                    "hrPer9": pd.get("hrPer9"), "kPct": pd.get("kPct"),
                    "pitchMix": pd.get("pitchMix"),
                    "recentMix": pd.get("recentMix"),
                    "recentStarts": pd.get("recentStarts"),
                    "mixDrift": pd.get("mixDrift"),
                    "crushedPitches": pd.get("crushedPitches"),
                    "shapeDrift": pd.get("shapeDrift"),
                    "usedZones": pd.get("_usedZones"),
                    "zoneShares": pd.get("_zoneShares")}

        merged.append({
            "gameId": g.get("gamePk"),
            "gameTime": g.get("gameDate"),
            "venue": (g.get("venue") or {}).get("name"),
            "venueParkFactor": park,
            "weather": {"tempF": weather["tempF"], "windMph": weather["windMph"],
                       "windDir": weather["windDir"], "condition": weather["condition"]},
            "homeplateUmpire": umpire,
            "homeTeam": {"name": home.get("name"), "abbr": home_abbr},
            "awayTeam": {"name": away.get("name"), "abbr": away_abbr},
            "homeProbable": slim_probable(hp),
            "awayProbable": slim_probable(ap),
            "homeMatchups": home_hitters,
            "awayMatchups": away_hitters,
        })

    all_rows = []
    for gm in merged:
        all_rows.extend(gm["homeMatchups"])
        all_rows.extend(gm["awayMatchups"])
    apply_view_scores(all_rows)
    for gm in merged:
        gm["homeMatchups"].sort(key=lambda x: x["viewScore"], reverse=True)
        gm["awayMatchups"].sort(key=lambda x: x["viewScore"], reverse=True)

    return {
        "schemaVersion": BOARD_SCHEMA_VERSION,
        "modelVersion": M.MODEL_VERSION,
        "builtAt": TODAY,
        "builtTs": datetime.datetime.now(ET).isoformat(),
        "gamesCount": len(merged),
        "calibrationApplied": bool(calibration),
        "poolSize": len(pool),
        "leagueRates": {
            "hitRatePerPA": round(league["hitRatePerPA"], 4),
            "hrRatePerPA": round(league["hrRatePerPA"], 4),
        },
        "dataHealth": HEALTH,
        "games": merged,
    }


def atomic_write_json(path, obj):
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
