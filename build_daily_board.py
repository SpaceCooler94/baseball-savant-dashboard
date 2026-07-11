#!/usr/bin/env python3
# ============================================================================
# build_daily_board.py -- Part 2 of the model port (the data layer).
#
# Runs OFF the phone (on GitHub Actions). Pulls Savant hitter data + today's
# schedule/probable pitchers + pitcher rate stats, feeds them into the proven
# math in mlb_model.py, and writes daily_board.json in the exact shape
# MLB_Daily.js renders. Render serves that file; the phone just displays it.
#
# HONESTY NOTE: unlike mlb_model.py (which I could prove identical to the JS in a
# sandbox), this file hits LIVE data sources. I can build it, guard it, and
# dry-run its shape, but its real data output must be confirmed by an actual run
# (locally or in the GitHub Action). Savant column names in particular drift
# between library versions -- every column access here goes through find_col with
# multiple candidate names so a rename degrades gracefully instead of KeyError-ing.
#
# Reuses the helper patterns from the repo's app.py (find_col / rename_if_exists /
# df_to_records) so behavior is consistent with the dashboard already in production.
# ============================================================================

import datetime
import json
import math
import sys
import time

import requests

import mlb_model as M


YEAR = datetime.datetime.now().year
STATS_API = "https://statsapi.mlb.com/api/v1"

# Park factors + stadium coords, ported verbatim from Daily_Matchups.js constants.
PARK_FACTORS = {
    "COL": 122, "CIN": 108, "BOS": 108, "PHI": 107, "NYY": 106, "TEX": 105,
    "HOU": 104, "ATL": 104, "MIL": 103, "ARI": 103, "WSH": 102, "CHC": 102,
    "NYM": 101, "DET": 101, "MIN": 101, "LAD": 100, "STL": 100, "TOR": 100,
    "CLE": 99, "BAL": 99, "SEA": 99, "PIT": 98, "CHW": 98, "MIA": 98,
    "KCR": 97, "TBR": 97, "LAA": 97, "OAK": 116, "SDP": 94, "SFG": 93,
}

# Full team-abbreviation normalization, ported from the JS TEAM_MAP.
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

# StatsAPI returns full team names; map those to our abbreviations.
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


def norm_team(abbr):
    if not abbr:
        return None
    return TEAM_MAP.get(str(abbr).upper(), str(abbr).upper())


# ---- helper functions mirrored from app.py so behavior matches the dashboard ----

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def nv(v):
    """Same null-safe numeric coercion as mlb_model._nv / the JS nv()."""
    if v is None or v == "":
        return None
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def http_json(url, tries=2):
    """GET with one retry -- mirrors the pacing/retry discipline from the JS side."""
    last = None
    for attempt in range(tries):
        try:
            r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            if attempt < tries - 1:
                time.sleep(2)
    raise last


# --------------------------- batter data (Savant) ---------------------------

def build_batter_pool():
    """Pull the full qualified-hitter pool and shape it into the record format
    mlb_model expects (obp/bbPct/hrRate/avg/vsLAvg/vsRAvg/etc).

    Batter base = StatsAPI season hitting stats (obp/avg/hr/pa/bb) plus vl/vr platoon
    splits (which feed the model's hit projection). Savant advanced metrics (barrel%,
    xBA, xSLG, etc.) are merged on top via merge_savant_metrics() for the detail view.

    Still null in this version: recent-form OPS (l5/l10, would need per-hitter gameLog)
    and lineup slot (unknown until lineups post). The model handles both as null safely.
    """
    # StatsAPI season hitting leaders -- gives obp, avg, hr, pa, bb, so, per team roster.
    # We pull by team so we also get team affiliation for the schedule join.
    teams = http_json(f"{STATS_API}/teams?sportId=1&season={YEAR}").get("teams", [])
    pool = []
    for t in teams:
        tid = t.get("id")
        tabbr = norm_team(t.get("abbreviation"))
        if not tid:
            continue
        try:
            roster = http_json(f"{STATS_API}/teams/{tid}/roster?rosterType=active&season={YEAR}")
        except Exception:
            continue
        for entry in roster.get("roster", []):
            person = entry.get("person", {})
            pid = person.get("id")
            pos = entry.get("position", {}).get("abbreviation")
            if not pid or pos == "P":  # skip pitchers for the hitter pool
                continue
            try:
                stats = http_json(
                    f"{STATS_API}/people/{pid}/stats?stats=season&group=hitting&season={YEAR}"
                )
            except Exception:
                continue
            splits = (stats.get("stats") or [{}])[0].get("splits") or []
            if not splits:
                continue
            st = splits[0].get("stat", {})
            pa = nv(st.get("plateAppearances"))
            if pa is None or pa < 25:  # same MIN_PA floor as the model's pool
                continue
            # Platoon splits: one extra call, but fetched via statSplits with vl,vr in a
            # single request (mirrors the old Rankings.js approach). vsLAvg/vsRAvg feed
            # directly into the model's hit projection. Best-effort -- if the split call
            # fails, we keep the hitter with null splits (model falls back to season rate,
            # flagged partial) rather than dropping them.
            vs_l_avg, vs_r_avg = None, None
            try:
                sp = http_json(
                    f"{STATS_API}/people/{pid}/stats?stats=statSplits&group=hitting"
                    f"&season={YEAR}&sitCodes=vl,vr"
                )
                for s in (sp.get("stats") or [{}])[0].get("splits") or []:
                    code = str((s.get("split") or {}).get("code") or "").lower()
                    sst = s.get("stat", {})
                    if code == "vl":
                        vs_l_avg = nv(sst.get("avg"))
                    elif code == "vr":
                        vs_r_avg = nv(sst.get("avg"))
            except Exception:
                pass
            pool.append({
                "id": pid,
                "name": person.get("fullName"),
                "teamAbbr": tabbr,
                "pos": pos,
                "batSide": (person.get("batSide") or {}).get("code"),
                "pa": pa,
                "obp": nv(st.get("obp")),
                "avg": nv(st.get("avg")),
                "bbPct": (nv(st.get("baseOnBalls")) / pa * 100) if (nv(st.get("baseOnBalls")) is not None and pa) else None,
                "hr": nv(st.get("homeRuns")),
                "hrRate": (nv(st.get("homeRuns")) / pa * 100) if (nv(st.get("homeRuns")) is not None and pa) else None,
                "vsLAvg": vs_l_avg,
                "vsRAvg": vs_r_avg,
                # Recent-form OPS still null in this version -- would need gameLog per
                # hitter (another call each). Model treats null recent-form safely.
                "l5Ops": None, "l10Ops": None,
                "orderAvg": None,  # lineup slot not known until lineups post; null -> model uses whole-lineup PA
            })
    return pool


# --------------------------- Savant metrics (bulk) --------------------------

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


def fetch_savant_metrics():
    """Pull barrel%, hard-hit%, xBA, xSLG, etc. in BULK (one pybaseball call each, not
    per-player) and return a dict keyed by normalized name. Best-effort: any endpoint
    that fails is skipped, and every column access goes through find_col so a Savant
    rename degrades to null rather than crashing. Returns {} on total failure so the
    board still builds on rate stats alone."""
    metrics = {}
    try:
        from pybaseball import statcast_batter_exitvelo_barrels, statcast_batter_expected_stats
    except Exception:
        return metrics

    def col(df, cands):
        return find_col(df, cands)

    # Exit velo / barrels
    try:
        ev = statcast_batter_exitvelo_barrels(YEAR, minBBE=25)
        name_c = col(ev, ["last_name, first_name", "player_name", "name"])
        brl_c = col(ev, ["barrel_batted_rate", "brl_percent"])
        hh_c = col(ev, ["hard_hit_percent", "hard_hit_rate"])
        ev_c = col(ev, ["avg_hit_speed", "exit_velocity_avg", "avg_hit_velo"])
        if name_c:
            for _, row in ev.iterrows():
                k = _norm_name(row.get(name_c))
                if not k:
                    continue
                m = metrics.setdefault(k, {})
                if brl_c: m["barrelPct"] = nv(row.get(brl_c))
                if hh_c: m["hardHitPct"] = nv(row.get(hh_c))
                if ev_c: m["avgEV"] = nv(row.get(ev_c))
    except Exception:
        pass

    # Expected stats (xBA, xSLG, xwOBA)
    try:
        xs = statcast_batter_expected_stats(YEAR, minPA=25)
        name_c = col(xs, ["last_name, first_name", "player_name", "name"])
        xba_c = col(xs, ["est_ba", "xba", "expected_batting_avg"])
        xslg_c = col(xs, ["est_slg", "xslg", "expected_slg"])
        xwoba_c = col(xs, ["est_woba", "xwoba", "expected_woba"])
        if name_c:
            for _, row in xs.iterrows():
                k = _norm_name(row.get(name_c))
                if not k:
                    continue
                m = metrics.setdefault(k, {})
                if xba_c: m["xBA"] = nv(row.get(xba_c))
                if xslg_c: m["xSLG"] = nv(row.get(xslg_c))
                if xwoba_c: m["xwOBA"] = nv(row.get(xwoba_c))
    except Exception:
        pass

    return metrics


# --------------------------- schedule + pitchers ----------------------------

def get_pitcher_rates(pid):
    """Season hit-rate-allowed and HR-rate-allowed per batter faced, from StatsAPI.
    Mirrors the fields buildPitcherProfile() computes in the JS."""
    try:
        stats = http_json(
            f"{STATS_API}/people/{pid}/stats?stats=season&group=pitching&season={YEAR}"
        )
    except Exception:
        return None
    splits = (stats.get("stats") or [{}])[0].get("splits") or []
    if not splits:
        return None
    st = splits[0].get("stat", {})
    bf = nv(st.get("battersFaced"))
    if not bf:
        return None
    obp = nv(st.get("obp"))
    bb = nv(st.get("baseOnBalls"))
    hr = nv(st.get("homeRuns"))
    bb_pct = (bb / bf) if bb is not None else None
    hit_rate_allowed = max(0.01, obp - bb_pct) if (obp is not None and bb_pct is not None) else None
    hr_rate_allowed = (hr / bf) if hr is not None else None
    hand = (
        http_json(f"{STATS_API}/people/{pid}")
        .get("people", [{}])[0]
        .get("pitchHand", {})
        .get("code")
    )
    return {
        "hand": hand,
        "hitRateAllowedPerPA": hit_rate_allowed,
        "hrRateAllowedPerPA": hr_rate_allowed,
        "battersFaced": bf,
    }


def get_schedule():
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    data = http_json(
        f"{STATS_API}/schedule?sportId=1&date={today}"
        "&hydrate=probablePitcher(note),team,venue,linescore"
    )
    games = []
    for date in data.get("dates", []):
        games.extend(date.get("games", []))
    return games, today


# ------------------------------- assembly -----------------------------------

def load_calibration():
    """Best-effort read of a committed calibration.json (same file the Render route
    serves). None -> model runs uncalibrated. Version-gated like the JS side."""
    try:
        with open("calibration.json") as f:
            cal = json.load(f)
        if isinstance(cal, dict) and cal.get("modelVersion") == "log5-v5.0":
            return cal
    except Exception:
        pass
    return None


def build_board():
    pool = build_batter_pool()
    # Merge Savant advanced metrics (bulk fetch, name-joined) onto each hitter.
    savant = fetch_savant_metrics()
    for h in pool:
        sm = savant.get(_norm_name(h.get("name")), {})
        h["_savant"] = sm  # stashed for the display metrics block below
    league = M.league_rates(pool)
    calibration = load_calibration()
    games, today = get_schedule()

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
            hp["name"] = home_prob.get("fullName")
        if ap and away_prob:
            ap["name"] = away_prob.get("fullName")

        ctx = {"park": park, "weather": {}}  # weather omitted in v1; model treats temp=None safely

        def project_side(hitters, opp_pitcher):
            out = []
            for h in hitters:
                hit = M.project_hit(h, opp_pitcher, ctx, league, calibration)
                hr = M.project_hr(h, opp_pitcher, ctx, league, calibration)
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
                        "avg": nv(h.get("avg")), "obp": nv(h.get("obp")), "hr": nv(h.get("hr")),
                        "hrRate": nv(h.get("hrRate")), "kPct": None, "babip": None,
                        "barrelPct": (h.get("_savant") or {}).get("barrelPct"),
                        "hardHitPct": (h.get("_savant") or {}).get("hardHitPct"),
                        "avgEV": (h.get("_savant") or {}).get("avgEV"),
                        "pullPct": None, "fbPct": None,
                        "xBA": (h.get("_savant") or {}).get("xBA"),
                        "xSLG": (h.get("_savant") or {}).get("xSLG"),
                        "xwOBA": (h.get("_savant") or {}).get("xwOBA"),
                        "sweetSpotPct": None,
                    },
                })
            out.sort(key=lambda x: x["viewScore"], reverse=True)
            return out

        home_hitters = project_side(by_team.get(home_abbr, []), ap)
        away_hitters = project_side(by_team.get(away_abbr, []), hp)
        all_h = home_hitters + away_hitters

        merged.append({
            "gameId": g.get("gamePk"),
            "gameTime": g.get("gameDate"),
            "venue": (g.get("venue") or {}).get("name"),
            "venueParkFactor": park,
            "weather": {},
            "homeTeam": {"name": home.get("name"), "abbr": home_abbr},
            "awayTeam": {"name": away.get("name"), "abbr": away_abbr},
            "homeProbable": {"name": hp["name"], "hand": hp.get("hand")} if hp else None,
            "awayProbable": {"name": ap["name"], "hand": ap.get("hand")} if ap else None,
            "homeMatchups": home_hitters,
            "awayMatchups": away_hitters,
            "topHitTargets": sorted(all_h, key=lambda x: x["hitProb"], reverse=True)[:8],
            "topHrTargets": sorted(all_h, key=lambda x: x["hrProb"], reverse=True)[:8],
            "topOverall": sorted(all_h, key=lambda x: x["viewScore"], reverse=True)[:8],
        })

    board = {
        "schemaVersion": 5.1,
        "builtAt": today,
        "builtTs": datetime.datetime.now().isoformat(),
        "gamesCount": len(merged),
        "calibrationApplied": bool(calibration),
        "poolSize": len(pool),
        "leagueRates": {
            "hitRatePerPA": round(league["hitRatePerPA"], 4),
            "hrRatePerPA": round(league["hrRatePerPA"], 4),
        },
        "games": merged,
    }
    return board


def main():
    board = build_board()
    with open("daily_board.json", "w") as f:
        json.dump(board, f)
    print(f"Wrote daily_board.json: {board['gamesCount']} games, "
          f"{board['poolSize']} hitters in pool, "
          f"calibrated={board['calibrationApplied']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"BUILD FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
