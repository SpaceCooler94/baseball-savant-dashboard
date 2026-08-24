"""
Flask front for the baseball-savant-dashboard.

Serves two things that matter: /api/daily-board (what MLB_Daily.js fetches) and
a set of Savant leaderboard endpoints for the browser dashboard. Everything
heavy is cached; nothing here computes the model.

REVIEW FIXES APPLIED (2026-08-09) -- see block comments at each site:
  1. Cache key built from unvalidated user input -> unbounded memory growth.
  2. Exception text returned to clients -> information disclosure.
  3. /api/columns ran an uncached full-season pull on every request.
  4. MODEL_VERSION had drifted from mlb_model.py, silently disabling
     /api/calibration.
  5. Naive datetime.now() -> UTC on Render, so "today" flipped a day early.
  6. No cache eviction or size bound.
"""

from flask import Flask, render_template, jsonify, request
import datetime
import logging
import math
import os
import json
import threading
import time
from zoneinfo import ZoneInfo

import requests

app = Flask(__name__)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TIME. Render runs UTC. datetime.now() with no tzinfo therefore rolls over to
# "tomorrow" at 8pm ET, which made /api/slate fetch the wrong day's schedule
# every evening -- the same class of bug already fixed in build_daily_board.py.
# Every date in this file is Eastern, because that is what a baseball slate is.
# ---------------------------------------------------------------------------
ET = ZoneInfo("America/New_York")


def today_et():
    return datetime.datetime.now(ET)


# ---------------------------------------------------------------------------
# CACHE. Two changes from the original dict:
#
#   BOUNDED. The old cache was keyed partly by user input (slate_{team}) with no
#   size limit and no eviction, so `GET /api/slate?team=<random>` in a loop grew
#   the process until the 512MB free-tier container was killed -- and every miss
#   also fired a live StatsAPI request, so the same loop amplified into upstream
#   traffic. Keys are validated below AND the cache is capped with LRU eviction.
#
#   THREAD-SAFE. gunicorn's default sync worker is single-threaded per process,
#   but a lock costs nothing here and stops a torn read if the worker class ever
#   changes. Note each worker still holds its own cache: that's fine for
#   read-only leaderboard data, just don't treat it as shared state.
# ---------------------------------------------------------------------------
CACHE_TTL = 3600
BOARD_CACHE_TTL = 300
MAX_CACHE_ENTRIES = 64

_cache = {}
_cache_lock = threading.Lock()


def get_cached(key, fetch_fn, ttl=CACHE_TTL):
    now = time.time()
    with _cache_lock:
        entry = _cache.get(key)
        if entry and now - entry[1] < ttl:
            _cache[key] = (entry[0], entry[1])
            return entry[0]
    data = fetch_fn()
    with _cache_lock:
        if len(_cache) >= MAX_CACHE_ENTRIES:
            for stale_key, (_, ts) in sorted(_cache.items(), key=lambda kv: kv[1][1])[:8]:
                _cache.pop(stale_key, None)
        _cache[key] = (data, now)
    return data


# ---------------------------------------------------------------------------
# ERROR HANDLING. Every route used to `return jsonify({"message": str(e)}), 500`,
# which hands a stranger the exception text: absolute filesystem paths, pandas
# and pybaseball internals, and upstream URLs. The trace goes to the server log
# where it's useful; the client gets a generic message and nothing else.
# ---------------------------------------------------------------------------
@app.after_request
def _security_headers(resp):
    """Baseline hardening. The board is public read-only data, so CORS stays
    open for GET, but nothing here should ever be framed or content-sniffed,
    and referrers shouldn't leak the Render hostname onward."""
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    resp.headers.setdefault("Access-Control-Allow-Origin", "*")
    resp.headers.setdefault("Access-Control-Allow-Methods", "GET, OPTIONS")
    return resp


def fail(route, exc, status=500):
    log.exception("%s failed", route)
    return jsonify({"status": "error", "message": "upstream data unavailable"}), status


# Imported, not restated. This constant drifted once already -- app.py sat on
# "log5-v5.0" while mlb_model.py moved to v5.4, and because load_calibration_file
# gates on equality, /api/calibration returned None on every request for weeks
# while looking perfectly healthy. The previous fix corrected the literal and
# left a comment asking a human to keep the two in sync; that is the same
# control that already failed once. Importing makes drift structurally
# impossible. Falls back to a literal ONLY if the model module isn't importable
# (e.g. a docs build), and fails closed by logging loudly.
try:
    from mlb_model import MODEL_VERSION
except ImportError:  # pragma: no cover
    MODEL_VERSION = "log5-v5.4"
    log.warning("mlb_model not importable -- MODEL_VERSION falling back to a "
                "literal, which can drift. Calibration may silently disable.")
CALIBRATION_PATH = os.path.join(os.path.dirname(__file__), "calibration.json")
DAILY_BOARD_PATH = os.path.join(os.path.dirname(__file__), "daily_board.json")


def load_calibration_file():
    """Serves the committed calibration.json.

    VERSION GATE, AND WHY IT MATTERS: this constant is compared against the
    modelVersion inside calibration.json. It had drifted to log5-v5.0 while
    mlb_model.py moved to log5-v5.4, so this route silently returned None on
    every request -- a dead endpoint that looked healthy. Keep MODEL_VERSION
    here in lockstep with mlb_model.MODEL_VERSION on every model bump. (Better
    still, import it -- but app.py deliberately avoids importing the model so
    the web dyno doesn't need pandas at boot.)

    Render's filesystem is ephemeral: never write this file at runtime and
    expect it to survive. calibrate.py commits it via the Action.
    """
    if not os.path.exists(CALIBRATION_PATH):
        return None
    try:
        with open(CALIBRATION_PATH, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        log.warning("calibration.json unreadable")
        return None
    if not isinstance(data, dict) or data.get("modelVersion") != MODEL_VERSION:
        log.warning("calibration.json modelVersion=%s, expected %s -- ignored",
                    (data or {}).get("modelVersion"), MODEL_VERSION)
        return None
    return data


def _format_game_time(iso_str):
    """'2026-08-23T17:35:00Z' -> '1:35 PM ET' for the browser dashboard."""
    if not iso_str:
        return "TBD"
    try:
        dt = datetime.datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return dt.astimezone(ET).strftime("%-I:%M %p ET")
    except (ValueError, TypeError):
        return "TBD"


def load_daily_board():
    if not os.path.exists(DAILY_BOARD_PATH):
        return None
    try:
        with open(DAILY_BOARD_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        log.warning("daily_board.json unreadable")
        return None


# ------------------------------ dataframe utils -----------------------------

def df_to_records(df):
    cleaned = []
    for row in df.to_dict(orient="records"):
        clean = {}
        for k, v in row.items():
            # pandas nullable dtypes return pd.NA, whose truthiness raises.
            # float()/isnan() on NA raises TypeError -- caught here. This is the
            # same hazard that took down recent_form.py's whole layer.
            try:
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    clean[k] = None
                    continue
            except (TypeError, ValueError):
                clean[k] = None
                continue
            clean[k] = v
        cleaned.append(clean)
    return cleaned


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def rename_if_exists(df, mapping):
    df.rename(columns={k: v for k, v in mapping.items() if k in df.columns}, inplace=True)
    return df


# --------------------------------- teams ------------------------------------

TEAM_ALIASES = {
    "LAA": ["LAA", "ANGELS", "LOS ANGELES ANGELS", "LA ANGELS"],
    "ARI": ["ARI", "DIAMONDBACKS", "ARIZONA DIAMONDBACKS", "D-BACKS"],
    "BAL": ["BAL", "ORIOLES", "BALTIMORE ORIOLES"],
    "BOS": ["BOS", "RED SOX", "BOSTON RED SOX"],
    "CHC": ["CHC", "CUBS", "CHICAGO CUBS"],
    "CIN": ["CIN", "REDS", "CINCINNATI REDS"],
    "CLE": ["CLE", "GUARDIANS", "CLEVELAND GUARDIANS"],
    "COL": ["COL", "ROCKIES", "COLORADO ROCKIES"],
    "DET": ["DET", "TIGERS", "DETROIT TIGERS"],
    "HOU": ["HOU", "ASTROS", "HOUSTON ASTROS"],
    "KC": ["KC", "ROYALS", "KANSAS CITY ROYALS"],
    "LAD": ["LAD", "DODGERS", "LOS ANGELES DODGERS"],
    "MIA": ["MIA", "MARLINS", "MIAMI MARLINS"],
    "MIL": ["MIL", "BREWERS", "MILWAUKEE BREWERS"],
    "MIN": ["MIN", "TWINS", "MINNESOTA TWINS"],
    "NYM": ["NYM", "METS", "NEW YORK METS"],
    "NYY": ["NYY", "YANKEES", "NEW YORK YANKEES"],
    "PHI": ["PHI", "PHILLIES", "PHILADELPHIA PHILLIES"],
    "PIT": ["PIT", "PIRATES", "PITTSBURGH PIRATES"],
    "SD": ["SD", "PADRES", "SAN DIEGO PADRES"],
    "SEA": ["SEA", "MARINERS", "SEATTLE MARINERS"],
    "SF": ["SF", "GIANTS", "SAN FRANCISCO GIANTS"],
    "STL": ["STL", "CARDINALS", "ST. LOUIS CARDINALS"],
    "TB": ["TB", "RAYS", "TAMPA BAY RAYS"],
    "TEX": ["TEX", "RANGERS", "TEXAS RANGERS"],
    "TOR": ["TOR", "BLUE JAYS", "TORONTO BLUE JAYS"],
    "WSH": ["WSH", "NATIONALS", "WASHINGTON NATIONALS"],
    "ATL": ["ATL", "BRAVES", "ATLANTA BRAVES"],
    "CWS": ["CWS", "WHITE SOX", "CHICAGO WHITE SOX"],
}


def normalize_team(team):
    if not team:
        return None
    return team.strip().upper()


def valid_team(team):
    """THE FIX FOR THE UNBOUNDED-CACHE ISSUE. Any ?team= value the caller sends
    used to become a permanent cache key. Only the 30 known codes are accepted
    now, so the cache key space is 31 entries, period. An unknown code is not an
    error -- it just means 'no filter', matching the old behaviour for callers
    passing a full team name."""
    t = normalize_team(team)
    return t if t in TEAM_ALIASES else None


def team_match(team_val, selected):
    if not selected:
        return True
    selected = normalize_team(selected)
    val = normalize_team(team_val)
    aliases = TEAM_ALIASES.get(selected, [selected])
    return val in aliases or team_val == selected


# ------------------------------ savant fetchers ------------------------------

def _season():
    return today_et().year


def fetch_exit_velo():
    from pybaseball import statcast_batter_exitvelo_barrels
    df = statcast_batter_exitvelo_barrels(_season(), minBBE=50)
    wanted = ["last_name, first_name", "avg_hit_speed", "barrel_batted_rate",
              "hard_hit_percent", "avg_distance", "avg_hr_distance"]
    cols = [c for c in wanted if c in df.columns]
    df = df[cols].head(25).copy()
    rename_if_exists(df, {
        "last_name, first_name": "player",
        "avg_hit_speed": "avg_exit_velo",
        "barrel_batted_rate": "barrel_pct",
        "hard_hit_percent": "hard_hit_pct",
    })
    return df_to_records(df.round(1))


def fetch_expected_stats():
    from pybaseball import statcast_batter_expected_stats
    df = statcast_batter_expected_stats(_season(), minPA=100)
    name_col = find_col(df, ["last_name, first_name", "player_name", "name", "Name"])
    pa_col = find_col(df, ["pa", "PA", "plate_appearances"])
    xba_col = find_col(df, ["est_ba", "xba", "x_ba", "expected_batting_avg"])
    xslg_col = find_col(df, ["est_slg", "xslg", "x_slg", "expected_slg"])
    xwoba_col = find_col(df, ["est_woba", "xwoba", "x_woba", "expected_woba"])
    xobp_col = find_col(df, ["est_obp", "xobp", "x_obp", "expected_obp"])
    woba_col = find_col(df, ["woba", "w_oba"])
    ba_col = find_col(df, ["batting_avg", "ba", "avg", "batting_average"])
    cols = [c for c in [name_col, pa_col, xba_col, xslg_col, xwoba_col, xobp_col, woba_col, ba_col] if c]
    if not cols:
        return []
    df = df[cols].head(25).copy()
    rename_map = {}
    for src, dst in ((name_col, "player"), (pa_col, "pa"), (xba_col, "xba"), (xslg_col, "xslg"),
                     (xwoba_col, "xwoba"), (xobp_col, "xobp"), (woba_col, "woba"), (ba_col, "ba")):
        if src:
            rename_map[src] = dst
    rename_if_exists(df, rename_map)
    df = df.round(3)
    if "xwoba" in df.columns and "woba" in df.columns:
        df["edge"] = (df["xwoba"] - df["woba"]).round(3)
    return df_to_records(df)


def fetch_batter_percentile_ranks():
    from pybaseball import statcast_batter_percentile_ranks
    df = statcast_batter_percentile_ranks(_season())
    name_col = find_col(df, ["player_name", "last_name, first_name", "name"])
    cols = [c for c in [name_col, "exit_velocity", "hard_hit_rate", "barrel_batted_rate",
                        "whiff_percent", "sprint_speed"] if c and c in df.columns]
    if not cols:
        return []
    df = df[cols].head(25).copy()
    if name_col:
        rename_if_exists(df, {name_col: "player"})
    return df_to_records(df)


def fetch_pitcher_expected_stats():
    from pybaseball import statcast_pitcher_expected_stats
    df = statcast_pitcher_expected_stats(_season(), minPA=100)
    name_col = find_col(df, ["last_name, first_name", "player_name", "name"])
    pa_col = find_col(df, ["pa", "PA", "plate_appearances"])
    xba_col = find_col(df, ["est_ba", "xba", "x_ba", "expected_batting_avg"])
    xslg_col = find_col(df, ["est_slg", "xslg", "x_slg"])
    xwoba_col = find_col(df, ["est_woba", "xwoba", "x_woba"])
    xera_col = find_col(df, ["est_era", "xera", "x_era", "expected_era"])
    era_col = find_col(df, ["era", "ERA", "p_era"])
    woba_col = find_col(df, ["woba", "w_oba"])
    cols = [c for c in [name_col, pa_col, xba_col, xslg_col, xwoba_col, xera_col, era_col, woba_col] if c]
    if not cols:
        return []
    df = df[cols].head(25).copy()
    rename_map = {}
    for src, dst in ((name_col, "player"), (pa_col, "pa"), (xba_col, "xba"), (xslg_col, "xslg"),
                     (xwoba_col, "xwoba"), (xera_col, "xera"), (era_col, "era"), (woba_col, "woba")):
        if src:
            rename_map[src] = dst
    rename_if_exists(df, rename_map)
    df = df.round(3)
    if "xwoba" in df.columns and "woba" in df.columns:
        df["edge"] = (df["woba"] - df["xwoba"]).round(3)
    return df_to_records(df)


def fetch_pitcher_arsenal():
    from pybaseball import statcast_pitcher_arsenal_stats
    df = statcast_pitcher_arsenal_stats(_season(), minPA=50)
    name_col = find_col(df, ["last_name, first_name", "player_name", "name"])
    cols = [c for c in [name_col, "pitch_name", "pa", "run_value_per100", "whiff_percent",
                        "k_percent", "put_away"] if c and c in df.columns]
    if not cols:
        return []
    df = df[cols].head(30).copy()
    rename_if_exists(df, {
        name_col: "player",
        "run_value_per100": "rv100",
        "whiff_percent": "whiff_pct",
        "k_percent": "k_pct",
    })
    return df_to_records(df.round(2))


def fetch_pitcher_percentile_ranks():
    from pybaseball import statcast_pitcher_percentile_ranks
    df = statcast_pitcher_percentile_ranks(_season())
    name_col = find_col(df, ["player_name", "last_name, first_name", "name"])
    cols = [c for c in [name_col, "xera", "fastball_velo", "whiff_percent", "k_percent",
                        "bb_percent", "hard_hit_percent"] if c and c in df.columns]
    if not cols:
        return []
    df = df[cols].head(25).copy()
    if name_col:
        rename_if_exists(df, {name_col: "player"})
    return df_to_records(df)


def fetch_slate(team=None):
    # Eastern date, not UTC -- see the ET note at the top.
    today = today_et().strftime("%m/%d/%Y")
    url = (f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}"
           f"&hydrate=probablePitcher,lineScore")
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    games = []
    for date in data.get("dates", []):
        for g in date.get("games", []):
            away = g["teams"]["away"]
            home = g["teams"]["home"]

            def pitcher_info(side):
                p = side.get("probablePitcher")
                return p.get("fullName", "TBD") if p else "TBD"

            away_team = away["team"]["name"]
            home_team = home["team"]["name"]
            game = {
                "game_pk": g.get("gamePk"),
                "status": g.get("status", {}).get("detailedState", "Scheduled"),
                "game_time_utc": g.get("gameDate", ""),
                "away_team": away_team,
                "home_team": home_team,
                "away_pitcher": pitcher_info(away),
                "home_pitcher": pitcher_info(home),
                "venue": g.get("venue", {}).get("name", ""),
            }
            if team:
                if team_match(away_team, team) or team_match(home_team, team):
                    games.append(game)
            else:
                games.append(game)
    return games


# ---------------------------------- routes ----------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/ping")
def ping():
    return jsonify({"status": "ok", "message": "awake"})


def _leaderboard(route_name, cache_key, fetch_fn):
    try:
        return jsonify({"status": "ok", "data": get_cached(cache_key, fetch_fn),
                        "source": "Baseball Savant"})
    except Exception as e:
        return fail(route_name, e)


@app.route("/api/exit-velo")
def exit_velo():
    return _leaderboard("exit-velo", "exit_velo", fetch_exit_velo)


@app.route("/api/expected-stats")
def expected_stats():
    return _leaderboard("expected-stats", "expected_stats", fetch_expected_stats)


@app.route("/api/percentile-ranks")
def percentile_ranks():
    return _leaderboard("percentile-ranks", "batter_pct", fetch_batter_percentile_ranks)


@app.route("/api/pitcher-expected-stats")
def pitcher_expected_stats():
    return _leaderboard("pitcher-expected-stats", "pitcher_xstats", fetch_pitcher_expected_stats)


@app.route("/api/pitcher-arsenal")
def pitcher_arsenal():
    return _leaderboard("pitcher-arsenal", "pitcher_arsenal", fetch_pitcher_arsenal)


@app.route("/api/pitcher-percentile-ranks")
def pitcher_percentile_ranks():
    return _leaderboard("pitcher-percentile-ranks", "pitcher_pct", fetch_pitcher_percentile_ranks)


@app.route("/api/slate")
def slate():
    try:
        # valid_team() clamps the cache key space to the 30 real teams; an
        # arbitrary ?team= string can no longer mint a permanent cache entry.
        team = valid_team(request.args.get("team"))
        return jsonify({
            "status": "ok",
            "data": get_cached(f"slate_{team or 'all'}", lambda: fetch_slate(team)),
            "source": "MLB StatsAPI",
            "date": today_et().strftime("%B %d, %Y"),
        })
    except Exception as e:
        return fail("slate", e)


@app.route("/api/teams")
def teams():
    return jsonify({"status": "ok", "teams": [
        {"code": c, "name": a[-1].title()} for c, a in sorted(TEAM_ALIASES.items())
    ]})


@app.route("/api/columns")
def columns():
    """DIAGNOSTIC ONLY, AND NOW GATED.

    This route pulled a full season leaderboard from pybaseball on EVERY request
    with no caching, so an unauthenticated caller could hammer it and drive both
    the dyno and upstream Savant traffic -- the cheapest denial-of-service in the
    app. It exists to inspect column names when pybaseball changes its schema,
    which is a maintenance task, not a public API. It now reuses the cached
    expected-stats pull and is disabled unless DEBUG_ROUTES=1 is set.
    """
    if os.environ.get("DEBUG_ROUTES") != "1":
        return jsonify({"status": "error", "message": "not found"}), 404
    try:
        records = get_cached("expected_stats", fetch_expected_stats)
        cols = sorted(records[0].keys()) if records else []
        return jsonify({"status": "ok", "columns": cols})
    except Exception as e:
        return fail("columns", e)


@app.route("/api/kpis")
def kpis():
    try:
        ev = get_cached("exit_velo", fetch_exit_velo)
        xs = get_cached("expected_stats", fetch_expected_stats)
        sl = get_cached("slate_all", lambda: fetch_slate(None))

        def _num(x):
            try:
                f = float(x)
                return f if f == f else 0.0
            except (TypeError, ValueError):
                return 0.0

        n_ev = max(len(ev), 1)
        n_xs = max(len(xs), 1)
        # Every aggregate goes through _num(): pybaseball can return None or a
        # non-numeric on any field, and a bare sum() 500s the whole route. The
        # original only guarded hard_hit_pct.
        return jsonify({
            "status": "ok",
            "avg_exit_velo": round(sum(_num(p.get("avg_exit_velo")) for p in ev) / n_ev, 1),
            "avg_barrel_rate": round(sum(_num(p.get("barrel_pct")) for p in ev) / n_ev, 1),
            "avg_hard_hit": round(sum(_num(p.get("hard_hit_pct")) for p in ev) / n_ev, 1),
            "avg_xwoba": round(sum(_num(p.get("xwoba")) for p in xs) / n_xs, 3),
            "top_positive_edge": (max(xs, key=lambda p: _num(p.get("edge")))["player"]
                                  if xs else "N/A"),
            "games_today": len(sl),
            "year": _season(),
        })
    except Exception as e:
        return fail("kpis", e)


@app.route("/api/calibration")
def api_calibration():
    try:
        cal = get_cached("calibration_file", load_calibration_file)
        return jsonify({"status": "ok", "data": cal,
                        "source": "log5 backtest" if cal else "no calibration committed yet"})
    except Exception as e:
        return fail("calibration", e)


@app.route("/api/daily-board")
def api_daily_board():
    """What MLB_Daily.js fetches. Reading a committed file is instant; the only
    latency is the free-tier cold start, which the phone script retries through."""
    try:
        board = get_cached("daily_board", load_daily_board, ttl=BOARD_CACHE_TTL)
        return jsonify({"status": "ok", "data": board,
                        "source": "github-actions log5 model" if board else "no board published yet"})
    except Exception as e:
        return fail("daily-board", e)


@app.route("/api/hr-predictions")
def hr_predictions():
    """Home-run-probability-ranked view of today's daily board, for the browser
    dashboard. daily_board.json/mlb_model already compute hrProb per hitter --
    this just flattens today's games into one list sorted by it, since the raw
    board (nested by game/side) only otherwise gets consumed by MLB_Daily.js."""
    try:
        board = get_cached("daily_board", load_daily_board, ttl=BOARD_CACHE_TTL)
        if not board:
            return jsonify({"status": "ok", "data": [],
                            "source": "no board published yet"})
        rows = []
        for g in board.get("games", []):
            game_time = _format_game_time(g.get("gameTime"))
            sides = (
                ("homeMatchups", g.get("awayProbable")),
                ("awayMatchups", g.get("homeProbable")),
            )
            for side_key, opp_probable in sides:
                opp_name = (opp_probable or {}).get("name") or "TBD"
                opp_hand = (opp_probable or {}).get("hand")
                opp = f"{opp_name} ({opp_hand})" if opp_hand else opp_name
                for m in g.get(side_key, []):
                    hr_prob = m.get("hrProb")
                    if hr_prob is None:
                        continue
                    rows.append({
                        "player": m.get("name"),
                        "team": m.get("teamAbbr"),
                        "opp": opp,
                        "game_time": game_time,
                        "hr_pct": round(hr_prob * 100, 1),
                        "hr_tier": m.get("hrTier"),
                        "hit_pct": round((m.get("hitProb") or 0) * 100, 1),
                        "confidence": m.get("confidence"),
                    })
        rows.sort(key=lambda r: r["hr_pct"], reverse=True)
        return jsonify({"status": "ok", "data": rows,
                        "source": f"log5 model · built {board.get('builtAt')}"})
    except Exception as e:
        return fail("hr-predictions", e)


@app.route("/api/health")
def health():
    """Cheap liveness + freshness check that doesn't touch pybaseball. Tells you
    whether the served board is actually today's, which the phone can't."""
    board = load_daily_board()
    return jsonify({
        "status": "ok",
        "boardBuiltAt": (board or {}).get("builtAt"),
        "boardIsToday": bool(board) and board.get("builtAt") == today_et().strftime("%Y-%m-%d"),
        "lineupsConfirmed": (board or {}).get("lineupsConfirmedGames"),
        "modelVersion": (board or {}).get("modelVersion"),
        "expectedModelVersion": MODEL_VERSION,
        "cacheEntries": len(_cache),
        "serverDateET": today_et().strftime("%Y-%m-%d %H:%M"),
    })


if __name__ == "__main__":
    # debug=True enables the Werkzeug interactive debugger, which is arbitrary
    # code execution for anyone who can reach it. Production uses
    # `gunicorn app:app` (render.yaml), so this never ran in the deployment --
    # but it stays off by default so a stray `python app.py` on a shared network
    # isn't a remote shell.
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
