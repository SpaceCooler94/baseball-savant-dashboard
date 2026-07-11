from flask import Flask, render_template, jsonify, request
import datetime
import time
import requests
import math
import os
import json

app = Flask(__name__)

_cache = {}
CACHE_TTL = 3600

def get_cached(key, fetch_fn):
    now = time.time()
    if key in _cache:
        data, ts = _cache[key]
        if now - ts < CACHE_TTL:
            return data
    data = fetch_fn()
    _cache[key] = (data, now)
    return data

# ============================================================================
# Calibration for Daily_Matchups.js v5.1 (log5 model).
# Keep MODEL_VERSION in lockstep with MODEL_VERSION in Daily_Matchups.js. If the
# log5 formula there changes, bump both -- Scriptable ignores a calibration file
# whose modelVersion doesn't match and falls back to raw/uncalibrated output.
#
# RENDER FREE TIER: the filesystem is ephemeral (wiped on redeploy and on dyno
# sleep after ~15 min idle). Do NOT write calibration.json at runtime and expect
# it to persist. Instead run the backtest in your Windows Python stack
# (C:\Users\astro\hitter\), then COMMIT calibration.json into this repo so it
# ships with each deploy. This route just serves that committed file.
# ============================================================================
MODEL_VERSION = "log5-v5.0"
CALIBRATION_PATH = os.path.join(os.path.dirname(__file__), "calibration.json")

def load_calibration_file():
    if not os.path.exists(CALIBRATION_PATH):
        return None
    try:
        with open(CALIBRATION_PATH, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    # Version gate: refuse to serve calibration fit for a different model version.
    if not isinstance(data, dict) or data.get("modelVersion") != MODEL_VERSION:
        return None
    return data

def df_to_records(df):
    records = df.to_dict(orient="records")
    cleaned = []
    for row in records:
        clean_row = {}
        for k, v in row.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean_row[k] = None
            else:
                clean_row[k] = v
        cleaned.append(clean_row)
    return cleaned

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def rename_if_exists(df, mapping):
    df.rename(columns={k: v for k, v in mapping.items() if k in df.columns}, inplace=True)
    return df

def normalize_team(team):
    if not team:
        return None
    return team.strip().upper()

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

def team_match(team_val, selected):
    if not selected:
        return True
    selected = normalize_team(selected)
    val = normalize_team(team_val)
    aliases = TEAM_ALIASES.get(selected, [selected])
    return val in aliases or team_val == selected

def fetch_exit_velo():
    from pybaseball import statcast_batter_exitvelo_barrels
    df = statcast_batter_exitvelo_barrels(datetime.datetime.now().year, minBBE=50)
    wanted = ["last_name, first_name", "avg_hit_speed", "barrel_batted_rate", "hard_hit_percent", "avg_distance", "avg_hr_distance"]
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
    df = statcast_batter_expected_stats(datetime.datetime.now().year, minPA=100)
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
    if name_col: rename_map[name_col] = "player"
    if pa_col: rename_map[pa_col] = "pa"
    if xba_col: rename_map[xba_col] = "xba"
    if xslg_col: rename_map[xslg_col] = "xslg"
    if xwoba_col: rename_map[xwoba_col] = "xwoba"
    if xobp_col: rename_map[xobp_col] = "xobp"
    if woba_col: rename_map[woba_col] = "woba"
    if ba_col: rename_map[ba_col] = "ba"
    rename_if_exists(df, rename_map)
    df = df.round(3)
    if "xwoba" in df.columns and "woba" in df.columns:
        df["edge"] = (df["xwoba"] - df["woba"]).round(3)
    return df_to_records(df)

def fetch_batter_percentile_ranks():
    from pybaseball import statcast_batter_percentile_ranks
    df = statcast_batter_percentile_ranks(datetime.datetime.now().year)
    name_col = find_col(df, ["player_name", "last_name, first_name", "name"])
    cols = [c for c in [name_col, "exit_velocity", "hard_hit_rate", "barrel_batted_rate", "whiff_percent", "sprint_speed"] if c and c in df.columns]
    if not cols:
        return []
    df = df[cols].head(25).copy()
    if name_col:
        rename_if_exists(df, {name_col: "player"})
    return df_to_records(df)

def fetch_pitcher_expected_stats():
    from pybaseball import statcast_pitcher_expected_stats
    df = statcast_pitcher_expected_stats(datetime.datetime.now().year, minPA=100)
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
    if name_col: rename_map[name_col] = "player"
    if pa_col: rename_map[pa_col] = "pa"
    if xba_col: rename_map[xba_col] = "xba"
    if xslg_col: rename_map[xslg_col] = "xslg"
    if xwoba_col: rename_map[xwoba_col] = "xwoba"
    if xera_col: rename_map[xera_col] = "xera"
    if era_col: rename_map[era_col] = "era"
    if woba_col: rename_map[woba_col] = "woba"
    rename_if_exists(df, rename_map)
    df = df.round(3)
    if "xwoba" in df.columns and "woba" in df.columns:
        df["edge"] = (df["woba"] - df["xwoba"]).round(3)
    return df_to_records(df)

def fetch_pitcher_arsenal():
    from pybaseball import statcast_pitcher_arsenal_stats
    df = statcast_pitcher_arsenal_stats(datetime.datetime.now().year, minPA=50)
    name_col = find_col(df, ["last_name, first_name", "player_name", "name"])
    cols = [c for c in [name_col, "pitch_name", "pa", "run_value_per100", "whiff_percent", "k_percent", "put_away"] if c and c in df.columns]
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
    df = statcast_pitcher_percentile_ranks(datetime.datetime.now().year)
    name_col = find_col(df, ["player_name", "last_name, first_name", "name"])
    cols = [c for c in [name_col, "xera", "fastball_velo", "whiff_percent", "k_percent", "bb_percent", "hard_hit_percent"] if c and c in df.columns]
    if not cols:
        return []
    df = df[cols].head(25).copy()
    if name_col:
        rename_if_exists(df, {name_col: "player"})
    return df_to_records(df)

def fetch_slate(team=None):
    today = datetime.datetime.now().strftime("%m/%d/%Y")
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}&hydrate=probablePitcher,lineScore"
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

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/ping")
def ping():
    return jsonify({"status": "ok", "message": "awake"})

@app.route("/api/exit-velo")
def exit_velo():
    try:
        return jsonify({"status": "ok", "data": get_cached("exit_velo", fetch_exit_velo), "source": "Baseball Savant"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/expected-stats")
def expected_stats():
    try:
        return jsonify({"status": "ok", "data": get_cached("expected_stats", fetch_expected_stats), "source": "Baseball Savant"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/percentile-ranks")
def percentile_ranks():
    try:
        return jsonify({"status": "ok", "data": get_cached("batter_pct", fetch_batter_percentile_ranks), "source": "Baseball Savant"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/pitcher-expected-stats")
def pitcher_expected_stats():
    try:
        return jsonify({"status": "ok", "data": get_cached("pitcher_xstats", fetch_pitcher_expected_stats), "source": "Baseball Savant"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/pitcher-arsenal")
def pitcher_arsenal():
    try:
        return jsonify({"status": "ok", "data": get_cached("pitcher_arsenal", fetch_pitcher_arsenal), "source": "Baseball Savant"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/pitcher-percentile-ranks")
def pitcher_percentile_ranks():
    try:
        return jsonify({"status": "ok", "data": get_cached("pitcher_pct", fetch_pitcher_percentile_ranks), "source": "Baseball Savant"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/slate")
def slate():
    try:
        team = request.args.get("team")
        return jsonify({"status": "ok", "data": get_cached(f"slate_{team or 'all'}", lambda: fetch_slate(team)), "source": "MLB StatsAPI", "date": datetime.datetime.now().strftime("%B %d, %Y")})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/teams")
def teams():
    return jsonify({"status": "ok", "teams": [{"code": "NYY", "name": "New York Yankees"}, {"code": "LAD", "name": "Los Angeles Dodgers"}, {"code": "PHI", "name": "Philadelphia Phillies"}, {"code": "ATL", "name": "Atlanta Braves"}, {"code": "HOU", "name": "Houston Astros"}, {"code": "SD", "name": "San Diego Padres"}, {"code": "TB", "name": "Tampa Bay Rays"}, {"code": "TOR", "name": "Toronto Blue Jays"}, {"code": "SEA", "name": "Seattle Mariners"}, {"code": "MIL", "name": "Milwaukee Brewers"}]})

@app.route("/api/columns")
def columns():
    try:
        from pybaseball import statcast_batter_expected_stats
        df = statcast_batter_expected_stats(datetime.datetime.now().year, minPA=100)
        return jsonify({"status": "ok", "columns": list(df.columns)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/kpis")
def kpis():
    try:
        ev = get_cached("exit_velo", fetch_exit_velo)
        xs = get_cached("expected_stats", fetch_expected_stats)
        sl = get_cached("slate_all", lambda: fetch_slate(None))
        avg_ev = round(sum(p.get("avg_exit_velo", 0) for p in ev) / max(len(ev), 1), 1)
        avg_brl = round(sum(p.get("barrel_pct", 0) for p in ev) / max(len(ev), 1), 1)
        # Frontend (static/app.js) reads json.avg_hard_hit for the "Avg Hard Hit%" tile,
        # but this route never returned it -> the tile showed "undefined%". The exit-velo
        # records already carry hard_hit_pct (fetch_exit_velo renames hard_hit_percent to
        # hard_hit_pct), so average that. Guard each value: pybaseball can return None or
        # non-numeric for a field on some rows, and a bare sum() would 500 the whole route.
        def _num(x):
            try:
                f = float(x)
                return f if f == f else 0.0  # f != f screens out NaN
            except (TypeError, ValueError):
                return 0.0
        avg_hh = round(sum(_num(p.get("hard_hit_pct")) for p in ev) / max(len(ev), 1), 1)
        avg_xwoba = round(sum(p.get("xwoba", 0) for p in xs) / max(len(xs), 1), 3)
        top_edge = max(xs, key=lambda p: p.get("edge") or 0)["player"] if xs else "N/A"
        return jsonify({"status": "ok", "avg_exit_velo": avg_ev, "avg_barrel_rate": avg_brl, "avg_hard_hit": avg_hh, "avg_xwoba": avg_xwoba, "top_positive_edge": top_edge, "games_today": len(sl), "year": datetime.datetime.now().year})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/calibration")
def api_calibration():
    try:
        cal = get_cached("calibration_file", load_calibration_file)
        # data:None is a valid, non-error response -- Daily_Matchups.js reads it as
        # "run uncalibrated" and falls back to raw log5. Cleaner for the JS than a 404.
        source = "log5 backtest" if cal else "no calibration committed yet"
        return jsonify({"status": "ok", "data": cal, "source": source})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Serves the daily board that the GitHub Action builds and commits. This is what
# MLB_Daily.js (the one thin-client script) fetches. Reading a committed file is
# instant -- the only slowness is the free-tier cold-start wake, which the phone
# script already retries through. data:None means the Action hasn't produced a board
# yet (e.g. before the first run), which the phone script shows as "not published".
def load_daily_board():
    path = os.path.join(os.path.dirname(__file__), "daily_board.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

@app.route("/api/daily-board")
def api_daily_board():
    try:
        # Short cache (5 min) -- the file only changes once a day when the Action runs,
        # but a brief cache spares disk reads if the phone is hit repeatedly.
        board = get_cached("daily_board", load_daily_board)
        source = "github-actions log5 model" if board else "no board published yet"
        return jsonify({"status": "ok", "data": board, "source": source})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
