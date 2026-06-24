from flask import Flask, render_template, jsonify
import datetime
import time
import requests
import math

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

def fetch_slate():
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
            games.append({
                "game_pk": g.get("gamePk"),
                "status": g.get("status", {}).get("detailedState", "Scheduled"),
                "game_time_utc": g.get("gameDate", ""),
                "away_team": away["team"]["name"],
                "home_team": home["team"]["name"],
                "away_pitcher": pitcher_info(away),
                "home_pitcher": pitcher_info(home),
                "venue": g.get("venue", {}).get("name", ""),
            })
    return games

@app.route("/")
def home():
    return render_template("index.html")

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
        return jsonify({"status": "ok", "data": get_cached("slate", fetch_slate), "source": "MLB StatsAPI", "date": datetime.datetime.now().strftime("%B %d, %Y")})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

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
        sl = get_cached("slate", fetch_slate)
        avg_ev = round(sum(p.get("avg_exit_velo", 0) for p in ev) / max(len(ev), 1), 1)
        avg_brl = round(sum(p.get("barrel_pct", 0) for p in ev) / max(len(ev), 1), 1)
        avg_xwoba = round(sum(p.get("xwoba", 0) for p in xs) / max(len(xs), 1), 3)
        top_edge = max(xs, key=lambda p: p.get("edge") or 0)["player"] if xs else "N/A"
        return jsonify({"status": "ok", "avg_exit_velo": avg_ev, "avg_barrel_rate": avg_brl, "avg_xwoba": avg_xwoba, "top_positive_edge": top_edge, "games_today": len(sl), "year": datetime.datetime.now().year})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
