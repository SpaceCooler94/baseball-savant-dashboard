from flask import Flask, render_template, jsonify
import datetime
import time

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

def fetch_exit_velo():
    from pybaseball import statcast_batter_exitvelo_barrels
    df = statcast_batter_exitvelo_barrels(datetime.datetime.now().year, minBBE=50)
    cols = ["last_name, first_name","avg_hit_speed","barrel_batted_rate","hard_hit_percent","avg_distance","avg_hr_distance"]
    df = df[cols].head(25)
    df.columns = ["player","avg_exit_velo","barrel_pct","hard_hit_pct","avg_distance","avg_hr_distance"]
    return df.round(1).to_dict(orient="records")

def fetch_expected_stats():
    from pybaseball import statcast_batter_expected_stats
    df = statcast_batter_expected_stats(datetime.datetime.now().year, minPA=100)
    cols = ["last_name, first_name","pa","xba","xslg","xwoba","xobp","woba","batting_avg"]
    df = df[cols].head(25)
    df.columns = ["player","pa","xba","xslg","xwoba","xobp","woba","ba"]
    df = df.round(3)
    df["edge"] = (df["xwoba"] - df["woba"]).round(3)
    return df.to_dict(orient="records")

def fetch_percentile_ranks():
    from pybaseball import statcast_batter_percentile_ranks
    df = statcast_batter_percentile_ranks(datetime.datetime.now().year)
    cols = ["player_name","exit_velocity","hard_hit_rate","barrel_batted_rate","whiff_percent","sprint_speed"]
    available = [c for c in cols if c in df.columns]
    df = df[available].head(25)
    df.rename(columns={"player_name": "player"}, inplace=True)
    return df.to_dict(orient="records")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/exit-velo")
def exit_velo():
    try:
        data = get_cached("exit_velo", fetch_exit_velo)
        return jsonify({"status": "ok", "data": data, "source": "Baseball Savant / pybaseball"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/expected-stats")
def expected_stats():
    try:
        data = get_cached("expected_stats", fetch_expected_stats)
        return jsonify({"status": "ok", "data": data, "source": "Baseball Savant / pybaseball"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/percentile-ranks")
def percentile_ranks():
    try:
        data = get_cached("percentile_ranks", fetch_percentile_ranks)
        return jsonify({"status": "ok", "data": data, "source": "Baseball Savant / pybaseball"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/kpis")
def kpis():
    try:
        ev_data = get_cached("exit_velo", fetch_exit_velo)
        xstats = get_cached("expected_stats", fetch_expected_stats)
        avg_ev = round(sum(p["avg_exit_velo"] for p in ev_data) / len(ev_data), 1)
        avg_barrel = round(sum(p["barrel_pct"] for p in ev_data) / len(ev_data), 1)
        avg_hh = round(sum(p["hard_hit_pct"] for p in ev_data) / len(ev_data), 1)
        avg_xwoba = round(sum(p["xwoba"] for p in xstats) / len(xstats), 3)
        top_edge = max(xstats, key=lambda p: p["edge"])["player"]
        return jsonify({
            "status": "ok",
            "avg_exit_velo": avg_ev,
            "avg_barrel_rate": avg_barrel,
            "avg_hard_hit": avg_hh,
            "avg_xwoba": avg_xwoba,
            "top_positive_edge": top_edge,
            "year": datetime.datetime.now().year
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
