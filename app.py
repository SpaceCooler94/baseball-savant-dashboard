from flask import Flask, render_template, jsonify
import pandas as pd
from pathlib import Path

app = Flask(__name__)

DATA_FILE = Path("data/sample_players.csv")

def load_players():
    if DATA_FILE.exists():
        df = pd.read_csv(DATA_FILE)
    else:
        df = pd.DataFrame([
            {"player": "Juan Soto", "xba": 0.311, "xslg": 0.612, "barrel_pct": 19.4, "sweet_spot_pct": 37.8, "edge": 8.2},
            {"player": "Austin Riley", "xba": 0.289, "xslg": 0.574, "barrel_pct": 15.8, "sweet_spot_pct": 35.1, "edge": 6.4},
            {"player": "Bobby Witt Jr.", "xba": 0.301, "xslg": 0.541, "barrel_pct": 11.7, "sweet_spot_pct": 33.9, "edge": 5.7},
            {"player": "Corbin Carroll", "xba": 0.276, "xslg": 0.522, "barrel_pct": 12.4, "sweet_spot_pct": 32.7, "edge": 2.3},
            {"player": "Kyle Tucker", "xba": 0.298, "xslg": 0.559, "barrel_pct": 13.1, "sweet_spot_pct": 36.4, "edge": 4.9},
        ])
    return df

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/players")
def players():
    df = load_players()
    return jsonify(df.to_dict(orient="records"))

@app.route("/api/kpis")
def kpis():
    return jsonify({
        "avg_exit_velo": 91.8,
        "hard_hit_rate": 46.2,
        "barrel_rate": 11.4,
        "whiff_rate": 29.1,
        "expected_run_index": 118
    })

if __name__ == "__main__":
    app.run(debug=True)
