#!/usr/bin/env python3
"""
hrhit_data.py — data layer for the HR/Hit sim model.

Pulls season-to-date Statcast (chunked monthly, cached to parquet so Actions
runs are incremental) + today's schedule/probables/lineups from StatsAPI.

Outputs:
  data/statcast_2026.parquet   (slim pitch-level, PA-ending rows only)
  data/slate.json              (games, probables, lineups, park)

Deterministic, no ML, no scraping beyond official endpoints.
"""

import json, os, sys, time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from pybaseball import statcast

DATA = Path("data"); DATA.mkdir(exist_ok=True)
CACHE = DATA / "statcast_2026.parquet"
SEASON_START = "2026-03-25"          # adjust to actual Opening Day
TODAY = date.today().isoformat()

KEEP = [
    "game_date", "batter", "pitcher", "stand", "p_throws", "events",
    "launch_speed", "launch_angle", "launch_speed_angle", "bb_type",
    "estimated_ba_using_speedangle", "hc_x", "hc_y", "home_team",
]

def pull_statcast():
    """Incremental: only fetch dates newer than what's cached."""
    if CACHE.exists():
        df = pd.read_parquet(CACHE)
        start = (pd.to_datetime(df["game_date"]).max() + timedelta(days=1)).date().isoformat()
    else:
        df, start = pd.DataFrame(), SEASON_START
    if start > TODAY:
        print("cache current"); return df

    chunks, cur = [], datetime.fromisoformat(start).date()
    end = date.today()
    while cur <= end:
        stop = min(cur + timedelta(days=13), end)
        print(f"statcast {cur} -> {stop}", flush=True)
        try:
            c = statcast(start_dt=cur.isoformat(), end_dt=stop.isoformat())
            if c is not None and len(c):
                c = c[c["events"].notna()][KEEP]        # PA-ending rows only
                chunks.append(c)
        except Exception as e:
            print(f"WARN chunk {cur}: {e}", file=sys.stderr)
        cur = stop + timedelta(days=1)
        time.sleep(2)

    if chunks:
        df = pd.concat([df, *chunks], ignore_index=True) if len(df) else pd.concat(chunks, ignore_index=True)
        df.to_parquet(CACHE, index=False)
    print(f"statcast rows: {len(df)}")
    return df

API = "https://statsapi.mlb.com/api/v1"

def get(url, **params):
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def pull_slate():
    sched = get(f"{API}/schedule", sportId=1, date=TODAY,
                hydrate="probablePitcher,lineups,team")
    games = []
    for d in sched.get("dates", []):
        for g in d.get("games", []):
            home = g["teams"]["home"]; away = g["teams"]["away"]
            gm = {
                "gamePk": g["gamePk"],
                "home": home["team"].get("abbreviation") or home["team"]["name"],
                "away": away["team"].get("abbreviation") or away["team"]["name"],
                "venue": g.get("venue", {}).get("name", ""),
                "probables": {
                    "home": (home.get("probablePitcher") or {}).get("id"),
                    "away": (away.get("probablePitcher") or {}).get("id"),
                },
                "lineups": {"home": [], "away": []},
            }
            lu = g.get("lineups") or {}
            for side, key in (("home", "homePlayers"), ("away", "awayPlayers")):
                gm["lineups"][side] = [p["id"] for p in lu.get(key, [])]
            games.append(gm)
    (DATA / "slate.json").write_text(json.dumps({"date": TODAY, "games": games}, indent=1))
    print(f"slate: {len(games)} games")
    # name map for board rendering
    ids = set()
    for gm in games:
        ids |= set(gm["lineups"]["home"]) | set(gm["lineups"]["away"])
        ids |= {gm["probables"]["home"], gm["probables"]["away"]} - {None}
    names = {}
    for i in range(0, len(ids := list(ids)), 100):
        batch = get(f"{API}/people", personIds=",".join(map(str, ids[i:i+100])))
        for p in batch.get("people", []):
            names[str(p["id"])] = p.get("fullName", str(p["id"]))
    (DATA / "names.json").write_text(json.dumps(names))

if __name__ == "__main__":
    pull_statcast()
    pull_slate()
