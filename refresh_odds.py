#!/usr/bin/env python3
# ============================================================================
# refresh_odds.py -- pulls real DraftKings/FanDuel milestone (Over 0.5/1.5/
# 2.5) prices for HR and Hits props from The Odds API, joins them to today's
# board by player name, and patches in the actual, correctly-computed edge:
#
#   edge = model_probability * book_decimal_odds - 1
#
# using this pipeline's own calibrated milestone_prob() (mlb_model.py) priced
# at the SAME threshold the book is offering -- not just the Over 0.5 number
# the board already had. Also computes the book's OWN no-vig fair price
# (devig_two_way) so "do we agree on the true rate" and "is this bet
# profitable at the price offered" are two separate, clearly labeled numbers,
# never conflated into one.
#
# Same safety discipline as refresh_lineups.py: atomic write, refuses to
# patch a board that isn't today's, retries with backoff on 429 (this
# matters more here than for StatsAPI -- The Odds API explicitly rate-limits
# bursts), and reports quota usage every run so cost stays visible.
#
# OPERATIONAL COST (read the docs before changing MARKETS/BOOKMAKERS in
# odds_api.py -- this math changes if you do):
#   2 markets (batter_home_runs_alternate, batter_hits_alternate)
#   x 1 region-equivalent (2 named bookmakers, and the docs price every group
#     of <=10 named bookmakers as ONE region)
#   = 2 credits PER EVENT, charged only for markets actually present in the
#     response (empty markets/events are free).
#   A 15-game slate = ~30 credits per full run. Three runs/day (paired with
#   the existing lineup-refresh cadence) = ~90 credits/day, ~2,700/month --
#   comfortably inside the lowest paid tier, and each run's actual cost is
#   printed from the x-requests-* response headers rather than estimated.
#
# Usage: python refresh_odds.py [path/to/daily_board.json]
# Requires env var ODDS_API_KEY. Exits 0 (not an error) when no odds are
# posted yet for a game -- normal well before lineups are out.
# ============================================================================

import datetime
import json
import os
import sys
import tempfile
import time
from zoneinfo import ZoneInfo

import requests

import mlb_model as M
import odds_api as O

ET = ZoneInfo("America/New_York")
ODDS_API = "https://api.the-odds-api.com/v4"
SPORT = "baseball_mlb"
BOARD_PATH = "daily_board.json"
CAL_PATH = "calibration.json"
THRESHOLDS_K = (1, 2, 3)   # Over 0.5 / 1.5 / 2.5 -- what DK/FD actually post

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "mlb-daily-board-odds/1.0 (personal analytics pipeline)"})


def http_json(url, params, tries=4):
    """Same retry shape as build_daily_board.py/refresh_lineups.py's
    http_json, tuned for The Odds API's documented rate limiting (429 on
    bursts) rather than StatsAPI's -- slightly longer backoff, since this
    module is explicitly warned about bursts in the docs, StatsAPI isn't."""
    last = None
    for attempt in range(tries):
        try:
            r = SESSION.get(url, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(min(3.0 * (attempt + 1), 15))
                last = requests.HTTPError("429 rate limited")
                continue
            r.raise_for_status()
            return r.json(), r.headers
        except requests.HTTPError as e:
            last = e
            if attempt < tries - 1:
                time.sleep(1.5 * (attempt + 1))
        except (requests.ConnectionError, requests.Timeout) as e:
            last = e
            if attempt < tries - 1:
                time.sleep(1.5 * (attempt + 1))
    raise last


def _nv(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def fetch_events(api_key):
    """Free endpoint -- does not count against quota. Returns today's/live
    MLB events with id, home_team, away_team, commence_time."""
    data, _ = http_json(f"{ODDS_API}/sports/{SPORT}/events", {"apiKey": api_key})
    return data if isinstance(data, list) else []


def match_event_to_game(event, board_games):
    """Team-name match, not ID match -- The Odds API and StatsAPI don't share
    event identifiers. Full team names on both sides make this a plain exact
    match in the normal case; falls back to substring match for the rare
    naming mismatch (e.g. 'Athletics' vs a market-specific alt name)."""
    home, away = event.get("home_team", ""), event.get("away_team", "")
    for g in board_games:
        gh = (g.get("homeTeam") or {}).get("name") or (g.get("homeTeam") or {}).get("abbr", "")
        ga = (g.get("awayTeam") or {}).get("name") or (g.get("awayTeam") or {}).get("abbr", "")
        if (home and (home == gh or home in gh or gh in home)) and \
           (away and (away == ga or away in ga or ga in away)):
            return g
    return None


def fetch_event_odds(api_key, event_id):
    """The one paid call, one event at a time -- see module docstring for the
    exact quota math. Returns (parsed_dict, headers) so the caller can log
    x-requests-remaining without a second request."""
    params = {
        "apiKey": api_key,
        "bookmakers": O.BOOKMAKERS,
        "markets": O.MARKETS,
        "oddsFormat": "american",
    }
    data, headers = http_json(f"{ODDS_API}/sports/{SPORT}/events/{event_id}/odds", params)
    return O.parse_event_odds(data or {}), headers


def apply_odds_to_row(row, odds_for_player, cal_by_stat):
    """Patches row["bookOdds"] = {"hr": {...}, "hit": {...}} with, per
    threshold actually offered by DK/FD: the model's calibrated price at that
    SAME threshold (via mlb_model.milestone_prob, not just the board's
    existing Over-0.5 number), the book's own devigged fair price, and the
    real edge at each book.

    cal_by_stat: {"hr": cal_block_or_None, "hit": cal_block_or_None} -- HR and
    Hit calibrate independently (different scale/offset in calibration.json),
    so this takes both rather than one shared block; passing the wrong one to
    the wrong stat would silently mis-price every HR-side edge.

    Returns True if anything was written."""
    wrote = False
    for stat in ("hr", "hit"):
        by_point = (odds_for_player or {}).get(stat)
        if not by_point:
            continue
        inputs = row.get("hrInputs" if stat == "hr" else "hitInputs") or {}
        raw_per_pa = _nv(inputs.get("rawPerPA"))
        n = _nv(row.get("expectedPA"))
        if raw_per_pa is None or n is None:
            continue
        cal_block = (cal_by_stat or {}).get(stat)
        stat_out = {}
        for point, books in by_point.items():
            k = O.milestone_threshold_to_k(point)
            if k is None or k not in THRESHOLDS_K:
                continue
            raw_p, cal_p = M.milestone_prob(raw_per_pa, n, k, cal_block)
            entry = {"point": point, "modelRaw": raw_p, "modelFair": cal_p, "books": {}}
            for book_key, prices in books.items():
                over, under = prices.get("over"), prices.get("under")
                book_fair_over, _ = O.devig_two_way(over, under)
                edge = O.compute_edge(cal_p, over) if over is not None else None
                entry["books"][book_key] = {
                    "overPrice": over, "underPrice": under,
                    "bookFairProb": book_fair_over,
                    "edge": edge,
                }
                wrote = True
            stat_out[str(point)] = entry
        if stat_out:
            row.setdefault("bookOdds", {})[stat] = stat_out
    return wrote


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
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        print("ODDS_API_KEY not set -- skipping odds refresh", file=sys.stderr)
        return

    path = sys.argv[1] if len(sys.argv) > 1 else BOARD_PATH
    if not os.path.exists(path):
        print(f"No board at {path} -- nothing to patch")
        return
    with open(path) as f:
        board = json.load(f)

    today = datetime.datetime.now(ET).strftime("%Y-%m-%d")
    if board.get("builtAt") != today:
        print(f"Board is for {board.get('builtAt')}, today is {today} -- refusing to patch a stale board")
        return

    cal_hit, cal_hr = None, None
    try:
        with open(CAL_PATH) as f:
            cal = json.load(f)
        if isinstance(cal, dict) and cal.get("modelVersion") == M.MODEL_VERSION:
            cal_hit, cal_hr = cal.get("hit"), cal.get("hr")
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    cal_by_stat = {"hr": cal_hr, "hit": cal_hit}

    events = fetch_events(api_key)
    if not events:
        print("No MLB events returned -- nothing to patch (normal in the off-season or very early morning)")
        return

    by_pid = {}
    for g in board.get("games", []):
        for side in ("homeMatchups", "awayMatchups"):
            for row in (g.get(side) or []):
                by_pid.setdefault(O.norm_name(row.get("name")), []).append(row)

    events_matched = 0
    rows_patched = 0
    last_headers = {}
    for ev in events:
        game = match_event_to_game(ev, board.get("games", []))
        if not game:
            continue
        try:
            parsed, headers = fetch_event_odds(api_key, ev["id"])
        except Exception as e:
            print(f"  odds fetch failed for {ev.get('away_team')} @ {ev.get('home_team')}: "
                  f"{type(e).__name__}: {e}", file=sys.stderr)
            continue
        last_headers = headers
        events_matched += 1
        all_players = set()
        for stat in ("hr", "hit"):
            all_players |= set((parsed.get(stat) or {}).keys())
        for player_norm in all_players:
            rows = by_pid.get(player_norm)
            if not rows:
                continue
            odds_for_player = {stat: (parsed.get(stat) or {}).get(player_norm)
                                for stat in ("hr", "hit")}
            for row in rows:
                if apply_odds_to_row(row, odds_for_player, cal_by_stat):
                    rows_patched += 1
        time.sleep(1.0)  # spacing between per-event calls, per the docs' 429 guidance

    if events_matched == 0:
        print("Matched 0 events to tonight's board -- team-name matching may need a look")
        return

    board["oddsRefreshedAt"] = datetime.datetime.now(ET).isoformat(timespec="minutes")
    board["oddsEventsMatched"] = events_matched
    atomic_write_json(path, board)
    remaining = last_headers.get("x-requests-remaining", "?")
    used_last = last_headers.get("x-requests-last", "?")
    print(f"Odds patched: {events_matched} events matched, {rows_patched} player-rows updated. "
          f"Quota remaining: {remaining} (last call cost {used_last})")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ODDS REFRESH FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
