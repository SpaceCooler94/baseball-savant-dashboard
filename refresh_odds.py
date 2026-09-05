#!/usr/bin/env python3
# ============================================================================
# refresh_odds.py -- pulls real HR milestone prices from SportsGameOdds and
# patches today's board with row["bookOdds"]["hr"][point]["books"][book] =
# {overPrice, edge}. Same output shape the frontend (MLB_Daily.js) already
# expects and was fixed this session to handle correctly -- any threshold,
# any book, not a fixed 0.5/four-book assumption.
#
# STRUCTURAL CHANGE FROM THE PRIOR (The Odds API) INTEGRATION, confirmed
# against real, live, authenticated data before writing any of this:
#   - ONE call for the whole slate, not one per event. SportsGameOdds
#     charges per event returned, not per market+bookmaker, so a single
#     /v2/events?leagueID=MLB&oddIDs=...&limit=N pull with a high enough
#     limit returns every game AND every player's odds embedded in one
#     response. Confirmed on a real pull: 1 event, 1186 odds entries,
#     genuinely 1 object cost.
#   - No fixed threshold. Each bookmaker quotes its own "main" HR line
#     independently (real example: DraftKings/Pinnacle at Over 0.5,
#     FanDuel/BetMGM at Over 1.5, ESPN Bet/Hard Rock at Over 2.5, same
#     player, same moment) -- apply_odds_to_row() below prices the
#     model's probability at EACH book's own actual threshold, not a
#     single assumed one.
#   - `available` matters. A real pull showed only ~38.5% of returned
#     bookmaker quotes marked available=true -- the rest are stale/closed
#     lines still present in the response. odds_api.parse_event_odds()
#     already filters to available=true only; nothing here needs to
#     re-check that, but it's why row-level coverage will look thinner
#     than the raw number of bookmakers in a response might suggest, and
#     that's correct behavior, not a bug to chase.
#   - Player identity has no crosswalk to MLBAM hitterId -- joins are by
#     normalized name (see odds_api.norm_name()), same discipline as
#     every other cross-source join in this pipeline.
#
# Usage: python refresh_odds.py [path/to/daily_board.json]
# Requires env var SPORTSGAMEODDS_API_KEY.
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
BOARD_PATH = "daily_board.json"
CAL_PATH = "calibration.json"
THRESHOLDS_K = (1, 2, 3)   # Over 0.5 / 1.5 / 2.5 -- what real books actually post
EDGE_SANITY_CEILING = 0.35  # see MLB_Daily.js / prior refresh_odds.py for the
                             # real, live-data reasoning behind this number

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "mlb-daily-board-odds/2.0 (personal analytics pipeline)"})


def http_json(url, params, tries=4):
    last = None
    for attempt in range(tries):
        try:
            r = SESSION.get(url, params=params, timeout=25)
            if r.status_code == 429:
                time.sleep(min(3.0 * (attempt + 1), 15))
                last = requests.HTTPError("429 rate limited")
                continue
            r.raise_for_status()
            return r.json()
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


def fetch_slate_odds(api_key, limit=30):
    """One call, whole slate. limit=30 comfortably covers even a full
    15-game MLB night (each game is 1 event) with room to spare."""
    params = {
        "apiKey": api_key,
        "leagueID": "MLB",
        "oddsAvailable": "true",
        "oddIDs": O.ODD_IDS,
        "limit": limit,
    }
    data = http_json(O.BASE_URL, params)
    if not data.get("success"):
        raise RuntimeError(f"SportsGameOdds returned success=false: {data.get('error')}")
    return data.get("data") or []


def match_event_to_game(event, board_games):
    """Team-name match -- SportsGameOdds' eventID has no relationship to
    MLB Stats API's gamePk. Uses the short team name (e.g. 'NYM') which
    this pipeline's board rows already carry as teamAbbr."""
    teams = event.get("teams") or {}
    home = ((teams.get("home") or {}).get("names") or {}).get("short", "")
    away = ((teams.get("away") or {}).get("names") or {}).get("short", "")
    for g in board_games:
        gh = (g.get("homeTeam") or {}).get("abbr", "")
        ga = (g.get("awayTeam") or {}).get("abbr", "")
        if home == gh and away == ga:
            return g
    return None


def apply_odds_to_row(row, odds_for_player, cal_hr):
    """Patches row["bookOdds"]["hr"][point] = {point, modelRaw, modelFair,
    books: {book: {overPrice, underPrice, edge}}}. odds_for_player is
    {point: {book: {"over": price, "under": price}}} from
    odds_api.parse_event_odds() -- already keyed by each book's own real
    threshold, so this function pricing the model at THAT threshold (not
    a fixed one) is the whole point of this rewrite, not an afterthought.

    Ledger-stamps a book_edge_* angle under the same quality bar as
    before (HIGH confidence, no small-sample flag, point==0.5 only since
    settle.py only grades at-least-one-HR outcomes) so measure_signals.py
    keeps working unchanged on the new data source."""
    if not odds_for_player:
        return False
    inputs = row.get("hrInputs") or {}
    raw_per_pa = _nv(inputs.get("rawPerPA"))
    n = _nv(row.get("expectedPA"))
    if raw_per_pa is None or n is None:
        return False
    wrote = False
    stat_out = {}
    for point, books in odds_for_player.items():
        k = O.milestone_threshold_to_k(point)
        if k is None or k not in THRESHOLDS_K:
            continue
        raw_p, cal_p = M.milestone_prob(raw_per_pa, n, k, cal_hr)
        entry = {"point": point, "modelRaw": raw_p, "modelFair": cal_p, "books": {}}
        for book_key, prices in books.items():
            over, under = prices.get("over"), prices.get("under")
            book_fair_over, _ = O.devig_two_way(over, under) if under is not None else (None, None)
            edge = O.compute_edge(cal_p, over) if over is not None else None
            entry["books"][book_key] = {
                "overPrice": over, "underPrice": under,
                "bookFairProb": book_fair_over, "edge": edge,
            }
            wrote = True
            if (edge is not None and 0.08 <= edge < EDGE_SANITY_CEILING and point == 0.5
                    and row.get("confidence") == "high"
                    and not any("small sample" in str(x.get("label", "")).lower()
                                for x in (row.get("hrRisks") or []))):
                row.setdefault("angles", []).append({
                    "key": f"book_edge_{book_key}",
                    "label": f"HR edge vs {book_key}: model {cal_p*100:.1f}% vs "
                             f"{'+' if over>0 else ''}{over:.0f} ({edge*100:+.1f}%)",
                    "cls": "green",
                })
        stat_out[str(point)] = entry
    if stat_out:
        row.setdefault("bookOdds", {})["hr"] = stat_out
    return wrote


def archive_board(board, today):
    try:
        os.makedirs("boards", exist_ok=True)
        atomic_write_json(os.path.join("boards", f"{today}.json"), board)
    except Exception as e:
        print(f"  archive re-write failed (live board is fine): {type(e).__name__}: {e}",
              file=sys.stderr)


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
    api_key = os.environ.get("SPORTSGAMEODDS_API_KEY")
    if not api_key:
        print("SPORTSGAMEODDS_API_KEY not set -- skipping odds refresh", file=sys.stderr)
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

    cal_hr = None
    try:
        with open(CAL_PATH) as f:
            cal = json.load(f)
        if isinstance(cal, dict) and cal.get("modelVersion") == M.MODEL_VERSION:
            cal_hr = cal.get("hr")
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    try:
        events = fetch_slate_odds(api_key)
    except Exception as e:
        print(f"  odds fetch failed: {type(e).__name__}: {e}", file=sys.stderr)
        return
    if not events:
        print("No MLB events returned -- nothing to patch")
        return

    by_norm_name = {}
    for g in board.get("games", []):
        for side in ("homeMatchups", "awayMatchups"):
            for row in (g.get(side) or []):
                by_norm_name.setdefault(O.norm_name(row.get("name")), []).append(row)

    events_matched = 0
    rows_patched = 0
    for ev in events:
        game = match_event_to_game(ev, board.get("games", []))
        if not game:
            continue
        parsed = O.parse_event_odds(ev)
        events_matched += 1
        for player_norm, odds_for_player in parsed.items():
            rows = by_norm_name.get(player_norm)
            if not rows:
                continue
            for row in rows:
                if apply_odds_to_row(row, odds_for_player, cal_hr):
                    rows_patched += 1

    if events_matched == 0:
        print("Matched 0 events to tonight's board -- team-name matching may need a look")
        return

    board["oddsRefreshedAt"] = datetime.datetime.now(ET).isoformat(timespec="minutes")
    board["oddsEventsMatched"] = events_matched
    board["oddsProvider"] = "sportsgameodds"
    atomic_write_json(path, board)
    archive_board(board, today)
    print(f"Odds patched: {events_matched} events matched, {rows_patched} player-rows updated.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ODDS REFRESH FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
