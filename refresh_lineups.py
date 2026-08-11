#!/usr/bin/env python3
# ============================================================================
# refresh_lineups.py -- upgrade the board from PROJECTED to CONFIRMED lineups.
#
# The daily build runs in the morning, hours before any lineup card is posted,
# so every hitter ships with a projected slot (recent_form.lineup_slots) and a
# "#N~" tilde in the UI. This job re-reads StatsAPI once the real lineups exist
# and patches daily_board.json in place.
#
# WHY THIS IS CHEAP: the only thing a confirmed lineup changes is expected PA.
# Since v5.4 the board stores the RAW per-PA rate on every row, and per-game is
# just 1-(1-rawPerPA)^expectedPA. So this script recomputes probabilities
# EXACTLY from data already in the payload -- one schedule call total, no
# StatsAPI player fetches, no Statcast, no model refetch. It reuses
# mlb_model.game_prob / apply_calibration / expected_pa so the arithmetic is
# literally the same code the build ran, not a reimplementation that could drift.
#
# WHAT IT CHANGES per patched row:
#   expectedPA, hitProb/hrProb (+ raw), tiers, slotSource -> "confirmed",
#   batOrderAvg -> the actual slot, lineupUnconfirmed -> False.
#   Hitters on a confirmed lineup card's bench get slotSource "scratched" and
#   are flagged, not deleted -- settle.py already voids 0-PA rows, and a
#   scratched row is information ("he's out"), not an error.
#
# SCHEDULING (all ET; lineups post ~2h45m before first pitch):
#   11:00am -> day games                     (~15% of the slate)
#    4:30pm -> + late afternoon, east night  (~70%; 4:00pm only reaches 35%,
#              because prime-time east lineups land around 4:20)
#    7:30pm -> + central and west            (100%)
# Each run patches whatever is confirmed AT THAT MOMENT and leaves the rest
# projected, so running it more often is always safe and never destructive.
#
# Usage: python refresh_lineups.py [path/to/daily_board.json]
# Exit 0 when nothing is confirmed yet -- that is a normal outcome, not failure.
# ============================================================================

import datetime
import json
import os
import random
import sys
import tempfile
import time
from zoneinfo import ZoneInfo

import requests

import mlb_model as M
# tier_hit/tier_hr used to be reimplemented HERE as a second copy of the
# cutoffs in mlb_model.py -- two definitions of one rule, and the one that
# drifts is the one silently grading confirmed-lineup rows wrong. Removed;
# M.tier_hit/M.tier_hr below are the only copy that exists now.

ET = ZoneInfo("America/New_York")
STATS_API = "https://statsapi.mlb.com/api/v1"
BOARD_PATH = "daily_board.json"
CAL_PATH = "calibration.json"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "mlb-daily-board-lineups/1.0 (personal analytics pipeline)"})


def http_json(url, tries=3, timeout=20):
    last = None
    for attempt in range(tries):
        try:
            r = SESSION.get(url, timeout=timeout)
            if r.status_code == 429:
                time.sleep(min(float(r.headers.get("Retry-After", 5)), 30))
                last = requests.HTTPError("429")
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            if attempt < tries - 1:
                time.sleep((2 ** attempt) + random.uniform(0, 0.5))
    raise last


def nv(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


# ------------------------------ pure transforms ------------------------------

def parse_lineups(sched):
    """schedule(hydrate=lineups) -> {gamePk: {batterId: slot 1-9}}.
    Only games whose card is actually posted appear; a partial card (fewer
    than 9) is ignored rather than half-applied."""
    out = {}
    for date in sched.get("dates", []):
        for g in date.get("games", []):
            gpk = g.get("gamePk")
            lu = g.get("lineups") or {}
            slots = {}
            for side in ("homePlayers", "awayPlayers"):
                players = lu.get(side) or []
                if len(players) < 9:
                    continue
                for i, p in enumerate(players[:9], start=1):
                    pid = p.get("id")
                    if pid:
                        slots[int(pid)] = i
            if slots:
                out[gpk] = slots
    return out


def repatch_row(row, slot, calibration):
    """Recompute one row for a known lineup slot. Returns True if changed.
    Uses the model's own game_prob/apply_calibration/expected_pa so this can
    never drift from what the build computed."""
    raw_hit = nv((row.get("hitInputs") or {}).get("rawPerPA"))
    raw_hr = nv((row.get("hrInputs") or {}).get("rawPerPA"))
    if raw_hit is None or raw_hr is None:
        return False
    n = M.expected_pa(slot)
    row["expectedPA"] = n
    row["batOrderAvg"] = slot
    row["slotSource"] = "confirmed"
    row["slotLast"] = slot
    row["lineupUnconfirmed"] = False

    rg_hit = M._round3(M.game_prob(raw_hit, n))
    rg_hr = M._round3(M.game_prob(raw_hr, n))
    row["hitRawPerGame"] = rg_hit
    row["hrRawPerGame"] = rg_hr
    cal = calibration or {}
    p_hit, _ = M.apply_calibration(rg_hit, cal.get("hit"))
    p_hr, _ = M.apply_calibration(rg_hr, cal.get("hr"))
    row["hitProb"] = M._round3(p_hit)
    row["hrProb"] = M._round3(p_hr)
    row["hitTier"] = M.tier_hit(row["hitProb"])
    row["hrTier"] = M.tier_hr(row["hrProb"])
    return True


def mark_scratched(row):
    """On a confirmed card but not in the nine. Kept and flagged, not removed:
    'he isn't playing' is information, and settle.py voids 0-PA rows anyway."""
    row["slotSource"] = "scratched"
    row["lineupUnconfirmed"] = False
    row["batOrderAvg"] = None


def apply_lineups(board, lineups, calibration):
    stats = {"gamesConfirmed": 0, "rowsConfirmed": 0, "rowsScratched": 0, "gamesPending": 0}
    for g in board.get("games", []):
        gpk = g.get("gameId")
        slots = lineups.get(gpk)
        if not slots:
            stats["gamesPending"] += 1
            continue
        stats["gamesConfirmed"] += 1
        for side in ("homeMatchups", "awayMatchups"):
            for row in (g.get(side) or []):
                pid = row.get("hitterId")
                slot = slots.get(pid)
                if slot:
                    if repatch_row(row, slot, calibration):
                        stats["rowsConfirmed"] += 1
                else:
                    mark_scratched(row)
                    stats["rowsScratched"] += 1
    return stats


def resort(board):
    """viewScore is a slate z-score; probabilities moved, so recompute it and
    re-sort. Mirrors build_daily_board.apply_view_scores exactly."""
    rows = []
    for g in board.get("games", []):
        rows.extend(g.get("homeMatchups") or [])
        rows.extend(g.get("awayMatchups") or [])
    if len(rows) < 2:
        return

    def zs(vals):
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        sd = var ** 0.5
        if sd < 1e-9:
            return [0.0] * n
        return [(v - mean) / sd for v in vals]

    zh = zs([r.get("hitProb") or 0 for r in rows])
    zr = zs([r.get("hrProb") or 0 for r in rows])
    for r, a, b in zip(rows, zh, zr):
        r["viewScore"] = round(a + b, 3)
    for g in board.get("games", []):
        for side in ("homeMatchups", "awayMatchups"):
            (g.get(side) or []).sort(key=lambda x: x.get("viewScore", 0), reverse=True)


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
    path = sys.argv[1] if len(sys.argv) > 1 else BOARD_PATH
    if not os.path.exists(path):
        print(f"No board at {path} -- nothing to refresh")
        return
    with open(path) as f:
        board = json.load(f)

    today = datetime.datetime.now(ET).strftime("%Y-%m-%d")
    if board.get("builtAt") != today:
        print(f"Board is for {board.get('builtAt')}, today is {today} -- "
              f"refusing to patch a stale board")
        return

    calibration = None
    try:
        with open(CAL_PATH) as f:
            cal = json.load(f)
        if isinstance(cal, dict) and cal.get("modelVersion") == M.MODEL_VERSION:
            calibration = cal
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    sched = http_json(f"{STATS_API}/schedule?sportId=1&date={today}&hydrate=lineups")
    lineups = parse_lineups(sched)
    if not lineups:
        print("No lineups posted yet -- board left projected (normal for an early run)")
        return

    stats = apply_lineups(board, lineups, calibration)
    if not stats["rowsConfirmed"]:
        print("Lineups found but no rows matched -- board unchanged")
        return

    resort(board)
    board["lineupRefreshedAt"] = datetime.datetime.now(ET).isoformat(timespec="minutes")
    board["lineupsConfirmedGames"] = stats["gamesConfirmed"]
    atomic_write_json(path, board)
    print(f"Lineups patched: {stats['gamesConfirmed']} games confirmed, "
          f"{stats['gamesPending']} still pending, "
          f"{stats['rowsConfirmed']} rows updated, {stats['rowsScratched']} scratched")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"LINEUP REFRESH FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
