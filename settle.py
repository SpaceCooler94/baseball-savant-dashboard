#!/usr/bin/env python3
# ============================================================================
# settle.py -- nightly settlement job for the calibration ledger.
#
# Reads the archived board for a given date (boards/YYYY-MM-DD.json), pulls the
# final box scores from StatsAPI, and appends one JSONL row per hitter to
# ledger/ledger.jsonl pairing the model's RAW per-game predictions with what
# actually happened. calibrate.py fits on these rows; later, odds can be joined
# onto the same rows for ROI/CLV tracking.
#
# SIM JOIN (added): if boards/props_YYYY-MM-DD.json exists for the same date,
# the hrhit Monte Carlo probabilities are stamped onto the SAME row as extra
# fields (simHitRaw / simHrRaw / simModelVersion). They are NOT separate rows:
# calibrate.py dedupes on (date, hitterId), so a second row per hitter would
# silently clobber the log5 row. One row, one outcome, two predictions.
#
# DESIGN RULES (do not relax casually):
#   - VOID RULE: hitters with 0 actual PA are excluded (counted, not settled).
#     A scratch is not a model miss, and sportsbooks void those props too --
#     the ledger mirrors settlement reality. Sim fields inherit this for free.
#   - RAW ONLY: predictions recorded are hitRawPerGame/hrRawPerGame (pre-
#     calibration). Fitting on calibrated outputs is a feedback loop. For 5.3
#     boards (no rawPerGame field) it is reconstructed from inputs.rawPerPA +
#     expectedPA -- identical math, since 5.3 calibration was never applied.
#     Sim rows follow the same rule: pHRraw/pHitRaw preferred, falling back to
#     pHR/pHit for boards built before hrhit learned to self-calibrate.
#   - SIM STALENESS GUARD: a props board whose own "date" != the settle date is
#     ignored entirely. A stale board (cron ran before lineups posted, so the
#     file still holds an older slate) would otherwise pair yesterday's
#     predictions with today's outcomes -- worse than no data at all.
#   - IDEMPOTENT: a date already in ledger/settled_dates.json is skipped, so
#     re-runs and workflow retries can't double-write rows.
#   - Rows carry modelVersion from the board; the fit filters on it. Sim rows
#     carry simModelVersion SEPARATELY so a log5 bump doesn't discard sim
#     history (and vice versa).
#
# Usage: python settle.py [YYYY-MM-DD]   (default: yesterday, US/Eastern)
# Exit codes: 0 = settled or cleanly skipped; 1 = hard failure.
# ============================================================================

import datetime
import json
import os
import random
import sys
import time
from zoneinfo import ZoneInfo

import requests

ET = ZoneInfo("America/New_York")
STATS_API = "https://statsapi.mlb.com/api/v1"
BOARDS_DIR = "boards"
LEDGER_DIR = "ledger"
LEDGER_PATH = os.path.join(LEDGER_DIR, "ledger.jsonl")
SETTLED_PATH = os.path.join(LEDGER_DIR, "settled_dates.json")

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "mlb-daily-board-settle/1.1 (personal analytics pipeline)"})


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
# Kept side-effect-free so they can be unit-tested without a network.

def parse_boxscore(box):
    """gamePk boxscore -> {hitterId: {"pa": int, "hits": int, "hr": int}}.
    Only players with a batting stat line appear; pitchers who never batted and
    inactive players simply won't be present, which the void rule handles."""
    out = {}
    for side in ("home", "away"):
        players = (((box.get("teams") or {}).get(side) or {}).get("players") or {})
        for key, pdata in players.items():
            pid = (pdata.get("person") or {}).get("id")
            bat = ((pdata.get("stats") or {}).get("batting") or {})
            if pid is None or not bat:
                continue
            pa = nv(bat.get("plateAppearances"))
            if pa is None:
                continue
            out[int(pid)] = {
                "pa": int(pa),
                "hits": int(nv(bat.get("hits")) or 0),
                "hr": int(nv(bat.get("homeRuns")) or 0),
            }
    return out


def raw_per_game(row, market):
    """Raw (pre-calibration) per-game probability for a ledger row.
    5.4+ boards carry it directly; 5.3 boards get it reconstructed from
    rawPerPA + expectedPA (exact, because 5.3 never applied calibration)."""
    direct = row.get(market + "RawPerGame")
    if direct is not None:
        return direct
    inputs = row.get(market + "Inputs") or {}
    raw_pa = nv(inputs.get("rawPerPA"))
    n = nv(row.get("expectedPA"))
    if raw_pa is None or n is None:
        return None
    return round(1 - (1 - max(0.001, min(0.999, raw_pa))) ** n, 3)


def load_sim_board(date_str, boards_dir=BOARDS_DIR):
    """Archived hrhit props board for date_str -> ({hitterId: player}, version).
    Returns ({}, None) when absent, unreadable, empty, or -- critically -- when
    the board's own date does not match: a stale props_*.json from a run that
    fired before lineups posted must never be paired with today's outcomes."""
    path = os.path.join(boards_dir, f"props_{date_str}.json")
    if not os.path.exists(path):
        return {}, None
    try:
        with open(path) as f:
            board = json.load(f)
    except (json.JSONDecodeError, OSError):
        print(f"WARN: unreadable sim board {path}", file=sys.stderr)
        return {}, None
    if board.get("date") != date_str:
        print(f"WARN: sim board date {board.get('date')} != {date_str} -- ignoring (stale)",
              file=sys.stderr)
        return {}, None
    by_id = {}
    for p in board.get("players") or []:
        pid = p.get("id")
        if pid is not None:
            by_id[int(pid)] = p
    return by_id, board.get("model")


def sim_raw(player, market):
    """Raw (pre-calibration) sim probability. hrhit >= 1.1 publishes pHRraw /
    pHitRaw alongside the calibrated pHR / pHit; older boards only have the
    latter, which were uncalibrated anyway, so the fallback is exact."""
    pub, raw = ("pHR", "pHRraw") if market == "hr" else ("pHit", "pHitRaw")
    v = player.get(raw)
    if v is None:
        v = player.get(pub)
    return nv(v)


def settle_rows(board, outcomes_by_game, date_str, sim_by_id=None, sim_version=None):
    """Board + parsed boxscores (+ optional sim board) -> (ledger_rows, stats)."""
    rows = []
    voided = 0
    missing_pred = 0
    sim_matched = 0
    sim_by_id = sim_by_id or {}
    model_version = board.get("modelVersion") or ("schema-%s" % board.get("schemaVersion"))
    for g in board.get("games", []):
        game_id = g.get("gameId")
        outcomes = outcomes_by_game.get(game_id) or {}
        for r in (g.get("homeMatchups") or []) + (g.get("awayMatchups") or []):
            pid = r.get("hitterId")
            oc = outcomes.get(pid)
            if not oc or oc["pa"] <= 0:
                voided += 1  # scratch / DNP / no batting line -> void, not a miss
                continue
            hit_raw = raw_per_game(r, "hit")
            hr_raw = raw_per_game(r, "hr")
            if hit_raw is None or hr_raw is None:
                missing_pred += 1
                continue
            row = {
                "date": date_str,
                "gameId": game_id,
                "hitterId": pid,
                "name": r.get("name"),
                "modelVersion": model_version,
                "hitRaw": hit_raw,
                "hrRaw": hr_raw,
                "hitPred": r.get("hitProb"),   # as-published (calibrated if it was)
                "hrPred": r.get("hrProb"),
                "hitTier": r.get("hitTier"),
                "hrTier": r.get("hrTier"),
                "confidence": r.get("confidence"),
                "expectedPA": r.get("expectedPA"),
                "actualPA": oc["pa"],
                "gotHit": 1 if oc["hits"] > 0 else 0,
                "gotHR": 1 if oc["hr"] > 0 else 0,
                # Angle stamps: measure each angle's residual lift later by
                # comparing observed rates for flagged vs unflagged rows at
                # the same raw probability. Absent on pre-5.5 boards -> [].
                "angles": [a.get("key") for a in (r.get("angles") or []) if a.get("key")],
                "profile": (r.get("recentForm") or {}).get("profile"),
                "zoneScore": r.get("zoneScore"),
            }
            # --- sim join: same row, same outcome, second prediction source ---
            sp = sim_by_id.get(pid)
            if sp:
                s_hit = sim_raw(sp, "hit")
                s_hr = sim_raw(sp, "hr")
                if s_hit is not None and s_hr is not None:
                    row["simHitRaw"] = s_hit
                    row["simHrRaw"] = s_hr
                    row["simModelVersion"] = sim_version
                    sim_matched += 1
            rows.append(row)
    return rows, {"settled": len(rows), "voided": voided, "missingPred": missing_pred,
                  "simMatched": sim_matched, "modelVersion": model_version,
                  "simVersion": sim_version}


# --------------------------------- job glue ----------------------------------

def load_settled():
    try:
        with open(SETTLED_PATH) as f:
            return set(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def mark_settled(dates):
    os.makedirs(LEDGER_DIR, exist_ok=True)
    with open(SETTLED_PATH, "w") as f:
        json.dump(sorted(dates), f)


def main():
    date_str = sys.argv[1] if len(sys.argv) > 1 else \
        (datetime.datetime.now(ET) - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    settled = load_settled()
    if date_str in settled:
        print(f"{date_str} already settled -- skipping (idempotent)")
        return

    board_path = os.path.join(BOARDS_DIR, f"{date_str}.json")
    if not os.path.exists(board_path):
        print(f"No archived board at {board_path} -- nothing to settle")
        return
    with open(board_path) as f:
        board = json.load(f)

    sim_by_id, sim_version = load_sim_board(date_str)
    if sim_by_id:
        print(f"Sim board: {len(sim_by_id)} hitters, model={sim_version}")
    else:
        print("Sim board: none for this date (log5 rows only)")

    outcomes_by_game = {}
    failures = 0
    for g in board.get("games", []):
        gid = g.get("gameId")
        if not gid:
            continue
        try:
            box = http_json(f"{STATS_API}/game/{gid}/boxscore")
            outcomes_by_game[gid] = parse_boxscore(box)
        except Exception as e:
            failures += 1
            print(f"WARN: boxscore fetch failed game={gid} ({type(e).__name__})", file=sys.stderr)

    if failures and not outcomes_by_game:
        print("All boxscore fetches failed -- not marking settled, will retry next run",
              file=sys.stderr)
        sys.exit(1)

    rows, stats = settle_rows(board, outcomes_by_game, date_str, sim_by_id, sim_version)

    os.makedirs(LEDGER_DIR, exist_ok=True)
    with open(LEDGER_PATH, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    # Only mark fully settled when every game's boxscore was retrieved;
    # partial days stay unmarked so the next run completes them. The ledger
    # append is still idempotent-enough in practice because a retry re-appends
    # the same date -- calibrate.py dedupes on (date, hitterId) defensively.
    if failures == 0:
        settled.add(date_str)
        mark_settled(settled)

    print(f"Settled {date_str}: {stats['settled']} rows, {stats['voided']} voided (0 PA), "
          f"{stats['missingPred']} missing predictions, {stats['simMatched']} with sim, "
          f"{failures} boxscore failures, model={stats['modelVersion']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"SETTLE FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
