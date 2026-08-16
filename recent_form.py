#!/usr/bin/env python3
# ============================================================================
# recent_form.py -- recent-form layer for the daily board (DTP components 1-4).
#
# ONE bulk Statcast pull (last ~LOOKBACK_DAYS of pitch-level data) yields:
#   - Per batter: last-10-games batted-ball aggregates -- barrel%, hard-hit%,
#     air/FB/LD/GB%, xISO-on-contact, HRs, near-HRs -- and a DTP power PROFILE.
#   - Per pitcher: last-3-starts pitch mix with per-pitch damage allowed, for
#     mix-DRIFT detection vs the season arsenal and a recent-mix arsenal fit.
#
# DESIGN RULES:
#   - REFERENCE ONLY. Nothing here touches log5 or raw probabilities;
#     MODEL_VERSION does not bump. These ship as angles + display fields, get
#     stamped onto ledger rows at settle time, and graduate into the model only
#     if the ledger later shows real residual lift (the v5.5+ decision is made
#     with evidence, not vibes).
#   - Small-sample honesty: 10 games is ~25-35 BBE; barrel% at that n swings
#     +/-5 points on noise alone. Profiles therefore require MIN_BBE and are
#     flags, never numbers that move probability. Near-HRs are kept because
#     they are direct evidence of HR-quality contact the box score hid.
#   - Pitch mix is a DECISION, not luck -- it stabilizes in one start. That is
#     why last-3-start usage is trustworthy where 30-BBE batter stats are not.
#   - Pure compute functions take DataFrames; the network lives in
#     fetch_statcast() alone, so everything else unit-tests offline.
#
# PROFILE THRESHOLDS (v1 -- documented so future-you can re-tune against the
# ledger; priority order top to bottom, first match wins, MIN_BBE gate first):
#   insane     barrel% >= 18 and (HR + nearHR) >= 3
#   elite      barrel% >= 12 and air% >= 50
#   flyball    FB% >= 45 and hardHit% >= 40
#   line_drive LD% >= 28 and xISO >= .200
# ============================================================================

import datetime
import sys
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

LOOKBACK_DAYS = 12          # covers ~10 team games plus off-days
MIN_BBE = 15                # below this, no profile is assigned
RECENT_STARTS = 3           # pitcher window
DRIFT_PP = 8.0              # usage change (percentage points) that counts as drift
CORE_USAGE = 15.0           # DTP rule 4: attack pitches thrown >= 15%
CRUSHED_XSLG = 0.550        # recent xSLG-allowed on a core pitch = liability

PROFILES = {
    "insane":     {"emoji": "\U0001F4A3", "label": "Insane power profile"},
    "elite":      {"emoji": "\U0001F680", "label": "Elite power profile"},
    "flyball":    {"emoji": "\U0001F357", "label": "Fly-ball power profile"},
    "line_drive": {"emoji": "\U0001F3AF", "label": "Line-drive power profile"},
}


STATCAST_CHUNK_DAYS = 4   # see fetch_statcast() -- this is the fix for a real
                          # 2026-08-13 outage, not a preemptive guess

def fetch_statcast(days=LOOKBACK_DAYS):
    """Bulk pitch-level Statcast for the trailing window, pulled in
    STATCAST_CHUNK_DAYS-sized pieces and concatenated, not one request for the
    whole window.

    ROOT CAUSE THIS FIXES: on 2026-08-13 a single statcast() call over the
    full LOOKBACK_DAYS window failed with Baseball Savant's own "Query
    Timeout. Please try to limit your query to less data" -- and because this
    is the ONE network call the whole module makes, that single failure took
    out batter_form, pitcher_recent, lineup_slots, AND bullpen_fatigue
    together (98 of 372 hitters fell back to a league-default slot that
    morning, with zero projected slots at all). Same chunking discipline
    zone_engine.py's update_cache() already uses for its incremental backfill
    (chunk_days=7 there) -- smaller requests avoid the timeout in the first
    place, and this module gets the added benefit that a bad chunk now costs
    only that chunk's days, not the whole pull.

    DELIBERATELY DIFFERENT from zone_engine.update_cache()'s chunk loop in one
    way: that loop STOPS on the first failed chunk, because it advances a
    sequential asOf watermark that must never claim coverage it doesn't have.
    This function has no such invariant -- it's gathering a trailing window,
    not extending a cache -- so a failed chunk is skipped and the loop
    continues, maximizing how much of the window survives one bad request
    instead of discarding everything after it.
    """
    from pybaseball import statcast
    import pandas as pd
    end = datetime.datetime.now(ET).date()
    start = end - datetime.timedelta(days=days)

    frames = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + datetime.timedelta(days=STATCAST_CHUNK_DAYS - 1), end)
        try:
            piece = statcast(start_dt=cur.isoformat(), end_dt=chunk_end.isoformat())
            if piece is not None and len(piece):
                frames.append(piece)
        except Exception as e:
            print(f"WARN: statcast chunk {cur}..{chunk_end} failed "
                  f"({type(e).__name__}: {e}) -- continuing with remaining chunks",
                  file=sys.stderr)
        cur = chunk_end + datetime.timedelta(days=1)

    if not frames:
        raise RuntimeError(
            f"all statcast chunks failed for the {days}-day window -- no data to fold")
    return pd.concat(frames, ignore_index=True)


def _col(df, name):
    return name if name in df.columns else None


def _iv(v):
    """NA/malformed-safe int coercion for groupby keys (batter/pitcher ids).
    Real Statcast data never sends a non-numeric id -- this exists because a
    stress test surfaced that int(pid) crashes if it ever did, a latent
    assumption batter_form()/lineup_slots() already share and this doesn't
    change; new code here just doesn't inherit the same crash risk."""
    try:
        import pandas as _pd
        if v is None or _pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _f(v):
    """NA-safe float coercion.

    The old body was `f = float(v); return f if f == f else None`, which broke
    on 2026-08-08: modern pybaseball hands back pandas NULLABLE dtypes, where a
    column that is entirely null has a mean of pd.NA. float(pd.NA) raises
    TypeError -- caught here -- but the same self-comparison idiom used inline
    elsewhere raised "boolean value of NA is ambiguous" and killed the whole
    recent-form pull, taking L10 profiles, pitch mix, pull%, AND lineup slots
    with it. Never test NA-ness with `x == x`; always use pd.isna()."""
    try:
        import pandas as _pd
        if v is None or _pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


# ----------------------------- batter form (1+2) -----------------------------

def batter_form(df):
    """DataFrame -> {batterId: form dict}. Last 10 games of batted-ball events
    per batter. barrel via launch_speed_angle==6 (Savant's own classification);
    xISO-on-contact = mean(xSLG) - mean(xBA) over BBE, which sidesteps needing
    AB accounting from pitch-level rows."""
    need = ["batter", "game_pk", "type"]
    if any(_col(df, c) is None for c in need):
        return {}
    bbe = df[(df["type"].astype(str) == "X")].copy()   # 'X' = ball put in play
    if bbe.empty:
        return {}
    has = {c: _col(bbe, c) for c in
           ["game_date", "events", "bb_type", "launch_speed", "launch_angle",
            "launch_speed_angle", "estimated_ba_using_speedangle",
            "estimated_slg_using_speedangle", "hit_distance_sc",
            "hc_x", "hc_y", "stand"]}
    out = {}
    for pid, g in bbe.groupby("batter"):
        # last 10 distinct games for THIS batter
        if has["game_date"]:
            order = g.groupby("game_pk")[has["game_date"]].max().sort_values()
            last_games = set(order.index[-10:])
        else:
            last_games = set(g["game_pk"].unique()[-10:])
        g = g[g["game_pk"].isin(last_games)]
        n = len(g)
        if n == 0:
            continue
        ev = g[has["events"]].astype(str) if has["events"] else None
        hr = int((ev == "home_run").sum()) if ev is not None else 0
        barrels = int((g[has["launch_speed_angle"]] == 6).sum()) if has["launch_speed_angle"] else 0
        hard = int((g[has["launch_speed"]] >= 95).sum()) if has["launch_speed"] else 0
        fb = ld = gb = 0
        if has["bb_type"]:
            bt = g[has["bb_type"]].astype(str)
            fb = int((bt == "fly_ball").sum())
            ld = int((bt == "line_drive").sum())
            gb = int((bt == "ground_ball").sum())
        near = 0
        if ev is not None and has["launch_speed"] and has["launch_angle"]:
            not_hr = ev != "home_run"
            ls, la = g[has["launch_speed"]], g[has["launch_angle"]]
            cond = not_hr & (ls >= 100) & (la >= 20) & (la <= 38)
            if has["hit_distance_sc"]:
                cond = cond | (not_hr & (g[has["hit_distance_sc"]] >= 385))
            near = int(cond.sum())
        pull_pct = _pull_pct(g, has)
        x_iso = None
        if has["estimated_ba_using_speedangle"] and has["estimated_slg_using_speedangle"]:
            xba = _f(g[has["estimated_ba_using_speedangle"]].mean())
            xslg = _f(g[has["estimated_slg_using_speedangle"]].mean())
            if xba is not None and xslg is not None:
                x_iso = round(xslg - xba, 3)
        form = {
            "bbe": n,
            "games": len(last_games),
            "hr": hr,
            "nearHr": near,
            "barrelPct": round(barrels / n * 100, 1),
            "hardHitPct": round(hard / n * 100, 1),
            "fbPct": round(fb / n * 100, 1),
            "ldPct": round(ld / n * 100, 1),
            "gbPct": round(gb / n * 100, 1),
            "airPct": round((fb + ld) / n * 100, 1),
            "pullPct": pull_pct,
            "xIso": x_iso,
        }
        form["profile"] = classify_profile(form)
        out[int(pid)] = form
    return out


_HOME_X, _HOME_Y = 125.42, 198.27
_PULL_DEG = 15.0

def _pull_pct(g, has):
    if not (has["hc_x"] and has["hc_y"] and has["stand"]):
        return None
    import pandas as _pd
    x = _pd.to_numeric(g[has["hc_x"]], errors="coerce").astype(float)
    y = _pd.to_numeric(g[has["hc_y"]], errors="coerce").astype(float)
    side = g[has["stand"]].astype(str)
    dy = (_HOME_Y - y)
    valid = (dy > 1) & x.notna() & y.notna()
    valid = valid.fillna(False)
    if int(valid.sum()) < 5:
        return None
    import math as _m
    angle = ((x[valid] - _HOME_X) / dy[valid]).apply(lambda v: _m.atan(v) * 180 / _m.pi * 0.75)
    stand = side[valid]
    pulled = ((stand == "R") & (angle < -_PULL_DEG)) | ((stand == "L") & (angle > _PULL_DEG))
    pulled = pulled.fillna(False)
    return round(float(pulled.sum()) / len(angle) * 100, 1)


def classify_profile(f):
    if f["bbe"] < MIN_BBE:
        return None
    if f["barrelPct"] >= 18 and (f["hr"] + f["nearHr"]) >= 3:
        return "insane"
    if f["barrelPct"] >= 12 and f["airPct"] >= 50:
        return "elite"
    if f["fbPct"] >= 45 and f["hardHitPct"] >= 40:
        return "flyball"
    if f["ldPct"] >= 28 and (f["xIso"] is not None and f["xIso"] >= 0.200):
        return "line_drive"
    return None


# --------------------------- bat speed (swing mechanics) ---------------------

BAT_SPEED_MIN_SWINGS = 15   # same role as MIN_BBE -- floor before an average
                            # means anything, same magnitude given a similar
                            # per-game swing count to BBE count

def bat_speed_form(df):
    """DataFrame -> {batterId: {"swings": n, "games": k, "avgBatSpeed": mph,
    "avgSwingLength": ft}} over the batter's last 10 distinct games -- same
    recency window batter_form() uses, for the same reason (DTP's whole
    premise is that recent form is a real, separate read from season rates).

    DELIBERATELY NOT restricted to balls in play, unlike batter_form(): bat
    speed and swing length are tracked by Hawk-Eye on every swing it follows,
    contact or not -- a swing that missed still tells you about bat speed,
    and restricting to BBE would silently bias the average toward swings that
    already succeeded. Filtering on bat_speed.notna() rather than enumerating
    which 'description' values count as a swing is deliberate: it lets the
    data define what's tracked instead of guessing at Statcast's exact
    swing-vs-take vocabulary, same instinct as _f()'s NA-safety fix above.

    REFERENCE ONLY, same footing as every other signal in this module -- an
    unproven angle until measure_signals.py's ledger check says otherwise.
    """
    need = ["batter", "game_pk", "bat_speed"]
    if any(_col(df, c) is None for c in need):
        return {}
    import pandas as _pd
    speed_col = _col(df, "bat_speed")
    length_col = _col(df, "swing_length")
    speed_num = _pd.to_numeric(df[speed_col], errors="coerce")
    swings = df[speed_num.notna()].copy()
    if swings.empty:
        return {}
    has_date = _col(swings, "game_date")
    out = {}
    for pid, g in swings.groupby("batter"):
        pid = _iv(pid)
        if pid is None:
            continue
        if has_date:
            order = g.groupby("game_pk")[has_date].max().sort_values()
            last_games = set(order.index[-10:])
        else:
            last_games = set(g["game_pk"].unique()[-10:])
        g = g[g["game_pk"].isin(last_games)]
        n = len(g)
        if n < BAT_SPEED_MIN_SWINGS:
            continue
        avg_speed = _f(g[speed_col].mean())
        avg_length = _f(g[length_col].mean()) if length_col else None
        out[pid] = {
            "swings": n,
            "games": len(last_games),
            "avgBatSpeed": round(avg_speed, 1) if avg_speed is not None else None,
            "avgSwingLength": round(avg_length, 2) if avg_length is not None else None,
        }
    return out


# ------------------------------ lineup slots ---------------------------------

SLOT_DECAY = 0.88
MIN_SLOT_GAMES = 3


def lineup_slots(df):
    need = ["batter", "game_pk", "at_bat_number", "inning_topbot"]
    if any(_col(df, c) is None for c in need):
        return {}
    has_date = _col(df, "game_date")
    first = df

    per_batter = {}
    team_games = {}
    game_dates = {}
    for (gpk, half), g in first.groupby(["game_pk", "inning_topbot"]):
        g = g.sort_values("at_bat_number")
        seen = []
        for b in g["batter"]:
            b = int(b)
            if b not in seen:
                seen.append(b)
            if len(seen) == 9:
                break
        if len(seen) < 9:
            continue
        key = (gpk, half)
        team_games[key] = True
        if has_date:
            game_dates[key] = str(g[has_date].iloc[0])
        for slot, b in enumerate(seen, start=1):
            per_batter.setdefault(b, []).append((key, slot))

    ordered = sorted(team_games.keys(),
                     key=lambda k: game_dates.get(k, ""), reverse=True)
    rank = {k: i for i, k in enumerate(ordered)}

    club_games = {}
    for b, entries in per_batter.items():
        club_games[b] = len({k for k, _ in entries})

    out = {}
    for b, entries in per_batter.items():
        entries.sort(key=lambda e: rank.get(e[0], 999))
        wsum = 0.0
        vsum = 0.0
        for i, (key, slot) in enumerate(entries):
            w = SLOT_DECAY ** rank.get(key, i)
            wsum += w
            vsum += w * slot
        if not wsum or len(entries) < MIN_SLOT_GAMES:
            continue
        out[b] = {
            "orderAvg": round(vsum / wsum, 2),
            "games": len(entries),
            "lastSlot": entries[0][1],
        }
    if out:
        maxg = max(v["games"] for v in out.values()) or 1
        for v in out.values():
            v["startPct"] = round(min(1.0, v["games"] / maxg) * 100, 1)
    return out


# ---------------------------- pitcher recent (3) -----------------------------

SHAPE_MIN_N = 8   # floor before a pitch's recent velo/spin/movement average
                  # means anything -- lower than CORE_USAGE's implied count
                  # deliberately: shape decay on even a show-me pitch is worth
                  # surfacing, unlike usage/xSlg-based angles which need it to
                  # be a CORE pitch to matter

def pitcher_recent(df):
    """DataFrame -> {pitcherId: {"starts": k, "pitches": n, "mix": [
    {"pitch", "usage", "xSlg", "hr", "n", "avgVelo", "avgSpin", "avgExtension",
    "avgHBreak", "avgVBreak"}]}} over each pitcher's last RECENT_STARTS games.
    xSlg is mean xSLG allowed on BBE off that pitch -- 'is this pitch getting
    crushed lately'. The shape fields (added alongside, same window) are
    'is this pitch MOVING like it used to' -- a different and earlier
    question than xSlg: velocity/spin/movement are measured on every pitch of
    that type thrown in the window, not just the ones that got hit, so a
    pitch can show shape decay before the results catch up. release_speed/
    release_spin_rate/release_extension/pfx_x/pfx_z are standard raw Statcast
    columns already present on every row this module's fetch_statcast()
    pulls -- no new network call, same instinct as bat_speed_form() above."""
    need = ["pitcher", "game_pk", "pitch_name"]
    if any(_col(df, c) is None for c in need):
        return {}
    has_date = _col(df, "game_date")
    has_ev = _col(df, "events")
    has_xslg = _col(df, "estimated_slg_using_speedangle")
    has_type = _col(df, "type")
    has_velo = _col(df, "release_speed")
    has_spin = _col(df, "release_spin_rate")
    has_ext = _col(df, "release_extension")
    has_hbreak = _col(df, "pfx_x")
    has_vbreak = _col(df, "pfx_z")
    out = {}
    for pid, g in df.groupby("pitcher"):
        pid = _iv(pid)
        if pid is None:
            continue
        if has_date:
            order = g.groupby("game_pk")[has_date].max().sort_values()
            recent_games = list(order.index[-RECENT_STARTS:])
        else:
            recent_games = list(g["game_pk"].unique()[-RECENT_STARTS:])
        g = g[g["game_pk"].isin(recent_games)]
        total = len(g)
        if total < 30:
            continue
        mix = []
        for pitch, pg in g.groupby("pitch_name"):
            pitch = str(pitch).strip()
            if not pitch or pitch == "nan":
                continue
            entry = {"pitch": pitch, "usage": round(len(pg) / total * 100, 1), "n": len(pg)}
            if has_ev:
                entry["hr"] = int((pg[has_ev].astype(str) == "home_run").sum())
            if has_xslg and has_type:
                bip = pg[pg[has_type] == "X"]
                if len(bip) >= 5:
                    xs = _f(bip[has_xslg].mean())
                    if xs is not None:
                        entry["xSlg"] = round(xs, 3)
            if len(pg) >= SHAPE_MIN_N:
                if has_velo:
                    v = _f(pg[has_velo].mean())
                    if v is not None:
                        entry["avgVelo"] = round(v, 1)
                if has_spin:
                    s = _f(pg[has_spin].mean())
                    if s is not None:
                        entry["avgSpin"] = round(s, 0)
                if has_ext:
                    e = _f(pg[has_ext].mean())
                    if e is not None:
                        entry["avgExtension"] = round(e, 2)
                if has_hbreak:
                    hb = _f(pg[has_hbreak].mean())
                    if hb is not None:
                        entry["avgHBreak"] = round(hb, 2)
                if has_vbreak:
                    vb = _f(pg[has_vbreak].mean())
                    if vb is not None:
                        entry["avgVBreak"] = round(vb, 2)
            mix.append(entry)
        mix.sort(key=lambda m: m["usage"], reverse=True)
        out[pid] = {"starts": len(recent_games), "pitches": total,
                    "gamePks": [int(x) for x in recent_games], "mix": mix[:8]}
    return out


# ------------------------- drift + recent fit (3+4) --------------------------

def mix_drift(recent, season_mix):
    drifts, crushed = [], []
    if not recent:
        return drifts, crushed
    season_by = {str(m.get("pitch")).strip().lower(): _f(m.get("usage"))
                 for m in (season_mix or []) if m.get("pitch")}
    for m in recent.get("mix", []):
        key = m["pitch"].strip().lower()
        season_u = season_by.get(key)
        if season_u is not None and abs(m["usage"] - season_u) >= DRIFT_PP:
            drifts.append({"pitch": m["pitch"], "recent": m["usage"],
                           "season": round(season_u, 1),
                           "delta": round(m["usage"] - season_u, 1)})
        if m["usage"] >= CORE_USAGE and (
                (m.get("xSlg") is not None and m["xSlg"] >= CRUSHED_XSLG)
                or (m.get("hr", 0) >= 2)):
            crushed.append({"pitch": m["pitch"], "usage": m["usage"],
                            "xSlg": m.get("xSlg"), "hr": m.get("hr", 0)})
    recent_keys = {m["pitch"].strip().lower() for m in recent.get("mix", [])}
    for key, season_u in season_by.items():
        if season_u is not None and season_u >= DRIFT_PP and key not in recent_keys:
            drifts.append({"pitch": key.title(), "recent": 0.0,
                           "season": round(season_u, 1),
                           "delta": round(-season_u, 1)})
    return drifts, crushed


SHAPE_DRIFT_VELO_MPH = 1.5    # recent velo down this much vs season = notable
SHAPE_DRIFT_SPIN_RPM = 150    # recent spin down this much vs season = notable

def pitch_shape_drift(recent, season_shape):
    """recent: one pitcher's pitcher_recent() entry (has 'mix' with avgVelo/
    avgSpin per pitch, when SHAPE_MIN_N cleared). season_shape: {pitchName
    (any case): {"avgVelo":.., "avgSpin":..}} -- a season baseline supplied
    BY THE CALLER. This module makes no network calls and has no opinion
    about where the baseline comes from; build_daily_board.py owns that.

    Returns entries for CORE pitches (>=15% recent usage -- DTP rule 4, same
    bar mix_drift's crushed-pitch check already uses) whose velo or spin
    DROPPED enough to flag vs that pitcher's own season number for the same
    pitch. One-directional on purpose, unlike mix_drift()'s usage check:
    throwing harder or spinning more than usual isn't a liability signal, so
    only decay in one direction is worth surfacing here."""
    out = []
    if not recent:
        return out
    season_by = {str(k).strip().lower(): v for k, v in (season_shape or {}).items()}
    for m in recent.get("mix", []):
        if _f(m.get("usage")) is None or m["usage"] < CORE_USAGE:
            continue
        key = m["pitch"].strip().lower()
        s = season_by.get(key)
        if not s:
            continue
        row = {"pitch": m["pitch"]}
        flagged = False
        rv, sv = m.get("avgVelo"), s.get("avgVelo")
        if rv is not None and sv is not None and (sv - rv) >= SHAPE_DRIFT_VELO_MPH:
            row.update({"veloRecent": rv, "veloSeason": sv, "veloDelta": round(rv - sv, 1)})
            flagged = True
        rs, ss = m.get("avgSpin"), s.get("avgSpin")
        if rs is not None and ss is not None and (ss - rs) >= SHAPE_DRIFT_SPIN_RPM:
            row.update({"spinRecent": rs, "spinSeason": ss, "spinDelta": round(rs - ss, 0)})
            flagged = True
        if flagged:
            out.append(row)
    return out


def core_mix(mix, min_usage=CORE_USAGE):
    return [m for m in (mix or []) if _f(m.get("usage")) and m["usage"] >= min_usage]


if __name__ == "__main__":
    df = fetch_statcast()
    b = batter_form(df)
    p = pitcher_recent(df)
    profiled = sum(1 for f in b.values() if f.get("profile"))
    print(f"statcast rows={len(df)}  batters={len(b)} (profiled {profiled})  pitchers={len(p)}",
          file=sys.stderr)
