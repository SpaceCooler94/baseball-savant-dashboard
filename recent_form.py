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


def fetch_statcast(days=LOOKBACK_DAYS):
    """Bulk pitch-level Statcast for the trailing window. The single network
    call of this module; heavy (~tens of thousands of rows) but one call."""
    from pybaseball import statcast
    end = datetime.datetime.now(ET).date()
    start = end - datetime.timedelta(days=days)
    return statcast(start_dt=start.isoformat(), end_dt=end.isoformat())


def _col(df, name):
    return name if name in df.columns else None


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
            # _f() rather than `== itself`: these means are pd.NA when the
            # column is all-null under nullable dtypes, and pd.NA in a boolean
            # context raises.
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


# Pull% via spray angle from hc_x/hc_y -- these are raw per-pitch Statcast
# coordinates (not a leaderboard aggregate), so they're already present in the
# bulk pull this module does; no extra network call. The hc_x/hc_y -> degrees
# formula below is the standard community reverse-engineering of Savant's
# coordinate system (MLB has never published it officially) and is what public
# tools like baseballr's calculate_spray_angle use. Left of center (negative
# angle) is pull for a RHB, right of center (positive) is pull for a LHB.
# REFERENCE metric: not fed into log5, same as everything else in this module.
_HOME_X, _HOME_Y = 125.42, 198.27
_PULL_DEG = 15.0   # +/- this band around center counts as "straightaway"

def _pull_pct(g, has):
    if not (has["hc_x"] and has["hc_y"] and has["stand"]):
        return None
    import pandas as _pd
    # to_numeric + astype(float) collapses nullable dtypes to plain NaN, so the
    # comparisons below can never produce an NA-bearing mask (which is not a
    # valid indexer and raises the same ambiguous-NA TypeError).
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
    """DTP profile buckets. Returns profile key or None. Priority order:
    insane > elite > flyball > line_drive. MIN_BBE gate first -- a profile off
    8 batted balls is astrology."""
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


# ------------------------------ lineup slots ---------------------------------

SLOT_DECAY = 0.88      # per game back; ~10-game effective window
MIN_SLOT_GAMES = 3     # below this, no orderAvg -- the model falls back to 3.9


def lineup_slots(df):
    """{batterId: {"orderAvg": float, "games": int, "lastSlot": int, "startPct": float}}

    Derived from the bulk Statcast pull already made for recent form -- NO extra
    network call. The trick: in the first inning, batters come up in lineup
    order, so the first nine distinct batters of each half-inning ARE that
    team's batting order. Anyone appearing later (pinch hitter, defensive sub)
    correctly gets no slot for that game.

    WHY THIS MATTERS MORE THAN IT SOUNDS: expected_pa() spans 4.6 PA (leadoff)
    to 3.75 (9-hole), which at league hit rates is ~8 points of hit probability.
    Until now every hitter was projected at 3.9 PA because lineups aren't posted
    at build time, so ALL of that variance was discarded -- more spread than the
    model's measured end-to-end discrimination (6.9 pts, per the 2026-08-08
    reliability table). Recovering it is the single largest available win.

    Slots are recency-weighted (SLOT_DECAY per game back) because managers move
    people; startPct is the share of the window's team-games this batter started,
    which is the honest read on whether he's a regular or a bench bat."""
    need = ["batter", "game_pk", "at_bat_number", "inning", "inning_topbot"]
    if any(_col(df, c) is None for c in need):
        return {}
    has_date = _col(df, "game_date")
    import pandas as _pd
    inning = _pd.to_numeric(df["inning"], errors="coerce")
    first = df[(inning == 1).fillna(False)]
    if first.empty:
        return {}

    # (game, half) -> ordered first nine distinct batters = that lineup
    per_batter = {}
    team_games = {}          # (game_pk, half) counted once, keyed per lineup
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
            continue         # incomplete half-inning; not a readable lineup
        key = (gpk, half)
        team_games[key] = True
        if has_date:
            game_dates[key] = str(g[has_date].iloc[0])
        for slot, b in enumerate(seen, start=1):
            per_batter.setdefault(b, []).append((key, slot))

    # order the windows newest-first so decay weights are meaningful
    ordered = sorted(team_games.keys(),
                     key=lambda k: game_dates.get(k, ""), reverse=True)
    rank = {k: i for i, k in enumerate(ordered)}

    # how many team-games each batter's club actually played in the window
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
    # startPct needs the team's game count, which we approximate by the max
    # games any of that batter's lineup-mates appeared in -- computed per game
    # set rather than per club, since we never resolve team identity here.
    if out:
        maxg = max(v["games"] for v in out.values()) or 1
        for v in out.values():
            v["startPct"] = round(min(1.0, v["games"] / maxg) * 100, 1)
    return out


# ---------------------------- pitcher recent (3) -----------------------------

def pitcher_recent(df):
    """DataFrame -> {pitcherId: {"starts": k, "pitches": n, "mix": [
    {"pitch", "usage", "xSlg", "hr", "n"}]}} over each pitcher's last
    RECENT_STARTS games. xSlg is mean xSLG allowed on BBE off that pitch --
    'is this pitch getting crushed lately'."""
    need = ["pitcher", "game_pk", "pitch_name"]
    if any(_col(df, c) is None for c in need):
        return {}
    has_date = _col(df, "game_date")
    has_ev = _col(df, "events")
    has_xslg = _col(df, "estimated_slg_using_speedangle")
    has_type = _col(df, "type")
    out = {}
    for pid, g in df.groupby("pitcher"):
        if has_date:
            order = g.groupby("game_pk")[has_date].max().sort_values()
            recent_games = list(order.index[-RECENT_STARTS:])
        else:
            recent_games = list(g["game_pk"].unique()[-RECENT_STARTS:])
        g = g[g["game_pk"].isin(recent_games)]
        total = len(g)
        if total < 30:      # reliever cameo / opener -- not a mix read
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
            mix.append(entry)
        mix.sort(key=lambda m: m["usage"], reverse=True)
        out[int(pid)] = {"starts": len(recent_games), "pitches": total,
                         "gamePks": [int(x) for x in recent_games], "mix": mix[:8]}
    return out


# ------------------------- drift + recent fit (3+4) --------------------------

def mix_drift(recent, season_mix):
    """Compare last-3-start usage vs season arsenal. Returns (drift_list,
    crushed_list). Drift: any pitch whose usage moved >= DRIFT_PP percentage
    points either way. Crushed: a CORE (>= 15%) recent pitch with recent
    xSLG-allowed >= CRUSHED_XSLG or >= 2 HR allowed in the window."""
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
    # a season pitch that vanished from the recent mix is also drift
    recent_keys = {m["pitch"].strip().lower() for m in recent.get("mix", [])}
    for key, season_u in season_by.items():
        if season_u is not None and season_u >= DRIFT_PP and key not in recent_keys:
            drifts.append({"pitch": key.title(), "recent": 0.0,
                           "season": round(season_u, 1),
                           "delta": round(-season_u, 1)})
    return drifts, crushed


def core_mix(mix, min_usage=CORE_USAGE):
    """DTP rule 4: the pitches a batter will actually see enough of."""
    return [m for m in (mix or []) if _f(m.get("usage")) and m["usage"] >= min_usage]


if __name__ == "__main__":
    # standalone smoke: fetch + print summary counts (needs network)
    df = fetch_statcast()
    b = batter_form(df)
    p = pitcher_recent(df)
    profiled = sum(1 for f in b.values() if f.get("profile"))
    print(f"statcast rows={len(df)}  batters={len(b)} (profiled {profiled})  pitchers={len(p)}",
          file=sys.stderr)
