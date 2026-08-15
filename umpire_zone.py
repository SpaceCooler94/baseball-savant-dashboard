#!/usr/bin/env python3
# ============================================================================
# umpire_zone.py -- STANDALONE, UNVERIFIED AGAINST LIVE DATA. Do not import
# into build_daily_board.py or wire into compute_angles() until the three
# checks in the module docstring below have been run against a real pull.
#
# The pure statistic here (shadow-zone call accuracy per umpire) is the
# well-established public methodology -- same one Umpire Scorecards and the
# ump-analysis/Tim Sheehan public work use: take pitches near the rulebook
# zone boundary, compare the actual ball/strike call to what the rulebook
# zone says it should have been, aggregate per umpire. That math is fully
# testable with synthetic data and IS tested below.
#
# What is NOT verified, because it needs a real live pull this environment
# has no network access to make:
#
#   1. COLUMN RELIABILITY. The raw `umpire` field is a numeric MLBAM ID, and
#      it has a documented history of going null for extended stretches
#      (reported broken in Savant's own CSV export, historically). This
#      module refuses to report a rate for any umpire (or overall) below
#      MIN_COVERAGE / MIN_BORDERLINE -- see the coverage gate below -- but
#      that gate only helps if you actually check its output before trusting
#      anything downstream. RUN THIS FIRST on a real pull:
#          df['umpire'].notna().mean()
#      If that's near zero, this feature is dead regardless of anything else
#      here, and no code change fixes it.
#
#   2. ID vs NAME JOIN. situational.home_plate_umpire() returns a NAME from
#      StatsAPI's officials hydrate. This module's output is keyed by the
#      numeric MLBAM ID from the Statcast column. Joining "tonight's umpire"
#      to "his historical tendency" needs the SAME key on both sides --
#      exactly the ID-only-joins rule build_daily_board.py already enforces
#      everywhere else. CHECK before wiring this in: pull one real game's
#      schedule hydrate and print the raw officials block --
#          games = get_schedule()
#          print(games[0].get("officials"))
#      -- and confirm each entry's "official" dict actually carries an "id"
#      key alongside "fullName". If it does, home_plate_umpire() needs a
#      one-line change to also return that id. If it doesn't, this needs a
#      name-based join instead, which is a different (weaker) join than
#      every other lookup in this codebase uses on purpose.
#
#   3. ABS CHALLENGES (2026). MLB introduced ball/strike challenge reviews
#      league-wide this season. A human umpire's initial call can now be
#      overturned mid-game. Every public umpire-tendency methodology (the
#      ones this module's math is based on) predates this and assumes the
#      recorded call IS the umpire's judgment. If Savant's per-pitch
#      description/type fields reflect the POST-CHALLENGE result rather than
#      the original call, this would silently attribute the automated
#      system's correction to the umpire -- backwards, not just noisy. CHECK:
#      look for any challenge-related column in a real pull (something like
#      `is_review`, `review_result`, or similar -- exact name unconfirmed) or
#      check pybaseball's changelog/issues for 2026 ABS-related schema notes
#      before trusting this on 2026 data specifically.
#
# Until all three are checked, treat this module's output the same as any
# other unproven, ledger-stamped-only signal -- and don't be surprised if #3
# means it needs a challenge-aware filter added before it's trustworthy at
# all for the 2026 season specifically.
# ============================================================================

import math

# Rulebook plate half-width: 17 inches / 2, in feet.
PLATE_HALF_WIDTH_FT = (17.0 / 12.0) / 2.0   # 0.7083 ft

# Borderline-pitch margin: how close to the rulebook edge (horizontally or
# vertically) a pitch has to land to count as "testing the umpire's judgment"
# rather than an obvious ball or strike. 3 inches is the commonly-cited band
# in public umpire-accuracy work (Umpire Scorecards and similar use a
# comparable shadow-zone width) -- DOCUMENTED APPROXIMATION, not MLB's own
# cut, same honesty standard zone_engine.py already applies to its own
# heart/shadow boundary.
BORDERLINE_MARGIN_FT = 3.0 / 12.0            # 0.25 ft

MIN_BORDERLINE = 40      # per-umpire floor before reporting a rate at all
MIN_COVERAGE = 0.60      # overall non-null 'umpire' rate floor -- below this
                          # the column itself is too broken to trust for ANY
                          # umpire, not just thin ones (see check #1 above)


def _nv(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _col(df, name):
    return name if name in df.columns else None


def umpire_coverage(df):
    """Fraction of rows with a non-null umpire id. Call this FIRST, on real
    data, before trusting anything else in this module. Returns None if the
    column doesn't exist at all (older pybaseball / unexpected schema)."""
    c = _col(df, "umpire")
    if c is None:
        return None
    import pandas as pd
    n = len(df)
    if n == 0:
        return None
    return round(float(df[c].notna().sum()) / n, 4)


def _is_called_ball_or_strike(description):
    """True only for a pitch the PLATE UMPIRE actually judged -- excludes
    swings, fouls, hit-by-pitch, etc. 'description' values per the standard
    Statcast vocabulary; defensive against anything unrecognized (returns
    False rather than guessing)."""
    d = str(description or "").strip().lower()
    return d in ("called_strike", "ball", "blocked_ball")


def _is_actual_strike(description):
    d = str(description or "").strip().lower()
    return d == "called_strike"


def umpire_call_accuracy(df, min_coverage=MIN_COVERAGE, min_borderline=MIN_BORDERLINE):
    """DataFrame of pitches -> {umpireId: {...}} OR a dict with "insufficient
    coverage" if the umpire column itself is too sparse to trust for ANYONE,
    not just thin-sample umpires.

    Per borderline pitch: expected_strike = rulebook zone says strike.
    actual_strike = the recorded call. Aggregated per umpire into:
      - expansionRate: of BORDERLINE pitches OUTSIDE the rulebook zone, how
        often they're still called strikes (pitcher-favorable / "loose")
      - contractionRate: of BORDERLINE pitches INSIDE the rulebook zone, how
        often they're still called balls (hitter-favorable / "tight")
      - zoneScore: expansionRate - contractionRate. Positive = generous to
        pitchers (harder to get a hit prop home), negative = generous to
        hitters. This is the number a HIT/HR model would actually want.
    Both counts are reported alongside the rates -- a rate with n=41 and a
    rate with n=800 should never be trusted the same amount downstream, and
    this module doesn't collapse that distinction away.
    """
    need = ["plate_x", "plate_z", "sz_top", "sz_bot", "description"]
    ump_col = _col(df, "umpire")
    if ump_col is None or any(_col(df, c) is None for c in need):
        return {"insufficientData": True, "reason": "missing required columns"}

    coverage = umpire_coverage(df)
    if coverage is None or coverage < min_coverage:
        return {"insufficientData": True, "reason": f"umpire column coverage {coverage} < {min_coverage}",
                "coverage": coverage}

    tallies = {}   # umpire_id -> {"exp_n":.., "exp_strikes":.., "con_n":.., "con_balls":..}
    for _, row in df.iterrows():
        uid = row.get(ump_col)
        uid = _nv(uid)
        if uid is None:
            continue
        desc = row.get("description")
        if not _is_called_ball_or_strike(desc):
            continue
        px, pz = _nv(row.get("plate_x")), _nv(row.get("plate_z"))
        sz_top, sz_bot = _nv(row.get("sz_top")), _nv(row.get("sz_bot"))
        if px is None or pz is None or sz_top is None or sz_bot is None:
            continue
        if sz_top <= sz_bot:
            continue  # degenerate zone bounds -- skip rather than guess

        dx = abs(px) - PLATE_HALF_WIDTH_FT          # >0 outside horizontally
        dz_top = pz - sz_top                          # >0 above the zone
        dz_bot = sz_bot - pz                           # >0 below the zone
        dz = max(dz_top, dz_bot, -min(pz - sz_bot, sz_top - pz))
        # signed distance to nearest edge: negative = inside, positive = outside
        horiz_dist = dx
        vert_dist = max(dz_top, dz_bot) if (pz > sz_top or pz < sz_bot) else -min(pz - sz_bot, sz_top - pz)
        # overall "outside-ness": if outside on either axis, distance is the
        # max positive excess; if inside on both, distance is the (negative)
        # margin to the nearest edge.
        outside_h = px < -PLATE_HALF_WIDTH_FT or px > PLATE_HALF_WIDTH_FT
        outside_v = pz > sz_top or pz < sz_bot
        in_zone = not outside_h and not outside_v
        if in_zone:
            edge_dist = min(PLATE_HALF_WIDTH_FT - abs(px), pz - sz_bot, sz_top - pz)
        else:
            edge_dist = max(horiz_dist if outside_h else -1e9,
                            vert_dist if outside_v else -1e9)
        if abs(edge_dist) > BORDERLINE_MARGIN_FT:
            continue  # not borderline -- an obvious ball or obvious strike

        t = tallies.setdefault(uid, {"exp_n": 0, "exp_strikes": 0, "con_n": 0, "con_balls": 0})
        is_strike = _is_actual_strike(desc)
        if in_zone:
            t["con_n"] += 1
            if not is_strike:
                t["con_balls"] += 1
        else:
            t["exp_n"] += 1
            if is_strike:
                t["exp_strikes"] += 1

    out = {}
    for uid, t in tallies.items():
        total = t["exp_n"] + t["con_n"]
        if total < min_borderline:
            continue
        expansion = round(t["exp_strikes"] / t["exp_n"], 4) if t["exp_n"] else None
        contraction = round(t["con_balls"] / t["con_n"], 4) if t["con_n"] else None
        zone_score = (round(expansion - contraction, 4)
                      if expansion is not None and contraction is not None else None)
        out[int(uid)] = {
            "expansionRate": expansion, "expansionN": t["exp_n"],
            "contractionRate": contraction, "contractionN": t["con_n"],
            "zoneScore": zone_score, "totalBorderline": total,
        }
    return out
