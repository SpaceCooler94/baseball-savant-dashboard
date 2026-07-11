# ============================================================================
# mlb_model.py -- Python port of the log5 daily projection model.
#
# This is the model that currently lives in Daily_Matchups.js (v5.1). Porting it
# here lets it run on GitHub Actions instead of on the phone, producing a finished
# daily-board JSON that Render serves and MLB_Daily.js just displays.
#
# PORT DISCIPLINE: every function here mirrors its JavaScript counterpart exactly.
# The companion test (test_mlb_model.py) feeds identical inputs to both the JS and
# this Python and asserts the outputs match to the same rounding. Nothing here is
# trusted until that parity test passes -- a silent Python/JS divergence would make
# the served board wrong in ways no single-file review would catch.
#
# This file (Part 1) is the pure math only: log5, expected PA, game probability,
# league rates, batter/pitcher rate helpers, and the two projection functions.
# Data fetching (Savant via pybaseball, schedule via StatsAPI) comes in Part 2 once
# this core is proven identical to the JS.
# ============================================================================

import math


def clamp01(v):
    return max(0.001, min(0.999, v))


def clamp_range(v, lo, hi):
    return max(lo, min(hi, v))


def log5(a, b, l):
    """Matchup probability blend. Mirrors log5() in Daily_Matchups.js exactly.
    P = (A*B/L) / (A*B/L + (1-A)(1-B)/(1-L))"""
    a = clamp01(a)
    b = clamp01(b)
    l = clamp01(l)
    num = (a * b) / l
    den = num + ((1 - a) * (1 - b)) / (1 - l)
    return num / den


def expected_pa(order_avg):
    """PA-per-lineup-slot. Mirrors expectedPA() in Daily_Matchups.js.
    Grounded in real 2023 full-season data (leadoff ~4.6, 9-hole ~3.75,
    ~-0.11 PA per slot down). Unknown slot -> whole-lineup middle."""
    if order_avg is None or not _finite(order_avg):
        return 3.9
    slot = clamp_range(order_avg, 1, 9)
    return _js_round((4.6 - 0.11 * (slot - 1)) * 100) / 100


def game_prob(per_pa, n):
    """1 - (1-p)^n. Mirrors gameProb() in Daily_Matchups.js."""
    return 1 - math.pow(1 - clamp01(per_pa), n)


def _finite(v):
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _nv(v):
    """Mirror of the JS nv(): None/''/non-numeric -> None, else float.
    Critically NOT coercing None->0 (the bug we killed earlier)."""
    if v is None or v == "":
        return None
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


# --------------------------- league + rate helpers ---------------------------

def league_rates(players):
    """PA-weighted league baseline rates from today's full pool. Mirrors
    leagueRates() in Daily_Matchups.js. Hit rate uses OBP - BB% as a hits-per-PA
    approximation (omits the small HBP term, ~1-2% of PA, documented); HR rate is
    exact (hrRate field is already HR/PA)."""
    pa_sum = 0.0
    hit_sum = 0.0
    hr_sum = 0.0
    for pl in players:
        pa = _nv(pl.get("pa"))
        obp = _nv(pl.get("obp"))
        bb_pct = _nv(pl.get("bbPct"))
        hr_rate = _nv(pl.get("hrRate"))
        if pa is None or pa <= 0 or obp is None or bb_pct is None or hr_rate is None:
            continue
        pa_sum += pa
        hit_sum += max(0.01, obp - bb_pct / 100) * pa
        hr_sum += (hr_rate / 100) * pa
    return {
        "hitRatePerPA": hit_sum / pa_sum if pa_sum > 0 else 0.315,
        "hrRatePerPA": hr_sum / pa_sum if pa_sum > 0 else 0.030,
        "poolPA": pa_sum,
    }


def batter_hit_rate_per_pa(h, pitcher_hand):
    """Platoon-adjusted hits-per-PA. Mirrors batterHitRatePerPA() in the JS.
    Base = OBP-BB%; multiplied by split-AVG/season-AVG ratio (clamped +/-40%)."""
    obp = _nv(h.get("obp"))
    bb_pct = _nv(h.get("bbPct"))
    if obp is None or bb_pct is None:
        return None
    base = max(0.01, obp - bb_pct / 100)
    season_avg = _nv(h.get("avg"))
    if pitcher_hand == "L":
        split_avg = _nv(h.get("vsLAvg"))
    elif pitcher_hand == "R":
        split_avg = _nv(h.get("vsRAvg"))
    else:
        split_avg = None
    if split_avg is not None and season_avg is not None and season_avg > 0:
        mult = clamp_range(split_avg / season_avg, 0.6, 1.4)
        return base * mult
    return base


def batter_hr_rate_per_pa(h):
    """Season HR/PA. Mirrors batterHrRatePerPA(). No split HR data exists anywhere
    in the pipeline, so this is season-only by necessity, not omission."""
    r = _nv(h.get("hrRate"))
    return r / 100 if r is not None else None


# --------------------------- calibration (Platt) ----------------------------

def _logit(p):
    c = clamp01(p)
    return math.log(c / (1 - c))


def _sigmoid(z):
    return 1 / (1 + math.exp(-z))


def apply_calibration(raw_per_pa, cal_block):
    """Mirror of applyCalibration() in the JS. Logit-space scale+offset."""
    if not cal_block or cal_block.get("scale") is None:
        return raw_per_pa, False
    z = _logit(raw_per_pa)
    calibrated = _sigmoid(cal_block["scale"] * z + (cal_block.get("offset") or 0))
    return clamp01(calibrated), True


# ------------------------------- projections --------------------------------

def _js_round(v):
    """JavaScript Math.round: rounds half UP (toward +inf), unlike Python's built-in
    round() which uses banker's rounding (half to even). The parity test caught these
    diverging on exact .5 boundaries (e.g. 304.5 -> JS 305, Python 304), which would
    have made the served board silently disagree with the phone model. Since the JS is
    the shipping reference we're porting to match, we replicate its behavior."""
    return math.floor(v + 0.5)


def _round3(v):
    return _js_round(v * 1000) / 1000


def project_hit(h, p, ctx, league, calibration=None):
    """Mirror of projectHit(). Returns the same field shape the JS emits."""
    p_hand = p.get("hand") if p else None
    batter_rate = batter_hit_rate_per_pa(h, p_hand)
    pitcher_rate = p.get("hitRateAllowedPerPA") if p else None
    if batter_rate is not None and pitcher_rate is not None:
        data_quality = "full"
    elif batter_rate is not None or pitcher_rate is not None:
        data_quality = "partial"
    else:
        data_quality = "thin"
    a = batter_rate if batter_rate is not None else league["hitRatePerPA"]
    b = pitcher_rate if pitcher_rate is not None else league["hitRatePerPA"]
    raw_per_pa = log5(a, b, league["hitRatePerPA"])
    cal_hit = calibration.get("hit") if calibration else None
    per_pa, applied = apply_calibration(raw_per_pa, cal_hit)
    n = expected_pa(h.get("orderAvg"))
    per_game = game_prob(per_pa, n)
    tier = "A" if per_game >= .70 else "B" if per_game >= .60 else "C" if per_game >= .50 else "D"
    sig, risk = [], []
    season_rate = None
    if _nv(h.get("obp")) is not None and _nv(h.get("bbPct")) is not None:
        season_rate = max(0.01, _nv(h.get("obp")) - _nv(h.get("bbPct")) / 100)
    if p_hand and batter_rate is not None and season_rate is not None:
        if batter_rate > season_rate * 1.05:
            sig.append({"label": "Favorable platoon split", "cls": "green"})
        elif batter_rate < season_rate * 0.95:
            risk.append({"label": "Unfavorable platoon split", "cls": "orange"})
    if pitcher_rate is not None and pitcher_rate > league["hitRatePerPA"] * 1.08:
        sig.append({"label": "Hittable pitcher", "cls": "green"})
    elif pitcher_rate is not None and pitcher_rate < league["hitRatePerPA"] * 0.92:
        risk.append({"label": "Tough pitcher matchup", "cls": "orange"})
    oa = h.get("orderAvg")
    if oa is not None and oa <= 3:
        sig.append({"label": "Top-of-order PA volume", "cls": "cyan"})
    elif oa is not None and oa >= 8:
        risk.append({"label": "Bottom-of-order, fewer PA", "cls": "orange"})
    l10, l5 = _nv(h.get("l10Ops")), _nv(h.get("l5Ops"))
    if (l5 is not None and l5 >= .900) or (l10 is not None and l10 >= .860):
        sig.append({"label": "Recent form", "cls": "cyan"})
    elif (l5 is not None and l5 < .620) or (l10 is not None and l10 < .670):
        risk.append({"label": "Cold recent form", "cls": "orange"})
    if data_quality != "full":
        risk.append({"label": "Thin data, used league baseline" if data_quality == "thin"
                     else "Partial data, one side used league baseline", "cls": "orange"})
    return {
        "perPA": _round3(per_pa),
        "perGame": _round3(per_game),
        "expectedPA": n,
        "tier": tier,
        "dataQuality": data_quality,
        "calibrationApplied": applied,
        "signals": sig[:5],
        "risks": risk[:5],
        "inputs": {
            "batterRatePerPA": _round3(batter_rate) if batter_rate is not None else None,
            "pitcherRateAllowedPerPA": _round3(pitcher_rate) if pitcher_rate is not None else None,
            "leagueRatePerPA": _round3(league["hitRatePerPA"]),
            "rawPerPA": _round3(raw_per_pa),
        },
    }


def project_hr(h, p, ctx, league, calibration=None):
    """Mirror of projectHR(). Park factor multiplicative, conservative temp nudge,
    NO wind (needs stadium azimuth we don't have)."""
    batter_rate = batter_hr_rate_per_pa(h)
    pitcher_rate = p.get("hrRateAllowedPerPA") if p else None
    if batter_rate is not None and pitcher_rate is not None:
        data_quality = "full"
    elif batter_rate is not None or pitcher_rate is not None:
        data_quality = "partial"
    else:
        data_quality = "thin"
    a = batter_rate if batter_rate is not None else league["hrRatePerPA"]
    b = pitcher_rate if pitcher_rate is not None else league["hrRatePerPA"]
    raw_per_pa = log5(a, b, league["hrRatePerPA"])
    park = _nv(ctx.get("park")) or 100
    raw_per_pa = raw_per_pa * (park / 100)
    temp = _nv((ctx.get("weather") or {}).get("temp"))
    if temp is not None and temp >= 82:
        raw_per_pa *= 1.06
    elif temp is not None and temp <= 45:
        raw_per_pa *= 0.94
    raw_per_pa = clamp01(raw_per_pa)
    cal_hr = calibration.get("hr") if calibration else None
    per_pa, applied = apply_calibration(raw_per_pa, cal_hr)
    n = expected_pa(h.get("orderAvg"))
    per_game = game_prob(per_pa, n)
    tier = "A" if per_game >= .20 else "B" if per_game >= .13 else "C" if per_game >= .07 else "D"
    sig, risk = [], []
    if pitcher_rate is not None and pitcher_rate > league["hrRatePerPA"] * 1.25:
        sig.append({"label": "HR-prone pitcher", "cls": "green"})
    elif pitcher_rate is not None and pitcher_rate < league["hrRatePerPA"] * 0.75:
        risk.append({"label": "Stingy HR pitcher", "cls": "orange"})
    if park >= 110:
        sig.append({"label": "Hitter park", "cls": "cyan"})
    elif park <= 94:
        risk.append({"label": "Pitcher park", "cls": "orange"})
    if temp is not None and temp >= 82:
        sig.append({"label": "Warm carry weather", "cls": "cyan"})
    if data_quality != "full":
        risk.append({"label": "Thin data, used league baseline" if data_quality == "thin"
                     else "Partial data, one side used league baseline", "cls": "orange"})
    return {
        "perPA": _round3(per_pa),
        "perGame": _round3(per_game),
        "calibrationApplied": applied,
        "expectedPA": n,
        "tier": tier,
        "dataQuality": data_quality,
        "signals": sig[:5],
        "risks": risk[:5],
        "inputs": {
            "batterRatePerPA": _round3(batter_rate) if batter_rate is not None else None,
            "pitcherRateAllowedPerPA": _round3(pitcher_rate) if pitcher_rate is not None else None,
            "leagueRatePerPA": _round3(league["hrRatePerPA"]),
            "parkFactor": park,
            "tempF": temp,
            "rawPerPA": _round3(raw_per_pa),
        },
    }
