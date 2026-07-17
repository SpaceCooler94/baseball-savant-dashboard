# ============================================================================
# mlb_model.py -- log5 daily projection model. v5.4
#
# LINEAGE: v5.0-5.2 were a parity-locked port of Daily_Matchups.js. As of v5.3
# this Python file IS the reference implementation -- the JS parity freeze ends
# at v5.2 (test_mlb_model.py still proves the v5.2 subset matches the JS; the
# v5.3 additions are covered by their own unit tests, not JS parity).
#
# v5.4 -- CALIBRATION MOVES PER-GAME:
#   Calibration now applies to the per-GAME probability (after game_prob), not
#   the per-PA rate. The settled outcome is binary per game ("did he get a
#   hit"), so that is the only level a Platt fit can be estimated at honestly;
#   calibrating per-PA against per-game outcomes would require pretending a
#   game's PAs are independent Bernoulli trials, which they are not. Output
#   changes: perPA is now always the RAW rate, perGame is calibrated when a
#   valid calibration block is present, and rawPerGame is emitted alongside so
#   the settlement ledger can always record the uncalibrated prediction (fits
#   must NEVER see calibrated outputs -- that is a feedback loop). MODEL_VERSION
#   below is stamped on boards and ledger rows; calibrate.py filters on it so a
#   fit never mixes rows from different model math.
#
# v5.3 -- SHRINKAGE EVERYWHERE (the Wagaman fix):
#   v5.2 shrank platoon *splits* toward season average, but every other input
#   rate was taken at face value. Result on live boards: a 39-PA bench bat with
#   2 HR (5.1% HR/PA) ranked as the #2 HR play, ahead of a 32-HR slugger, and
#   0-HR small samples projected impossible ~0% HR rates. v5.3 applies the same
#   empirical-Bayes discipline to all four rates feeding log5:
#     - batter hit base (OBP-BB%)      -> shrunk toward league, prior 150 PA
#     - batter HR/PA                   -> shrunk toward league, prior 250 PA
#       (HR rate is the noisiest input; it stabilizes far slower than AVG)
#     - pitcher hit-rate allowed       -> shrunk toward league, prior 200 BF
#     - pitcher HR-rate allowed        -> shrunk toward league, prior 250 BF
#   Split shrinkage from v5.2 is unchanged and stacks on top (the split ratio
#   multiplies the now-shrunk base).
#   Unknown sample size falls back to a conservative 0.25 weight -- same rule
#   v5.2 used for unknown split PA.
#
# Everything else (log5, expectedPA, gameProb, calibration, tiers, signals,
# _js_round half-up rounding, output field shape) is byte-for-byte v5.2 logic.
# ============================================================================

import math

# Stamped on every board and ledger row; calibration fits filter on it so a
# Platt fit never mixes predictions produced by different model math. Bump
# whenever any change alters the raw probabilities (priors, log5 inputs,
# park/temp handling, expectedPA...). The fitting ledger effectively restarts
# at each bump; old rows remain for ROI history.
MODEL_VERSION = "log5-v5.4"

# Empirical-Bayes priors (in PA for batters, BF for pitchers). Larger prior =
# less trust in small samples. HR gets the biggest prior because HR/PA is the
# slowest rate to stabilize; pitcher priors sit between (rates-per-BF stabilize
# faster than batter HR, slower than batter contact).
HIT_BASE_PRIOR_PA = 150
HR_PRIOR_PA = 250
PITCHER_HIT_PRIOR_BF = 200
PITCHER_HR_PRIOR_BF = 250
UNKNOWN_SAMPLE_WEIGHT = 0.25


def clamp01(v):
    return max(0.001, min(0.999, v))


def clamp_range(v, lo, hi):
    return max(lo, min(hi, v))


def log5(a, b, l):
    """Matchup probability blend.
    P = (A*B/L) / (A*B/L + (1-A)(1-B)/(1-L))"""
    a = clamp01(a)
    b = clamp01(b)
    l = clamp01(l)
    num = (a * b) / l
    den = num + ((1 - a) * (1 - b)) / (1 - l)
    return num / den


def shrink(rate, n, prior_n, league_rate):
    """Empirical-Bayes shrinkage: weight = n / (n + prior_n).
    None rate -> None (caller falls back to league). None/invalid n -> the
    conservative UNKNOWN_SAMPLE_WEIGHT. None league -> rate unshrunk (lets the
    v5.2 parity tests keep passing by simply not passing a league rate)."""
    if rate is None:
        return None
    if league_rate is None:
        return rate
    if n is None or not _finite(n) or n <= 0:
        w = UNKNOWN_SAMPLE_WEIGHT
    else:
        w = n / (n + prior_n)
    return w * rate + (1 - w) * league_rate


def expected_pa(order_avg):
    """PA-per-lineup-slot. Grounded in real 2023 full-season data (leadoff ~4.6,
    9-hole ~3.75, ~-0.11 PA per slot down). Unknown slot -> whole-lineup middle."""
    if order_avg is None or not _finite(order_avg):
        return 3.9
    slot = clamp_range(order_avg, 1, 9)
    return _js_round((4.6 - 0.11 * (slot - 1)) * 100) / 100


def game_prob(per_pa, n):
    """1 - (1-p)^n."""
    return 1 - math.pow(1 - clamp01(per_pa), n)


def _finite(v):
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _nv(v):
    """None/''/non-numeric -> None, else float. Critically NOT coercing
    None->0 (the bug we killed earlier)."""
    if v is None or v == "":
        return None
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


# --------------------------- league + rate helpers ---------------------------

def league_rates(players):
    """PA-weighted league baseline rates from today's full pool. Hit rate uses
    OBP - BB% as a hits-per-PA approximation (omits the small HBP term, ~1-2%
    of PA, documented); HR rate is exact (hrRate field is already HR/PA).
    NOTE: baselines are built from RAW rates on purpose -- shrinkage targets
    this baseline, so the baseline itself must not be shrunk."""
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


def batter_hit_rate_per_pa(h, pitcher_hand, league_rate=None):
    """Platoon-adjusted hits-per-PA.
    v5.3: the BASE rate (OBP-BB%) is now shrunk toward the league rate by
    season PA (prior HIT_BASE_PRIOR_PA) before the platoon multiplier applies.
    A 200-PA rookie's .350 OBP no longer outranks a 650-PA star at face value.
    The v5.2 split shrinkage (prior 150 PA toward season avg) is unchanged and
    multiplies the shrunk base."""
    obp = _nv(h.get("obp"))
    bb_pct = _nv(h.get("bbPct"))
    if obp is None or bb_pct is None:
        return None
    raw_base = max(0.01, obp - bb_pct / 100)
    base = shrink(raw_base, _nv(h.get("pa")), HIT_BASE_PRIOR_PA, league_rate)
    season_avg = _nv(h.get("avg"))
    if pitcher_hand == "L":
        split_avg = _nv(h.get("vsLAvg"))
        split_pa = _nv(h.get("vsLPa"))
    elif pitcher_hand == "R":
        split_avg = _nv(h.get("vsRAvg"))
        split_pa = _nv(h.get("vsRPa"))
    else:
        split_avg = None
        split_pa = None
    if split_avg is not None and season_avg is not None and season_avg > 0:
        # v5.2 logic, unchanged: shrink split_avg toward season_avg.
        SPLIT_PRIOR_PA = 150
        w = (split_pa / (split_pa + SPLIT_PRIOR_PA)) if split_pa is not None else 0.25
        shrunk_split = w * split_avg + (1 - w) * season_avg
        mult = clamp_range(shrunk_split / season_avg, 0.6, 1.4)
        return base * mult
    return base


def batter_hr_rate_per_pa(h, league_rate=None):
    """Season HR/PA, shrunk toward league by season PA (prior HR_PRIOR_PA).
    v5.3: this is THE Wagaman fix -- 2 HR in 39 PA now projects ~3.5% HR/PA
    instead of 5.1%, and 0 HR in 62 PA projects ~2.6% instead of ~0%.
    No split HR data exists anywhere in the pipeline, so season-only remains
    a necessity, not an omission."""
    r = _nv(h.get("hrRate"))
    if r is None:
        return None
    return shrink(r / 100, _nv(h.get("pa")), HR_PRIOR_PA, league_rate)


# --------------------------- calibration (Platt) ----------------------------

def _logit(p):
    c = clamp01(p)
    return math.log(c / (1 - c))


def _sigmoid(z):
    return 1 / (1 + math.exp(-z))


def apply_calibration(raw_per_pa, cal_block):
    """Logit-space scale+offset."""
    if not cal_block or cal_block.get("scale") is None:
        return raw_per_pa, False
    z = _logit(raw_per_pa)
    calibrated = _sigmoid(cal_block["scale"] * z + (cal_block.get("offset") or 0))
    return clamp01(calibrated), True


# ------------------------------- projections --------------------------------

def _js_round(v):
    """JavaScript Math.round: rounds half UP (toward +inf), unlike Python's
    built-in round() (banker's rounding). Kept from the parity era -- boards
    must stay comparable across versions."""
    return math.floor(v + 0.5)


def _round3(v):
    return _js_round(v * 1000) / 1000


def _pitcher_rate(p, field, prior_bf, league_rate):
    """v5.3: pitcher rates are shrunk toward league by batters faced. A 45-BF
    spot starter's inflated/deflated rates no longer swing every hitter in
    that game to a board extreme."""
    if not p:
        return None
    return shrink(_nv(p.get(field)), _nv(p.get("battersFaced")), prior_bf, league_rate)


def project_hit(h, p, ctx, league, calibration=None):
    """Returns the same field shape v5.0-5.2 emitted."""
    p_hand = p.get("hand") if p else None
    batter_rate = batter_hit_rate_per_pa(h, p_hand, league["hitRatePerPA"])
    pitcher_rate = _pitcher_rate(p, "hitRateAllowedPerPA",
                                 PITCHER_HIT_PRIOR_BF, league["hitRatePerPA"])
    if batter_rate is not None and pitcher_rate is not None:
        data_quality = "full"
    elif batter_rate is not None or pitcher_rate is not None:
        data_quality = "partial"
    else:
        data_quality = "thin"
    a = batter_rate if batter_rate is not None else league["hitRatePerPA"]
    b = pitcher_rate if pitcher_rate is not None else league["hitRatePerPA"]
    raw_per_pa = log5(a, b, league["hitRatePerPA"])
    n = expected_pa(h.get("orderAvg"))
    raw_per_game = game_prob(raw_per_pa, n)
    # v5.4: Platt calibration applies to the per-GAME probability -- the level
    # at which outcomes are actually observed and fits are estimated.
    cal_hit = calibration.get("hit") if calibration else None
    per_game, applied = apply_calibration(raw_per_game, cal_hit)
    per_pa = raw_per_pa  # perPA is always raw as of v5.4
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
    # v5.3: small-sample flag -- shrinkage already discounts the rate, but the
    # reader should still see WHY a hot small sample isn't ranked higher.
    pa = _nv(h.get("pa"))
    if pa is not None and pa < HIT_BASE_PRIOR_PA:
        risk.append({"label": "Small sample (%d PA), rate regressed" % int(pa), "cls": "orange"})
    if data_quality != "full":
        risk.append({"label": "Thin data, used league baseline" if data_quality == "thin"
                     else "Partial data, one side used league baseline", "cls": "orange"})
    return {
        "perPA": _round3(per_pa),
        "perGame": _round3(per_game),
        "rawPerGame": _round3(raw_per_game),
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
    """Park factor multiplicative, conservative temp nudge, NO wind (needs
    stadium azimuth we don't have)."""
    batter_rate = batter_hr_rate_per_pa(h, league["hrRatePerPA"])
    pitcher_rate = _pitcher_rate(p, "hrRateAllowedPerPA",
                                 PITCHER_HR_PRIOR_BF, league["hrRatePerPA"])
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
    n = expected_pa(h.get("orderAvg"))
    raw_per_game = game_prob(raw_per_pa, n)
    cal_hr = calibration.get("hr") if calibration else None
    per_game, applied = apply_calibration(raw_per_game, cal_hr)
    per_pa = raw_per_pa  # perPA is always raw as of v5.4
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
    pa = _nv(h.get("pa"))
    if pa is not None and pa < HR_PRIOR_PA:
        risk.append({"label": "Small sample (%d PA), HR rate regressed" % int(pa), "cls": "orange"})
    if data_quality != "full":
        risk.append({"label": "Thin data, used league baseline" if data_quality == "thin"
                     else "Partial data, one side used league baseline", "cls": "orange"})
    return {
        "perPA": _round3(per_pa),
        "perGame": _round3(per_game),
        "rawPerGame": _round3(raw_per_game),
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
