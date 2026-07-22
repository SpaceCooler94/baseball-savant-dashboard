#!/usr/bin/env python3
# ============================================================================
# calibrate.py -- weekly Platt calibration fit for the daily board.
#
# Reads ledger/ledger.jsonl, fits P(outcome) = sigmoid(scale * logit(rawProb)
# + offset) per market (hit, HR) with plain Newton-Raphson IRLS -- two
# parameters, deterministic, no sklearn/scipy -- and writes calibration.json
# only when the fit clears every gate. build_daily_board.py picks the file up
# on its next run; mlb_model.apply_calibration consumes {scale, offset}.
#
# SIM MARKETS (added): the same ledger rows may also carry simHitRaw/simHrRaw
# from the hrhit Monte Carlo board. Those are fit SEPARATELY into simHit/simHr
# blocks, which hrhit_model.py consumes on its next build. Two reasons they are
# never pooled with the log5 fit:
#   - Different generating process. A Platt curve fit on a mixture of two
#     models' probabilities is wrong for both -- it splits the difference on a
#     miscalibration that only one of them has.
#   - Different version cadence. Sim rows are gated on simModelVersion, not
#     MODEL_VERSION, so a log5 bump doesn't throw away sim history (and a sim
#     bump doesn't throw away log5 history).
# Each sim market clears the same MIN_ROWS gate independently, so expect
# identity stubs for weeks after the log5 markets have published.
#
# GATES (a fit that fails any gate publishes IDENTITY for that market):
#   1. n >= MIN_ROWS settled rows for the current MODEL_VERSION only
#      (rows from older model math are a different distribution -- excluded).
#   2. Time-ordered 70/30 split (never shuffled -- shuffling leaks the future
#      into training). Fit on the first 70% of dates, validate on the last 30%.
#   3. Published only if validation log loss improves on identity (raw) by at
#      least MIN_IMPROVEMENT. Ship nothing rather than noise.
#   4. Fitted scale must stay in a sane band (0.2..3.0) -- a wild scale means
#      the ledger is contaminated or too thin, not that the model is that wrong.
# After passing, parameters are refit on ALL rows (standard practice: the split
# exists to validate the procedure, the final fit uses every observation).
#
# calibration.json also carries a decile reliability table per market
# (predicted vs observed by bucket) -- the audit trail for whether the A/B/C/D
# tier thresholds match reality.
#
# Usage: python calibrate.py    Exit 0 always unless the ledger is unreadable;
# "no publish" is a normal outcome, not an error.
# ============================================================================

import datetime
import json
import math
import os
import sys
from zoneinfo import ZoneInfo

from mlb_model import MODEL_VERSION

ET = ZoneInfo("America/New_York")
LEDGER_PATH = os.path.join("ledger", "ledger.jsonl")
OUT_PATH = "calibration.json"

MIN_ROWS = 2000
MIN_IMPROVEMENT = 0.0005     # absolute validation log-loss improvement required
SCALE_BAND = (0.2, 3.0)
EPS = 1e-6

# Sim model version to fit. Bump when hrhit's math changes, exactly like
# MODEL_VERSION -- older sim rows then stop counting toward MIN_ROWS.
SIM_MODEL_VERSION = "hrhit-1.0"


def _logit(p):
    p = min(1 - EPS, max(EPS, p))
    return math.log(p / (1 - p))


def _sigmoid(z):
    if z >= 0:
        e = math.exp(-z)
        return 1 / (1 + e)
    e = math.exp(z)
    return e / (1 + e)


def log_loss(probs, ys):
    s = 0.0
    for p, y in zip(probs, ys):
        p = min(1 - EPS, max(EPS, p))
        s += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return s / len(probs)


def fit_platt(raw_probs, ys, iters=50, ridge=1e-6):
    """Newton-Raphson (IRLS) logistic regression on the single feature
    z = logit(rawProb): minimizes log loss of sigmoid(a*z + b).
    Deterministic; the tiny ridge keeps the 2x2 Hessian invertible on
    degenerate inputs. Returns (scale a, offset b)."""
    zs = [_logit(p) for p in raw_probs]
    a, b = 1.0, 0.0  # start at identity
    n = len(zs)
    for _ in range(iters):
        g_a = g_b = 0.0
        h_aa = h_ab = h_bb = 0.0
        for z, y in zip(zs, ys):
            mu = _sigmoid(a * z + b)
            d = mu - y
            w = mu * (1 - mu)
            g_a += d * z
            g_b += d
            h_aa += w * z * z
            h_ab += w * z
            h_bb += w
        g_a /= n; g_b /= n
        h_aa = h_aa / n + ridge; h_ab /= n; h_bb = h_bb / n + ridge
        det = h_aa * h_bb - h_ab * h_ab
        if abs(det) < 1e-12:
            break
        da = (h_bb * g_a - h_ab * g_b) / det
        db = (h_aa * g_b - h_ab * g_a) / det
        a -= da
        b -= db
        if abs(da) < 1e-10 and abs(db) < 1e-10:
            break
    return a, b


def reliability_table(raw_probs, ys, buckets=10):
    """Decile table: rows sorted by predicted prob, split into equal-count
    buckets; each reports n, mean predicted, observed rate."""
    order = sorted(range(len(raw_probs)), key=lambda i: raw_probs[i])
    out = []
    n = len(order)
    for k in range(buckets):
        lo = k * n // buckets
        hi = (k + 1) * n // buckets
        idx = order[lo:hi]
        if not idx:
            continue
        out.append({
            "n": len(idx),
            "meanPredicted": round(sum(raw_probs[i] for i in idx) / len(idx), 4),
            "observedRate": round(sum(ys[i] for i in idx) / len(idx), 4),
        })
    return out


def load_rows(version_field="modelVersion", version=None, required=("hitRaw", "hrRaw")):
    """Ledger rows for one model version, deduped on (date, hitterId) -- a
    partially-settled day that was retried can appear twice; last wins.

    version_field lets the sim markets filter on simModelVersion instead, so
    the two models' histories are gated independently even though they share
    rows. required drops rows missing that market's predictions (e.g. log5
    rows from dates where no sim board existed)."""
    if version is None:
        version = MODEL_VERSION
    if not os.path.exists(LEDGER_PATH):
        return []
    dedup = {}
    with open(LEDGER_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get(version_field) != version:
                continue
            if any(r.get(k) is None for k in required):
                continue
            dedup[(r.get("date"), r.get("hitterId"))] = r
    rows = list(dedup.values())
    rows.sort(key=lambda r: (r.get("date") or "", r.get("hitterId") or 0))
    return rows


def fit_market(rows, raw_key, y_key):
    """Run the full gated procedure for one market. Returns (block_or_None,
    report dict). block is the {scale, offset, ...} dict for calibration.json;
    None means identity (gates not cleared)."""
    raws = [r[raw_key] for r in rows]
    ys = [r[y_key] for r in rows]
    report = {"n": len(rows)}

    if len(rows) < MIN_ROWS:
        report["verdict"] = f"identity: only {len(rows)} rows (< {MIN_ROWS})"
        return None, report

    # Time-ordered split: rows are date-sorted already.
    cut = int(len(rows) * 0.7)
    a, b = fit_platt(raws[:cut], ys[:cut])
    val_raw = raws[cut:]
    val_y = ys[cut:]
    ll_identity = log_loss(val_raw, val_y)
    ll_cal = log_loss([_sigmoid(a * _logit(p) + b) for p in val_raw], val_y)
    report["valLogLossRaw"] = round(ll_identity, 5)
    report["valLogLossCal"] = round(ll_cal, 5)
    report["trainScale"] = round(a, 4)
    report["trainOffset"] = round(b, 4)

    if not (SCALE_BAND[0] <= a <= SCALE_BAND[1]):
        report["verdict"] = f"identity: scale {a:.3f} outside sane band {SCALE_BAND}"
        return None, report
    if ll_identity - ll_cal < MIN_IMPROVEMENT:
        report["verdict"] = ("identity: improvement %.5f < %.5f"
                             % (ll_identity - ll_cal, MIN_IMPROVEMENT))
        return None, report

    # Gates cleared: refit on everything for the published parameters.
    a_full, b_full = fit_platt(raws, ys)
    if not (SCALE_BAND[0] <= a_full <= SCALE_BAND[1]):
        report["verdict"] = f"identity: full-refit scale {a_full:.3f} left the sane band"
        return None, report
    report["verdict"] = "published"
    block = {
        "scale": round(a_full, 4),
        "offset": round(b_full, 4),
        "n": len(rows),
        "valLogLossRaw": report["valLogLossRaw"],
        "valLogLossCal": report["valLogLossCal"],
    }
    return block, report


def main():
    rows = load_rows()
    sim_rows = load_rows(version_field="simModelVersion", version=SIM_MODEL_VERSION,
                         required=("simHitRaw", "simHrRaw"))
    print(f"Ledger: {len(rows)} rows for {MODEL_VERSION}, "
          f"{len(sim_rows)} rows for {SIM_MODEL_VERSION}")

    hit_block, hit_report = fit_market(rows, "hitRaw", "gotHit")
    hr_block, hr_report = fit_market(rows, "hrRaw", "gotHR")
    shit_block, shit_report = fit_market(sim_rows, "simHitRaw", "gotHit")
    shr_block, shr_report = fit_market(sim_rows, "simHrRaw", "gotHR")
    for label, rep in (("hit", hit_report), ("hr ", hr_report),
                       ("sim hit", shit_report), ("sim hr ", shr_report)):
        print(f"{label}:", rep.get("verdict"), "|",
              {k: v for k, v in rep.items() if k != "verdict"})

    out = {
        "modelVersion": MODEL_VERSION,
        "simModelVersion": SIM_MODEL_VERSION,
        "fitDate": datetime.datetime.now(ET).strftime("%Y-%m-%d"),
        "reports": {"hit": hit_report, "hr": hr_report,
                    "simHit": shit_report, "simHr": shr_report},
    }
    if hit_block:
        out["hit"] = hit_block
        out["reliabilityHit"] = reliability_table([r["hitRaw"] for r in rows],
                                                  [r["gotHit"] for r in rows])
    if hr_block:
        out["hr"] = hr_block
        out["reliabilityHr"] = reliability_table([r["hrRaw"] for r in rows],
                                                 [r["gotHR"] for r in rows])
    if shit_block:
        out["simHit"] = shit_block
        out["reliabilitySimHit"] = reliability_table([r["simHitRaw"] for r in sim_rows],
                                                     [r["gotHit"] for r in sim_rows])
    if shr_block:
        out["simHr"] = shr_block
        out["reliabilitySimHr"] = reliability_table([r["simHrRaw"] for r in sim_rows],
                                                    [r["gotHR"] for r in sim_rows])

    # Always write the file: even an all-identity file documents WHY (reports),
    # and build's loader treats missing scale as identity per market.
    tmp = OUT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=1)
    os.replace(tmp, OUT_PATH)
    print(f"Wrote {OUT_PATH}: hit={'published' if hit_block else 'identity'}, "
          f"hr={'published' if hr_block else 'identity'}, "
          f"simHit={'published' if shit_block else 'identity'}, "
          f"simHr={'published' if shr_block else 'identity'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"CALIBRATE FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
