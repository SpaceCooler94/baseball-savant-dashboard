#!/usr/bin/env python3
# ============================================================================
# measure_signals.py -- the analysis this whole angle/ledger architecture was
# built around, and never actually run until now. Every reference signal
# shipped this project (DTP profiles, zone overlap, heart-zone xSLG, SP HR/9,
# bullpen fatigue, wind, rest, arsenal overlap, book edges) has been stamped
# onto ledger rows as an angle key for weeks specifically so this question
# could eventually be answered: does knowing the angle fired tell you
# anything ABOUT THE OUTCOME that the model's own probability didn't already
# tell you -- or is it decoration.
#
# METHOD. For each angle key, fit:
#     P(outcome) = sigmoid(b0 + b1*logit(raw_prob) + b2*[angle fired])
# via Newton-Raphson IRLS -- the exact numerical method calibrate.py already
# uses for its own 2-parameter Platt fit (scale on logit(raw_prob), offset),
# generalized here to an arbitrary design matrix. calibrate.py's fit is the
# special case of this with no b2 term at all. b2 is the number that
# actually answers the question: after controlling for what the model
# already knows via raw_prob, does this angle ALSO shift the odds of the
# outcome. A b2 whose z-score clears +-1.96 (95%) is real residual lift (or
# real residual drag, if negative) -- coefficient indistinguishable from zero
# means the angle adds nothing the probability didn't already capture.
#
# This is not a new statistical method invented for this script -- it's the
# same hand-rolled IRLS logistic fit already shipped and trusted in
# calibrate.py, extended by one column. No numpy, no sklearn, same
# discipline that codebase already established for exactly this reason.
#
# Usage: python measure_signals.py [ledger/ledger.jsonl]
# Writes signal_validation.json. Never touches the model or calibration --
# this is read-only measurement, the same posture settle.py/calibrate.py
# already have toward the live board.
# ============================================================================

import datetime
import json
import math
import os
import sys
from zoneinfo import ZoneInfo

import mlb_model as M

ET = ZoneInfo("America/New_York")
LEDGER_PATH = "ledger/ledger.jsonl"
OUT_PATH = "signal_validation.json"
MIN_FLAGGED = 30   # need at least this many rows WITH the angle to attempt a
                   # fit at all. Below this, "no signal detected" would be
                   # indistinguishable from "not enough data to tell" -- the
                   # verdict says the latter explicitly, never the former.


# ------------------------------ linear algebra -------------------------------
# Hand-rolled, no numpy -- same "no heavy stats dependency" discipline
# calibrate.py already established for its own fit. Fine at this scale: the
# design matrix here never exceeds 3 columns (intercept, raw-prob logit, one
# angle flag), so these are always tiny systems.

def _solve(A, b):
    """Gaussian elimination with partial pivoting, A x = b."""
    n = len(b)
    M_ = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M_[r][col]))
        if abs(M_[piv][col]) < 1e-12:
            raise ValueError("singular matrix")
        M_[col], M_[piv] = M_[piv], M_[col]
        for r in range(col + 1, n):
            factor = M_[r][col] / M_[col][col]
            for c in range(col, n + 1):
                M_[r][c] -= factor * M_[col][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = M_[i][n] - sum(M_[i][j] * x[j] for j in range(i + 1, n))
        x[i] = s / M_[i][i]
    return x


def _matinv(A):
    """Gauss-Jordan matrix inverse. Only used to get the diagonal (variances)
    of the inverse Fisher information for standard errors -- small enough
    matrices here that computing the full inverse is simpler than a smarter
    partial approach, and just as fast at this size."""
    n = len(A)
    M_ = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M_[r][col]))
        if abs(M_[piv][col]) < 1e-12:
            raise ValueError("singular matrix")
        M_[col], M_[piv] = M_[piv], M_[col]
        p = M_[col][col]
        M_[col] = [v / p for v in M_[col]]
        for r in range(n):
            if r == col:
                continue
            factor = M_[r][col]
            M_[r] = [M_[r][c] - factor * M_[col][c] for c in range(2 * n)]
    return [row[n:] for row in M_]


def fit_logistic(X, y, max_iter=50, tol=1e-8):
    """Newton-Raphson IRLS. X: list of rows (each a list of covariate values,
    including the intercept column). y: list of 0/1 outcomes. Returns
    (beta, se) -- coefficients and their standard errors. Raises on
    non-convergence or a singular design (e.g. an angle collinear with
    another, or degenerate data)."""
    n, k = len(y), len(X[0])
    beta = [0.0] * k
    for _ in range(max_iter):
        eta = [sum(X[i][j] * beta[j] for j in range(k)) for i in range(n)]
        p = [M._sigmoid(e) for e in eta]
        w = [max(pi * (1 - pi), 1e-6) for pi in p]
        z = [eta[i] + (y[i] - p[i]) / w[i] for i in range(n)]
        XtWX = [[sum(X[i][a] * w[i] * X[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
        XtWz = [sum(X[i][a] * w[i] * z[i] for i in range(n)) for a in range(k)]
        beta_new = _solve(XtWX, XtWz)
        delta = max(abs(beta_new[j] - beta[j]) for j in range(k))
        beta = beta_new
        if delta < tol:
            break
    else:
        raise RuntimeError("IRLS did not converge")
    eta = [sum(X[i][j] * beta[j] for j in range(k)) for i in range(n)]
    p = [M._sigmoid(e) for e in eta]
    w = [max(pi * (1 - pi), 1e-6) for pi in p]
    XtWX = [[sum(X[i][a] * w[i] * X[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
    inv = _matinv(XtWX)
    se = [math.sqrt(max(inv[j][j], 0.0)) for j in range(k)]
    return beta, se


# --------------------------------- ledger IO ----------------------------------

def load_ledger(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def build_samples(rows, stat):
    """rows -> [(raw_prob, outcome, angle_set), ...] for one stat ('hit'/'hr'),
    filtered to the CURRENT model version only -- same convention
    calibrate.py already uses; an angle's meaning can shift across model
    versions the same way raw probabilities do."""
    raw_key = "hitRaw" if stat == "hit" else "hrRaw"
    out_key = "gotHit" if stat == "hit" else "gotHR"
    samples = []
    for r in rows:
        if r.get("modelVersion") != M.MODEL_VERSION:
            continue
        raw = r.get(raw_key)
        outcome = r.get(out_key)
        if raw is None or outcome is None:
            continue
        try:
            raw = float(raw)
        except (TypeError, ValueError):
            continue
        if not (0.0 < raw < 1.0):
            continue
        angles = set(r.get("angles") or [])
        samples.append((raw, int(outcome), angles))
    return samples


# ------------------------------- per-angle test -------------------------------

def test_angle(samples, angle_key):
    """The actual measurement. See module docstring for the model."""
    flagged_n = sum(1 for _, _, a in samples if angle_key in a)
    if flagged_n < MIN_FLAGGED:
        return {"verdict": f"insufficient data ({flagged_n} < {MIN_FLAGGED} flagged rows)",
                "n": len(samples), "flaggedN": flagged_n}

    X, y = [], []
    for raw, outcome, angles in samples:
        X.append([1.0, M._logit(raw), 1.0 if angle_key in angles else 0.0])
        y.append(outcome)

    try:
        beta, se = fit_logistic(X, y)
    except Exception as e:
        return {"verdict": f"fit failed ({type(e).__name__}: {e})",
                "n": len(X), "flaggedN": flagged_n}

    b2, se2 = beta[2], se[2]
    zscore = b2 / se2 if se2 > 0 else 0.0
    if zscore >= 1.96 and b2 > 0:
        verdict = "residual lift (positive, p<0.05) -- real signal beyond raw_prob"
    elif zscore <= -1.96:
        verdict = "residual DRAG (negative, p<0.05) -- flagged rows underperform raw_prob"
    else:
        verdict = "no detectable residual signal -- decoration, not signal, so far"

    flagged_rate = sum(o for r_, o, a in samples if angle_key in a) / flagged_n
    unflagged = [o for r_, o, a in samples if angle_key not in a]
    unflagged_rate = (sum(unflagged) / len(unflagged)) if unflagged else None

    return {
        "verdict": verdict, "n": len(X), "flaggedN": flagged_n,
        "coefRaw": round(beta[1], 4), "coefAngle": round(b2, 4),
        "seAngle": round(se2, 4), "z": round(zscore, 3),
        "observedRateFlagged": round(flagged_rate, 4),
        "observedRateUnflagged": round(unflagged_rate, 4) if unflagged_rate is not None else None,
    }


def all_angle_keys(samples):
    keys = set()
    for _, _, angles in samples:
        keys |= angles
    return sorted(keys)


# ---------------------------------- main ---------------------------------------

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else LEDGER_PATH
    rows = load_ledger(path)
    if not rows:
        print(f"No ledger data at {path} -- nothing to measure")
        return

    out = {"fitDate": datetime.datetime.now(ET).strftime("%Y-%m-%d"),
           "modelVersion": M.MODEL_VERSION, "ledgerRows": len(rows), "stats": {}}

    for stat in ("hit", "hr"):
        samples = build_samples(rows, stat)
        keys = all_angle_keys(samples)
        print(f"\n=== {stat.upper()} -- n={len(samples)} rows at {M.MODEL_VERSION}, "
              f"{len(keys)} distinct angles seen ===")
        stat_out = {}
        for key in keys:
            result = test_angle(samples, key)
            stat_out[key] = result
            tag = "  <-- REAL LIFT" if "positive" in result["verdict"] else \
                  "  <-- DRAG" if "DRAG" in result["verdict"] else ""
            print(f"  {key:<22} flagged={result.get('flaggedN','?'):<5} {result['verdict']}{tag}")
        out["stats"][stat] = stat_out

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
