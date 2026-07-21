#!/usr/bin/env python3
"""
hrhit_model.py — HR + Hit prop model, season-to-date, Monte Carlo.

HR MODEL (10,000 sims/player)
  Ranks by genuine power, not contact volume:
  per-BBE HR prob = blend( shrunk HR/BBE , barrel% x lg P(HR|barrel) )
  x pull-air multiplier x hard-hit multiplier x pitcher factor x park factor.
  Contact volume (BBE/PA) only converts per-BBE -> per-PA; it never boosts rank
  beyond what plate appearances physically allow.

Hit MODEL (25,000 sims/player)
  Recency-weighted (exp decay, half-life 21d) contact quality:
  xBA-on-contact + LD% adjustment, times contact rate (1 - K/AB) matched
  against the starting pitcher via odds-ratio (log5) on K and contact quality,
  mild park factor.

All shrinkage is empirical Bayes toward league means. Fixed RNG seed per
(date, player) => fully reproducible boards. No fitted ML anywhere.

Reads  data/statcast_2026.parquet, data/slate.json, data/names.json
Writes props_board.json
"""

import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

MODEL_VERSION = "hrhit-1.0"
N_SIMS_HR, N_SIMS_HIT = 10_000, 25_000
HALF_LIFE_DAYS = 21           # hit-model recency decay
K_BBE = 60                    # EB pseudo-BBE for power rates
K_PA  = 120                   # EB pseudo-PA for K%/BB%
W_HRBBE, W_BARREL = 0.55, 0.45          # HR core blend
PULL_AIR_SPAN, HARDHIT_SPAN = 0.30, 0.15  # +/- max multiplier effect
PITCH_CLIP = (0.70, 1.35)
SLOT_PA = [4.65, 4.55, 4.45, 4.35, 4.25, 4.15, 4.05, 3.95, 3.85]

# HR park factor (100 = neutral). EDIT ANNUALLY from Savant Park Factors.
PARK_HR = {"CIN":131,"NYY":119,"LAA":114,"LAD":113,"PHI":112,"CWS":111,"MIL":109,
 "TEX":104,"COL":103,"ATL":103,"BAL":102,"HOU":101,"TOR":100,"CHC":100,"MIN":99,
 "ARI":98,"WSH":97,"NYM":96,"BOS":96,"CLE":95,"SD":94,"TB":93,"SEA":92,"STL":91,
 "DET":90,"ATH":103,"KC":87,"PIT":87,"MIA":85,"SF":82}
# Hits park factor (much flatter).
PARK_H = {"COL":110,"BOS":106,"KC":104,"CIN":102,"ARI":102,"MIA":101,"PIT":101,
 "TEX":100,"WSH":100,"LAA":100,"ATH":101,"CHC":99,"ATL":99,"BAL":99,"TOR":99,
 "PHI":99,"MIN":98,"DET":98,"CLE":98,"HOU":98,"SD":97,"STL":97,"NYM":97,
 "LAD":96,"CWS":96,"TB":96,"NYY":95,"MIL":95,"SF":94,"SEA":92}

D = Path("data")

def shrink(x, n, prior, k):
    return (x * n + prior * k) / (n + k)

def spray_deg(hc_x, hc_y):
    return np.degrees(np.arctan2(hc_x - 125.42, 198.27 - hc_y))

def load():
    df = pd.read_parquet(D / "statcast_2026.parquet")
    df["game_date"] = pd.to_datetime(df["game_date"])
    bbe = df["launch_speed"].notna() & df["bb_type"].notna()
    df["is_bbe"]    = bbe
    df["is_hr"]     = df["events"].eq("home_run")
    df["is_barrel"] = df["launch_speed_angle"].eq(6)
    df["is_hard"]   = bbe & (df["launch_speed"] >= 95)
    df["is_ld"]     = df["bb_type"].eq("line_drive")
    spray = spray_deg(df["hc_x"], df["hc_y"])
    pulled = np.where(df["stand"].eq("R"), spray < -15, spray > 15)
    df["is_pullair"] = bbe & pulled & (df["launch_angle"] >= 20)
    df["is_k"]  = df["events"].isin(["strikeout", "strikeout_double_play"])
    df["is_bb"] = df["events"].isin(["walk", "hit_by_pitch", "intent_walk"])
    df["is_hit"] = df["events"].isin(["single", "double", "triple", "home_run"])
    df["is_ab"] = ~df["is_bb"] & ~df["events"].isin(["sac_fly", "sac_bunt", "catcher_interf"])
    return df

def league(df):
    b = df[df["is_bbe"]]
    return dict(
        hr_bbe=b["is_hr"].mean(), barrel=b["is_barrel"].mean(),
        pullair=b["is_pullair"].mean(), hard=b["is_hard"].mean(),
        p_hr_barrel=b.loc[b["is_barrel"], "is_hr"].mean(),   # ~0.55
        bbe_pa=df["is_bbe"].mean(), k_pa=df["is_k"].mean(), bb_pa=df["is_bb"].mean(),
        xba_con=b["estimated_ba_using_speedangle"].fillna(0).mean(),
        ld=b["is_ld"].mean(),
        ba_con=b["is_hit"].mean(),
    )

def batter_power(df, lg):
    g = df.groupby("batter")
    n = g["is_bbe"].sum().rename("bbe")
    b = df[df["is_bbe"]].groupby("batter")
    out = pd.DataFrame({"bbe": n})
    for col, name, prior in (("is_hr","hr_bbe","hr_bbe"),("is_barrel","barrel","barrel"),
                             ("is_pullair","pullair","pullair"),("is_hard","hard","hard")):
        out[name] = shrink(b[col].mean().reindex(out.index).fillna(lg[prior]),
                           out["bbe"], lg[prior], K_BBE)
    pa = g.size().rename("pa")
    out["pa"] = pa
    out["bbe_pa"] = shrink((out["bbe"] / out["pa"]).fillna(lg["bbe_pa"]),
                           out["pa"], lg["bbe_pa"], K_PA)
    return out

def batter_contact(df, lg, asof):
    """Recency-weighted contact quality for the hit model."""
    w = 0.5 ** ((asof - df["game_date"]).dt.days / HALF_LIFE_DAYS)
    d = df.assign(w=w)
    g = d.groupby("batter")
    wsum_pa = g["w"].sum()
    def wmean(mask_col, sub=None):
        dd = d if sub is None else d[d[sub]]
        num = dd.groupby("batter").apply(lambda x: (x[mask_col] * x["w"]).sum())
        den = dd.groupby("batter")["w"].sum()
        return (num / den)
    out = pd.DataFrame(index=wsum_pa.index)
    out["w_pa"]  = wsum_pa
    out["w_bbe"] = d[d["is_bbe"]].groupby("batter")["w"].sum().reindex(out.index).fillna(0)
    out["k_ab"]  = shrink(wmean("is_k").fillna(lg["k_pa"]), out["w_pa"], lg["k_pa"], K_PA)
    out["bb_pa"] = shrink(wmean("is_bb").fillna(lg["bb_pa"]), out["w_pa"], lg["bb_pa"], K_PA)
    xba = d[d["is_bbe"]].groupby("batter").apply(
        lambda x: (x["estimated_ba_using_speedangle"].fillna(0) * x["w"]).sum() / x["w"].sum())
    out["xba_con"] = shrink(xba.reindex(out.index).fillna(lg["xba_con"]),
                            out["w_bbe"], lg["xba_con"], K_BBE)
    out["ld"] = shrink(wmean("is_ld", "is_bbe").reindex(out.index).fillna(lg["ld"]),
                       out["w_bbe"], lg["ld"], K_BBE)
    return out

def pitchers(df, lg):
    g = df.groupby("pitcher")
    b = df[df["is_bbe"]].groupby("pitcher")
    out = pd.DataFrame({"pa": g.size(), "bbe": g["is_bbe"].sum()})
    out["hr_bbe"] = shrink(b["is_hr"].mean().reindex(out.index).fillna(lg["hr_bbe"]),
                           out["bbe"], lg["hr_bbe"], K_BBE)
    out["barrel"] = shrink(b["is_barrel"].mean().reindex(out.index).fillna(lg["barrel"]),
                           out["bbe"], lg["barrel"], K_BBE)
    out["xba_con"] = shrink(df[df["is_bbe"]]
                            .assign(x=lambda d: d["estimated_ba_using_speedangle"].fillna(0))
                            .groupby("pitcher")["x"].mean()
                            .reindex(out.index).fillna(lg["xba_con"]),
                            out["bbe"], lg["xba_con"], K_BBE)
    out["k_pa"] = shrink(g["is_k"].mean(), out["pa"], lg["k_pa"], K_PA)
    return out

def log5(b, p, lg):
    """odds-ratio matchup blend"""
    num = b * p / lg
    return num / (num + (1 - b) * (1 - p) / (1 - lg))

def seed_for(day, pid):
    return int(hashlib.sha256(f"{day}:{pid}:{MODEL_VERSION}".encode()).hexdigest()[:8], 16)

def sim_atleast1(p_event, exp_n, n_sims, rng):
    n = rng.choice([max(2, round(exp_n) - 1), round(exp_n), round(exp_n) + 1],
                   size=n_sims, p=[0.2, 0.6, 0.2])
    return float((rng.binomial(n, p_event) >= 1).mean())

def main():
    df = load()
    slate = json.loads((D / "slate.json").read_text())
    names = json.loads((D / "names.json").read_text())
    lg = league(df)
    bp, pit = batter_power(df, lg), pitchers(df, lg)
    bc = batter_contact(df, lg, pd.Timestamp(slate["date"]))

    rows = []
    for gm in slate["games"]:
        for side in ("home", "away"):
            opp_sp = gm["probables"]["away" if side == "home" else "home"]
            park = gm["home"]
            hr_pf = PARK_HR.get(park, 100) / 100
            h_pf  = 1 + (PARK_H.get(park, 100) / 100 - 1) * 0.5   # damp hit PF
            P = pit.loc[opp_sp] if opp_sp in pit.index else None
            for slot, bid in enumerate(gm["lineups"][side]):
                if bid not in bp.index:  # no season data
                    continue
                B, C = bp.loc[bid], bc.loc[bid]
                exp_pa = SLOT_PA[slot] if slot < 9 else 4.1

                # ---- HR ----
                core = W_HRBBE * B["hr_bbe"] + W_BARREL * B["barrel"] * lg["p_hr_barrel"]
                m_pull = 1 + PULL_AIR_SPAN * (B["pullair"] / lg["pullair"] - 1)
                m_hard = 1 + HARDHIT_SPAN * (B["hard"] / lg["hard"] - 1)
                if P is not None:
                    m_pit = np.clip(0.5 * (P["barrel"] / lg["barrel"])
                                  + 0.5 * (P["hr_bbe"] / lg["hr_bbe"]), *PITCH_CLIP)
                    bbe_pa = B["bbe_pa"] * np.clip((1 - P["k_pa"]) / (1 - lg["k_pa"]), 0.8, 1.2)
                else:
                    m_pit, bbe_pa = 1.0, B["bbe_pa"]
                p_hr_pa = float(np.clip(core * m_pull * m_hard * m_pit * hr_pf * bbe_pa, 1e-4, 0.20))

                # ---- HIT ----
                ba_con = C["xba_con"] * (1 + 0.25 * (C["ld"] / lg["ld"] - 1))
                if P is not None:
                    k = log5(C["k_ab"], P["k_pa"], lg["k_pa"])
                    ba_con = log5(ba_con, P["xba_con"], lg["xba_con"])
                else:
                    k = C["k_ab"]
                # xBA-on-contact conditions on contact; scale by matchup K to get per-AB
                p_hit_ab = float(np.clip(ba_con * h_pf * (1 - k) / (1 - lg["k_pa"]), 0.05, 0.45))
                exp_ab = exp_pa * (1 - C["bb_pa"])

                rng_hr  = np.random.default_rng(seed_for(slate["date"], f"hr{bid}"))
                rng_hit = np.random.default_rng(seed_for(slate["date"], f"h{bid}"))
                rows.append({
                    "id": bid, "name": names.get(str(bid), str(bid)),
                    "team": gm[side], "opp": gm["away" if side == "home" else "home"],
                    "sp": names.get(str(opp_sp), "?"), "slot": slot + 1, "park": park,
                    "pHR": round(sim_atleast1(p_hr_pa, exp_pa, N_SIMS_HR, rng_hr), 4),
                    "pHit": round(sim_atleast1(p_hit_ab, exp_ab, N_SIMS_HIT, rng_hit), 4),
                    "hr_detail": {"hrBBE": round(B["hr_bbe"], 4), "barrel": round(B["barrel"], 3),
                                  "pullAir": round(B["pullair"], 3), "hard": round(B["hard"], 3),
                                  "mPit": round(float(m_pit), 3), "parkPF": PARK_HR.get(park, 100),
                                  "pHRperPA": round(p_hr_pa, 4)},
                    "hit_detail": {"xbaCon": round(C["xba_con"], 3), "ld": round(C["ld"], 3),
                                   "kAB": round(float(k), 3), "pHitAB": round(p_hit_ab, 3),
                                   "expAB": round(exp_ab, 2)},
                })

    rows.sort(key=lambda r: -r["pHR"])
    board = {"date": slate["date"], "model": MODEL_VERSION,
             "sims": {"hr": N_SIMS_HR, "hit": N_SIMS_HIT},
             "league": {k: round(v, 4) for k, v in lg.items()},
             "players": rows}
    Path("props_board.json").write_text(json.dumps(board, indent=1))
    print(f"board: {len(rows)} hitters, top pHR = "
          f"{rows[0]['name']} {rows[0]['pHR']:.1%}" if rows else "board: empty (no lineups yet)")

if __name__ == "__main__":
    main()
