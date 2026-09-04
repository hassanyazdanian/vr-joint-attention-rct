"""
Final sensitivity run - STEPS 6 and 9: stratified permutation, and multiplicity.

SECTION 6 - stratified permutation
  Treatment labels are permuted WITHIN the original verbal/nonverbal strata,
  reflecting the stratified randomization. The analysis uses 100,000
  permutations and retains the verbal-status adjustment term. Covariance
  parameters are estimated once from the observed verbal-adjusted model and
  then held fixed across permutations.

  Implementation note: with X = [B | g*C], B = [1, T2, T3, verbal] and
  C = [1, T2, T3], and group constant within participant, the GLS normal
  equations collapse to per-participant matrices that are precomputed once.
  The matrix reduction introduces no additional approximation conditional on
  the fixed covariance estimate.

SECTION 9 - multiplicity
  Same strategy as the current revised analysis: Holm and Benjamini-Hochberg
  within the protocol-defined families
      PRIMARY   = joint attention      (behavioural sample, C-JARS, JAST)
      SECONDARY = social communication (GARS-3, ACSF:SC)
  Final outcome set only: 16 outcomes. GARS_tot_6, the verbal-only GARS
  subscales, and the reaction-time variables are all excluded from the
  multiplicity families. Raw and adjusted p-values are kept in separate columns.

Reads the deposited participant-level analysis file and contrast results.
Writes results/stratified_permutation_results.csv and
results/multiplicity_results.csv.
"""
import hashlib
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from scipy import optimize

import fs00_common as K

RESULTS = K.ROOT / "results"
LONG = (
    pd.read_csv(K.ROOT / "data" / "analysis_long.csv")
    .sort_values(["id", "time"])
    .reset_index(drop=True)
)
CON = pd.read_csv(RESULTS / "contrast_results.csv")
TIDX = {t: i for i, t in enumerate(K.TPS)}
BASE_SEED = 20260825
NPERM = 100_000

KEY9 = ["CJARS_sjas", "JAST_arja", "JAST_aija", "GARS_rr", "GARS_si",
        "GARS_sc", "GARS_er", "ACSF_cc", "BS_RJA"]


def make_rng(outcome):
    """Return a deterministic RNG whose stream does not depend on KEY9 order."""
    token = f"{BASE_SEED}:{outcome}".encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(token).digest()[:8], "little")
    return np.random.default_rng(seed)


def permutation_p(exceedances, n_permutations):
    """Monte Carlo permutation p-value with the standard plus-one correction."""
    return (exceedances + 1) / (n_permutations + 1)


def build_sigma(theta):
    s2b, s2e = np.exp(theta[0]), np.exp(theta[1])
    return s2b * np.ones((3, 3)) + s2e * np.eye(3)


def m2reml(theta, blocks, p):
    Sig = build_sigma(theta)
    XtSX = np.zeros((p, p)); XtSy = np.zeros(p); logdet = 0.0
    for X, y, oi in blocks:
        Si_ = Sig[np.ix_(oi, oi)]
        try:
            L = np.linalg.cholesky(Si_)
        except np.linalg.LinAlgError:
            return 1e10
        logdet += 2 * np.log(np.diag(L)).sum()
        Sinv = np.linalg.inv(Si_)
        XtSX += X.T @ Sinv @ X; XtSy += X.T @ Sinv @ y
    try:
        Lx = np.linalg.cholesky(XtSX)
    except np.linalg.LinAlgError:
        return 1e10
    beta = np.linalg.solve(XtSX, XtSy)
    quad = sum((y - X @ beta) @ np.linalg.solve(Sig[np.ix_(oi, oi)], y - X @ beta)
               for X, y, oi in blocks)
    N = sum(len(y) for _, y, _ in blocks)
    return logdet + quad + 2 * np.log(np.diag(Lx)).sum() + (N - p) * np.log(2 * np.pi)


print("=" * 78)
print(f"SECTION 6 — STRATIFIED PERMUTATION ({NPERM:,} permutations)")
print("=" * 78)
prows = []
for v in KEY9:
    rng = make_rng(v)
    sub_all = []
    for pid, s in LONG.groupby("id", sort=True):
        s = s[s[v].notna()]
        if len(s):
            sub_all.append((pid, s))
    # covariance estimated once from the observed data, verbal-adjusted model
    blocks_fit = []
    for pid, s in sub_all:
        t2 = (s["time"] == "T2").astype(float).values
        t3 = (s["time"] == "T3").astype(float).values
        g = s["group"].astype(float).values
        w = s["verbal"].astype(float).values
        X = np.column_stack([np.ones(len(s)), t2, t3, w, g, g * t2, g * t3])
        blocks_fit.append((X, s[v].astype(float).values,
                           [TIDX[t] for t in s["time"]]))
    yv = LONG[v].dropna()
    x0 = np.array([np.log(np.var(yv) / 2 + 1e-6), np.log(np.var(yv) / 2 + 1e-6)])
    r = optimize.minimize(m2reml, x0, args=(blocks_fit, 7), method="Nelder-Mead",
                          options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-10})
    Sig = build_sigma(r.x)

    P, Q, R, U, V, g0, strat = [], [], [], [], [], [], []
    for pid, s in sub_all:
        oi = [TIDX[t] for t in s["time"]]
        Si = np.linalg.inv(Sig[np.ix_(oi, oi)])
        y = s[v].astype(float).values
        t2 = (s["time"] == "T2").astype(float).values
        t3 = (s["time"] == "T3").astype(float).values
        w = s["verbal"].astype(float).values
        B = np.column_stack([np.ones(len(s)), t2, t3, w])   # non-group block
        C = np.column_stack([np.ones(len(s)), t2, t3])      # multiplied by group
        P.append(B.T @ Si @ B); Q.append(B.T @ Si @ C); R.append(C.T @ Si @ C)
        U.append(B.T @ Si @ y); V.append(C.T @ Si @ y)
        g0.append(float(s["group"].iloc[0])); strat.append(int(s["verbal"].iloc[0]))
    P, Q, R = np.array(P), np.array(Q), np.array(R)
    U, V = np.array(U), np.array(V)
    g0 = np.array(g0); strat = np.array(strat)

    def solve(gv):
        A11 = P.sum(0)
        A12 = (Q * gv[:, None, None]).sum(0)
        A22 = (R * gv[:, None, None]).sum(0)
        XtSX = np.block([[A11, A12], [A12.T, A22]])
        XtSy = np.concatenate([U.sum(0), (V * gv[:, None]).sum(0)])
        cov = np.linalg.inv(XtSX)
        return cov @ XtSy, cov

    beta, cov = solve(g0)
    # index map in [B | g*C]: 4 = group, 5 = group:T2, 6 = group:T3
    CS = {"T1->T2 difference in change": np.eye(7)[5],
          "T1->T3 difference in change": np.eye(7)[6]}
    L = np.zeros((2, 7)); L[0, 5] = 1; L[1, 6] = 1
    obs_t = {lab: (c @ beta) / np.sqrt(c @ cov @ c) for lab, c in CS.items()}
    d = L @ beta
    obs_F = float(d @ np.linalg.solve(L @ cov @ L.T, d) / 2)

    null_t = {lab: np.empty(NPERM) for lab in CS}
    null_F = np.empty(NPERM)
    for b in range(NPERM):
        gp = g0.copy()
        for s_ in np.unique(strat):                # permute WITHIN stratum
            m = strat == s_
            gp[m] = rng.permutation(g0[m])
        bb, cc = solve(gp)
        for lab, c in CS.items():
            null_t[lab][b] = (c @ bb) / np.sqrt(c @ cc @ c)
        dd = L @ bb
        null_F[b] = float(dd @ np.linalg.solve(L @ cc @ L.T, dd) / 2)

    om = CON[(CON.outcome == v) &
             (CON.contrast == "Omnibus Group x Time (2 df)")]
    prows.append({
        "outcome": v, "contrast": "Omnibus Group x Time",
        "observed_statistic": obs_F, "statistic_type": "Wald F (2 df)",
        "permutation_p": permutation_p(
            int((null_F >= obs_F).sum()), NPERM
        ),
        "n_permutations": NPERM,
        "p_parametric_verbal_adjusted": float(om["adj_p"].iloc[0]) if len(om) else np.nan,
    })
    for lab, c in CS.items():
        plab = ("Treatment difference in change T1->T2" if "T2" in lab
                else "Treatment difference in change T1->T3")
        pr = CON[(CON.outcome == v) & (CON.contrast == plab)]
        prows.append({
            "outcome": v, "contrast": lab,
            "observed_statistic": obs_t[lab], "statistic_type": "t (Wald)",
            "permutation_p": permutation_p(
                int((np.abs(null_t[lab]) >= abs(obs_t[lab])).sum()), NPERM
            ),
            "n_permutations": NPERM,
            "p_parametric_verbal_adjusted": float(pr["adj_p"].iloc[0]) if len(pr) else np.nan,
        })
    print(f"  {v:11s} done")

PERM = pd.DataFrame(prows)
PERM["permutation_mcse"] = np.sqrt(
    PERM.permutation_p * (1.0 - PERM.permutation_p) / (NPERM + 1)
)
PERM["near_0_05"] = (
    (PERM.permutation_p - 0.05).abs() <= 2 * PERM.permutation_mcse
)
PERM["stratification"] = "labels permuted within verbal/nonverbal strata"
PERM.to_csv(RESULTS / "stratified_permutation_results.csv", index=False)
print()
print(PERM[["outcome", "contrast", "observed_statistic", "permutation_p",
            "permutation_mcse", "p_parametric_verbal_adjusted", "near_0_05"]]
      .round(4).to_string(index=False))
print()
print(f"Tests within two Monte Carlo SEs of 0.05: "
      f"{int(PERM.near_0_05.sum())}/{len(PERM)}")

# ==================================================== multiplicity
print()
print("=" * 78)
print("SECTION 9 — MULTIPLICITY (final outcome set)")
print("=" * 78)
PRIMARY_FAM = ["CJARS_ss", "CJARS_ps", "CJARS_sjas", "BS_RJA", "BS_IJA",
               "BS_EC", "BS_FL", "BS_T", "JAST_arja", "JAST_aija"]
SECOND_FAM = ["GARS_rr", "GARS_si", "GARS_sc", "GARS_er", "ACSF_tp", "ACSF_cc"]
FAM = {**{v: "PRIMARY (joint attention)" for v in PRIMARY_FAM},
       **{v: "SECONDARY (social communication)" for v in SECOND_FAM}}
assert "GARS_tot_6" not in FAM and not any(v.startswith("JAST_RT") for v in FAM)


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    adj = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(adj, 1.0); return out


def holm(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    adj = np.maximum.accumulate(p[o] * (n - np.arange(n)))
    out = np.empty(n); out[o] = np.minimum(adj, 1.0); return out


LABS = ["Treatment difference in change T1->T2",
        "Treatment difference in change T1->T3"]
d = CON[CON.outcome.isin(FAM) & CON.contrast.isin(LABS)].copy()
d["family"] = d.outcome.map(FAM)
d["test"] = d.outcome + " | " + d.contrast.str.replace(
    "Treatment difference in change ", "", regex=False)
d = d[["outcome", "family", "contrast", "test", "adj_p"]].rename(
    columns={"adj_p": "p_raw"}).reset_index(drop=True)
d["Holm_within_family"] = np.nan
d["BH_within_family"] = np.nan
for fam, idx in d.groupby("family").groups.items():
    idx = list(idx)
    d.loc[idx, "Holm_within_family"] = holm(d.loc[idx, "p_raw"].values)
    d.loc[idx, "BH_within_family"] = bh(d.loc[idx, "p_raw"].values)
d["Holm_all"] = holm(d.p_raw.values)
d["BH_all"] = bh(d.p_raw.values)
for c in ["p_raw", "Holm_within_family", "BH_within_family", "Holm_all", "BH_all"]:
    d[c + "_sig"] = d[c] < 0.05
d["model"] = "verbal-adjusted primary LMM"
d["estimand"] = "difference in change from T1 (Group x Time contrast)"
d["excluded_from_families"] = "GARS_tot_6, GARS_cs, GARS_ms, JAST_RT_t1..t4"
d.to_csv(RESULTS / "multiplicity_results.csv", index=False)
print(d[["test", "family", "p_raw", "Holm_within_family", "BH_within_family",
         "Holm_all", "BH_all"]].round(4).to_string(index=False))
print()
print(f"  raw p < .05                : {int(d.p_raw_sig.sum())}/{len(d)}")
print(f"  Holm within family < .05   : {int(d.Holm_within_family_sig.sum())}/{len(d)}")
print(f"  BH within family < .05     : {int(d.BH_within_family_sig.sum())}/{len(d)}")
print(f"  Holm across all {len(d)} < .05  : {int(d.Holm_all_sig.sum())}/{len(d)}")
print(f"  BH across all {len(d)} < .05    : {int(d.BH_all_sig.sum())}/{len(d)}")
print()
print("DONE.")
