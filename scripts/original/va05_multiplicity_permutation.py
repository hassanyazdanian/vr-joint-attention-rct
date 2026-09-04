"""
Verbal-adjusted sensitivity analysis - STEP 5.

  (10) Multiplicity: regenerate Holm and Benjamini-Hochberg adjusted p-values on
       the CORRECTED manuscript outcome set (16 outcomes; GARS_tot_6 EXCLUDED),
       for both the unadjusted and the verbal-adjusted model. The multiplicity
       strategy itself is unchanged - families follow the protocol hierarchy:
           PRIMARY   = joint attention      (behavioural sample, C-JARS, JAST)
           SECONDARY = social communication (GARS-3, ACSF:SC)
       Raw and adjusted p-values are kept in separate, clearly named columns.

  (11) Stratified permutation: treatment labels permuted WITHIN verbal/nonverbal
       strata, reflecting the stratified randomization. Kept entirely separate
       from the primary mixed-model results.

  (exploratory) An explicitly labelled Group x Verbal / Time x Verbal /
       three-way diagnostic. NOT used for any primary comparison and NOT part of
       the sensitivity conclusion.

Writes:
    analysis_verbal_adjusted/multiplicity_adjusted.csv
    analysis_verbal_adjusted/stratified_permutation.csv
    analysis_verbal_adjusted/exploratory_verbal_interaction.csv
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
from scipy import stats, optimize
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
DER = HERE / "derived"
LONG = pd.read_csv(DER / "va_long.csv")
CON = pd.read_csv(HERE / "contrast_results.csv")
TPS = ["T1", "T2", "T3"]
TIDX = {t: i for i, t in enumerate(TPS)}
rng = np.random.default_rng(20260824)

PRIMARY = ["CJARS_ss", "CJARS_ps", "CJARS_sjas", "BS_RJA", "BS_IJA",
           "BS_EC", "BS_FL", "BS_T", "JAST_arja", "JAST_aija"]
SECONDARY = ["GARS_rr", "GARS_si", "GARS_sc", "GARS_er", "ACSF_tp", "ACSF_cc"]
OUTCOMES = PRIMARY + SECONDARY
assert "GARS_tot_6" not in OUTCOMES and len(OUTCOMES) == 16
FAMILY = {**{v: "PRIMARY (joint attention)" for v in PRIMARY},
          **{v: "SECONDARY (social communication)" for v in SECONDARY}}


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    adj = p[o] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(adj, 1.0)
    return out


def holm(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    adj = np.maximum.accumulate(p[o] * (n - np.arange(n)))
    out = np.empty(n); out[o] = np.minimum(adj, 1.0)
    return out


# ============================================================ multiplicity
print("=" * 78)
print("MULTIPLICITY — 16 outcomes, GARS_tot_6 EXCLUDED")
print("=" * 78)
rows = []
CONTRAST_SETS = {
    "difference-in-change (T1->T2, T1->T3)":
        ["Treatment difference in change T1->T2",
         "Treatment difference in change T1->T3"],
    "marginal group difference (at T2, at T3)":
        ["Group difference at T2", "Group difference at T3"],
}
for setname, labels in CONTRAST_SETS.items():
    sub = CON[CON.outcome.isin(OUTCOMES) & CON.contrast.isin(labels)].copy()
    sub["family"] = sub.outcome.map(FAMILY)
    sub["test"] = sub.outcome + " | " + sub.contrast
    for model, pcol in (("A_unadjusted", "unadj_p"), ("B_verbal_adjusted", "adj_p")):
        d = sub[["outcome", "family", "contrast", "test", pcol]].copy()
        d = d.rename(columns={pcol: "p_raw"}).reset_index(drop=True)
        d["contrast_set"] = setname
        d["model"] = model
        d["BH_within_family"] = np.nan
        d["Holm_within_family"] = np.nan
        for fam, idx in d.groupby("family").groups.items():
            idx = list(idx)
            d.loc[idx, "BH_within_family"] = bh(d.loc[idx, "p_raw"].values)
            d.loc[idx, "Holm_within_family"] = holm(d.loc[idx, "p_raw"].values)
        d["BH_all_32"] = bh(d["p_raw"].values)
        d["Holm_all_32"] = holm(d["p_raw"].values)
        rows.append(d)
MUL = pd.concat(rows, ignore_index=True)
for c in ["p_raw", "BH_within_family", "Holm_within_family", "BH_all_32",
          "Holm_all_32"]:
    MUL[c + "_sig"] = MUL[c] < 0.05
MUL.to_csv(HERE / "multiplicity_adjusted.csv", index=False)

for setname in CONTRAST_SETS:
    print(f"\n{setname}")
    for model in ("A_unadjusted", "B_verbal_adjusted"):
        s = MUL[(MUL.contrast_set == setname) & (MUL.model == model)]
        print(f"   {model:18s} raw p<.05 {int(s.p_raw_sig.sum()):2d}/{len(s)}   "
              f"BH(family) {int(s.BH_within_family_sig.sum()):2d}   "
              f"Holm(family) {int(s.Holm_within_family_sig.sum()):2d}   "
              f"BH(all 32) {int(s.BH_all_32_sig.sum()):2d}   "
              f"Holm(all 32) {int(s.Holm_all_32_sig.sum()):2d}")
print()
print("Outcomes whose BH-within-family status differs between models:")
piv = MUL.pivot_table(index=["contrast_set", "test"], columns="model",
                      values="BH_within_family_sig", aggfunc="first")
diff = piv[piv["A_unadjusted"] != piv["B_verbal_adjusted"]]
print("   none" if not len(diff) else diff.to_string())

# ================================================== stratified permutation
print()
print("=" * 78)
print("STRATIFIED PERMUTATION — labels permuted WITHIN verbal strata")
print("=" * 78)
NPERM = 5000
# column order [1, t2, t3 | g, g*t2, g*t3]  (+ verbal appended when adjusted)
C_T2_MARG = np.array([0, 0, 0, 1, 1, 0.0])
C_T3_MARG = np.array([0, 0, 0, 1, 0, 1.0])
C_T2_DIC = np.array([0, 0, 0, 0, 1, 0.0])
C_T3_DIC = np.array([0, 0, 0, 0, 0, 1.0])
CSET = {"Group difference at T2": C_T2_MARG,
        "Group difference at T3": C_T3_MARG,
        "Treatment difference in change T1->T2": C_T2_DIC,
        "Treatment difference in change T1->T3": C_T3_DIC}


def build_sigma(theta):
    s2b, s2e = np.exp(theta[0]), np.exp(theta[1])
    return s2b * np.ones((3, 3)) + s2e * np.eye(3)


def m2reml_simple(theta, blocks, p):
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


prows = []
for v in OUTCOMES:
    # precompute per-participant pieces; permuting arm labels only moves a
    # participant between the g and non-g partitions
    ids, A_list, c_list, g0, strat, Bs = [], [], [], [], [], []
    sub_all = []
    for pid, sub in LONG.groupby("id"):
        sub = sub[sub[v].notna()]
        if len(sub) == 0:
            continue
        sub_all.append((pid, sub))
    # fit once to get Sigma (unadjusted model)
    blocks_fit = []
    for pid, sub in sub_all:
        t2 = (sub["time"] == "T2").astype(float).values
        t3 = (sub["time"] == "T3").astype(float).values
        g = sub["group"].astype(float).values
        X = np.column_stack([np.ones(len(sub)), g, t2, t3, g * t2, g * t3])
        blocks_fit.append((X, sub[v].astype(float).values,
                           [TIDX[t] for t in sub["time"]]))
    yv = LONG[v].dropna()
    x0 = np.array([np.log(np.var(yv) / 2 + 1e-6), np.log(np.var(yv) / 2 + 1e-6)])
    r = optimize.minimize(m2reml_simple, x0, args=(blocks_fit, 6),
                          method="Nelder-Mead",
                          options={"maxiter": 20000, "xatol": 1e-10,
                                   "fatol": 1e-10})
    Sig = build_sigma(r.x)

    for pid, sub in sub_all:
        oi = [TIDX[t] for t in sub["time"]]
        Si = np.linalg.inv(Sig[np.ix_(oi, oi)])
        y = sub[v].astype(float).values
        t2 = (sub["time"] == "T2").astype(float).values
        t3 = (sub["time"] == "T3").astype(float).values
        B = np.column_stack([np.ones(len(sub)), t2, t3])
        A_list.append(B.T @ Si @ B); c_list.append(B.T @ Si @ y)
        g0.append(int(sub["group"].iloc[0]))
        strat.append(int(sub["verbal"].iloc[0]))
    A = np.array(A_list); Cc = np.array(c_list)
    g0 = np.array(g0, float); strat = np.array(strat)

    def solve(gv):
        S0 = A.sum(0); S1 = (A * gv[:, None, None]).sum(0)
        t0 = Cc.sum(0); t1 = (Cc * gv[:, None]).sum(0)
        cov = np.linalg.inv(np.block([[S0, S1], [S1, S1]]))
        return cov @ np.concatenate([t0, t1]), cov

    beta, cov = solve(g0)
    obs = {lab: (c @ beta) / np.sqrt(c @ cov @ c) for lab, c in CSET.items()}
    null = {lab: np.empty(NPERM) for lab in CSET}
    for b in range(NPERM):
        gp = g0.copy()
        for s_ in np.unique(strat):            # permute WITHIN verbal stratum
            m = strat == s_
            gp[m] = rng.permutation(g0[m])
        bb, cc = solve(gp)
        for lab, c in CSET.items():
            null[lab][b] = (c @ bb) / np.sqrt(c @ cc @ c)
    for lab, c in CSET.items():
        mod = CON[(CON.outcome == v) & (CON.contrast == lab)]
        prows.append({
            "outcome": v, "family": FAMILY[v], "contrast": lab,
            "p_model_unadjusted": float(mod["unadj_p"].iloc[0]),
            "p_model_verbal_adjusted": float(mod["adj_p"].iloc[0]),
            "p_permutation_stratified": float(
                (np.abs(null[lab]) >= abs(obs[lab])).mean()),
            "n_permutations": NPERM,
        })
PERM = pd.DataFrame(prows)
PERM["model_sig"] = PERM.p_model_unadjusted < 0.05
PERM["perm_sig"] = PERM.p_permutation_stratified < 0.05
PERM["disagreement"] = PERM.model_sig != PERM.perm_sig
PERM.to_csv(HERE / "stratified_permutation.csv", index=False)
print(PERM[["outcome", "contrast", "p_model_unadjusted",
            "p_model_verbal_adjusted", "p_permutation_stratified",
            "disagreement"]].round(4).to_string(index=False))
print()
print(f"Contrasts where the stratified permutation test disagrees with the "
      f"model: {int(PERM.disagreement.sum())}/{len(PERM)}")
if PERM.disagreement.any():
    print(PERM[PERM.disagreement][["outcome", "contrast", "p_model_unadjusted",
                                   "p_permutation_stratified"]]
          .round(4).to_string(index=False))

# ============================================ EXPLORATORY diagnostic only
print()
print("=" * 78)
print("EXPLORATORY DIAGNOSTIC — Group x Verbal / Time x Verbal / three-way")
print("*** NOT part of the primary comparison. Reported for diagnosis only. ***")
print("=" * 78)
erows = []
for v in OUTCOMES:
    blocks = []
    for pid, sub in LONG.groupby("id"):
        sub = sub[sub[v].notna()]
        if len(sub) == 0:
            continue
        n = len(sub)
        t2 = (sub["time"] == "T2").astype(float).values
        t3 = (sub["time"] == "T3").astype(float).values
        g = sub["group"].astype(float).values
        w = sub["verbal"].astype(float).values
        X = np.column_stack([np.ones(n), g, t2, t3, g * t2, g * t3, w,
                             g * w, w * t2, w * t3, g * w * t2, g * w * t3])
        blocks.append((X, sub[v].astype(float).values,
                       [TIDX[t] for t in sub["time"]]))
    p = 12
    yv = LONG[v].dropna()
    x0 = np.array([np.log(np.var(yv) / 2 + 1e-6), np.log(np.var(yv) / 2 + 1e-6)])
    r = optimize.minimize(m2reml_simple, x0, args=(blocks, p),
                          method="Nelder-Mead",
                          options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-10})
    Sig = build_sigma(r.x)
    XtSX = np.zeros((p, p)); XtSy = np.zeros(p)
    for X, y, oi in blocks:
        Si = np.linalg.inv(Sig[np.ix_(oi, oi)])
        XtSX += X.T @ Si @ X; XtSy += X.T @ Si @ y
    cov = np.linalg.inv(XtSX); beta = cov @ XtSy
    N = sum(len(b[1]) for b in blocks)
    dfr = N - p
    L3 = np.zeros((2, p)); L3[0, 10] = 1; L3[1, 11] = 1     # 3-way
    d = L3 @ beta; V = L3 @ cov @ L3.T
    F3 = float(d @ np.linalg.solve(V, d) / 2)
    erows.append({"outcome": v, "family": FAMILY[v],
                  "three_way_F": F3, "three_way_df1": 2, "three_way_df2": dfr,
                  "three_way_p": float(stats.f.sf(F3, 2, dfr)),
                  "note": "EXPLORATORY ONLY - not used for any primary comparison"})
EXP = pd.DataFrame(erows)
EXP["three_way_sig"] = EXP.three_way_p < 0.05
EXP.to_csv(HERE / "exploratory_verbal_interaction.csv", index=False)
print(EXP[["outcome", "three_way_F", "three_way_p", "three_way_sig"]]
      .round(4).to_string(index=False))
print(f"\nThree-way Group x Time x Verbal significant at p<.05: "
      f"{int(EXP.three_way_sig.sum())}/16 (uncorrected; with 16 tests roughly "
      f"{16*0.05:.1f} are expected by chance alone)")
print()
print("DONE.")
