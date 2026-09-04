"""
Final sensitivity run - STEP 2: behavioural-sample count models, verbal-adjusted.

PRIMARY behavioural analysis (unchanged, not refitted here) is the linear mixed
model on the MEAN of raters M and Z.

This script fits two count-model sensitivity analyses:

  (A) RATER-SUM negative binomial   y = BS_<dom>_M + BS_<dom>_Z
      Outcome ~ Group*Time + Verbal + (1 | participant)

      *** INTERPRETATION WARNING, stated explicitly in the outputs ***
      The sum M+Z is used ONLY because a count likelihood requires integers and
      the rater mean is a half-integer. It is a SENSITIVITY REPRESENTATION OF
      THE TWO RATINGS OF THE SAME BEHAVIOURAL SAMPLE. It is NOT twice the number
      of observed behavioural events: the two raters coded the SAME session, so
      the sum double-counts each event by construction. Rate ratios are
      interpretable (the factor 2 cancels in any ratio); absolute counts are not.

  (B) RATER-SPECIFIC negative binomial, preserving each rater's own integer count
      Outcome ~ Group*Time + Verbal + Rater + (1 | participant)
      Reported separately as an EXPLORATORY robustness check. It does not
      replace the primary mean-of-raters analysis.
      Limitation recorded: the two rater rows for one participant-occasion are
      not independent; with only a participant-level random intercept plus the
      NB dispersion term, residual within-occasion correlation is not fully
      modelled, so its standard errors may be slightly anti-conservative.

Writes behavioral_nb_results.csv and behavioral_rater_specific_results.csv
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
from scipy import stats
import fs00_common as K

HERE = K.HERE
LONG = pd.read_csv(K.DER / "fs_long.csv")
RLONG = pd.read_csv(K.DER / "fs_rater_long.csv")
PRIM = pd.read_csv(K.ROOT / "analysis_verbal_adjusted" / "contrast_results.csv")

CONTRASTS = {
    "T1->T2 difference in change": K.IDX_GT2,
    "T1->T3 difference in change": K.IDX_GT3,
}


def run(frame, ycol, extra_cols, tag):
    sub = frame[frame[ycol].notna()].copy()
    extra = [(nm, sub[src].astype(float).values) for nm, src in extra_cols]
    X, names = K.design(sub, extra)
    y = sub[ycol].astype(float).values
    subj = sub["id"].values
    res = {}
    for fam in ("poisson", "nb"):
        res[fam] = K.fit_count(y, X, subj, family=fam)
    lrt = 2 * (res["poisson"]["nll"] - res["nb"]["nll"])
    p_lrt = 0.5 * stats.chi2.sf(max(lrt, 0.0), 1)
    f = res["nb"]
    out = []
    L = np.zeros((2, X.shape[1])); L[0, K.IDX_GT2] = 1; L[1, K.IDX_GT3] = 1
    om = K.wald_omnibus(f["beta"], f["cov"], L)
    out.append({"model": tag, "outcome": ycol, "contrast": "Omnibus Group x Time",
                "estimate": np.nan, "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "IRR": np.nan, "IRR_ci_lo": np.nan, "IRR_ci_hi": np.nan,
                "test": f"Wald chi2({om['df']}) = {om['chi2']:.4f}",
                "p": om["p"]})
    for lab, idx in CONTRASTS.items():
        c = np.zeros(X.shape[1]); c[idx] = 1.0
        w = K.wald(f["beta"], f["cov"], c)
        out.append({"model": tag, "outcome": ycol, "contrast": lab,
                    "estimate": w["estimate"], "se": w["se"],
                    "ci_lo": w["ci_lo"], "ci_hi": w["ci_hi"],
                    "IRR": np.exp(w["estimate"]),
                    "IRR_ci_lo": np.exp(w["ci_lo"]),
                    "IRR_ci_hi": np.exp(w["ci_hi"]),
                    "test": f"z = {w['z']:.4f}", "p": w["p"]})
    for r in out:
        r.update({
            "dispersion_alpha": f["alpha"],
            "dispersion_note": ("NB alpha; larger = more overdispersion. "
                                "Poisson is alpha -> 0."),
            "LRT_nb_vs_poisson": lrt, "p_LRT_overdispersion": p_lrt,
            "overdispersed": p_lrt < 0.05,
            "re_sd": f["sd_re"], "converged": f["converged"],
            "n_subj": f["n_subj"], "n_obs": f["n_obs"],
            "n_params": len(f["par"]),
        })
    return out


print("=" * 78)
print("(A) RATER-SUM NEGATIVE BINOMIAL, verbal-adjusted")
print("=" * 78)
rowsA = []
for dom in K.BS_DOM:
    LONG[f"BS_{dom}_sum"] = LONG[f"BS_{dom}_M"] + LONG[f"BS_{dom}_Z"]
    rowsA += run(LONG, f"BS_{dom}_sum", [], "A_rater_sum_NB")
A = pd.DataFrame(rowsA)
A["outcome"] = A["outcome"].str.replace("_sum", "", regex=False)
A["representation"] = ("SUM of the two raters' counts of the SAME session. "
                       "A sensitivity representation of the two ratings, "
                       "NOT twice the observed behavioural events.")
A.to_csv(HERE / "behavioral_nb_results.csv", index=False)
print(A[["outcome", "contrast", "estimate", "se", "ci_lo", "ci_hi", "IRR",
         "p", "dispersion_alpha", "converged"]].round(4).to_string(index=False))

print()
print("=" * 78)
print("(B) RATER-SPECIFIC NEGATIVE BINOMIAL — exploratory robustness check")
print("=" * 78)
RLONG["rater_Z"] = (RLONG["rater"] == "Z").astype(float)
rowsB = []
for dom in K.BS_DOM:
    rowsB += run(RLONG, f"BS_{dom}", [("rater_Z", "rater_Z")], "B_rater_specific_NB")
B = pd.DataFrame(rowsB)
B["representation"] = ("Each rater's own integer count retained as a separate "
                       "row; Rater (M/Z) included as a fixed effect.")
B["limitation"] = ("Two rater rows per participant-occasion are not independent; "
                   "only a participant-level random intercept is fitted, so SEs "
                   "may be slightly anti-conservative. Exploratory only.")
B.to_csv(HERE / "behavioral_rater_specific_results.csv", index=False)
print(B[["outcome", "contrast", "estimate", "se", "ci_lo", "ci_hi", "IRR",
         "p", "dispersion_alpha", "converged"]].round(4).to_string(index=False))

# ------------------------------------------------ comparison with primary LMM
print()
print("=" * 78)
print("COMPARISON WITH THE PRIMARY VERBAL-ADJUSTED LMM (mean of raters)")
print("=" * 78)
cmp_rows = []
MAP = {"Omnibus Group x Time": "Omnibus Group x Time (2 df)",
       "T1->T2 difference in change": "Treatment difference in change T1->T2",
       "T1->T3 difference in change": "Treatment difference in change T1->T3"}
for dom in K.BS_DOM:
    v = f"BS_{dom}"
    for lab, plab in MAP.items():
        pr = PRIM[(PRIM.outcome == v) & (PRIM.contrast == plab)]
        p_prim = float(pr["adj_p"].iloc[0]) if len(pr) else np.nan
        pa = A[(A.outcome == v) & (A.contrast == lab)]["p"]
        pb = B[(B.outcome == v) & (B.contrast == lab)]["p"]
        pa = float(pa.iloc[0]) if len(pa) else np.nan
        pb = float(pb.iloc[0]) if len(pb) else np.nan
        cmp_rows.append({
            "outcome": v, "contrast": lab,
            "p_primary_LMM_verbal_adj": p_prim,
            "p_NB_rater_sum": pa, "p_NB_rater_specific": pb,
            "primary_sig": p_prim < 0.05, "nb_sum_sig": pa < 0.05,
            "nb_rater_sig": pb < 0.05,
            "material_difference_vs_primary":
                "YES" if (p_prim < 0.05) != (pa < 0.05) else "no",
        })
C = pd.DataFrame(cmp_rows)
C.to_csv(K.DER / "behavioural_nb_vs_primary.csv", index=False)
print(C.round(4).to_string(index=False))
print()
n_diff = int((C.material_difference_vs_primary == "YES").sum())
print(f"Behavioural outcomes where NB (rater-sum) inference differs materially "
      f"from the primary LMM: {n_diff}/{len(C)}")
if n_diff:
    print(C[C.material_difference_vs_primary == "YES"]
          [["outcome", "contrast", "p_primary_LMM_verbal_adj", "p_NB_rater_sum"]]
          .round(4).to_string(index=False))
print()
print("DONE.")
