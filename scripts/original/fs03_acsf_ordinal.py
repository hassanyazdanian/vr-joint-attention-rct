"""
Final sensitivity run - STEP 3: ACSF:SC ordinal (cumulative-logit) mixed models.

    Outcome ~ Group*Time + Verbal + (1 | participant)
proportional-odds cumulative logit, Gaussian participant random intercept,
Gauss-Hermite quadrature.

ODDS-RATIO DIRECTION - stated explicitly, because ACSF:SC is reverse-scored.
The model is parameterised as
        P(Y <= k) = logistic(theta_k - X'beta)
so a POSITIVE beta shifts probability towards HIGHER ACSF categories.
ACSF:SC is scored 1 = Level I (highest skill) ... 5 = Level V (lowest skill),
therefore:
        OR > 1  =>  higher odds of a WORSE (higher-numbered) ACSF level
        OR < 1  =>  higher odds of a BETTER (lower-numbered) ACSF level
An intervention BENEFIT therefore appears as OR < 1 on the Group x Time terms.

PROPORTIONAL-ODDS ASSESSMENT: for each cut point k the outcome is dichotomised
(Y >= k+1) and a binary logistic mixed model with the same fixed effects is
fitted. If proportional odds holds, the Group x Time coefficients should be
similar across cut points. Reported as an approximate Brant-type diagnostic -
the per-threshold fits are not independent, so the homogeneity test is
indicative rather than exact.

Writes acsf_ordinal_results.csv
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
from scipy import stats, optimize
import fs00_common as K

HERE = K.HERE
LONG = pd.read_csv(K.DER / "fs_long.csv")
PRIM = pd.read_csv(K.ROOT / "analysis_verbal_adjusted" / "contrast_results.csv")

# design without intercept (absorbed by the cutpoints)
NAMES = ["group", "time_T2", "time_T3", "group:T2", "group:T3", "verbal"]
I_GT2, I_GT3 = 3, 4


def build(sub):
    n = len(sub)
    t2 = (sub["time"] == "T2").astype(float).values
    t3 = (sub["time"] == "T3").astype(float).values
    g = sub["group"].astype(float).values
    return np.column_stack([g, t2, t3, g * t2, g * t3,
                            sub["verbal"].astype(float).values])


def binary_mixed_logit(yb, X, subj):
    """Binary logistic GLMM with a Gaussian random intercept (GH quadrature)."""
    Yp, Xp, Mk = K.pad_by_subject(yb.astype(float), X, subj)

    def nll(par):
        beta = par[:X.shape[1]]
        a = par[X.shape[1]]
        sd = np.exp(np.clip(par[-1], -8, 4))
        eta0 = Xp @ beta + a
        acc = np.zeros(Yp.shape[0])
        for xq, wq in zip(K._gx, K._gw):
            lin = eta0 + sd * xq
            p = 1 / (1 + np.exp(-np.clip(lin, -30, 30)))
            p = np.clip(p, 1e-12, 1 - 1e-12)
            ll = Yp * np.log(p) + (1 - Yp) * np.log(1 - p)
            acc += wq * np.exp(np.clip(np.where(Mk, ll, 0.0).sum(1), -700, 700))
        return -np.log(np.maximum(acc, 1e-300)).sum()

    p0 = np.concatenate([np.zeros(X.shape[1]), [0.0], [0.0]])
    best = None
    for meth in ("Nelder-Mead", "Powell"):
        r = optimize.minimize(nll, p0 if best is None else best.x, method=meth,
                              options={"maxiter": 40000, "maxfev": 80000})
        if best is None or r.fun < best.fun:
            best = r
    cov = K.numeric_cov(nll, best.x)
    return best.x[:X.shape[1]], np.sqrt(np.abs(np.diag(cov))[:X.shape[1]]), best


rows, po_rows = [], []
print("=" * 78)
print("ACSF:SC ORDINAL CUMULATIVE-LOGIT MIXED MODELS (verbal-adjusted)")
print("=" * 78)
for v in ["ACSF_tp", "ACSF_cc"]:
    sub = LONG[LONG[v].notna()].copy()
    levels = np.sort(sub[v].unique())
    Kn = len(levels)
    ycat = np.searchsorted(levels, sub[v].values)
    X = build(sub)
    subj = sub["id"].values
    f = K.fit_clmm(ycat, X, subj, Kn)

    L = np.zeros((2, X.shape[1])); L[0, I_GT2] = 1; L[1, I_GT3] = 1
    om = K.wald_omnibus(f["beta"], f["cov"], L)
    prim_om = PRIM[(PRIM.outcome == v) &
                   (PRIM.contrast == "Omnibus Group x Time (2 df)")]["adj_p"]
    rows.append({
        "outcome": v, "contrast": "Overall Group x Time", "levels_K": Kn,
        "estimate_logOR": np.nan, "se": np.nan, "OR": np.nan,
        "or_ci_lo": np.nan, "or_ci_hi": np.nan,
        "test": f"Wald chi2({om['df']}) = {om['chi2']:.4f}", "p": om["p"],
        "p_primary_LMM": float(prim_om.iloc[0]) if len(prim_om) else np.nan,
        "re_sd": f["sd_re"], "converged": f["converged"],
        "quasi_separated": f["separated"],
        "n_subj": f["n_subj"], "n_obs": f["n_obs"],
    })
    for lab, idx, plab in (
            ("T1->T2 treatment-related change", I_GT2,
             "Treatment difference in change T1->T2"),
            ("T1->T3 treatment-related change", I_GT3,
             "Treatment difference in change T1->T3")):
        c = np.zeros(X.shape[1]); c[idx] = 1.0
        w = K.wald(f["beta"], f["cov"], c)
        pr = PRIM[(PRIM.outcome == v) & (PRIM.contrast == plab)]["adj_p"]
        rows.append({
            "outcome": v, "contrast": lab, "levels_K": Kn,
            "estimate_logOR": w["estimate"], "se": w["se"],
            "OR": np.exp(w["estimate"]),
            "or_ci_lo": np.exp(w["ci_lo"]), "or_ci_hi": np.exp(w["ci_hi"]),
            "test": f"z = {w['z']:.4f}", "p": w["p"],
            "p_primary_LMM": float(pr.iloc[0]) if len(pr) else np.nan,
            "re_sd": f["sd_re"], "converged": f["converged"],
            "quasi_separated": f["separated"],
            "n_subj": f["n_subj"], "n_obs": f["n_obs"],
        })

    # ---------------- proportional-odds diagnostic
    print(f"\n{v}: proportional-odds check across {Kn - 1} cut points")
    for k in range(Kn - 1):
        yb = (ycat >= k + 1).astype(float)
        if yb.mean() in (0.0, 1.0) or min(yb.sum(), len(yb) - yb.sum()) < 5:
            po_rows.append({"outcome": v, "cutpoint": f">= level {levels[k+1]:.0f}",
                            "n_above": int(yb.sum()), "estimable": False,
                            "beta_groupT2": np.nan, "se_groupT2": np.nan,
                            "beta_groupT3": np.nan, "se_groupT3": np.nan})
            print(f"   cut >= {levels[k+1]:.0f}: too sparse "
                  f"({int(yb.sum())} of {len(yb)}) — not estimable")
            continue
        b, se, _ = binary_mixed_logit(yb, X, subj)
        po_rows.append({"outcome": v, "cutpoint": f">= level {levels[k+1]:.0f}",
                        "n_above": int(yb.sum()), "estimable": True,
                        "beta_groupT2": b[I_GT2], "se_groupT2": se[I_GT2],
                        "beta_groupT3": b[I_GT3], "se_groupT3": se[I_GT3]})
        print(f"   cut >= {levels[k+1]:.0f}: n above = {int(yb.sum())}/{len(yb)}, "
              f"group:T2 beta = {b[I_GT2]:+.3f} (SE {se[I_GT2]:.3f}), "
              f"group:T3 beta = {b[I_GT3]:+.3f} (SE {se[I_GT3]:.3f})")

PO = pd.DataFrame(po_rows)
R = pd.DataFrame(rows)

# approximate homogeneity test across cut points
po_sum = []
for v in ["ACSF_tp", "ACSF_cc"]:
    s = PO[(PO.outcome == v) & PO.estimable]
    for term in ("groupT2", "groupT3"):
        b = s[f"beta_{term}"].values
        se = s[f"se_{term}"].values
        if len(b) < 2 or not np.all(np.isfinite(se)) or np.any(se <= 0):
            po_sum.append({"outcome": v, "term": term, "n_cutpoints": len(b),
                           "spread_max_minus_min": np.nan,
                           "max_threshold_SE": np.nan, "Q": np.nan,
                           "df": np.nan, "p_homogeneity": np.nan,
                           "test_is_informative": False,
                           "assumption": "not assessable (too few estimable cut points)"})
            continue
        w = 1 / se ** 2
        bbar = float((w * b).sum() / w.sum())
        Q = float((w * (b - bbar) ** 2).sum())
        dfq = len(b) - 1
        p = float(stats.chi2.sf(Q, dfq))
        # A per-threshold SE above ~10 on the log-odds scale means that binary
        # model is separated / unidentified. The homogeneity test then has
        # essentially no power and a non-significant Q is NOT evidence that
        # proportional odds holds - it is evidence the check is uninformative.
        unstable = bool(np.any(se > 10))
        verdict = ("UNINFORMATIVE (per-threshold models unstable: "
                   f"max SE = {se.max():.0f}; sparse categories cause "
                   "separation, so the test has no power)" if unstable
                   else ("reasonable" if p >= 0.05 else "QUESTIONABLE"))
        po_sum.append({"outcome": v, "term": term, "n_cutpoints": len(b),
                       "spread_max_minus_min": float(b.max() - b.min()),
                       "max_threshold_SE": float(se.max()),
                       "Q": Q, "df": dfq, "p_homogeneity": p,
                       "test_is_informative": not unstable,
                       "assumption": verdict})
POS = pd.DataFrame(po_sum)

R = R.merge(POS.groupby("outcome").agg(
    po_min_p=("p_homogeneity", "min"),
    po_verdict=("assumption", lambda s: "QUESTIONABLE"
                if s.str.startswith("QUESTIONABLE").any() else
                ("reasonable" if (s == "reasonable").any()
                 else "NOT ASSESSABLE / UNINFORMATIVE"))
).reset_index(), on="outcome", how="left")
R["OR_direction_note"] = ("Model: P(Y<=k)=logistic(theta_k - X'beta). OR > 1 = "
                          "higher odds of a WORSE (higher-numbered) ACSF level; "
                          "OR < 1 = benefit.")
R.to_csv(HERE / "acsf_ordinal_results.csv", index=False)
PO.to_csv(K.DER / "acsf_proportional_odds_cutpoints.csv", index=False)
POS.to_csv(K.DER / "acsf_proportional_odds_test.csv", index=False)

print()
print("=" * 78)
print("RESULTS")
print("=" * 78)
print(R[["outcome", "contrast", "estimate_logOR", "se", "OR", "or_ci_lo",
         "or_ci_hi", "p", "p_primary_LMM", "converged", "quasi_separated"]]
      .round(4).to_string(index=False))
print()
print("Proportional-odds diagnostic:")
print(POS.round(4).to_string(index=False))
print()
for _, r in R[R.contrast != "Overall Group x Time"].iterrows():
    same = (r.p < 0.05) == (r.p_primary_LMM < 0.05)
    print(f"  {r.outcome:8s} {r.contrast:33s} ordinal p={r.p:.4f} vs "
          f"primary LMM p={r.p_primary_LMM:.4f}  -> "
          f"{'same conclusion' if same else '*** DIFFERS ***'}")
print()
print("DONE.")
