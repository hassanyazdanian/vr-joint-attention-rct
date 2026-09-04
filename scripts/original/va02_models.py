"""
Verbal-adjusted sensitivity analysis - STEP 2: fit Model A and Model B.

  Model A (current) : Outcome ~ Group * Time              + (1 | participant)
  Model B (adjusted): Outcome ~ Group * Time + Verbal     + (1 | participant)

Time is a categorical factor with T1 as reference. Random-effects / covariance
specification is UNCHANGED from the validated pipeline: compound symmetry
(equivalently a participant random intercept), REML, all available observations,
no imputation, participants analysed as randomized.

Verbal status enters as a FIXED BASELINE ADJUSTMENT TERM ONLY. No Group x Verbal,
Time x Verbal or three-way term is fitted here (an explicitly labelled exploratory
diagnostic is in va05_exploratory_interaction.py and is NOT used for any primary
comparison).

DEGREES OF FREEDOM: contrast-specific Satterthwaite df are computed for EVERY
contrast, not reused from the omnibus test. For a contrast c'beta with variance
g(theta) = c'C(theta)c, df = 2 g^2 / (grad_g' A grad_g), where A is the asymptotic
covariance of the REML covariance-parameter estimates, A = 2 * inv(Hessian of the
-2logL_R objective). Verified against SPSS 26 in va04_spss_crosscheck.py.

Writes:
    analysis_verbal_adjusted/verbal_adjusted_model_results.csv
    analysis_verbal_adjusted/unadjusted_vs_verbal_adjusted.csv
    analysis_verbal_adjusted/contrast_results.csv
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
from scipy import optimize, stats
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent
DER = HERE / "derived"

LONG = pd.read_csv(DER / "va_long.csv")
TPS = ["T1", "T2", "T3"]
TIDX = {t: i for i, t in enumerate(TPS)}

FAMILIES = {
    "C-JARS":             ["CJARS_ss", "CJARS_ps", "CJARS_sjas"],
    "Behavioural sample": ["BS_RJA", "BS_IJA", "BS_EC", "BS_FL", "BS_T"],
    "JAST":               ["JAST_arja", "JAST_aija"],
    "GARS-3":             ["GARS_rr", "GARS_si", "GARS_sc", "GARS_er"],
    "ACSF:SC":            ["ACSF_tp", "ACSF_cc"],
}
FAM_OF = {v: f for f, vs in FAMILIES.items() for v in vs}
OUTCOMES = [v for vs in FAMILIES.values() for v in vs]
assert "GARS_tot_6" not in OUTCOMES and len(OUTCOMES) == 16

DIRECTION = {"ACSF_tp": -1, "ACSF_cc": -1, "GARS_rr": -1, "GARS_si": -1,
             "GARS_sc": -1, "GARS_er": -1, "CJARS_ss": -1, "CJARS_ps": +1,
             "CJARS_sjas": +1, "BS_RJA": +1, "BS_IJA": +1, "BS_EC": +1,
             "BS_FL": +1, "BS_T": +1, "JAST_arja": +1, "JAST_aija": +1}

# ---------------------------------------------------------------- design
NAMES_A = ["Intercept", "group", "time_T2", "time_T3", "group:T2", "group:T3"]
NAMES_B = NAMES_A + ["verbal"]


def design(sub, adjusted):
    n = len(sub)
    t2 = (sub["time"] == "T2").astype(float).values
    t3 = (sub["time"] == "T3").astype(float).values
    g = sub["group"].astype(float).values
    cols = [np.ones(n), g, t2, t3, g * t2, g * t3]
    if adjusted:
        cols.append(sub["verbal"].astype(float).values)
    return np.column_stack(cols)


def blocks_for(var, adjusted):
    out = []
    for pid, sub in LONG.groupby("id"):
        sub = sub[sub[var].notna()]
        if len(sub) == 0:
            continue
        out.append((pid, design(sub, adjusted), sub[var].astype(float).values,
                    [TIDX[t] for t in sub["time"]]))
    return out


def build_sigma(theta):
    """Compound symmetry == participant random intercept."""
    s2b, s2e = np.exp(theta[0]), np.exp(theta[1])
    return s2b * np.ones((3, 3)) + s2e * np.eye(3)


def gls(blocks, Sig, p):
    XtSX = np.zeros((p, p))
    XtSy = np.zeros(p)
    for _, X, y, oi in blocks:
        Si = np.linalg.inv(Sig[np.ix_(oi, oi)])
        XtSX += X.T @ Si @ X
        XtSy += X.T @ Si @ y
    cov = np.linalg.inv(XtSX)
    return cov @ XtSy, cov


def m2reml(theta, blocks, p):
    Sig = build_sigma(theta)
    XtSX = np.zeros((p, p))
    XtSy = np.zeros(p)
    logdet = 0.0
    for _, X, y, oi in blocks:
        Si_ = Sig[np.ix_(oi, oi)]
        try:
            L = np.linalg.cholesky(Si_)
        except np.linalg.LinAlgError:
            return 1e10
        logdet += 2 * np.log(np.diag(L)).sum()
        Sinv = np.linalg.inv(Si_)
        XtSX += X.T @ Sinv @ X
        XtSy += X.T @ Sinv @ y
    try:
        Lx = np.linalg.cholesky(XtSX)
    except np.linalg.LinAlgError:
        return 1e10
    logdetX = 2 * np.log(np.diag(Lx)).sum()
    beta = np.linalg.solve(XtSX, XtSy)
    quad = 0.0
    for _, X, y, oi in blocks:
        r = y - X @ beta
        quad += r @ np.linalg.solve(Sig[np.ix_(oi, oi)], r)
    N = sum(len(y) for _, y, _ in [(b[0], b[2], b[3]) for b in blocks])
    return logdet + quad + logdetX + (N - p) * np.log(2 * np.pi)


def fit(var, adjusted):
    blocks = blocks_for(var, adjusted)
    p = 7 if adjusted else 6
    y_all = LONG[var].dropna()
    x0 = np.array([np.log(np.var(y_all) / 2 + 1e-6), np.log(np.var(y_all) / 2 + 1e-6)])
    r = optimize.minimize(m2reml, x0, args=(blocks, p), method="Nelder-Mead",
                          options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-10})
    r = optimize.minimize(m2reml, r.x, args=(blocks, p), method="Nelder-Mead",
                          options={"maxiter": 20000, "xatol": 1e-12, "fatol": 1e-12})
    theta = r.x
    Sig = build_sigma(theta)
    beta, cov = gls(blocks, Sig, p)
    N = sum(len(b[2]) for b in blocks)

    # --- asymptotic covariance of theta: A = 2 * inv(Hessian of -2logL_R)
    h = 1e-5
    k = len(theta)
    H = np.zeros((k, k))
    f0 = m2reml(theta, blocks, p)
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k); ei[i] = h
            ej = np.zeros(k); ej[j] = h
            val = (m2reml(theta + ei + ej, blocks, p)
                   - m2reml(theta + ei - ej, blocks, p)
                   - m2reml(theta - ei + ej, blocks, p)
                   + m2reml(theta - ei - ej, blocks, p)) / (4 * h * h)
            H[i, j] = H[j, i] = val
    A = 2 * np.linalg.pinv(H)

    def var_contrast(th, c):
        Sg = build_sigma(th)
        _, C = gls(blocks, Sg, p)
        return float(c @ C @ c)

    def satterthwaite(c):
        g0 = var_contrast(theta, c)
        grad = np.zeros(k)
        for i in range(k):
            e = np.zeros(k); e[i] = h
            grad[i] = (var_contrast(theta + e, c)
                       - var_contrast(theta - e, c)) / (2 * h)
        vg = float(grad @ A @ grad)
        if vg <= 0 or not np.isfinite(vg):
            return np.nan
        return 2 * g0 ** 2 / vg

    def satterthwaite_multi(L):
        """Fai & Cornelius (1996) spectral decomposition, as SPSS/SAS DDFM=SATTERTH
        uses for a multi-row contrast. Reduces L to q orthogonal single-df rows,
        takes each row's own Satterthwaite df, then pools them."""
        q = L.shape[0]
        C = L @ cov @ L.T
        w, Gam = np.linalg.eigh(C)                 # C = Gam diag(w) Gam'
        rows_ = Gam.T @ L                          # orthogonalised contrast rows
        nus = []
        for m in range(q):
            nu = satterthwaite(rows_[m])
            if np.isfinite(nu) and nu > 2:
                nus.append(nu)
        if not nus:
            return np.nan
        E = float(np.sum([nu / (nu - 2) for nu in nus]))
        return 2 * E / (E - q) if E > q else np.nan

    return {"theta": theta, "Sigma": Sig, "beta": beta, "cov": cov, "p": p,
            "N_obs": N, "n_subj": len(blocks), "m2RLL": f0,
            "blocks": blocks, "satterthwaite": satterthwaite,
            "satterthwaite_multi": satterthwaite_multi,
            "converged": bool(r.success)}


# contrasts, on beta = [1, group, T2, T3, gT2, gT3 (, verbal)]
def cvec(spec, p):
    c = np.zeros(p)
    for i, val in spec.items():
        c[i] = val
    return c


CONTRASTS = {
    "Group difference at T1":              {1: 1},
    "Group difference at T2":              {1: 1, 4: 1},
    "Group difference at T3":              {1: 1, 5: 1},
    "Treatment difference in change T1->T2": {4: 1},
    "Treatment difference in change T1->T3": {5: 1},
}

# baseline pooled SD for standardised effects (identical in both models)
BASE_SD = {}
for v in OUTCOMES:
    b = LONG[(LONG.time == "T1") & LONG[v].notna()]
    n0 = int((b.group == 0).sum()); n1 = int((b.group == 1).sum())
    s0 = b.loc[b.group == 0, v].std(ddof=1); s1 = b.loc[b.group == 1, v].std(ddof=1)
    BASE_SD[v] = np.sqrt(((n0 - 1) * s0 ** 2 + (n1 - 1) * s1 ** 2) / (n0 + n1 - 2))

full_rows, cmp_rows, con_rows = [], [], []

print("=" * 78)
print("FITTING MODEL A (unadjusted) AND MODEL B (verbal-adjusted)")
print("=" * 78)
for v in OUTCOMES:
    fits = {"A_unadjusted": fit(v, False), "B_verbal_adjusted": fit(v, True)}
    for mname, f in fits.items():
        se = np.sqrt(np.diag(f["cov"]))
        names = NAMES_B if f["p"] == 7 else NAMES_A
        for i, nm in enumerate(names):
            c = np.zeros(f["p"]); c[i] = 1.0
            dfree = f["satterthwaite"](c)
            t = f["beta"][i] / se[i]
            full_rows.append({
                "outcome": v, "family": FAM_OF[v], "model": mname,
                "term": nm, "estimate": f["beta"][i], "se": se[i],
                "df_satterthwaite": dfree,
                "t": t, "p": 2 * stats.t.sf(abs(t), dfree),
                "ci_lo": f["beta"][i] - stats.t.ppf(0.975, dfree) * se[i],
                "ci_hi": f["beta"][i] + stats.t.ppf(0.975, dfree) * se[i],
                "n_subj": f["n_subj"], "N_obs": f["N_obs"],
                "var_between": float(np.exp(f["theta"][0])),
                "var_residual": float(np.exp(f["theta"][1])),
                "m2RLL": f["m2RLL"], "converged": f["converged"],
            })

    # ---- contrasts, each with its OWN Satterthwaite df
    per = {}
    for label, spec in CONTRASTS.items():
        row = {"outcome": v, "family": FAM_OF[v], "contrast": label,
               "scoring": "higher = better" if DIRECTION[v] > 0 else "lower = better"}
        for mname, f in fits.items():
            tag = "unadj" if mname.startswith("A") else "adj"
            c = cvec(spec, f["p"])
            est = float(c @ f["beta"])
            sev = float(np.sqrt(c @ f["cov"] @ c))
            dfree = f["satterthwaite"](c)
            t = est / sev
            pv = 2 * stats.t.sf(abs(t), dfree)
            crit = stats.t.ppf(0.975, dfree)
            row.update({
                f"{tag}_estimate": est, f"{tag}_se": sev,
                f"{tag}_df": dfree, f"{tag}_t": t, f"{tag}_p": pv,
                f"{tag}_ci_lo": est - crit * sev, f"{tag}_ci_hi": est + crit * sev,
                f"{tag}_d": est / BASE_SD[v],
            })
        per[label] = row
        con_rows.append(row)

    # ---- omnibus group x time (2 df joint Wald F), own Satterthwaite denominator
    om = {"outcome": v, "family": FAM_OF[v],
          "contrast": "Omnibus Group x Time (2 df)"}
    for mname, f in fits.items():
        tag = "unadj" if mname.startswith("A") else "adj"
        L = np.zeros((2, f["p"])); L[0, 4] = 1; L[1, 5] = 1
        d = L @ f["beta"]
        V = L @ f["cov"] @ L.T
        F = float(d @ np.linalg.solve(V, d) / 2)
        # denominator df by the Fai-Cornelius spectral rule (matches SPSS)
        dd = float(f["satterthwaite_multi"](L))
        om.update({f"{tag}_F": F, f"{tag}_num_df": 2, f"{tag}_den_df": dd,
                   f"{tag}_p": float(stats.f.sf(F, 2, dd))})
    con_rows.append(om)
    per["Omnibus Group x Time (2 df)"] = om

    # ---- side-by-side comparison rows
    for label in ["Omnibus Group x Time (2 df)",
                  "Treatment difference in change T1->T2",
                  "Treatment difference in change T1->T3",
                  "Group difference at T2", "Group difference at T3"]:
        r = per[label]
        if label.startswith("Omnibus"):
            up, ap = r["unadj_p"], r["adj_p"]
            ue = ae = us = as_ = np.nan
        else:
            up, ap = r["unadj_p"], r["adj_p"]
            ue, ae = r["unadj_estimate"], r["adj_estimate"]
            us, as_ = r["unadj_se"], r["adj_se"]
        changed = ("YES" if (up < 0.05) != (ap < 0.05) else "no")
        borderline = "YES" if (0.04 < ap < 0.06) or (0.04 < up < 0.06) else "no"
        cmp_rows.append({
            "outcome": v, "family": FAM_OF[v], "effect_contrast": label,
            "unadjusted_estimate": ue, "adjusted_estimate": ae,
            "unadjusted_se": us, "adjusted_se": as_,
            "unadjusted_p": up, "adjusted_p": ap,
            "unadjusted_sig": up < 0.05, "adjusted_sig": ap < 0.05,
            "change_in_conclusion": changed, "borderline_0.04_0.06": borderline,
            "abs_p_shift": abs(ap - up),
        })
    print(f"  {v:11s} A: -2RLL {fits['A_unadjusted']['m2RLL']:9.3f}   "
          f"B: -2RLL {fits['B_verbal_adjusted']['m2RLL']:9.3f}   "
          f"n={fits['A_unadjusted']['n_subj']}, obs={fits['A_unadjusted']['N_obs']}")

FULL = pd.DataFrame(full_rows)
CMP = pd.DataFrame(cmp_rows)
CON = pd.DataFrame(con_rows)
FULL.to_csv(HERE / "verbal_adjusted_model_results.csv", index=False)
CMP.to_csv(HERE / "unadjusted_vs_verbal_adjusted.csv", index=False)
CON.to_csv(HERE / "contrast_results.csv", index=False)

print()
print("=" * 78)
print("VERBAL-STATUS ADJUSTMENT TERM (Model B)")
print("=" * 78)
vt = FULL[(FULL.model == "B_verbal_adjusted") & (FULL.term == "verbal")]
print(vt[["outcome", "estimate", "se", "df_satterthwaite", "t", "p"]]
      .round(4).to_string(index=False))
print(f"\nVerbal term significant at p<.05 in {int((vt.p < 0.05).sum())}/16 outcomes")

print()
print("=" * 78)
print("CHANGE IN CONCLUSION")
print("=" * 78)
ch = CMP[CMP.change_in_conclusion == "YES"]
print(f"Contrasts whose significance status changes: {len(ch)}/{len(CMP)}")
if len(ch):
    print(ch[["outcome", "effect_contrast", "unadjusted_p", "adjusted_p",
              "unadjusted_sig", "adjusted_sig"]].round(4).to_string(index=False))
print()
bl = CMP[CMP["borderline_0.04_0.06"] == "YES"]
print(f"Borderline contrasts (0.04 < p < 0.06 in either model): {len(bl)}")
if len(bl):
    print(bl[["outcome", "effect_contrast", "unadjusted_p", "adjusted_p"]]
          .round(4).to_string(index=False))
print()
print(f"Largest |p shift|: {CMP.abs_p_shift.max():.4f} "
      f"({CMP.loc[CMP.abs_p_shift.idxmax(), 'outcome']} / "
      f"{CMP.loc[CMP.abs_p_shift.idxmax(), 'effect_contrast']})")
print()
print("DONE.")
