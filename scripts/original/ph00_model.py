"""
Post-hoc verbal-adjusted analysis - shared model machinery.

This is a SIDE-EFFECT-FREE EXTRACTION of the fitting and Satterthwaite machinery
in analysis_verbal_adjusted/scripts/va02_models.py. The functions below are
copied verbatim in substance; only the module-level code that WRITES
verbal_adjusted_model_results.csv / unadjusted_vs_verbal_adjusted.csv /
contrast_results.csv has been removed, because importing va02_models.py would
overwrite those archived files.

ph01 asserts that this module reproduces the archived
analysis_verbal_adjusted/contrast_results.csv exactly, so the extraction is
verified rather than assumed.

MODEL (final primary framework):
    Outcome ~ Group * Time + VerbalStatus + (1 | participant)
    Time categorical, T1 reference; compound symmetry; REML; all available
    observations; no imputation; NO verbal-status interactions.

NOTE: syntax/05b_itt_mixed.py is deliberately NOT used anywhere here - that is
the older UNADJUSTED model.
"""
import numpy as np
import pandas as pd
from scipy import optimize, stats
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent          # posthoc_verbal_adjusted/
ROOT = HERE.parent                                     
RAW = ROOT / "data_raw" / "VR_JA_DATA_reviewed_20260821.xlsx"
EXPECTED_SHA = "844f2d0e10f35908c1bb43ee8d8f061f394e80907c9fca46f27eddcdf0d5eb31"

GCOL = "Group (A=0, B=1)"
TPS = ["T1", "T2", "T3"]
TIDX = {t: i for i, t in enumerate(TPS)}
SHEETS = dict(zip(TPS, ["Pre(T1)", "Post(T2)", "Flu(T3)"]))
BS_DOM = ["RJA", "IJA", "EC", "FL", "T"]

FAMILIES = {
    "C-JARS":             ["CJARS_ss", "CJARS_ps", "CJARS_sjas"],
    "Behavioural sample": ["BS_RJA", "BS_IJA", "BS_EC", "BS_FL", "BS_T"],
    "JAST":               ["JAST_arja", "JAST_aija"],
    "GARS-3":             ["GARS_rr", "GARS_si", "GARS_sc", "GARS_er"],
    "ACSF:SC":            ["ACSF_tp", "ACSF_cc"],
}
FAM_OF = {v: f for f, vs in FAMILIES.items() for v in vs}
OUTCOMES = [v for vs in FAMILIES.values() for v in vs]
EXCLUDED_FROM_INFERENCE = ["GARS_tot_6", "GARS_cs", "GARS_ms",
                           "CJARS_ass", "CJARS_aps",
                           "JAST_RT_t1", "JAST_RT_t2", "JAST_RT_t3", "JAST_RT_t4"]

# --- required assertions on the outcome set
assert len(OUTCOMES) == 16, f"expected 16 outcomes, got {len(OUTCOMES)}"
for _v in EXCLUDED_FROM_INFERENCE:
    assert _v not in OUTCOMES, f"{_v} must be excluded from inference"

DIRECTION = {"ACSF_tp": -1, "ACSF_cc": -1, "GARS_rr": -1, "GARS_si": -1,
             "GARS_sc": -1, "GARS_er": -1, "CJARS_ss": -1, "CJARS_ps": +1,
             "CJARS_sjas": +1, "BS_RJA": +1, "BS_IJA": +1, "BS_EC": +1,
             "BS_FL": +1, "BS_T": +1, "JAST_arja": +1, "JAST_aija": +1}

# beta = [Intercept, group, time_T2, time_T3, group:T2, group:T3, verbal]
NAMES = ["Intercept", "group", "time_T2", "time_T3",
         "group:T2", "group:T3", "verbal"]
NPAR = 7


def verify_input():
    import hashlib
    h = hashlib.sha256(RAW.read_bytes()).hexdigest()
    if h != EXPECTED_SHA:
        raise SystemExit(
            f"ABORT: raw workbook SHA-256 mismatch.\n  expected {EXPECTED_SHA}"
            f"\n  found    {h}\nEvery downstream number assumes the original file.")
    return h


def load_long():
    """Participant x timepoint long format, behavioural sample = MEAN of M and Z."""
    def rd(sheet):
        d = pd.read_excel(RAW, sheet_name=sheet, keep_default_na=False,
                          na_values=["NA"])
        return d[d["id"].notna()].assign(id=lambda x: x["id"].astype(int))

    S = {t: rd(s) for t, s in SHEETS.items()}
    for t in TPS:
        for dom in BS_DOM:
            S[t][f"BS_{dom}"] = S[t][[f"BS_{dom}_M", f"BS_{dom}_Z"]].mean(axis=1)
    pre = S["T1"]
    ids = list(pre["id"])
    grp = pre.set_index("id")[GCOL]
    verb = pre.set_index("id")["demog_Verbal"]
    rows = []
    for t in TPS:
        d = S[t].set_index("id")
        for pid in ids:
            rec = {"id": pid, "group": int(grp[pid]), "time": t,
                   "verbal": int(verb[pid])}
            for v in OUTCOMES:
                rec[v] = d.at[pid, v]
            rows.append(rec)
    return pd.DataFrame(rows).sort_values(["id", "time"]).reset_index(drop=True)


# ------------------------------------------------- machinery from va02_models.py
def design(sub):
    n = len(sub)
    t2 = (sub["time"] == "T2").astype(float).values
    t3 = (sub["time"] == "T3").astype(float).values
    g = sub["group"].astype(float).values
    return np.column_stack([np.ones(n), g, t2, t3, g * t2, g * t3,
                            sub["verbal"].astype(float).values])


def blocks_for(LONG, var):
    out = []
    for pid, sub in LONG.groupby("id"):
        sub = sub[sub[var].notna()]
        if len(sub) == 0:
            continue
        out.append((pid, design(sub), sub[var].astype(float).values,
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
    N = sum(len(b[2]) for b in blocks)
    return logdet + quad + logdetX + (N - p) * np.log(2 * np.pi)


def fit(LONG, var):
    """Fit the verbal-adjusted model. Returns beta, cov and the two
    Satterthwaite routines (single contrast and Fai-Cornelius multi-row)."""
    blocks = blocks_for(LONG, var)
    p = NPAR
    y_all = LONG[var].dropna()
    x0 = np.array([np.log(np.var(y_all) / 2 + 1e-6),
                   np.log(np.var(y_all) / 2 + 1e-6)])
    r = optimize.minimize(m2reml, x0, args=(blocks, p), method="Nelder-Mead",
                          options={"maxiter": 20000, "xatol": 1e-10,
                                   "fatol": 1e-10})
    r = optimize.minimize(m2reml, r.x, args=(blocks, p), method="Nelder-Mead",
                          options={"maxiter": 20000, "xatol": 1e-12,
                                   "fatol": 1e-12})
    theta = r.x
    Sig = build_sigma(theta)
    beta, cov = gls(blocks, Sig, p)
    N = sum(len(b[2]) for b in blocks)

    h = 1e-5
    k = len(theta)
    H = np.zeros((k, k))
    f0 = m2reml(theta, blocks, p)
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k); ei[i] = h
            ej = np.zeros(k); ej[j] = h
            H[i, j] = H[j, i] = (m2reml(theta + ei + ej, blocks, p)
                                 - m2reml(theta + ei - ej, blocks, p)
                                 - m2reml(theta - ei + ej, blocks, p)
                                 + m2reml(theta - ei - ej, blocks, p)) / (4 * h * h)
    A = 2 * np.linalg.pinv(H)

    def var_contrast(th, c):
        _, C = gls(blocks, build_sigma(th), p)
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
        """Fai & Cornelius (1996) spectral rule, as SPSS DDFM=SATTERTH uses."""
        q = L.shape[0]
        C = L @ cov @ L.T
        _, Gam = np.linalg.eigh(C)
        rows_ = Gam.T @ L
        nus = [nu for nu in (satterthwaite(rows_[m]) for m in range(q))
               if np.isfinite(nu) and nu > 2]
        if not nus:
            return np.nan
        E = float(np.sum([nu / (nu - 2) for nu in nus]))
        return 2 * E / (E - q) if E > q else np.nan

    return {"theta": theta, "Sigma": Sig, "beta": beta, "cov": cov, "p": p,
            "N_obs": N, "n_subj": len(blocks), "m2RLL": f0,
            "satterthwaite": satterthwaite,
            "satterthwaite_multi": satterthwaite_multi,
            "converged": bool(r.success)}


# ---------------------------------------------------------------- contrasts
def cvec(spec):
    c = np.zeros(NPAR)
    for i, val in spec.items():
        c[i] = val
    return c


# within-arm pairwise, on beta = [1, g, T2, T3, gT2, gT3, verbal]
WITHIN_ARM = {
    ("Control", "T2 - T1"):      cvec({2: 1}),
    ("Control", "T3 - T2"):      cvec({2: -1, 3: 1}),
    ("Control", "T3 - T1"):      cvec({3: 1}),
    ("Intervention", "T2 - T1"): cvec({2: 1, 4: 1}),
    ("Intervention", "T3 - T2"): cvec({2: -1, 3: 1, 4: -1, 5: 1}),
    ("Intervention", "T3 - T1"): cvec({3: 1, 5: 1}),
}

# omnibus tests. Time is averaged over the two arms with equal weight, matching
# the SPSS Type III test of `time` in the presence of group*time.
L_TIME = np.array([[0, 0, 1, 0, 0.5, 0, 0],
                   [0, 0, 0, 1, 0, 0.5, 0]], float)
L_INTER = np.array([[0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0]], float)


def contrast_stats(f, c, bonferroni_m=1):
    """Estimate, SE, contrast-specific Satterthwaite df, t, unadjusted p,
    Bonferroni-adjusted p, and a Bonferroni-widened CI consistent with it."""
    est = float(c @ f["beta"])
    se = float(np.sqrt(c @ f["cov"] @ c))
    dfv = float(f["satterthwaite"](c))
    t = est / se
    p_un = float(2 * stats.t.sf(abs(t), dfv))
    p_bf = float(min(1.0, p_un * bonferroni_m))
    alpha_adj = 0.05 / bonferroni_m
    crit = float(stats.t.ppf(1 - alpha_adj / 2, dfv))
    return {"estimate": est, "se": se, "df_satterthwaite": dfv, "t": t,
            "p_unadjusted": p_un, "p_bonferroni": p_bf,
            "ci_lo_bonferroni": est - crit * se,
            "ci_hi_bonferroni": est + crit * se,
            "bonferroni_m": bonferroni_m,
            "ci_level": f"{100 * (1 - alpha_adj):.4g}% per-comparison "
                        f"(= 95% familywise over {bonferroni_m})"}


def omnibus(f, L):
    d = L @ f["beta"]
    V = L @ f["cov"] @ L.T
    W = float(d @ np.linalg.solve(V, d))
    F = W / L.shape[0]
    dfden = float(f["satterthwaite_multi"](L))
    return {"F": F, "df1": L.shape[0], "df2": dfden,
            "p": float(stats.f.sf(F, L.shape[0], dfden))}
