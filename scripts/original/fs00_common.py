"""
Final verbal-adjusted sensitivity analysis - shared machinery.

Model framework preserved throughout:
    Outcome ~ Group * Time + VerbalStatus + (1 | participant)
Time categorical (T1 reference). Verbal status is an ADJUSTMENT TERM ONLY -
no Group x Verbal, Time x Verbal or three-way term appears anywhere.

Provides:
  * data loaders (participant-level long, rater-level long)
  * negative-binomial / Poisson GLMM with a Gaussian participant random
    intercept, by adaptive-free Gauss-Hermite quadrature (vectorised)
  * cumulative-link (proportional-odds) mixed model, same quadrature
  * Wald contrast machinery: estimate, SE, 95% CI, z, p, and 2-df omnibus
"""
import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import gammaln
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent
RAW = ROOT / "data_raw" / "VR_JA_DATA_reviewed_20260821.xlsx"
DER = HERE / "derived"
DER.mkdir(parents=True, exist_ok=True)

GCOL = "Group (A=0, B=1)"
TPS = ["T1", "T2", "T3"]
SHEETS = dict(zip(TPS, ["Pre(T1)", "Post(T2)", "Flu(T3)"]))
BS_DOM = ["RJA", "IJA", "EC", "FL", "T"]
RT_VARS = [f"JAST_RT_t{i}" for i in (1, 2, 3, 4)]
WITHDRAWN = ['P24', 'P04']  # original identifiers replaced for publication

# ---- the FINAL outcome set. GARS_tot_6, GARS_cs, GARS_ms are absent by design.
FAMILIES = {
    "C-JARS":             ["CJARS_ss", "CJARS_ps", "CJARS_sjas"],
    "Behavioural sample": ["BS_RJA", "BS_IJA", "BS_EC", "BS_FL", "BS_T"],
    "JAST":               ["JAST_arja", "JAST_aija"],
    "GARS-3":             ["GARS_rr", "GARS_si", "GARS_sc", "GARS_er"],
    "ACSF:SC":            ["ACSF_tp", "ACSF_cc"],
}
FAM_OF = {v: f for f, vs in FAMILIES.items() for v in vs}
OUTCOMES = [v for vs in FAMILIES.values() for v in vs]
FORBIDDEN = ["GARS_tot_6", "GARS_cs", "GARS_ms", "CJARS_ass", "CJARS_aps"]
assert not any(f in OUTCOMES for f in FORBIDDEN)
assert len(OUTCOMES) == 16

# +1 higher is better, -1 lower is better
DIRECTION = {"ACSF_tp": -1, "ACSF_cc": -1, "GARS_rr": -1, "GARS_si": -1,
             "GARS_sc": -1, "GARS_er": -1, "CJARS_ss": -1, "CJARS_ps": +1,
             "CJARS_sjas": +1, "BS_RJA": +1, "BS_IJA": +1, "BS_EC": +1,
             "BS_FL": +1, "BS_T": +1, "JAST_arja": +1, "JAST_aija": +1}


def _rd(sheet):
    d = pd.read_excel(RAW, sheet_name=sheet, keep_default_na=False, na_values=["NA"])
    return d[d["id"].notna()].assign(id=lambda x: x["id"].astype(int))


def load_sheets():
    S = {t: _rd(s) for t, s in SHEETS.items()}
    for t in TPS:
        for dom in BS_DOM:
            S[t][f"BS_{dom}"] = S[t][[f"BS_{dom}_M", f"BS_{dom}_Z"]].mean(axis=1)
    return S


def load_long():
    """Participant-level long: one row per participant per timepoint."""
    S = load_sheets()
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
            for v in OUTCOMES + RT_VARS:
                rec[v] = d.at[pid, v]
            for dom in BS_DOM:
                for r in ("M", "Z"):
                    rec[f"BS_{dom}_{r}"] = d.at[pid, f"BS_{dom}_{r}"]
            rows.append(rec)
    return pd.DataFrame(rows).sort_values(["id", "time"]).reset_index(drop=True)


def load_rater_long():
    """Rater-level long: one row per participant per timepoint per rater.
    Preserves the two raters' own integer counts instead of combining them."""
    L = load_long()
    out = []
    for _, r in L.iterrows():
        for rater in ("M", "Z"):
            rec = {"id": r["id"], "group": r["group"], "time": r["time"],
                   "verbal": r["verbal"], "rater": rater}
            for dom in BS_DOM:
                rec[f"BS_{dom}"] = r[f"BS_{dom}_{rater}"]
            out.append(rec)
    return pd.DataFrame(out)


# ===================================================== design matrices
def design(sub, extra=()):
    """[1, group, T2, T3, group*T2, group*T3, verbal, <extra>]"""
    n = len(sub)
    t2 = (sub["time"] == "T2").astype(float).values
    t3 = (sub["time"] == "T3").astype(float).values
    g = sub["group"].astype(float).values
    cols = [np.ones(n), g, t2, t3, g * t2, g * t3,
            sub["verbal"].astype(float).values]
    names = ["Intercept", "group", "time_T2", "time_T3",
             "group:T2", "group:T3", "verbal"]
    for nm, vals in extra:
        cols.append(np.asarray(vals, float))
        names.append(nm)
    return np.column_stack(cols), names


# indices of group:T2 and group:T3 in the design above
IDX_GT2, IDX_GT3 = 4, 5

GH_N = 24
_gx, _gw = np.polynomial.hermite_e.hermegauss(GH_N)
_gw = _gw / np.sqrt(2 * np.pi)


def pad_by_subject(y, X, subj):
    uniq = np.unique(subj)
    mx = max(int((subj == s).sum()) for s in uniq)
    Yp = np.zeros((len(uniq), mx))
    Xp = np.zeros((len(uniq), mx, X.shape[1]))
    Mk = np.zeros((len(uniq), mx), bool)
    for i, s in enumerate(uniq):
        m = subj == s
        k = int(m.sum())
        Yp[i, :k] = y[m]
        Xp[i, :k] = X[m]
        Mk[i, :k] = True
    return Yp, Xp, Mk


# ===================================================== count GLMM
def count_nll(par, Yp, Xp, Mk, family):
    nb_ = Xp.shape[2]
    beta = par[:nb_]
    sd = np.exp(np.clip(par[nb_], -8, 4))
    alpha = np.exp(np.clip(par[-1], -8, 6)) if family == "nb" else None
    eta0 = Xp @ beta
    acc = np.zeros(Yp.shape[0])
    for xq, wq in zip(_gx, _gw):
        mu = np.exp(np.clip(eta0 + sd * xq, -20, 20))
        if family == "poisson":
            ll = Yp * np.log(mu) - mu - gammaln(Yp + 1)
        else:
            r = 1.0 / alpha
            ll = (gammaln(Yp + r) - gammaln(r) - gammaln(Yp + 1)
                  + r * np.log(r / (r + mu)) + Yp * np.log(mu / (r + mu)))
        acc += wq * np.exp(np.clip(np.where(Mk, ll, 0.0).sum(1), -700, 700))
    return -np.log(np.maximum(acc, 1e-300)).sum()


def fit_count(y, X, subj, family="nb"):
    Yp, Xp, Mk = pad_by_subject(y, X, subj)
    b0 = np.linalg.lstsq(X, np.log(np.maximum(y, 0.5)), rcond=None)[0]
    p0 = np.concatenate([b0, [0.0]] + ([[0.0]] if family == "nb" else []))
    best = None
    for meth in ("Nelder-Mead", "Powell", "Nelder-Mead"):
        r = optimize.minimize(count_nll, p0 if best is None else best.x,
                              args=(Yp, Xp, Mk, family), method=meth,
                              options={"maxiter": 60000, "maxfev": 120000,
                                       "xatol": 1e-9, "fatol": 1e-9}
                              if meth == "Nelder-Mead"
                              else {"maxiter": 60000, "maxfev": 120000})
        if best is None or r.fun < best.fun:
            best = r
    par, nll = best.x, best.fun
    cov = numeric_cov(lambda p: count_nll(p, Yp, Xp, Mk, family), par)
    nb_ = X.shape[1]
    return {"par": par, "beta": par[:nb_], "cov": cov[:nb_, :nb_],
            "full_cov": cov, "nll": nll,
            "sd_re": float(np.exp(par[nb_])),
            "alpha": float(np.exp(par[-1])) if family == "nb" else np.nan,
            "converged": bool(best.success), "n_obs": len(y),
            "n_subj": len(np.unique(subj)), "family": family}


# ===================================================== ordinal (CLMM)
def clmm_nll(par, Yc, Xp, Mk, K):
    ncut = K - 1
    cuts = np.concatenate([[par[0]], par[0] + np.cumsum(np.exp(par[1:ncut]))])
    nb_ = Xp.shape[2]
    beta = par[ncut:ncut + nb_]
    sd = np.exp(np.clip(par[-1], -8, 4))
    eta0 = Xp @ beta
    acc = np.zeros(Yc.shape[0])
    lo_i = np.clip(Yc - 1, 0, ncut - 1)
    hi_i = np.clip(Yc, 0, ncut - 1)
    for xq, wq in zip(_gx, _gw):
        lin = eta0 + sd * xq
        lo = np.where(Yc == 0, -np.inf, cuts[lo_i])
        hi = np.where(Yc == K - 1, np.inf, cuts[hi_i])
        pr = (1 / (1 + np.exp(-(hi - lin)))) - (1 / (1 + np.exp(-(lo - lin))))
        pr = np.clip(pr, 1e-12, 1.0)
        acc += wq * np.exp(np.clip(np.where(Mk, np.log(pr), 0.0).sum(1), -700, 700))
    return -np.log(np.maximum(acc, 1e-300)).sum()


def fit_clmm(ycat, X, subj, K, ridge=0.0):
    """X must NOT contain an intercept column (absorbed by the cutpoints)."""
    Yc, Xp, Mk = pad_by_subject(ycat, X, subj)
    Yc = Yc.astype(int)
    ncut = K - 1
    props = np.cumsum(np.bincount(ycat, minlength=K) / len(ycat))[:-1]
    props = np.clip(props, 1e-3, 1 - 1e-3)
    c0 = np.log(props / (1 - props))
    p0 = np.concatenate([[c0[0]], np.log(np.maximum(np.diff(c0), 1e-3)),
                         np.zeros(X.shape[1]), [0.0]])

    def obj(p):
        v = clmm_nll(p, Yc, Xp, Mk, K)
        if ridge:
            v += ridge * np.sum(p[ncut:ncut + X.shape[1]] ** 2)
        return v

    best = None
    for meth in ("Nelder-Mead", "Powell", "Nelder-Mead"):
        r = optimize.minimize(obj, p0 if best is None else best.x, method=meth,
                              options={"maxiter": 80000, "maxfev": 160000,
                                       "xatol": 1e-9, "fatol": 1e-9}
                              if meth == "Nelder-Mead"
                              else {"maxiter": 80000, "maxfev": 160000})
        if best is None or r.fun < best.fun:
            best = r
    par = best.x
    cov = numeric_cov(obj, par)
    sl = slice(ncut, ncut + X.shape[1])
    return {"par": par, "beta": par[sl], "cov": cov[sl, sl], "nll": best.fun,
            "sd_re": float(np.exp(par[-1])), "converged": bool(best.success),
            "n_obs": len(ycat), "n_subj": len(np.unique(subj)), "K": K,
            "separated": bool(np.any(np.abs(par[sl]) > 8))}


# ===================================================== utilities
def numeric_cov(f, par, h=1e-4):
    n = len(par)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n); ei[i] = h
            ej = np.zeros(n); ej[j] = h
            H[i, j] = H[j, i] = (f(par + ei + ej) - f(par + ei - ej)
                                 - f(par - ei + ej) + f(par - ei - ej)) / (4 * h * h)
    return np.linalg.pinv(H)


def wald(beta, cov, c):
    est = float(c @ beta)
    se = float(np.sqrt(max(c @ cov @ c, 0)))
    if se <= 0 or not np.isfinite(se):
        return dict(estimate=est, se=np.nan, z=np.nan, p=np.nan,
                    ci_lo=np.nan, ci_hi=np.nan)
    z = est / se
    return dict(estimate=est, se=se, z=z, p=float(2 * stats.norm.sf(abs(z))),
                ci_lo=est - 1.96 * se, ci_hi=est + 1.96 * se)


def wald_omnibus(beta, cov, L):
    d = L @ beta
    V = L @ cov @ L.T
    try:
        W = float(d @ np.linalg.solve(V, d))
    except np.linalg.LinAlgError:
        return dict(chi2=np.nan, df=L.shape[0], p=np.nan)
    return dict(chi2=W, df=L.shape[0],
                p=float(stats.chi2.sf(W, L.shape[0])))
