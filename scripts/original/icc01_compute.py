"""
Inter-rater reliability for the behavioural sample — source for manuscript S2 Table.

MODEL / TYPE / UNIT, stated explicitly:
    Model : two-way RANDOM effects (both participants and raters treated as random)
    Type  : ABSOLUTE AGREEMENT
    Unit  : AVERAGE MEASURES, k = 2   -> ICC(A,k), the headline S2 quantity
            SINGLE MEASURES           -> ICC(A,1), reported alongside for reference
    In McGraw & Wong (1996) notation ICC(A,1) and ICC(A,k); in Shrout & Fleiss (1979)
    notation ICC(2,1) and ICC(2,k).

Cells: 5 behavioural domains x 3 timepoints. Each cell uses the participants with
BOTH rater M and rater Z present at that timepoint (pairwise-complete). Cell n is
reported explicitly for every cell.

ICC(C,1) and ICC(C,k) — two-way MIXED, CONSISTENCY type — are also computed,
because the archived output/09_MASTER_REPORT.md section E2 table and the original
published value of 0.937 are consistency-type quantities. Reporting all four makes
the reconciliation arithmetic checkable.

Point estimates are computed twice: from the ANOVA mean squares directly
(McGraw & Wong formulas, implemented here) and with pingouin. Both must agree.

Writes icc_verbal_adjusted/icc_results.csv
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import hashlib
import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ROOT = HERE.parent
RAW = ROOT / "data_raw" / "VR_JA_DATA_reviewed_20260821.xlsx"
EXPECTED_SHA = "844f2d0e10f35908c1bb43ee8d8f061f394e80907c9fca46f27eddcdf0d5eb31"

h = hashlib.sha256(RAW.read_bytes()).hexdigest()
if h != EXPECTED_SHA:
    raise SystemExit(f"ABORT: workbook SHA-256 mismatch\n  expected {EXPECTED_SHA}"
                     f"\n  found    {h}")
print(f"Input verified: {RAW.name}  sha256 {h[:16]}...")

TPS = ["T1", "T2", "T3"]
SHEETS = dict(zip(TPS, ["Pre(T1)", "Post(T2)", "Flu(T3)"]))
DOMAINS = [("RJA", "Responding to Joint Attention"),
           ("IJA", "Initiating Joint Attention"),
           ("EC", "Eye Contact"),
           ("FL", "Follow Behaviour"),
           ("T", "Total Behavioural Score")]


def rd(sheet):
    d = pd.read_excel(RAW, sheet_name=sheet, keep_default_na=False, na_values=["NA"])
    return d[d["id"].notna()].assign(id=lambda x: x["id"].astype(int))


S = {t: rd(s) for t, s in SHEETS.items()}


# ------------------------------------------------ ICC from the ANOVA mean squares
def icc_from_anova(Y):
    """Y is n x k (targets x raters), complete. Returns the four ICC forms and
    the ANOVA quantities, following McGraw & Wong (1996) Table 4."""
    n, k = Y.shape
    grand = Y.mean()
    row_m = Y.mean(axis=1)
    col_m = Y.mean(axis=0)
    SSR = k * ((row_m - grand) ** 2).sum()          # between targets
    SSC = n * ((col_m - grand) ** 2).sum()          # between raters
    SST = ((Y - grand) ** 2).sum()
    SSE = SST - SSR - SSC
    MSR = SSR / (n - 1)
    MSC = SSC / (k - 1)
    MSE = SSE / ((n - 1) * (k - 1))
    a1 = (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)   # ICC(A,1)
    ak = (MSR - MSE) / (MSR + (MSC - MSE) / n)                       # ICC(A,k)
    c1 = (MSR - MSE) / (MSR + (k - 1) * MSE)                         # ICC(C,1)
    ck = (MSR - MSE) / MSR                                           # ICC(C,k)
    return dict(n=n, k=k, MSR=MSR, MSC=MSC, MSE=MSE,
                F_targets=MSR / MSE, df1=n - 1, df2=(n - 1) * (k - 1),
                ICC_A1=a1, ICC_Ak=ak, ICC_C1=c1, ICC_Ck=ck)


def icc_cis(A, alpha=0.05):
    """Exact-form CIs, McGraw & Wong (1996) Table 7. pingouin rounds its CI95 to
    two decimals, which is too coarse for a manuscript table, so the intervals
    are computed here at full precision and checked against pingouin's rounding."""
    n, k = A["n"], A["k"]
    MSR, MSC, MSE = A["MSR"], A["MSC"], A["MSE"]
    q = 1 - alpha / 2
    out = {}

    # --- consistency type: simple F-based
    Fo = MSR / MSE
    FL = Fo / stats.f.ppf(q, n - 1, (n - 1) * (k - 1))
    FU = Fo * stats.f.ppf(q, (n - 1) * (k - 1), n - 1)
    out["C1"] = ((FL - 1) / (FL + k - 1), (FU - 1) / (FU + k - 1))
    out["Ck"] = (1 - 1 / FL, 1 - 1 / FU)

    # --- absolute agreement, single measures
    r = A["ICC_A1"]
    a = k * r / (n * (1 - r))
    b = 1 + k * r * (n - 1) / (n * (1 - r))
    v = ((a * MSC + b * MSE) ** 2
         / ((a * MSC) ** 2 / (k - 1) + (b * MSE) ** 2 / ((n - 1) * (k - 1))))
    Fl = stats.f.ppf(q, n - 1, v)
    Fu = stats.f.ppf(q, v, n - 1)
    lo = n * (MSR - Fl * MSE) / (Fl * (k * MSC + (k * n - k - n) * MSE) + n * MSR)
    hi = n * (Fu * MSR - MSE) / (k * MSC + (k * n - k - n) * MSE + n * Fu * MSR)
    out["A1"] = (lo, hi)

    # --- absolute agreement, average measures
    rk = A["ICC_Ak"]
    a = rk / (n * (1 - rk))
    b = 1 + rk * (n - 1) / (n * (1 - rk))
    v = ((a * MSC + b * MSE) ** 2
         / ((a * MSC) ** 2 / (k - 1) + (b * MSE) ** 2 / ((n - 1) * (k - 1))))
    Fl = stats.f.ppf(q, n - 1, v)
    Fu = stats.f.ppf(q, v, n - 1)
    lo = n * (MSR - Fl * MSE) / (Fl * (MSC - MSE) + n * MSR)
    hi = n * (Fu * MSR - MSE) / (MSC - MSE + n * Fu * MSR)
    out["Ak"] = (lo, hi)
    return out


rows = []
print()
print("=" * 78)
print("ICC BY DOMAIN AND TIMEPOINT")
print("  Model: two-way random | Type: absolute agreement | Unit: average (k=2)")
print("=" * 78)
for dom, label in DOMAINS:
    for t in TPS:
        d = S[t]
        m, z = d[f"BS_{dom}_M"], d[f"BS_{dom}_Z"]
        both = m.notna() & z.notna()
        Y = np.column_stack([m[both].values, z[both].values]).astype(float)
        A = icc_from_anova(Y)

        # independent computation with pingouin (also supplies the McGraw & Wong CIs)
        ids = d.loc[both, "id"].values
        long = pd.DataFrame({
            "target": np.concatenate([ids, ids]),
            "rater": ["M"] * len(ids) + ["Z"] * len(ids),
            "score": np.concatenate([m[both].values, z[both].values]).astype(float)})
        ic = pg.intraclass_corr(data=long, targets="target", raters="rater",
                                ratings="score", nan_policy="omit").set_index("Type")

        CI = icc_cis(A)
        ak_lo, ak_hi = CI["Ak"]
        a1_lo, a1_hi = CI["A1"]
        c1_lo, c1_hi = CI["C1"]
        ck_lo, ck_hi = CI["Ck"]

        # sanity: our full-precision CIs must agree with pingouin's 2-dp rounding
        ci_dev = 0.0
        for tp, (lo_, hi_) in (("ICC(A,k)", CI["Ak"]), ("ICC(A,1)", CI["A1"]),
                               ("ICC(C,1)", CI["C1"]), ("ICC(C,k)", CI["Ck"])):
            plo, phi = ic.loc[tp, "CI95"]
            ci_dev = max(ci_dev, abs(round(lo_, 2) - float(plo)),
                         abs(round(hi_, 2) - float(phi)))

        # agreement between the two independent computations
        dev = max(abs(A["ICC_A1"] - ic.loc["ICC(A,1)", "ICC"]),
                  abs(A["ICC_Ak"] - ic.loc["ICC(A,k)", "ICC"]),
                  abs(A["ICC_C1"] - ic.loc["ICC(C,1)", "ICC"]),
                  abs(A["ICC_Ck"] - ic.loc["ICC(C,k)", "ICC"]))

        rows.append({
            "domain": f"BS_{dom}", "domain_label": label, "timepoint": t,
            "n_rated_cases": A["n"], "k_raters": A["k"],
            "model": "two-way random", "type_primary": "absolute agreement",
            "unit_primary": "average measures (k=2)",
            "ICC_A_k": A["ICC_Ak"], "ICC_A_k_lo95": ak_lo, "ICC_A_k_hi95": ak_hi,
            "ICC_A_1": A["ICC_A1"], "ICC_A_1_lo95": a1_lo, "ICC_A_1_hi95": a1_hi,
            "ICC_C_1": A["ICC_C1"], "ICC_C_1_lo95": c1_lo, "ICC_C_1_hi95": c1_hi,
            "ICC_C_k": A["ICC_Ck"], "ICC_C_k_lo95": ck_lo, "ICC_C_k_hi95": ck_hi,
            "MSR": A["MSR"], "MSC": A["MSC"], "MSE": A["MSE"],
            "F_targets": A["F_targets"], "df1": A["df1"], "df2": A["df2"],
            "p_F": float(stats.f.sf(A["F_targets"], A["df1"], A["df2"])),
            "max_dev_anova_vs_pingouin": dev,
            "max_ci_dev_vs_pingouin_2dp": ci_dev,
            "ICC_A_k_3dp": round(A["ICC_Ak"], 3),
            "display_A_k": f"{A['ICC_Ak']:.3f} ({ak_lo:.3f}–{ak_hi:.3f})",
        })

R = pd.DataFrame(rows)
R.to_csv(HERE / "icc_results.csv", index=False)

print(R[["domain", "timepoint", "n_rated_cases", "ICC_A_k", "ICC_A_k_lo95",
         "ICC_A_k_hi95", "ICC_A_1", "ICC_C_1"]].round(4).to_string(index=False))
print()
print(f"Max deviation, ANOVA-formula vs pingouin point estimates : "
      f"{R.max_dev_anova_vs_pingouin.max():.3e}")
print(f"Max deviation, own CIs vs pingouin CIs at pingouin's 2 dp: "
      f"{R.max_ci_dev_vs_pingouin_2dp.max():.3e}")
assert R.max_dev_anova_vs_pingouin.max() < 1e-10, "point estimates disagree"
assert R.max_ci_dev_vs_pingouin_2dp.max() < 5e-3, "CIs disagree"
print("  -> the two independent computations agree on estimates and CIs.")

# ------------------------------------------------ pooled across timepoints
print()
print("=" * 78)
print("POOLED ACROSS TIMEPOINTS (each participant-timepoint = one target)")
print("=" * 78)
prows = []
for dom, label in DOMAINS:
    tg, rt, sc = [], [], []
    for t in TPS:
        d = S[t]
        m, z = d[f"BS_{dom}_M"], d[f"BS_{dom}_Z"]
        both = m.notna() & z.notna()
        ids = d.loc[both, "id"].values
        keys = [f"{t}_{i}" for i in ids]
        tg += keys + keys
        rt += ["M"] * len(keys) + ["Z"] * len(keys)
        sc += list(m[both].values) + list(z[both].values)
    long = pd.DataFrame({"target": tg, "rater": rt, "score": np.array(sc, float)})
    ic = pg.intraclass_corr(data=long, targets="target", raters="rater",
                            ratings="score").set_index("Type")
    prows.append({"domain": f"BS_{dom}", "n_target_occasions": long.target.nunique(),
                  "ICC_A_k_pooled": float(ic.loc["ICC(A,k)", "ICC"]),
                  "ICC_A_1_pooled": float(ic.loc["ICC(A,1)", "ICC"]),
                  "ICC_C_1_pooled": float(ic.loc["ICC(C,1)", "ICC"])})
P = pd.DataFrame(prows)
P.to_csv(HERE / "derived" / "icc_pooled.csv", index=False)
print(P.round(4).to_string(index=False))

# ------------------------------------------------ reconcile with E2
print()
print("=" * 78)
print("RECONCILIATION WITH output/09_MASTER_REPORT.md SECTION E2")
print("=" * 78)
E2 = {"BS_RJA": (0.898, 0.738, 0.845, 0.907),
      "BS_IJA": (0.943, 0.919, 0.911, 0.961),
      "BS_EC":  (0.617, 0.669, 0.610, 0.763),
      "BS_FL":  (0.929, 0.662, 0.777, 0.897),
      "BS_T":   (0.937, 0.733, 0.772, 0.907)}
rec = []
for dom, vals in E2.items():
    for i, t in enumerate(TPS):
        r = R[(R.domain == dom) & (R.timepoint == t)].iloc[0]
        cand = {"ICC_A_1": r.ICC_A_1, "ICC_A_k": r.ICC_A_k,
                "ICC_C_1": r.ICC_C_1, "ICC_C_k": r.ICC_C_k}
        best = min(cand, key=lambda kk: abs(cand[kk] - vals[i]))
        rec.append({"source": "E2 per-timepoint", "domain": dom, "timepoint": t,
                    "E2_value": vals[i], "best_match_quantity": best,
                    "best_match_value": cand[best],
                    "abs_diff": abs(cand[best] - vals[i]),
                    "matches_to_3dp": round(cand[best], 3) == vals[i]})
    p = P[P.domain == dom].iloc[0]
    candp = {"ICC_A_k_pooled": p.ICC_A_k_pooled, "ICC_A_1_pooled": p.ICC_A_1_pooled,
             "ICC_C_1_pooled": p.ICC_C_1_pooled}
    bestp = min(candp, key=lambda kk: abs(candp[kk] - vals[3]))
    rec.append({"source": "E2 pooled column", "domain": dom, "timepoint": "pooled",
                "E2_value": vals[3], "best_match_quantity": bestp,
                "best_match_value": candp[bestp],
                "abs_diff": abs(candp[bestp] - vals[3]),
                "matches_to_3dp": round(candp[bestp], 3) == vals[3]})
REC = pd.DataFrame(rec)
REC.to_csv(HERE / "derived" / "e2_reconciliation.csv", index=False)
print(REC.round(4).to_string(index=False))
print()
per_tp = REC[REC.source == "E2 per-timepoint"]
print("E2 per-timepoint column identified as:",
      per_tp.best_match_quantity.value_counts().to_dict())
print("E2 pooled column identified as:",
      REC[REC.source == "E2 pooled column"].best_match_quantity.value_counts().to_dict())

# ------------------------------------------------ draft S2 verification
print()
print("=" * 78)
print("DRAFT S2 VALUES — VERIFICATION (to three decimals)")
print("=" * 78)
DRAFT = {"BS_RJA": (0.946, 0.844, 0.911), "BS_IJA": (0.970, 0.958, 0.954),
         "BS_EC": (0.767, 0.799, 0.718), "BS_FL": (0.958, 0.797, 0.872),
         "BS_T": (0.968, 0.847, 0.858)}
drows = []
for dom, vals in DRAFT.items():
    for i, t in enumerate(TPS):
        r = R[(R.domain == dom) & (R.timepoint == t)].iloc[0]
        ok = round(r.ICC_A_k, 3) == vals[i]
        drows.append({"domain": dom, "timepoint": t, "draft_value": vals[i],
                      "computed_ICC_A_k": r.ICC_A_k,
                      "computed_3dp": round(r.ICC_A_k, 3),
                      "reproduces": "YES" if ok else "NO",
                      "n_rated_cases": int(r.n_rated_cases),
                      "ci95": f"({r.ICC_A_k_lo95:.3f}, {r.ICC_A_k_hi95:.3f})"})
        print(f"  {dom:7s} {t}  draft {vals[i]:.3f}  computed {r.ICC_A_k:.6f} "
              f"-> {round(r.ICC_A_k,3):.3f}  {'REPRODUCES' if ok else '*** NO ***'}"
              f"   n={int(r.n_rated_cases)}")
D = pd.DataFrame(drows)
D.to_csv(HERE / "derived" / "draft_s2_verification.csv", index=False)
print()
print(f"Draft S2 values reproducing: {int((D.reproduces=='YES').sum())}/{len(D)}")

# ------------------------------------------------ the 0.937 arithmetic
print()
print("=" * 78)
print("THE 0.937 -> AVERAGE-MEASURES ARITHMETIC (BS_T at T1)")
print("=" * 78)
r = R[(R.domain == "BS_T") & (R.timepoint == "T1")].iloc[0]
c1, ck, a1, ak = r.ICC_C_1, r.ICC_C_k, r.ICC_A_1, r.ICC_A_k
sb = 2 * c1 / (1 + c1)
print(f"  published ICC (consistency, single measures)  ICC(C,1) = {c1:.6f}  -> {c1:.3f}")
print(f"  Spearman-Brown  k*r/(1+(k-1)r) with k=2       = 2({c1:.6f})/(1+{c1:.6f})")
print(f"                                                = {sb:.6f}  -> {sb:.3f}")
print(f"  computed ICC(C,k) directly                    = {ck:.6f}  -> {ck:.3f}")
print(f"  |Spearman-Brown - ICC(C,k)|                   = {abs(sb-ck):.3e}")
print(f"  ICC(A,1) absolute agreement, single           = {a1:.6f}  -> {a1:.3f}")
print(f"  ICC(A,k) absolute agreement, average  [S2]    = {ak:.6f}  -> {ak:.3f}")
print(f"  ICC(A,k) - ICC(C,k)                           = {ak-ck:+.6f}")
pd.DataFrame([{"quantity": "ICC(C,1) published", "value": c1},
              {"quantity": "Spearman-Brown of ICC(C,1)", "value": sb},
              {"quantity": "ICC(C,k) computed", "value": ck},
              {"quantity": "ICC(A,1)", "value": a1},
              {"quantity": "ICC(A,k) = S2 value", "value": ak},
              {"quantity": "MSC (between-rater MS)", "value": r.MSC},
              {"quantity": "MSE", "value": r.MSE},
              {"quantity": "n", "value": r.n_rated_cases}]).to_csv(
    HERE / "derived" / "icc_0937_arithmetic.csv", index=False)
print()
print("DONE.")
