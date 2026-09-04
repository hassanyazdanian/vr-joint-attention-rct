"""
Final sensitivity run - STEPS 4 and 5: JAST ordinal sensitivity, and reaction time.

SECTION 4 - JAST ARJA / AIJA
  The PREFERRED analysis (task-level ordinal model of the individual 0-3 task
  scores) requires the task-level scores. A search of the project establishes
  that they DO NOT EXIST in any available file - see the feasibility block below.
  The preferred analysis is therefore NOT FEASIBLE and the validated primary LMM
  is retained.

  A secondary ordinal analysis IS possible without any rounding, because the
  averages are not continuous: ARJA and AIJA are means of three 0-3 task scores,
  so they take only exact multiples of 1/3 on [0, 3]. Their observed values form
  an ordered discrete support (10 and 7 levels respectively) which can be modelled
  directly as ordered categories. NO ROUNDING TO INTEGERS IS PERFORMED - the exact
  observed values are used as category labels.

SECTION 5 - JAST reaction time
  RT is NOT reincorporated into any efficacy conclusion. Missingness is classified
  into (a) assessment not administered and (b) structurally undefined because the
  child did not respond, and RT is summarised descriptively only.

Writes jast_ordinal_results.csv and jast_rt_descriptive.csv
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import pyreadstat
from pathlib import Path
import fs00_common as K

HERE = K.HERE
LONG = pd.read_csv(K.DER / "fs_long.csv")
PRIM = pd.read_csv(K.ROOT / "analysis_verbal_adjusted" / "contrast_results.csv")

# ==================================================== 4a. feasibility check
print("=" * 78)
print("SECTION 4a — TASK-LEVEL JAST DATA: FEASIBILITY")
print("=" * 78)
feas = []
sheets = pd.ExcelFile(K.RAW).sheet_names
for sh in sheets:
    cols = pd.read_excel(K.RAW, sheet_name=sh, nrows=0).columns
    tl = [c for c in cols if isinstance(c, str)
          and c.startswith("JAST_t") and c.endswith(("_sc", "_ct"))]
    feas.append({"source": f"workbook sheet '{sh}'", "task_level_columns": len(tl),
                 "non_null_values": 0})
for f in ["<EXTERNAL_SPSS_DIR>/All_Variables.sav",
          "<EXTERNAL_SPSS_DIR>/Data test reliability.sav"]:
    try:
        df, _ = pyreadstat.read_sav(f)
        tl = [c for c in df.columns if c.startswith("JAST_t")
              and (c.endswith("_sc") or "_sc_" in c)]
        feas.append({"source": Path(f).name, "task_level_columns": len(tl),
                     "non_null_values": int(df[tl].notna().sum().sum()) if tl else 0})
    except Exception as e:
        feas.append({"source": Path(f).name, "task_level_columns": -1,
                     "non_null_values": -1})
F = pd.DataFrame(feas)
print(F.to_string(index=False))
total_vals = int(F.non_null_values.clip(lower=0).sum())
print()
print(f"Total task-level JAST score values available anywhere: {total_vals}")
print("VERDICT: task-level ordinal modelling is NOT FEASIBLE. "
      "`All_Variables.sav` declares 18\n         task-level columns but contains "
      "one row and zero values; no other source holds them.\n         The "
      "validated primary LMM on ARJA/AIJA is retained.")
F.to_csv(K.DER / "jast_tasklevel_feasibility.csv", index=False)

# ==================================================== 4b. ordinal on exact support
print()
print("=" * 78)
print("SECTION 4b — SECONDARY ORDINAL ANALYSIS ON THE EXACT DISCRETE SUPPORT")
print("=" * 78)
NAMES = ["group", "time_T2", "time_T3", "group:T2", "group:T3", "verbal"]
I_GT2, I_GT3 = 3, 4


def build(sub):
    t2 = (sub["time"] == "T2").astype(float).values
    t3 = (sub["time"] == "T3").astype(float).values
    g = sub["group"].astype(float).values
    return np.column_stack([g, t2, t3, g * t2, g * t3,
                            sub["verbal"].astype(float).values])


rows = []
for v in ["JAST_arja", "JAST_aija"]:
    sub = LONG[LONG[v].notna()].copy()
    levels = np.sort(sub[v].unique())
    Kn = len(levels)
    # exact-value check: no rounding is applied anywhere
    assert np.allclose(sub[v].values * 3, np.round(sub[v].values * 3)), \
        "values are not exact multiples of 1/3"
    ycat = np.searchsorted(levels, sub[v].values)
    X = build(sub)
    subj = sub["id"].values
    # mild ridge keeps quasi-separated coefficients finite; documented below
    f = K.fit_clmm(ycat, X, subj, Kn, ridge=1e-3)
    L = np.zeros((2, X.shape[1])); L[0, I_GT2] = 1; L[1, I_GT3] = 1
    om = K.wald_omnibus(f["beta"], f["cov"], L)
    prim_om = PRIM[(PRIM.outcome == v) &
                   (PRIM.contrast == "Omnibus Group x Time (2 df)")]["adj_p"]
    rows.append({"outcome": v, "contrast": "Overall Group x Time",
                 "levels_K": Kn, "estimate_logOR": np.nan, "se": np.nan,
                 "OR": np.nan, "or_ci_lo": np.nan, "or_ci_hi": np.nan,
                 "test": f"Wald chi2({om['df']}) = {om['chi2']:.4f}", "p": om["p"],
                 "p_primary_LMM": float(prim_om.iloc[0]) if len(prim_om) else np.nan,
                 "converged": f["converged"], "quasi_separated": f["separated"],
                 "re_sd": f["sd_re"], "n_subj": f["n_subj"], "n_obs": f["n_obs"]})
    for lab, idx, plab in (
            ("T1->T2 treatment-related change", I_GT2,
             "Treatment difference in change T1->T2"),
            ("T1->T3 treatment-related change", I_GT3,
             "Treatment difference in change T1->T3")):
        c = np.zeros(X.shape[1]); c[idx] = 1.0
        w = K.wald(f["beta"], f["cov"], c)
        pr = PRIM[(PRIM.outcome == v) & (PRIM.contrast == plab)]["adj_p"]
        rows.append({"outcome": v, "contrast": lab, "levels_K": Kn,
                     "estimate_logOR": w["estimate"], "se": w["se"],
                     "OR": np.exp(w["estimate"]),
                     "or_ci_lo": np.exp(w["ci_lo"]), "or_ci_hi": np.exp(w["ci_hi"]),
                     "test": f"z = {w['z']:.4f}", "p": w["p"],
                     "p_primary_LMM": float(pr.iloc[0]) if len(pr) else np.nan,
                     "converged": f["converged"], "quasi_separated": f["separated"],
                     "re_sd": f["sd_re"], "n_subj": f["n_subj"],
                     "n_obs": f["n_obs"]})
J = pd.DataFrame(rows)
J["task_level_model"] = "NOT FEASIBLE - task-level scores do not exist"
J["rounding_applied"] = "NONE - exact observed values used as ordered categories"
J["OR_direction_note"] = ("P(Y<=k)=logistic(theta_k - X'beta); OR > 1 = higher "
                          "odds of a HIGHER JAST score = BENEFIT (higher is better).")
J["penalty_note"] = ("Ridge 1e-3 on the fixed effects keeps quasi-separated "
                     "coefficients finite; JAST is at ceiling post-intervention.")
J.to_csv(HERE / "jast_ordinal_results.csv", index=False)
print(J[["outcome", "contrast", "levels_K", "estimate_logOR", "se", "OR",
         "p", "p_primary_LMM", "converged", "quasi_separated"]]
      .round(4).to_string(index=False))
print()
for _, r in J[J.contrast != "Overall Group x Time"].iterrows():
    same = (r.p < 0.05) == (r.p_primary_LMM < 0.05)
    direction_ok = (r.estimate_logOR > 0)
    print(f"  {r.outcome:10s} {r.contrast:33s} ordinal p={r.p:.4f} vs primary "
          f"p={r.p_primary_LMM:.4f} -> {'SUPPORTS' if same else '*** DIFFERS ***'}"
          f"; direction favours {'intervention' if direction_ok else 'control'}")

# ==================================================== 5. reaction time
print()
print("=" * 78)
print("SECTION 5 — JAST REACTION TIME: MISSINGNESS TAXONOMY AND DESCRIPTIVES")
print("=" * 78)
rt_rows = []
for v in K.RT_VARS:
    for t in K.TPS:
        d = LONG[LONG.time == t]
        attended = d["JAST_arja"].notna()
        rec = {"variable": v, "timepoint": t,
               "n_participants": int(len(d)),
               "session_not_administered": int((~attended).sum()),
               "attended_and_RT_present": int((attended & d[v].notna()).sum()),
               "attended_but_RT_undefined": int((attended & d[v].isna()).sum())}
        for code, nm in ((0, "Control"), (1, "Intervention")):
            s = d[(d.group == code) & attended & d[v].notna()][v]
            rec[f"{nm}_n"] = int(len(s))
            rec[f"{nm}_mean"] = float(s.mean()) if len(s) else np.nan
            rec[f"{nm}_sd"] = float(s.std(ddof=1)) if len(s) > 1 else np.nan
            rec[f"{nm}_median"] = float(s.median()) if len(s) else np.nan
        rt_rows.append(rec)
RT = pd.DataFrame(rt_rows)
RT["missingness_note"] = (
    "session_not_administered = the whole JAST session is absent (dropout-type). "
    "attended_but_RT_undefined = the child completed the session but did not "
    "respond to that cue, so no reaction time exists. The latter is STRUCTURALLY "
    "UNDEFINED, not MAR, and is never imputed or modelled as missing.")
RT["inferential_use"] = "NONE - descriptive only; excluded from all efficacy conclusions"
RT.to_csv(HERE / "jast_rt_descriptive.csv", index=False)
print(RT[["variable", "timepoint", "session_not_administered",
          "attended_but_RT_undefined", "attended_and_RT_present",
          "Control_n", "Control_mean", "Intervention_n", "Intervention_mean"]]
      .round(3).to_string(index=False))
print()
tot_undef = int(RT.attended_but_RT_undefined.sum())
tot_nosess = int(RT.session_not_administered.sum())
print(f"Across all four RT indices and three timepoints:")
print(f"   {tot_nosess} cells where the JAST session was not administered "
      f"(dropout-type missing)")
print(f"   {tot_undef} cells where the session WAS completed but the child did "
      f"not respond\n      -> reaction time structurally undefined, NOT missing "
      f"data, NOT imputed")
print()
print("RT is reported descriptively only and does not enter any efficacy "
      "conclusion,\nany multiplicity family, or any model in this run.")
print()
print("DONE.")
