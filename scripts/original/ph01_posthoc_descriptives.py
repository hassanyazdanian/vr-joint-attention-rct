"""
TASK 1 - within-arm pairwise comparisons under the verbal-adjusted model.
TASK 2 - ITT observed descriptives on all available observations.

Qualification for post-hoc testing is RE-DERIVED under the adjusted model:
an outcome qualifies if it has a significant omnibus time effect OR a
significant omnibus group x time interaction. The list is NOT inherited from
output/07_interaction_inventory.csv, which was produced under the older
unadjusted model.

Writes:
    posthoc_verbal_adjusted/posthoc_pairwise_adjusted.csv
    posthoc_verbal_adjusted/itt_descriptives_adjusted.csv
    posthoc_verbal_adjusted/qualifying_outcomes.csv
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
import ph00_model as M

HERE = M.HERE
ROOT = M.ROOT

sha = M.verify_input()
print(f"Input verified: VR_JA_DATA_reviewed_20260821.xlsx  sha256 {sha[:16]}...")
print(f"Outcomes: {len(M.OUTCOMES)}  |  excluded from inference: "
      f"{', '.join(M.EXCLUDED_FROM_INFERENCE)}")

LONG = M.load_long()
LONG.to_csv(M.HERE / "derived" / "ph_long.csv", index=False)

# ================================================== fit all 16 adjusted models
print()
print("=" * 78)
print("FITTING THE VERBAL-ADJUSTED MODEL FOR ALL 16 OUTCOMES")
print("=" * 78)
FITS = {}
for v in M.OUTCOMES:
    FITS[v] = M.fit(LONG, v)
    print(f"  {v:11s} n={FITS[v]['n_subj']:2d}  obs={FITS[v]['N_obs']:3d}  "
          f"-2RLL={FITS[v]['m2RLL']:10.4f}  converged={FITS[v]['converged']}")

# ---- equivalence check against the archived adjusted analysis
ARCH = pd.read_csv(ROOT / "analysis_verbal_adjusted" / "contrast_results.csv")
diffs = []
for v in M.OUTCOMES:
    f = FITS[v]
    for lab, c, col in (
            ("Group difference at T2", M.cvec({1: 1, 4: 1}), "adj_estimate"),
            ("Group difference at T3", M.cvec({1: 1, 5: 1}), "adj_estimate"),
            ("Treatment difference in change T1->T2", M.cvec({4: 1}), "adj_estimate"),
            ("Treatment difference in change T1->T3", M.cvec({5: 1}), "adj_estimate")):
        a = ARCH[(ARCH.outcome == v) & (ARCH.contrast == lab)]
        if len(a):
            diffs.append(abs(float(c @ f["beta"]) - float(a[col].iloc[0])))
maxd = max(diffs)
print()
print(f"EQUIVALENCE CHECK vs analysis_verbal_adjusted/contrast_results.csv: "
      f"max |difference| over {len(diffs)} estimates = {maxd:.3e}")
assert maxd < 1e-8, "extracted machinery does not reproduce the archived model"
print("  -> the extracted machinery reproduces the archived adjusted model exactly.")

# ================================================== TASK 1a: qualification
print()
print("=" * 78)
print("TASK 1a - QUALIFICATION, RE-DERIVED UNDER THE ADJUSTED MODEL")
print("=" * 78)
OLD = pd.read_csv(ROOT / "output" / "07_interaction_inventory.csv")
qrows = []
for v in M.OUTCOMES:
    f = FITS[v]
    tm = M.omnibus(f, M.L_TIME)
    it = M.omnibus(f, M.L_INTER)
    qual = (tm["p"] < 0.05) or (it["p"] < 0.05)
    o = OLD[OLD.outcome == v]
    old_qual = bool(o["posthoc_warranted"].iloc[0]) if len(o) else np.nan
    qrows.append({
        "outcome": v, "family": M.FAM_OF[v],
        "adj_time_F": tm["F"], "adj_time_df2": tm["df2"], "adj_time_p": tm["p"],
        "adj_time_sig": tm["p"] < 0.05,
        "adj_interaction_F": it["F"], "adj_interaction_df2": it["df2"],
        "adj_interaction_p": it["p"], "adj_interaction_sig": it["p"] < 0.05,
        "qualifies_adjusted": qual,
        "qualified_in_07_inventory_unadjusted": old_qual,
        "qualification_status_changed":
            ("n/a - not in old file" if pd.isna(old_qual)
             else ("YES" if bool(qual) != bool(old_qual) else "no")),
    })
Q = pd.DataFrame(qrows)
Q.to_csv(HERE / "qualifying_outcomes.csv", index=False)
print(Q[["outcome", "adj_time_p", "adj_time_sig", "adj_interaction_p",
         "adj_interaction_sig", "qualifies_adjusted",
         "qualified_in_07_inventory_unadjusted",
         "qualification_status_changed"]].round(4).to_string(index=False))
QUAL = [r["outcome"] for r in qrows if r["qualifies_adjusted"]]
print()
print(f"Qualifying under the ADJUSTED model: {len(QUAL)}/16 -> {QUAL}")
chg = Q[Q.qualification_status_changed == "YES"]
print(f"Outcomes whose qualification status differs from "
      f"output/07_interaction_inventory.csv: {len(chg)}")
if len(chg):
    print(chg[["outcome", "adj_time_p", "adj_interaction_p",
               "qualifies_adjusted",
               "qualified_in_07_inventory_unadjusted"]].round(4).to_string(index=False))

# ================================================== TASK 1b: pairwise
print()
print("=" * 78)
print("TASK 1b - WITHIN-ARM PAIRWISE (Bonferroni over 3 comparisons per arm)")
print("=" * 78)
prows = []
for v in QUAL:
    f = FITS[v]
    for (arm, lab), c in M.WITHIN_ARM.items():
        s = M.contrast_stats(f, c, bonferroni_m=3)
        dirn = M.DIRECTION[v]
        prows.append({
            "outcome": v, "family": M.FAM_OF[v], "arm": arm, "comparison": lab,
            **s,
            "sig_unadjusted": s["p_unadjusted"] < 0.05,
            "sig_bonferroni": s["p_bonferroni"] < 0.05,
            "scoring": "higher = better" if dirn > 0 else "lower = better",
            "improvement": bool(s["estimate"] * dirn > 0),
            "n_subj": f["n_subj"], "N_obs": f["N_obs"],
            "model": "Outcome ~ Group*Time + VerbalStatus + (1|participant)",
        })
P = pd.DataFrame(prows)
P.to_csv(HERE / "posthoc_pairwise_adjusted.csv", index=False)
print(f"{P.outcome.nunique()} outcomes x 6 comparisons = {len(P)} rows")
print()
print(P[["outcome", "arm", "comparison", "estimate", "se", "df_satterthwaite",
         "t", "p_unadjusted", "p_bonferroni", "sig_bonferroni"]]
      .round(4).to_string(index=False))
print()
print(f"Significant after Bonferroni — Intervention "
      f"{int((P.sig_bonferroni & (P.arm=='Intervention')).sum())}/"
      f"{int((P.arm=='Intervention').sum())}, Control "
      f"{int((P.sig_bonferroni & (P.arm=='Control')).sum())}/"
      f"{int((P.arm=='Control').sum())}")

# ================================================== TASK 2: descriptives
print()
print("=" * 78)
print("TASK 2 - ITT OBSERVED DESCRIPTIVES (all available observations)")
print("=" * 78)
drows = []
for v in M.OUTCOMES:
    for t in M.TPS:
        for code, arm in ((1, "Intervention"), (0, "Control")):
            s = LONG[(LONG.time == t) & (LONG.group == code)][v].dropna()
            mean = float(s.mean()) if len(s) else np.nan
            sd = float(s.std(ddof=1)) if len(s) > 1 else np.nan
            drows.append({
                "outcome": v, "family": M.FAM_OF[v], "timepoint": t, "arm": arm,
                "n": int(len(s)), "mean": mean, "sd": sd,
                "mean_2dp": None if np.isnan(mean) else round(mean, 2),
                "sd_2dp": None if np.isnan(sd) else round(sd, 2),
                "display": ("" if np.isnan(mean) else
                            f"{mean:.2f} ± {sd:.2f}" if not np.isnan(sd)
                            else f"{mean:.2f}"),
                "basis": "all available observations (NOT listwise)",
                "behavioural_definition": ("mean of raters M and Z"
                                           if v.startswith("BS_") else "n/a"),
            })
D = pd.DataFrame(drows)
D.to_csv(HERE / "itt_descriptives_adjusted.csv", index=False)
print(f"{len(D)} rows = 16 outcomes x 3 timepoints x 2 arms")
print()
piv = D.pivot_table(index=["outcome", "timepoint"], columns="arm",
                    values="display", aggfunc="first")
npv = D.pivot_table(index=["outcome", "timepoint"], columns="arm",
                    values="n", aggfunc="first")
show = piv.join(npv, rsuffix="_n")
print(show.to_string())
print()
print("DONE.")
