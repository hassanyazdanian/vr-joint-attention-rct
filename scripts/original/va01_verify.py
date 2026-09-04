"""
Verbal-adjusted sensitivity analysis - STEP 1: pre-flight verification.

Confirms every participant-accounting claim in the analysis brief BEFORE any
model is fitted, and reports any inconsistency. Also confirms the coding of the
verbal-status variable and produces the baseline verbal-status balance table.

Writes:
    analysis_verbal_adjusted/verbal_balance.csv
    analysis_verbal_adjusted/sample_sizes_by_outcome.csv
    analysis_verbal_adjusted/derived/va_long.csv      (analysis dataset)
    analysis_verbal_adjusted/derived/va_long_for_spss.csv
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent          # analysis_verbal_adjusted/
ROOT = HERE.parent                                     
RAW = ROOT / "data_raw" / "VR_JA_DATA_reviewed_20260821.xlsx"
DER = HERE / "derived"
DER.mkdir(parents=True, exist_ok=True)

GCOL = "Group (A=0, B=1)"
TPS = ["T1", "T2", "T3"]
SHEETS = dict(zip(TPS, ["Pre(T1)", "Post(T2)", "Flu(T3)"]))
BS_DOM = ["RJA", "IJA", "EC", "FL", "T"]
WITHDRAWN = ['P24', 'P04']  # original identifiers replaced for publication

# ---- the 16 manuscript outcomes for this sensitivity analysis.
# GARS_tot_6 is DELIBERATELY ABSENT, as are GARS_cs / GARS_ms and the derived
# C-JARS averages. JAST reaction times are handled separately and never enter
# the primary comparison.
FAMILIES = {
    "C-JARS":            ["CJARS_ss", "CJARS_ps", "CJARS_sjas"],
    "Behavioural sample": ["BS_RJA", "BS_IJA", "BS_EC", "BS_FL", "BS_T"],
    "JAST":              ["JAST_arja", "JAST_aija"],
    "GARS-3":            ["GARS_rr", "GARS_si", "GARS_sc", "GARS_er"],
    "ACSF:SC":           ["ACSF_tp", "ACSF_cc"],
}
OUTCOMES = [v for vs in FAMILIES.values() for v in vs]
RT_VARS = [f"JAST_RT_t{i}" for i in (1, 2, 3, 4)]
FORBIDDEN = ["GARS_tot_6", "GARS_cs", "GARS_ms", "CJARS_ass", "CJARS_aps"]


def rd(sheet):
    d = pd.read_excel(RAW, sheet_name=sheet, keep_default_na=False, na_values=["NA"])
    return d[d["id"].notna()].assign(id=lambda x: x["id"].astype(int))


S = {t: rd(s) for t, s in SHEETS.items()}
for t in TPS:
    for dom in BS_DOM:                      # rater MEAN, as in the current pipeline
        S[t][f"BS_{dom}"] = S[t][[f"BS_{dom}_M", f"BS_{dom}_Z"]].mean(axis=1)

pre = S["T1"]
IDS = list(pre["id"])
GRP = pre.set_index("id")[GCOL]
VERB = pre.set_index("id")["demog_Verbal"]

checks = []


def chk(label, ok, detail=""):
    checks.append({"check": label, "result": "PASS" if ok else "FAIL",
                   "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  — {detail}" if detail else ""))
    return ok


print("=" * 78)
print("STEP 1 — PARTICIPANT ACCOUNTING")
print("=" * 78)
n_int = int((pre[GCOL] == 1).sum())
n_ctl = int((pre[GCOL] == 0).sum())
chk("43 participants randomized", len(pre) == 43, f"n = {len(pre)}")
chk("Intervention n = 21", n_int == 21, f"n = {n_int}")
chk("Control n = 22", n_ctl == 22, f"n = {n_ctl}")
chk("group codes are 0/1 only", set(pre[GCOL].unique()) == {0, 1},
    f"values {sorted(pre[GCOL].unique())}")
chk("both withdrawn participants present", all(w in IDS for w in WITHDRAWN))
chk("first withdrawal is Control", GRP[WITHDRAWN[0]] == 0, f"group code {GRP[WITHDRAWN[0]]}")
chk("second withdrawal is Intervention", GRP[WITHDRAWN[1]] == 1, f"group code {GRP[WITHDRAWN[1]]}")

DEMOG = [c for c in pre.columns if c.startswith("demog_")]
for w in WITHDRAWN:
    r1 = S["T1"].set_index("id").loc[w]
    chk(f"{w}: demographic data complete",
        bool(r1[DEMOG].notna().all()),
        f"{int(r1[DEMOG].notna().sum())}/{len(DEMOG)} fields")
    base_ok = r1[["CJARS_ss", "CJARS_ps", "CJARS_sjas",
                  "GARS_rr", "GARS_si", "GARS_sc", "GARS_er",
                  "ACSF_tp", "ACSF_cc"]].notna().all()
    chk(f"{w}: baseline C-JARS + GARS-3 + ACSF:SC present", bool(base_ok))
    bs_jast = r1[[f"BS_{d}" for d in BS_DOM] + ["JAST_arja", "JAST_aija"]]
    chk(f"{w}: NO baseline behavioural-sample or JAST data",
        bool(bs_jast.isna().all()),
        f"{int(bs_jast.notna().sum())} unexpected values")
    for t in ("T2", "T3"):
        rt = S[t].set_index("id").loc[w]
        chk(f"{w}: no {t} outcome data",
            bool(rt[OUTCOMES + RT_VARS].isna().all()),
            f"{int(rt[OUTCOMES + RT_VARS].notna().sum())} unexpected values")

print()
print("=" * 78)
print("STEP 2 — VERBAL-STATUS VARIABLE")
print("=" * 78)
chk("demog_Verbal exists", "demog_Verbal" in pre.columns)
chk("demog_Verbal coded 0/1 only", set(VERB.unique()) <= {0, 1},
    f"values {sorted(VERB.unique())}")
chk("demog_Verbal complete for all 43", bool(VERB.notna().all()))
chk("demog_Verbal identical across all three sheets",
    all(list(S[t].set_index("id")["demog_Verbal"].reindex(IDS))
        == list(VERB.reindex(IDS)) for t in TPS))
print("  README coding: 0 = Nonverbal, 1 = Verbal "
      "(ReadMe row 'demog_Verbal | Communication method | Nominal')")

print()
print("=" * 78)
print("STEP 3 — GUARD: forbidden variables must not enter the analysis")
print("=" * 78)
chk("GARS_tot_6 not in outcome list", "GARS_tot_6" not in OUTCOMES)
for f in FORBIDDEN:
    chk(f"{f} excluded", f not in OUTCOMES)
chk("exactly 16 outcomes", len(OUTCOMES) == 16, f"n = {len(OUTCOMES)}")
chk("JAST reaction times excluded from the outcome list",
    not any(v in OUTCOMES for v in RT_VARS))

print()
print("=" * 78)
print("STEP 4 — GROUP LABEL / COLUMN INTEGRITY")
print("=" * 78)
pada = pd.read_excel(RAW, sheet_name="PaDa", header=6,
                     keep_default_na=False, na_values=["NA"])
pada = pada[pada["id"].notna()].assign(id=lambda x: x["id"].astype(int))
lab = pada.set_index("id")["Group"].reindex(IDS)
code = pada.set_index("id")["Group_code"].astype(int).reindex(IDS)
chk("PaDa Group label matches Group_code for every id",
    bool(((lab == "Intervention") == (code == 1)).all()))
chk("PaDa Group_code matches the outcome-sheet group column",
    bool((code.values == GRP.reindex(IDS).values).all()))
chk("label mapping is 1 = Intervention, 0 = Control (NOT reversed)",
    bool((lab[code == 1] == "Intervention").all()
         and (lab[code == 0] == "Control").all()))
for t in TPS:
    chk(f"{t} sheet group column identical to T1",
        list(S[t][GCOL]) == list(pre[GCOL]))

# =============================================== verbal balance (section 9)
print()
print("=" * 78)
print("STEP 5 — BASELINE VERBAL-STATUS BALANCE (descriptive only)")
print("=" * 78)
bal = []
for code_, name in ((1, "Intervention"), (0, "Control")):
    ids = [i for i in IDS if GRP[i] == code_]
    v = int(sum(VERB[i] == 1 for i in ids))
    nv = len(ids) - v
    bal.append({"group": name, "n": len(ids),
                "verbal_n": v, "verbal_pct": round(100 * v / len(ids), 1),
                "nonverbal_n": nv, "nonverbal_pct": round(100 * nv / len(ids), 1)})
tot_v = int((VERB == 1).sum())
bal.append({"group": "Total", "n": len(IDS),
            "verbal_n": tot_v, "verbal_pct": round(100 * tot_v / len(IDS), 1),
            "nonverbal_n": len(IDS) - tot_v,
            "nonverbal_pct": round(100 * (len(IDS) - tot_v) / len(IDS), 1)})
BAL = pd.DataFrame(bal)
BAL.to_csv(HERE / "verbal_balance.csv", index=False)
print(BAL.to_string(index=False))

# =============================================== long dataset
rows = []
for t in TPS:
    d = S[t].set_index("id")
    for pid in IDS:
        rec = {"id": pid, "group": int(GRP[pid]), "time": t,
               "verbal": int(VERB[pid])}
        for v in OUTCOMES + RT_VARS:
            rec[v] = d.at[pid, v]
        rows.append(rec)
LONG = pd.DataFrame(rows).sort_values(["id", "time"])
LONG.to_csv(DER / "va_long.csv", index=False)
sp = LONG.copy()
sp["time"] = sp["time"].map({"T1": 1, "T2": 2, "T3": 3})
sp.to_csv(DER / "va_long_for_spss.csv", index=False)

# =============================================== sample sizes (section 8)
print()
print("=" * 78)
print("STEP 6 — SAMPLE SIZES BY OUTCOME")
print("=" * 78)
ss = []
for fam, vs in FAMILIES.items():
    for v in vs:
        w = LONG.pivot(index="id", columns="time", values=v)
        rec = {"family": fam, "outcome": v, "randomized_N": 43,
               "n_participants_with_any_obs": int(w.notna().any(axis=1).sum()),
               "n_observations_used": int(w.notna().sum().sum())}
        for t in TPS:
            rec[f"n_{t}"] = int(w[t].notna().sum())
            for code_, nm in ((0, "ctrl"), (1, "int")):
                ids = [i for i in IDS if GRP[i] == code_]
                rec[f"n_{t}_{nm}"] = int(w.loc[ids, t].notna().sum())
        rec["all_43_contribute_at_least_one_obs"] = (
            rec["n_participants_with_any_obs"] == 43)
        # NB: this does NOT mean complete data. No outcome has 129/129
        # observations; the maximum is 124 of a possible 129.
        ss.append(rec)
SS = pd.DataFrame(ss)
SS.to_csv(HERE / "sample_sizes_by_outcome.csv", index=False)
print(SS[["outcome", "n_participants_with_any_obs", "n_observations_used",
          "n_T1", "n_T2", "n_T3",
          "all_43_contribute_at_least_one_obs"]].to_string(index=False))

print()
n_fail = sum(1 for c in checks if c["result"] == "FAIL")
pd.DataFrame(checks).to_csv(DER / "va_preflight_checks.csv", index=False)
print("=" * 78)
print(f"PRE-FLIGHT: {len(checks) - n_fail}/{len(checks)} checks passed")
if n_fail:
    print("*** INCONSISTENCIES FOUND — resolve before fitting models ***")
    for c in checks:
        if c["result"] == "FAIL":
            print(f"   {c['check']}: {c['detail']}")
    sys.exit(1)
print("All participant-accounting claims verified. Safe to proceed.")
