#!/usr/bin/env python3
"""
Reproduce the primary treatment-effect estimates from the de-identified data.

Fits the trial's primary model for each of the 16 analysed outcomes

    Outcome ~ Group * Time + VerbalStatus + (1 | participant)

with time treated as categorical (baseline as reference), a participant-level
random intercept (compound-symmetry covariance for the repeated measures) and
restricted maximum likelihood estimation, then extracts the two treatment
contrasts reported in the manuscript:

    T1 -> T2  between-group difference in change from baseline
    T1 -> T3  between-group difference in change from baseline

and compares them against results/contrast_results.csv, the output of the
analysis pipeline that produced Tables 2-6.

Usage:
    python scripts/reproduce_primary.py

Confidence intervals in the published tables use contrast-specific Satterthwaite
denominator degrees of freedom, which are taken here from the shipped results
file rather than recomputed; the point estimates and standard errors are
refitted from the data.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "analysis_long.csv"
REF = ROOT / "results" / "contrast_results.csv"

OUTCOMES = [
    "CJARS_ss", "CJARS_ps", "CJARS_sjas",
    "BS_RJA", "BS_IJA", "BS_EC", "BS_FL", "BS_T",
    "JAST_arja", "JAST_aija",
    "GARS_rr", "GARS_si", "GARS_sc", "GARS_er",
    "ACSF_tp", "ACSF_cc",
]

CONTRASTS = {
    "T1->T2": "C(group)[T.1]:C(time)[T.T2]",
    "T1->T3": "C(group)[T.1]:C(time)[T.T3]",
}

TOL = 5e-3  # agreement tolerance on the point estimate


def load():
    d = pd.read_csv(DATA)
    d["time"] = pd.Categorical(d["time"], categories=["T1", "T2", "T3"])
    d["group"] = pd.Categorical(d["group"], categories=[0, 1])
    return d


def fit(d, outcome):
    sub = d.dropna(subset=[outcome])
    model = smf.mixedlm(
        f"{outcome} ~ C(group)*C(time) + verbal", sub, groups=sub["id"]
    )
    return model.fit(reml=True), sub


def main():
    d = load()
    ref = pd.read_csv(REF)
    ref = ref[ref["contrast"].str.startswith("Treatment difference in change")]
    ref["key"] = ref["outcome"] + "|" + ref["contrast"].str.replace(
        "Treatment difference in change ", "", regex=False
    )
    ref = ref.set_index("key")

    rows, failures = [], 0
    for outcome in OUTCOMES:
        res, sub = fit(d, outcome)
        for label, term in CONTRASTS.items():
            est, se = res.params[term], res.bse[term]
            key = f"{outcome}|{label}"
            if key in ref.index:
                published = float(ref.loc[key, "adj_estimate"])
                df = float(ref.loc[key, "adj_df"])
                delta = abs(est - published)
                ok = delta < TOL
            else:
                published, df, delta, ok = np.nan, np.nan, np.nan, False
            failures += (not ok)
            tcrit = stats.t.ppf(0.975, df) if np.isfinite(df) else 1.96
            rows.append({
                "outcome": outcome,
                "contrast": label,
                "n_participants": sub["id"].nunique(),
                "n_observations": len(sub),
                "estimate": round(est, 4),
                "se": round(se, 4),
                "ci_lo": round(est - tcrit * se, 4),
                "ci_hi": round(est + tcrit * se, 4),
                "published_estimate": published,
                "abs_difference": round(delta, 6) if np.isfinite(delta) else "",
                "agrees": "yes" if ok else "NO",
            })

    out = pd.DataFrame(rows)
    dest = ROOT / "results" / "reproduced_contrasts.csv"
    out.to_csv(dest, index=False)

    width = max(len(o) for o in OUTCOMES)
    print(f"{'outcome':<{width}}  {'contrast':<7} {'estimate':>10} "
          f"{'published':>10} {'diff':>10}  agrees")
    for r in rows:
        print(f"{r['outcome']:<{width}}  {r['contrast']:<7} {r['estimate']:>10.4f} "
              f"{r['published_estimate']:>10.4f} {r['abs_difference']:>10}  "
              f"{r['agrees']}")

    print()
    print(f"{len(rows) - failures} of {len(rows)} contrasts reproduce "
          f"within {TOL}.")
    print(f"Written to {dest.relative_to(ROOT)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
