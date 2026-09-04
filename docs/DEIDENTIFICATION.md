# De-identification note

The published data files are derived from the trial's source workbook. This note
records exactly what was changed, so that a reader can judge the residual
re-identification risk rather than take a general assurance on trust.

## What was removed

| Variable | Reason |
|---|---|
| `demog_Age` (age in months) | 33 distinct values across 43 children; 25 children had an age in months shared with no one else in the trial. The single most identifying field in the dataset. |
| `demog_mom_edu`, `demog_dad_edu` | Five and six ordered levels across 43 families. Combined with age and communication status, these made all 43 participants unique. |
| `demog_Comp_expr` (computer-game experience) | Four ordered levels; contributes to the same combination. |
| `demog_Sex` | Constant (all participants male). Removed as uninformative; recorded in the README instead. |
| `demog_VR_expr` | Constant (no participant had previous VR experience). Removed as uninformative; recorded in the README instead. |
| `No.` (row number) | Reflected enrolment order. |

## What was changed

Participant identifiers were replaced. The original identifiers encoded the
recruitment site in their leading digit and enrolment position in the remainder,
so they disclosed which of the four participating schools each child attended.
The replacement codes (`P01`-`P43`) were assigned after a seeded shuffle, so the
ordering of the new codes does not recover site membership. The crosswalk between
original and replacement identifiers is held by the study team and is not
published.

## What remains, and the residual risk

The published files retain group allocation, communication status, timepoint and
the outcome scores. The smallest cell in the group-by-communication-status
cross-tabulation contains 7 participants, so no participant is distinguishable
from the others on those two variables.

The irreducible residual risk is that the files contain 43 longitudinal profiles
across twenty outcome variables. Anyone holding the original scoring sheets, such
as one of the two behavioural-sample raters or a participating school, could in
principle match a row to a child. No amount of column removal eliminates this,
and it is stated here rather than left implicit. What the de-identification does
achieve is to prevent re-identification by a reader who does not already hold
trial records.

## What this cost the analysis

Nothing. The model fitted for every inferential result in the paper is
`Outcome ~ Group * Time + VerbalStatus + (1 | participant)`. Age, parental
education, computer-game experience, sex and previous VR experience appear in no
model. They populate the published baseline table only, which is printed in the
article. All thirty-two treatment contrasts reproduce from the de-identified file
(`scripts/reproduce_primary.py`).
