# VR-based joint attention training in autism: data and analysis code

[![DOI](https://zenodo.org/badge/1357580560.svg)](https://doi.org/10.5281/zenodo.22311114)

Data and analysis code underlying **"Virtual Reality as a Complementary Tool for
Joint Attention Training in Autism: Evidence from a Randomized Controlled Study."**

- **Trial registration:** IRCT20221224056907N1 (Iranian Registry of Clinical Trials)
- **Ethics approval:** IR.USWR.REC.1401.177, Ethics Commission of the University of
  Social Welfare and Rehabilitation Sciences, Tehran, Iran (21 December 2022)
- **Protocol:** https://doi.org/10.32598/jpcp.14.1.1077.1
- **Systematic review:** [Virtual/augmented reality for joint attention skills improvement in autism spectrum disorder: a systematic review](https://doi.org/10.1080/20473869.2023.2277604)
- **VR system design:** [Design and development of a VR-based system for training response to joint attention in autistic children: A preliminary study](https://doi.org/10.1109/ICBME61513.2023.10488524)

## The trial in one paragraph

Forty-three boys with autism spectrum disorder, aged 6 years 0 months to 11 years
11 months, were randomized 1:1 with stratification by communication status
(verbal / nonverbal) to VR-based joint attention training plus treatment as usual
(n = 21) or to sham VR plus treatment as usual (n = 22). The five-week
intervention targeted responding to and initiating joint attention. Outcomes were
assessed at baseline (T1), post-intervention (T2, week 5) and follow-up (T3, week 9).

## Repository layout

```
data/       de-identified participant-level data (see Data section)
scripts/    public-data reproduction scripts and redacted, publication-adapted archival scripts
results/    every result file behind Tables 1-6 and S2.1-S2.3
docs/       codebook and de-identification note
```

## Reproducing the primary analysis

```bash
pip install -r requirements.txt
python scripts/reproduce_primary.py
```

This script refits the point estimates and standard errors for all thirty-two treatment contrasts. It programmatically compares the point estimates against `results/contrast_results.csv` using a tolerance of 0.005. Confidence intervals are reconstructed using the refitted estimates and standard errors together with the stored Satterthwaite degrees of freedom. All thirty-two point-estimate checks pass.

## Archival and sensitivity scripts

Files under `scripts/original/` are redacted and, where necessary,
publication-adapted copies of scripts used during the analysis and revision.
Most require the non-deposited source workbook or intermediate directories and
are not intended as standalone reproduction workflows.

The supported primary reproduction is:

`python scripts/reproduce_primary.py`

The stratified permutation and multiplicity analyses can be regenerated from
the deposited data using:

`python scripts/original/fs06_permutation_multiplicity.py`

## Deviations from the protocol

The protocol planned independent-sample t-tests or Mann–Whitney U tests, Sidak
multiplicity correction, and cause-dependent imputation. The final analysis
instead used longitudinal mixed-effects models adjusted for communication
status, likelihood-based handling of incomplete observations without imputation,
and Holm correction with Benjamini–Hochberg sensitivity analyses. These
deviations are also reported in the manuscript.

## The model

Every primary mixed-model inferential result in the paper comes from one specification:

```
Outcome ~ Group * Time + VerbalStatus + (1 | participant)
```

Time is categorical with baseline as the reference level; the participant random
intercept gives a compound-symmetry covariance structure for the repeated
measures; estimation is by restricted maximum likelihood. Communication status
enters as a covariate because it was the stratification factor used in
randomization. No interactions involving communication status are included.

Participants are analysed as randomized using all available observations.
Missing outcome values are not imputed; likelihood-based estimation accommodates
incomplete longitudinal records under a missing-at-random assumption.

The principal inferential test for each outcome is the omnibus group x time
interaction. Treatment effects are between-group differences in change from
baseline (T1->T2 and T1->T3), with contrast-specific Satterthwaite denominator
degrees of freedom. Cohen's d is the model estimate, and each of its confidence
limits, divided by the pooled baseline standard deviation.

Multiplicity: Holm adjustment within the primary joint-attention family
(10 outcomes, 20 contrasts) and the secondary social-communication family
(6 outcomes, 12 contrasts); Benjamini-Hochberg reported as a sensitivity analysis.

## Data

`data/analysis_long.csv` is the analysis file: 129 rows, one per participant per
assessment, carrying group, timepoint, communication status and the twenty
outcome variables. `data/behavioral_raters_long.csv` carries the two raters'
scores separately, which the inter-rater reliability analysis needs.
`data/participant_flow.csv` gives per-participant assessment availability and
reconciles with the CONSORT diagram.

**These files are de-identified.** Age, sex, parental education, computer-game
experience and previous VR experience have been removed, and participant
identifiers have been replaced with opaque codes that do not encode recruitment
site or enrolment order. `docs/DEIDENTIFICATION.md` records what was removed and
why. None of the removed variables enters any model in the paper; they appear
only in the baseline table, which is published in the article.

All participants were male and none had previous VR experience, so those two
variables were constant across the sample and carry no information.

## Software

The primary mixed-model analyses were implemented in Python 3.12 with
statsmodels and independently cross-validated against the IBM SPSS Statistics
26 MIXED procedure; `results/spss_vs_python.csv` records that comparison.
The corresponding SPSS syntax is included in `scripts/original/`.
Machine-specific paths in the deposited scripts have been replaced by explicit
placeholders such as `<PROJECT_ROOT>` and `<EXTERNAL_SPSS_DIR>`.

## Licence

Code is released under the MIT Licence. Data and documentation are released
under CC BY 4.0. See `LICENSE`.
