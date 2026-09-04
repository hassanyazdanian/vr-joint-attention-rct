# Codebook

## data/analysis_long.csv

129 rows: 43 participants x 3 assessments. One row per participant per assessment.
Empty cells indicate no value available; values were not imputed.

| Variable | Description | Coding |
|---|---|---|
| `id` | Participant identifier | `P01`-`P43`, opaque |
| `group` | Randomized allocation | 0 = control (sham VR + TAU); 1 = intervention (VR + TAU) |
| `time` | Assessment | `T1` baseline; `T2` post-intervention (week 5); `T3` follow-up (week 9) |
| `verbal` | Communication status at baseline | 0 = nonverbal; 1 = verbal. Stratification factor and model covariate |
| `CJARS_ss` | C-JARS Social Symptoms | Higher = more social symptoms |
| `CJARS_ps` | C-JARS Prosocial Scale | Higher = better |
| `CJARS_sjas` | C-JARS Summary Joint Attention Scale | Higher = better |
| `BS_RJA` | Behavioural sample, responding to joint attention | Frequency count, mean of two raters. Higher = better |
| `BS_IJA` | Behavioural sample, initiating joint attention | as above |
| `BS_EC` | Behavioural sample, eye contact | as above |
| `BS_FL` | Behavioural sample, follow behaviour | as above |
| `BS_T` | Behavioural sample, total score | as above |
| `JAST_arja` | JAST average responding to joint attention | 0-3, higher = better |
| `JAST_aija` | JAST average initiating joint attention | 0-3, higher = better |
| `GARS_rr` | GARS-3 Restrictive/Repetitive Behaviors | T-score, higher = greater difficulty |
| `GARS_si` | GARS-3 Social Interaction | as above |
| `GARS_sc` | GARS-3 Social Communication | as above |
| `GARS_er` | GARS-3 Emotional Responses | as above |
| `ACSF_tp` | ACSF:SC typical performance | 1 = Level I (highest skill) to 5 = Level V (lowest) |
| `ACSF_cc` | ACSF:SC best capacity | as above |
| `JAST_RT_t1`-`t4` | JAST response-time indices | Seconds. Reported descriptively only; not analysed inferentially, because a response time is undefined when a child does not respond to the cue |

## data/behavioral_raters_long.csv

The two raters' behavioural-sample scores held separately, for the inter-rater
reliability analysis. Suffix `_M` and `_Z` denote the two independent raters.
The analysed variable in `analysis_long.csv` is the mean of the two.

## data/participant_flow.csv

Per-participant allocation, withdrawal and assessment availability. Reconciles
with the CONSORT flow diagram and with the per-cell sample sizes in Tables 2-6.

## results/

| File | Contents |
|---|---|
| `contrast_results.csv` | All model contrasts: omnibus interaction, cross-sectional between-group differences, and the T1->T2 / T1->T3 treatment contrasts with estimates, SEs, Satterthwaite df, CIs and Cohen's d |
| `multiplicity_results.csv` | The 32 treatment contrasts with raw, Holm-adjusted and Benjamini-Hochberg-adjusted p-values, within and across families |
| `posthoc_pairwise_adjusted.csv` | Within-group pairwise comparisons among timepoints, Bonferroni-adjusted, with 98.33% intervals |
| `qualifying_outcomes.csv` | Which outcomes qualified for within-group testing, and on which omnibus effect |
| `itt_descriptives_adjusted.csv` | Observed means, SDs and cell sizes |
| `icc_results.csv` | Inter-rater reliability, ICC(A,k), per behavioural domain per assessment |
| `behavioral_nb_results.csv` | Negative-binomial sensitivity models for the behavioural counts |
| `acsf_ordinal_results.csv`, `jast_ordinal_results.csv` | Cumulative-logit sensitivity models |
| `stratified_permutation_results.csv` | 10,000 permutations within verbal-status strata |
| `jast_rt_descriptive.csv` | Response-time descriptives, with the missingness taxonomy |
| `sample_sizes_by_outcome.csv` | Participants and observations contributing to each model |
| `spss_vs_python.csv` | Quantity-by-quantity comparison against IBM SPSS Statistics 26 |
| `reproduced_contrasts.csv` | Output of `scripts/reproduce_primary.py` |
