* ==========================================================================
* VERBAL-ADJUSTED SENSITIVITY ANALYSIS
* VR Joint Attention RCT - PLOS ONE revision
*
* Model A (current) : Outcome BY group time    /FIXED = group time group*time
* Model B (adjusted): Outcome BY group time WITH verbal
*                     /FIXED = group time group*time verbal
*
* Time categorical, T1 = 1 as reference. Covariance structure UNCHANGED from the
* validated pipeline: COVTYPE(CS) on /REPEATED, i.e. a participant random
* intercept. REML. All available observations. No imputation.
*
* Verbal status is a BASELINE ADJUSTMENT TERM ONLY - no group*verbal,
* time*verbal or three-way term is fitted.
*
* /EMMEANS ... COMPARE(group) yields the CONTRAST-SPECIFIC Satterthwaite df for
* the group difference at each timepoint. These are deliberately captured so the
* Python implementation can be validated contrast by contrast, rather than
* reusing the omnibus Group x Time denominator df.
*
* OUTCOMES: the 16 manuscript outcomes only.
*   GARS_tot_6 is DELIBERATELY EXCLUDED and must not appear anywhere below.
*   GARS_cs / GARS_ms are excluded (verbal-only subscales).
*   CJARS_ass / CJARS_aps are excluded (derived averages).
*   JAST reaction times are excluded from the primary comparison.
*
* Input: analysis_verbal_adjusted/derived/va_long_for_spss.csv
*        id group time verbal <outcomes>   (group 0=Control 1=Intervention;
*        time 1=T1 2=T2 3=T3; missing values already SYSMIS)
* ==========================================================================.

PRESERVE.
SET DECIMAL DOT.

GET DATA /TYPE=TXT
  /FILE="<PROJECT_ROOT>\analysis_verbal_adjusted\derived\va_long_for_spss.csv"
  /ENCODING='UTF8' /DELIMITERS="," /QUALIFIER='"' /ARRANGEMENT=DELIMITED
  /FIRSTCASE=2 /DATATYPEMIN PERCENTAGE=95.0
  /VARIABLES=
    id F8.0 group F8.0 time F8.0 verbal F8.0
    CJARS_ss F8.4 CJARS_ps F8.4 CJARS_sjas F8.4
    BS_RJA F8.4 BS_IJA F8.4 BS_EC F8.4 BS_FL F8.4 BS_T F8.4
    JAST_arja F8.4 JAST_aija F8.4
    GARS_rr F8.4 GARS_si F8.4 GARS_sc F8.4 GARS_er F8.4
    ACSF_tp F8.4 ACSF_cc F8.4
    JAST_RT_t1 F8.4 JAST_RT_t2 F8.4 JAST_RT_t3 F8.4 JAST_RT_t4 F8.4
  /MAP.
DATASET NAME va WINDOW=FRONT.

VALUE LABELS group 0 'Control' 1 'Intervention'
  / time 1 'T1 baseline' 2 'T2 post' 3 'T3 follow-up'
  / verbal 0 'Nonverbal' 1 'Verbal'.

* Sanity check on the imported file.
FREQUENCIES VARIABLES=group time verbal /ORDER=ANALYSIS.

* ==========================================================================
* MACRO: Model A (unadjusted) then Model B (verbal-adjusted), same outcome.
* ==========================================================================.

DEFINE !pair (v = !TOKENS(1))

* ---------- Model A : unadjusted ----------.
MIXED !v BY group time
  /CRITERIA=CIN(95) MXITER(200) MXSTEP(10) SCORING(1)
    SINGULAR(0.000000000001) HCONVERGE(0, ABSOLUTE) LCONVERGE(0, ABSOLUTE)
    PCONVERGE(0.000001, ABSOLUTE)
  /FIXED=group time group*time | SSTYPE(3)
  /METHOD=REML
  /PRINT=SOLUTION TESTCOV
  /REPEATED=time | SUBJECT(id) COVTYPE(CS)
  /EMMEANS=TABLES(group*time) COMPARE(group) ADJ(LSD).

* ---------- Model B : verbal-adjusted ----------.
MIXED !v BY group time WITH verbal
  /CRITERIA=CIN(95) MXITER(200) MXSTEP(10) SCORING(1)
    SINGULAR(0.000000000001) HCONVERGE(0, ABSOLUTE) LCONVERGE(0, ABSOLUTE)
    PCONVERGE(0.000001, ABSOLUTE)
  /FIXED=group time group*time verbal | SSTYPE(3)
  /METHOD=REML
  /PRINT=SOLUTION TESTCOV
  /REPEATED=time | SUBJECT(id) COVTYPE(CS)
  /EMMEANS=TABLES(group*time) WITH(verbal=MEAN) COMPARE(group) ADJ(LSD).

!ENDDEFINE.

* ==========================================================================
* OMS capture -> OXML, so every number can be read back exactly.
* ==========================================================================.
* Capture EVERY table produced by MIXED. A SUBTYPES filter was tried first and
* silently dropped the EMMEANS pairwise tables, which are exactly the ones
* carrying the contrast-specific Satterthwaite df, so no filter is used here.
OMS /SELECT TABLES
  /DESTINATION FORMAT=OXML
     OUTFILE='<PROJECT_ROOT>\analysis_verbal_adjusted\derived\va_spss26.xml'
  /TAG='va'.

* --- C-JARS ---.
!pair v = CJARS_ss.
!pair v = CJARS_ps.
!pair v = CJARS_sjas.

* --- Behavioural sample (rater mean) ---.
!pair v = BS_RJA.
!pair v = BS_IJA.
!pair v = BS_EC.
!pair v = BS_FL.
!pair v = BS_T.

* --- JAST ---.
!pair v = JAST_arja.
!pair v = JAST_aija.

* --- GARS-3 : FOUR subscales only. GARS_tot_6 is NOT run. ---.
!pair v = GARS_rr.
!pair v = GARS_si.
!pair v = GARS_sc.
!pair v = GARS_er.

* --- ACSF:SC ---.
!pair v = ACSF_tp.
!pair v = ACSF_cc.

OMSEND TAG=['va'].

RESTORE.
* ===================== end of file =====================.
