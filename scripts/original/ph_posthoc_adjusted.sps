* ==========================================================================
* POST-HOC WITHIN-ARM PAIRWISE COMPARISONS - VERBAL-ADJUSTED MODEL
* VR Joint Attention RCT - PLOS ONE revision
*
* Model (final primary framework):
*   MIXED <outcome> BY group time WITH verbal
*     /FIXED = group time group*time verbal
*     /REPEATED = time | SUBJECT(id) COVTYPE(CS)      <- compound symmetry
*     /METHOD = REML
*
* /EMMEANS=TABLES(group*time) COMPARE(time) ADJ(BONFERRONI) produces, WITHIN
* EACH ARM, the three pairwise time comparisons T2-T1, T3-T2, T3-T1 with
* Bonferroni adjustment across the 3 comparisons and contrast-specific
* Satterthwaite degrees of freedom. These are exactly the quantities the Python
* implementation reports, and are captured here for verification.
*
* Verbal status is an ADJUSTMENT TERM ONLY. No group*verbal, time*verbal or
* three-way term appears anywhere.
*
* OUTCOMES: only the 14 that qualify under the ADJUSTED model (significant
* omnibus time effect OR significant omnibus group*time interaction).
* BS_IJA and BS_FL do not qualify and are not run.
* GARS_tot_6, GARS_cs, GARS_ms, CJARS_ass, CJARS_aps and the JAST RT indices
* are excluded from inference and appear nowhere below.
*
* Input: posthoc_verbal_adjusted/derived/ph_long_for_spss.csv
*        id group time verbal <outcomes>
*        group 0=Control 1=Intervention; time 1=T1 2=T2 3=T3; missing = SYSMIS
* ==========================================================================.

PRESERVE.
SET DECIMAL DOT.

GET DATA /TYPE=TXT
  /FILE="<PROJECT_ROOT>\posthoc_verbal_adjusted\derived\ph_long_for_spss.csv"
  /ENCODING='UTF8' /DELIMITERS="," /QUALIFIER='"' /ARRANGEMENT=DELIMITED
  /FIRSTCASE=2 /DATATYPEMIN PERCENTAGE=95.0
  /VARIABLES=
    id F8.0 group F8.0 time F8.0 verbal F8.0
    CJARS_ss F8.4 CJARS_ps F8.4 CJARS_sjas F8.4
    BS_RJA F8.4 BS_IJA F8.4 BS_EC F8.4 BS_FL F8.4 BS_T F8.4
    JAST_arja F8.4 JAST_aija F8.4
    GARS_rr F8.4 GARS_si F8.4 GARS_sc F8.4 GARS_er F8.4
    ACSF_tp F8.4 ACSF_cc F8.4
  /MAP.
DATASET NAME ph WINDOW=FRONT.

VALUE LABELS group 0 'Control' 1 'Intervention'
  / time 1 'T1' 2 'T2' 3 'T3'
  / verbal 0 'Nonverbal' 1 'Verbal'.

DEFINE !ph (v = !TOKENS(1))
MIXED !v BY group time WITH verbal
  /CRITERIA=CIN(95) MXITER(200) MXSTEP(10) SCORING(1)
    SINGULAR(0.000000000001) HCONVERGE(0, ABSOLUTE) LCONVERGE(0, ABSOLUTE)
    PCONVERGE(0.000001, ABSOLUTE)
  /FIXED=group time group*time verbal | SSTYPE(3)
  /METHOD=REML
  /PRINT=SOLUTION TESTCOV
  /REPEATED=time | SUBJECT(id) COVTYPE(CS)
  /EMMEANS=TABLES(group*time) WITH(verbal=MEAN) COMPARE(time) ADJ(BONFERRONI).
!ENDDEFINE.

* Capture every MIXED table. A SUBTYPES filter is deliberately NOT used: in an
* earlier run such a filter silently dropped the EMMEANS tables, which are the
* ones carrying the pairwise comparisons and their Satterthwaite df.
OMS /SELECT TABLES
  /DESTINATION FORMAT=OXML
     OUTFILE='<PROJECT_ROOT>\posthoc_verbal_adjusted\derived\ph_spss26.xml'
  /TAG='ph'.

* --- C-JARS ---.
!ph v = CJARS_ss.
!ph v = CJARS_ps.
!ph v = CJARS_sjas.

* --- Behavioural sample (rater mean); BS_IJA and BS_FL do not qualify ---.
!ph v = BS_RJA.
!ph v = BS_EC.
!ph v = BS_T.

* --- JAST ---.
!ph v = JAST_arja.
!ph v = JAST_aija.

* --- GARS-3: the four universal subscales only ---.
!ph v = GARS_rr.
!ph v = GARS_si.
!ph v = GARS_sc.
!ph v = GARS_er.

* --- ACSF:SC ---.
!ph v = ACSF_tp.
!ph v = ACSF_cc.

OMSEND TAG=['ph'].

RESTORE.
* ===================== end of file =====================.
