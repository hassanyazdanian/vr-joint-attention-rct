* ==========================================================================
* INTER-RATER RELIABILITY — BEHAVIOURAL SAMPLE (manuscript S2 Table source)
* VR Joint Attention RCT - PLOS ONE revision
*
* Primary quantity (S2 Table):
*   RELIABILITY /ICC=MODEL(RANDOM) TYPE(ABSOLUTE)
*   -> two-way random effects, absolute agreement.
*      SPSS prints BOTH "Single Measures" = ICC(A,1) and
*      "Average Measures" = ICC(A,k) with k = 2, each with a 95% CI.
*
* Also run for BS Total at baseline only:
*   RELIABILITY /ICC=MODEL(MIXED) TYPE(CONSISTENCY)
*   -> reproduces the ORIGINAL published value of 0.937, which is
*      ICC(C,1), so the change to the average-measures figure is auditable.
*
* Raters M and Z; participants with both ratings present at that timepoint.
*
* Input: icc_verbal_adjusted/derived/icc_wide_for_spss.csv
*        id, then <DOM>_<TP>_<RATER> for DOM in RJA IJA EC FL T,
*        TP in T1 T2 T3, RATER in M Z.
* ==========================================================================.

PRESERVE.
SET DECIMAL DOT.

GET DATA /TYPE=TXT
  /FILE="<PROJECT_ROOT>\icc_verbal_adjusted\derived\icc_wide_for_spss.csv"
  /ENCODING='UTF8' /DELIMITERS="," /QUALIFIER='"' /ARRANGEMENT=DELIMITED
  /FIRSTCASE=2 /DATATYPEMIN PERCENTAGE=95.0
  /VARIABLES=
    id F8.0
    RJA_T1_M F8.4 RJA_T1_Z F8.4 IJA_T1_M F8.4 IJA_T1_Z F8.4
    EC_T1_M F8.4 EC_T1_Z F8.4 FL_T1_M F8.4 FL_T1_Z F8.4
    T_T1_M F8.4 T_T1_Z F8.4
    RJA_T2_M F8.4 RJA_T2_Z F8.4 IJA_T2_M F8.4 IJA_T2_Z F8.4
    EC_T2_M F8.4 EC_T2_Z F8.4 FL_T2_M F8.4 FL_T2_Z F8.4
    T_T2_M F8.4 T_T2_Z F8.4
    RJA_T3_M F8.4 RJA_T3_Z F8.4 IJA_T3_M F8.4 IJA_T3_Z F8.4
    EC_T3_M F8.4 EC_T3_Z F8.4 FL_T3_M F8.4 FL_T3_Z F8.4
    T_T3_M F8.4 T_T3_Z F8.4
  /MAP.
DATASET NAME icc WINDOW=FRONT.

* Capture every table. No SUBTYPES filter: an earlier run in this project showed
* such a filter can silently drop the tables of interest.
OMS /SELECT TABLES
  /DESTINATION FORMAT=OXML
     OUTFILE='<PROJECT_ROOT>\icc_verbal_adjusted\derived\icc_spss26.xml'
  /TAG='icc'.

DEFINE !icc (m = !TOKENS(1) / z = !TOKENS(1))
RELIABILITY
  /VARIABLES=!m !z
  /SCALE('ICC') ALL
  /MODEL=ALPHA
  /ICC=MODEL(RANDOM) TYPE(ABSOLUTE) CIN=95 TESTVAL=0.
!ENDDEFINE.

* --- T1 ---.
!icc m = RJA_T1_M z = RJA_T1_Z.
!icc m = IJA_T1_M z = IJA_T1_Z.
!icc m = EC_T1_M  z = EC_T1_Z.
!icc m = FL_T1_M  z = FL_T1_Z.
!icc m = T_T1_M   z = T_T1_Z.

* --- T2 ---.
!icc m = RJA_T2_M z = RJA_T2_Z.
!icc m = IJA_T2_M z = IJA_T2_Z.
!icc m = EC_T2_M  z = EC_T2_Z.
!icc m = FL_T2_M  z = FL_T2_Z.
!icc m = T_T2_M   z = T_T2_Z.

* --- T3 ---.
!icc m = RJA_T3_M z = RJA_T3_Z.
!icc m = IJA_T3_M z = IJA_T3_Z.
!icc m = EC_T3_M  z = EC_T3_Z.
!icc m = FL_T3_M  z = FL_T3_Z.
!icc m = T_T3_M   z = T_T3_Z.

* --- the ORIGINAL published analysis, for the 0.937 audit trail ---.
RELIABILITY
  /VARIABLES=T_T1_M T_T1_Z
  /SCALE('original consistency') ALL
  /MODEL=ALPHA
  /ICC=MODEL(MIXED) TYPE(CONSISTENCY) CIN=95 TESTVAL=0.

OMSEND TAG=['icc'].

RESTORE.
* ===================== end of file =====================.
