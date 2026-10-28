# Data

None of the raw data is committed to this repository. Some of it is licensed
and cannot be redistributed. This file lists what each script expects, under
the exact filename the code reads, where it comes from, how to download it,
and the vintage used for the results in the thesis.

Place every file below in `Data/`, then run `01_build_data.py`. The scripts
download nothing.

## Files

| File in `Data/` | Contents | Read by | Source |
|---|---|---|---|
| `Cash_RiskfreeTS.csv` | CRSP risk-free series, the 1-Month nominal rows | `01`, `Plot_Shortrate`, `Plot_Drivers` | CRSP via WRDS, licensed |
| `Bond_MonthlyTS.csv` | CRSP issue-level Treasury monthly time series, about 60 MB | `01` | CRSP via WRDS, licensed |
| `Bond_FixedTermTS.csv` | CRSP Fixed Term Indexes, diagnostic only, begins 1961-06 | `01` | CRSP via WRDS, licensed |
| `GoyalWelch_Monthly.csv` | `ret` (S&P 500 total return) and `ltr` (Ibbotson long-term government bond) | `01`, `06` | Amit Goyal's website, public |
| `Driver_DGS3MO.csv` | Three-month Treasury constant-maturity yield, daily | `01`, `Plot_Drivers` | FRED, public |
| `Driver_DGS10.csv` | Ten-year Treasury constant-maturity yield, daily | `01`, `Plot_Drivers` | FRED, public |
| `Driver_VIXCLS.csv` | VIX close, daily | `01`, `Plot_Drivers` | FRED, public |
| `Driver_ebp_csv.csv` | Excess bond premium, monthly, columns `date` and `ebp` | `01`, `Plot_Drivers` | Federal Reserve Board, public |

## How to download

Download every series over its full available history. Start in January 1950
where the source allows it, and otherwise as early as the series begins. Every
file must run to at least December 2025. A file that starts too late leaves
months of a panel empty, and `01_build_data.py` then stops.

### CRSP, via WRDS

All three files come from CRSP Treasuries on WRDS
(https://wrds-www.wharton.upenn.edu). Save each query as comma-delimited text.
WRDS adds the date `MCALDT` to every output without it being selected.

**`Cash_RiskfreeTS.csv`.** CRSP, Treasuries, Riskfree Series (1-month and
3-month). Select `KYTREASNOX`, `TIDXFAM`, `TTERMTYPE`, `TTERMLBL`,
`RMTREASNO`, `RMCRSPID`, `TMBIDYTM`, `TMASKYTM`, `TMYTM` and `TMDURATN`.
`01_build_data.py` uses `TTERMLBL`, `MCALDT`, `TMYTM` and `TMDURATN`. The
other columns are not read.

**`Bond_MonthlyTS.csv`.** CRSP, Treasuries, Monthly Time Series, all issues.
Do not filter by issue type or maturity in WRDS, since `01_build_data.py`
applies its own filters. Select `KYTREASNO`, `KYCRSPID`, `CRSPID`, `TCUSIP`,
`TDATDT`, `TMATDT`, `IWHY`, `TCOUPRT`, `TNIPPY`, `TVALFC`, `TFCPDT`,
`IFCPDTF`, `TFCALDT`, `TNOTICE`, `IYMCN`, `ITYPE`, `IUNIQ`, `ITAX`, `IFLWR`,
`TBANKDT`, `TSTRIPELIG`, `TFRGNTGT`, `TMBID`, `TMASK`, `TMNOMPRC`,
`TMNOMPRC_FLG`, `TMSOURCR`, `TMACCINT`, `TMRETNUA`, `TMYLD`, `TMDURATN`,
`TMTOTOUT`, `TMPUBOUT`, `TMPCYLD`, `TMRETNXS` and `TMPDINT`. 

**`Bond_FixedTermTS.csv`.** CRSP, Treasuries, Fixed Term Indexes. Set the
frequency to monthly. The variable list on the form shows the daily names.
Select `KYTREASNOX`, `TTERMLBL`, `TDYEARSTM`, `TDDURATN` and `TDRETADJ`.
With monthly frequency, WRDS delivers them under the monthly names
`TMYEARSTM`, `TMDURATN` and `TMRETADJ`. Check that
the downloaded header carries these `TM` names and `MCALDT`. A file with
`TD` names and `CALDT` is a daily extract and is not usable here. The index
begins in June 1961.

### Welch and Goyal

Amit Goyal's website, https://sites.google.com/view/agoyal145, under Goyal,
Welch and Zafirov (2024), "A Comprehensive 2022 Look at the Empirical
Performance of Equity Premium Prediction", *Review of Financial Studies*
37(11), 3490–3557. Download the file "All data up to 2025 (includes data from
GW (2008) and GWZ (2024) in one file) in excel format", the full-sample
version. Save its sheet "Monthly" as `GoyalWelch_Monthly.csv`. The zipped CSV
files on the same page hold only the predictors added in 2024 and contain
neither `ret` nor `ltr`.

### FRED

Download each series at daily frequency as CSV from its FRED page. The file
has two columns, `observation_date` and the series ID.

| File | FRED series | Page |
|---|---|---|
| `Driver_DGS3MO.csv` | `DGS3MO`, 3-month Treasury constant maturity | https://fred.stlouisfed.org/series/DGS3MO |
| `Driver_DGS10.csv` | `DGS10`, 10-year Treasury constant maturity | https://fred.stlouisfed.org/series/DGS10 |
| `Driver_VIXCLS.csv` | `VIXCLS`, CBOE Volatility Index close | https://fred.stlouisfed.org/series/VIXCLS |

### Excess bond premium

Favara, Gilchrist, Lewis and Zakrajšek (2016), "Updating the Recession Risk
and the Excess Bond Premium", FEDS Notes, Board of Governors of the Federal
Reserve System:
https://www.federalreserve.gov/econres/notes/feds-notes/updating-the-recession-risk-and-the-excess-bond-premium-20161006.html.
The note links the current estimates as a CSV file at
https://www.federalreserve.gov/econres/notes/feds-notes/ebp_csv.csv. Save it
as `Driver_ebp_csv.csv`. The measure itself is from Gilchrist and Zakrajšek
(2012).

## Vintages

The download date of each file used for the results in the thesis:

| File | Downloaded |
|---|---|
| `Bond_MonthlyTS.csv` | 2026-07-22 |
| `Cash_RiskfreeTS.csv` | 2026-07-22 |
| `Bond_FixedTermTS.csv` | 2026-08-21 |
| `Driver_DGS3MO.csv`, `Driver_DGS10.csv`, `Driver_VIXCLS.csv` | 2026-07-24 |
| `Driver_ebp_csv.csv` | 2026-07-24 |
| `GoyalWelch_Monthly.csv` | 2026-08-22 |

## What `01_build_data.py` makes of it

`01_build_data.py` writes two panels into `Data/`:

```
panel_replication.csv   1952-01 to 2000-12   the Ang and Bekaert sample
panel_modern.csv        1990-02 to 2025-12   the specification ladder
```

These are the only data inputs of the estimation scripts. After `01`, only
`06` (the Ibbotson series `ltr`) and the two `Plot` scripts read raw files.
The construction of the bond series, the dating of the covariates and the
excess bond premium's publication lag are described in Chapter 5 of the thesis
and in the docstrings of `01_build_data.py`.

The Fixed Term file is optional. It feeds only the comparison in
Section 5.2.1, which is skipped if the file is absent.

## Vintage and revisions

Several of these series are revised. The results in the thesis were produced
on the vintages listed above. A later vintage changes the panels, and through
them every estimate.

A different vintage also changes the AR(1) persistence of the term spread,
which `07d_noise_rung.py` estimates from the panel at the start of every run
and matches its artificial covariates to. On the vintage used here it is
0.9756, and the script prints it, so a run on another vintage shows at once
whether the placebo columns differ.

## If you cannot obtain the licensed data

No results are committed. Every table and figure in the thesis depends on
the CRSP files, directly or through the panels, and cannot be reproduced
without them.
