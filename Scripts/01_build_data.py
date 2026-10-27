"""
01_build_data.py
================
Builds the two monthly panels the estimator consumes.

    panel_replication.csv   1952-01 to 2000-12   Ang-Bekaert (2004) sample
    panel_modern.csv        1990-02 to 2025-12   ladder M0-M4

--------------------------------------------------------------------------
COLUMN CONTRACT
--------------------------------------------------------------------------
Row t holds:

    r_obs    T-bill yield quoted at the close of month t       (AB's r_t)
    r        T-bill yield quoted at the close of month t-1,    (AB's r_{t-1})
             i.e. the rate earned OVER month t
    rb_x     excess bond return earned during month t,   = rb - r
    re_x     excess equity return earned during month t, = re - r

    z_short  transition covariate: the short rate.  Identical to `r`; kept
             as its own column so the covariate list names what it uses.
    z_term   term spread,           close of month t-1
    z_vix    VIX,                   close of month t-1
    z_ebp    excess bond premium,   reference month t-2 (publication lag)

`r` is one number doing three jobs: AR lag in the short-rate equation,
argument of the transition logistic, and the rate netted out to form the
month's excess returns.  Everything here is built to keep those three the
same array.

Invariant:  r[t] == r_obs[t-1].  Not asserted here; 02_msm.prepare(),
03_replication.py and 07_ladder.load_modern() check it before estimation.
Both datings are produced by reindexing one series on the monthly
PeriodIndex, never by a positional .shift().  The index is checked for
contiguity first, so a gap cannot silently pair non-adjacent months.

--------------------------------------------------------------------------
UNITS
--------------------------------------------------------------------------
All return columns are percent per month, the unit AB report estimates in.
The portfolio code converts to decimals before forming weights.

The short rate is carried twice:

    r, r_obs, z_short   percent per month   what the model uses
    r_ann               percent per year    reporting and figures only

`r_ann` is the raw CRSP annualised yield.  `r` de-annualises it over a
uniform twelfth of a year, so the two are not exactly 12:1.  `r_hpr`
de-annualises over the bill's actual term and is a diagnostic only.  See
build_cash().

--------------------------------------------------------------------------
DIAGNOSTICS
--------------------------------------------------------------------------
Printed before the panels are written.  None of them feed the panel.

    verify_total_return   is the CRSP bond return a total return? (5.2.1)
    report                moments, short rate, AR(1) fit, realised maturity
                          and callable share of each bond series (5.1, 5.2.3,
                          5.6, Table 5.3 for the covariates)
    gw_benchmark          against AB's Table 2 and the Welch-Goyal Ibbotson
                          and S&P series (Table 5.1, Section 5.3)
    validate_fixed_term   ten-year series against CRSP's constant-maturity
                          index (5.2.1)
    callable_coverage     what a maturity band or a noncallable-only filter
                          would cost (5.2.1)
"""

import os
import numpy as np
import pandas as pd
# Load 00_common.py by its path.  A normal import is impossible because module
# names cannot start with a digit.  The file is found next to this script and
# registered under the name `common`.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "common", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "00_common.py"))
common = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(common)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DATA_DIR = common.DATA_DIR                 # raw inputs, read only
OUT_DIR  = common.DATA_DIR                 # the two panels are written here
ZLB_WINDOWS = common.ZLB_WINDOWS           # fed funds target range at 0-0.25%

REPL_START,   REPL_END   = "1952-01", "2000-12"   # Ang-Bekaert (2004) sample
MODERN_START, MODERN_END = "1990-02", "2025-12"   # ladder sample. VIX starts 1990-01

# --------------------------------------------------------------------------
# Bond series
# --------------------------------------------------------------------------
# Two bond series are built on both panels, a ten-year and a twenty-five-year
# one.  `rb_x` is a copy of whichever series the panel uses as its baseline.
# The twenty-five-year series is a diagnostic (Section 5.2.3).
BOND_TARGET_LONG = 25.0
BOND_TARGET_10Y  = 10.0

REPL_BASELINE_LEG   = "10y"
MODERN_BASELINE_LEG = "10y"

BOND_PIN_TOLERANCE = 1.0      # report() flags months this far off target

# CRSP issue types: 1 noncallable bond, 2 noncallable note, 5 callable bond.
# Excludes certificates (3), bills (4), flower bonds and TIPS (11, 12).
BOND_ITYPES = (1, 2, 5)

# Read only by validate_fixed_term(), never an input to the panel
FIXED_TERM_FILES = ["Bond_FixedTermTS.csv", "Cash_RiskfreeTS.csv"]

# Ang and Bekaert (2002), Table 2 Panel B, annualised percent.
AB_TABLE2 = {
    "excess bond":  {"mean": 1.34, "sd": 9.00},
    "excess stock": {"mean": 7.37, "sd": 14.33},
}


# --------------------------------------------------------------------------
# Helper
# --------------------------------------------------------------------------
# CRSP dates rows by the last trading day of the month, FRED-derived series
# by the last calendar day.  Joining on raw timestamps would drop rows, so
# everything below is keyed on a monthly Period.

def to_period(s):
    """
    Convert a column of dates into monthly periods.

    Parameters
    ----------
    s : pd.Series
        Dates, as text or timestamps.

    Returns
    -------
    pd.Series
        The same values as monthly periods, e.g. 1952-01-31 -> 1952-01.

    """
    return pd.to_datetime(s).dt.to_period("M")


# ==========================================================================
# 1.  Cash Asset
# ==========================================================================

def build_cash():
    """
    One-month T-bill rate, stored twice with different monthly dating.

    Reads
    -----
    Cash_RiskfreeTS.csv, filtered to the 1-Month nominal series.
        TMYTM     annualised yield to maturity, percent
        TMDURATN  days to maturity of the bill actually used

    Returns
    -------
    pd.DataFrame
        Monthly PeriodIndex spanning one month beyond the quoted series:
            r_obs, r_obs_ann, obs_date   quoted at the close of month t
            r, r_ann                     quoted at the close of month t-1,
                                         hence earned over month t
            r_hpr                        diagnostic, see Notes
        `r` and `r_obs` are percent per month, `r_ann` and `r_obs_ann`
        percent per year.

    Notes
    -----
    The yield is de-annualised over a uniform twelfth of a year (Section 5.1).
    `r_hpr` de-annualises over the bill's actual term instead and is a
    diagnostic only. nothing downstream uses it.
    """
    raw = pd.read_csv(os.path.join(DATA_DIR, "Cash_RiskfreeTS.csv"))

    # extract 1-Month nominal risk-free series
    m1 = raw[raw["TTERMLBL"].str.contains("1-Month", na=False)].copy()
    m1 = m1[["MCALDT", "TMYTM", "TMDURATN"]].dropna()  # drop incomplete months
    m1["obs_month"] = to_period(m1["MCALDT"])
    m1 = m1.sort_values("obs_month").set_index("obs_month")

    # De-annualise
    # uniform twelfth
    r_q   = 100.0 * ((1.0 + m1["TMYTM"] / 100.0) ** (1.0 / 12.0) - 1.0)    
    #  actual term
    r_hpr = 100.0 * ((1.0 + m1["TMYTM"] / 100.0) ** (m1["TMDURATN"] / 365.0) - 1.0)

    # one yield, two datings. union so both fit, ends trimmed by the joins
    idx  = m1.index                          # month at whose end the yield is quoted
    cash = pd.DataFrame(index=idx.union(idx + 1))   # quoted months + one more
    cash.index.name = "month"

    # quoted at the close of month t, widened onto the union, not moved
    cash["r_obs"]     = r_q.reindex(cash.index)                            # monthly %
    cash["r_obs_ann"] = m1["TMYTM"].reindex(cash.index)                    # annualised %
    cash["obs_date"]  = pd.to_datetime(m1["MCALDT"]).reindex(cash.index)   # quote date

    # same numbers relabelled one month on, so they sit where earned.
    # set_axis + reindex, not .shift(), so a gap cannot pair distant months.
    cash["r"]     = r_q.set_axis(idx + 1).reindex(cash.index)          # rate earned over t
    cash["r_ann"] = m1["TMYTM"].set_axis(idx + 1).reindex(cash.index)  # same, annualised
    cash["r_hpr"] = r_hpr.set_axis(idx + 1).reindex(cash.index)        # diagnostic
    return cash


# ==========================================================================
# 2.  Bond Series
# ==========================================================================

# transform into percent per month.  TMRETNUA (issue level) is a decimal, 
# TMRETADJ (fixed term) is already percent.
_RET_SCALE = {"tmretnua": 100.0, "tmret": 100.0,
              "tmretadj": 1.0, "tdretadj": 1.0}

# Module-level cache for _eligible_issues().  The issue-level file is 60 MB
# and build_bond() is called four times (two bond series, two panels), so without
# this it would be parsed and filtered four times over.  The filtering does
# not depend on the panel or the target maturity, so one copy serves all
# four. callers only ever read it or copy it.
_ELIGIBLE_CACHE = None


def _eligible_issues():
    """
    The Treasury notes and bonds the bond series may be built from, one row
    per issue per month.

    Starts from every issue in the CRSP monthly Treasury file and keeps only
    those that are fit to stand in for a constant-maturity bond: noncallable
    bonds, noncallable notes and callable bonds, excluding bills,
    certificates, inflation-indexed issues, flower bonds and partially
    tax-exempt issues, with a non-missing return and more than six months to
    run (Section 5.2.1).  build_bond() then picks, each month, the issue
    closest to the target maturity from this set.

    Reads
    -----
    Bond_MonthlyTS.csv (CRSP issue-level monthly time series).

    Returns
    -------
    pd.DataFrame
        Filtered to BOND_ITYPES, non-flower, fully taxable, non-missing
        return and more than six months to run, with `ttm` in years and
        `obs_month` as a monthly Period.  Cached across calls.
    """
    global _ELIGIBLE_CACHE
    if _ELIGIBLE_CACHE is not None:
        return _ELIGIBLE_CACHE

    cols = ["KYTREASNO", "MCALDT", "TMATDT", "ITYPE", "IFLWR", "ITAX",
            "TCOUPRT", "TMRETNUA", "TMTOTOUT", "TMDURATN",
            "TMNOMPRC", "TMACCINT", "TMPDINT"]
    b = pd.read_csv(os.path.join(DATA_DIR, "Bond_MonthlyTS.csv"),
                    usecols=cols, low_memory=False)   # usecols keeps the read small

    b["MCALDT"] = pd.to_datetime(b["MCALDT"])         # observation month-end
    b["TMATDT"] = pd.to_datetime(b["TMATDT"])         # maturity date
    b["ttm"]    = (b["TMATDT"] - b["MCALDT"]).dt.days / 365.25   # years to run

    elig = b[
        b["ITYPE"].isin(BOND_ITYPES)      # bonds and notes only
        & (b["IFLWR"] == 1)               # not a flower bond
        & (b["ITAX"] == 1)                # fully taxable
        & b["TMRETNUA"].notna()           # return must exist
        & (b["ttm"] > 0.5)                # more than six months to run
    ].copy()
    elig["obs_month"] = to_period(elig["MCALDT"])   # join key for build_bond()

    _ELIGIBLE_CACHE = elig
    return elig


def build_bond(target, label):
    """
    Bond series at a target maturity: each month, the return on the eligible
    issue whose remaining maturity is closest to `target` years.

    Parameters
    ----------
    target : float
        Target maturity in years.
    label : str
        Suffix appended to every returned column, e.g. "_10y".

    Returns
    -------
    pd.DataFrame
        Monthly PeriodIndex, columns suffixed with `label`:
            rb          total return, percent per month
            bond_ttm    realised maturity of the selected issue, years
            bond_id     CRSP issue key, so switches can be counted
            bond_dur    duration of the selected issue, days
            bond_itype  CRSP issue type, 5 meaning callable

    Notes
    -----
    Ties on maturity distance break towards the larger issue by amount
    outstanding.  No hard maturity band is imposed, so the realised maturity
    drifts where issuance was thin; report() prints it (Section 5.2).
    TMRETNUA is a decimal total return and is scaled to percent here.
    """
    elig = _eligible_issues()
    # Each month, rank the eligible issues by distance from the target
    # maturity, breaking ties towards the larger issue by amount outstanding,
    # and keep the first: one issue per month.
    sel = (elig.assign(gap=(elig["ttm"] - target).abs())
               .sort_values(["obs_month", "gap", "TMTOTOUT"],
                            ascending=[True, True, False],
                            na_position="last")
               .drop_duplicates("obs_month", keep="first")
               .set_index("obs_month"))

    bond = pd.DataFrame({
        "rb":       100.0 * sel["TMRETNUA"],   # total return, decimal -> percent
        "bond_ttm": sel["ttm"],                # realised maturity of the pick
        "bond_id":  sel["KYTREASNO"],          # issue key, so switches can be counted
        "bond_dur": sel["TMDURATN"],           # bond duration
        "bond_itype": sel["ITYPE"],            # 5 = callable
        })
    bond.index.name = "month"
    return bond.add_suffix(label)              # e.g. rb_10y, bond_ttm_10y


def load_fixed_term(term_label):
    """
    CRSP Fixed Term Index at one maturity, the benchmark the constructed bond
    series is measured against.

    Searches the files in FIXED_TERM_FILES in order and returns the index from
    the first one that contains it.  The index is expected in
    `Bond_FixedTermTS.csv`, the download DATA.md lists.  `Cash_RiskfreeTS.csv`
    is a fallback for a WRDS extract that returns the risk-free and the
    fixed-term series in one file.  Used only by validate_fixed_term() for the
    comparison in Section 5.2.1.  Never an input to the panel, because the
    index starts in 1961-06 and cannot cover the replication panel from 1952.

    Parameters
    ----------
    term_label : str
        Maturity to look for in the CRSP series label, matched regardless of
        case.  The only value used is "10-Year".

    Reads
    -----
    The first file in FIXED_TERM_FILES carrying a usable Fixed Term block.
    Column names are lowercased and matched by pattern:
        ttermlbl   series label, must contain "fixed term" and term_label
        mcaldt     month-end date
        tmretadj   monthly total return (see _RET_SCALE for accepted names)
        tmyearstm  years to maturity

    Returns
    -------
    pd.DataFrame or None
        Monthly PeriodIndex with `rb` in percent per month and `bond_ttm` in
        years.  None if no file holds a usable block.

    """
    for fname in FIXED_TERM_FILES:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            continue

        d = pd.read_csv(path, low_memory=False)
        d.columns = [c.strip().lower() for c in d.columns]   # WRDS casing varies

        # locate the columns by pattern rather than by fixed name
        lab_col = next((c for c in d.columns if "ttermlbl" in c), None)
        ret_col = next((c for c in _RET_SCALE if c in d.columns), None)
        if lab_col is None or ret_col is None or "mcaldt" not in d.columns:
            continue                                          # not this file

        # "fixed term" excludes the risk-free rows sharing the file,
        # term_label picks the maturity
        sel = d[d[lab_col].astype(str).str.contains("fixed term", case=False, na=False)
                & d[lab_col].astype(str).str.contains(term_label, case=False, na=False)]
        sel = sel.dropna(subset=["mcaldt", ret_col])
        if sel.empty:
            continue                             # label absent or unpopulated

        out = pd.DataFrame(                      # scale to percent per month
            {"rb": _RET_SCALE[ret_col] * pd.to_numeric(sel[ret_col], errors="coerce")})
        out.index = to_period(sel["mcaldt"])
        out.index.name = "month"
        out["bond_ttm"] = (pd.to_numeric(sel["tmyearstm"], errors="coerce").to_numpy()
                           if "tmyearstm" in sel.columns else np.nan)
        out["bond_dur"] = (pd.to_numeric(sel["tmduratn"], errors="coerce").to_numpy()
                   if "tmduratn" in sel.columns else np.nan)
        out = out.dropna(subset=["rb"])

        # a monthly bond return sits near 0.5 percent. anything else means
        # _RET_SCALE has the wrong unit for this extract
        med = out["rb"].abs().median()
        if not 0.05 < med < 5.0:
            raise ValueError(
                f"{fname}: {ret_col} has median |return| {med:.4g} after "
                f"scaling; the unit assumption in _RET_SCALE is wrong here")

        return out                                    # first usable file wins

    return None                                       # no extract present


# ==========================================================================
# 3.  equity asset
# ==========================================================================

def build_gw_equity():
    """
    S&P 500 total return from Welch-Goyal (2008), the equity asset.

    Reads
    -----
    GoyalWelch_Monthly.csv, column `ret` (decimal, total return including
    dividends), monthly from 1926.

    Returns
    -------
    pd.Series
        Monthly PeriodIndex, percent per month, earned during month t.
    """
    gw = pd.read_csv(os.path.join(DATA_DIR, "GoyalWelch_Monthly.csv"),
                 usecols=["yyyymm", "ret"])          # ret = S&P 500 total return
    gw.index = pd.PeriodIndex(gw["yyyymm"].astype(int).astype(str), freq="M")  # 199001 -> Period
    gw.index.name = "month"
    return (100.0 * pd.to_numeric(gw["ret"], errors="coerce")   # decimal -> percent
            ).rename("re").dropna()


# ==========================================================================
# 4.  Transition drivers (modern panel only)
# ==========================================================================

def build_drivers():
    """
    Term spread, VIX and excess bond premium, each placed on the row of the
    month whose transition it drives

    Reads
    -----
    Driver_DGS3MO.csv, Driver_DGS10.csv, Driver_VIXCLS.csv   FRED, daily
    Driver_ebp_csv.csv                                       Board, monthly

    Returns
    -------
    pd.DataFrame
        Monthly PeriodIndex with z_term, z_vix and z_ebp in levels.
    """
    # daily -> last trading day of each month
    t3  = common.load_fred_daily("Driver_DGS3MO.csv", "DGS3MO").resample("ME").last()
    t10 = common.load_fred_daily("Driver_DGS10.csv",  "DGS10" ).resample("ME").last()
    vix = common.load_fred_daily("Driver_VIXCLS.csv", "VIXCLS").resample("ME").last()
    # term spread
    term = (t10 - t3).rename("term")          # 10y minus 3m constant-maturity yield

    # excess bond premium
    ebp_raw = pd.read_csv(os.path.join(DATA_DIR, "Driver_ebp_csv.csv"))
    ebp = ebp_raw.set_index(to_period(ebp_raw["date"]))["ebp"]   # already monthly

    # month-end timestamps -> monthly Period, the panel's key
    drv = pd.DataFrame({
        "z_term": term.groupby(term.index.to_period("M")).last(),
        "z_vix":  vix.groupby(vix.index.to_period("M")).last(),
    })
    drv.index.name = "month"

    drv.index = drv.index + 1          # observed end of t-1 -> row t
    ebp.index = ebp.index + 2          # one extra month for the publication lag
    drv["z_ebp"] = ebp.reindex(drv.index)   # NaN where the premium is unpublished
    return drv


# ==========================================================================
# 5.  ASSEMBLY
# ==========================================================================

def assemble(start, end, with_drivers, baseline_leg):
    """
    Build one panel: join the cash, bond and equity series by month, form
    their excess returns over the bill rate, add the ZLB flag and, for the
    modern panel, the transition covariates, then trim to [start, end].

    Parameters
    ----------
    start, end : str
        First and last month of the panel, e.g. "1952-01".
    with_drivers : bool
        True adds the transition covariates z_term, z_ebp and z_vix.  Used for
        the modern panel only.
    baseline_leg : {"10y", "long"}
        Bond series copied into `rb` and `rb_x`, the columns the estimator
        reads: "10y" (ten-year baseline) or "long" (twenty-five-year
        diagnostic).

    Returns
    -------
    pd.DataFrame
        One row per month, with a monthly PeriodIndex and the columns listed
        in `ordered` at the end of this function.  Both bond series are always
        present under fixed names (`rb_x_10y`, `rb_x_long`), so a column name
        means the same asset on either panel.

    Notes
    -----
    Before any excess return is formed, the index is checked for gaps, so no
    return can be paired with the bill rate of a different month.  `re_x` is
    the equity asset, the S&P 500 total return.
    """
    cash = build_cash()
    bond_long = build_bond(BOND_TARGET_LONG, "_long")
    bond_10y  = build_bond(BOND_TARGET_10Y,  "_10y")

    # inner joins trim the ends where a series has no data
    df = (cash.join(bond_long, how="inner")
            .join(bond_10y,  how="inner")
            .sort_index())

    # no gaps: a missing month would pair non-adjacent observations downstream
    expected = pd.period_range(df.index.min(), df.index.max(), freq="M")
    if not df.index.equals(expected):
        missing = expected.difference(df.index)
        raise ValueError(
            f"panel has {len(missing)} missing month(s), first: "
            f"{list(missing[:6])}; fix the source series before continuing.")

    # excess returns, all against the cash return earned over the same month
    df["rb_x_long"] = df["rb_long"] - df["r"]
    df["rb_x_10y"]  = df["rb_10y"]  - df["r"]
    
    df["re"]        = build_gw_equity().reindex(df.index)   # S&P 500 total return
    df["re_x"]      = df["re"]      - df["r"]               # equity excess return

    # baseline alias: everything downstream reads `rb` and `rb_x`
    if baseline_leg not in ("long", "10y"):
        raise ValueError(f"baseline_leg must be 'long' or '10y', got {baseline_leg!r}")
    df["rb"]   = df["rb_"   + baseline_leg]
    df["rb_x"] = df["rb_x_" + baseline_leg]
    df["z_short"] = df["r"]         # same rate, named for the transition

    if with_drivers:
        df = df.join(build_drivers(), how="left")   # modern panel only

    df = df.loc[start:end].copy()                   # trim to the panel window

    df["zlb"] = False                               # subperiod flag
    for a, b in ZLB_WINDOWS:
        df.loc[a:b, "zlb"] = True

    # Whitelist: anything not named here is dropped from the panel.
    ordered = ["obs_date", "r_obs", "r_obs_ann",
               "r", "r_ann", "r_hpr",
               "rb", "rb_x", "re", "re_x",
               "rb_long", "rb_x_long", "rb_10y", "rb_x_10y",
               "bond_ttm_long", "bond_id_long", "bond_dur_long",
               "bond_ttm_10y",  "bond_id_10y",  "bond_dur_10y",
               "bond_itype_long", "bond_itype_10y", "z_short", "z_term",
               "z_ebp", "z_vix", "zlb"]
    return df[[c for c in ordered if c in df.columns]]


# ==========================================================================
# 6.  DIAGNOSTICS
# ==========================================================================

def report(df, label, baseline_leg):
    """
    Print the diagnostics to inspect before a panel is used.

    Covers the return moments, the short rate in both units, the AR(1) fit of
    the two cash constructions, the realised maturity and callable share of
    each bond series, the return correlations and any
    missing values.  Several of these are the figures quoted in Chapter 5.

    Parameters
    ----------
    df : pd.DataFrame
        Panel from assemble().
    label : str
        Panel name, printed in the header.
    baseline_leg : {"10y", "long"}
        Bond series the panel uses as its baseline, marked in the output.

    Returns
    -------
    None
        Prints only.
    """
    print("=" * 74)
    print(f"{label}:  {df.index.min()} to {df.index.max()}   ({len(df)} months)")
    print(f"baseline bond series: '{baseline_leg}'  "
          f"(rb_x is an alias for rb_x_{baseline_leg})")
    print("=" * 74)

    core = [c for c in ("r", "rb_x_long", "rb_x_10y", "re_x") if c in df.columns]
    stats = pd.DataFrame({
        "mean": df[core].mean(),
        "sd":   df[core].std(),
        "min":  df[core].min(),
        "max":  df[core].max(),
    })
    print("\nReturn columns (percent per month):")
    print(stats.round(4).to_string())

    print("\nShort rate, annualised percent:\n"
          f"   min {df['r_ann'].min():.2f}, max {df['r_ann'].max():.2f}, "
          f"mean {df['r_ann'].mean():.2f}, sd {df['r_ann'].std():.2f}")

    # AR(1) on both cash constructions. Section 5.1 reads this against AB's
    # 0.99 and 0.94
    for col in ("r", "r_hpr"):
        if col not in df.columns or df[col].isna().any():
            continue                                                 # skip if incomplete
        y, x = df[col].to_numpy()[1:], df[col].to_numpy()[:-1]   # level on own lag
        X = np.column_stack([np.ones_like(x), x])                # constant + lag
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)             # OLS; coef[1] is rho
        sd = float(np.std(y - X @ coef, ddof=2))                 # residual sd, 2 params
        print(f"\nAR(1) on {col:6s}: rho {coef[1]:.3f}, residual sd {sd:.3f}")

    # Check whether each bond series stays pinned to its target maturity, or
    # only centres on it
    for leg, tgt in (("long", BOND_TARGET_LONG), ("10y", BOND_TARGET_10Y)):
        col = f"bond_ttm_{leg}"
        if col not in df.columns or df[col].isna().all():
            print(f"\nBond series '{leg}': no issue-level maturity to report")
            continue                                    # a constant-maturity index

        ttm = df[col]
        off = (ttm - tgt).abs() > BOND_PIN_TOLERANCE    # months outside the band
        flag = "PINNED" if not off.any() else f"drifts in {off.mean():.0%} of months"
        star = "  <-- baseline" if leg == baseline_leg else ""

        print(f"\nBond series '{leg}' (target {tgt:.0f}y){star}")
        print(f"   realised maturity: mean {ttm.mean():.2f}, "
              f"median {ttm.median():.2f}, "
              f"range {ttm.min():.2f} to {ttm.max():.2f}")
        print(f"   within +/-{BOND_PIN_TOLERANCE:.0f}y of target: {flag}")


        # ITYPE 5 is a callable bond, whose price embeds a short call option
        col_it = f"bond_itype_{leg}"
        if col_it in df.columns:
            cal = (df[col_it] == 5).to_numpy()
            print(f"   callable issue selected in {cal.mean():.0%} of months")
            if cal.any():
                print(f"   years with a callable pick: "
                      f"{sorted(set(df.index[cal].year))}")
                print(f"   last callable month: {df.index[cal].max()}")

    print("\nCorrelations of the return series:")
    print(df[core].corr().round(3).to_string())

    miss = df.isna().sum()          # should be empty on both panels
    miss = miss[miss > 0]
    print("\nMissing values:", "none" if miss.empty else f"\n{miss.to_string()}")
    print()


def gw_benchmark(repl):
    """
    Print both risky assets of the replication panel next to Ang and
    Bekaert's Table 2 and the series of Welch and Goyal (2008), 1952 to 2000.

    Their data carry the two series the note to Ang and Bekaert's Table 2
    names: Ibbotson's long-term government bond and the S&P 500 total return.
    Setting them against the constructed series identifies the bond Ang and
    Bekaert actually used (Section 5.2.2) and checks the equity asset
    (Section 5.3) directly, rather than inferring both from reported moments.

    Parameters
    ----------
    repl : pd.DataFrame
        The replication panel.
        
    Reads
    -----
    GoyalWelch_Monthly.csv
        ltr    Ibbotson long-term government bond total return
        ret    S&P 500 total return including dividends
    Both are decimals.  They are scaled to percent per month here and taken in
    excess of the same one-month bill the panel uses.

    Returns
    -------
    None
        Prints only.
    """
    gw = pd.read_csv(os.path.join(DATA_DIR, "GoyalWelch_Monthly.csv"))
    gw.index = pd.PeriodIndex(gw["yyyymm"].astype(int).astype(str), freq="M")
    gw = gw.loc[REPL_START:REPL_END]        # AB's window

    def excess(col):
        """
        One Welch-Goyal column as an excess return: decimal -> percent per
        month, minus the panel's bill rate `r` of the same month.  Uses `gw`
        and `repl` from the enclosing function.
        """
        s = 100.0 * pd.to_numeric(gw[col], errors="coerce")
        return (s - repl["r"]).dropna()     # same one-month bill as the panel

    ltr_x, spx_x = excess("ltr"), excess("ret")

    print("=" * 74)
    print("WELCH-GOYAL AGAINST BOTH RISKY ASSETS, 1952-2000")
    print("=" * 74)

    # bond: check if AB's series is the Ibbotson long-term bond 
    print("\nBond, percent per month:")
    print(f"{'':30s}{'mean':>10s}{'sd':>10s}{'corr(equity)':>14s}")
    ab_b = AB_TABLE2["excess bond"]               # de-annualised to monthly
    print(f"{'AB Table 2':30s}{ab_b['mean'] / 12:10.3f}"
          f"{ab_b['sd'] / 12 ** 0.5:10.3f}")
    for name, s in (("Ibbotson ltr",               ltr_x),
                    ("here, 10y series",           repl["rb_x_10y"]),
                    ("here, long series (25y)",    repl["rb_x_long"])):
        print(f"{name:30s}{s.mean():10.3f}{s.std():10.3f}"
              f"{repl.loc[s.index, 're_x'].corr(s):14.3f}")

    # check which series is ltr, as an asset
    for col in ("rb_x_10y", "rb_x_long"):
        j = repl[col].loc[ltr_x.index]
        print(f"\ncorr(ltr, {col:10s}) {ltr_x.corr(j):+.3f}   "
              f"sd ratio {ltr_x.std() / j.std():.3f}")
        
    # equity: the S&P 500 total return against AB's reported moments, with
    # the distance in standard errors (Section 5.3).  Under normality the
    # standard error of the mean is sd / sqrt(T) and that of the standard
    # deviation sd / sqrt(2T).
    ab_e = AB_TABLE2["excess stock"]
    m_ab, s_ab = ab_e["mean"] / 12, ab_e["sd"] / 12 ** 0.5
    T = len(spx_x)
    se_m, se_s = spx_x.std() / T ** 0.5, spx_x.std() / (2 * T) ** 0.5
    print("\nEquity, percent per month:")
    print(f"{'':30s}{'mean':>10s}{'sd':>10s}")
    print(f"{'AB Table 2':30s}{m_ab:10.3f}{s_ab:10.3f}")
    print(f"{'S&P 500 total return':30s}{spx_x.mean():10.3f}{spx_x.std():10.3f}")
    print(f"{'standard error, T = ' + str(T):30s}{se_m:10.3f}{se_s:10.3f}")
    print(f"{'distance in standard errors':30s}"
          f"{(spx_x.mean() - m_ab) / se_m:10.2f}"
          f"{(spx_x.std() - s_ab) / se_s:10.2f}")
    print()


def validate_fixed_term(df, label):
    """
    Print the constructed ten-year bond series next to the CRSP Fixed Term
    Index, the constant-maturity series build_bond() approximates.

    Compares mean return, volatility, maturity and duration over the months
    both cover, with the correlation and tracking error between the two.
    These are the figures of Section 5.2.1.

    Parameters
    ----------
    df : pd.DataFrame
        Panel from assemble().
    label : str
        Panel name, printed in the header.

    Returns
    -------
    None
        Prints only.  Prints nothing if no file holds the index, and skips
        the comparison if fewer than 24 months overlap.
    """
    ft = load_fixed_term("10-Year")
    if ft is None:
        return                                  # no extract, nothing to compare

    both = df.index.intersection(ft.index)      # the index starts 1961-06
    if len(both) < 24:
        print(f"\n{label}: Fixed Term overlap is {len(both)} months; skipping\n")
        return

    a, b = df.loc[both, "rb_10y"], ft.loc[both, "rb"]   # series vs index, same months
    a_ttm, b_ttm = df.loc[both, "bond_ttm_10y"].mean(), ft.loc[both, "bond_ttm"].mean()
    a_dur = df.loc[both, "bond_dur_10y"].mean() / 365.25    # CRSP reports days
    b_dur = ft.loc[both, "bond_dur"].mean() / 365.25

    print(f"\n{label}: issue-level 10y series against the CRSP Fixed Term Index")
    print(f"   {both.min()} to {both.max()}, {len(both)} months\n")

    print(f"{'':22s}{'series':>10s}{'index':>10s}{'ratio':>10s}")
    for name, x, y in (("mean return, %/mo", a.mean(),  b.mean()),
                       ("std dev, %/mo",     a.std(),   b.std()),
                       ("maturity, years",   a_ttm,     b_ttm),
                       ("duration, years",   a_dur,     b_dur)):
        print(f"   {name:19s}{x:10.3f}{y:10.3f}{x / y:10.3f}")

    print(f"\n   correlation {a.corr(b):+.4f}      "
          f"tracking error {(a - b).std():.3f} %/mo\n")


def verify_total_return():
    """
    Check that CRSP's issue-level return TMRETNUA is a total return, by
    rebuilding it from price and coupon.

    The gross price is the quoted price plus accrued interest, TMNOMPRC +
    TMACCINT.  A total return for month t is then
        (gross_t + coupon paid in t - gross_{t-1}) / gross_{t-1}.
    If TMRETNUA equals this for every issue-month, it includes the coupon
    and is a total return, not a price return (Section 5.2.1).

    Only consecutive months of the same issue are compared, so the first
    month of each issue drops out.  Runs over the eligible issues of
    _eligible_issues(), not every issue-month in the file.

    Returns
    -------
    None
        Prints the largest and the median discrepancy and the number of
        issue-months above 1e-6.
    """
    b = _eligible_issues().sort_values(["KYTREASNO", "obs_month"])
    gross = b["TMNOMPRC"] + b["TMACCINT"]
    coupon = b["TMPDINT"].fillna(0.0)

    ordinal = b["obs_month"].astype("int64")               # months since epoch
    prev_ord = ordinal.groupby(b["KYTREASNO"]).shift(1)
    prev_gross = gross.groupby(b["KYTREASNO"]).shift(1)

    adjacent = (ordinal - prev_ord) == 1                   # no gap within issue
    ok = adjacent & prev_gross.notna() & (prev_gross > 0)

    recon = (gross + coupon - prev_gross) / prev_gross
    d = (recon[ok] - b["TMRETNUA"][ok]).abs()
    print(f"\nTMRETNUA reconstruction over {ok.sum():,} issue-months")
    print(f"   max |discrepancy| {d.max():.2e}, "
          f"median {d.median():.2e}, above 1e-6 in {(d > 1e-6).sum()} cases")
    print()

def callable_coverage(start, end, target=25.0, band=BOND_PIN_TOLERANCE):
    """
    Count the months in which no eligible issue lies within `band` years of
    the target maturity, once with callable bonds admitted and once without.

    build_bond() imposes no maturity band: it always takes the closest
    issue, however far from the target.  `band` only defines when a month
    counts as covered.  The output supports two choices of Section 5.2.1.
    The count with callables admitted shows how many months a hard band of
    that width would leave without a bond, which is why none is imposed.  The
    count without callables shows how many more months would be lost by
    excluding them, which is why they are admitted.

    Parameters
    ----------
    start, end : str
        First and last month, e.g. "1952-01".
    target : float
        Target maturity in years, 10 or 25 in the calls below.
    band : float
        Distance in years within which an issue covers a month.  Defaults to
        BOND_PIN_TOLERANCE, one year.

    Returns
    -------
    None
        Prints the number of uncovered months under both filters.

    Notes
    -----
    Counted after the other filters of _eligible_issues(), which already drop
    flower bonds, partially tax-exempt issues and anything with less than six
    months to run.
    """
    elig = _eligible_issues()
    win = elig[(elig["obs_month"] >= pd.Period(start, "M"))
               & (elig["obs_month"] <= pd.Period(end, "M"))]

    months = pd.period_range(start, end, freq="M")
    near = win[(win["ttm"] - target).abs() <= band]      # close enough to count
    noncall = near[near["ITYPE"].isin((1, 2))]           # drop callable bonds

    n_all = near["obs_month"].nunique()
    n_nc = noncall["obs_month"].nunique()

    print(f"\nCoverage within +/-{band:.0f}y of {target:.0f} years, "
          f"{start} to {end} ({len(months)} months)")
    print(f"   callables admitted : {len(months) - n_all} months uncovered")
    print(f"   noncallable only   : {len(months) - n_nc} months uncovered")
    print()
    
# ==========================================================================
# 7.  MAIN
# ==========================================================================

if __name__ == "__main__":

    os.makedirs(OUT_DIR, exist_ok=True)
    
    verify_total_return()                       # is TMRETNUA a total return?

    # ---- replication panel ------------------------------------------------
    repl = assemble(REPL_START, REPL_END, with_drivers=False,
                    baseline_leg=REPL_BASELINE_LEG)
    report(repl, "REPLICATION PANEL (Ang-Bekaert 2004 sample)", REPL_BASELINE_LEG)
    gw_benchmark(repl)                          # against the Ibbotson and S&P series
    validate_fixed_term(repl, "REPLICATION PANEL")   # against the constant-maturity index
    callable_coverage(REPL_START, REPL_END, target=10.0)   # what a band at ten years would cost
    callable_coverage(REPL_START, REPL_END, target=25.0)   # justifies admitting callables
    repl.to_csv(os.path.join(OUT_DIR, "panel_replication.csv"))

    # ---- modern panel -----------------------------------------------------
    modern = assemble(MODERN_START, MODERN_END, with_drivers=True,
                      baseline_leg=MODERN_BASELINE_LEG)
    report(modern, "MODERN PANEL (specification ladder M0-M4)", MODERN_BASELINE_LEG)
    validate_fixed_term(modern, "MODERN PANEL")

    # the transition covariates standardisation happens inside the estimator
    print("Driver columns, levels:")
    zcols = [c for c in modern.columns if c.startswith("z_")]
    print(modern[zcols].describe().T[["mean", "std", "min", "max"]].round(3).to_string())
    print(f"\nZLB months flagged: {modern['zlb'].sum()} of {len(modern)}\n")

    modern.to_csv(os.path.join(OUT_DIR, "panel_modern.csv"))

    print(f"written: {os.path.join(OUT_DIR, 'panel_replication.csv')}")
    print(f"written: {os.path.join(OUT_DIR, 'panel_modern.csv')}")