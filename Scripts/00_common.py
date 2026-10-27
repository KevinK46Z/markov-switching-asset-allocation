"""
00_common.py
============
Shared configuration and helpers.  A module, not a script: every other file
imports from it, so a path, a grid or a statistic is defined once.

Holds the project paths, the risk-aversion grid, the strategy tags, the ZLB
windows, the figure palette, the module loader and the shared statistics.

Filenames starting with a digit are not legal module names, so load_module()
imports a script by path.  This runs the script's module level, which is
configuration and function definitions only, since every script guards its
work with __main__.
"""

import os
import sys
import importlib.util
import numpy as np
import pandas as pd
from scipy.stats import norm


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# The project root is the parent of the folder holding this file, so a clone
# runs wherever it is placed.  MT_PROJECT overrides it when the data sits
# elsewhere
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT    = os.environ.get("MT_PROJECT", os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(PROJECT, "Data")
OUT_DIR  = os.path.join(PROJECT, "Output")

# --------------------------------------------------------------------------
# Shared settings
# --------------------------------------------------------------------------
GAMMAS = [2, 3, 4, 5, 10]     # risk aversion grid of Ang-Bekaert Table 8 (2002)

# Column tag and printed name of every strategy, used in every file
STRATEGIES = [("rs", "regime-dependent (hard)"),
              ("sf", "regime-dependent (soft)"),
              ("un", "non-regime-dependent")]

# Zero lower bound windows: Fed funds target range at 0-0.25 percent.  
# 01 writes these to the panel as `zlb`, the figures shade the same months.
ZLB_WINDOWS = [("2008-12", "2015-12"), ("2020-03", "2022-03")]

# One colour per rung, so a rung is the same colour in every figure
RUNG_COLOURS = {"M0": "#0072B2",     # blue
                "M1": "#E69F00",     # orange
                "M2": "#D55E00",     # vermillion
                "M3": "#009E73",     # green
                "M4": "#CC79A7"}     # purple
STATIC_COLOUR = "black"              # the benchmark, always dashed
# One colour per allocation rule, for figures that set the two rules of one
# model against each other.  Hard takes M1's colour, since the replication
# model is M1.  Soft takes the Okabe-Ito sky blue, which no rung uses.
RULE_COLOURS = {"hard": RUNG_COLOURS["M1"],   # orange
                "soft": "#56B4E9"}            # sky blue

# Figure style, applied by every plotting block.
PLOT_RC = {"font.family": "serif", "font.size": 9, "axes.linewidth": 0.6,
           "xtick.direction": "in", "ytick.direction": "in"}


# ==========================================================================
# Module loading
# ==========================================================================
def load_module(filename, alias):
    """
    Import a script by path under a legal module name

    Parameters
    ----------
    filename : str
        File in SCRIPT_DIR, e.g. "02_msm.py".
    alias : str
        Name to register it under in sys.modules.

    Returns
    -------
    module
        The module already registered under `alias` if there is one, so two
        scripts loading the same file in one process share one copy.
    """
    if alias in sys.modules:
        return sys.modules[alias]                 # already loaded this process
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    spec = importlib.util.spec_from_file_location(
        alias, os.path.join(SCRIPT_DIR, filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod

# ==========================================================================
# Panel access
# ==========================================================================
def read_panel(filename):
    """
    Read a panel written by 01_build_data.py.

    Parameters
    ----------
    filename : str
        File in DATA_DIR, e.g. "panel_replication.csv".

    Returns
    -------
    pd.DataFrame
        Monthly PeriodIndex.

    Raises
    ------
    SystemExit
        The file does not exist.
    """
    f = os.path.join(DATA_DIR, filename)
    if not os.path.exists(f):
        raise SystemExit(f"missing {f}; run 01_build_data.py")
    panel = pd.read_csv(f, index_col=0)
    # The CSV stores the month as text ("1952-01").  Convert it back to a
    # monthly PeriodIndex, so rows can be selected by month and shifted by
    # whole months, and so the index matches the other panels and backtests.
    panel.index = pd.PeriodIndex(panel.index, freq="M")
    return panel


def read_backtest(filename):
    """
    Read a backtest written by 03 or 07

    Parameters
    ----------
    filename : str
        File in OUT_DIR.

    Returns
    -------
    pd.DataFrame
        The backtest with its index converted back to monthly periods.  If
        the file has a `zlb` column (the ladder backtests of 07 do, those of
        03 do not), it is converted back to True/False, since a CSV stores it
        as text.

    Raises
    ------
    SystemExit
        The file does not exist.
    """
    f = os.path.join(OUT_DIR, filename)
    if not os.path.exists(f):
        raise SystemExit(f"missing {f}; run the script that writes it")
    bt = pd.read_csv(f, index_col=0)
    bt.index = pd.PeriodIndex(bt.index, freq="M")
    if "zlb" in bt.columns:
        bt["zlb"] = bt["zlb"].astype(bool)
    return bt


def promote_leg(panel, col):
    """
    Copy of the panel with `rb_x` set to the requested bond series.

    `rb_x` is the excess bond return the estimator and every allocation rule
    read.  The panel carries two bond series under fixed names,
    `rb_x_10y` (the ten-year baseline) and `rb_x_long` (the twenty-five-year
    diagnostic).  This function copies the chosen one into `rb_x`, so the
    same code runs on either series.

    Parameters
    ----------
    panel : pd.DataFrame
        Panel carrying both bond columns.
    col : str
        Column to copy into `rb_x`: "rb_x_10y" (ten-year baseline) or
        "rb_x_long" (twenty-five-year diagnostic).

    Returns
    -------
    pd.DataFrame
        A copy of the panel.  The input panel is left unchanged.

    Notes
    -----
    Everything downstream reads `rb_x`, so switching the series is one column
    swap rather than a parameter threaded through every function.
    """
    out = panel.copy()
    out["rb_x"] = panel[col]          # promote the requested series
    return out


def weight_path(bt, strat, gamma):
    """
    Target weight path of one strategy at one gamma

    Parameters
    ----------
    bt : pd.DataFrame
        A backtest written by 03 or 07.
    strat : str
        Strategy tag: "rs" (regime-dependent, hard classification), "sf"
        (regime-dependent, soft classification) or "un" (non-regime-dependent
        benchmark).
    gamma : int
        Risk aversion, one of 2, 3, 4, 5 or 10.

    Returns
    -------
    np.ndarray, shape (T, 2)
        Weights on (bond, equity), one row per holding month.
    """
    return bt[[f"w_bond_{strat}_g{gamma}",
               f"w_equity_{strat}_g{gamma}"]].to_numpy(dtype=float)


def load_fred_daily(filename, colname):
    """
    Read a daily series downloaded from FRED and return it as a clean series.

    A FRED download has two columns, the date and the value.  Days without an
    observation, such as holidays, are marked with "." instead of a number.
    This function renames the value column, parses the dates, turns "." into
    a missing value and drops those days, so only actual observations remain.

    Parameters
    ----------
    filename : str
        File in DATA_DIR.
    colname : str
        Name given to the value column.

    Returns
    -------
    pd.Series
        DatetimeIndex, gaps dropped.
    """
    d = pd.read_csv(os.path.join(DATA_DIR, filename))
    d.columns = ["date", colname]
    d["date"] = pd.to_datetime(d["date"])
    d[colname] = pd.to_numeric(d[colname], errors="coerce")
    return d.dropna().set_index("date")[colname]


# ==========================================================================
# Portfolio machinery
# ==========================================================================
def mv_weights(mu_pct, Sigma_pct, gamma):
    """
    Single-period mean-variance weights, Ang-Bekaert (2002) equation (10).

    Parameters
    ----------
    mu_pct : np.ndarray, length 2
        Expected excess return on (bond, equity), percent per month.
    Sigma_pct : np.ndarray, shape (2, 2)
        Conditional covariance, percent-squared per month.  The mixture, not
        the within-regime block.
    gamma : float
        Risk aversion.

    Returns
    -------
    np.ndarray, length 2
        Weights on (bond, equity).  Cash takes the residual 1 - w_b - w_e, so
        a risky total above one is a borrowed position.  No constraint is
        imposed, following Ang and Bekaert.

    Notes
    -----
    w = Sigma^-1 mu / gamma holds in decimals, so the percent inputs bring the
    factor 100/gamma.
    """
    return (100.0 / gamma) * np.linalg.solve(Sigma_pct, mu_pct)


def performance(R, rf, label):
    """
    Performance statistics for a monthly total-return series.

    Parameters
    ----------
    R : array_like
        Monthly total return, decimal.
    rf : array_like
        Monthly risk-free rate, decimal.
    label : str
        Printed name of the strategy, copied into the output row:
        "regime-dependent (hard)", "regime-dependent (soft)" or
        "non-regime-dependent".

    Returns
    -------
    dict
        mean_ann_%, mean_cagr_%, sd_ann_%, sharpe_ann, sharpe_excess,
        final_wealth and min_month_%.

    Notes
    -----
    mean_ann_% is twelve times the arithmetic monthly mean, the construction
    Ang and Bekaert use.  mean_cagr_% is NaN if leverage wipes the portfolio
    out.
    """
    R  = np.asarray(R, dtype=float)
    rf = np.asarray(rf, dtype=float)
    T  = len(R)

    growth = np.prod(1.0 + R)                  # wealth multiple over the window
    cagr = np.nan if growth <= 0 else growth ** (12.0 / T) - 1.0

    mean_ann = 12.0 * R.mean()                 # arithmetic, as in AB
    rf_ann   = 12.0 * rf.mean()
    sd_ann   = np.sqrt(12.0) * R.std(ddof=1)   # ddof=1: divide by N-1
    xs       = R - rf                          # excess over cash
    return {
        "strategy":      label,
        "mean_ann_%":    100.0 * mean_ann,
        "mean_cagr_%":   100.0 * cagr,         # compound, for reference
        "sd_ann_%":      100.0 * sd_ann,
        "sharpe_ann":    (mean_ann - rf_ann) / sd_ann,
        "sharpe_excess": np.sqrt(12.0) * xs.mean() / xs.std(ddof=1),
        "final_wealth":  growth,
        "min_month_%":   100.0 * R.min(),
    }


# ==========================================================================
# Statistics
# ==========================================================================
def sharpe(xs):
    """
    Annualised Sharpe ratio of a monthly excess-return series.

    Parameters
    ----------
    xs : array_like
        Monthly excess return, decimal.

    Returns
    -------
    float
        NaN if the series has no variation (avoid dividing by 0)
    """
    xs = np.asarray(xs, dtype=float)
    sd = xs.std(ddof=1)
    return np.sqrt(12.0) * xs.mean() / sd if sd > 0 else np.nan


def gamma_free(xs, gamma):
    """
    Gamma-free summary of a monthly excess-return series held at `gamma`. The 
    decomposition is derived in Appendix A.3 of the thesis.

    Parameters
    ----------
    xs : array_like
        Monthly excess return, decimal.
    gamma : float
        Risk aversion the series was generated at.

    Returns
    -------
    B : float
        Annualised mean excess return times gamma, percent.
    C : float
        Annualised standard deviation of the excess return times gamma,
        percent.
    SR : float
        B/C, the Sharpe ratio the rule delivers at every gamma.

    Notes
    -----
    Used for every backtest.  gamma_decomposition() estimates the same
    constants from a grid of total returns, for published tables that report
    no return series.  Its B is identical, but its C differs slightly, because
    the total return includes the bill rate, which varies from month to month.
    """
    xs = np.asarray(xs, dtype=float)
    # x 12 annualises the monthly mean, x 100 turns decimals into percent
    B = 1200.0 * gamma * xs.mean()
    # a standard deviation annualises with sqrt(12), not 12
    C = gamma * 100.0 * np.sqrt(12.0) * xs.std(ddof=1)
    return B, C, B / C


def gamma_decomposition(means, sds, gammas):
    """
    The decomposition of gamma_free(), estimated from a table of annualised
    total returns across a grid of risk aversions. A and B are fitted by 
    regressing the means on 1/gamma, and C is the average of sd * gamma over 
    the grid.  This is the only way to obtain B and C for a published table, 
    such as Ang and Bekaert's, that reports total returns at several gammas but
    no return series. The decomposition is derived in Appendix A.3 of the 
    thesis.

    Parameters
    ----------
    means, sds : array_like
        Annualised mean and standard deviation of the total return, one per
        gamma, in percent.
    gammas : array_like
        Risk aversion grid.

    Returns
    -------
    A : float
        Intercept, which should equal the mean annualised bill rate.
    B : float
        Slope on 1/gamma, the annualised mean excess return times gamma.
    max_abs_resid : float
        Largest fitted residual of the regression.  Near zero confirms that
        the mean is linear in 1/gamma, as the unconstrained rule implies.
    mean_sd_gamma : float
        Average of sd * gamma over the grid.  Not exactly constant when the
        bill rate varies, and not checked here.
    sharpe_invariant : float
        B / mean_sd_gamma.  Agrees with the excess-return Sharpe ratio of
        gamma_free() to within about 0.001.
    """
    g = np.asarray(gammas, dtype=float)
    m = np.asarray(means, dtype=float)
    s = np.asarray(sds, dtype=float)
    X = np.column_stack([np.ones_like(g), 1.0 / g])        # regress on 1/gamma
    A, B = np.linalg.lstsq(X, m, rcond=None)[0]
    resid = float(np.abs(m - X @ np.array([A, B])).max())  # linearity check
    sd_gamma = float((s * g).mean())                       # average over the grid
    return float(A), float(B), resid, sd_gamma, float(B / sd_gamma)


def sharpe_diff_test(xs1, xs2):
    """
    Memmel (2003) correction of the Jobson-Korkie test for two Sharpe ratios
    estimated on the same sample.

    Parameters
    ----------
    xs1, xs2 : array_like
        Monthly excess returns of the two strategies, same months.

    Returns
    -------
    d : float
        Annualised Sharpe difference.
    se : float
        Annualised standard error of that difference.
    t : float
        The Jobson-Korkie statistic with the variance as corrected by Memmel
        (2003), d / se (Appendix A.7 of the thesis).  Compared with the
        standard normal, so 1.645, 1.96 and 2.576 are the two-sided 10, 5 and
        1 percent thresholds.
    p : float
        Two-sided normal p-value.
    rho : float
        Correlation of the two return series.

    Notes
    -----
    Checked against simulation in Appendix A.7.  The formula understates the
    standard error by 0.6 to 0.7 percent at T = 192, differs by at most 0.1
    percent at T = 587, and understates by 2.5 percent at T = 60.
    """
    x1 = np.asarray(xs1, dtype=float)
    x2 = np.asarray(xs2, dtype=float)
    T = len(x1)
    s1 = x1.mean() / x1.std(ddof=1)              # monthly Sharpe ratios
    s2 = x2.mean() / x2.std(ddof=1)
    rho = float(np.corrcoef(x1, x2)[0, 1])
    # Variance of the estimated Sharpe difference, Memmel's correction of
    # Jobson-Korkie (Appendix A.7)
    var = (1.0 / T) * (2.0 - 2.0 * rho + 0.5 * (s1 ** 2 + s2 ** 2)
                       - s1 * s2 * rho ** 2)
    se = np.sqrt(max(var, 0.0))
    t = (s1 - s2) / se if se > 0 else np.nan
    # two-sided p-value: probability of a |t| at least this large under the
    # standard normal. missing if t could not be computed
    p = float(2.0 * (1.0 - norm.cdf(abs(t)))) if np.isfinite(t) else np.nan
    # difference and standard error annualised by sqrt(12). t, p and rho
    # have no time unit
    return (np.sqrt(12.0) * (s1 - s2), np.sqrt(12.0) * se, float(t), p, rho)


def sharpe_diff_from_moments(sr1, sr2, rho, T):
    """
    Memmel (2003) test from two annualised Sharpe ratios and their
    correlation, for published results that report no return series.

    Parameters
    ----------
    sr1, sr2 : float
        Annualised Sharpe ratios.
    rho : float
        Correlation of the two return series, taken from a comparable pair.
    T : int
        Months.

    Returns
    -------
    d, se, t, p : float
        Annualised difference, its annualised standard error, the t
        statistic and the two-sided normal p-value.
    """
    s1, s2 = sr1 / np.sqrt(12.0), sr2 / np.sqrt(12.0)
    var = (2.0 - 2.0 * rho + 0.5 * (s1 ** 2 + s2 ** 2)
           - s1 * s2 * rho ** 2) / T
    se = np.sqrt(var)
    t = (s1 - s2) / se
    p = float(2.0 * (1.0 - norm.cdf(abs(t))))
    return np.sqrt(12.0) * (s1 - s2), np.sqrt(12.0) * se, float(t), p


def shade_zlb(ax, dates, zlb):
    """
    Grey bands over the flagged months of a time-series axis.

    Parameters
    ----------
    ax : matplotlib axis
    dates : array_like
        Timestamps, one per month.
    zlb : array_like of bool
        Flag, same length as `dates`.

    Returns
    -------
    None
    """
    run = None
    for i, v in enumerate(np.asarray(zlb, dtype=bool)):
        if v and run is None:
            run = i
        if (not v or i == len(zlb) - 1) and run is not None:
            ax.axvspan(dates[run], dates[i], color="0.90", lw=0, zorder=0)
            run = None