"""
06_benchmark_check.py
=====================
Checks that the non-regime-dependent benchmark of 03_replication.py is the
benchmark Ang and Bekaert describe, by running it on their bond series
(Section 6.4, Table 6.4).

Their definition, from the published article: "a non-regime-dependent strategy
that incorporated means and covariances estimated from data up to time t".
The `un` strategy of part_b in 03 implements this.  The check is whether the
same benchmark, applied to their bond series, reproduces their reported
Sharpe ratio of 0.577.

The benchmark uses no regime model, so this runs in seconds.  It runs the
expanding-window benchmark, with an estimation window that starts in 1952,
on three bond series over the same evaluation window:

    rb_x_10y     the ten-year baseline series
    rb_x_long    the twenty-five-year diagnostic series
    ltr_x        the Ibbotson long-term government bond of Welch and Goyal,
                 which Section 5.2.2 identifies as the series they used

Prints, for each series, the bond Sharpe ratio over the full sample and over
the evaluation window, the mean bond weight at gamma = 5 and the realised
Sharpe ratio of the benchmark (Table 6.4).  The published row carries the
bond Sharpe ratio of Ang and Bekaert's Table 2, Panel B (2002), over 1952 to
2000, and their benchmark Sharpe ratio recomputed as in Section 6.3.  Then
the difference between the benchmark on the Ibbotson series and their
reported 0.577, as a fraction of the standard error of a single Sharpe
ratio over the evaluation window (Section 6.4).

verify_against_03() asserts that benchmark_path() reproduces the `un`
columns 03 recorded on the ten-year series to machine precision.
run_benchmark() is built on the same benchmark_path(), so the code run on
the three series is the benchmark of 03, not a separate implementation.

Reads panel_replication.csv, GoyalWelch_Monthly.csv, 03's backtest of the
ten-year series and Ang and Bekaert's published tables from 03
(AB_GAMMA_FREE).  Writes benchmark_check.csv.
"""

import os

import numpy as np
import pandas as pd

# 00_common.py cannot load itself through common.load_module, so it is
# loaded by path here.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "common", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "00_common.py"))
common = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(common)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
DATA_DIR  = common.DATA_DIR
OUT_DIR   = common.OUT_DIR

# Welch and Goyal predictor file, the same one 01_build_data.py reads.  The
# only column needed here is `ltr`, the Ibbotson long-term government bond
# total return, alongside the yyyymm date column.
GW_FILE   = os.path.join(DATA_DIR, "GoyalWelch_Monthly.csv")

OOS_START = "1985-01"
GAMMA     = 5
SPEC_TAG  = "v5"          # must match 03_replication.py, for the cross-check

# The three bond series, as (panel column, label)
SERIES = (("rb_x_10y",  "10y series (baseline)"),
          ("rb_x_long", "long series (25y)"),
          ("ltr_x",     "Ibbotson ltr (theirs)"))

repl = common.load_module("03_replication.py", "repl")   # Ang and Bekaert's published tables

# Ang and Bekaert's out-of-sample benchmark in gamma-free form, computed in
# 03 from their Table 8 (Section 6.3).
_ab = repl.AB_GAMMA_FREE.loc[("out-of-sample", "non-regime-dependent")]
AB_BENCH = {"B": float(_ab["B"]), "C": float(_ab["sd_gamma"])}
AB_BENCH["SR"] = AB_BENCH["B"] / AB_BENCH["C"]

# Bond Sharpe ratio of Ang and Bekaert's Table 2, Panel B (2002), 1952 to
# 2000: annualised mean over annualised standard deviation of the excess
# bond return.  Same values as AB_TABLE2 in 01_build_data.py.
AB_BOND_SR = 1.34 / 9.00


# ==========================================================================
# Helpers
# ==========================================================================
def load_panel():
    """
    Read the replication panel and append the Ibbotson excess return.

    Returns
    -------
    pd.DataFrame
        The panel of 01_build_data.py with one further column, `ltr_x`: the
        Ibbotson long-term government bond return in excess of the same
        one-month bill as the other two bond series, in percent per month.

    Raises
    ------
    SystemExit
        The Welch-Goyal file is missing or has no `ltr` column, the
        reconstructed Ibbotson excess return does not match the moments 01
        prints, or it has gaps over the replication window.

    Notes
    -----
    Row t of the panel holds `r`, the bill rate earned over month t, and the
    bond excess returns `rb_x_10y` and `rb_x_long` net of that rate.
    `ltr_x` is formed the same way, so all three bond series are measured
    against the same bill in the same month.
    """
    panel = common.read_panel("panel_replication.csv")

    if not os.path.exists(GW_FILE):
        raise SystemExit(f"missing {GW_FILE}; point GW_FILE at the file "
                         f"01_build_data.py reads")
    gw = pd.read_csv(GW_FILE)
    gw.columns = [str(c).strip().lower() for c in gw.columns]
    if "ltr" not in gw.columns:
        raise SystemExit(f"{os.path.basename(GW_FILE)} has no `ltr` column; "
                         f"columns are {list(gw.columns)}")
    datecol = next(c for c in gw.columns if c in ("yyyymm", "date", "month"))
    # the date column is a yyyymm integer, sometimes read back as a float
    gw.index = pd.PeriodIndex(
        pd.to_numeric(gw[datecol], errors="coerce").dropna().astype(int)
          .astype(str), freq="M")
    gw = gw.loc[~gw.index.duplicated()]
    ltr = pd.to_numeric(gw["ltr"], errors="coerce")
    ltr = ltr * 100.0      # Welch-Goyal stores decimals, the panel percent

    panel["ltr_x"] = ltr.reindex(panel.index) - panel["r"]

    # 01_build_data.py prints 0.122 and 2.597 for this series over the
    # replication window.  A mismatch means the timing or the unit is wrong,
    # and every number below would be wrong with it.
    m, s = panel["ltr_x"].mean(), panel["ltr_x"].std()
    if not (abs(m - 0.122) < 0.002 and abs(s - 2.597) < 0.005):
        raise SystemExit(f"ltr_x reconstructs to mean {m:.4f}, sd {s:.4f}; "
                         f"01 reports 0.122 and 2.597.  Check the unit and "
                         f"the timing before reading anything below.  A mean "
                         f"near 0.60 means the wrong column was picked up, "
                         f"and a mean near 0.0012 means the file was not in "
                         f"decimals.")
    if panel["ltr_x"].isna().any():
        raise SystemExit("ltr_x has gaps over the replication window")
    return panel


def benchmark_path(panel, bond_col):
    """
    Monthly weights and excess returns of the expanding-window benchmark on
    one bond series over the evaluation window.

    Parameters
    ----------
    panel : pd.DataFrame
        Output of load_panel().
    bond_col : str
        Column holding the bond excess return: "rb_x_10y", "rb_x_long" or
        "ltr_x".

    Returns
    -------
    pd.DataFrame
        One row per holding month from OOS_START, with the bond weight, the
        equity weight and the realised excess return as a decimal.

    Notes
    -----
    Built the same way as the `un` strategy of part_b in 03_replication.py.
    At each holding month h the benchmark uses the sample mean and
    covariance of the two excess returns over rows 0 to h-1, from the first
    month of the panel in 1952 up to the decision.  Nothing dated after the
    decision enters.
    """
    months = panel.index
    first = months.get_loc(pd.Period(OOS_START, freq="M"))
    x_all = panel[[bond_col, "re_x"]].to_numpy(dtype=float)   # percent

    rows = []
    for h in range(first, len(months)):
        # expanding window: all months before the holding month h
        win = x_all[:h]
        mu, S = win.mean(axis=0), np.cov(win.T, ddof=1)
        w = common.mv_weights(mu, S, GAMMA)
        rows.append({
            "month": months[h],
            "w_bond": w[0], "w_equity": w[1],
            "xs": float(w @ (x_all[h] / 100.0)),       # excess return, decimal
        })
    return pd.DataFrame(rows).set_index("month")


def run_benchmark(panel, bond_col):
    """
    Summarise the benchmark on one bond series over the evaluation window.

    Parameters
    ----------
    panel : pd.DataFrame
        Output of load_panel().
    bond_col : str
        Column holding the bond excess return.

    Returns
    -------
    dict
        B, C, SR : float
            Gamma-free form of the benchmark from common.gamma_free()
            (Appendix A.3), with SR = B / C.
        w_bond, w_equity : float
            Mean bond and equity weight at GAMMA.
        months : int
            Number of holding months.
    """
    bt = benchmark_path(panel, bond_col)
    B, C, SR = common.gamma_free(bt["xs"], GAMMA)
    return {
        "B": B, "C": C, "SR": SR,
        "w_bond": bt["w_bond"].mean(), "w_equity": bt["w_equity"].mean(),
        "months": len(bt),
    }


def verify_against_03(panel, bond_col="rb_x_10y", tag="10y"):
    """
    Check that benchmark_path() reproduces the benchmark 03 recorded.

    Parameters
    ----------
    panel : pd.DataFrame
        Output of load_panel().
    bond_col : str
        Panel column of the bond series 03 was run on.
    tag : str
        Label of that bond series in 03's file names.

    Returns
    -------
    bool
        True if the check ran and passed, False if 03's backtest is absent.

    Raises
    ------
    SystemExit
        The weights or the excess returns differ from 03's `un` columns by
        more than 1e-12.  This file is then not running the strategy 03
        runs, and nothing below can be read as a statement about it.
    """
    f = os.path.join(OUT_DIR, f"repl_backtest_{tag}_{SPEC_TAG}.csv")
    if not os.path.exists(f):
        print("   (03's backtest absent; cross-check skipped)")
        return False

    bt03 = common.read_backtest(os.path.basename(f))
    bt = benchmark_path(panel, bond_col)

    ref_W = common.weight_path(bt03, "un", GAMMA)
    # 03 records total returns, so the bill rate is taken off for the check
    ref_xs = (bt03[f"R_un_g{GAMMA}"] - bt03["rf"]).to_numpy(dtype=float)
    W = bt[["w_bond", "w_equity"]].to_numpy(dtype=float)
    if not (np.allclose(W, ref_W, atol=1e-12)
            and np.allclose(bt["xs"].to_numpy(dtype=float), ref_xs,
                            atol=1e-12)):
        raise SystemExit(
            "this benchmark does not reproduce the `un` columns of "
            f"{os.path.basename(f)}; the two implementations have diverged")
    print(f"   benchmark reproduces 03's `un` columns on the {tag} series "
          f"over {len(W)} months, to machine precision")
    return True


# ==========================================================================
# MAIN
# ==========================================================================
if __name__ == "__main__":
    panel = load_panel()
    verify_against_03(panel)
    win = panel.loc[pd.Period(OOS_START, "M"):]

    def ann_sr(x):
        """Annualised Sharpe ratio of a monthly excess return."""
        return x.mean() / x.std() * 12 ** 0.5

    # ---- Table 6.4 -------------------------------------------------------
    # Bond Sharpe ratio over the full sample and over the evaluation window,
    # then the mean bond weight and the realised Sharpe ratio of the
    # benchmark on each bond series.
    print("=" * 78)
    print("TABLE 6.4   the same benchmark code on each bond series")
    print("=" * 78)
    full_lab = f"{str(panel.index.min())[:4]}-{str(panel.index.max())[2:4]}"
    oos_lab = f"{OOS_START[2:4]}-{str(panel.index.max())[2:4]}"
    print(f"{'bond series':28s}{'SR ' + full_lab:>11s}{'SR ' + oos_lab:>10s}"
          f"{'w_bond':>9s}{'bench SR':>10s}")
    recs = {}
    for col, lab in SERIES:
        r = run_benchmark(panel, col)
        r["bond_SR_full"], r["bond_SR_oos"] = ann_sr(panel[col]), ann_sr(win[col])
        recs[lab] = r
        print(f"   {lab:25s}{r['bond_SR_full']:11.3f}{r['bond_SR_oos']:10.3f}"
              f"{r['w_bond']:9.3f}{r['SR']:10.3f}")
    # their bond Sharpe ratio over 1952 to 2000, Table 2, Panel B (2002).
    # They report none for the evaluation window or for the mean weight.
    print(f"   {'Ang-Bekaert, as published':25s}{AB_BOND_SR:11.3f}"
          f"{'--':>10s}{'--':>9s}{AB_BENCH['SR']:10.3f}")

    # ---- their series against their reported value (Section 6.4) ---------
    # The difference is compared with the standard error of a single
    # Sharpe ratio over the evaluation window, sqrt((1 + SR^2 / 2) / T) on
    # monthly values, annualised by sqrt(12).
    ibb = recs["Ibbotson ltr (theirs)"]
    T = ibb["months"]
    sr_m = ibb["SR"] / 12 ** 0.5
    se_one = np.sqrt((1 + sr_m ** 2 / 2) / T) * 12 ** 0.5
    diff = ibb["SR"] - AB_BENCH["SR"]
    print("\n" + "-" * 78)
    print(f"   Benchmark on their own bond series: {ibb['SR']:.3f}")
    print(f"   Benchmark as they report it:        {AB_BENCH['SR']:.3f}")
    print(f"   Difference:                         {diff:+.3f}")
    print(f"   s.e. of one Sharpe ratio, T = {T}:  {se_one:.3f}")
    print(f"   difference / s.e.:                  {diff / se_one:.3f}")

    pd.DataFrame(recs).T.to_csv(os.path.join(OUT_DIR, "benchmark_check.csv"))
    print(f"\nwritten: benchmark_check.csv in {OUT_DIR}")