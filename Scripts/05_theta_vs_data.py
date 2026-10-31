"""
05_theta_vs_data.py
===================
Tests whether the shortfall against Ang and Bekaert lies in the estimated
parameters or in the realised returns (Section 6.5, Table 6.6).  Run after
03_replication.py and before 04_diagnostics.py.  One filter pass per
configuration, nothing is re-estimated.

The experiment.  Ang and Bekaert's published Table 6 vector is fed into the
same allocation on the same realised returns as the fitted vector.  If their
parameters recover their Sharpe ratio, the difference lies in the estimates.
If they do not, it lies in the return series, and no change of specification
closes it.

Sample.  The full-sample rows use full-sample parameters and full-sample
benchmark moments.  They therefore contain look-ahead by construction, as
Ang and Bekaert's in-sample table does.  They are a specification check, not
a strategy.

Reads:
    panel_replication.csv
    repl_M1_theta_<tag>_<SPEC_TAG>.csv written by 03, one per bond series
    Ang and Bekaert's Model II vector, from 02 (AB_MODEL_II_THETA), and
        their published tables in gamma-free form, from 03 (AB_GAMMA_FREE)

Prints, for each bond series:
    the Sharpe ratios and the Memmel tests of four configurations, the
        fitted and the published vector, each over the full sample and over
        1985 to 2000 (Table 6.6)
    Ang and Bekaert's printed Sharpe ratios and gaps (Table 6.6)
    the mean regime path and positions, among them the mean bond weight of
        the benchmark 
    the test of Ang and Bekaert's in-sample gap at the correlation and
        sample length measured here (Table 6.6)

Writes repl_param_vs_data_<tag>_<SPEC_TAG>.csv per bond series.  04 reads
its row "fitted theta, full sample" for the in-sample configurations of the
simulation in Appendix A.7.
"""

import os

import numpy as np
import pandas as pd

# 00_common.py cannot load itself through common.load_module, so it is
# loaded by path here.  Every other module goes through common.load_module.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "common", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "00_common.py"))
common = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(common)


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
OUT_DIR    = common.OUT_DIR
STRATEGIES = common.STRATEGIES

SPEC_TAG  = "v5"               # must match 03_replication.py
BOND_LEGS = [("10y", "rb_x_10y"), ("long", "rb_x_long")]
GAMMA     = 5
OOS_START = "1985-01"          # first month of the out-of-sample rows

msm  = common.load_module("02_msm.py", "msm")            # the estimator
repl = common.load_module("03_replication.py", "repl")   # Ang and Bekaert's published tables

# Ang and Bekaert in gamma-free form, as (B, C), Model II against the naive
# strategy.  Computed in 03 from their printed Tables 7 (in-sample) and 8
# (out-of-sample).
AB = {(s, n): (repl.AB_GAMMA_FREE.loc[(s, n), "B"],
               repl.AB_GAMMA_FREE.loc[(s, n), "sd_gamma"])
      for s in ("in-sample", "out-of-sample")
      for n in ("regime-dependent", "non-regime-dependent")}

# Their Sharpe ratios, SR = B / C, and the gap of the regime-dependent
# strategy against the benchmark, per sample.
AB_SR  = {key: B / C for key, (B, C) in AB.items()}
AB_GAP = {s: AB_SR[(s, "regime-dependent")] - AB_SR[(s, "non-regime-dependent")]
          for s in ("in-sample", "out-of-sample")}



# ==========================================================================
# Allocation with a fixed parameter vector
# ==========================================================================
def allocate(panel, theta, mu_bench, Sig_bench):
    """
    Run all three allocation rules over the panel with a fixed theta.

    Parameters
    ----------
    panel : pd.DataFrame
        Replication panel with `rb_x` set to the chosen bond series, as
        returned by common.promote_leg().
    theta : array_like, length 22
        Parameter vector for k = 1.
    mu_bench : np.ndarray, length 2
        Benchmark mean excess return on (bond, equity), percent per month.
    Sig_bench : np.ndarray, shape (2, 2)
        Benchmark covariance, percent squared per month.

    Returns
    -------
    pd.DataFrame
        One row per holding month, with a monthly PeriodIndex.  Columns: the
        filtered probability of regime 1 known at the decision, the
        classified regime (1 or 2), the mixing weight pi1 under each rule,
        the bond and equity weight of each strategy, and the realised excess
        return of each strategy as a decimal.

    Raises
    ------
    SystemExit
        The filter returns a different number of rows than the panel has,
        which means 02_msm.prepare() is not the current version.

    Notes
    -----
    prepare() drops no row, so res.xi_filt[i] is the filtered probability of
    panel.index[i].  Holding month j is decided at the end of month j-1, so
    the loop starts at j = 1 and reads res.xi_filt[j-1].  This is the timing
    of 03.  The benchmark weights are constant, so they are formed once
    outside the loop.
    """
    res = msm.evaluate(panel, ["z_short"], theta)
    if len(res.xi_filt) != len(panel):
        raise SystemExit(f"xi_filt has {len(res.xi_filt)} entries for a "
                         f"{len(panel)}-row panel; 02_msm.prepare() is stale")

    x = panel[["rb_x", "re_x"]].to_numpy(dtype=float) / 100.0
    z = panel[["z_short"]].to_numpy(dtype=float)
    w_un = common.mv_weights(mu_bench, Sig_bench, GAMMA)

    recs = []
    for j in range(1, len(panel)):
        xi_d = float(res.xi_filt[j - 1])
        # hard rule: classify into the more likely regime, 0-based here
        regime = 0 if xi_d > 0.5 else 1
        mu_h, Sig_h, pi1_h, _, _ = msm.conditional_moments(
            res.params, z[j], regime=regime)
        # soft rule: mix the two regimes with the filtered probability
        mu_s, Sig_s, pi1_s, _, _ = msm.conditional_moments(
            res.params, z[j], xi=xi_d)
        w = {"rs": common.mv_weights(mu_h, Sig_h, GAMMA),
             "sf": common.mv_weights(mu_s, Sig_s, GAMMA),
             "un": w_un}
        row = {"month": panel.index[j], "xi_filt_d": xi_d,
               "regime": regime + 1, "pi1_h": pi1_h, "pi1_s": pi1_s}
        for s, _ in STRATEGIES:
            row[f"w_bond_{s}"]   = w[s][0]
            row[f"w_equity_{s}"] = w[s][1]
            row[f"xs_{s}"]       = float(w[s] @ x[j])
        recs.append(row)

    out = pd.DataFrame(recs).set_index("month")
    out.index = pd.PeriodIndex(out.index, freq="M")
    return out


def summarise(alloc, label, window):
    """
    Gamma-free summary of one allocation over one window, with the tests.

    Parameters
    ----------
    alloc : pd.DataFrame
        Output of allocate().
    label : str
        Configuration name, carried into the output row.
    window : tuple of str
        First and last month, inclusive, as "YYYY-MM".

    Returns
    -------
    dict
        For each strategy B, C and SR (common.gamma_free, Appendix A.3).
        For the three pairs hard-benchmark ("hb"),
        soft-benchmark ("sb") and hard-soft ("hs") the Memmel test: gap,
        standard error, t statistic and correlation.  Then the share of
        months classified regime 1, the mean pi1 under each rule and the
        mean weights.
    """
    # restrict the allocation to the months of the window, both ends included
    m = ((alloc.index >= pd.Period(window[0], "M"))
         & (alloc.index <= pd.Period(window[1], "M")))
    a = alloc.loc[m]

    rec = {"config": label, "months": int(m.sum())}

    # B, C and the Sharpe ratio of each strategy (common.gamma_free)
    for s, _ in STRATEGIES:
        B, C, SR = common.gamma_free(a[f"xs_{s}"], GAMMA)
        rec[f"B_{s}"], rec[f"C_{s}"], rec[f"SR_{s}"] = B, C, SR

    # Memmel tests of the three pairs: hard against benchmark ("hb"), soft
    # against benchmark ("sb"), hard against soft ("hs").  Each gap is the
    # first strategy's Sharpe ratio minus the second's.  The p-value is not
    # stored.
    for a1, a2, key in (("rs", "un", "hb"), ("sf", "un", "sb"),
                        ("rs", "sf", "hs")):
        d, se, t, _, rho = common.sharpe_diff_test(a[f"xs_{a1}"],
                                                   a[f"xs_{a2}"])
        rec[f"gap_{key}"], rec[f"se_{key}"] = d, se
        rec[f"t_{key}"], rec[f"rho_{key}"] = t, rho

    # Regime path: share of months the hard rule classifies as regime 1, and
    # the mean mixing weight pi1 under each rule
    rec["regime1_share"] = float((a["regime"] == 1).mean())
    rec["pi1_h_mean"] = float(a["pi1_h"].mean())
    rec["pi1_s_mean"] = float(a["pi1_s"].mean())

    # mean bond and equity weight of each strategy over the window
    for s, _ in STRATEGIES:
        rec[f"w_eq_{s}_mean"] = float(a[f"w_equity_{s}"].mean())
        rec[f"w_bd_{s}_mean"] = float(a[f"w_bond_{s}"].mean())
    return rec


def load_theta(tag):
    """
    Read the fitted M1 vector that 03 saved for one bond series.

    Parameters
    ----------
    tag : str
        Label of the bond series, "10y" or "long".

    Returns
    -------
    np.ndarray, length 22

    Raises
    ------
    SystemExit
        The SPEC_TAG-stamped file is missing.
    """
    f = os.path.join(OUT_DIR, f"repl_M1_theta_{tag}_{SPEC_TAG}.csv")
    if not os.path.exists(f):
        raise SystemExit(f"missing {f}; run 03_replication.py at SPEC_TAG "
                         f"{SPEC_TAG}")
    return pd.read_csv(f)["M1"].to_numpy(dtype=float)


# ==========================================================================
# One bond series
# ==========================================================================
def run_leg(tag, col):
    """
    Run the parameter-against-data experiment for one bond series.

    Parameters
    ----------
    tag : str
        Label of the bond series, "10y" or "long".
    col : str
        Panel column holding that series' excess return, "rb_x_10y" or
        "rb_x_long".

    Returns
    -------
    pd.DataFrame
        The four configuration rows, indexed by config.  Also written to
        repl_param_vs_data_<tag>_<SPEC_TAG>.csv.

    Raises
    ------
    SystemExit
        The fitted vector has the wrong length for k = 1.
    """
    print("=" * 78)
    print(f"BOND SERIES '{tag}'")
    print("=" * 78)

    panel = common.promote_leg(common.read_panel("panel_replication.csv"), col)

    theta_here = load_theta(tag)
    theta_ab   = np.asarray(msm.AB_MODEL_II_THETA, dtype=float)
    if len(theta_here) != len(theta_ab):
        raise SystemExit(f"fitted theta has length {len(theta_here)}, "
                         f"expected {len(theta_ab)} for k = 1")

    # full-sample benchmark moments, as in their in-sample table
    xw = panel[["rb_x", "re_x"]].to_numpy(dtype=float)
    mu_bench, Sig_bench = xw.mean(axis=0), np.cov(xw.T, ddof=1)

    alloc_here = allocate(panel, theta_here, mu_bench, Sig_bench)
    alloc_ab   = allocate(panel, theta_ab,   mu_bench, Sig_bench)

    # The out-of-sample rows evaluate the same full-sample allocation over
    # the window of 03's backtest.  The parameters are not re-estimated.
    full = (str(panel.index.min()), str(panel.index.max()))
    oos  = (OOS_START, str(panel.index.max()))
    # window label taken from OOS_START, so it cannot drift from the setting
    oos_lab = f"{OOS_START[:4]}-{str(panel.index.max())[:4]}"
    tab = pd.DataFrame([
        summarise(alloc_here, "fitted theta, full sample",         full),
        summarise(alloc_ab,   "AB Table 6 theta, full sample",     full),
        summarise(alloc_here, f"fitted theta, {oos_lab}",          oos),
        summarise(alloc_ab,   f"AB Table 6 theta, {oos_lab}",      oos),
    ]).set_index("config")

    print("\n--- gamma-free summary " + "-" * 55)
    print(tab[["months", "SR_rs", "SR_sf", "SR_un"]].round(3).to_string())

    print("\n--- Memmel (2003) tests " + "-" * 54)
    print(f"{'':34s}{'h-b':>8s}{'t':>6s}{'s-b':>8s}{'t':>6s}"
          f"{'h-s':>8s}{'t':>6s}{'rho(h,b)':>10s}{'months':>8s}")
    for cfg in tab.index:
        r = tab.loc[cfg]
        print(f"{cfg:34s}{r['gap_hb']:8.3f}{r['t_hb']:6.2f}"
              f"{r['gap_sb']:8.3f}{r['t_sb']:6.2f}"
              f"{r['gap_hs']:8.3f}{r['t_hs']:6.2f}"
              f"{r['rho_hb']:10.4f}{int(r['months']):8d}")

    print("\n--- Ang and Bekaert, printed tables " + "-" * 42)
    for s in ("in-sample", "out-of-sample"):
        print(f"   {s:14s} SR_rs {AB_SR[(s, 'regime-dependent')]:6.3f}"
              f"   SR_un {AB_SR[(s, 'non-regime-dependent')]:6.3f}"
              f"   gap {AB_GAP[s]:+6.3f}")

    # ---- regime path and average positions -------------------------------
    print("\n--- regime path and average positions " + "-" * 40)
    print(tab[["regime1_share", "pi1_h_mean", "pi1_s_mean",
               "w_eq_rs_mean", "w_bd_rs_mean", "w_eq_sf_mean",
               "w_bd_sf_mean", "w_eq_un_mean", "w_bd_un_mean"]]
          .round(3).to_string())
    # Ang and Bekaert's in-sample gap tested at this bond series'
    # correlation and sample length (Table 6.6).  They report no return
    # series, so the test runs on the moments.
    r = tab.loc["fitted theta, full sample"]
    _, se, t, p = common.sharpe_diff_from_moments(
        AB_SR[("in-sample", "regime-dependent")],
        AB_SR[("in-sample", "non-regime-dependent")],
        float(r["rho_hb"]), int(r["months"]))
    print(f"   AB in-sample gap at rho {r['rho_hb']:.3f}, T {int(r['months'])}: "
          f"s.e. {se:.3f}, t {t:.2f}, p {p:.3f}\n")

    tab.to_csv(os.path.join(OUT_DIR,
                            f"repl_param_vs_data_{tag}_{SPEC_TAG}.csv"))
    return tab


# ==========================================================================
# MAIN
# ==========================================================================
if __name__ == "__main__":
    for tag, col in BOND_LEGS:
        run_leg(tag, col)

    print(f"\nwritten to {OUT_DIR}")