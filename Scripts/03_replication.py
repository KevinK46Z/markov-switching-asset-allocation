"""
03_replication.py
=================
Step one of the design in Chapter 4: reproduce Ang and Bekaert (2004) on
independently sourced data over January 1952 to December 2000, with
out-of-sample evaluation from January 1985.  Runs once for each bond series,
the ten-year baseline and the twenty-five-year diagnostic.

Run 01_build_data.py first.  Loads the estimator from 02_msm.py.

PART A  Full-sample estimation of M0 and M1, the comparison with Table 6 of
        the working paper, and the likelihood-ratio test of constant
        transition probabilities.  Feeds Table 6.1, the transition figure
        and the saved parameter vector that 04 and 05 read.
PART B  Recursive out-of-sample backtest from January 1985.  Independent of
        Part A, since every month is re-estimated on its own window.  The
        backtest file it writes is read by 04 and 06.
PART C  Performance of the three strategies at gamma in {2, 3, 4, 5, 10},
        and the gamma-free decomposition of Appendix A.3.  Feeds Table 6.3.

Strategy tags, used in every column name (common.STRATEGIES):
    rs   regime-dependent, hard classification, the Ang-Bekaert rule
    sf   regime-dependent, soft classification
    un   non-regime-dependent benchmark, sample moments of the same window

AB_GAMMA_FREE holds Ang and Bekaert's published tables in the consistent
form of Section 6.3.  The results are compared against it.

Approx. running time: 3400 s
"""

import os
import tempfile
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "common", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "00_common.py"))
common = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(common)

OUT_DIR     = common.OUT_DIR

OOS_START   = "1985-01"          # first month whose returns the strategy earns
GAMMAS      = common.GAMMAS      # risk aversion grid of Ang-Bekaert Table 8
STRATEGIES  = common.STRATEGIES  # (tag, printed name) per strategy

# Stamps every cache and output file.  Bump it when the specification
# changes, so RESUME cannot restore an estimate from an earlier version.
SPEC_TAG = "v5"

# Bond series, one run each.  The first is the baseline.
BOND_LEGS = [("10y", "rb_x_10y"), ("long", "rb_x_long")]

# How many randomly shifted starting points each search uses, in addition to
# the data-driven start (Section 4.4).  More starts make it less likely that
# the optimiser stops on a local peak, but each start costs a full search.
N_RANDOM_FULL  = 8    # Part A: the full-sample fits, run once, so many starts
N_RANDOM_REFIT = 3    # backtest: first month and every REFIT_EVERY-th month
N_RANDOM_OOS   = 0    # backtest: all other months start from last month's
                      # estimate only
REFIT_EVERY    = 24   # months between the wider searches in the backtest

# Resume an interrupted run from its checkpoint instead of starting over.
RESUME = True

# Checkpoints are rewritten every month, so they live in the temporary folder
# rather than in OneDrive, whose sync can lock a file mid-write and stop the
# run.
CKPT_DIR = os.path.join(tempfile.gettempdir(), "ab2004_replication")

msm = common.load_module("02_msm.py", "msm")      # estimator


# --------------------------------------------------------------------------
# Ang and Bekaert's published performance numbers
# --------------------------------------------------------------------------
# Typed in from the working paper (2002) as printed.  Only Model II and the
# non-regime-dependent benchmark, the two strategies Ang and Bekaert run, so
# there is no soft-rule counterpart.  Everything derived from them below is
# computed.

# Table 8 (2002) (= Table 4 of the published article, 2004): out-of-sample,
# 1985 to 2000.
AB_OOS_PUBLISHED = pd.DataFrame({
    "gamma":      GAMMAS * 2,      # the grid twice: once per strategy
    "strategy":   ["regime-dependent (hard)"] * 5 + ["non-regime-dependent"] * 5,
    "mean_ann_%": [25.29, 17.69, 13.89, 11.61, 7.05,
                   17.65, 12.60, 10.07, 8.55, 5.52],
    "sd_ann_%":   [34.53, 23.02, 17.27, 13.82, 6.91,
                   26.25, 17.50, 13.13, 10.50, 5.26],
    "sharpe_printed": [0.58, 0.54, 0.50, 0.47, 0.27,
                       0.48, 0.42, 0.37, 0.32, 0.07],
}).set_index(["gamma", "strategy"])

# Table 7: in-sample, 1952 to 2000.
AB_IS_PUBLISHED = pd.DataFrame({
    "gamma":      GAMMAS * 2,      # the grid twice: once per strategy
    "strategy":   ["regime-dependent (hard)"] * 5 + ["non-regime-dependent"] * 5,
    "mean_ann_%": [24.52, 18.05, 14.81, 12.87, 8.98,
                   18.34, 13.92, 11.72, 10.39, 7.74],
    "sd_ann_%":   [29.79, 19.89, 14.85, 11.87, 5.91,
                   25.65, 17.07, 12.79, 10.22, 5.10],
    "sharpe_printed": [0.65] * 5 + [0.51] * 5,
}).set_index(["gamma", "strategy"])

# Gamma-free summary of both tables (Appendix A.3): A and B from a regression
# of the printed means on 1/gamma, sd_gamma the average of sd * gamma over the
# grid.  The regression residuals are below 0.006 everywhere.
# one row per table and strategy: (sample, strategy, A, B, sd_gamma)
_rows = []
for sample, tab in (("out-of-sample", AB_OOS_PUBLISHED),
                    ("in-sample",     AB_IS_PUBLISHED)):
    for name, label in (("regime-dependent (hard)", "regime-dependent"),
                        ("non-regime-dependent",    "non-regime-dependent")):
        # the five rows of this strategy, one per gamma
        t = tab.xs(name, level="strategy")
        # A and B from the regression of the means on 1/gamma, sd_gamma the
        # average of sd * gamma; the residual and the ratio are not needed
        A, B, _, sdg, _ = common.gamma_decomposition(
            t["mean_ann_%"], t["sd_ann_%"], GAMMAS)
        _rows.append((sample, label, A, B, sdg))

# collect the four rows into one table, indexed by sample and strategy
AB_GAMMA_FREE = pd.DataFrame(
    _rows, columns=["sample", "strategy", "A", "B", "sd_gamma"]
).set_index(["sample", "strategy"])
# the Sharpe ratio at every gamma, B / C
AB_GAMMA_FREE["sharpe_invariant"] = AB_GAMMA_FREE["B"] / AB_GAMMA_FREE["sd_gamma"]


# ==========================================================================
# PART A: full-sample estimation
# ==========================================================================
def part_a(panel, tag):
    """
    Full-sample estimation of M0 and M1 on one bond series, with the
    comparison against Ang and Bekaert's Table 6 and the likelihood-ratio
    test of constant transition probabilities.

    Prints the estimates of both models, M1 parameter by parameter against
    Table 6 (2002), the implied standard deviations and correlations, the long-run
    probability of regime 1 and the test of M0 against M1.  Saves the M1
    parameter vector for 04 and 05.

    Parameters
    ----------
    panel : pd.DataFrame
        Replication panel with `rb_x` set to the chosen bond series.
    tag : str
        Label of the bond series, "10y" or "long", used in the file names.

    Returns
    -------
    m0, m1 : msm.MSResult
        The constant-transition fit and the fit with the short rate driving
        the transitions.
    se1 : np.ndarray
        Standard errors of the M1 estimates, in the units of theta.

    Notes
    -----
    The estimates and the M1 standard errors are cached in an .npz file.  On
    a later run the filter is re-run at the cached estimates instead of
    estimating again, which gives an identical result.  Delete the .npz to
    force a new estimation.
    """
    print("=" * 74)
    print(f"PART A   Full-sample estimation, {panel.index.min()} to "
          f"{panel.index.max()}   [bond series: {tag}]")
    print("=" * 74)

    cache = os.path.join(OUT_DIR, f"repl_partA_{tag}_{SPEC_TAG}.npz")
    if RESUME and os.path.exists(cache):
        # only theta is cached. evaluate() re-runs the filter at it, which is
        # deterministic, so the restored fit is identical to the original
        z = np.load(cache)                                      # theta only
        m0  = msm.evaluate(panel, [],          z["theta_m0"])   # refilter, no refit
        m1  = msm.evaluate(panel, ["z_short"], z["theta_m1"])   # same for M1
        se1 = z["se_m1"]                               # stored, not recomputed
        print("\n   (restored from cache)")
    else:
        t0 = time.time()
        print("\nEstimating M0 (constant transition probabilities)...")
        m0 = msm.fit(panel, zcols=[], n_random=N_RANDOM_FULL)
        print("Estimating M1 (lagged short rate drives the transitions)...")
        # start M1 at the converged M0 with both slopes at zero
        m1 = msm.fit(panel, zcols=["z_short"], n_random=N_RANDOM_FULL,
                     theta0=msm.nest_start(m0.theta, 0, 1))
        se1 = msm.standard_errors(m1, panel, ["z_short"])   # numerical Hessian
        np.savez(cache, theta_m0=m0.theta, theta_m1=m1.theta, se_m1=se1)
        print(f"   ({time.time() - t0:.0f} seconds)")

    print("\n--- M0: constant transitions " + "-" * 45)
    print(m0.summary())
    print("\n--- M1: Ang-Bekaert baseline " + "-" * 45)
    print(m1.summary(se=se1))

    # ---- parameter-by-parameter comparison against Table 6 ---------------
    # Both vectors are converted to Ang and Bekaert's own naming and units, so
    # the Cholesky block is reported as R_rr, R_br and so on rather than in the
    # packed log form the optimiser searches over.  Section 6.1.
    here  = msm.natural_params(m1.theta, 1)              # theta in Table 6 naming
    there = msm.natural_params(msm.AB_MODEL_II_THETA, 1) # their published theta
    print("\n--- M1 against Table 6, Model II " + "-" * 41)
    print(f"{'':14s}{'here':>11s}{'AB':>11s}{'diff':>10s}"
          f"{'AB s.e.':>10s}{'diff/s.e.':>11s}")
    for name in there:
        h, t = here[name], there[name]
        se = msm.AB_MODEL_II_SE.get(name, float("nan"))
        # the last column is the difference in units of their standard error,
        # the closeness criterion of Section 4.7.  Table 6 prints 0.00 for
        # parameters estimated more precisely than two decimals, so those rows
        # print "--" instead of dividing by zero.
        ratio = f"{(h - t) / se:11.2f}" if (np.isfinite(se) and se > 0) else f"{'--':>11s}"
        print(f"{name:14s}{h:11.4f}{t:11.4f}{h - t:10.4f}{se:10.2f}{ratio}")

    # ---- derived moments, the form the published text quotes -------------
    dh = msm.derived_moments(m1.theta, 1) # sd of each shock, corr(b,e)
    dt = msm.derived_moments(msm.AB_MODEL_II_THETA, 1) # same, at their theta
    print("\n--- implied standard deviations and correlations " + "-" * 25)
    print(f"{'':22s}{'here':>11s}{'AB':>11s}{'diff':>10s}")
    for name in dt:
        print(f"{name:22s}{dh[name]:11.4f}{dt[name]:11.4f}{dh[name] - dt[name]:10.4f}")

    # ---- long-run probability of the calm regime (Table 6.1) -------------
    # Simulated from the joint (r, s) process, as Ang and Bekaert do
    print("\n--- long-run P(regime 1) " + "-" * 49)
    print(f"   simulated from the fitted model : "
          f"{msm.stationary_prob(m1.params, n=200_000):.4f}")
    print("   Ang and Bekaert, Model II       : 0.7014   (Table 6 rounds mu_r")
    print("                                      and rho, so not reproducible)")

    # ---- likelihood-ratio test of constant transitions -------------------
    stat, dof, pval = msm.lr_test(m0, m1)
    print("\n--- Likelihood-ratio test, M0 restriction b1 = b2 = 0 " + "-" * 20)
    print(f"   LR statistic {stat:.3f} on {dof} degrees of freedom, p = {pval:.3g}")
    print(f"   {'REJECTED' if pval < 0.05 else 'NOT rejected'} at the 5 percent "
          f"level.  Ang and Bekaert: 'strongly statistically rejected'.")

    # ---- save the fitted M1 vector for 04 and 05 -------------------------
    # Stamped with SPEC_TAG so a downstream script cannot silently pick up a
    # vector fitted under an earlier build.  05 reads the "M1" column.
    pd.DataFrame({
        "parameter": [f"theta_{i}" for i in range(len(m1.theta))],
        "M1":        m1.theta,
        "M1_se":     se1,
    }).to_csv(os.path.join(OUT_DIR, f"repl_M1_theta_{tag}_{SPEC_TAG}.csv"),
              index=False)
    return m0, m1, se1


# ==========================================================================
# PART B: recursive out-of-sample backtest
# ==========================================================================
def part_b(panel, zcols, tag, label="M1"):
    """
    Recursive out-of-sample backtest from OOS_START: each month, estimate the
    model on the data available at the decision date, form the three
    portfolios and record what they earn the following month.

    Parameters
    ----------
    panel : pd.DataFrame
        Replication panel with `rb_x` set to the chosen bond series.
    zcols : list of str
        Transition covariate columns, ["z_short"] for M1.
    tag : str
        Label of the bond series, used in the checkpoint file names.
    label : str
        Model name, used in the printed output only.

    Returns
    -------
    pd.DataFrame
        One row per holding month, indexed by month.  Holds the filtered and
        predicted probabilities of regime 1, the classified regime, P and Q,
        the mixing weights of both rules (pi1_h, pi1_s), and the weights and
        realised returns of all three strategies at every gamma.

    Notes
    -----
    For each decision month d:
        1. estimate the model on the panel from its start through d;
        2. classify month d as regime 1 if its filtered probability exceeds
           one half;
        3. form the moments of next month's returns from the covariates
           observed at the close of d (panel row d+1);
        4. compute the weights of the three strategies and apply them to the
           realised returns of month d+1.
    The hard and soft rules share the estimation and differ only in the
    mixing weight.  Nothing dated after the decision enters.

    After every month the rows so far and the latest parameter vector are
    saved to two checkpoint files in CKPT_DIR, so an interrupted run resumes
    where it stopped.  Delete both files to force a clean run.
    """
    print("\n" + "=" * 74)
    print(f"PART B   Recursive out-of-sample backtest, {label}, "
          f"from {OOS_START}   [bond series: {tag}]")
    print("=" * 74)

    months = panel.index
    first_hold = months.get_loc(pd.Period(OOS_START, freq="M"))

    # ---- checkpointing ---------------------------------------------------
    os.makedirs(CKPT_DIR, exist_ok=True)                    # no-op if it exists
    ckpt_rows  = os.path.join(CKPT_DIR, f"backtest_{tag}_{SPEC_TAG}_partial.csv")
    ckpt_theta = os.path.join(CKPT_DIR, f"backtest_{tag}_{SPEC_TAG}_partial_theta.npy")

    rows = []                                # one dict per holding month
    theta_prev = None                        # last month's solution, warm start
    start_h = first_hold                     # first month to compute

    if RESUME and os.path.exists(ckpt_rows) and os.path.exists(ckpt_theta):
        prev = pd.read_csv(ckpt_rows)        # months completed before the stop
        if len(prev):
            rows = prev.to_dict("records")   # back to the loop's own format
            for row in rows:                 # CSV loses the Period type
                row["month"] = pd.Period(row["month"], freq="M")
            theta_prev = np.load(ckpt_theta)      # warm start of the stopped run
            start_h = first_hold + len(rows)      # resume where it stopped
            print(f"   resuming: {len(rows)} months already done, "
                  f"restarting at {months[start_h]}")

    t0 = time.time()
    for h in range(start_h, len(months)):
        d_idx = h - 1                        # decision month
        window = panel.iloc[: d_idx + 1]     # data available at the decision

        # full multi-start refit periodically, warm start otherwise
        n_since = h - first_hold             # months since the backtest began
        full = (theta_prev is None) or (n_since % REFIT_EVERY == 0)   # refit month?
        res = msm.fit(window, zcols,
                      n_random=N_RANDOM_REFIT if full else N_RANDOM_OOS,
                      theta0=theta_prev,     # None on the first month
                      polish=full,           # Nelder-Mead only on refit months
                      include_default_start=full)   # cold start only on refits
        theta_prev = res.theta               # warm start for next month

        # covariate driving the transition into the holding month.  For the
        # baseline this is the short rate of month d, known at the decision.
        z_next = (panel[zcols].iloc[h].to_numpy(dtype=float)
                  if zcols else np.zeros(0))
        xi_next = msm.predict_regime(res, z_next)      # diagnostic only
        xi_now  = float(res.xi_filt[-1])               # filtered, decision month

        # classify the current regime on the filtered probability
        regime = 0 if xi_now > 0.5 else 1

        # moments the allocation is built on, once per rule.  Same parameters,
        # same covariate, different mixing weight.
        mu_h, Sig_h, pi1_h, P_t, Q_t = msm.conditional_moments(
            res.params, z_next, regime=regime)  # hard: xi rounded first
        mu_s, Sig_s, pi1_s, _, _ = msm.conditional_moments(
            res.params, z_next, xi=xi_now)      # soft: xi used as it is

        # benchmark: unconditional moments over the same window
        x_win = window[["rb_x", "re_x"]].to_numpy(dtype=float)
        mu_u  = x_win.mean(axis=0)
        Sig_u = np.cov(x_win.T, ddof=1)

        # realised excess returns of the holding month, percent -> decimal
        x_next  = panel[["rb_x", "re_x"]].iloc[h].to_numpy(dtype=float) / 100.0
        rf_next = panel["r"].iloc[h] / 100.0

        row = {
            "month":     months[h],
            "xi_filt_d": xi_now,              # what the classification uses
            "xi_pred":   xi_next,             # one-step forecast, diagnostic
            "regime":    regime + 1,
            "P_t":       P_t,
            "Q_t":       Q_t,
            "pi1_h":     pi1_h,               # mixing weight, hard rule
            "pi1_s":     pi1_s,               # mixing weight, soft rule
            "loglik":    res.loglik,
            "refit":     full,
            "rf":        rf_next,
        }
        for g in GAMMAS:
            w = {
                "rs": common.mv_weights(mu_h, Sig_h, g),   # regime-dependent, hard
                "sf": common.mv_weights(mu_s, Sig_s, g),   # regime-dependent, soft
                "un": common.mv_weights(mu_u, Sig_u, g),   # static benchmark
            }
            for strat, w_s in w.items():    # one column set per rule
                row[f"w_bond_{strat}_g{g}"]   = w_s[0]
                row[f"w_equity_{strat}_g{g}"] = w_s[1]
                # cash pays rf on the residual 1 - w_b - w_e, and the two risky
                # assets pay their excess returns, so the total is rf + w'x
                row[f"R_{strat}_g{g}"] = rf_next + w_s @ x_next
        rows.append(row)                       # one row per holding month

        # flush the checkpoint.  A failed write is never fatal: the checkpoint
        # exists to save work, so it must not be able to destroy it.  The next
        # month rewrites it anyway.
        try:
            pd.DataFrame(rows).to_csv(ckpt_rows, index=False)   # every row so far
            np.save(ckpt_theta, theta_prev)                     # the warm start
        except OSError as e:
            print(f"   (checkpoint write skipped: {e})", flush=True)

        if n_since % REFIT_EVERY == 0:         # progress line on refit months
            print(f"   {months[h]}  loglik {res.loglik:10.2f}  "
                  f"xi_filt {xi_now:.3f}  regime {regime + 1}  "
                  f"pi1 {pi1_h:.3f}/{pi1_s:.3f}  "
                  f"[{time.time() - t0:5.0f}s]", flush=True)

    bt = pd.DataFrame(rows).set_index("month")
    # building from dicts leaves an object-dtype index, restore a PeriodIndex
    bt.index = pd.PeriodIndex(bt.index, freq="M")

    print(f"   done: {len(bt)} months in {time.time() - t0:.0f} seconds")

    # the diagnostic that says whether the mixing is working (Section 6.2).  With the raw
    # within-regime covariance the equity weight takes at most two values.
    n_uniq = bt["w_equity_rs_g5"].round(6).nunique()
    print(f"   distinct hard-rule equity weights at gamma=5: {n_uniq}"
          f" of {len(bt)} months")
    if n_uniq <= 2:
        print("   WARNING: the weights are a step function.  That is the "
              "signature of using the raw within-regime covariance instead of "
              "the mixture of Equation 4.27.")
    # mean and range of the mixing weight (Section 6.2)
    for c, name in (("pi1_h", "hard"), ("pi1_s", "soft")):
        print(f"   blend weight pi1, {name}: mean {bt[c].mean():.3f}, "
              f"range {bt[c].min():.3f} to {bt[c].max():.3f}")
    # how far apart the two rules are (Section 6.4).  Zero in every month
    # where the filter is confident, so the mean absolute gap is the size of
    # the disagreement.
    gap = (bt["pi1_h"] - bt["pi1_s"]).abs()
    print(f"   |pi1 hard - pi1 soft|: mean {gap.mean():.3f}, "
          f"max {gap.max():.3f}, above 0.05 in "
          f"{(gap > 0.05).mean():.1%} of months")

    for f in (ckpt_rows, ckpt_theta):        # series finished, drop the partials
        try:
            if os.path.exists(f):
                os.remove(f)
        except OSError:
            pass
    return bt


# ==========================================================================
# PART C: performance table
# ==========================================================================
def part_c(bt, tag):
    """
    Out-of-sample performance of the three strategies, at every gamma and in
    gamma-free form, set against Ang and Bekaert's published table.

    Prints the performance table at gamma in {2, 3, 4, 5, 10} (Table 6.3,
    Panel B, at gamma = 5)) and their printed Table 8 (2002), then the gamma-free
    decomposition of Appendix A.3 for each strategy next to theirs (Table
    6.3, Panel A).

    Parameters
    ----------
    bt : pd.DataFrame
        Output of part_b().
    tag : str
        Label of the bond series, used in the printed output only.

    Returns
    -------
    tab : pd.DataFrame
        Annualised mean, standard deviation, Sharpe ratios, compound growth,
        final wealth, target turnover and worst month, indexed by (gamma,
        strategy).
    gamma_free : pd.DataFrame
        For each strategy A, B, the largest regression residual, sd * gamma
        and the Sharpe ratio B / sd_gamma, from gamma_decomposition() applied
        to the total returns in tab.
    """
    print("\n" + "=" * 74)
    print(f"PART C   Out-of-sample performance, {bt.index.min()} to "
          f"{bt.index.max()}   [bond series: {tag}]")
    print("=" * 74)

    recs = []                                     # one dict per (gamma, strategy)
    for g in GAMMAS:
        for strat, name in STRATEGIES:
            s = common.performance(bt[f"R_{strat}_g{g}"].to_numpy(),   # realised returns
                                   bt["rf"].to_numpy(), name)
            s["gamma"] = g  # becomes part of the index
            # one-way turnover in the two risky assets, before drift adjustment.
            # Reported because AB's defence for ignoring costs is a claim about
            # turnover (Section 4.6).
            dw = (bt[[f"w_bond_{strat}_g{g}", f"w_equity_{strat}_g{g}"]]
                  .diff().abs().sum(axis=1))      # |dw_b| + |dw_e| each month
            s["turnover_pm"] = dw.iloc[1:].mean()     # first diff is NaN
            recs.append(s)

    tab = pd.DataFrame(recs).set_index(["gamma", "strategy"])
    tab = tab[["mean_ann_%", "sd_ann_%", "sharpe_ann", "sharpe_excess",   # column order
               "mean_cagr_%", "final_wealth", "turnover_pm", "min_month_%"]]

    print("\nReplication (returns and standard deviations annualised, percent):")
    print(tab.round(3).to_string())
    print("\nAng and Bekaert, out-of-sample table, as printed:")
    print(AB_OOS_PUBLISHED.round(3).to_string())

    # ---- the gamma-free comparison ---------------------------------------
    rf_ann_here = 100.0 * 12.0 * bt["rf"].mean()
    print("\n" + "-" * 74)
    print("GAMMA-FREE COMPARISON   mean_ann = A + B/gamma,  sd_ann = C/gamma")
    print("-" * 74)
    print(f"{'':34s}{'A':>9s}{'B':>10s}{'resid':>9s}{'sd*gam':>10s}{'SR':>8s}")
    rows = []
    for strat, name in STRATEGIES:
        means = [tab.loc[(g, name), "mean_ann_%"] for g in GAMMAS]
        sds   = [tab.loc[(g, name), "sd_ann_%"]   for g in GAMMAS]
        A, B, resid, sdg, sr = common.gamma_decomposition(means, sds, GAMMAS)
        print(f"{'here, ' + name:34s}{A:9.3f}{B:10.3f}{resid:9.4f}"
              f"{sdg:10.2f}{sr:8.3f}")
        rows.append((name, A, B, resid, sdg, sr))
    for sample in ("out-of-sample", "in-sample"):          # AB, both tables
        for name in ("regime-dependent", "non-regime-dependent"):
            r = AB_GAMMA_FREE.loc[(sample, name)]
            print(f"{'AB ' + sample + ', ' + name:34s}{r['A']:9.3f}{r['B']:10.3f}"
                  f"{'--':>9s}{r['sd_gamma']:10.2f}{r['sharpe_invariant']:8.3f}")

    print(f"\n   mean annualised risk-free rate over this window: "
          f"{rf_ann_here:.3f} percent.  A should equal it.")

    gamma_free = pd.DataFrame(
        rows, columns=["strategy", "A", "B", "resid", "sd_gamma", "sharpe_invariant"]
    ).set_index("strategy")
    return tab, gamma_free


# ==========================================================================
# Figures
# ==========================================================================
def _ann_to_monthly(r_ann_pct):
    """
    Convert an annual rate into the equivalent monthly rate, both in percent.

    Used for the transition figure.  Its axis shows the short rate per year,
    which is easier to read, but the transition equations are estimated on
    the monthly rate, so the axis values are converted before P and Q are
    evaluated.  The conversion compounds, as in build_cash(): a monthly rate
    that, earned twelve times, gives the annual rate.

    Parameters
    ----------
    r_ann_pct : array_like or float
        Annual rate in percent, e.g. 6.0 for 6 %.

    Returns
    -------
    np.ndarray or float
        Monthly rate in percent, e.g. 0.487 for 6 % a year.
    """
    return ((1.0 + np.asarray(r_ann_pct) / 100.0) ** (1.0 / 12.0) - 1.0) * 100.0


def figures(m1, tag):
    """
    Draw the transition probabilities of the fitted M1 against the short
    rate, Figure 6.1 of the thesis and the counterpart of Ang and Bekaert's
    Figure 5 (2004).

    Parameters
    ----------
    m1 : msm.MSResult
        The full-sample fit of M1, which supplies the transition
        intercepts and slopes.
    tag : str
        Label of the bond series, appended to the file name.

    Returns
    -------
    None
        Writes repl_transition_probs_<tag>.pdf to OUT_DIR.
    """
    plt.rcParams.update(common.PLOT_RC)

    # The x-axis is annualised for readability, the functions are evaluated
    # at the monthly rate.
    p = m1.params
    grid_ann = np.linspace(0.0, 18.0, 200)
    grid_mo  = _ann_to_monthly(grid_ann)
    P_grid = msm._logistic(p["a"][0] + p["b"][0][0] * grid_mo)
    Q_grid = msm._logistic(p["a"][1] + p["b"][1][0] * grid_mo)
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    ax.plot(grid_ann, P_grid, color="black", lw=1.0,
            label="$P(r_{t-1})$: stay in regime 1")
    ax.plot(grid_ann, Q_grid, color="black", lw=1.0, ls=(0, (4, 3)),
            label="$Q(r_{t-1})$: stay in regime 2")
    ax.set_xlabel("Annualised short rate (%)", fontsize=8.5)
    ax.set_ylabel("Transition probability", fontsize=8.5)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT_DIR, f"repl_transition_probs_{tag}.pdf"),
                bbox_inches="tight")
    plt.close(fig)

    print(f"\nfigure written to {OUT_DIR}")


# ==========================================================================
# MAIN
# ==========================================================================
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    full = common.read_panel("panel_replication.csv")

    need = ["r_obs", "r", "z_short", "rb_x_10y", "rb_x_long", "re_x"]
    miss = [c for c in need if c not in full.columns]
    if miss:
        raise SystemExit(f"panel_replication.csv is missing {miss}; "
                         f"re-run 01_build_data.py")
    if full[need].isna().any().any():
        bad = full[need].isna().any()
        raise SystemExit(f"NaN in {list(bad[bad].index)}; the panel must be "
                         f"complete over the whole window")

    # the panel's defining invariant, checked at the door rather than
    # discovered inside the estimator
    if not np.allclose(full["r"].to_numpy()[1:], full["r_obs"].to_numpy()[:-1]):
        raise SystemExit("panel violates r_t == r_obs_{t-1}; re-run 01")
    if not np.allclose(full["z_short"], full["r"]):
        raise SystemExit("z_short != r; the panel predates the timing fix")

    # The exercise runs once per bond series.  Nothing else differs between
    # the two passes, so a difference in results comes from the bond alone.
    for tag, col in BOND_LEGS:
        print("\n" + "#" * 74)
        print(f"#  BOND SERIES '{tag}'  (panel column {col})")
        print("#" * 74 + "\n")

        panel = common.promote_leg(full, col)
        m0, m1, se1 = part_a(panel, tag)
        bt = part_b(panel, zcols=["z_short"], tag=tag, label="M1")
        bt.to_csv(os.path.join(OUT_DIR, f"repl_backtest_{tag}_{SPEC_TAG}.csv"))
        tab, gf = part_c(bt, tag)
        tab.to_csv(os.path.join(OUT_DIR, f"repl_performance_{tag}_{SPEC_TAG}.csv"))
        gf.to_csv(os.path.join(OUT_DIR, f"repl_gamma_free_{tag}_{SPEC_TAG}.csv"))
        figures(m1, tag)

    print("\nAll output in", OUT_DIR)