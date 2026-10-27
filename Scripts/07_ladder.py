"""
07_ladder.py
============
The specification ladder M0 to M4 on the modern panel (Chapter 7).  Run
01_build_data.py first.  Imports the estimator from 02_msm.py.  07c, 07d and
08 build on the caches and backtest files this script writes, the way 04 and
05 build on 03.

PART A  Full-sample estimation of every rung: the nested likelihood-ratio
        tests and information criteria (Table 7.1), the transition
        coefficients (Table 7.2), the within-regime moments (Table 7.3) and
        the correlation of the covariates (Section 7.2).
PART B  Recursive out-of-sample backtest of each rung from OOS_START.  One
        estimation per month per rung, nothing dated after the decision.
PART C  Performance of every rung against the static benchmark, gross of
        transaction costs (Table 7.4), the ZLB split (Table 7.6) and the
        January 2009 holding month (Section 7.5).
Figures 7.1 to 7.4 are written at the end, and 07c redraws them from disk.

Ladder.  Each rung adds one transition covariate to the rung below it, so
every rung is nested in the next and each comparison carries two degrees of
freedom.  LADDER is the only place the order is set.

    M0  none                              constant transitions
    M1  z_short                           Ang-Bekaert baseline
    M2  z_short, z_term
    M3  z_short, z_term, z_ebp
    M4  z_short, z_term, z_ebp, z_vix

Strategy tags, as in 03_replication.py (common.STRATEGIES):
    rs   regime-dependent, hard classification
    sf   regime-dependent, soft classification
    un   non-regime-dependent benchmark, sample moments of the same window

Standardisation.  Each covariate is rescaled to zero mean and unit variance
before it reaches the estimator.  Part A uses full-sample moments, Part B the
estimation window alone.  prepare() is called with strict=False throughout,
since a standardised z_short no longer equals the AR lag.

Running in pieces.  Rungs can be passed on the command line, e.g.

    python 07_ladder.py M3 M4

Part A is cached per rung and Part B checkpoints per rung, so an interrupted
overnight run resumes where it stopped.

Approx. running time: 8500s
"""

import os
import sys
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

PANEL_FILE  = "panel_modern.csv"
BOND_COL    = "rb_x_10y"         # ten-year series only, Section 5.2.3
BOND_TAG    = "10y"

OOS_START   = "2005-01"          # first month whose returns the strategy earns
GAMMAS      = common.GAMMAS      # same grid as 03, so the tables line up

# One rung per entry, each nested in the next.  Order set here and nowhere
# else.
LADDER = [
    ("M0", []),
    ("M1", ["z_short"]),
    ("M2", ["z_short", "z_term"]),
    ("M3", ["z_short", "z_term", "z_ebp"]),
    ("M4", ["z_short", "z_term", "z_ebp", "z_vix"]),
]

LADDER_RUNGS = tuple(n for n, _ in LADDER)   # frozen; 07d appends noise rungs later

STRATEGIES = common.STRATEGIES   # (tag, printed name) per strategy

# Bumped when the specification changes, so RESUME cannot silently restore
# estimates produced under an earlier build.  08_costs.py must carry the same
# tag.  Delete the npz or the backtest CSV to force a refit under one tag.
SPEC_TAG = "e1"      # e1: first ladder run, M0-M4 on the modern panel

RUNG_COLOURS  = common.RUNG_COLOURS   # one colour per rung, every figure
STATIC_COLOUR = common.STATIC_COLOUR  # the benchmark, always dashed

# Optimiser starts.  Part A runs once per rung so it can afford a wide search.
# The backtest warm-starts each month from the previous month's solution and
# does a full multi-start refit every REFIT_EVERY months.
N_RANDOM_FULL  = 8    # Part A, full-sample estimation
N_RANDOM_FIRST = 8    # first month of a rung's backtest, no warm start yet
N_RANDOM_REFIT = 3    # periodic refit inside the backtest
N_RANDOM_OOS   = 0    # ordinary warm-started month
REFIT_EVERY    = 24

# Crash recovery, not estimation. pick up a partial run from its checkpoint.
RESUME = True

# Checkpoints are rewritten every month, so they must stay off OneDrive.
CKPT_DIR = os.path.join(tempfile.gettempdir(), "ext_ladder")

msm = common.load_module("02_msm.py", "msm")      # the estimator

# Short names for the shared statistics.  07d reaches sharpe and
# sharpe_diff_test through this module.
sharpe              = common.sharpe
sharpe_diff_test    = common.sharpe_diff_test
gamma_decomposition = common.gamma_decomposition
mv_weights          = common.mv_weights
performance         = common.performance

# Converged theta of the first backtest month of each ladder rung, written to
# disk by part_b.  The rung above starts its first month there with the new
# slopes at zero, which uses only data available at that date.  The file is
# read from disk rather than held in memory, so a rung run on its own or
# resumed from a checkpoint gets the same seed as a full run.
def first_theta_path(name):
    """
    Path of the first-month estimate of a rung's backtest.

    Parameters
    ----------
    name : str
        Rung label, e.g. "M1".

    Returns
    -------
    str
        Path of ext_first_theta_<name>_<SPEC_TAG>.npy in OUT_DIR.  The
        file seeds the first month of the rung above.
    """
    return os.path.join(OUT_DIR, f"ext_first_theta_{name}_{SPEC_TAG}.npy")

# ==========================================================================
# Data
# ==========================================================================
def all_drivers():
    """
    List every covariate used anywhere in LADDER.

    Returns
    -------
    list of str
        Covariate columns in the order they first enter the ladder, e.g.
        ["z_short", "z_term", "z_ebp", "z_vix"].
    """
    out = []
    for _, zc in LADDER:
        for c in zc:
            if c not in out:
                out.append(c)
    return out


def lower_rung(name):
    """
    Name of the rung immediately below `name`.

    Parameters
    ----------
    name : str

    Returns
    -------
    str or None
        None for the bottom rung.
    """
    names = [n for n, _ in LADDER]
    i = names.index(name)
    return names[i - 1] if i else None


def load_modern():
    """
    Read the modern panel and trim it to the months every ladder covariate
    covers.

    Returns
    -------
    pd.DataFrame
        Monthly PeriodIndex, with `rb_x` set to BOND_COL and `zlb` as bool.

    Raises
    ------
    SystemExit
        A required column is missing, the timing checks fail, a covariate
        has a gap inside the kept window, or fewer than 24 months remain.

    Notes
    -----
    The excess bond premium is published with a lag, so the panel ends at
    the last month every covariate covers.  01_build_data.py already dates
    the covariates to the row they condition, so they are not lagged here.
    """
    panel = common.read_panel(PANEL_FILE)

    need = ["r_obs", "r", BOND_COL, "re_x", "zlb"] + all_drivers()
    miss = [c for c in need if c not in panel.columns]
    if miss:
        raise SystemExit(f"{PANEL_FILE} is missing {miss}; re-run "
                         f"01_build_data.py")

    # the panel's defining invariant, checked at the door
    if not np.allclose(panel["r"].to_numpy()[1:], panel["r_obs"].to_numpy()[:-1]):
        raise SystemExit("panel violates r_t == r_obs_{t-1}; re-run 01")
    if not np.allclose(panel["z_short"], panel["r"]):
        raise SystemExit("z_short != r; the panel predates the timing fix")

    # keep the longest run of rows in which every covariate is present
    ok = panel[all_drivers() + [BOND_COL, "re_x", "r", "r_obs"]].notna().all(axis=1)
    if not ok.any():
        raise SystemExit("no row of the panel has all ladder covariates")
    lo, hi = panel.index[ok][0], panel.index[ok][-1]
    kept = panel.loc[lo:hi]
    if not kept[all_drivers()].notna().all().all():
        gaps = kept[all_drivers()].isna().sum()
        raise SystemExit(f"gaps inside the covariate window:\n"
                         f"{gaps[gaps > 0].to_string()}")
    if len(kept) < 24:
        raise SystemExit(f"only {len(kept)} usable months")
    if len(kept) < len(panel):
        print(f"   panel trimmed to {lo} to {hi}: "
              f"{len(panel) - len(kept)} row(s) dropped where a covariate is "
              f"unpublished")

    out = kept.copy()
    out["rb_x"] = out[BOND_COL]          # the ten-year series as the bond
    out["zlb"] = out["zlb"].astype(bool)
    return out


def standardised(df, zcols):
    """
    Return a copy of a frame with its covariate columns standardised to
    zero mean and unit variance over the frame itself.

    Parameters
    ----------
    df : pd.DataFrame
        Frame to transform, the full sample in Part A or the estimation
        window in Part B.
    zcols : list of str
        Covariate columns, empty for M0.

    Returns
    -------
    out : pd.DataFrame
        df with zcols standardised.  df itself for M0.
    mu, sd : np.ndarray, length k
        Mean and standard deviation used, so the same transform can be
        applied to the covariate vector of the holding month.
    """
    if not zcols:
        return df, np.zeros(0), np.ones(0)       # nothing to scale for M0
    Z = df[zcols].to_numpy(dtype=float)
    Zs, mu, sd = msm.standardise(Z, ref=Z)
    out = df.copy()
    out[zcols] = Zs
    return out, mu, sd


# ==========================================================================
# PART A: full-sample estimation of the ladder
# ==========================================================================
def part_a(panel):
    """
    Estimate every rung on the full sample and print Tables 7.1 to 7.3 and
    the covariate correlations.

    Parameters
    ----------
    panel : pd.DataFrame
        Output of load_modern().

    Returns
    -------
    dict of str -> msm.MSResult
        One fit per rung name.

    Notes
    -----
    Every rung is fitted even when Part B runs only a subset, since each
    rung starts from the one below.  Covariates are standardised over the
    full sample.  Fits are cached per rung as ext_partA_<rung>_<SPEC_TAG>.npz.
    """
    print("=" * 78)
    print(f"PART A   Full-sample ladder, {panel.index.min()} to "
          f"{panel.index.max()}   [bond series: {BOND_TAG}]")
    print("=" * 78)

    fits, ses, prev = {}, {}, None
    for name, zcols in LADDER:
        std, mu, sd = standardised(panel, zcols)              # full-sample moments
        cache = os.path.join(OUT_DIR, f"ext_partA_{name}_{SPEC_TAG}.npz")

        if RESUME and os.path.exists(cache):
            z = np.load(cache)                                # theta and se only
            res = msm.evaluate(std, zcols, z["theta"], strict=False)  # refilter
            se = z["se"]
            print(f"\n{name}: restored from cache")
        else:
            t0 = time.time()
            print(f"\n{name}: estimating with {len(zcols)} covariate(s)...")
            # start at the rung below with the new slopes at zero.  The two
            # models are numerically identical there, so this rung begins at
            # the lower rung's likelihood and cannot report a worse fit.
            th0 = (msm.nest_start(prev.theta, prev.k, len(zcols))
                   if prev is not None else None)
            res = msm.fit(std, zcols, n_random=N_RANDOM_FULL, theta0=th0,
                          strict=False)
            se = msm.standard_errors(res, std, zcols, strict=False)  # num. Hessian
            np.savez(cache, theta=res.theta, se=se, mu=mu, sd=sd)
            print(f"   ({time.time() - t0:.0f} seconds)")

        fits[name], ses[name] = res, se
        prev = res

        # coefficients and within-regime moments of this rung (Tables 7.2
        # and 7.3).  b is per standard deviation of the covariate, since the
        # columns were standardised above.
        print(res.summary(se=se))

    # ---- ladder table ----------------------------------------------------
    print("\n" + "-" * 78)
    print("LADDER: fit and nested likelihood-ratio tests")
    print("-" * 78)
    print(f"{'rung':6s}{'k':>3s}{'npar':>6s}{'loglik':>12s}{'AIC':>11s}"
          f"{'BIC':>11s}{'added':>10s}{'LR':>9s}{'df':>4s}{'p':>10s}")
    recs = []
    for i, (name, zcols) in enumerate(LADDER):
        r = fits[name]
        added = zcols[-1] if zcols else "--"
        if i == 0:
            stat, dof, pval = np.nan, 0, np.nan
        else:
            stat, dof, pval = msm.lr_test(fits[LADDER[i - 1][0]], r)
        print(f"{name:6s}{r.k:3d}{r.npar:6d}{r.loglik:12.3f}{r.aic:11.3f}"
              f"{r.bic:11.3f}{added:>10s}"
              + (f"{stat:9.3f}{dof:4d}{pval:10.3g}" if i else
                 f"{'--':>9s}{'--':>4s}{'--':>10s}"))
        recs.append({"rung": name, "k": r.k, "npar": r.npar,
                     "loglik": r.loglik, "aic": r.aic, "bic": r.bic,
                     "added": added, "LR_vs_prev": stat, "df": dof,
                     "p_vs_prev": pval})


    tab = pd.DataFrame(recs).set_index("rung")
    tab.to_csv(os.path.join(OUT_DIR, f"ext_ladder_fit_{SPEC_TAG}.csv"))

    # ---- transition coefficients side by side (Table 7.2) ----------------
    # Standardised units, so the columns are comparable across rungs and the
    # size of a slope is the effect of a one standard deviation move.
    print("\n" + "-" * 78)
    print("TRANSITION COEFFICIENTS across the ladder (per standard deviation)")
    print("-" * 78)
    drivers = all_drivers()
    for eq, lab in ((0, "b1: stay in regime 1"), (1, "b2: stay in regime 2")):
        print(f"\n{lab}")
        print(f"{'':10s}" + "".join(f"{n:>18s}" for n, _ in LADDER))
        for j, drv in enumerate(drivers):
            cells = []
            for name, zcols in LADDER:
                if drv in zcols:
                    idx = zcols.index(drv)
                    b = fits[name].params["b"][eq][idx]
                    s = ses[name][msm.N_FIXED + eq * fits[name].k + idx]
                    cells.append(f"{b:9.3f} ({s:.2f})")
                else:
                    cells.append("--")
            print(f"{drv:10s}" + "".join(f"{c:>18s}" for c in cells))
        cells = []
        for name, _ in LADDER:                       # the intercept row
            a = fits[name].params["a"][eq]
            s = ses[name][msm.N_FIXED - 2 + eq]
            cells.append(f"{a:9.3f} ({s:.2f})")
        print(f"{'a':10s}" + "".join(f"{c:>18s}" for c in cells))

    # ---- overlap of the covariates (Section 7.2) --------------------------
    # Standardising does not change a correlation, so the raw columns serve.
    print("\n" + "-" * 78)
    print("CORRELATION OF THE COVARIATES over the estimation sample")
    print("-" * 78)
    print(panel[drivers].corr().round(3).to_string())

    return fits


# ==========================================================================
# PART B: recursive out-of-sample backtest of one rung
# ==========================================================================
def part_b(panel, name, zcols):
    """
    Recursive out-of-sample evaluation of one rung.

    Parameters
    ----------
    panel : pd.DataFrame
        Modern panel with `rb_x` set to the ten-year series.
    name : str
        Rung label, used in the checkpoint and output file names.
    zcols : list of str
        Transition covariates of this rung, empty for M0.

    Returns
    -------
    pd.DataFrame
        One row per holding month, PeriodIndex, carrying the filtered and
        predicted probabilities, the classified regime, P_t, Q_t, the two
        mixing weights pi1_h and pi1_s, the standardised covariate vector, the
        ZLB flag, and the weights and realised returns of all three
        strategies at every gamma.

    Notes
    -----
    For each decision month d the model is estimated on the panel through d,
    the covariates are standardised on that window alone, the regime is
    classified on xi_{d|d}, the conditional moments for month d+1 are formed
    at z_{d+1}, and the weights are applied to month d+1's realised returns.
    Nothing dated after the decision enters.

    Every month is appended to a partial CSV with the last parameter vector
    beside it.  Delete the two _partial files to force a clean run.
    """
    print("\n" + "=" * 78)
    print(f"PART B   Recursive backtest of {name} from {OOS_START}   "
          f"[{len(zcols)} covariate(s): {', '.join(zcols) or 'none'}]")
    print("=" * 78)

    months = panel.index
    first_hold = months.get_loc(pd.Period(OOS_START, freq="M"))

    # ---- checkpointing ---------------------------------------------------
    os.makedirs(CKPT_DIR, exist_ok=True)                    # no-op if it exists
    ckpt_rows  = os.path.join(CKPT_DIR, f"ext_{name}_{SPEC_TAG}_partial.csv")
    ckpt_theta = os.path.join(CKPT_DIR, f"ext_{name}_{SPEC_TAG}_partial_theta.npy")

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
        # covariate moments from the estimation window only
        win_std, mu, sd = standardised(window, zcols)

        # full multi-start refit periodically, warm start otherwise
        n_since = h - first_hold             # months since the backtest began
        full = (theta_prev is None) or (n_since % REFIT_EVERY == 0)  # refit month?
        if theta_prev is not None:
            th0, n_rand = theta_prev, (N_RANDOM_REFIT if full else N_RANDOM_OOS)
        else:
            # first month of this rung: seed from the rung below at the same
            # date.  Read from disk, so the result does not depend on which
            # rungs ran in this process or on a resume.  Noise rungs of 07d
            # are not in LADDER_RUNGS and start cold.
            th0 = None
            if name in LADDER_RUNGS and lower_rung(name) is not None:
                f = first_theta_path(lower_rung(name))
                if not os.path.exists(f):
                    raise SystemExit(f"{os.path.basename(f)} missing; run "
                                     f"rung {lower_rung(name)} first")
                seed = np.load(f)
                k_lo = (len(seed) - msm.N_FIXED) // 2
                th0 = msm.nest_start(seed, k_lo, len(zcols))
            n_rand = N_RANDOM_FIRST
        res = msm.fit(win_std, zcols, n_random=n_rand, theta0=th0,
                      polish=full,           # Nelder-Mead only on refit months
                      include_default_start=full,   # cold start only on refits
                      strict=False)          # Z is standardised
        theta_prev = res.theta               # warm start for next month
        if h == first_hold and name in LADDER_RUNGS:
            np.save(first_theta_path(name), res.theta)   # seed for the rung above

        # covariate vector driving the transition into the holding month, put
        # through the same transform as the estimation window
        z_raw = (panel[zcols].iloc[h].to_numpy(dtype=float) if zcols
                 else np.zeros(0))
        z_next = (z_raw - mu) / sd if zcols else np.zeros(0)

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
            "rung":      name,
            "xi_filt_d": xi_now,              
            "xi_pred":   xi_next,            
            "regime":    regime + 1,
            "P_t":       P_t,
            "Q_t":       Q_t,
            "pi1_h":     pi1_h,              
            "pi1_s":     pi1_s,              
            "loglik":    res.loglik,
            "refit":     full,
            "rf":        rf_next,
            "zlb":       bool(panel["zlb"].iloc[h]),
        }
        for j, c in enumerate(zcols):
            row[f"{c}_std"] = float(z_next[j])   # covariate as the model saw it
        for g in GAMMAS:
            w = {
                "rs": mv_weights(mu_h, Sig_h, g),   # regime-dependent, hard
                "sf": mv_weights(mu_s, Sig_s, g),   # regime-dependent, soft
                "un": mv_weights(mu_u, Sig_u, g),   # static benchmark
            }
            for strat, w_s in w.items():    # one column set per rule
                row[f"w_bond_{strat}_g{g}"]   = w_s[0]   # w_bond_<rule>_g<gamma>
                row[f"w_equity_{strat}_g{g}"] = w_s[1]
                # cash pays rf on the residual 1 - w_b - w_e, and the two risky
                # assets pay their excess returns, so the total is rf + w'x
                row[f"R_{strat}_g{g}"] = rf_next + w_s @ x_next
        rows.append(row)                       # one row per holding month

        # flush the checkpoint.  A failed write is never fatal: the checkpoint
        # exists to save work, so it must not be able to destroy it.
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
    # so date arithmetic and .to_timestamp() work downstream
    bt.index = pd.PeriodIndex(bt.index, freq="M")
    bt["zlb"] = bt["zlb"].astype(bool)

    print(f"   done: {len(bt)} months in {time.time() - t0:.0f} seconds")

    # the diagnostic that says whether the mixture is working.  With the raw
    # within-regime covariance the equity weight takes at most two values.
    n_uniq = bt["w_equity_rs_g5"].round(6).nunique()
    print(f"   distinct hard-rule equity weights at gamma=5: {n_uniq}"
          f" of {len(bt)} months")
    if n_uniq <= 2:
        print("   WARNING: the weights are a step function.  That is the "
              "signature of using the raw within-regime covariance instead of "
              "the mixture.")
    # how far the two mixing weights lie apart (Section 7.4)
    gap = (bt["pi1_h"] - bt["pi1_s"]).abs()
    print(f"   |pi1 hard - pi1 soft|: mean {gap.mean():.3f}, "
          f"max {gap.max():.3f}, above 0.05 in "
          f"{(gap > 0.05).mean():.1%} of months")

    for f in (ckpt_rows, ckpt_theta):        # rung finished, drop the partials
        try:
            if os.path.exists(f):
                os.remove(f)
        except OSError:
            pass
    return bt


# ==========================================================================
# PART C: performance across the ladder
# ==========================================================================
def bt_path(name):
    """
    Path of one rung's backtest file.

    Parameters
    ----------
    name : str
        Rung label, e.g. "M1".

    Returns
    -------
    str
        Path of ext_backtest_<name>_<SPEC_TAG>.csv in OUT_DIR.
    """
    return os.path.join(OUT_DIR, f"ext_backtest_{name}_{SPEC_TAG}.csv")


def load_backtests():
    """
    Read the backtest of every rung that exists on disk.

    Returns
    -------
    dict of str -> pd.DataFrame
        One backtest per rung name, in LADDER order, with a monthly
        PeriodIndex.

    Raises
    ------
    SystemExit
        No backtest file exists, or two rungs cover different months.
    """
    out = {}
    for name, _ in LADDER:
        f = bt_path(name)
        if not os.path.exists(f):
            continue
        out[name] = common.read_backtest(os.path.basename(f))
    if not out:
        raise SystemExit("no backtest files found; run Part B first")
    spans = {n: (b.index.min(), b.index.max(), len(b)) for n, b in out.items()}
    if len(set(spans.values())) > 1:
        raise SystemExit(f"rungs cover different months: {spans}")
    return out


def part_c(bts):
    """
    Evaluate every rung gross of costs: performance and gamma-free form
    (Table 7.4), Memmel tests against the benchmark and against M1, the ZLB
    split (Table 7.6) and the January 2009 holding month (Section 7.5).

    Parameters
    ----------
    bts : dict of str -> pd.DataFrame
        Output of load_backtests().

    Returns
    -------
    tab : pd.DataFrame
        Performance by (rung, gamma, strategy).
    gf : pd.DataFrame
        Gamma-free summary by (rung, strategy).

    Raises
    ------
    SystemExit
        The benchmark differs between two rungs, which means they were not
        run on the same panel.
    """
    any_bt = next(iter(bts.values()))
    print("\n" + "=" * 78)
    print(f"PART C   Gross out-of-sample performance, {any_bt.index.min()} to "
          f"{any_bt.index.max()}   [bond series: {BOND_TAG}]")
    print("=" * 78)

    # the benchmark carries no regime model, so it must be identical in every
    # rung.  A difference means the rungs were run on different panels.
    ref = any_bt["R_un_g5"].to_numpy(dtype=float)
    for name, bt in bts.items():
        if not np.allclose(bt["R_un_g5"].to_numpy(dtype=float), ref, atol=1e-12):
            raise SystemExit(f"benchmark differs in rung {name}; the rungs were "
                             f"not run on the same panel")

    # ---- performance at every gamma ----------------------------------------
    # One row per (rung, gamma, strategy).  The benchmark is taken from the
    # first rung only and labelled "static".
    recs = []
    for name, bt in bts.items():
        for g in GAMMAS:
            for strat, label in STRATEGIES:
                if strat == "un" and name != list(bts)[0]:
                    continue                      # benchmark reported once
                s = performance(bt[f"R_{strat}_g{g}"].to_numpy(),
                                bt["rf"].to_numpy(), label)
                s["rung"] = "static" if strat == "un" else name
                s["gamma"] = g
                # one-way target turnover in the two risky assets, before
                # drift (Table 7.4).  08 recomputes it against the drifted
                # position.
                dw = (bt[[f"w_bond_{strat}_g{g}", f"w_equity_{strat}_g{g}"]]
                      .diff().abs().sum(axis=1))
                s["turnover_pm"] = dw.iloc[1:].mean()   # first diff is NaN
                recs.append(s)

    tab = pd.DataFrame(recs).set_index(["rung", "gamma", "strategy"]).sort_index()
    tab = tab[["mean_ann_%", "sd_ann_%", "sharpe_ann", "sharpe_excess",
               "mean_cagr_%", "final_wealth", "turnover_pm", "min_month_%"]]
    print("\nGross of transaction costs, annualised percent:")
    print(tab.round(3).to_string())

    # ---- gamma-free summary (Table 7.4, B and C) ---------------------------
    print("\n" + "-" * 78)
    print("GAMMA-FREE COMPARISON   mean_ann = A + B/gamma,  sd_ann = C/gamma")
    print("-" * 78)
    print(f"{'':34s}{'A':>9s}{'B':>10s}{'resid':>9s}{'sd*gam':>10s}{'SR':>8s}")
    rows = []
    for rung in list(tab.index.get_level_values("rung").unique()):
        for strat, label in STRATEGIES:
            # the benchmark only under "static", the two rules only under a rung
            if (rung == "static") != (strat == "un"):
                continue
            key = [(rung, g, label) for g in GAMMAS]
            if any(k not in tab.index for k in key):
                continue
            means = [tab.loc[k, "mean_ann_%"] for k in key]
            sds   = [tab.loc[k, "sd_ann_%"]   for k in key]
            A, B, resid, sdg, sr = gamma_decomposition(means, sds, GAMMAS)
            print(f"{rung + ', ' + label:34s}{A:9.3f}{B:10.3f}{resid:9.4f}"
                  f"{sdg:10.2f}{sr:8.3f}")
            rows.append((rung, label, A, B, resid, sdg, sr))
    gf = pd.DataFrame(rows, columns=["rung", "strategy", "A", "B", "resid",
                                     "sd_gamma", "sharpe_invariant"]
                      ).set_index(["rung", "strategy"])

    # A is the part of the mean that does not scale with 1/gamma, which is
    # the bill rate.  This line checks that.
    rf_ann = 100.0 * 12.0 * any_bt["rf"].mean()
    print(f"\n   mean annualised risk-free rate over this window: "
          f"{rf_ann:.3f} percent.  A should equal it.")

    # ---- Memmel tests against the benchmark and against M1 (Table 7.4) ---
    # Tests run on excess returns at gamma = 5.  Every weight scales with
    # 1/gamma, so the excess-return Sharpe ratio and the tests are the same
    # at every gamma.
    print("\n" + "-" * 78)
    print("SHARPE DIFFERENCES, gross of costs (Memmel 2003)")
    print("-" * 78)
    xs = {}
    for name, bt in bts.items():
        for strat, _ in STRATEGIES:
            xs[(name, strat)] = (bt[f"R_{strat}_g5"] - bt["rf"]).to_numpy(float)
    bench = xs[(list(bts)[0], "un")]

    print(f"{'':34s}{'SR':>8s}{'gap':>8s}{'se':>8s}{'t':>7s}{'p':>8s}{'rho':>8s}")
    print(f"{'static benchmark':34s}{sharpe(bench):8.3f}")
    diffs = []
    # each rung under both rules against the benchmark.  STRATEGIES[:2] is
    # the hard and the soft rule.
    for name in [n for n, _ in LADDER if n in bts]:
        for strat, label in STRATEGIES[:2]:
            s = xs[(name, strat)]
            d, se, t, p, rho = sharpe_diff_test(s, bench)
            print(f"{name + ', ' + label:34s}{sharpe(s):8.3f}{d:8.3f}"
                  f"{se:8.3f}{t:7.2f}{p:8.3f}{rho:8.4f}")
            diffs.append({"rung": name, "strategy": label, "vs": "benchmark",
                          "SR": sharpe(s), "gap": d, "se": se, "t": t,
                          "p": p, "rho": rho})

    # each rung against M1, the Ang-Bekaert specification
    if "M1" in bts:
        print(f"\n{'against M1, same rule':34s}{'gap':>8s}{'se':>8s}{'t':>7s}"
              f"{'p':>8s}{'rho':>8s}")
        for name in [n for n, _ in LADDER if n in bts and n != "M1"]:
            for strat, label in STRATEGIES[:2]:
                d, se, t, p, rho = sharpe_diff_test(xs[(name, strat)],
                                                    xs[("M1", strat)])
                print(f"{name + ', ' + label:34s}{d:8.3f}{se:8.3f}{t:7.2f}"
                      f"{p:8.3f}{rho:8.4f}")
                diffs.append({"rung": name, "strategy": label, "vs": "M1",
                              "SR": sharpe(xs[(name, strat)]), "gap": d,
                              "se": se, "t": t, "p": p, "rho": rho})
    print(f"   T = {len(bench)} months")

    # ---- the zero lower bound split (Table 7.6) ----------------------------
    zlb = any_bt["zlb"].to_numpy(bool)
    print("\n" + "-" * 78)
    print(f"ZERO LOWER BOUND SPLIT   {zlb.sum()} of {len(zlb)} months flagged")
    print("-" * 78)
    if zlb.sum() >= 12 and (~zlb).sum() >= 12:
        print(f"{'':34s}{'ZLB':>9s}{'non-ZLB':>10s}{'all':>9s}")
        for name in [n for n, _ in LADDER if n in bts]:
            for strat, label in STRATEGIES[:2]:
                s = xs[(name, strat)]
                print(f"{name + ', ' + label:34s}{sharpe(s[zlb]):9.3f}"
                      f"{sharpe(s[~zlb]):10.3f}{sharpe(s):9.3f}")
        print(f"{'static benchmark':34s}{sharpe(bench[zlb]):9.3f}"
              f"{sharpe(bench[~zlb]):10.3f}{sharpe(bench):9.3f}")
        print("   Short windows: read the levels, not the differences.")
    else:
        print("   one side of the split has fewer than twelve months; skipped")

    # ---- January 2009 (Section 7.5) ---------------------------------------
    d = pd.Period("2009-01", freq="M")
    if d in any_bt.index:
        print("\n" + "-" * 78)
        print("JANUARY 2009 HOLDING MONTH   filtered on data through 2008-12")
        print("-" * 78)
        print(f"{'':8s}{'P(calm)':>9s}{'regime':>8s}{'pi1 hard':>10s}"
              f"{'pi1 soft':>10s}")
        for name in [n for n, _ in LADDER if n in bts]:
            r = bts[name].loc[d]
            print(f"   {name:5s}{r['xi_filt_d']:9.3f}{int(r['regime']):8d}"
                  f"{r['pi1_h']:10.3f}{r['pi1_s']:10.3f}")

    tab.to_csv(os.path.join(OUT_DIR, f"ext_performance_{SPEC_TAG}.csv"))
    gf.to_csv(os.path.join(OUT_DIR, f"ext_gamma_free_{SPEC_TAG}.csv"))
    pd.DataFrame(diffs).to_csv(
        os.path.join(OUT_DIR, f"ext_sharpe_tests_{SPEC_TAG}.csv"), index=False)
    return tab, gf


# ==========================================================================
# Figures
# ==========================================================================
def figures(panel, fits, bts):
    """
    Four figures for the extension chapter.

    Parameters
    ----------
    panel : pd.DataFrame
        Modern panel, for the ZLB windows.
    fits : dict of str -> msm.MSResult
        Full-sample fits from part_a.
    bts : dict of str -> pd.DataFrame
        Backtests from load_backtests().

    Returns
    -------
    None
        Writes four PDFs to OUT_DIR.

    Notes
    -----
    The two backtest figures (wealth, mixing weight and equity weight) use
    the hard rule, the primary rule of the thesis, so they match the numbers
    of Table 7.4 and the text of Section 7.4.
    """
    plt.rcParams.update(common.PLOT_RC)
    top = [n for n, _ in LADDER if n in fits][-1]        # richest fitted rung
    dates = fits[top].index.to_timestamp("M")
    zlb = panel.loc[fits[top].index, "zlb"].to_numpy(bool)

    # ---- filtered probability of the volatile regime ---------------------
    fig, ax = plt.subplots(figsize=(6.8, 2.6))
    common.shade_zlb(ax, dates, zlb)
    ax.plot(dates, 1 - fits[top].xi_filt, color="black", lw=0.7)
    ax.set_ylabel("P(volatile regime)", fontsize=8.5)
    ax.set_ylim(0, 1); ax.margins(x=0.01)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT_DIR, f"ext_filtered_prob_{SPEC_TAG}.pdf"),
                bbox_inches="tight")
    plt.close(fig)

    # ---- stay-probabilities against each covariate -----------------------
    # One panel per driver of the richest rung, the others held at their mean,
    # which is zero after standardisation.
    zc = fits[top].zcols
    if zc:
        fig, axes = plt.subplots(1, len(zc), figsize=(1.9 * len(zc) + 0.6, 2.4),
                                 sharey=True)
        axes = np.atleast_1d(axes)
        grid = np.linspace(-2.5, 2.5, 200)
        p = fits[top].params
        for j, (c, ax) in enumerate(zip(zc, axes)):
            Zg = np.zeros((len(grid), len(zc)))
            Zg[:, j] = grid                          # move one driver at a time
            P, Q = msm.transition_probs(p, Zg)
            ax.plot(grid, P, color="black", lw=1.0, label="$P$")
            ax.plot(grid, Q, color="black", lw=1.0, ls=(0, (4, 3)), label="$Q$")
            ax.set_xlabel({"z_short": "Short rate", "z_term": "Term spread",
                           "z_ebp": "Excess bond premium", "z_vix": "VIX"}[c]
                          + " (sd)", fontsize=8.5)
            ax.set_ylim(0, 1)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        axes[0].set_ylabel("stay probability", fontsize=8.5)
        handles, _ = axes[0].get_legend_handles_labels()
        fig.legend(handles, ["$P$: stay in regime 1", "$Q$: stay in regime 2"],
                   loc="lower center", ncol=2, frameon=False, fontsize=8,
                   bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(pad=0.4, rect=(0, 0.14, 1, 1))   # room below the x labels
        fig.savefig(os.path.join(OUT_DIR, f"ext_transition_probs_{SPEC_TAG}.pdf"),
                    bbox_inches="tight")
        plt.close(fig)

    # ---- out-of-sample wealth by rung, hard rule -------------------------
    any_bt = next(iter(bts.values()))
    idx = any_bt.index.to_timestamp("M")
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    for name, bt in bts.items():
        ax.plot(idx, (1 + bt["R_rs_g5"]).cumprod(),
                color=RUNG_COLOURS.get(name, "0.5"), lw=1.0, label=name)
    ax.plot(idx, (1 + any_bt["R_un_g5"]).cumprod(), color=STATIC_COLOUR,
            lw=1.0, ls=(0, (4, 3)), label="static")
    ax.set_ylabel(f"Wealth from \\$1 in {OOS_START}", fontsize=8.5)
    ax.set_yscale("log")                          # compare growth, not levels
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper left")
    ax.margins(x=0.01)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT_DIR, f"ext_oos_wealth_{SPEC_TAG}.pdf"),
                bbox_inches="tight")
    plt.close(fig)

    # ---- mixing weight and equity weight, simplest against richest rung ---
    # Hard rule, so pi1 is P or 1 - Q in every month.
    lo = [n for n, _ in LADDER if n in bts][0]
    hi = [n for n, _ in LADDER if n in bts][-1]
    c_lo = RUNG_COLOURS.get(lo, "0.55")
    c_hi = RUNG_COLOURS.get(hi, "black")
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 4.2), sharex=True)
    axes[0].plot(idx, bts[lo]["pi1_h"], color=c_lo, lw=0.8, label=lo)
    axes[0].plot(idx, bts[hi]["pi1_h"], color=c_hi, lw=0.8, label=hi)
    axes[0].set_ylabel("$\\pi_1$, hard rule", fontsize=8.5)
    axes[0].set_ylim(0, 1)
    axes[1].plot(idx, bts[lo]["w_equity_rs_g5"], color=c_lo, lw=0.8, label=lo)
    axes[1].plot(idx, bts[hi]["w_equity_rs_g5"], color=c_hi, lw=0.8, label=hi)
    axes[1].axhline(0.0, color="0.85", lw=0.5)
    axes[1].set_ylabel("Equity weight ($\\gamma = 5$)", fontsize=8.5)
    for ax in axes:
        ax.margins(x=0.01)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles, _ = axes[0].get_legend_handles_labels()
    fig.legend(handles, [lo, hi], loc="lower center", ncol=2, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(pad=0.4, rect=(0, 0.06, 1, 1))     # room for the legend
    fig.savefig(os.path.join(OUT_DIR, f"ext_rung_paths_{SPEC_TAG}.pdf"),
                bbox_inches="tight")
    plt.close(fig)

    print(f"\nfigures written to {OUT_DIR}")


# ==========================================================================
# MAIN
# ==========================================================================
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    names = [n for n, _ in LADDER]
    rungs = [a.upper() for a in sys.argv[1:]] or names      # CLI subset
    bad = [r for r in rungs if r not in names]
    if bad:
        raise SystemExit(f"unknown rung(s) {bad}; choose from {names}")

    panel = load_modern()
    print(f"\nmodern panel: {panel.index.min()} to {panel.index.max()}, "
          f"{len(panel)} months, bond series '{BOND_TAG}'")
    print(f"running rungs: {', '.join(rungs)}\n")

    fits = part_a(panel)

    for name, zcols in LADDER:
        if name not in rungs:
            continue
        bt = part_b(panel, name, zcols)
        bt.to_csv(bt_path(name))

    bts = load_backtests()
    part_c(bts)
    figures(panel, fits, bts)

    print("\nAll output in", OUT_DIR)