"""
07d_noise_rung.py
=================
Placebo test of the M1 to M2 step (Section 7.4.1, Table 7.5).  Each placebo
rung replaces the term spread with an artificial AR(1) covariate that has
the persistence of the term spread and no content.  If placebo rungs match
M2, the gain at M2 is dilution of the short rate.  If they stay near M1,
the term spread carries information.

Run after 07_ladder.py (Part A cache, backtests of M1 and M2).  About 2.2
hours per seed.  The README describes the parallel run.  Writes
ext_noise_rungs_<SPEC_TAG>.csv.
"""

import os
import time

import numpy as np
import pandas as pd

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "common", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "00_common.py"))
common = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(common)

lad = common.load_module("07_ladder.py", "ladder")   # config, helpers, part_b
msm = lad.msm


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
# A first run used the first twelve seeds as a check.  The list was then
# extended to 100 seeds to build a null distribution, keeping the twelve in
# place as its first rows.  The full list was fixed before the extended run,
# and every draw in it is reported.
SEEDS = (101, 202, 303, 404, 505, 606, 707, 808, 909, 111, 222, 333) \
        + tuple(range(1001, 1089))                  # 12 + 88 = 100

# Parallel execution.  Worker CHUNK takes every NCHUNKS-th seed.  Each seed
# writes its own cache files, so workers never collide.  With NCHUNKS = 1 the
# script runs serially and produces the full output.
CHUNK   = int(os.environ.get("NOISE_CHUNK", "0"))
NCHUNKS = int(os.environ.get("NOISE_NCHUNKS", "1"))

# The covariate the artificial column stands in for, and the rung it is
# compared against.  BASE is what both share.
BASE_DRIVER    = "z_short"
REPLACED       = "z_term"      # persistence is matched to this column
RUNG_BASE      = "M1"          # z_short alone
RUNG_REFERENCE = "M2"          # z_short plus the term spread

N_RANDOM_FULL = 8              # starts for the full-sample fit, as in 07
SATURATION    = 10.0           # |slope| above this is a spurious mode, not a fit


# ==========================================================================
# The artificial covariate
# ==========================================================================
def ar1_rho(x):
    """
    Estimate the AR(1) coefficient of a series by ordinary least squares.

    Parameters
    ----------
    x : array_like
        Series in time order, without gaps.

    Returns
    -------
    float
        Slope of x_t on a constant and x_{t-1}, capped at 0.999 so the
        generated covariate stays stationary.
    """
    x = np.asarray(x, dtype=float)
    # regress x_t on a constant and x_{t-1}
    X = np.column_stack([np.ones(len(x) - 1), x[:-1]])
    rho = float(np.linalg.lstsq(X, x[1:], rcond=None)[0][1])
    return min(rho, 0.999)


def ar1_noise(n, rho, seed):
    """
    Draw a stationary AR(1) path with unit unconditional variance.

    Parameters
    ----------
    n : int
        Length of the panel.
    rho : float
        Persistence, from ar1_rho() on the term spread.
    seed : int
        Generator seed, so the path is identical on every rerun.

    Returns
    -------
    np.ndarray, length n

    Notes
    -----
    The first value is drawn from N(0, 1) and the innovations are scaled
    by sqrt(1 - rho^2), so the path is stationary from the start.
    """
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0]                                   # unconditional draw
    s = np.sqrt(1.0 - rho ** 2)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + s * e[t]
    return x


# ==========================================================================
# Evaluation
# ==========================================================================
def excess(bt, strat, gamma=5):
    """
    Return the monthly excess return of one strategy at one gamma.

    Parameters
    ----------
    bt : pd.DataFrame
        A backtest written by part_b of 07_ladder.py.
    strat : str
        Strategy tag: "rs", "sf" or "un".
    gamma : int, default 5
        Risk aversion.  The Sharpe ratio of the result does not depend on
        it, since every weight scales with 1/gamma.

    Returns
    -------
    np.ndarray, length T
        R_<strat>_g<gamma> minus rf, as decimals.
    """
    return (bt[f"R_{strat}_g{gamma}"] - bt["rf"]).to_numpy(dtype=float)


def load_rung(name):
    """
    Read one rung's backtest from disk.

    Parameters
    ----------
    name : str
        Rung label, e.g. "M1", or the label of a placebo rung.

    Returns
    -------
    pd.DataFrame
        The backtest at lad.bt_path(name), with a monthly PeriodIndex.

    Raises
    ------
    SystemExit
        The file is missing.
    """
    f = lad.bt_path(name)
    if not os.path.exists(f):
        raise SystemExit(f"missing {os.path.basename(f)}; run 07_ladder.py "
                         f"first")
    return common.read_backtest(os.path.basename(f))


def permutation_p(draws, observed):
    """
    Compute the one-sided permutation p-value of an observed statistic
    against the placebo draws.

    Parameters
    ----------
    draws : array_like
        The null draws, here the Sharpe ratios of the placebo rungs.
    observed : float
        The statistic tested, here the Sharpe ratio of M2.

    Returns
    -------
    count : int
        Number of draws at or above the observed value.
    p : float
        (1 + count) / (N + 1).

    Notes
    -----
    The observed value counts as one of the N + 1 possible draws, so a
    result above every draw gives 1 / (N + 1), not zero.
    """
    draws = np.asarray(draws, dtype=float)
    n_beat = int(np.sum(draws >= observed))
    return n_beat, (n_beat + 1) / (len(draws) + 1)


# ==========================================================================
# MAIN
# ==========================================================================
if __name__ == "__main__":
    # load_modern() checks the panel against the drivers named in LADDER, so
    # it has to run before the artificial rungs are appended to it
    panel = lad.load_modern()

    rho = ar1_rho(panel[REPLACED].to_numpy(dtype=float))
    print(f"\npanel {panel.index.min()} to {panel.index.max()}, "
          f"{len(panel)} months")
    print(f"AR(1) persistence of {REPLACED}: {rho:.4f}   "
          f"artificial columns are matched to it")

    seeds_run = SEEDS[CHUNK::NCHUNKS]
    if NCHUNKS > 1:
        print(f"worker {CHUNK + 1} of {NCHUNKS}, {len(seeds_run)} of "
              f"{len(SEEDS)} seeds")
    else:
        print(f"{len(seeds_run)} seeds")
    print()

    base_bt = load_rung(RUNG_BASE)
    ref_bt  = load_rung(RUNG_REFERENCE)
    bench   = excess(base_bt, "un")

    # M1's cached full-sample fit, for the likelihood-ratio comparison
    f_base = os.path.join(lad.OUT_DIR,
                          f"ext_partA_{RUNG_BASE}_{lad.SPEC_TAG}.npz")
    if not os.path.exists(f_base):
        raise SystemExit(f"missing {os.path.basename(f_base)}; run "
                         f"07_ladder.py first")
    zc_base = dict(lad.LADDER)[RUNG_BASE]
    std_base, _, _ = lad.standardised(panel, zc_base)
    m_base = msm.evaluate(std_base, zc_base, np.load(f_base)["theta"],
                          strict=False)                    # refilter, no refit

    # the reference rung's own in-sample p-value, from its cache
    f_ref = os.path.join(lad.OUT_DIR,
                         f"ext_partA_{RUNG_REFERENCE}_{lad.SPEC_TAG}.npz")
    zc_ref = dict(lad.LADDER)[RUNG_REFERENCE]
    std_ref, _, _ = lad.standardised(panel, zc_ref)
    m_ref = msm.evaluate(std_ref, zc_ref, np.load(f_ref)["theta"],
                         strict=False)
    _, _, p_ref = msm.lr_test(m_base, m_ref)

    results = []
    for seed in seeds_run:
        name = f"N{seed}"
        zcols = [BASE_DRIVER, "z_noise"]

        # inject the artificial column, then register the rung so the helpers
        # of 07 recognise it.  LADDER is only read, never written back to disk.
        panel = panel.copy()
        panel["z_noise"] = ar1_noise(len(panel), rho, seed)
        if name not in [n for n, _ in lad.LADDER]:
            lad.LADDER.append((name, zcols))

        print("=" * 78)
        print(f"NOISE RUNG {name}   seed {seed}")
        print("=" * 78)

        # ---- full sample, against the base rung --------------------------
        cache = os.path.join(lad.OUT_DIR,
                             f"ext_partA_{name}_{lad.SPEC_TAG}.npz")
        std, mu, sd = lad.standardised(panel, zcols)
        if lad.RESUME and os.path.exists(cache):
            res = msm.evaluate(std, zcols, np.load(cache)["theta"],
                               strict=False)
            print("   full-sample fit restored from cache")
        else:
            t0 = time.time()
            res = msm.fit(std, zcols, n_random=N_RANDOM_FULL,
                          theta0=msm.nest_start(m_base.theta, 1, 2),
                          strict=False)
            np.savez(cache, theta=res.theta)
            print(f"   full-sample fit ({time.time() - t0:.0f} seconds)")
        stat, dof, pval = msm.lr_test(m_base, res)
        b1, b2 = res.params["b"][0][1], res.params["b"][1][1]
        print(f"   LR against {RUNG_BASE}: {stat:.3f} on {dof} df, "
              f"p = {pval:.3g}")
        print(f"   slopes on the artificial column: {b1:+.3f} and {b2:+.3f}")
        # a slope of this size in standardised units drives the logistic to
        # its bounds, which is a spurious mode rather than a fit.  It happens
        # because a switching likelihood of this dimension has near-flat
        # ridges the optimiser can climb on any column at all.
        if max(abs(b1), abs(b2)) > SATURATION:
            print("   WARNING: the logistic is saturated on a column that "
                  "contains nothing.  Read the out-of-sample columns below "
                  "and disregard this rung's in-sample p-value.")

        # ---- recursive backtest -----------------------------------------
        if lad.RESUME and os.path.exists(lad.bt_path(name)):
            bt = load_rung(name)
            print(f"   backtest restored from "
                  f"{os.path.basename(lad.bt_path(name))}")
        else:
            bt = lad.part_b(panel, name, zcols)
            bt.to_csv(lad.bt_path(name))

        results.append({"name": name, "seed": seed, "lr": stat, "p": pval,
                        "xs_h": excess(bt, "rs"), "xs_s": excess(bt, "sf")})

    # a worker holds only part of the seed set, so neither the table nor the
    # permutation p-value would be meaningful.  The caches are what it is for.
    if NCHUNKS > 1:
        print(f"\nworker {CHUNK + 1} of {NCHUNKS} finished, "
              f"{len(seeds_run)} caches written.  Rerun with NCHUNKS unset "
              f"and RESUME on for the full table.")
        raise SystemExit

    # ---- the comparison ---------------------------------------------------
    print("\n" + "=" * 78)
    print("DOES A COLUMN THAT CONTAINS NOTHING REPRODUCE THE GAIN?")
    print("=" * 78)
    print(f"{'':22s}{'SR hard':>9s}{'SR soft':>9s}"
          f"{'gap vs ' + RUNG_BASE:>14s}{'t':>7s}{'p':>8s}"
          f"{'in-sample p':>13s}")

    # ---- comparison rows: benchmark, M1 and M2 -----------------------------
    # One line each, in the layout of Table 7.5.  The gap, t and p are the
    # Memmel test of the hard rule against M1 (RUNG_BASE) under the hard
    # rule.  They are left blank for the benchmark and for M1 itself.  The
    # in-sample p is the LR test against M1, so it exists only for M2.
    rows = []
    for lab, xs_h, xs_s, ins_p in (
            ("benchmark", bench, bench, np.nan),
            (RUNG_BASE, excess(base_bt, "rs"), excess(base_bt, "sf"), np.nan),
            (RUNG_REFERENCE, excess(ref_bt, "rs"), excess(ref_bt, "sf"),
             p_ref)):
        d, se, t, p, _ = lad.sharpe_diff_test(xs_h, excess(base_bt, "rs"))
        gap = "--" if lab in ("benchmark", RUNG_BASE) else f"{d:14.3f}"
        tt  = "--" if lab in ("benchmark", RUNG_BASE) else f"{t:7.2f}"
        pp  = "--" if lab in ("benchmark", RUNG_BASE) else f"{p:8.3f}"
        ip  = "--" if not np.isfinite(ins_p) else f"{ins_p:13.3f}"
        print(f"{lab:22s}{lad.sharpe(xs_h):9.3f}{lad.sharpe(xs_s):9.3f}"
              f"{gap:>14s}{tt:>7s}{pp:>8s}{ip:>13s}")

    # ---- one line per placebo rung (Appendix table) ------------------------
    # Sharpe ratio under both rules, the Memmel test of the hard rule
    # against M1, and the in-sample LR p-value against M1.  Each row is also
    # kept for the CSV.
    print()
    for r in results:
        d, se, t, p, _ = lad.sharpe_diff_test(r["xs_h"], excess(base_bt, "rs"))
        print(f"{r['name'] + ' (noise)':22s}{lad.sharpe(r['xs_h']):9.3f}"
              f"{lad.sharpe(r['xs_s']):9.3f}{d:14.3f}{t:7.2f}{p:8.3f}"
              f"{r['p']:13.3f}")
        rows.append({"rung": r["name"], "seed": r["seed"],
                     "SR_hard": lad.sharpe(r["xs_h"]),
                     "SR_soft": lad.sharpe(r["xs_s"]),
                     "gap_vs_base": d, "t": t, "p": p,
                     "LR_insample": r["lr"], "p_insample": r["p"]})

    # Sharpe ratios for the summary below: the placebo draws under each
    # rule, M1 under the hard rule, and M2 under both rules
    sr      = np.array([lad.sharpe(r["xs_h"]) for r in results])
    sr_soft = np.array([lad.sharpe(r["xs_s"]) for r in results])
    sr_base   = lad.sharpe(excess(base_bt, "rs"))
    sr_ref    = lad.sharpe(excess(ref_bt, "rs"))
    sr_ref_s  = lad.sharpe(excess(ref_bt, "sf"))

    print("\n" + "-" * 78)
    print(f"   {RUNG_BASE} {sr_base:.3f}   {RUNG_REFERENCE} {sr_ref:.3f}   "
          f"noise mean {sr.mean():.3f}, range {sr.min():.3f} to {sr.max():.3f}")
    # how far the average noise draw travels from the base rung towards the
    # reference rung.
    share = (sr.mean() - sr_base) / (sr_ref - sr_base) if sr_ref != sr_base \
        else np.nan
    print(f"   the average noise draw closes {share:.0%} of the distance "
          f"from {RUNG_BASE} to {RUNG_REFERENCE}")

    print()
    for lab, draws, obs in (("hard", sr, sr_ref),
                            ("soft", sr_soft, sr_ref_s)):
        n_beat, p_perm = permutation_p(draws, obs)
        z = (obs - draws.mean()) / draws.std(ddof=1)
        print(f"   {lab}: {n_beat} of {len(draws)} draws reach "
              f"{RUNG_REFERENCE}, permutation p = {p_perm:.4f} "
              f"(floor {1 / (len(draws) + 1):.4f}), "
              f"{z:+.2f} sd above the placebo mean")

    # ---- counts quoted in Section 7.4.1  ------------------------------------
    nz = pd.DataFrame(rows)
    sr_bench = lad.sharpe(bench)
    t_ref = lad.sharpe_diff_test(excess(ref_bt, "rs"), excess(base_bt, "rs"))[2]
    fit = pd.read_csv(os.path.join(lad.OUT_DIR,
                                   f"ext_ladder_fit_{lad.SPEC_TAG}.csv"),
                      index_col=0)
    print("\n" + "-" * 78)
    print(f"   draws beating {RUNG_BASE}, hard: "
          f"{(nz['SR_hard'] > sr_base).sum()} of {len(nz)}")
    print(f"   draws beating the benchmark, hard: "
          f"{(nz['SR_hard'] > sr_bench).sum()} of {len(nz)}")
    print(f"   mean gain over {RUNG_BASE}: {nz['gap_vs_base'].mean():.3f}, "
          f"s.e. {nz['gap_vs_base'].std(ddof=1) / np.sqrt(len(nz)):.4f}")
    print(f"   draws with t >= 1.96: {(nz['t'] >= 1.96).sum()}, "
          f"largest t {nz['t'].max():.2f}")
    print(f"   {RUNG_REFERENCE} t against {RUNG_BASE}: {t_ref:.2f}, "
          f"exceeded by {(nz['t'] > t_ref).sum()} draws")
    for rung in ("M2", "M3", "M4"):
        lr = float(fit.loc[rung, "LR_vs_prev"])
        print(f"   placebo LR above {rung}'s {lr:.3f}: "
              f"{(nz['LR_insample'] > lr).sum()} of {len(nz)}")

    pd.DataFrame(rows).to_csv(
        os.path.join(lad.OUT_DIR, f"ext_noise_rungs_{lad.SPEC_TAG}.csv"),
        index=False)
    print(f"\nwritten: ext_noise_rungs_{lad.SPEC_TAG}.csv in {lad.OUT_DIR}")