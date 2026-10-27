"""
08_costs.py
===========
Transaction costs on the ladder backtests (Chapter 8).  Run 07_ladder.py
first.  Reads panel_modern.csv and the SPEC_TAG-stamped ext_backtest files.
Nothing is re-estimated.

Turnover is one-way in the two risky assets, as defined in Section 4.6.
Cash is the residual, so its trade is implied.  The charge in month t is the
cost c times the traded turnover, the distance from the drifted position to
the new target.

    PART A  target against traded turnover (Table 8.1)
    PART B  proportional costs, rebalanced every month: the gamma spread of
            the net Sharpe ratio, the cost grid with break-even costs
            (Table 8.2), the gross advantage per unit of extra turnover, and
            the Memmel test at REF_BP (Section 8.2, Chapter 9)
    PART C  no-trade band (Table 8.3)
    PART D  ZLB split net of REF_BP (Table 8.4)
    Figures 8.1 and 8.2

Net results depend on gamma, so every one is at COST_GAMMA.

Approx. running time: 60s
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "common", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "00_common.py"))
common = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(common)

lad = common.load_module("07_ladder.py", "ladder")   # the ladder's settings


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
# Everything the two scripts must agree on is read off 07 rather than
# restated here, so there is nothing left to keep in step by hand.
OUT_DIR       = common.OUT_DIR
PANEL_FILE    = lad.PANEL_FILE
BOND_COL      = lad.BOND_COL
SPEC_TAG      = lad.SPEC_TAG
STRATEGIES    = common.STRATEGIES
GAMMAS        = common.GAMMAS
RUNG_COLOURS  = common.RUNG_COLOURS
STATIC_COLOUR = common.STATIC_COLOUR

COST_GAMMA = 5                 # net results are gamma-dependent, see the header

sharpe           = common.sharpe
sharpe_diff_test = common.sharpe_diff_test
weights          = common.weight_path

# One-way proportional cost, basis points of the weight traded.
COST_BP = [0.0, 5.0, 10.0, 25.0, 50.0]

# No-trade band, in weight units.  An asset is left alone until its drift
# away from target exceeds the threshold.
BANDS = [0.01, 0.025, 0.05]

# The setup trade out of cash in the first month is a real cost and is
# charged.  Reported turnover excludes it, so the mean is comparable with the
# turnover column of 07.
CHARGE_SETUP = True

# Break-even search range, basis points one way.
BE_MAX_BP = 2000.0

# Reference cost for the gamma spread, the Memmel test and the ZLB split,
# basis points one way.
REF_BP = 25.0


# ==========================================================================
# Loading
# ==========================================================================
def load_panel(index):
    """
    Read the realised excess returns of the two risky assets over the
    backtest months.

    Parameters
    ----------
    index : pd.PeriodIndex
        Months of the backtest.

    Returns
    -------
    np.ndarray, shape (T, 2)
        Excess returns on (bond, equity), decimal.

    Raises
    ------
    SystemExit
        The panel does not cover the backtest.
    """
    panel = common.read_panel(PANEL_FILE)
    missing = index.difference(panel.index)
    if len(missing):
        raise SystemExit(f"panel does not cover {list(missing[:6])}")
    return panel.loc[index, [BOND_COL, "re_x"]].to_numpy(dtype=float) / 100.0


def load_backtests():
    """
    Read every rung's backtest written by 07 and check it against the panel.

    Returns
    -------
    bts : dict of str -> pd.DataFrame
        One backtest per rung name, in LADDER order.
    x : np.ndarray, shape (T, 2)
        Realised excess returns on (bond, equity), as decimals.

    Raises
    ------
    SystemExit
        No backtest exists, the rungs cover different months, or the weights
        and the panel do not reproduce the returns 07 recorded.
    """
    # every rung 07 has written, all covering the same months
    bts = lad.load_backtests()
    idx = next(iter(bts.values())).index

    # Rebuilding each recorded return as rf + w'x from the stored weights
    # checks that the panel, the bond series and the weights are the ones 07
    # used.  The cost calculations rely on exactly these weights and returns.
    x = load_panel(idx)
    for name, bt in bts.items():
        rf = bt["rf"].to_numpy(dtype=float)
        for s, _ in STRATEGIES:
            for g in GAMMAS:
                W = weights(bt, s, g)
                recon = rf + (W * x).sum(axis=1)
                if not np.allclose(recon, bt[f"R_{s}_g{g}"].to_numpy(float),
                                   atol=1e-12):
                    raise SystemExit(
                        f"reconstruction mismatch, rung {name}, strategy "
                        f"'{s}', gamma {g}: check BOND_COL and SPEC_TAG "
                        f"against 07_ladder.py")
    return bts, x


# ==========================================================================
# Cost accounting
# ==========================================================================
def drifted(w_held, x_row, rf):
    """
    Return the weights of a held position after one month of returns,
    before any trade.

    Parameters
    ----------
    w_held : np.ndarray, length 2
        Weights on (bond, equity) held over the month.
    x_row : np.ndarray, length 2
        Realised excess returns of that month, as decimals.
    rf : float
        Bill return of that month, as a decimal.

    Returns
    -------
    np.ndarray, length 2
        Drifted weights w * (1 + rf + x) / (1 + rf + w'x).

    Notes
    -----
    If the portfolio loses all its value, 1 + rf + w'x <= 0, the held
    weights are returned unchanged so the accounting can continue.
    """
    R_assets = rf + x_row                       # total return of each asset
    R_port = rf + float(w_held @ x_row)         # total return of the portfolio
    if 1.0 + R_port <= 0.0:
        return w_held.copy()                    # portfolio wiped out
    return w_held * (1.0 + R_assets) / (1.0 + R_port)


def run_costs(W, x, rf, cost, band=None):
    """
    Compute the net return path of one weight path under one cost
    treatment.

    Parameters
    ----------
    W : np.ndarray, shape (T, 2)
        Target weights on (bond, equity), one row per holding month.
    x : np.ndarray, shape (T, 2)
        Realised excess returns, as decimals.
    rf : np.ndarray, length T
        Bill return of each month, as decimals.
    cost : float
        One-way proportional cost per unit of weight traded, as a decimal.
    band : float, optional
        No-trade threshold in weight units.  None rebalances to target every
        month.

    Returns
    -------
    dict
        xs_net : np.ndarray, length T
            Net excess return.
        xs_gross : np.ndarray, length T
            Gross excess return of the held weights.
        traded : np.ndarray, length T
            Traded turnover charged in each month.
        target : np.ndarray, length T
            Target turnover, NaN in the first month.
        n_trades : int
            Months in which at least one asset was traded.

    Notes
    -----
    The trade at the start of month t is charged against month t's return.
    The first month's setup trade out of cash is charged only if
    CHARGE_SETUP is set.  Under a band the held weights depend on the path,
    so the months are walked forward in order.
    """
    T = len(W)
    held = np.zeros((T, 2))
    traded = np.zeros(T)
    target = np.full(T, np.nan)
    n_trades = 0

    pos = np.zeros(2)                            # start in cash
    for t in range(T):
        tgt = W[t]
        dev = tgt - pos                          # distance from target
        if band is None:
            new = tgt.copy()                     # rebalance every month
        else:
            new = pos.copy()
            hit = np.abs(dev) > band             # assets outside the band
            new[hit] = tgt[hit]                  # trade back to target
        trade = float(np.abs(new - pos).sum())   # one-way turnover
        if trade > 0:
            n_trades += 1
        if t == 0 and not CHARGE_SETUP:
            traded[t] = 0.0
        else:
            traded[t] = trade
        if t > 0:
            target[t] = float(np.abs(W[t] - W[t - 1]).sum())
        held[t] = new
        pos = drifted(new, x[t], rf[t])          # carry into the next month

    xs_gross = (held * x).sum(axis=1)
    xs_net = xs_gross - cost * traded
    return {"xs_net": xs_net, "xs_gross": xs_gross, "traded": traded,
            "target": target, "n_trades": n_trades}


def break_even(W_s, W_u, x, rf, band=None, hi_bp=BE_MAX_BP):
    """
    Find the cost at which a strategy's net Sharpe ratio falls to the
    benchmark's.

    Parameters
    ----------
    W_s, W_u : np.ndarray, shape (T, 2)
        Target weight paths of the strategy and of the static benchmark.
    x : np.ndarray, shape (T, 2)
        Realised excess returns, as decimals.
    rf : np.ndarray, length T
        Bill return of each month, as decimals.
    band : float, optional
        No-trade threshold applied to both paths.  None rebalances every
        month.
    hi_bp : float, default BE_MAX_BP
        Upper end of the search, in basis points one way.

    Returns
    -------
    float
        Break-even cost in basis points one way.  0 if the strategy is
        already behind at zero cost, NaN if it is still ahead at hi_bp.

    Notes
    -----
    Both paths pay the same cost, so the benchmark pays for its own trading
    too.  The search is a bisection and assumes the two net Sharpe ratios
    cross only once.
    """
    def gap(bp):
        """Net Sharpe of the strategy minus that of the benchmark, at bp."""
        c = bp / 10000.0
        a = run_costs(W_s, x, rf, c, band)["xs_net"]
        b = run_costs(W_u, x, rf, c, band)["xs_net"]
        return sharpe(a) - sharpe(b)

    g0 = gap(0.0)
    if not np.isfinite(g0) or g0 <= 0:
        return 0.0                               # behind before any cost
    if gap(hi_bp) > 0:
        return np.nan                            # still ahead at the top
    lo, hi = 0.0, hi_bp
    for _ in range(40):                          # 40 halvings, well below 0.01 bp
        mid = 0.5 * (lo + hi)
        if gap(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ==========================================================================
# PART A: turnover
# ==========================================================================
def part_a(bts, x):
    """
    Compare target and traded turnover for every rung (Table 8.1).

    Parameters
    ----------
    bts : dict of str -> pd.DataFrame
        Output of load_backtests().
    x : np.ndarray, shape (T, 2)
        Realised excess returns on (bond, equity), as decimals.

    Returns
    -------
    pd.DataFrame
        One row per (rung, strategy), with the monthly turnover statistics
        at COST_GAMMA.  Also written to ext_turnover_<SPEC_TAG>.csv.

    Notes
    -----
    Turnover is one-way in the two risky assets.  The setup trade is left
    out of every statistic, so target turnover matches the turnover column
    of 07.  The last column is traded turnover times twelve.
    """
    print("=" * 78)
    print(f"PART A   Turnover, gamma = {COST_GAMMA}, one way, two risky assets")
    print("=" * 78)
    print(f"{'':34s}{'target':>9s}{'traded':>9s}{'ratio':>8s}{'max':>8s}"
          f"{'ann.':>8s}")

    rf = next(iter(bts.values()))["rf"].to_numpy(dtype=float)
    recs = []
    for name, bt in bts.items():
        for strat, label in STRATEGIES:
            if strat == "un" and name != list(bts)[0]:
                continue                          # identical in every rung
            # run at zero cost: only the turnover is needed here
            W = weights(bt, strat, COST_GAMMA)
            r = run_costs(W, x, rf, 0.0)
            tgt = np.nanmean(r["target"])         # first month is NaN
            trd = r["traded"][1:].mean()          # setup trade excluded
            rung = "static" if strat == "un" else name
            print(f"{rung + ', ' + label:34s}{tgt:9.4f}{trd:9.4f}"
                  f"{trd / tgt if tgt else np.nan:8.3f}"
                  f"{r['traded'][1:].max():8.4f}"
                  f"{12 * trd:8.3f}")
            recs.append({"rung": rung, "strategy": label,
                         "target_pm": tgt, "traded_pm": trd,
                         "traded_max": r["traded"][1:].max(),
                         "traded_ann": 12 * trd})
    print("\n   `target` is the distance between two consecutive target")
    print("   weights, `traded` the distance from the drifted position to the")
    print("   new target.  The ratio is traded over target turnover.")
    tab = pd.DataFrame(recs).set_index(["rung", "strategy"])
    tab.to_csv(os.path.join(OUT_DIR, f"ext_turnover_{SPEC_TAG}.csv"))
    return tab


# ==========================================================================
# PART B: proportional costs, full rebalancing
# ==========================================================================
def part_b(bts, x):
    """
    Compute net Sharpe ratios on the cost grid, rebalancing to target every
    month (Table 8.2).

    Parameters
    ----------
    bts : dict of str -> pd.DataFrame
        Output of load_backtests().
    x : np.ndarray, shape (T, 2)
        Realised excess returns on (bond, equity), as decimals.

    Returns
    -------
    pd.DataFrame
        Net annualised Sharpe ratio by (rung, strategy) at each cost level,
        and the break-even cost against the benchmark charged the same cost.
    """
    print("\n" + "=" * 78)
    print("PART B   Proportional costs, rebalanced every month")
    print("=" * 78)

    first = list(bts)[0]
    rf = bts[first]["rf"].to_numpy(dtype=float)
    W_un = weights(bts[first], "un", COST_GAMMA)      # benchmark, same in every rung

    # ---- how far the net Sharpe ratio moves with gamma --------------------
    # Costs break the 1/gamma invariance, since a smaller position drifts
    # less in weight units.  Checked on the first rung at REF_BP.
    print(f"\n--- net Sharpe ratio across the gamma grid, {REF_BP:.0f} bp "
          + "-" * 28)
    print(f"{'':34s}" + "".join(f"{'g=' + str(g):>9s}" for g in GAMMAS)
          + f"{'spread':>9s}")
    for name, bt in list(bts.items())[:1]:
        for strat, label in STRATEGIES:
            rung = "static" if strat == "un" else name
            vals = [sharpe(run_costs(weights(bt, strat, g), x, rf,
                                     REF_BP / 10000.0)["xs_net"])
                    for g in GAMMAS]
            print(f"{rung + ', ' + label:34s}"
                  + "".join(f"{v:9.4f}" for v in vals)
                  + f"{np.nanmax(vals) - np.nanmin(vals):9.4f}")
    print(f"   Everything below is at gamma = {COST_GAMMA}.")

    # ---- the cost grid (Table 8.2) ----------------------------------------
    print("\n--- net annualised Sharpe ratio " + "-" * 46)
    print(f"{'':34s}" + "".join(f"{str(int(c)) + 'bp':>9s}" for c in COST_BP)
          + f"{'break-even':>12s}")
    recs = []
    for name, bt in bts.items():
        for strat, label in STRATEGIES:
            if strat == "un" and name != first:
                continue                          # benchmark reported once
            W = weights(bt, strat, COST_GAMMA)
            srs = [sharpe(run_costs(W, x, rf, c / 10000.0)["xs_net"])
                   for c in COST_BP]
            rung = "static" if strat == "un" else name
            if strat == "un":
                be = np.nan                       # no break-even against itself
                be_txt = f"{'--':>12s}"
            else:
                be = break_even(W, W_un, x, rf)
                be_txt = (f"{be:12.1f}" if np.isfinite(be)
                          else f"{'>' + str(int(BE_MAX_BP)):>12s}")
            print(f"{rung + ', ' + label:34s}"
                  + "".join(f"{v:9.3f}" for v in srs) + be_txt)
            rec = {"rung": rung, "strategy": label, "break_even_bp": be}
            rec.update({f"SR_{int(c)}bp": v for c, v in zip(COST_BP, srs)})
            recs.append(rec)
    print("\n   Break-even is the one-way cost in basis points at which the")
    print("   strategy's net Sharpe ratio meets the benchmark's under the same")
    print("   cost.  Zero means it is behind already at no cost.")

    # ---- gross advantage per unit of extra turnover (Section 8.2) ---------
    # The break-even cost orders the rungs by the gross Sharpe advantage over
    # the benchmark divided by the excess traded turnover.  Hard rule.
    print("\n--- gross advantage per unit of extra traded turnover, hard rule "
          + "-" * 13)
    print(f"{'':34s}{'SR gap':>9s}{'extra turn':>12s}{'ratio':>8s}")
    sr_un0 = sharpe(run_costs(W_un, x, rf, 0.0)["xs_net"])
    trd_un = run_costs(W_un, x, rf, 0.0)["traded"][1:].mean()
    for name, bt in bts.items():
        r0 = run_costs(weights(bt, "rs", COST_GAMMA), x, rf, 0.0)
        d_sr = sharpe(r0["xs_net"]) - sr_un0
        d_tr = r0["traded"][1:].mean() - trd_un
        print(f"{name:34s}{d_sr:9.3f}{d_tr:12.3f}"
              f"{d_sr / d_tr if d_tr else np.nan:8.3f}")

    # ---- Memmel test on net returns at REF_BP (Chapter 9) -----------------
    # Both sides pay the same cost.
    print(f"\n--- net Sharpe against the benchmark at {REF_BP:.0f} bp "
          f"(Memmel 2003) " + "-" * 19)
    print(f"{'':34s}{'gap':>8s}{'se':>8s}{'t':>7s}{'p':>8s}{'rho':>8s}")
    tests = []
    xs_u = run_costs(W_un, x, rf, REF_BP / 10000.0)["xs_net"]
    for name, bt in bts.items():
        for strat, label in STRATEGIES[:2]:       # hard and soft rule
            xs_s = run_costs(weights(bt, strat, COST_GAMMA), x, rf,
                             REF_BP / 10000.0)["xs_net"]
            d, se, t, p, rho = sharpe_diff_test(xs_s, xs_u)
            print(f"{name + ', ' + label:34s}{d:8.3f}{se:8.3f}"
                  f"{t:7.2f}{p:8.3f}{rho:8.4f}")
            tests.append({"rung": name, "strategy": label, "cost_bp": REF_BP,
                          "gap": d, "se": se, "t": t, "p": p, "rho": rho})
    print(f"   T = {len(rf)} months")

    tab = pd.DataFrame(recs).set_index(["rung", "strategy"])
    tab.to_csv(os.path.join(OUT_DIR, f"ext_costs_full_{SPEC_TAG}.csv"))
    pd.DataFrame(tests).to_csv(
        os.path.join(OUT_DIR, f"ext_cost_tests_{SPEC_TAG}.csv"), index=False)
    return tab


# ==========================================================================
# PART C: no-trade band
# ==========================================================================
def part_c(bts, x):
    """
    Compute traded turnover, trade count and break-even cost under a no-trade
    band (Table 8.3).

    Parameters
    ----------
    bts : dict of str -> pd.DataFrame
        Output of load_backtests().
    x : np.ndarray, shape (T, 2)
        Realised excess returns on (bond, equity), as decimals.

    Returns
    -------
    pd.DataFrame
        Traded turnover, trade count and break-even cost by (rung, band) at
        COST_GAMMA, hard rule and benchmark.  Also written to
        ext_costs_band_<SPEC_TAG>.csv.

    Notes
    -----
    The band is in weight units and weights scale with 1/gamma, so the
    results hold at COST_GAMMA only.  No band is the monthly rebalancing of
    Part B.  For the break-even, the benchmark trades under the same band.
    """
    print("\n" + "=" * 78)
    print(f"PART C   No-trade band, gamma = {COST_GAMMA}, hard rule, trade to "
          f"target")
    print("=" * 78)
    print(f"{'':20s}{'band':>7s}{'turn':>8s}{'trades':>8s}{'break-even':>12s}")

    first = list(bts)[0]
    rf = bts[first]["rf"].to_numpy(dtype=float)
    W_un = weights(bts[first], "un", COST_GAMMA)      # benchmark, same in every rung
    T = len(rf)

    # the benchmark, then the hard rule of every rung
    paths = [("static", W_un)] + [(name, weights(bt, "rs", COST_GAMMA))
                                  for name, bt in bts.items()]
    recs = []
    for rung, W in paths:
        for band in [None] + BANDS:               # None: rebalance every month
            # turnover and trade count do not depend on the cost
            r0 = run_costs(W, x, rf, 0.0, band)
            if rung == "static":
                be_txt, be = f"{'--':>12s}", np.nan
            else:
                be = break_even(W, W_un, x, rf, band)
                be_txt = (f"{be:12.1f}" if np.isfinite(be)
                          else f"{'>' + str(int(BE_MAX_BP)):>12s}")
            lab = rung if band is None else ""
            print(f"{lab:20s}"
                  f"{'none' if band is None else f'{band:.3f}':>7s}"
                  f"{r0['traded'][1:].mean():8.4f}"
                  f"{r0['n_trades']:8d}" + be_txt)
            recs.append({"rung": rung,
                         "band": 0.0 if band is None else band,
                         "traded_pm": r0["traded"][1:].mean(),
                         "n_trades": r0["n_trades"], "months": T,
                         "break_even_bp": be})
    print("\n   `trades` counts months in which at least one asset was traded,")
    print(f"   out of {T}.")

    tab = pd.DataFrame(recs).set_index(["rung", "band"])
    tab.to_csv(os.path.join(OUT_DIR, f"ext_costs_band_{SPEC_TAG}.csv"))
    return tab


# ==========================================================================
# PART D: the zero lower bound split
# ==========================================================================
def part_d(bts, x):
    """
    Split turnover and net performance into ZLB and non-ZLB months
    (Table 8.4).

    Parameters
    ----------
    bts : dict of str -> pd.DataFrame
        Output of load_backtests().
    x : np.ndarray, shape (T, 2)
        Realised excess returns on (bond, equity), as decimals.

    Returns
    -------
    pd.DataFrame
        Traded turnover and net Sharpe ratio at REF_BP and
        COST_GAMMA, inside and outside the months the panel's `zlb` flag
        marks.  Empty if either side has fewer than twelve months.  Also
        written to ext_costs_zlb_<SPEC_TAG>.csv.

    Notes
    -----
    The full path is run first and then split by month, so the weights are
    the ones the strategy actually held.  Each month carries its own cost.
    """
    first = list(bts)[0]
    zlb = bts[first]["zlb"].to_numpy(bool)
    rf = bts[first]["rf"].to_numpy(dtype=float)

    print("\n" + "=" * 78)
    print(f"PART D   Zero lower bound split, {zlb.sum()} of {len(zlb)} months")
    print("=" * 78)
    if zlb.sum() < 12 or (~zlb).sum() < 12:
        print("   one side has fewer than twelve months; skipped")
        return pd.DataFrame()

    print(f"{'':34s}{'turn ZLB':>10s}{'turn other':>12s}"
          f"{'SR ZLB':>9s}{'SR other':>10s}{'SR all':>9s}")
    recs = []
    for name, bt in bts.items():
        for strat, label in STRATEGIES:
            if strat == "un" and name != first:
                continue                          # benchmark reported once
            rung = "static" if strat == "un" else name
            # full path at REF_BP, rebalanced every month, then split by month
            r = run_costs(weights(bt, strat, COST_GAMMA), x, rf,
                          REF_BP / 10000.0)
            xs, td = r["xs_net"], r["traded"]
            # the setup trade out of cash belongs to no subperiod, so the two
            # turnover means are taken over months two onwards
            t_z, t_o = td[1:][zlb[1:]].mean(), td[1:][~zlb[1:]].mean()
            print(f"{rung + ', ' + label:34s}{t_z:10.4f}"
                  f"{t_o:12.4f}{sharpe(xs[zlb]):9.3f}"
                  f"{sharpe(xs[~zlb]):10.3f}{sharpe(xs):9.3f}")
            recs.append({"rung": rung, "strategy": label,
                         "turnover_zlb": t_z, "turnover_other": t_o,
                         "SR_zlb": sharpe(xs[zlb]),
                         "SR_other": sharpe(xs[~zlb]),
                         "SR_all": sharpe(xs)})
    print(f"\n   Net of {REF_BP:.0f} basis points one way, rebalanced every "
          f"month.")
    print("   Short windows: read the levels, not the differences.")

    tab = pd.DataFrame(recs).set_index(["rung", "strategy"])
    tab.to_csv(os.path.join(OUT_DIR, f"ext_costs_zlb_{SPEC_TAG}.csv"))
    return tab


# ==========================================================================
# Figures
# ==========================================================================
def figures(bts, x):
    """
    Draw the two figures of Chapter 8: traded turnover over time (Figure
    8.1) and net Sharpe ratio against the cost level (Figure 8.2).

    Parameters
    ----------
    bts : dict of str -> pd.DataFrame
        Output of load_backtests().
    x : np.ndarray, shape (T, 2)
        Realised excess returns on (bond, equity), as decimals.

    Returns
    -------
    None
        Writes ext_turnover_path_<SPEC_TAG>.pdf and
        ext_net_sharpe_vs_cost_<SPEC_TAG>.pdf to OUT_DIR.
    """
    plt.rcParams.update(common.PLOT_RC)
    first = list(bts)[0]
    rf = bts[first]["rf"].to_numpy(dtype=float)
    idx = bts[first].index.to_timestamp("M")

    # ---- net Sharpe ratio against the cost level (Figure 8.2) ------------
    # Hard rule of every rung and the benchmark, rebalanced every month, on
    # a grid from 0 to 100 bp one way
    grid = np.linspace(0.0, 100.0, 41)                # basis points, one way
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    for name, bt in bts.items():
        W = weights(bt, "rs", COST_GAMMA)
        ax.plot(grid, [sharpe(run_costs(W, x, rf, c / 10000.0)["xs_net"])
                       for c in grid], color=RUNG_COLOURS.get(name, "0.5"),
                lw=1.0, label=name)
    W_un = weights(bts[first], "un", COST_GAMMA)
    ax.plot(grid, [sharpe(run_costs(W_un, x, rf, c / 10000.0)["xs_net"])
                   for c in grid], color=STATIC_COLOUR, lw=1.0, ls=(0, (4, 3)),
            label="static")
    ax.set_xlabel("One-way cost (basis points)", fontsize=8.5)
    ax.set_ylabel("Net Sharpe ratio", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    ax.margins(x=0.01)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT_DIR, f"ext_net_sharpe_vs_cost_{SPEC_TAG}.pdf"),
                bbox_inches="tight")
    plt.close(fig)

    # ---- traded turnover over time (Figure 8.1) --------------------------
    # Hard rule of the richest rung against the benchmark, with the ZLB
    # windows shaded.  Turnover does not depend on the cost, so cost 0.
    top = list(bts)[-1]
    r_top = run_costs(weights(bts[top], "rs", COST_GAMMA), x, rf, 0.0)
    r_un = run_costs(W_un, x, rf, 0.0)
    fig, ax = plt.subplots(figsize=(6.8, 2.6))
    common.shade_zlb(ax, idx, bts[first]["zlb"].to_numpy(bool))
    ax.plot(idx[1:], r_top["traded"][1:], color=RUNG_COLOURS.get(top, "0.3"),
            lw=0.7, label=top)
    ax.plot(idx[1:], r_un["traded"][1:], color=STATIC_COLOUR, lw=0.7,
            ls=(0, (4, 3)), label="static")
    ax.set_ylabel(f"Traded turnover ($\\gamma={COST_GAMMA}$)", fontsize=8.5)
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    ax.margins(x=0.01)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT_DIR, f"ext_turnover_path_{SPEC_TAG}.pdf"),
                bbox_inches="tight")
    plt.close(fig)

    print(f"\nfigures written to {OUT_DIR}")


# ==========================================================================
# MAIN
# ==========================================================================
if __name__ == "__main__":
    bts, x = load_backtests()
    print(f"rungs loaded: {', '.join(bts)}   "
          f"{next(iter(bts.values())).index.min()} to "
          f"{next(iter(bts.values())).index.max()}, "
          f"{len(next(iter(bts.values())))} months\n")

    part_a(bts, x)
    part_b(bts, x)
    part_c(bts, x)
    part_d(bts, x)
    figures(bts, x)

    print("\nAll output in", OUT_DIR)