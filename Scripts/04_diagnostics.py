"""
04_diagnostics.py
=================
The tables and figures of Chapter 6 that come from the stored replication
results, and the diagnostics behind them.  Run after 03_replication.py and
05_theta_vs_data.py.  Nothing is re-estimated.

Reads:
    panel_replication.csv
    repl_M1_theta_<tag>_<SPEC_TAG>.csv and repl_backtest_<tag>_<SPEC_TAG>.csv
        written by 03, one pair per bond series
    Ang and Bekaert's Model II vector, from 02 (AB_MODEL_II_THETA), and
        their published out-of-sample table, from 03 (AB_OOS_PUBLISHED,
        AB_GAMMA_FREE)
    repl_param_vs_data_<tag>_<SPEC_TAG>.csv written by 05, for the
        in-sample configurations of the simulation

Prints, in order:
    switch points, implied bond volatility and limiting portfolios
        (Tables 6.1 and 6.2), for both bond series and Ang and Bekaert
    for each bond series: B, C, the Sharpe ratio and the leverage factor
        (Table 6.3, Panel A), the Memmel tests (Table 6.5), the mean weights
        (Table 6.2) and the regime classification counts (Section 6.2)
    the arithmetic of the published out-of-sample table and the test of its
        gap at the correlations measured here (Appendix B.1, Table 6.5)
    the simulation check of the Memmel standard error (Appendix A.7),
        unless RUN_SIM is False

Writes the regime-path figure of each bond series (Figure 6.2 for the
ten-year series) and repl_oos_decomposition_<SPEC_TAG>.csv.

The leverage factor of a strategy S against the benchmark U is
lambda = C_S / C_U (Appendix A.3), printed as k.

Units.  C here comes from common.gamma_free(), computed on the excess
return.  part_c of 03 estimates the same constant from the grid of total
returns.  The two differ slightly, because the total return includes the
bill rate, which varies from month to month.
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


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
OUT_DIR    = common.OUT_DIR
STRATEGIES = common.STRATEGIES

SPEC_TAG  = "v5"               # must match 03_replication.py
BOND_LEGS = [("10y", "rb_x_10y"), ("long", "rb_x_long")]
GAMMA     = 5                  # every quantity below is gamma-invariant

msm  = common.load_module("02_msm.py", "msm")            # the estimator
repl = common.load_module("03_replication.py", "repl")   # Ang and Bekaert's published tables

RUN_SIM = True        # Appendix A.7 simulation, a few minutes
N_SIM   = 400_000     # simulated samples per configuration

# Ang and Bekaert's out-of-sample table in gamma-free form, computed in 03
# from their printed Table 8 (2002)
AB = {n: {"B": repl.AB_GAMMA_FREE.loc[("out-of-sample", n), "B"],
          "C": repl.AB_GAMMA_FREE.loc[("out-of-sample", n), "sd_gamma"]}
      for n in ("regime-dependent", "non-regime-dependent")}

# ==========================================================================
# Loading
# ==========================================================================
def load(tag, col):
    """
    Read 03's backtest for one bond series, with the realised returns and the
    weight path of each strategy, and check that they fit together.

    Parameters
    ----------
    tag : str
        Label of the bond series, "10y" or "long".
    col : str
        Panel column holding that series' excess return, "rb_x_10y" or
        "rb_x_long".

    Returns
    -------
    bt : pd.DataFrame
        The backtest written by 03, indexed by month.
    x : np.ndarray, shape (T, 2)
        Realised excess returns on (bond, equity) over the backtest months,
        as decimals.
    W : dict of str -> np.ndarray
        Weight path of each strategy at GAMMA, keyed by tag ("rs", "sf",
        "un").

    Raises
    ------
    SystemExit
        The backtest file is missing, or the weights applied to the panel's
        returns do not reproduce the returns 03 recorded.  That points to a
        wrong bond series, a stale backtest or a changed GAMMA.
    """
    bt_file = os.path.join(OUT_DIR, f"repl_backtest_{tag}_{SPEC_TAG}.csv")
    if not os.path.exists(bt_file):
        raise SystemExit(f"missing {bt_file}; run 03_replication.py at "
                         f"SPEC_TAG {SPEC_TAG}")

    bt = common.read_backtest(os.path.basename(bt_file))
    panel = common.read_panel("panel_replication.csv")
    x = panel.loc[bt.index, [col, "re_x"]].to_numpy(dtype=float) / 100.0

    W = {s: common.weight_path(bt, s, GAMMA) for s, _ in STRATEGIES}

    # Everything below is rebuilt from the weights and the panel, but 03
    # already recorded the returns.  A mismatch means swapped bond series,
    # a stale backtest or a changed GAMMA, and nothing else would catch it.
    rf = bt["rf"].to_numpy(dtype=float)
    for s, _ in STRATEGIES:
        recon = rf + (W[s] * x).sum(axis=1)
        if not np.allclose(recon, bt[f"R_{s}_g{GAMMA}"].to_numpy(float),
                           atol=1e-12):
            raise SystemExit(f"reconstruction mismatch for '{s}' on bond "
                             f"series '{tag}': check BOND_LEGS and GAMMA "
                             f"against 03")

    return bt, x, W


# ==========================================================================
# Analysis
# ==========================================================================
def analyse(tag, col):
    """
    The replication diagnostics for one bond series.

    Prints B, C, the Sharpe ratio and the leverage factor of each strategy
    (Table 6.3), the Memmel tests of Table 6.5, and the mean weights and
    regime classification counts of Section 6.2.  Draws the regime-path
    figure, Figure 6.2 for the ten-year series.

    Parameters
    ----------
    tag : str
        Label of the bond series, "10y" or "long".
    col : str
        Panel column holding that series' excess return, "rb_x_10y" or
        "rb_x_long".

    Returns
    -------
    dict
        The key results of this bond series: B, C and Sharpe ratios,
        leverage factors, the Memmel tests and the number of months.
        __main__ collects one per series into repl_oos_decomposition CSV.
        Also writes repl_oos_regime_path_<tag>.pdf to OUT_DIR.
    """
    print("=" * 78)
    print(f"BOND SERIES '{tag}'")
    print("=" * 78)
    bt, x, W = load(tag, col)

    # realised monthly excess return of each strategy: weights times the
    # month's excess returns on bond and equity
    xs = {s: (W[s] * x).sum(axis=1) for s, _ in STRATEGIES}

    # B and C of Appendix A.3 for each strategy, as (B, C)
    BC = {s: common.gamma_free(xs[s], GAMMA)[:2] for s, _ in STRATEGIES}

    # ---- gamma-free decomposition ----------------------------------------
    # B, C and the Sharpe ratio B/C of each strategy
    print("\n--- gamma-free decomposition " + "-" * 49)
    print(f"{'':28s}{'B':>10s}{'C=sd*gam':>11s}{'SR=B/C':>9s}")
    for s, name in STRATEGIES:
        B, C = BC[s]
        print(f"{name:28s}{B:10.3f}{C:11.2f}{B / C:9.3f}")

    # leverage factor of each regime-dependent strategy against the
    # benchmark, lambda = C_S / C_U (Table 6.3), and the same for Ang and
    # Bekaert's published pair
    C_un = BC["un"][1]
    ab_k = AB["regime-dependent"]["C"] / AB["non-regime-dependent"]["C"]
    print(f"\n{'':28s}{'k=C_S/C_U':>11s}")
    k = {}
    for s, name in STRATEGIES[:2]:            # hard and soft, not the benchmark
        k[s] = BC[s][1] / C_un
        print(f"{name:28s}{k[s]:11.3f}")
    print(f"{'AB out-of-sample':28s}{ab_k:11.3f}")

    # ---- are the gaps distinguishable from zero? -------------------------
    # Memmel test of each Sharpe difference (Table 6.5)
    print("\n--- Memmel (2003) tests " + "-" * 54)
    print(f"{'':40s}{'gap':>8s}{'se':>8s}{'t':>7s}{'p':>8s}{'rho':>8s}")
    tests = {}
    for a, b, lab in (("rs", "un", "hard minus benchmark"),
                      ("sf", "un", "soft minus benchmark"),
                      ("rs", "sf", "hard minus soft")):
        d, se, t, p, rho = common.sharpe_diff_test(xs[a], xs[b])
        tests[lab] = (d, se, t, p, rho)
        print(f"{lab:40s}{d:8.3f}{se:8.3f}{t:7.2f}{p:8.3f}{rho:8.4f}")
    print(f"   T = {len(xs['rs'])} months")

    # ---- realised mean weights -------------------------------------------
    # average position of each strategy over the window (Table 6.2); cash is
    # the residual 1 - bond - equity
    print(f"\n--- realised mean weights, gamma = {GAMMA} " + "-" * 42)
    print(f"{'':30s}{'bond':>9s}{'equity':>9s}{'cash':>9s}")
    for s, name in STRATEGIES:
        b, e = float(W[s][:, 0].mean()), float(W[s][:, 1].mean())
        print(f"   {name:27s}{b:9.3f}{e:9.3f}{1.0 - b - e:9.3f}")
    print(f"   {len(W['rs'])} months")

    # how often the hard rule classified the calm regime, and how often the
    # classification changed from one month to the next (Section 6.2)
    print(f"\n   months classified regime 1: {(bt['regime'] == 1).mean():.1%}")
    flips = int((bt["regime"].diff().fillna(0) != 0).sum())
    print(f"   classification changed in {flips} of {len(bt)} months")

    # ---- figure ----------------------------------------------------------
    # Figure 6.2: regime probabilities above, equity weights below
    plt.rcParams.update(common.PLOT_RC)
    c_h, c_s = common.RULE_COLOURS["hard"], common.RULE_COLOURS["soft"]
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 4.2), sharex=True)
    idx = bt.index.to_timestamp("M")          # months as dates for the axis

    # top: the filtered probability the rules start from, in grey, and the
    # mixing weight each rule turns it into
    axes[0].plot(idx, bt["xi_filt_d"], color="0.55", lw=0.8,
                 label=r"$\xi_{t|t}$")
    axes[0].plot(idx, bt["pi1_h"], color=c_h, lw=0.8, label=r"$\pi_1$ hard")
    axes[0].plot(idx, bt["pi1_s"], color=c_s, lw=0.8, label=r"$\pi_1$ soft")
    axes[0].axhline(0.5, color="0.85", lw=0.5)        # classification threshold
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("probability", fontsize=8.5)
    axes[0].legend(frameon=False, fontsize=7.5, ncol=3)

    # bottom: equity weights, benchmark dashed black as in every figure
    axes[1].plot(idx, W["rs"][:, 1], color=c_h, lw=0.8, label="hard")
    axes[1].plot(idx, W["sf"][:, 1], color=c_s, lw=0.8, label="soft")
    axes[1].plot(idx, W["un"][:, 1], color=common.STATIC_COLOUR, lw=0.8,
                 ls=(0, (4, 3)), label="static")
    axes[1].axhline(0.0, color="0.85", lw=0.5)
    axes[1].set_ylabel(f"Equity weight ($\\gamma={GAMMA}$)", fontsize=8.5)
    axes[1].legend(frameon=False, fontsize=7.5, ncol=3, loc="lower right")

    for ax in axes:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.margins(x=0.01)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT_DIR, f"repl_oos_regime_path_{tag}.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"\n   figure: repl_oos_regime_path_{tag}.pdf\n")

    # ---- record for the summary table ------------------------------------
    rec = {"tag": tag}
    rec["T"] = len(xs["rs"])                  # months of the evaluation window
    for s, _ in STRATEGIES:                   # B, C and SR of each strategy
        rec[f"B_{s}"], rec[f"C_{s}"] = BC[s]
        rec[f"SR_{s}"] = BC[s][0] / BC[s][1]
    for s in ("rs", "sf"):                    # leverage against the benchmark
        rec[f"k_{s}"] = k[s]
    for lab, key in (("hard minus benchmark", "hb"),
                     ("soft minus benchmark", "sb"),
                     ("hard minus soft", "hs")):     # the three Memmel tests
        d, se, t, p, rho = tests[lab]
        rec[f"gap_{key}"], rec[f"se_{key}"] = d, se
        rec[f"t_{key}"], rec[f"p_{key}"], rec[f"rho_{key}"] = t, p, rho
    return rec


def regime_tables():
    """
    Switch points, implied unconditional bond volatility (Table 6.1) and
    limiting portfolios at GAMMA (Table 6.2), for the M1 estimates on both
    bond series and for Ang and Bekaert's published Table 6 vector.

    Switch point
        The short rate at which a regime is as likely to end as to persist,
        -a_i / b_i (Appendix A.1).
    Implied unconditional bond volatility
        The two regime volatilities of the bond shock combined with the
        probability of regime 1 (Appendix A.6).  That
        probability is the mean filtered probability for the fitted vectors
        and Ang and Bekaert's reported 0.7014 for theirs.
    Limiting portfolios
        The weights of the rule if the investor were certain of one regime,
        pi1 = 1 or pi1 = 0: Sigma_x(i)^-1 mu / gamma, the two curves of their
        Figure 6 approach these.

    Returns
    -------
    None
        Prints only.  Reads repl_M1_theta_<tag>_<SPEC_TAG>.csv for each bond
        series and runs the filter once per fitted vector for the mean
        filtered probability.
    """
    full = common.read_panel("panel_replication.csv")

    # One entry per parameter vector: (label, bond column, theta).  The two
    # fitted M1 vectors 03 saved, one per bond series, then Ang and Bekaert's
    # published vector, which has no bond column of its own.
    rows = [(tag, col, pd.read_csv(os.path.join(
                 OUT_DIR, f"repl_M1_theta_{tag}_{SPEC_TAG}.csv"))["M1"]
             .to_numpy(dtype=float)) for tag, col in BOND_LEGS]
    rows.append(("AB", None, np.asarray(msm.AB_MODEL_II_THETA, dtype=float)))

    print("=" * 78)
    print("TABLES 6.1 AND 6.2   switch points, implied bond volatility, limits")
    print("=" * 78)
    print(f"{'':7s}{'switch calm':>12s}{'switch vol':>12s}{'pi_bar':>9s}"
          f"{'sd_b uncond':>13s}")
    for tag, col, th in rows:
        p = msm.unpack(th, 1)
        # probability of regime 1 used to combine the two regime volatilities
        if col is None:
            pibar = 0.7014                               # their reported value
        else:
            # mean filtered probability: run the filter at the fitted vector
            # on the panel with this bond series
            pibar = float(msm.evaluate(common.promote_leg(full, col),
                                       ["z_short"], th).xi_filt.mean())
        # standard deviation of the bond shock in each regime; [1, 1] is the
        # bond variance, since y is ordered (short rate, bond, equity)
        s1, s2 = (np.sqrt(p["Sigma"][i][1, 1]) for i in (0, 1))
        print(f"   {tag:4s}{-p['a'][0] / p['b'][0][0]:12.3f}"
              f"{-p['a'][1] / p['b'][1][0]:12.3f}{pibar:9.3f}"
              f"{np.sqrt(pibar * s1 ** 2 + (1 - pibar) * s2 ** 2):13.3f}")
    print("   Switch points -a_i/b_i in percent per month.  pi_bar is the mean")
    print("   filtered probability, or Ang and Bekaert's 0.7014 for their vector.")

    print(f"\n{'limiting portfolio, gamma = ' + str(GAMMA):34s}"
          f"{'bond':>8s}{'equity':>8s}{'cash':>8s}")
    for i, lab in ((0, "calm, pi1 = 1"), (1, "volatile, pi1 = 0")):
        for tag, _, th in rows:
            p = msm.unpack(th, 1)
            w = common.mv_weights(np.array([p["mu_b"], p["mu_e"]]),
                                  p["Sigma"][i][1:, 1:], GAMMA)
            print(f"   {tag + ', ' + lab:31s}{w[0]:8.3f}{w[1]:8.3f}"
                  f"{1 - w.sum():8.3f}")
    print()


def ab_published(recs):
    """
    Check the arithmetic of Ang and Bekaert's published out-of-sample table
    (Appendix B.1) and test their reported Sharpe gap (Table 6.5).

    For each of their two strategies, prints sd * gamma at every gamma, A, B
    and C of the gamma-free decomposition, and the bill rate their printed
    Sharpe ratios imply, set against the actual mean bill rate over the
    window.  Then tests their gap with the Memmel test.  They report no
    return series and no correlation, so the test borrows the correlation
    between the hard rule and the benchmark measured here on each bond
    series.

    Parameters
    ----------
    recs : list of dict
        Output of analyse(), one per bond series, each carrying `tag` and
        `rho_hb`, the correlation used for the test.

    Returns
    -------
    None
        Prints only.
    """
    print("=" * 78)
    print("APPENDIX B.1 AND TABLE 6.5   Ang and Bekaert's published table")
    print("=" * 78)
    ab = repl.AB_OOS_PUBLISHED
    g = np.array(common.GAMMAS, dtype=float)
    for name in ("regime-dependent (hard)", "non-regime-dependent"):
        t = ab.xs(name, level="strategy")
        A, B, _, C, _ = common.gamma_decomposition(t["mean_ann_%"],
                                                   t["sd_ann_%"], g)
        x = t["mean_ann_%"] - t["sharpe_printed"] * t["sd_ann_%"]
        print(f"   {name}")
        print("      sd * gamma          "
              + "".join(f"{v:8.2f}" for v in t["sd_ann_%"] * g))
        print(f"      A {A:.3f}   B {B:.3f}   C {C:.2f}")
        print("      rate implied by the "
              + "".join(f"{v:8.2f}" for v in x) + f"   mean {x.mean():.3f}")
        print("      printed Sharpe ratio")
    bt = common.read_backtest(f"repl_backtest_10y_{SPEC_TAG}.csv")
    print(f"\n   mean annualised bill rate over the window: "
          f"{1200 * bt['rf'].mean():.3f}")

    sr_rs = AB["regime-dependent"]["B"] / AB["regime-dependent"]["C"]
    sr_un = AB["non-regime-dependent"]["B"] / AB["non-regime-dependent"]["C"]
    print(f"\n   AB out-of-sample gap {sr_rs - sr_un:+.3f}, tested at the "
          f"correlations measured here, T = {len(bt)}")
    for rec in recs:
        d, se, t, p = common.sharpe_diff_from_moments(sr_rs, sr_un,
                                                      rec["rho_hb"], len(bt))
        print(f"      rho {rec['rho_hb']:.3f} ({rec['tag']}): s.e. {se:.3f}, "
              f"t {t:.2f}, p {p:.3f}")
    print()


def memmel_simulation(recs):
    """
    Check the Memmel standard error against simulation (Appendix A.7).

    For each configuration, draws N_SIM samples of T months of independent,
    bivariate normal returns with the given Sharpe ratios and correlation,
    computes the Sharpe difference in each sample, and compares the standard
    deviation of those differences with the formula's standard error.  The
    generator is seeded, so the result is the same on every run.

    Parameters
    ----------
    recs : list of dict
        Output of analyse(), one per bond series.

    Returns
    -------
    None
        Prints the formula and the simulated standard error and by how many
        percent the formula understates the latter.  Skips the in-sample
        configurations, with a message, if 05 has not been run.
    """
    print("=" * 78)
    print("APPENDIX A.7   Memmel s.e. against simulation")
    print("=" * 78)

    # (label, SR of the hard rule, SR of the benchmark, correlation, months)
    configs = [(f"OOS {r['tag']}", r["SR_rs"], r["SR_un"], r["rho_hb"], r["T"])
               for r in recs]
    for tag, _ in BOND_LEGS:
        f = os.path.join(OUT_DIR, f"repl_param_vs_data_{tag}_{SPEC_TAG}.csv")
        if not os.path.exists(f):
            print(f"   {os.path.basename(f)} missing, run 05 first; "
                  f"in-sample {tag} skipped")
            continue
        r = pd.read_csv(f, index_col=0).loc["fitted theta, full sample"]
        configs.append((f"in-sample {tag}", r["SR_rs"], r["SR_un"],
                        r["rho_hb"], int(r["months"])))
    r0 = recs[0]                                       # the ten-year series
    configs.append(("short window", r0["SR_rs"], r0["SR_un"], r0["rho_hb"], 60))

    print(f"   {'configuration':28s}{'formula':>10s}{'empirical':>11s}"
          f"{'understates':>13s}")
    rng = np.random.default_rng(0)
    for lab, sr1, sr2, rho, T in configs:
        m1, m2 = sr1 / np.sqrt(12), sr2 / np.sqrt(12)  # monthly Sharpe ratios
        d = np.empty(N_SIM)
        for i in range(0, N_SIM, 10_000):              # in blocks, to save memory
            k = min(10_000, N_SIM - i)
            e1, e2 = rng.standard_normal((k, T)), rng.standard_normal((k, T))
            x1 = m1 + e1
            x2 = m2 + rho * e1 + np.sqrt(1 - rho ** 2) * e2   # correlation rho
            d[i:i + k] = (x1.mean(1) / x1.std(1, ddof=1)
                          - x2.mean(1) / x2.std(1, ddof=1))
        emp = d.std(ddof=1)                            # simulated standard error
        form = np.sqrt((2 - 2 * rho + 0.5 * (m1 ** 2 + m2 ** 2)
                        - m1 * m2 * rho ** 2) / T)     # Memmel formula
        print(f"   {lab + ', T = ' + str(T):28s}{form:10.5f}{emp:11.5f}"
              f"{100 * (1 - form / emp):12.1f}%")
    print()


# ==========================================================================
# MAIN
# ==========================================================================
if __name__ == "__main__":
    regime_tables()
    recs = [analyse(tag, col) for tag, col in BOND_LEGS]
    ab_published(recs)
    if RUN_SIM:
        memmel_simulation(recs)

    # the results of both bond series as one table, for the record
    tab = pd.DataFrame(recs).set_index("tag")
    tab.to_csv(os.path.join(OUT_DIR, f"repl_oos_decomposition_{SPEC_TAG}.csv"))
    print(f"written: repl_oos_decomposition_{SPEC_TAG}.csv in {OUT_DIR}")