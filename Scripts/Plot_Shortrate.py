"""
Plot_Shortrate.py
=================
Figure 2.1: the one-month Treasury bill rate, 1952 to 2025, with the two ZLB
episodes shaded and the end of the Ang and Bekaert evaluation window marked.
Also prints the range over 1952 to 2000 and the mean over 2010 to 2015
quoted in the caption.

Depends only on 00_common.py and Cash_RiskfreeTS.csv, the file
01_build_data.py uses for the cash asset.  Plots TMYTM, the annualised yield
in percent as CRSP quotes it.  The shaded windows are common.ZLB_WINDOWS, so
the figure and the panel's `zlb` column cannot disagree.

Writes 2_shortrate.pdf to OUT_DIR.
"""

import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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
START  = "1952-01-01"
AB_END = "2000-12-31"          # last month of Ang and Bekaert's evaluation

# ZLB windows from the first to the last day of their months
ZLB = [(pd.Period(a, "M").start_time, pd.Period(b, "M").end_time)
       for a, b in common.ZLB_WINDOWS]


# ==========================================================================
# MAIN
# ==========================================================================
if __name__ == "__main__":
    # the one-month bill from the CRSP risk-free file
    rf = pd.read_csv(os.path.join(common.DATA_DIR, "Cash_RiskfreeTS.csv"))
    s = rf[rf.TTERMLBL.str.contains("1-Month", na=False)][["MCALDT", "TMYTM"]]
    s = s.dropna()
    s["MCALDT"] = pd.to_datetime(s["MCALDT"])
    s = s.sort_values("MCALDT").reset_index(drop=True)
    s = s[s.MCALDT >= START]

    # ---- caption numbers -------------------------------------------------
    ab = s[s.MCALDT <= AB_END].TMYTM
    zl = s[(s.MCALDT >= "2010-01-01") & (s.MCALDT <= "2015-12-31")].TMYTM
    print(f"{START[:4]}-{AB_END[:4]}: min {ab.min():.2f}, max {ab.max():.2f}")
    print(f"2010-2015: mean {zl.mean():.2f}")

    # ---- figure ----------------------------------------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 10,
        "axes.linewidth": 0.6,
    })

    fig, ax = plt.subplots(figsize=(7.0, 3.4))

    for a, b in ZLB:                                  # zero lower bound
        ax.axvspan(a, b, color="0.85", zorder=0, lw=0)

    # end of the Ang and Bekaert out-of-sample window
    ax.axvline(pd.Timestamp(AB_END), color="0.45", lw=0.8,
               ls=(0, (4, 3)), zorder=1)
    ax.text(pd.Timestamp("1999-06-01"), 15.4, "end of AB (2004)\nevaluation",
            ha="right", va="top", fontsize=7.5, color="0.35", linespacing=1.3)

    ax.plot(s.MCALDT, s.TMYTM, color="#1a1a1a", lw=0.9, zorder=3)

    ax.set_ylabel("Annualised yield (%)", fontsize=9.5)
    ax.set_ylim(-0.5, 17)
    ax.set_xlim(pd.Timestamp(START), pd.Timestamp("2026-01-01"))
    ax.set_yticks(range(0, 18, 4))
    ax.axhline(0, color="0.7", lw=0.5, zorder=2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9, length=3, width=0.6)
    ax.grid(axis="y", color="0.92", lw=0.5, zorder=0)
    ax.set_axisbelow(True)

    ax.legend(handles=[Patch(facecolor="0.85", label="Zero lower bound")],
              loc="upper right", frameon=False, fontsize=8.5)

    fig.tight_layout(pad=0.4)
    out = os.path.join(common.OUT_DIR, "2_shortrate.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"written: {out}")