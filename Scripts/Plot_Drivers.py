"""
Plot_Drivers.py
===============
Figure 2.2 and the dispersion table behind Chapter 2: the four candidate
transition drivers, 1990 to 2025, with the two ZLB episodes shaded.

Depends only on 00_common.py and the raw files in Data/.  The short rate is
the CRSP one-month bill, the series the model uses.  The other drivers are
read through common.load_fred_daily, the loader 01_build_data.py uses, and
the ZLB windows come from common.ZLB_WINDOWS.

Daily series are sampled on the last trading day of each month, the
decision date of the allocation.  The EBP is shown at the month it refers
to.  Its publication lag is applied in 01_build_data.py, not here.

Reads   Cash_RiskfreeTS.csv, Driver_DGS3MO.csv, Driver_DGS10.csv,
        Driver_VIXCLS.csv, Driver_ebp_csv.csv
Writes  drivers_monthly.csv, drivers_zlb_table.csv, drivers_variation.pdf
        to OUT_DIR
"""

import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
SAMPLE_START = "1990-01-31"
SAMPLE_END   = "2025-12-31"

# ZLB windows as month-end timestamps, to match the index of the figure
ZLB = [(pd.Period(a, "M").to_timestamp("M"), pd.Period(b, "M").to_timestamp("M"))
       for a, b in common.ZLB_WINDOWS]

# one panel per driver: (column, axis label)
PANELS = [("short", "Short rate (%)"),
          ("term",  "Term spread (pp)"),
          ("ebp",   "Excess bond premium (pp)"),
          ("vix",   "VIX (index points)")]


# ==========================================================================
# Loading
# ==========================================================================
def load_ebp():
    """
    Read the Federal Reserve Board excess bond premium.

    Returns
    -------
    pd.Series
        Monthly EBP in percentage points, month-end DatetimeIndex, named
        `ebp`.
    """
    e = pd.read_csv(os.path.join(common.DATA_DIR, "Driver_ebp_csv.csv"))
    e["date"] = pd.to_datetime(e["date"])
    s = e.set_index("date")["ebp"]
    s.index = s.index + pd.offsets.MonthEnd(0)       # first of month -> month end
    return s.rename("ebp")


def load_bill():
    """
    Read the CRSP one-month bill, the short rate of Figure 2.1 and of the
    model.

    Returns
    -------
    pd.Series
        Annualised yield in percent as CRSP quotes it, last value of each
        month, month-end DatetimeIndex, named `short`.
    """
    rf = pd.read_csv(os.path.join(common.DATA_DIR, "Cash_RiskfreeTS.csv"))
    s = rf[rf.TTERMLBL.str.contains("1-Month", na=False)][["MCALDT", "TMYTM"]]
    s = s.dropna()
    s.index = pd.to_datetime(s["MCALDT"])
    return s["TMYTM"].resample("ME").last().rename("short")


# ==========================================================================
# MAIN
# ==========================================================================
if __name__ == "__main__":
    # daily -> last trading day of each month
    t3  = common.load_fred_daily("Driver_DGS3MO.csv", "DGS3MO").resample("ME").last()
    t10 = common.load_fred_daily("Driver_DGS10.csv",  "DGS10").resample("ME").last()
    vix = common.load_fred_daily("Driver_VIXCLS.csv", "VIXCLS").resample("ME").last()

    df = pd.concat([load_bill(),
                    (t10 - t3).rename("term"),       # ten-year less three-month
                    load_ebp(),
                    vix.rename("vix")], axis=1, sort=True)
    df = df.loc[SAMPLE_START:SAMPLE_END]
    if df.isna().any().any():
        raise SystemExit(f"missing values:\n{df.isna().sum()}")
    df.to_csv(os.path.join(common.OUT_DIR, "drivers_monthly.csv"))

    # ---- dispersion inside the ZLB windows against the full sample -------
    zlb_idx = pd.DatetimeIndex([])
    for a, b in ZLB:
        zlb_idx = zlb_idx.union(df.loc[a:b].index)

    table = pd.DataFrame({
        "full_sd":    df.std(),
        "zlb_sd":     df.loc[zlb_idx].std(),
        "zlb_range":  df.loc[zlb_idx].max() - df.loc[zlb_idx].min(),
        "full_range": df.max() - df.min(),
    })
    table["ratio"] = table["zlb_sd"] / table["full_sd"]
    table = table[["full_sd", "zlb_sd", "ratio",
                   "zlb_range", "full_range"]].round(3)
    table.to_csv(os.path.join(common.OUT_DIR, "drivers_zlb_table.csv"))

    print(table)
    print(f"\nZLB months: {len(zlb_idx)} of {len(df)}")
    print(f"Sample: {df.index.min().date()} to {df.index.max().date()}")

    # ---- figure ----------------------------------------------------------
    plt.rcParams.update(common.PLOT_RC)
    plt.rcParams.update({"xtick.major.width": 0.6, "ytick.major.width": 0.6})

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(6.3, 7.2), sharex=True)
    for ax, (col, label) in zip(axes, PANELS):
        for a, b in ZLB:
            ax.axvspan(a, b, color="0.88", lw=0, zorder=0)
        ax.plot(df.index, df[col], color="black", lw=0.8, zorder=2)
        if col in ("term", "ebp"):                   # both cross zero
            ax.axhline(0, color="0.6", lw=0.5, zorder=1)
        ax.set_ylabel(label, fontsize=8.5)
        ax.margins(x=0.01)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    # one label for the shading, in the top panel over the first window
    axes[0].text(pd.Timestamp("2012-06-30"), 6.6, "Zero lower bound",
                 fontsize=7.5, color="0.35", ha="center")

    fig.align_ylabels(axes)
    plt.tight_layout(h_pad=0.6)
    plt.savefig(os.path.join(common.OUT_DIR, "drivers_variation.pdf"),
                bbox_inches="tight")
    print(f"\nfigure written to {common.OUT_DIR}")