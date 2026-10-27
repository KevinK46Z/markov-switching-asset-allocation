"""
07c_figures.py
==============
Redraws every figure of 07_ladder.py and 08_costs.py from output already on
disk.  Run after both, and again whenever the figure code or the palette
changes.  Nothing is re-estimated.

Part A of 07 is restored from its per-rung cache, which also rewrites
ext_ladder_fit_<SPEC_TAG>.csv with identical content.  The backtests are read
from their files, so this takes seconds where a full run of 07 takes more
than two hours.  The figure functions are the ones 07 and 08 use themselves,
loaded by path, so the redrawn figures cannot differ from theirs.
"""

import os

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "common", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "00_common.py"))
common = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(common)

lad  = common.load_module("07_ladder.py", "ladder")   # Part A, backtests, figures
cost = common.load_module("08_costs.py", "costs")     # the cost figures


# ==========================================================================
# MAIN
# ==========================================================================
if __name__ == "__main__":
    panel = lad.load_modern()

    # Part A from cache.  evaluate() refilters at the stored theta, which is
    # deterministic, so the restored fits are identical to the originals.
    # RESUME is forced on, so a missing or disabled cache cannot start a
    # refit.
    lad.RESUME = True
    missing = [n for n, _ in lad.LADDER
               if not os.path.exists(os.path.join(
                   lad.OUT_DIR, f"ext_partA_{n}_{lad.SPEC_TAG}.npz"))]
    if missing:
        raise SystemExit(f"Part A cache missing for {missing}; run "
                         f"07_ladder.py first")
    fits = lad.part_a(panel)

    bts = lad.load_backtests()                    # Part B from the CSVs
    print(f"\nrungs loaded: {', '.join(bts)}")
    lad.figures(panel, fits, bts)

    # 08's two figures come from the same backtests, so they are redrawn
    # here as well.  08 recomputes only the cost curves the figures need.
    bts8, x8 = cost.load_backtests()
    cost.figures(bts8, x8)