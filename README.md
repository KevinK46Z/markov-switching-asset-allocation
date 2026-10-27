# Macro-Financially Driven Regime Switching in Dynamic Asset Allocation

Code for the master's thesis *Macro-Financially Driven Regime Switching in
Dynamic Asset Allocation*, Kevin Kaufmann, MiQE/F, University of St.Gallen.

The thesis does two things. It replicates the regime-switching market-timing
model of Ang and Bekaert (2004) on independently sourced data, and it extends
the transition equation from one driver to several, then prices the resulting
turnover.

## Requirements

Python 3 with the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

Those are the versions used for the reported results. Later versions of SciPy
can shift L-BFGS-B optima slightly.

## Layout

The code expects three folders side by side in one project folder:

```
<project folder>/
├── Scripts/    all .py files, including 00_common.py
├── Data/       raw inputs (not committed) and the two built panels
└── Output/     caches, result tables and figures (not committed)
```

The name and location of the project folder do not matter. `00_common.py`
finds its own folder, `Scripts/`, takes the folder above it as the project
folder, and reads from `Data/` and writes to `Output/` inside it. No path has
to be edited after cloning. Create `Data/` and `Output/` if they are missing,
and place the raw input files listed in `DATA.md` in `Data/`.

If `Data/` and `Output/` sit somewhere else, for example on another drive, set
the environment variable `MT_PROJECT` to the folder that contains them before
starting Python:

```
set MT_PROJECT=E:\thesis              (Windows)
export MT_PROJECT=/path/to/thesis     (Linux or macOS)
```

The code then uses `E:\thesis\Data` and `E:\thesis\Output`, wherever the
scripts are.

## Data

Not committed. The CRSP extracts are licensed. `DATA.md` lists every file each
script expects, under the exact filename the code reads, with its source and
the vintage used.

## The two modules

`00_common.py` and `02_msm.py` are modules, not scripts. `00_common.py` holds
the paths, the risk-aversion grid, the strategy tags, the ZLB windows, the
figure palette and the shared statistics. `02_msm.py` is the estimator: the
likelihood, the Hamilton filter, the multi-start optimiser and the conditional
moments. Running `02_msm.py` directly executes a self-test that simulates from
a known parameter vector and checks that the estimator recovers it.

Filenames beginning with a digit are not legal module names, so every script
loads its dependencies by path through `common.load_module`. In Spyder, run
each script in a fresh console or with autoreload switched off, since
autoreload cannot refresh modules loaded this way and keeps stale copies.

## Order of execution

Two chains, both starting at `01_build_data.py`. Each script prints the
numbers of the thesis tables it feeds, labelled by table or section.

**Replication chain**

| Script | What it does | Runtime |
|---|---|---|
| `01_build_data.py` | builds `panel_replication.csv` and `panel_modern.csv`, and prints the data diagnostics of Chapter 5: the moments of both panels, the AR(1) fit of the short rate, the realised maturity and callable share of each bond series, the comparison with the CRSP Fixed Term Index, the Ibbotson and S&P 500 series of Welch and Goyal against Ang and Bekaert's Table 2, and the covariates in levels | a few minutes |
| `03_replication.py` | full-sample M0 and M1 against Table 6 of Ang and Bekaert (Table 6.1), the likelihood-ratio test of constant transitions, the recursive backtest from 1985 and its performance at five levels of risk aversion (Table 6.3), on both bond series | ~3400 s |
| `05_theta_vs_data.py` | feeds Ang and Bekaert's published parameter vector through the same allocation on the same returns as the fitted one, and tests their in-sample gap (Table 6.6) | under a minute |
| `04_diagnostics.py` | switch points, implied bond volatility and limiting portfolios (Tables 6.1 and 6.2), B, C, the Sharpe ratio and the leverage factor of each strategy (Table 6.3), the Memmel tests (Table 6.5), mean weights and classification counts, the arithmetic of the published table (Appendix B.1), and the simulation check of the Memmel standard error (Appendix A.7) | under a minute, a few minutes with the simulation |
| `06_benchmark_check.py` | the same benchmark code on three bond series, including the Ibbotson series Welch and Goyal supply (Table 6.4), and the gap to Ang and Bekaert's reported benchmark in standard errors | seconds |

`04`, `05` and `06` re-estimate nothing. They read the `v5`-stamped files `03`
writes and run the filter at stored parameter vectors. `04` also reads the
in-sample results of `05` for its simulation, so run `05` first. Set
`RUN_SIM = False` at the top of `04` to skip the simulation of Appendix A.7.

**Extension chain**

| Script | What it does | Runtime |
|---|---|---|
| `01_build_data.py` | as above | a few minutes |
| `07_ladder.py` | the ladder M0 to M4: full-sample fit, likelihood-ratio tests and coefficients (Tables 7.1 to 7.3), recursive backtest from 2005 and its performance (Table 7.4), the ZLB split (Table 7.6) and each rung’s regime classification for January 2009 (Section 7.5) | ~8500 s |
| `07d_noise_rung.py` | the hundred-draw placebo test (Table 7.5), with the permutation p-value and the counts against M1, the benchmark and the ladder's likelihood-ratio statistics | see below |
| `08_costs.py` | target against traded turnover (Table 8.1), proportional costs and break-even costs (Table 8.2), the no-trade band (Table 8.3) and the ZLB split net of costs (Table 8.4) | ~60 s |
| `07c_figures.py` | redraws every figure of `07` and `08` from files already on disk | seconds |

`07c_figures.py` runs last because it needs both `07` and `08`. Run it again
whenever the figure code or the palette changes, instead of rerunning either.
It forces `RESUME` on, so it never refits.

The performance tables of `07` (Part C) need only the backtest files. To print
them again without rerunning the backtests, run `07c_figures.py` and then, in
the same console:

```
lad.part_c(lad.load_backtests())
```

**Standalone figures**

| Script | What it does | Runtime |
|---|---|---|
| `Plot_Shortrate.py` | Figure 2.1: the one-month bill rate 1952 to 2025, with the ZLB episodes shaded and the end of the Ang and Bekaert evaluation window marked. Prints the range over 1952 to 2000 and the mean over 2010 to 2015 quoted in the caption | seconds |
| `Plot_Drivers.py` | Figure 2.2: the four candidate transition drivers 1990 to 2025, with the ZLB episodes shaded. Prints the dispersion table behind Chapter 2 and writes it to `Output/` as `drivers_zlb_table.csv`, with the monthly series as `drivers_monthly.csv` | seconds |

Both depend only on `00_common.py` and the raw files in `Data/`. They use no
panel, cache or result of the two chains, so they can run at any point once
`Data/` is populated. The short rate in both is the CRSP one-month bill, the
same series the model uses. The ZLB windows come from `common.ZLB_WINDOWS`,
so the figures and the panel's `zlb` column cannot disagree.

## Caching and resumption

Every long script caches. `RESUME = True` restores what exists rather than
refitting it.

Full-sample fits are cached as `.npz` files holding the parameter vector,
and for `03` and `07` its standard errors. The filter is deterministic given that vector, so a
restored fit is identical to the original. Delete the `.npz` to force a
refit.

The recursive backtests checkpoint month by month into the system temporary
folder, not into the project. The checkpoints are rewritten every month and a
cloud sync client that locks a file mid-rewrite kills the run, which is why
they live outside any synced folder. An interrupted run resumes at the month
it stopped.

`SPEC_TAG` stamps every cache and output file so a restored estimate cannot
silently come from an earlier build. The replication chain uses `v5` and the
extension chain `e1`. Bump the tag rather than deleting caches when the
specification changes.

`07_ladder.py` accepts a subset of rungs on the command line:

```
python 07_ladder.py M3 M4
```

Part A still fits every rung, because each rung supplies the starting value of
the next, but only the named rungs are backtested. A backtested rung starts
its first month from the seed file of the rung below,
`ext_first_theta_<rung>_e1.npy`, and stops with a message if that file is
missing. A full run of `07` writes these files itself. A subset run or a
resumed run therefore gives the same result as a full run.

## The placebo test

`07d_noise_rung.py` re-estimates and re-backtests a hundred rungs, each
pairing the short rate with one artificial AR(1) column matched to the
persistence of the term spread. It requires `07_ladder.py` to have written its
Part A cache and the backtests for M1 and M2, and exits with a message if they
are missing. The placebo rungs start their first backtest month from the
default and random starting points only, without a seed file.

It takes about 2.2 hours per seed, so roughly 220 hours serially. Split it
across workers by seed. The reported run used six workers and took about 36
hours.

Linux or macOS:

```
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for i in $(seq 0 5); do
  NOISE_CHUNK=$i NOISE_NCHUNKS=6 python 07d_noise_rung.py > noise_$i.log 2>&1 &
done
wait
python 07d_noise_rung.py
```

Windows, in one Anaconda Prompt that stays open for the whole run:

```
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
for /L %i in (0,1,5) do start "noise%i" /B cmd /c "set NOISE_CHUNK=%i&& set NOISE_NCHUNKS=6&& python 07d_noise_rung.py > noise_%i.log 2>&1"
```

Then, once all six have finished, the serial pass:

```
python 07d_noise_rung.py
```

Four things to know. Closing the prompt kills the workers, because they share
its console. Pinning the BLAS threads matters, since without it each worker
spawns a full thread pool and the parallel run is slower than the serial one.
The thread variables have to be set before Python starts, so this cannot be
done from inside Spyder. A worker writes caches only, and the serial pass at
the end restores all of them and produces the table, the permutation p-value
and the CSV in a couple of minutes.

To check progress, count the rows in the partial files in the system temporary
folder. Python buffers redirected output, so a log can sit empty for a long
time while the worker is running normally.

## Outputs

No output is committed. The scripts write their tables as CSV and their
figures as PDF into `Output/`, stamped with the `SPEC_TAG` that produced them
where they depend on one. `07d_noise_rung.py` writes the hundred placebo
draws to `ext_noise_rungs_e1.csv`, one row per seed, which holds the numbers
of Appendix B.2 and the placebo distribution in Table 7.5.

## Reproducibility

The multi-start optimiser is seeded. `starting_values` draws its jittered
starts from `default_rng(0)` unless a generator is passed, so the start set is
the same on every run. The simulation of `04` is seeded as well.

The artificial covariates of the placebo test are generated from a fixed seed
list. The AR(1) persistence they are matched to is estimated from the panel
at the start of every run and printed. It is 0.9756 on the data vintage used
for the reported results, so a revised vintage changes the columns.

Several checks are assertions rather than comments. A failure in any of them
means a stale file, not a rounding difference.

- `01_build_data.py` refuses to write a panel with a missing month, and
  reports how closely the CRSP bond return reconstructs from price and
  coupon.
- `02_msm.prepare` refuses a panel that violates `r_t == r_obs_{t-1}`.
- `04_diagnostics.py` and `08_costs.py` rebuild every recorded return from
  the weights and the panel and stop if they disagree.
- `06_benchmark_check.py` stops unless its benchmark reproduces the one `03`
  recorded to machine precision, and unless the Ibbotson series matches the
  moments `01` prints.
- `07_ladder.py` stops if the benchmark differs between two rungs, which
  would mean they were run on different panels.

## Citation

Kaufmann, Kevin (2026). *Macro-Financially Driven Regime Switching in Dynamic
Asset Allocation*. Master's thesis, University of St.Gallen.

## Licence

No licence is granted. The code may be read and run to check the results of
the thesis. The data are not part of this repository and remain under the
terms of their providers, see `DATA.md`.
