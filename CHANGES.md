# Changelog

## 6.0.2

* **No more committed .spec file.** PyInstaller regenerates a spec on every
  run, so a checked-in one is a build artifact pretending to be source: edit
  the bat and the spec silently wins, edit the spec and the next build
  overwrites it. Every option now lives in `BUILD_EXE.bat` as command-line
  flags, the generated spec is deleted when the build finishes, and `*.spec`
  is git-ignored.
* **Removed two hidden imports that never existed.** `scipy.special._cdflib`
  and `scipy._lib.array_api_compat.numpy.fft` were raising
  `ERROR: Hidden import not found` on every build. Found by running the
  command-line build for real rather than only checking that the spec parsed.
* Icon validation moved into the bat, so a PNG renamed to `.ico` is caught
  before PyInstaller reaches its resource writer, where the failure is a
  struct unpacking error that never mentions the icon.
* **README rewritten for people running the program**, not for people
  reviewing its design. It now covers what it does, how to install it, how to
  start it and how to drive each tab, with the reasoning and verification
  moved here where it belongs. Version numbers removed from it as well: the
  release is a dated zip, and a hand-edited version line in a README only ever
  drifts out of step with the one in the code.
* Known trade-off: the command line has no equivalent of the spec's
  `filter_submodules`, so two harmless warnings return, for `pydeck.widget`
  (wants ipywidgets) and `pyarrow.tests.parquet` (wants pytest). The build
  now says so up front rather than leaving them to look like problems.

## 6.0.1

* **Fan spread widened.** The default ray ratios were 1x2, 1x1 and 2x1, which
  on a typical chart occupy screen angles of 14, 26 and 44 degrees — a 31
  degree band that reads as one bundle at slightly different tilts rather than
  a fan. Squaring the chart in 6.0.0 fixed the tilt but not the spread.
  Default is now 1x8 through 8x1, spanning 3 to 76 degrees across eight
  distinct angle bands.
* **Added 1x16 and 16x1** to the ratio set. On a wide chart even 8x1 only
  reaches 76 degrees, so the near-vertical rays a classical fan drawing shows
  were not reachable at all.
* **Removed the "include fast ratios" filter.** It silently discarded 1x4,
  1x8, 4x1 and 8x1 — precisely the ratios that give a fan its spread. With
  ratios chosen explicitly in the picker, a second filter throwing some of
  them away was a trap rather than a convenience.
* Ray origins default lowered from 8 to 5, since seven ratios across four
  quadrants is 28 rays per dot. Density is now controlled by origin count,
  which is the correct dial.
* **Build now warns about stale files** from a previous version left behind by
  extracting an archive over an existing checkout. `stock_forecast/dataset.py`
  still imports skyfield, which is no longer a dependency, and both `gui/` and
  `stock_forecast/` are bundled wholesale into the executable, so leftovers
  ship inside it.

## 6.0.0

The degree-space Gann grid becomes the primary chart. Everything below was
found in shipped code and verified, not hypothesised.

### The main chart

* **New degree-space grid** (`kairos/skygrid.py`, first tab). Vertical axis is
  degrees 0–180, not price. Planet peak altitudes plot at their measured angle,
  rays radiate in all four directions, stocks overlay as scaled curves.
* **Axis direction was inverted in 5.3.** The first version mapped sky angles
  into price space; it is the stock that scales into degree space. That single
  flip also removed the entire per-instrument slope calibration problem, since
  in degree space the 1x1 is one degree per day on every instrument ever traded.
* **Ray intersections solved arithmetically** in (day, degree) space and
  classified `past-past`, `past-future`, `future-future`. `past-future` is the
  decision candidate: a future planetary position is exact today, so a ray cast
  backward from it crosses already-happened geometry.
* **Confluence zones** group crossings landing within a few days and degrees.
* **Peak altitude** via `alt_max = 90 - |latitude - declination|`, verified
  against PyEphem's own observer transit to 0.04 degrees and reproducing
  textbook solstice values (65.62° June, 18.75° December at latitude 47.8).
* **Squaring the chart.** The 1x1 rate now derives from the window width by
  default. A fixed 1°/day crosses 180° in 180 days, which is 150% of a
  four-month window and 14% of a three-year one — the same rays reading as a
  gentle diagonal at one scale and vertical at the other. Squared gives
  `180 ÷ window days`; measured median screen slope 1.00 at any window.
* **Every interval dot across the whole timeline.** An earlier version trimmed
  to the 12 dots nearest now, which silently overrode the interval setting. Dots
  are cheap and are all drawn; the limit moved onto ray origins alone, with
  modes for nearest-now, future-only, highest-angle and spread-evenly.
* **Future extension separated from the history window.** Sidebar dates control
  history; a separate control sets projection length.
* **Uniform ray styling.** 1x1 was drawn at double width with steep ratios
  faded, to encode reliability. That imposed an opinion about the method on the
  drawing of it. All rays are now width 1 at opacity 0.40, coloured by planet.
* **Confluence markers off by default**, available in the table instead.
* Auto-fit of stock to planet by least squares, with r² reported and a warning
  that two free parameters on two smooth curves reach a decent fit regardless.

### Why it would not build

* **Skyfield required a runtime download.** `load('de421.bsp')` fetches a 17 MB
  kernel from NASA on first use. In a one-file exe the file lands in a temp
  directory deleted on exit, and PyInstaller cannot see it at build time. This
  alone made a working one-file build impossible. Replaced with **PyEphem**,
  which embeds VSOP87 and the Chapront lunar theory in its compiled extension:
  no data files, no network, ~1 MB, 1.9s for ten bodies across 6500 days. The
  version-detection code in `_planet_positions` existed only to paper over
  Skyfield API drift and is gone.
* **No frozen entry point existed** — no launcher, no spec, no build script.
  Added `kairos_app.py`, which starts the server in-process through Streamlit's
  bootstrap, and `Kairos-StarVector.spec`, which collects the frontend assets
  and package metadata the import scanner cannot see.
* **`gui/gann_app.py` used `pd.Timedelta` without importing pandas.** NameError
  the moment the Build button was pressed.
* **yfinance returns MultiIndex columns**, so `df['Close']` gave a DataFrame,
  not a Series, and everything downstream received shape (N, 1). This is the
  real cause of the `.reshape(-1)` calls and the "fixes ValueError: shape (N,1)"
  comment in `dataset.py`. Normalised once at the boundary in `kairos/market.py`.
* **`lgbm.fit(verbose=False)`** raises TypeError on LightGBM 4.x. Both call
  sites swallowed it in a bare `except`, so residual fusion silently disabled
  itself and the report showed a missing model rather than an error.
* **transformers 5.x removed `return_all_scores`.** Also swallowed, so FinBERT
  sentiment returned zeros for every date. Now `top_k=None`.
* **The short-date-range path crashed.** With `train_len: 756` hard-coded, any
  range under roughly four years produced zero walk-forward windows, and the
  fallback branch then read `r["window"]["start"]`, a key it never set.

### Results that were wrong

* **Lookahead leakage.** `astro_gann_prox` computed round-number proximity from
  the closing price of the bar being predicted, so the feature contained the
  answer and any accuracy it produced was leakage. Removed. A leakage check is
  now part of the test suite: every feature correlates below 0.04 with the
  same-bar return on random data.
* **The backtest traded on information it did not have**, setting a bar's
  position from that same bar's prediction and multiplying by that bar's return.
  Signals are now lagged one bar, costs charged on the bar the position changes.
* **Diebold–Mariano used the plain sample variance**, which understates the
  standard error whenever the loss differential is serially correlated — and
  forecast errors nearly always are, so every model comparison read as
  significant. Now Newey–West with a Bartlett kernel and a data-driven lag.
* **Synodic periods were fitted over the loaded date range** and used geocentric
  longitudes, where Sun and Mercury share a mean motion so their relative
  longitude never accumulates. Jupiter–Saturn came out at 5196 days against a
  true 7253; Sun–Mercury at 23524 against a true 116. Now measured from
  long-baseline mean motions, heliocentric except for lunar pairs. All seven
  verified against published values.
* **Gann 1x1 slope was `(price range) / (days in view)`**, so the same market
  drew different fan angles at different zoom levels. Then briefly median
  absolute *daily* move, which is noise rate rather than trend rate and gave
  41.5 points/day on an index — projecting a 1x1 to 60,000 over four years.
  Now the median pivot-to-pivot rate, 12.26 points/day on the same data.
* **Fans expired silently.** A fan anchored years back has every ratio above the
  highest price ever traded, which reads as an extremely weak market when it
  means the fan ran out. A computed horizon now masks the projection.
* **Fans anchored only on alignment dates**, landing mid-trend at arbitrary
  levels. Added swing-pivot detection; alignment anchoring is one option.
* **The GAN generator was deterministic**, taking no noise input, so it could
  not represent a distribution and collapsed to a point estimate, leaving the
  adversarial objective meaningless. Added a noise vector; prediction averages
  32 draws for the conditional mean.
* **ARIMA order (3,1,2)** over-differenced an input already in log returns. Now
  (2,0,1), with the fallback returning a recent mean rather than the last value.
* **Only one walk-forward window on long histories.** Validation and test blocks
  were sized as a fraction of the data, so ten years left room for exactly one
  window and no distribution of scores to average. Both capped near six months:
  7 windows on 2200 bars, 17 on 5000.
* **MAPE on log returns** was a headline metric, but log returns cross zero
  constantly so the denominator explodes. Dropped from the default set.
* **No baseline existed.** Added persistence: a model that cannot beat it has
  learned nothing worth keeping.
* **FinBERT was re-entered once per calendar date** inside a Python loop, several
  thousand model calls over ten years. Now batched across the corpus.
* **The feature cache ignored its own inputs.** `assemble_conditional` returned
  any cached CSV regardless of ticker, range or aspect settings, so changing
  symbol reused the previous symbol's features and a non-intersecting index
  crashed the run.

### Crashes found and fixed during this work

* **Negative-index slice in `lead_lag_correlation`.** At `lag > n`,
  `len(x) - lag` goes negative and `x[:-76]` returns 148 elements from the front
  of the array while `y[lag:]` correctly returns none. The guard checked only
  `len(xa)`, so numpy failed deep inside `np.cov` where the cause is
  unrecognisable. Bounds are now clipped before slicing, the scan is clamped to
  `n - 32`, and constant slices are skipped. Audited the two other sites doing
  similar arithmetic; both were already guarded.
* **A broken PyTorch install took the whole app down.** Availability was checked
  with `try: import torch / except Exception`, which looks safe and is not: a
  half-installed torch crashes the interpreter with SIGBUS, which no `except`
  can intercept. Hit exactly that here after an install ran out of disk.
  Detection now uses `importlib.util.find_spec`, which never executes the
  module, with the real import deferred to the button press. Verified with a
  stub that announces its own import: torch reports available while never
  entering `sys.modules`.
* **Slider bounds collapsing.** Streamlit raises when `min_value` is not
  strictly less than `max_value`, and computed bounds can collapse. "Train /
  test split" computed calendar-day margins that crossed on short ranges;
  "Anchors to include" ran 2 to `min(len(pivots), 40)` and died at two pivots;
  "Candidate row number" died at one search result. An AST scan found all nine
  widgets with computed bounds; the three that could collapse now go through a
  guard that reports the single value as text and clamps defaults into range.
  Split bounds are also derived from bar positions rather than calendar days,
  which is what the underlying constraint actually is.
* **Plotly folds shape coordinates into autorange.** An unclipped 8x1 fan ray
  reached roughly 474,000 on an index near 6000, the axis stretched to 200,000
  to contain it, and the price series collapsed into a line at the bottom of the
  plot. Every line is now clipped where it leaves the visible band, and the price
  axis is pinned with `autorange: False`.
* **Ray length made charts unreadable.** The automatic fan horizon came to 979
  days on a 1300-day chart, so 40 anchors × 3 near-parallel ratios produced 120
  lines each crossing a third of the plot. Added a ray-length control; at 120
  days a fan on all 195 alignments spans 8% of the width instead of 35%.
* **Sign inversion in the forward plan.** With a positive `signal_offset` the
  position is driven by the wave weeks later, but the table printed the current
  date's wave value beside it, so a LONG could appear next to a negative number.
  Now reports the value actually used plus a `driven_by_date` column.
* **The candlestick legend entry** carried no useful label and was the source of
  a stray "undefined" string drawn over its own colour swatch. Removed; the price
  scale is named on the y axis. `st.plotly_chart` also now receives
  `theme=None`, since Streamlit defaults to `theme="streamlit"` and overrides
  the `plotly_dark` template these figures are built against.
* **`pd.Timedelta(days=N)` warns under NumPy 2.5.** Rewritten to the explicit
  unit form throughout, found by running the suite with
  `-W error::DeprecationWarning`.
* **Batch files ship CRLF.** `cmd.exe` mis-parses labels and `goto` targets in an
  LF-terminated `.bat`, which silently breaks every error path.

### Honest statistics

Price and the planetary curves are both smooth and strongly autocorrelated, so a
naive correlation between them is large regardless of any relationship and its
textbook p-value is wrong by orders of magnitude. Every test now reports a
surrogate p-value from phase-randomised series preserving the data's own
spectrum, alongside the naive one so the gap is visible.

Measured on a pure random walk:

| Test | Result | Honest comparison |
|---|---|---|
| Wave vs log price | naive p = 0.005 | surrogate p = 0.31 |
| Cycle matcher | 7 of 8 matched within 5% | ~60 candidate periods exist |
| Parameter search | 0.574 train accuracy | 0.482 test, 0.498 median |
| Rule of all angles | 42.6% | 59.3% matched control |

Added a reality-check mode that reruns every test against a phase-randomised
copy of the user's own series, offline demo data that doubles as a control case,
and `miss_pct` / `cycles_in_sample` columns on the cycle matcher.

### Calibrate and forecast

Fit wave parameters on a training window, re-score on held-out data, compare
against the same search run on phase-randomised prices, then project dated
LONG/SHORT/FLAT positions forward. Random search over roughly ten dimensions,
with candidates rejected for trading too rarely or sitting flat — without that
filter the search reliably wins by taking three trades in a decade and getting
them right.

`signal_offset` may read the wave ahead of the current date. Deliberate and not
lookahead: planetary positions are computable years in advance. Future prices
are never read in any module.

### Gann analysis from the literature

* Square of Nine as price levels, and longitude mapped onto the same spiral.
* Proper Gann time counts as degrees of a solar year.
* Market strength read from which angle price trades above, as an ordinal count
  rather than a hand-assigned label, so it can be correlated.
* The rule of all angles, measured with a matched static-target control. Split by
  direction, because a rising fan line recedes as time passes while a falling one
  descends to meet price: up-breaks hit 28%, down-breaks 55%.
* Price clusters where angles from several anchors converge.
* Gann eighths and thirds, and angle crossings of retracement levels.
* Squaring the range: elapsed time equal to the price range.
* Harmonic rounding of the 1x1 to a divisor of 360.

### Build system

* `BUILD_EXE.bat` and `START.bat` follow the existing repo conventions:
  `py -3.13` launcher check pinned to 3.13.12, `.\venv`, `[STEP n/m]` output,
  `goto :error` paths. `BUILD_EXE.bat full` opts into the ML stack.
* Installs with `--only-binary :all:` so a drifted pin fails loudly instead of
  quietly compiling from source, with a fallback that says so.
* An import and ephemeris self-test runs before PyInstaller, so a broken
  environment fails in seconds rather than after a 20-minute build.
* `requirements.txt` pins every runtime dependency with `==`, each verified to
  publish a CPython 3.13 Windows wheel. `requirements-ml.txt` pins only direct
  dependencies, deliberately: transformers 5.15 caps tokenizers at `<=0.23.0`
  and the only published build in that family is 0.23.1, so pinning inside a
  moving cap breaks the install rather than freezing it.
* Spec filters submodules bridging to libraries not shipped
  (`plotly.matplotlylib`, `pydeck.widget`, `pyarrow.tests`), which logged
  harmless warnings on every build. UPX left off: it corrupts some extension
  modules and attracts Defender. Console left on so a failed launch shows why.
* `.gitignore` uses `dir/*` plus `!dir/.gitkeep` rather than a bare `dir/`, since
  git does not descend into an excluded directory and the negation would be
  unreachable. Does not contain `*.spec`, which would exclude the build recipe.

### Removed

`gui/streamlit_app.py` and `gui/gann_app.py`, superseded by `gui/app.py`.
`stock_forecast/dataset.py` and `gann_grid.py`, superseded by `kairos/`.
`run_gui.bat`, `run_gui.sh`, `start.bat`, `setup.cfg`.
`scripts/fetch_news_yf.py` (yfinance news no longer carries usable timestamps)
and `scripts/run_optuna.py` (the search in `kairos/calibrate.py` covers the same
ground with train/test separation, which `run_optuna.py` lacked).

### Verified

App serves HTTP 200 and the full script body executes with no exceptions and no
deprecation warnings under `-W error`. Ephemeris matches published positions;
all seven synodic periods match published values. Spec resolves 451 Streamlit
frontend assets and 11 metadata directories. Precompute-and-filter shortcut
proven identical to direct computation. No feature leaks the target. Split
sizing yields multiple windows from 90 to 5000 bars. Scaler round-trips to
1.2e-9. Ray geometry verified diagonal at every window length, with every short
ray accounted for by a boundary. Verified from a clean archive extract.

Not tested: the Windows PyInstaller run itself and the torch training loops, for
want of Windows and disk in the test environment. The spec was validated by
executing its collection logic directly; torch paths degrade to a clear message
rather than crashing.
