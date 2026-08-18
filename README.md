# Kairos StarVector

**6.0.0** — MIT licensed.

A degree-space Gann grid. Planetary peak altitudes are plotted as points on a
0–180 degree axis, Gann rays radiate from each point in all four directions,
your stocks are overlaid as scaled curves, and every ray crossing is solved
and tabulated. Because planetary positions are computable in both directions,
crossings between a ray from a past point and a ray from a **future** point
can be calculated today.

Also included: an alignment-wave model, cycle detection, parameter calibration
with held-out testing, and an optional neural benchmark.

---

## Quick start

```
START.bat
```

Creates a virtual environment, installs the pinned dependencies on first run
only, and opens the app in your browser.

```
BUILD_EXE.bat
```

Produces `dist\Kairos-StarVector.exe`, a single file needing no Python on the
target machine. Add `full` to bundle the machine-learning extras.

Both require **exactly Python 3.13.12**, located through the `py` launcher so
a different `python` on your PATH cannot interfere.

---

## The Gann sky grid

This is the first tab and the main event.

**The vertical axis is degrees, 0 at the bottom to 180 at the top.** Not
price. Each dot is the chosen planet's peak altitude — its height at
culmination, seen from your latitude — for one interval, sitting at its
measured angle with no conversion applied.

**Dots appear at every interval across the whole displayed timeline**, past and
future. Choose day, week, month, quarter or year. Nothing is sampled or
skipped, because planetary positions are deterministic and cheap to compute.

**Gann rays radiate from each dot in all four directions**: forward, backward,
up and down. Every ray is drawn at the same width and opacity — a Gann line is
a Gann line, and where price sits relative to it is the information.

**Your stocks are overlaid**, scaled into the degree band by a ratio and offset
you adjust until past movement lines up with the planetary geometry. Auto-fit
gives a least-squares starting position.

**The red line is now.** Dots to the right of it are future positions, exact
rather than forecast.

### Squaring the chart

The 1x1 rate is derived from the window width by default, which is what Gann
meant by squaring a chart. This matters more than it sounds:

| Window | 1x1 at a fixed 1°/day covers | Looks like |
|---|---|---|
| 4 months | 150% of chart width | gentle diagonal |
| 2 years | 20% | steep |
| 3 years | 14% | vertical |

Same arithmetic, unusable at one scale and correct at the other. Squaring gives
`180° ÷ window days`, so the 1x1 stays on the diagonal at any window length.
Measured on a three-year window: median screen slope 1.00, exactly diagonal. A
manual rate is available, with the squared value always shown beside it.

Dots are cheap; rays are not. Every interval dot is always drawn, and a
separate control limits how many emit rays, with a mode for choosing which —
nearest now, future only, highest angle, or spread evenly.

### Crossings

Every ray pair is solved arithmetically in (day, degree) space and classified:

| kind | meaning |
|---|---|
| `past-future` | one ray from a point behind us, one from a future point |
| `future-future` | both ahead |
| `past-past` | both behind |

`past-future` is the decision candidate and the reason backward rays exist. A
future planetary position is exact today, so a ray cast back from it crosses
already-happened geometry, and that crossing is computable now rather than in
hindsight.

Crossings clustering within a few days and degrees are grouped into confluence
zones. One crossing of two lines is weak; a cluster of eight is what the method
looks for. Zone markers are off by default on the chart and available in the
table, which reads better.

---

## Other tabs

**Chart** — price with Gann fans anchored to swing pivots or alignment dates,
Square of Nine levels, round-number grids, time counts, planetary price lines,
plus an angle-analysis panel: market strength relative to the fan, the rule of
all angles measured against a control, price clusters, retracements, squaring
the range.

**Alignment wave** — the trickle-down model. Each planetary alignment emits an
influence that decays forward through time; overlapping tails sum into a wave,
projected past the last bar.

**Statistics** — correlations and event studies with p-values that account for
autocorrelation. The tab that decides whether anything else means anything.

**Calibrate & forecast** — fit wave parameters on a training window, re-score on
held-out data, compare against the same search run on phase-randomised prices,
then project dated LONG / SHORT / FLAT positions forward.

**Cycles** — periodogram of price on a calendar-day grid, dominant cycle
detection, matched against measured synodic periods.

**Calendar** — every aspect, station and ingress, past and future, with
zodiacal positions and daily motion.

**Forecast models** — optional. ARIMA, a conditional LSTM and a conditional
WGAN-GP on walk-forward windows, against a persistence baseline.

**Data** — the underlying series, exportable, plus environment details.

---

## Read this before trusting a number

The program will show you correlations, hit rates and fits. Those numbers on
their own are close to meaningless, for a reason worth understanding.

Price and the planetary curves are both smooth and strongly autocorrelated.
Correlate any two smooth series and you get a large number almost regardless of
whether they are related — two independent random walks routinely correlate
above 0.8. The textbook p-value assumes independent observations, so on data
like this it is wrong by orders of magnitude.

Every test therefore reports two p-values side by side:

- **p (naive)** — the textbook calculation. Almost always tiny. Ignore it.
- **p (surrogate)** — computed against series preserving your own data's power
  spectrum and autocorrelation but carrying no relationship. This is the number
  that means something.

Measured examples from this codebase's own test suite, all run on a **pure
random walk** with no connection to planetary geometry:

| Test | Result | Honest comparison |
|---|---|---|
| Wave vs log price | naive p = 0.005 | surrogate p = 0.31 |
| Cycle matcher | 7 of 8 cycles matched a synodic period within 5% | ~60 candidate periods exist, so a match is near-guaranteed |
| Parameter search | 0.574 direction accuracy on train | 0.482 on test, median 0.498 across candidates |
| Rule of all angles | 42.6% hit rate | 59.3% for a matched control |

Every one of those looks like a discovery and is not. That is what the
machinery is for.

Two tools are built in for calibrating expectations:

- **Offline demo data** — a synthetic random walk. Anything the analysis reports
  here is your noise floor for the settings you are using.
- **Reality check** — reruns every test against a phase-randomised copy of
  *your* series, matching its volatility and autocorrelation exactly while
  destroying any real relationship.

None of this argues your hypothesis is wrong. It is the machinery for finding
out, and it is deliberately hard to pass. A result that survives it deserves
attention. One that does not was never there.

### The one legitimate edge

`signal_offset` in the calibration, and the backward rays in the sky grid, both
read planetary values from the **future**. That is not lookahead cheating, and
it is the only real structural advantage here: planetary positions are
computable decades ahead, so future planetary values are genuinely known today
in a way no price-derived indicator's ever are.

Future *prices* are never read anywhere, in any module. Only planetary geometry
is read ahead. Every position is also lagged one bar before it meets a return,
so nothing is scored against a price it could not have been acted on.

This program is analysis and visualisation software. It is not financial
advice, and nothing in it is a recommendation to buy or sell anything.

---

## Which build do I want

|  | Standard | `full` |
|---|---|---|
| Command | `BUILD_EXE.bat` | `BUILD_EXE.bat full` |
| Size | ~350–450 MB | ~1.5–2.5 GB |
| Startup | a few seconds | ~1 minute, every launch |
| Everything except the neural benchmark | yes | yes |
| Forecast models tab | no | yes |

One-file executables re-extract their whole archive on every launch, so bundle
size becomes startup time. The benchmark is not lost in the standard build —
its source ships intact and runs from a source install:

```
START.bat ml
python scripts/run_benchmark.py
```

---

## Command-line tools

```
python scripts/upcoming_alignments.py --days 365 --orb 1.5
python scripts/upcoming_alignments.py --bodies JUPITER SATURN --csv out.csv
python scripts/build_features.py --ticker SPY --start 2010-01-01
python scripts/run_benchmark.py --epochs 40 --windows 4      # needs ML extras
python scripts/fetch_news_rss.py                             # needs ML extras
```

---

## Project layout

```
kairos/                  the engine, no heavy dependencies
  skygrid.py             degree-space grid, peak altitudes, radiating rays,
                         ray intersections, confluence zones
  astro.py               longitudes, aspects, stations, ingresses,
                         synodic periods, harmonic indices
  waves.py               trickle-down composite, cycle detection,
                         surrogate and permutation significance tests
  calibrate.py           parameter search with train/test separation,
                         null model, forward signal projection
  gann.py                Square of Nine, fans, pivots, time counts,
                         angle state, rule of all angles, clusters
  market.py              price loading, caching, offline fallback
  charting.py            Plotly figure builders
  paths.py               resource paths for source and frozen runs

gui/app.py               the Streamlit application
kairos_app.py            launcher, used by START.bat and the exe

stock_forecast/          optional ML benchmark
  pipeline.py            feature assembly and walk-forward evaluation
  models/                ARIMA, conditional LSTM, conditional WGAN-GP
  train_lstm.py          training with early stopping on direction accuracy
  train_gan.py           WGAN-GP training, noise-averaged prediction
  metrics.py             errors, direction accuracy, Diebold-Mariano
  backtest.py            long/short with costs and lagged signals
  meta_labeling.py       signal accept/reject filter
  splits.py              walk-forward window generation

scripts/                 command-line entry points
configs/default.yaml     defaults for the scripts
requirements.txt         pinned runtime dependencies
requirements-ml.txt      pinned optional ML extras
Kairos-StarVector.spec   PyInstaller build recipe
BUILD_EXE.bat            build the executable
START.bat                run from source
CHANGES.md               full changelog
```

Runtime output goes to `data/cache/` (price data, so the app works offline
after the first fetch) and `artifacts/` (script results). Both sit next to the
executable when frozen.

---

## Notes on the astronomy

Positions come from **PyEphem**, which embeds VSOP87 for the planets and the
Chapront lunar theory directly in its compiled extension. No data files, no
network access, accurate to a few arc-seconds — orders of magnitude finer than
the orbs any aspect study uses.

This replaced Skyfield, which requires the DE421 kernel downloaded from NASA on
first run. Workable from source, impossible in a one-file executable where the
download lands in a temp directory wiped on exit and PyInstaller cannot see the
file at build time.

**Peak altitude** uses the standard relation for a body's highest point:

```
alt_max = 90 - |latitude - declination|
```

Verified against PyEphem's own observer transit calculation to within 0.04
degrees, and it reproduces the textbook solstice altitudes: at latitude 47.8
the Sun peaks at 65.62° in June and 18.75° in December.

Both frames are available. **Geocentric** is what Gann and astrologers used.
**Heliocentric** is what the Bradley siderograph uses, with Earth replacing the
Sun.

Synodic periods are measured from long-baseline mean motions rather than from
your loaded date range, and verify against published values:

| Pair | Computed | Published |
|---|---|---|
| Moon–Sun | 29.53 d | 29.53 d |
| Mercury–Earth | 115.88 d | 115.88 d |
| Venus–Jupiter | 236.99 d | 236.99 d |
| Sun–Jupiter | 398.90 d | 398.88 d |
| Mars–Jupiter | 816.42 d | 816.4 d |
| Jupiter–Saturn | 7253.98 d | 7253.5 d |
| Saturn–Uranus | 16561 d | 16568 d |

Pairs involving the Moon are computed geocentrically, the rest
heliocentrically. That split is necessary rather than stylistic — the Sun and
Mercury share a mean *geocentric* motion, so their relative longitude never
accumulates and a geocentric calculation diverges instead of returning 116
days.

---

## Troubleshooting

**"Python 3.13 was not found via the py launcher."** Install Python 3.13.12 and
enable the py launcher during setup. The build refuses other versions because
the pinned wheels are the CPython 3.13 Windows builds.

**Every ray looks vertical.** The 1x1 rate is set manually and is too steep for
the window. Switch the 1x1 rate to "square the chart".

**The chart is an opaque lattice.** In order: reduce "dots that emit rays", set
a ray length instead of unlimited, remove ray ratios, turn off fast ratios.

**Build fails partway through.** Usually antivirus locking a file in `build\` or
`dist\`. Add the project folder to your exclusions, or delete `venv\` and rerun.

**Exe takes a long time to start.** Expected for one-file builds — the archive
extracts on every launch. Run from source with `START.bat` if startup time
matters.

**"No data returned for ..."** Check the symbol on Yahoo Finance directly.
Indices need a caret: `^GSPC`. Futures use `=F`: `GC=F`. Crypto uses a dash:
`BTC-USD`. If the network is unavailable the app falls back to cached data, and
offline demo mode always works.

**Blank white page in the browser.** The server started but the frontend assets
are missing, meaning the build did not collect Streamlit's static files.
Rebuild with `BUILD_EXE.bat`, which uses the spec file that handles it.

**Statistics tab is slow.** Surrogate and permutation tests are the cost of
honest p-values. Lower the iteration counts while exploring, raise them for a
result you intend to rely on.
