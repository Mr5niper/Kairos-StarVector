# Kairos StarVector

Plots planetary geometry against stock prices on a degree scale, draws Gann
rays from each planetary point in all four directions, and works out where
those rays cross — including crossings between points that have already
happened and points still in the future.

<img width="2525" height="1860" alt="image" src="https://github.com/user-attachments/assets/456f7cd9-c6e2-4e54-a91a-a39142614263" />


---

## What you need

**Windows** and **Python 3.13.12**, nothing else. Everything else installs
itself the first time you run it.

Get Python from
[python.org/downloads/release/python-31312](https://www.python.org/downloads/release/python-31312/)
and tick **"py launcher"** during installation. The scripts look for Python
through that launcher, so a different Python already on your PATH will not
interfere.

If you were given a prebuilt `.exe`, you do not need Python at all — skip to
[Running the executable](#running-the-executable).

---

## Setup

Unzip anywhere. Keep the folder structure as it is; the program looks for its
own files by relative path.

Then double-click:

```
START.bat
```

The first run creates a virtual environment and downloads the dependencies,
which takes a few minutes. Runs after that start in seconds. Your browser
opens automatically at `http://localhost:8501`.

A console window stays open while the program runs. **Closing it stops the
program.** That is the normal way to quit.

### Other things START.bat can do

| Command | What it does |
|---|---|
| `START.bat` | Start the program |
| `START.bat ml` | Also install the optional machine-learning extras (large download, only needed for the Forecast models tab) |
| `START.bat reset` | Delete the virtual environment so the next run rebuilds it from scratch |

---

## Building a standalone .exe

If you want a single file you can copy to a machine without Python:

```
BUILD_EXE.bat
```

Takes several minutes and produces `dist\Kairos-StarVector.exe`. Add `full` to
include the machine-learning extras, which makes it far larger and slower to
start:

| | `BUILD_EXE.bat` | `BUILD_EXE.bat full` |
|---|---|---|
| Size | ~350–450 MB | ~1.5–2.5 GB |
| Startup | a few seconds | about a minute, every launch |
| Forecast models tab | not included | included |

Everything except that one tab works in the standard build. Unless you plan to
run the neural benchmark, build the standard one.

### Running the executable

Double-click it. A console window opens showing the local address, then your
browser opens. Keep the console open; closing it stops the program.

The first few seconds of startup are the executable unpacking itself to a temp
folder. It does that on every launch — that is how one-file executables work,
not a fault.

---

## Using it

### Gann sky grid

The first tab, and the main one.

**The vertical axis is degrees, 0 at the bottom to 180 at the top.** Not price.
Each dot is your chosen planet's peak height in the sky for one interval, at
its actual measured angle. Your stock is scaled onto that same degree scale as
an overlay. The red vertical line is today.

Getting started:

1. Enter a ticker. Yahoo Finance symbols — `^GSPC` for the S&P 500, `AAPL`,
   `GC=F` for gold futures, `BTC-USD`. Several separated by commas will be
   overlaid together.
2. Set **Start** and **End** in the sidebar for how much history to show.
3. Pick a **Dot interval** — day, week, month, quarter or year. You get a dot
   for every one of those across the whole chart, past and future.
4. Pick a **planet of choice**. One is easiest to read; each extra planet adds
   its own dots and rays in its own colour.
5. Set **Future days beyond the window** for how far past today to project.

**Fitting the stock to the planet.** Leave *Auto-fit* on and the program
picks a starting ratio and offset for you. Turn it off to set them yourself:
**ratio** stretches or squashes the price curve, **offset** slides it up and
down. Adjust until the past movement lines up with the planetary geometry the
way you want.

**If the chart looks like a solid mess**, turn things down in this order:

1. **Dots that emit rays** — lower it. This is the big one. Every dot is still
   drawn; only some of them radiate.
2. **Ray length** — set a number of days instead of unlimited.
3. **Ray ratios** — remove some. Keep a wide spread rather than adjacent ones,
   since the spread is what makes it look like a fan.

**If every ray looks vertical**, the 1x1 rate is set manually and is too steep
for your window. Set **1x1 rate** back to *square the chart*.

**Crossings.** Below the chart is a table of every place two rays cross,
labelled:

| | |
|---|---|
| `past-future` | one ray from a point behind us, one from a point ahead |
| `future-future` | both ahead |
| `past-past` | both behind |

`past-future` crossings are the interesting ones — both halves are exact
geometry, and they can be worked out today. **Confluence zones** group
crossings that land within a few days and degrees of each other. Tick *Mark
confluence zones* to show them on the chart as well as in the table.

Everything is downloadable as CSV.

### The other tabs

| Tab | What it is for |
|---|---|
| **Chart** | Ordinary price chart with Gann fans from swing pivots, Square of Nine levels, time counts and planetary price lines |
| **Alignment wave** | Treats each planetary alignment as an influence that fades over the following weeks, and adds them up into one wave |
| **Statistics** | Measures whether the patterns you are looking at hold up against chance |
| **Calibrate & forecast** | Tunes settings on older data, checks them on data it was not tuned on, then projects buy/sell/flat dates forward |
| **Cycles** | Finds repeating cycles in the price and compares them against planetary cycle lengths |
| **Calendar** | Every alignment, retrograde turn and sign change, past and future |
| **Forecast models** | Optional neural benchmark; needs the machine-learning extras |
| **Data** | The raw numbers behind everything, exportable |

---

## A word on the numbers

The program shows correlations, hit rates and fit scores. Treat them
carefully: price curves and planetary curves are both smooth, and any two
smooth lines will appear to correlate strongly even when nothing connects
them.

Because of that, every statistic is reported twice — the ordinary figure, and
one measured against scrambled data that has the same shape but no real
relationship. **The second number is the one that means something.**

Two things help you calibrate:

- **Offline demo data** in the sidebar replaces the price with a random
  series. Anything the program reports there is what your settings produce
  from pure noise.
- **Reality check** in the Statistics tab reruns everything against a
  scrambled copy of your own data.

This is analysis and charting software. It is not financial advice and does
not tell you what to buy.

---

## When something goes wrong

**"Python 3.13 was not found via the py launcher."**
Install Python 3.13.12 from the link above and enable the py launcher option.

**"No data returned for ..."**
Check the symbol on Yahoo Finance. Indices need a caret (`^GSPC`), futures use
`=F` (`GC=F`), crypto uses a dash (`BTC-USD`). With no internet the program
falls back to previously downloaded data, and offline demo mode always works.

**The build fails partway through.**
Usually antivirus locking a file while PyInstaller writes it. Add the project
folder to your exclusions. Otherwise delete the `venv` folder and run again.

**Blank white page in the browser.**
Wait a few seconds and refresh. If it persists on a built `.exe`, rebuild with
`BUILD_EXE.bat`.

**The Statistics tab is slow.**
It is doing thousands of comparisons to produce honest numbers. Lower the
iteration counts while exploring and raise them when you want a result you
intend to rely on.

**The build warns about `pydeck.widget` or `pyarrow.tests.parquet`.**
Expected and harmless. Those are optional add-ons to libraries the program
uses; they are not needed and the build continues fine.

---

## Where things are saved

| Folder | Contents |
|---|---|
| `data\cache\` | Downloaded price data, so the program works offline afterwards |
| `artifacts\` | Output from the command-line scripts |
| `dist\` | The built executable |

When running the `.exe`, these sit next to the executable.

---

## Command-line tools

Not needed for normal use. Run them from the project folder after `START.bat`
has set up the environment once.

```
python scripts/upcoming_alignments.py --days 365
python scripts/build_features.py --ticker SPY --start 2010-01-01
python scripts/run_benchmark.py            (needs the ml extras)
```

---

MIT licensed. See `CHANGES.md` for what has changed between releases.
