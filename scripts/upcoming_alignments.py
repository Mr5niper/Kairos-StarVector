#!/usr/bin/env python3
"""
Print the alignment calendar ahead, without launching the GUI.

    python scripts/upcoming_alignments.py
    python scripts/upcoming_alignments.py --days 730 --orb 1.0 --csv out.csv
    python scripts/upcoming_alignments.py --bodies JUPITER SATURN URANUS

Planetary positions are deterministic, so this is exact geometry computed
forward rather than a forecast of anything.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from kairos import astro as A


def main() -> int:
    ap = argparse.ArgumentParser(description="Upcoming planetary alignments.")
    ap.add_argument("--start", default=None, help="default: today")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--orb", type=float, default=1.5)
    ap.add_argument("--aspects", type=float, nargs="*",
                    default=[0, 90, 120, 180])
    ap.add_argument("--bodies", nargs="*", default=None)
    ap.add_argument("--frame", default="geocentric",
                    choices=["geocentric", "heliocentric"])
    ap.add_argument("--min-weight", type=float, default=0.0,
                    help="drop events weaker than this")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    start = pd.Timestamp(args.start) if args.start else pd.Timestamp.today().normalize()
    bodies = args.bodies or ["SUN", "VENUS", "MARS", "JUPITER",
                             "SATURN", "URANUS", "NEPTUNE"]

    spec = A.EventSpec(aspects=args.aspects, orb_deg=args.orb,
                       min_separation_days=3)
    ev = A.upcoming_events(start, days_ahead=args.days, bodies=bodies,
                           spec=spec, frame=args.frame)

    if ev.empty:
        print("No alignments found for those settings.")
        return 0

    if args.min_weight > 0:
        ev = ev[ev["weight"] >= args.min_weight]

    ev = ev.sort_values("date").reset_index(drop=True)
    show = ev.copy()
    show["date"] = pd.to_datetime(show["date"]).dt.strftime("%Y-%m-%d")
    show["separation"] = show["separation"].round(2)
    show["offset"] = show["offset"].round(3)
    show["weight"] = show["weight"].round(4)

    print(f"{len(show)} alignments from {start.date()} "
          f"over {args.days} days ({args.frame}, orb {args.orb} deg)\n")
    print(show[["date", "pair", "aspect_name", "separation",
                "offset", "weight"]].to_string(index=False))

    print("\nStrongest ten by weight:")
    print(show.nlargest(10, "weight")[["date", "pair", "aspect_name", "weight"]]
          .to_string(index=False))

    if args.csv:
        show.to_csv(args.csv, index=False)
        print(f"\nWrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
