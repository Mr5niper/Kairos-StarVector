"""
Kairos StarVector
=================
Market price data aligned against planetary geometry and Gann structure,
with the statistical machinery needed to tell a real relationship from a
coincidence.

Modules
-------
skygrid   the primary chart: degree-space Gann grid, planetary peak
          altitudes, radiating rays, ray intersections
astro     planetary longitudes, aspects, stations, synodic periods (ephem,
          no data downloads)
waves     alignment events -> decaying influence wave; cycle detection;
          surrogate and permutation significance tests
gann      square of nine, fans, time counts, longitude-to-price conversion
market    price loading with caching and an offline fallback
charting  plotly figures
paths     resource locations for source and frozen runs
"""
__version__ = "6.0.0"
__all__ = ["astro", "waves", "gann", "market", "charting", "paths", "calibrate",
           "skygrid"]

from . import (astro, calibrate, charting, gann, market, paths,  # noqa: E402,F401
               skygrid, waves)
