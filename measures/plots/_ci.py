"""Run-level bootstrap intervals for the figures, on the appendix's machinery.

The chapter already has one bootstrap, in `tools/uncertainty.py`, and it fixes
three things that a second implementation would get subtly different: the run is
the resampling unit, the generator is seeded from the target's name through
SHA-256 so intervals do not move when a target is added, and the interval is a
percentile interval at 95 per cent over ten thousand resamples. This module
imports those pieces rather than restating them, so a figure and the appendix
cannot drift apart on a free parameter neither of them mentions.

What is left to decide here is only the statistic being resampled, and there are
two of them:

`band` --- a curve of per-round means, for a quantity a run carries once per
round. Runs are drawn with replacement and the mean is taken over the draw at
every round at once, so the band is a pointwise interval on the cell mean and
not a simultaneous band over the whole curve. A reader who takes the widest
excursion of a jagged curve as significant because it leaves the band is reading
more out of it than a pointwise interval can carry.

`pooled_median` --- the median of everything the drawn runs contain, for a
quantity that occurs a variable number of times per run. Resampling the runs and
not the events is what keeps the interval honest: a cell where one run supplies
a fifth of the fights should not be reported as though those fights were
independent draws.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _m in ("core", "_shared", "tools"):
    sys.path.insert(0, str(HERE.parent / _m))

from uncertainty import LEVEL, RESAMPLES, _pctl, _rng  # noqa: E402

__all__ = ["band", "band_difference", "pooled_median", "LEVEL", "RESAMPLES"]


def band(naam: str, per_run) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mean, lo, hi) per column, resampling the rows.

    `per_run` is one row per run and one column per round.
    """
    arr = np.asarray(per_run, dtype=float)
    n = len(arr)
    rng = _rng(naam)
    idx = rng.integers(0, n, size=(RESAMPLES, n))
    boot = arr[idx].mean(axis=1)                     # (resamples, rounds)
    a = (1 - LEVEL) / 2
    return (arr.mean(axis=0),
            np.quantile(boot, a, axis=0),
            np.quantile(boot, 1 - a, axis=0))


def pooled_median(naam: str, per_run) -> tuple[float, float, float]:
    """(median, lo, hi) over everything the runs hold, resampling the runs."""
    runs = [np.asarray(r, dtype=float) for r in per_run if len(r)]
    n = len(runs)
    if not n:
        return float("nan"), float("nan"), float("nan")
    rng = _rng(naam)
    idx = rng.integers(0, n, size=(RESAMPLES, n))
    boot = np.empty(RESAMPLES)
    for i in range(RESAMPLES):
        boot[i] = np.median(np.concatenate([runs[j] for j in idx[i]]))
    lo, hi = _pctl(boot)
    return float(np.median(np.concatenate(runs))), lo, hi


def band_difference(naam: str, a, b) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(difference, lo, hi) per column for the mean of `a` less the mean of `b`.

    The two arms are resampled separately, so each keeps the size it was run at
    --- the stratified rule of `tools/uncertainty.py`. An interval on the
    difference is not recoverable from the two intervals drawn beside each
    other: two bands can overlap while the difference between them still
    excludes zero, and reading significance off overlapping bands is the error
    this function exists to avoid.
    """
    A = np.asarray(a, dtype=float)
    B = np.asarray(b, dtype=float)
    na, nb = len(A), len(B)
    rng = _rng(naam)
    boot = (A[rng.integers(0, na, size=(RESAMPLES, na))].mean(axis=1)
            - B[rng.integers(0, nb, size=(RESAMPLES, nb))].mean(axis=1))
    q = (1 - LEVEL) / 2
    return (A.mean(axis=0) - B.mean(axis=0),
            np.quantile(boot, q, axis=0),
            np.quantile(boot, 1 - q, axis=0))
