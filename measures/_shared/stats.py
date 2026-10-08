"""Effect sizes with intervals, and the two subset arms — written once.

Two things kept breaking in the ad-hoc scripts, and both live here now.

**An effect size without an interval.** A p-value alone says whether an effect is
distinguishable from zero, not how large it is, and a partial eta squared alone
says how large it looks in this sample and nothing about how precisely it was
estimated. Every effect size this module returns comes with an interval:
Fisher-z for a correlation, the noncentral F for partial eta squared.

**"The first five runs" is not the old sample.** Measures test whether a rebuilt
pipeline reproduces a published figure by re-running on a subset. The obvious
subset — the first five runs of each cell — is wrong for that purpose. In
`prod_L3_knife` the five runs the ledger figures were computed on sit at r06-r10;
r01-r05 are a batch added afterwards. A "first five" check there compares the new
code against the *new* runs and reproduces nothing.

So there are two arms, and they answer different questions:

`legacy`  Selection on the index column `bron`: every run except those added
          after the ledger figures were computed. This is the old sample and
          therefore the reproduction test.
`max5`    First five runs per cell on index order. Equal cell sizes, so it
          answers whether a finding survives a balanced design. It is a
          robustness check and must not be reported as reproduction.

⚠️ Even the `legacy` arm reproduces on one axis only. The scripts behind the
ledger read from an older copy that missed the cleanup and counted four
fallback-polluted runs as valid. Selecting the old runs
out of the current index restores the *sample*, not the *source tree*. Where a
figure moves, both axes have to be named as candidates.
"""
from __future__ import annotations

from math import atanh, sqrt, tanh

from scipy import stats as _st
from scipy.optimize import brentq

import runset

# Runs that did not exist when the ledger figures were computed. `openrouter` is
# the top-up that took L3 and L4 from 5 to 10 runs per cell and L2 scarce from 14
# to 15; `job 25401637` is the second L3 knife batch. Everything else appears in
# the manifest of the script that fed the old numbers.
NIEUWE_BRONNEN = ("openrouter", "job 25401637")

ARMS = ("full", "legacy", "max5")


def arm_paths(naam: str, arm: str = "full") -> tuple[list, list[dict]]:
    """(paths, skipped) for one cell under one subset arm.

    `skipped` is always returned and always carries a reason, so a measure can
    report what it left out instead of quietly reporting a smaller n.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; pick from {ARMS}")
    rijen = sorted((r for r in runset.rijen() if r["cel"] == naam),
                   key=lambda r: r["nieuw_id"])
    paths = runset.cel(naam)                      # same order, existence checked
    if len(rijen) != len(paths):
        raise runset.RunsetError(
            f"cel {naam}: {len(rijen)} index rows against {len(paths)} paths")
    pairs = list(zip(rijen, paths))
    if arm == "full":
        keep, drop = pairs, []
    elif arm == "legacy":
        keep = [(r, p) for r, p in pairs if r["bron"] not in NIEUWE_BRONNEN]
        drop = [(r, p) for r, p in pairs if r["bron"] in NIEUWE_BRONNEN]
    else:
        keep, drop = pairs[:5], pairs[5:]
    reden = ("added after the ledger figures" if arm == "legacy"
             else "outside the first five of the cell")
    return ([p for _, p in keep],
            [{"run": r["nieuw_id"], "bron": r["bron"], "reason": reden}
             for r, _ in drop])


def pearson(xs, ys, conf: float = 0.95) -> dict | None:
    """Pearson r with n, p and a Fisher-z interval — or None with a reason.

    Returns None when a variable has no spread. That is not a correlation of
    zero: the relation is undefined, and reporting 0.0 there is the mistake this
    module exists to prevent.
    """
    xs = [float(x) for x in xs]
    ys = [float(y) for y in ys]
    n = len(xs)
    if n != len(ys):
        raise ValueError("pearson: unequal lengths")
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    r = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sqrt(sxx * syy)
    r = max(-0.999999999, min(0.999999999, r))
    t = r * sqrt((n - 2) / (1 - r * r))
    p = float(2 * _st.t.sf(abs(t), n - 2))
    if n > 3:
        se = 1 / sqrt(n - 3)
        z = _st.norm.ppf(0.5 + conf / 2)
        lo, hi = tanh(atanh(r) - z * se), tanh(atanh(r) + z * se)
    else:
        lo = hi = None
    return {"r": r, "p": p, "n": n, "ci": None if lo is None else (lo, hi)}


def _ncp(F: float, df1: int, df2: int, target: float) -> float:
    """Noncentrality lambda for which P(F_ncf(lambda) > F_obs) equals `target`."""
    def g(lam):
        return float(_st.ncf.sf(F, df1, df2, lam)) - target
    if g(0.0) >= 0:
        return 0.0
    hi = 1.0
    while g(hi) < 0 and hi < 1e9:
        hi *= 2
    if g(hi) < 0:
        return hi
    return float(brentq(g, 0.0, hi))


def partial_eta_ci(F: float, df1: int, df2: int, conf: float = 0.90):
    """Confidence interval on partial eta squared, from the noncentral F.

    90% by default, and deliberately: the interval that corresponds to the
    one-sided F-test is the 90% one, so a 95% interval on an effect size tested
    at alpha .05 would not agree with its own test (lakens2013 after
    steiger2004). n = df1 + df2 + 1.
    """
    if not (F > 0) or df2 <= 0:
        return None
    alpha = 1 - conf
    lam_lo = _ncp(F, df1, df2, alpha / 2)
    lam_hi = _ncp(F, df1, df2, 1 - alpha / 2)
    n = df1 + df2 + 1
    return (lam_lo / (lam_lo + n), lam_hi / (lam_hi + n))


def zscores(xs) -> list[float] | None:
    """Standardised values, or None when the series has no spread.

    None rather than a list of zeros. A cell in which every run has the same
    value carries no information about a within-cell relation; feeding it in as
    zeros adds points at the origin and drags a pooled correlation toward it.
    """
    xs = [float(x) for x in xs]
    n = len(xs)
    if n < 2:
        return None
    m = sum(xs) / n
    sd = (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5
    if sd == 0:
        return None
    return [(x - m) / sd for x in xs]
