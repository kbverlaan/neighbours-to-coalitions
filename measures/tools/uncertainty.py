"""Run-level uncertainty intervals for the chapter's central claims.

Every figure in the package reports a point estimate over a cell of ten or
fifteen runs. None of them reports what that point estimate is worth. Two
numbers differing by 0.006 are called "indistinguishable" in the chapter and
two differing by 0.14 are called a difference; nothing in the package says
which of those readings the data supports.

Three rules hold here and they are the whole design:

  1. **The run is the resampling unit.** Never an agent-turn, never a message.
     A cell of fifteen runs carries fifteen independent draws, not the hundred
     thousand agent-turns inside them; bootstrapping the turns would produce
     intervals ten times too narrow for a quantity that varies between runs.
     Resampling is stratified: a contrast between two cells resamples inside
     each cell separately, so the arm sizes stay fixed at what was run.

  2. **A degenerate count gets an exact interval, never a bootstrap.** Thirty
     of thirty L4 runs end at the commons capacity. A bootstrap over those
     thirty resamples thirty ones and returns [1.00, 1.00] every time, which
     says the next run is certain to do the same. Clopper-Pearson says
     [0.884, 1.000] and Jeffreys says [0.941, 1.000]. The bootstrap answer is
     not conservative or approximate --- it is wrong, and it is wrong in the
     direction that flatters the claim.

  3. **Five runs is marked as five runs.** A percentile interval from n=5 has
     at most 2^5 distinct resample means below its own median; its endpoints
     are the order statistics of a five-point sample and move by whole runs.
     Every such interval carries `fragile: true` and its five raw values, so a
     reader can see the interval is a restatement of the five numbers rather
     than an inference from them.

Free parameters, both reported with their alternative as the package requires:
the interval method (percentile is primary; BCa is computed alongside wherever
n >= 10 and the two are compared in the output) and the correlation definition
(Pearson primary, Spearman alongside).

    python3 tools/uncertainty.py

Writes `out/uncertainty.json` and prints a summary table. Nothing else in the
package reads it; this is a standalone audit, not a registered figure.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HIER = Path(__file__).resolve().parent
WORTEL = HIER.parent
for _m in ("core", "_shared"):
    sys.path.insert(0, str(WORTEL / _m))

import lexicons as LEX      # noqa: E402
import runset               # noqa: E402
import combat               # noqa: E402
import runstat              # noqa: E402
import text                 # noqa: E402
import turns                # noqa: E402

SEED = 20260817
RESAMPLES = 10_000
LEVEL = 0.95
FRAGILE_AT = 5          # arms this small get their raw values printed
BCA_FROM = 10           # below this, a jackknife acceleration is noise
UIT = WORTEL / "out" / "uncertainty.json"

RUNGS = ("L1", "L2", "L3", "L4")
PAYOFFS = ("scar", "knife", "abund")
PRODUCTIE = [f"prod_{r}_{p}" for r in RUNGS for p in PAYOFFS]


# --- deterministic streams --------------------------------------------------
#
# One generator per target, seeded from the target's name through SHA-256
# rather than through `hash()`. Python salts `hash()` per process, so a script
# seeded that way returns different intervals on every run while looking
# reproducible; `tests/test_standards.py::test_no_unstable_seeds` exists
# because that had already happened once in this package. SHA-256 of the name
# also makes each target independent of the order the targets are computed in,
# so adding a target does not move the intervals of the ones before it.

def _rng(naam: str) -> np.random.Generator:
    h = hashlib.sha256(naam.encode()).digest()[:8]
    return np.random.default_rng([SEED, int.from_bytes(h, "big")])


# --- the intervals ----------------------------------------------------------

def _pctl(boot: np.ndarray) -> tuple[float, float]:
    a = (1 - LEVEL) / 2
    return float(np.quantile(boot, a)), float(np.quantile(boot, 1 - a))


def _bca(boot: np.ndarray, punt: float, jack: np.ndarray) -> tuple | None:
    """Bias-corrected and accelerated endpoints, or None where undefined.

    Returns None rather than a number in the two cases where BCa has no
    meaning: a constant bootstrap distribution (every resample identical, so
    the bias correction z0 is +-inf) and a constant jackknife (acceleration
    0/0). Both occur here --- the L1 scarce cell is fifteen zeroes --- and both
    would otherwise silently produce nan endpoints that read like numbers.
    """
    if np.std(boot) == 0 or len(jack) < 3 or np.std(jack) == 0:
        return None
    onder = float(np.mean(boot < punt) + 0.5 * np.mean(boot == punt))
    if not 0 < onder < 1:
        return None
    z0 = stats.norm.ppf(onder)
    d = jack.mean() - jack
    noemer = 6 * (np.sum(d ** 2) ** 1.5)
    if noemer == 0:
        return None
    a = float(np.sum(d ** 3) / noemer)
    uit = []
    for q in ((1 - LEVEL) / 2, 1 - (1 - LEVEL) / 2):
        z = stats.norm.ppf(q)
        nz = z0 + (z0 + z) / (1 - a * (z0 + z))
        uit.append(float(np.quantile(boot, stats.norm.cdf(nz))))
    if not np.all(np.isfinite(uit)):
        return None
    return uit[0], uit[1], z0, a


def _interval(naam: str, xs, stat=np.mean, extra: dict | None = None) -> dict:
    """Percentile CI on one arm, with BCa alongside where n is large enough."""
    xs = np.asarray(xs, dtype=float)
    n = len(xs)
    rng = _rng(naam)
    idx = rng.integers(0, n, size=(RESAMPLES, n))
    boot = stat(xs[idx], axis=1)
    lo, hi = _pctl(boot)
    punt = float(stat(xs))
    uit = {
        "estimate": round(punt, 6),
        "ci_lo": round(lo, 6),
        "ci_hi": round(hi, 6),
        "method": "percentile",
        "resamples": RESAMPLES,
        "n": n,
        "denominator": n,
        "unit": "runs",
        "fragile": n <= FRAGILE_AT,
        "sd": round(float(np.std(xs, ddof=1)), 6) if n > 1 else 0.0,
        "median": round(float(np.median(xs)), 6),
        "min": round(float(xs.min()), 6),
        "max": round(float(xs.max()), 6),
    }
    if n >= BCA_FROM:
        jack = np.array([stat(np.delete(xs, i)) for i in range(n)])
        b = _bca(boot, punt, jack)
        if b is None:
            uit["bca"] = None
            uit["bca_note"] = "undefined: constant bootstrap or jackknife"
        else:
            blo, bhi, z0, a = b
            uit["bca"] = {"ci_lo": round(blo, 6), "ci_hi": round(bhi, 6),
                          "z0": round(z0, 4), "acceleration": round(a, 4)}
            # "Materially" is fixed here rather than judged per case: a shift of
            # a tenth of the percentile interval's own width is the point at
            # which the two methods would be read as disagreeing in a table.
            breedte = hi - lo
            verschuiving = max(abs(blo - lo), abs(bhi - hi))
            uit["bca_agrees"] = bool(breedte == 0 or verschuiving <= 0.1 * breedte)
    if n <= FRAGILE_AT:
        uit["per_run"] = [round(float(x), 6) for x in xs]
    if extra:
        uit.update(extra)
    return uit


def _verschil(naam: str, a, b, labels=("a", "b"), stat=np.mean) -> dict:
    """CI on stat(a) - stat(b), resampling inside each arm separately.

    Stratified because the arms are fixed by design, not sampled: fifteen runs
    were commissioned at L2 and five in the no-channel control. Pooling the
    twenty and redrawing twenty would let a resample contain nineteen speaking
    runs and one silent one, which is not a dataset anyone could have collected.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    rng = _rng(naam)
    boot = (stat(a[rng.integers(0, na, size=(RESAMPLES, na))], axis=1)
            - stat(b[rng.integers(0, nb, size=(RESAMPLES, nb))], axis=1))
    lo, hi = _pctl(boot)
    punt = float(stat(a) - stat(b))
    fragiel = min(na, nb) <= FRAGILE_AT
    uit = {
        "estimate": round(punt, 6),
        "ci_lo": round(lo, 6),
        "ci_hi": round(hi, 6),
        "method": "percentile (stratified, two-sample)",
        "resamples": RESAMPLES,
        "n": {labels[0]: na, labels[1]: nb},
        "denominator": na + nb,
        "unit": "runs",
        "straddles_zero": bool(lo <= 0 <= hi),
        "ci_width": round(hi - lo, 6),
        "arm_means": {labels[0]: round(float(stat(a)), 6),
                      labels[1]: round(float(stat(b)), 6)},
        "fragile": fragiel,
    }
    if min(na, nb) >= BCA_FROM:
        jack = np.array([stat(np.delete(a, i)) - stat(b) for i in range(na)]
                        + [stat(a) - stat(np.delete(b, j)) for j in range(nb)])
        r = _bca(boot, punt, jack)
        if r is None:
            uit["bca"] = None
            uit["bca_note"] = "undefined: constant bootstrap or jackknife"
        else:
            blo, bhi, z0, acc = r
            uit["bca"] = {"ci_lo": round(blo, 6), "ci_hi": round(bhi, 6),
                          "z0": round(z0, 4), "acceleration": round(acc, 4)}
            breedte = hi - lo
            uit["bca_agrees"] = bool(
                breedte == 0
                or max(abs(blo - lo), abs(bhi - hi)) <= 0.1 * breedte)
    if fragiel:
        uit["per_run"] = {labels[0]: [round(float(x), 6) for x in a],
                          labels[1]: [round(float(x), 6) for x in b]}
    return uit


def _gepaard(naam: str, d) -> dict:
    """CI on the mean of matched-seed differences, resampling the pairs.

    The pair is the unit. Resampling the two arms independently would throw
    away the matching that the shared seeds bought and inflate the interval by
    the between-seed variance the design was built to remove.
    """
    d = np.asarray(d, dtype=float)
    n = len(d)
    rng = _rng(naam)
    boot = np.mean(d[rng.integers(0, n, size=(RESAMPLES, n))], axis=1)
    lo, hi = _pctl(boot)
    return {
        "estimate": round(float(d.mean()), 6),
        "ci_lo": round(lo, 6),
        "ci_hi": round(hi, 6),
        "method": "percentile (paired on shared seed)",
        "resamples": RESAMPLES,
        "n": n,
        "denominator": n,
        "unit": "seed pairs",
        "straddles_zero": bool(lo <= 0 <= hi),
        "fragile": n <= FRAGILE_AT,
        "per_run": [round(float(x), 6) for x in d],
    }


def _correlatie(naam: str, paren_per_cel: dict) -> dict:
    """Pooled r with a CI, resampling runs inside each cell and refitting.

    r is recomputed on every resample rather than transformed through Fisher's
    z, because the pooled sample mixes cells with different marginals and the
    z-interval assumes a bivariate normal it has no reason to satisfy.
    Spearman is carried alongside as the free parameter's alternative: where
    the two disagree the Pearson figure is being carried by a few extreme runs.
    """
    cellen = sorted(paren_per_cel)
    xs = np.concatenate([np.asarray([p[0] for p in paren_per_cel[c]], float)
                         for c in cellen])
    ys = np.concatenate([np.asarray([p[1] for p in paren_per_cel[c]], float)
                         for c in cellen])
    grenzen, k = [], 0
    for c in cellen:
        m = len(paren_per_cel[c])
        grenzen.append((k, k + m))
        k += m
    rng = _rng(naam)
    boot_p, boot_s = [], []
    for _ in range(RESAMPLES):
        idx = np.concatenate([rng.integers(lo, hi, size=hi - lo)
                              for lo, hi in grenzen])
        bx, by = xs[idx], ys[idx]
        if np.std(bx) == 0 or np.std(by) == 0:
            continue
        boot_p.append(float(np.corrcoef(bx, by)[0, 1]))
        boot_s.append(float(stats.spearmanr(bx, by).statistic))
    boot_p = np.asarray(boot_p)
    lo, hi = _pctl(boot_p)
    r = float(np.corrcoef(xs, ys)[0, 1])
    rs = float(stats.spearmanr(xs, ys).statistic)
    slo, shi = _pctl(np.asarray(boot_s))
    return {
        "estimate": round(r, 6),
        "ci_lo": round(lo, 6),
        "ci_hi": round(hi, 6),
        "method": "percentile (stratified by cell, r refitted per resample)",
        "resamples": RESAMPLES,
        "resamples_used": int(len(boot_p)),
        "n": int(len(xs)),
        "denominator": int(len(xs)),
        "unit": "runs",
        "cells": {c: len(paren_per_cel[c]) for c in cellen},
        "straddles_zero": bool(lo <= 0 <= hi),
        "spearman": {"estimate": round(rs, 6),
                     "ci_lo": round(slo, 6), "ci_hi": round(shi, 6)},
        "fragile": bool(len(xs) <= FRAGILE_AT),
    }


def _exact(k: int, n: int, wat: str, bron: str) -> dict:
    """Clopper-Pearson and Jeffreys for a success count. No bootstrap.

    Both are given because they answer the reader's question differently at the
    boundary: Clopper-Pearson is guaranteed-coverage and conservative, Jeffreys
    is the Beta(1/2,1/2) posterior interval and is the one to quote when the
    question is "what does 30 out of 30 license about the next run".
    """
    a = (1 - LEVEL) / 2
    cp_lo = 0.0 if k == 0 else float(stats.beta.ppf(a, k, n - k + 1))
    cp_hi = 1.0 if k == n else float(stats.beta.ppf(1 - a, k + 1, n - k))
    j_lo = 0.0 if k == 0 else float(stats.beta.ppf(a, k + 0.5, n - k + 0.5))
    j_hi = 1.0 if k == n else float(stats.beta.ppf(1 - a, k + 0.5, n - k + 0.5))
    return {
        "what": wat,
        "successes": k,
        "n": n,
        "denominator": n,
        "unit": "runs",
        "estimate": round(k / n, 6),
        "clopper_pearson": {"ci_lo": round(cp_lo, 6), "ci_hi": round(cp_hi, 6)},
        "jeffreys": {"ci_lo": round(j_lo, 6), "ci_hi": round(j_hi, 6)},
        "method": "clopper_pearson + jeffreys (exact; a bootstrap on a "
                  "degenerate count returns a zero-width interval)",
        "degenerate": bool(k == 0 or k == n),
        "fragile": bool(n <= FRAGILE_AT),
        "source": bron,
    }


# --- the per-run quantities -------------------------------------------------
#
# The package already memoises `final_gini` and `transfer_pct` for the 150
# production runs in out/scalars.json, computed from the full reasoning traces
# by figures/paths.py. Those are the numbers the chapter reports, so they are
# what gets resampled. The control and model arms are not in that memo and are
# read here from `_log.jsonl`, which carries the same action and resource
# fields; `provenance` below checks the two readings agree on every run that
# has both, so the mixture cannot hide a discrepancy.

_MEMO = None


def _memo() -> dict:
    global _MEMO
    if _MEMO is None:
        pad = WORTEL / "out" / "scalars.json"
        _MEMO = json.loads(pad.read_text()) if pad.exists() else {}
    return _MEMO


def _final_gini(p: Path) -> float:
    m = _memo().get(f"final_gini|{p.name}")
    return float(m) if m is not None else runstat.final_gini(runset.log_path(p))


def _transfer_pct(p: Path) -> float:
    m = _memo().get(f"transfer_pct|{p.name}")
    if m is not None:
        return float(m)
    return float(turns.share([runset.log_path(p)],
                             turns.is_action("transfer")).value)


def _mean_holding(p: Path) -> float:
    return float(runstat.mean_holding(runset.log_path(p)))


def _fights(p: Path) -> float:
    """Resolved fights in a run, counted from the combat record."""
    return float(sum(1 for _ in combat.fights(runset.log_path(p))))


def _welfare(p: Path) -> dict:
    """Welfare sentences and rescues for one run, as counts and as rates.

    The counts reproduce `figures/capacities.py::welfare_rules`. The rates are
    the same numerators over their own denominators --- sentences spoken, and
    agent-rounds spent below one resource --- and a run that never had an agent
    below one has no rescue rate at all rather than a rate of zero.
    """
    lp = runset.log_path(p)
    treffers = tot = 0
    for _, _, _, s in text.public([lp]):
        tot += 1
        if LEX.WELFARE.search(s):
            treffers += 1
    kansen, gered = runstat.destitute_and_rescues(lp)
    return {
        "welfare_sentences": treffers,
        "sentences": tot,
        "welfare_pct": 100 * treffers / tot if tot else None,
        "rescues": gered,
        "destitute_agent_rounds": kansen,
        "rescue_pct": 100 * gered / kansen if kansen else None,
    }


def _provenance() -> dict:
    """Does the memo agree with a fresh read of the compact log, run by run?

    The reason to ask: the memoised scalars were computed from the reasoning
    traces and everything added here reads `_log.jsonl`. If those two ever
    disagreed, a contrast between a memoised arm and a freshly-read one would
    be measuring the file format. They do not, and this says so with a number
    instead of with a claim.
    """
    ergst_g = ergst_t = 0.0
    n = 0
    for c in PRODUCTIE:
        for p in runset.cel(c):
            g = _memo().get(f"final_gini|{p.name}")
            t = _memo().get(f"transfer_pct|{p.name}")
            if g is None or t is None:
                continue
            lp = runset.log_path(p)
            ergst_g = max(ergst_g, abs(float(g) - runstat.final_gini(lp)))
            ergst_t = max(ergst_t, abs(
                float(t) - turns.share([lp], turns.is_action("transfer")).value))
            n += 1
    return {
        "memoised_runs_checked": n,
        "max_abs_diff_final_gini": round(ergst_g, 9),
        "max_abs_diff_transfer_pct": round(ergst_t, 9),
        "note": "memoised scalars (from the reasoning traces) against a fresh "
                "read of _log.jsonl; transfer_pct is memoised rounded to two "
                "decimals, so its tolerance is 0.005",
        "agrees": bool(ergst_g < 1e-9 and ergst_t <= 0.005),
    }


# --- the targets ------------------------------------------------------------

def gini_per_cel(gini: dict) -> dict:
    uit = {}
    for c in PRODUCTIE:
        xs = gini[c]
        uit[c] = _interval(f"gini|{c}", xs)
        uit[c]["median_ci"] = {
            k: v for k, v in _interval(f"ginimed|{c}", xs, np.median).items()
            if k in ("estimate", "ci_lo", "ci_hi")}
    return uit


def gini_contrasten(gini: dict) -> dict:
    uit = {
        "L2_scar_minus_knife": _verschil(
            "c|L2sk", gini["prod_L2_scar"], gini["prod_L2_knife"],
            ("scar", "knife")),
        "L3_scar_minus_knife": _verschil(
            "c|L3sk", gini["prod_L3_scar"], gini["prod_L3_knife"],
            ("scar", "knife")),
    }
    uit["L2_scar_minus_knife"]["claim"] = "chapter: indistinguishable (0.006)"
    uit["L3_scar_minus_knife"]["claim"] = "chapter: indistinguishable (0.004)"
    for r in ("L1", "L4"):
        k = f"{r}_scar_minus_knife"
        uit[k] = _verschil(f"c|{r}sk", gini[f"prod_{r}_scar"],
                           gini[f"prod_{r}_knife"], ("scar", "knife"))
        uit[k]["claim"] = "chapter: scarcity and knife-edge separate at L1 and L4"
    for r in RUNGS:
        k = f"{r}_abund_minus_knife"
        uit[k] = _verschil(f"c|{r}ak", gini[f"prod_{r}_abund"],
                           gini[f"prod_{r}_knife"], ("abund", "knife"))
        uit[k]["claim"] = "chapter: abundance is the most unequal at every rung"
    return uit


def kanaal_contrasten(gini: dict) -> dict:
    """Knife-edge with the channel against knife-edge without it, per rung."""
    uit = {}
    for r in RUNGS:
        spreek = runset.cel(f"prod_{r}_knife")
        stil = runset.cel(f"prod_{r}_knife_nocomm")
        paren = {
            "final_gini": (gini[f"prod_{r}_knife"],
                           [_final_gini(p) for p in stil]),
            "mean_final_holding": ([_mean_holding(p) for p in spreek],
                                   [_mean_holding(p) for p in stil]),
            "transfer_pct": ([_transfer_pct(p) for p in spreek],
                             [_transfer_pct(p) for p in stil]),
            "fights_per_run": ([_fights(p) for p in spreek],
                               [_fights(p) for p in stil]),
        }
        rij = {}
        for q, (a, b) in paren.items():
            d = _verschil(f"ch|{r}|{q}", a, b, ("speaking", "silent"))
            # Both arms are printed in full whatever their size: with five
            # control runs the interval is a restatement of five numbers, and
            # the fifteen speaking runs are what it is being restated against.
            d["per_run"] = {"speaking": [round(float(x), 6) for x in a],
                            "silent": [round(float(x), 6) for x in b]}
            rij[q] = d
        uit[r] = rij
    return uit


def correlaties(gini: dict) -> dict:
    uit = {}
    # (a) L1 transfer share against final inequality, pooled over the rung.
    per_cel = {}
    for p_ in PAYOFFS:
        c = f"prod_L1_{p_}"
        paths = runset.cel(c)
        per_cel[c] = [(_transfer_pct(p), _final_gini(p)) for p in paths]
    uit["L1_transfer_share_vs_final_gini"] = _correlatie("r|L1tg", per_cel)
    uit["L1_transfer_share_vs_final_gini"]["claim"] = "chapter: r = +0.71"

    # (b) Welfare language against rescue, over the three L2 cells. Reported
    #     twice: on the raw counts, which is what figures.json stores per run,
    #     and on the rates, which is what "rate" in the chapter means. Runs
    #     with no agent below one resource have no rescue rate and are named.
    tellingen, tarieven, overgeslagen = {}, {}, []
    for p_ in PAYOFFS:
        c = f"prod_L2_{p_}"
        tellingen[c], tarieven[c] = [], []
        for p in runset.cel(c):
            w = _welfare(p)
            tellingen[c].append((w["welfare_sentences"], w["rescues"]))
            if w["rescue_pct"] is None or w["welfare_pct"] is None:
                overgeslagen.append(p.name)
            else:
                tarieven[c].append((w["welfare_pct"], w["rescue_pct"]))
    uit["L2_welfare_counts_vs_rescue_counts"] = _correlatie("r|L2wc", tellingen)
    uit["L2_welfare_counts_vs_rescue_counts"]["claim"] = (
        "chapter: r = +0.03, welfare language does not predict rescue at all")
    uit["L2_welfare_counts_vs_rescue_counts"]["skipped"] = []
    t = _correlatie("r|L2wr", {k: v for k, v in tarieven.items() if v})
    t["skipped"] = sorted(overgeslagen)
    t["skipped_note"] = ("no agent-round below one resource, so the run has no "
                         "rescue rate; dropping it is not the same as scoring "
                         "it zero and it is named rather than dropped silently")
    uit["L2_welfare_rate_vs_rescue_rate"] = t
    return uit


def exacte_tellingen() -> dict:
    F = "out/figures.json::m:stock-survives-silence"
    return {
        "gemma_L4_stock_at_capacity": _exact(30, 30,
            "Gemma L4 runs ending with the commons stock at capacity "
            "(10+10+10 over scarce, knife-edge and abundant)", F),
        "silent_L4_stock_at_capacity": _exact(0, 5,
            "no-channel L4 knife-edge control runs ending at capacity", F),
        "qwen_L4_stock_at_capacity": _exact(0, 5,
            "Qwen L4 knife-edge runs ending at capacity", F),
        "rota_scar": _exact(10, 10, "L4 scarce runs forming a rota",
                            "chapter (hand count), supplied with the brief"),
        "rota_knife": _exact(7, 10, "L4 knife-edge runs forming a rota",
                             "chapter (hand count), supplied with the brief"),
        "rota_abund": _exact(8, 10, "L4 abundant runs forming a rota",
                             "chapter (hand count), supplied with the brief"),
        "explicit_ceiling_all": _exact(20, 30,
            "L3 runs stating an explicit ceiling on a hand read",
            "chapter (hand read), supplied with the brief"),
        "explicit_ceiling_scar": _exact(6, 10, "hand-read ceiling, L3 scarce",
                                        "chapter (hand read)"),
        "explicit_ceiling_knife": _exact(9, 10, "hand-read ceiling, L3 knife",
                                         "chapter (hand read)"),
        "explicit_ceiling_abund": _exact(5, 10, "hand-read ceiling, L3 abundant",
                                         "chapter (hand read)"),
    }


def model_arm(gini: dict) -> dict:
    """Gemma against Qwen on the seeds both models were run on.

    Five seeds per rung, matched exactly, so the difference is taken within a
    seed before anything is averaged. The five differences are printed in full
    because at n=5 they are the finding and the interval is a summary of them.
    """
    rijen = runset.rijen()
    uit = {}
    for r in ("L2", "L3", "L4"):
        g = {x["seed"]: x for x in rijen if x["cel"] == f"prod_{r}_knife"}
        q = {x["seed"]: x for x in rijen if x["cel"] == f"robust_qwen_{r}_knife"}
        gedeeld = sorted(set(g) & set(q))
        pad_g = runset.WORTEL / f"prod_{r}_knife"
        pad_q = runset.WORTEL / f"robust_qwen_{r}_knife"
        paren = [(s,
                  _final_gini(pad_g / g[s]["bestand"]),
                  _final_gini(pad_q / q[s]["bestand"])) for s in gedeeld]
        d = [gv - qv for _, gv, qv in paren]
        rij = _gepaard(f"m|{r}", d)
        rij["direction"] = "gemma minus qwen"
        rij["pairs"] = [{"seed": s, "gemma": round(gv, 6), "qwen": round(qv, 6),
                         "difference": round(gv - qv, 6)} for s, gv, qv in paren]
        rij["skipped"] = sorted(
            [f"prod_{r}_knife/{v['bestand']}" for s, v in g.items()
             if s not in q])
        rij["skipped_note"] = ("Gemma runs on seeds Qwen was never run on; they "
                               "are in the unmatched cell figure and cannot "
                               "enter a paired comparison")
        uit[r] = rij
    return uit


# --- output -----------------------------------------------------------------

def _print(doc: dict) -> None:
    def rij(naam, e, lo, hi, extra=""):
        print(f"  {naam:<44} {e:>8.3f}  [{lo:>7.3f}, {hi:>7.3f}] {extra}")

    t = doc["targets"]
    print(f"\nrun-level uncertainty  ---  {RESAMPLES} resamples, "
          f"{int(LEVEL*100)}% intervals, seed {SEED}")
    pv = doc["provenance"]
    print(f"  provenance: {pv['memoised_runs_checked']} memoised runs agree "
          f"with a fresh log read (max diff {pv['max_abs_diff_final_gini']:.2e})")

    print("\nfinal Gini per production cell")
    for c, v in t["gini_by_cell"].items():
        rij(c, v["estimate"], v["ci_lo"], v["ci_hi"],
            "FRAGILE" if v["fragile"] else ("bca~" if not v.get("bca_agrees", True) else ""))

    print("\ncontrasts on final Gini")
    for k, v in t["gini_contrasts"].items():
        rij(k, v["estimate"], v["ci_lo"], v["ci_hi"],
            "straddles 0" if v["straddles_zero"] else "excludes 0")

    print("\nchannel contrasts, knife-edge (speaking minus silent)")
    for r, rijen_ in t["channel_contrasts"].items():
        for q, v in rijen_.items():
            rij(f"{r} {q}", v["estimate"], v["ci_lo"], v["ci_hi"],
                ("straddles 0" if v["straddles_zero"] else "excludes 0")
                + (" FRAGILE" if v["fragile"] else ""))

    print("\ncorrelations")
    for k, v in t["correlations"].items():
        rij(k, v["estimate"], v["ci_lo"], v["ci_hi"],
            f"n={v['n']} " + ("straddles 0" if v["straddles_zero"] else "excludes 0"))

    print("\nexact intervals for success counts (Clopper-Pearson | Jeffreys)")
    for k, v in t["exact_counts"].items():
        cp, j = v["clopper_pearson"], v["jeffreys"]
        print(f"  {k:<44} {v['successes']:>3}/{v['n']:<3} "
              f"[{cp['ci_lo']:.3f}, {cp['ci_hi']:.3f}] | "
              f"[{j['ci_lo']:.3f}, {j['ci_hi']:.3f}]")

    print("\nmodel arm, matched seeds (Gemma minus Qwen final Gini)")
    for r, v in t["model_arm_paired_seeds"].items():
        rij(f"{r} knife-edge", v["estimate"], v["ci_lo"], v["ci_hi"],
            "FRAGILE " + ("straddles 0" if v["straddles_zero"] else "excludes 0"))
        print(f"      pairs: {[p['difference'] for p in v['pairs']]}")
    print()


def main() -> None:
    gini = {c: [_final_gini(p) for p in runset.cel(c)] for c in PRODUCTIE}
    doc = {
        "generated_by": "tools/uncertainty.py",
        "rng_seed": SEED,
        "resamples": RESAMPLES,
        "ci_level": LEVEL,
        "resampling_unit": "the run; contrasts are stratified within cell",
        "free_parameters": {
            "interval_method": "percentile primary; BCa reported alongside "
                               "wherever n >= 10, with bca_agrees flagging a "
                               "shift of more than a tenth of the interval",
            "correlation": "Pearson primary; Spearman alongside",
            "fragile_threshold": FRAGILE_AT,
        },
        "provenance": _provenance(),
        "runset": {"cells": len(PRODUCTIE),
                   "production_runs": sum(len(v) for v in gini.values())},
        "targets": {
            "gini_by_cell": gini_per_cel(gini),
            "gini_contrasts": gini_contrasten(gini),
            "channel_contrasts": kanaal_contrasten(gini),
            "correlations": correlaties(gini),
            "exact_counts": exacte_tellingen(),
            "model_arm_paired_seeds": model_arm(gini),
        },
    }
    UIT.parent.mkdir(exist_ok=True)
    UIT.write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n")
    _print(doc)
    print(f"written to {UIT}")


if __name__ == "__main__":
    main()
