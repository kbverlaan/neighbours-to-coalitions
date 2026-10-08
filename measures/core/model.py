"""The tests: variance attributable to the design, association, contingency.

Four things are asked of the run-level numbers in this chapter, and each is
asked in several places, so each is written once here.

  anova2      how the two levers divide an outcome, with the interaction
  correlate   whether two run-level quantities move together
  contingency a two-by-two count with an exact test
  arms        the same quantity computed on subsets of the run set

`arms` is the one that keeps the reporting honest. Three findings in this study
turned out to differ between the cluster and the OpenRouter runs, and none of
them would have been visible from a single pooled number. Any figure that moved
when the run set grew is required by the standards test to report its arms.

The interaction term is returned by `anova2` and not left as an afterthought:
an additive split of variance between two levers is only a fair description of
a design in which the levers do not interact, and on this study's economic
outcomes they interact as strongly as the main effects.
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))

from result import Result      # noqa: E402
import runset                  # noqa: E402


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def anova2(cells: dict[tuple[str, str], list[float]]) -> Result:
    """Two-way ANOVA with interaction, Type II sums of squares, partial eta squared.

    `cells` maps (levelA, levelB) -> the run-level values in that cell. Unequal
    cell sizes are handled by Type II: each main effect is tested after the
    other main effect but before the interaction.
    """
    alle = [x for v in cells.values() for x in v]
    N = len(alle)
    grand = _mean(alle)
    A = sorted({a for a, _ in cells})
    B = sorted({b for _, b in cells})

    def ss_model(groepen):
        return sum(len(v) * (_mean(v) - grand) ** 2 for v in groepen.values())

    per_a, per_b = defaultdict(list), defaultdict(list)
    for (a, b), v in cells.items():
        per_a[a].extend(v)
        per_b[b].extend(v)

    ss_tot = sum((x - grand) ** 2 for x in alle)
    ss_cellen = sum(len(v) * (_mean(v) - grand) ** 2 for v in cells.values())
    ss_res = ss_tot - ss_cellen
    ss_a, ss_b = ss_model(per_a), ss_model(per_b)
    ss_ab = ss_cellen - ss_a - ss_b

    df_a, df_b = len(A) - 1, len(B) - 1
    df_ab = df_a * df_b
    df_res = N - len(cells)
    ms_res = ss_res / df_res if df_res else float("nan")

    def term(ss, df):
        eta = ss / (ss + ss_res) if (ss + ss_res) else 0.0
        F = (ss / df) / ms_res if df and ms_res else float("nan")
        return {"eta2p": round(eta, 3), "F": round(F, 2), "df": [df, df_res],
                "p": round(_f_p(F, df, df_res), 5)}

    return Result(
        value={"A": term(ss_a, df_a), "B": term(ss_b, df_b),
               "AxB": term(ss_ab, df_ab),
               # How much of the total spread the design accounts for at all.
               # The complement is what remains between runs sharing every
               # parameter, which is the subject of the path-divergence section.
               "model_r2": round(ss_cellen / ss_tot, 3) if ss_tot else None,
               "residual_share": round(ss_res / ss_tot, 3) if ss_tot else None},
        n=N, denominator=N, unit="runs",
        note="Type II SS; partial eta squared uses the residual from the full "
             "model including the interaction")


def cohen_d(xs, ys) -> float:
    """Standardised mean difference on the pooled within-group SD.

    The registration fixes SESOI at d = 1.0, so this is the quantity a
    confirmatory decision turns on and not a summary added afterwards. Returns
    nan where either group has fewer than two runs or the pooled SD is zero ---
    a cell with no variance cannot produce a standardised effect, and reporting
    0.0 there would read as "no effect" instead of "not defined".
    """
    n1, n2 = len(xs), len(ys)
    if n1 < 2 or n2 < 2:
        return float("nan")
    m1, m2 = _mean(xs), _mean(ys)
    v1 = sum((x - m1) ** 2 for x in xs) / (n1 - 1)
    v2 = sum((y - m2) ** 2 for y in ys) / (n2 - 1)
    sp = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
    return (m1 - m2) / math.sqrt(sp) if sp > 0 else float("nan")


def anova1(groups: dict[str, list[float]]) -> dict:
    """One-way ANOVA: F, p, partial eta squared, and every pairwise Cohen's d.

    Written for the registered P2 test, which asks whether the payoff sweep
    moves an outcome *within* a rung. The pairwise d's come back with it because
    the registration requires an effect at or above SESOI as well as p < alpha,
    and an omnibus F cannot say which contrast carries it.
    """
    alle = [x for v in groups.values() for x in v]
    N, k = len(alle), len(groups)
    grand = _mean(alle)
    ss_b = sum(len(v) * (_mean(v) - grand) ** 2 for v in groups.values())
    ss_w = sum((x - _mean(v)) ** 2 for v in groups.values() for x in v)
    df_b, df_w = k - 1, N - k
    F = (ss_b / df_b) / (ss_w / df_w) if df_w and ss_w else float("nan")
    namen = sorted(groups)
    paren = {}
    for i, a in enumerate(namen):
        for b in namen[i + 1:]:
            d = cohen_d(groups[a], groups[b])
            paren[f"{a}_vs_{b}"] = None if d != d else round(d, 2)
    geldig = [abs(v) for v in paren.values() if v is not None]
    return {"F": round(F, 2) if F == F else None,
            "df": [df_b, df_w],
            "p": round(_f_p(F, df_b, df_w), 5) if F == F else None,
            "eta2p": round(ss_b / (ss_b + ss_w), 3) if (ss_b + ss_w) else 0.0,
            "means": {a: round(_mean(v), 3) for a, v in groups.items()},
            "n": {a: len(v) for a, v in groups.items()},
            "pairwise_d": paren,
            "largest_abs_d": round(max(geldig), 2) if geldig else None}


def _f_p(F, df1, df2):
    """Upper-tail p for an F statistic, via the regularised incomplete beta."""
    if not (F == F) or F <= 0 or df1 <= 0 or df2 <= 0:
        return float("nan")
    x = df2 / (df2 + df1 * F)
    return _betainc(df2 / 2, df1 / 2, x)


def _betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lb = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
          + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return math.exp(lb) * _betacf(a, b, x) / a
    return 1 - math.exp(lb) * _betacf(b, a, 1 - x) / b


def _betacf(a, b, x, it=200, eps=1e-12):
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > 1e-30 else 1e-30)
    h = d
    for m in range(1, it):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)),
                   -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1 + aa * d
            d = 1 / (d if abs(d) > 1e-30 else 1e-30)
            c = 1 + aa / (c if abs(c) > 1e-30 else 1e-30)
            h *= d * c
        if abs(d * c - 1) < eps:
            break
    return h


def correlate(xs, ys) -> Result:
    """Pearson r with a two-sided p, over paired run-level values."""
    n = len(xs)
    if n != len(ys) or n < 3:
        return Result(value=None, n=n, note="too few pairs")
    mx, my = _mean(xs), _mean(ys)
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    if sxx == 0 or syy == 0:
        return Result(value=None, n=n, note="a series has no variance")
    r = sxy / math.sqrt(sxx * syy)
    r = max(-1.0, min(1.0, r))
    if abs(r) >= 1:
        p = 0.0
    else:
        t = r * math.sqrt((n - 2) / (1 - r * r))
        p = _f_p(t * t, 1, n - 2)
    return Result(value=round(r, 3), n=n, denominator=n, unit="runs",
                  sensitivity={"p": round(p, 5)})


def contingency(a: int, b: int, c: int, d: int) -> Result:
    """Two-by-two counts with a two-sided Fisher exact test."""
    def logC(n, k):
        return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

    n = a + b + c + d
    rij1, kol1 = a + b, a + c
    waar = logC(kol1, a) + logC(n - kol1, rij1 - a) - logC(n, rij1)
    p = 0.0
    for i in range(max(0, rij1 - (n - kol1)), min(rij1, kol1) + 1):
        lp = logC(kol1, i) + logC(n - kol1, rij1 - i) - logC(n, rij1)
        if lp <= waar + 1e-9:
            p += math.exp(lp)
    return Result(value=[[a, b], [c, d]], n=n, denominator=n, unit="runs",
                  sensitivity={"fisher_p": round(min(1.0, p), 5)},
                  note="two-sided Fisher exact")


def arms(cell: str) -> dict[str, list[Path]]:
    """Split one cell by where its runs were produced.

    Three findings in this study differ between the cluster and OpenRouter runs.
    Any figure that moved when the run set grew reports both.
    """
    rijen = [r for r in runset.rijen() if r["cel"] == cell]
    if not rijen:
        raise runset.RunsetError(f"cell '{cell}' is not in the index")
    uit: dict[str, list[Path]] = {"all": [], "cluster": [], "openrouter": []}
    for r in sorted(rijen, key=lambda r: r["nieuw_id"]):
        p = runset.WORTEL / cell / r["bestand"]
        uit["all"].append(p)
        uit["openrouter" if r["bron"] == "openrouter" else "cluster"].append(p)
    return {k: v for k, v in uit.items() if v}


def seeded(naam: str):
    """A random generator that gives the same draws in every process.

    `hash()` on a string is salted per interpreter, so `random.Random(hash(x))`
    produces different numbers on every run. A shuffle baseline seeded that way
    moves by a point or two between two invocations of the same code, which is
    the sixth standard --- running twice gives the same answer --- failing
    quietly. It was caught by comparing a figure against its own recomputation.

    CRC32 is not a good hash and does not need to be; it needs only to be the
    same one tomorrow.
    """
    import random
    from zlib import crc32
    return random.Random(crc32(naam.encode()))
