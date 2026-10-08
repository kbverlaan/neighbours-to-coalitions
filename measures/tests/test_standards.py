"""The seven standards, checked rather than remembered.

Every failure mode here has actually happened in this project. The tests exist
because discipline did not hold: an unreadable directory was silently reported
as an empty cell three times in one day, two figures disagreed about a
denominator for a month, and a definition with an unstated window produced a
number that could not be reproduced from its own description.

    python3 tests/test_standards.py
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

HIER = Path(__file__).resolve().parent
WORTEL = HIER.parent
for _m in ("figures", "core", "_shared"):
    sys.path.insert(0, str(WORTEL / _m))

import runset            # noqa: E402
import text as T         # noqa: E402
import turns             # noqa: E402
from result import Result  # noqa: E402

FOUTEN: list[str] = []


# Two of the checks below need the generated figures. The file runs as a script
# (`python3 tests/test_standards.py`), and under pytest those two would error on
# a missing argument rather than run, which is a test that silently does not
# test. This fixture gives pytest the same object the script builds.
try:
    import pytest

    @pytest.fixture
    def fig() -> dict:
        pad = WORTEL / "out" / "figures.json"
        if not pad.exists():
            pytest.skip("out/figures.json missing — run registry.py first")
        return json.loads(pad.read_text())["figures"]
except ImportError:      # running as a plain script
    pass


def eis(voorwaarde, boodschap):
    if not voorwaarde:
        FOUTEN.append(boodschap)
    return bool(voorwaarde)


# --- 1. One run set --------------------------------------------------------

def test_one_runset():
    """No module may reach for files outside the declared run set."""
    verboden = re.compile(r"Desktop/thesis-runs|~/origins|\.claude/jobs|/scratch/")
    for p in list(WORTEL.glob("core/*.py")) + list(WORTEL.glob("figures/*.py")):
        t = p.read_text()
        eis(not verboden.search(t), f"{p.name} contains a path outside the run set")
        if p.parent.name == "figures":
            eis("def rounds" not in t and "json.loads" not in t,
                f"{p.name} parses the log itself; that belongs in core/")


# --- 2. n and denominator travel with every value --------------------------

def test_denominators(fig: dict):
    def loop(k, v):
        if isinstance(v, dict):
            if "value" in v and "unit" in v and v.get("unit"):
                if v["unit"] in ("agent-turns", "sentences", "messages"):
                    # `in`, not truthiness, for the same reason as n below: a
                    # denominator of zero is a real denominator, and a share
                    # over it must be None rather than 0.0.
                    eis("denominator" in v, f"{k}: share without a denominator")
                    eis(v.get("denominator") or v.get("value") is None,
                        f"{k}: a share over an empty denominator must be None")
                # `in`, not truthiness: n = 0 is a legitimate value --- a cell
                # where no run had a fight --- and the whole point of this
                # package is that zero and missing are different claims.
                eis("n" in v, f"{k}: result without an n")
            for kk, vv in v.items():
                loop(f"{k}.{kk}", vv)
        elif isinstance(v, list):
            for i, vv in enumerate(v):
                loop(f"{k}[{i}]", vv)
    for sleutel, blok in fig.items():
        loop(sleutel, blok.get("result"))


# --- 3. Counts carry a chance expectation ----------------------------------

def test_baselines():
    """A text share must be able to state what chance yields on the same corpus."""
    paden = runset.cel("prod_L1_knife")[:2]
    r = T.repetition(paden)
    eis(r.baseline is not None, "text.repetition returns no baseline")
    eis(r.denominator, "text.repetition returns no denominator")


# --- 4. An unreadable source is never an empty one -------------------------

def test_unreadable_is_an_error():
    for slecht in ("prod_L9_nonexistent", "", "prod_L1"):
        try:
            runset.cel(slecht)
        except runset.RunsetError:
            continue
        except Exception as e:
            FOUTEN.append(f"cel({slecht!r}) raised {type(e).__name__}, not RunsetError")
            continue
        FOUTEN.append(f"cel({slecht!r}) returned a list instead of failing")


# --- 5. Skipped runs are named ---------------------------------------------

def test_skipped_is_reportable():
    eis("skipped" in Result(value=1).as_dict() or Result(value=1).skipped == [],
        "Result has no field for skipped runs")


# --- 6. Twice is the same --------------------------------------------------

def test_determinism():
    import grid
    for naam in ("severings", "l1_message_collapse"):
        fn = getattr(grid, naam)
        a = json.dumps(fn(), sort_keys=True, default=str)
        b = json.dumps(fn(), sort_keys=True, default=str)
        eis(a == b, f"grid.{naam}() is not deterministic")


# --- 7. A free parameter reports its alternative ----------------------------

VRIJE_PARAMETERS = {
    "m:scarce-transfers": ("attack_on_giver_only", "rule_order"),
    "m:pairing-capacity-level": ("if_one_direction_suffices",),
    "m:support-stays-inside": ("published_reading",),
}


def test_sensitivity(fig: dict):
    for sleutel, verwacht in VRIJE_PARAMETERS.items():
        blok = fig.get(sleutel)
        if not blok:
            FOUTEN.append(f"{sleutel}: not in figures.json, cannot check sensitivity")
            continue
        plat = json.dumps(blok, default=str)
        for v in verwacht:
            eis(v in plat, f"{sleutel}: free parameter '{v}' has no reported alternative")


# --- the cell counts must match the index ----------------------------------

def test_cell_counts():
    telling = runset.count()
    for c in runset.PRODUCTION:
        eis(len(runset.cel(c)) == telling[c],
            f"{c}: cel() returns {len(runset.cel(c))} of {telling[c]} indexed runs")
    eis(sum(telling[c] for c in runset.PRODUCTION) == 150,
        f"production arm is {sum(telling[c] for c in runset.PRODUCTION)}, expected 150")


if __name__ == "__main__":
    pad = WORTEL / "out" / "figures.json"
    fig = json.loads(pad.read_text())["figures"] if pad.exists() else {}
    if not fig:
        print("out/figures.json is empty — run registry.py first "
              "(tests 2 and 7 need it)\n")

    test_one_runset()
    test_unreadable_is_an_error()
    test_skipped_is_reportable()
    test_cell_counts()
    test_baselines()
    test_determinism()
    if fig:
        test_denominators(fig)
        test_sensitivity(fig)

    if FOUTEN:
        print(f"{len(FOUTEN)} standard(s) violated:\n")
        for f in FOUTEN:
            print(f"  - {f}")
        raise SystemExit(1)
    print(f"all standards hold ({len(fig)} figures checked)")


def test_no_unstable_seeds():
    """A seed derived from `hash()` gives different draws in every process.

    `random.Random(hash(name))` looks deterministic and is not: string hashing is
    salted per interpreter. Two shuffle baselines were seeded that way and moved
    by a point or two between one computation of the figure and the next, which
    is the sixth standard failing without saying so. `model.seeded()` is the
    replacement.
    """
    # Read as code, not as text. The pattern matched the docstring of
    # `model.seeded()` --- the one place that explains why the pattern is
    # forbidden --- and the test failed on its own documentation.
    for p in list(WORTEL.glob("figures/*.py")) + list(WORTEL.glob("core/*.py")):
        boom = ast.parse(p.read_text())
        letterlijk = {id(n) for n in ast.walk(boom) if isinstance(n, ast.Constant)}
        code = "".join(
            ast.unparse(n) for n in ast.walk(boom)
            if isinstance(n, ast.Call) and id(n) not in letterlijk).replace(" ", "")
        eis("Random(hash(" not in code,
            f"{p.name} seeds a generator from hash(); use model.seeded()")


def test_nothing_was_violated():
    """Under pytest the checks above only collect; this is what fails the run.

    `eis` appends rather than raises so the script can print every violation at
    once instead of stopping at the first. That made the pytest run green while
    two standards were being broken, which is worse than having no test. This
    runs last and asserts the collected list is empty.
    """
    assert not FOUTEN, "\n".join(FOUTEN)
