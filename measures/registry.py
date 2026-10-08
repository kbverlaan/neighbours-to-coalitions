"""One entry point: every reported figure, computed and written to out/figures.json.

    python3 registry.py                 # everything
    python3 registry.py m:scarce-transfers …   # a subset
    python3 registry.py --section grid

The keys are the `\\meth{}` labels used in the thesis, so a claim in the text and
the code that produced it share a name.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

HIER = Path(__file__).resolve().parent
for _m in ("figures", "core", "_shared"):
    sys.path.insert(0, str(HIER / _m))

import capacities, channel, grid, levers, models, paths  # noqa: E402

SECTIONS = {"grid": grid, "channel": channel, "paths": paths,
            "models": models, "capacities": capacities, "levers": levers}
UIT = HIER / "out" / "figures.json"


def commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=HIER, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def all_figures() -> dict:
    uit = {}
    for naam, mod in SECTIONS.items():
        for sleutel, fn in mod.FIGURES.items():
            uit[sleutel] = (naam, fn)
    return uit


def run(sleutels=None, section=None, bewaar=None) -> dict:
    reg = all_figures()
    if section:
        reg = {k: v for k, v in reg.items() if v[0] == section}
    if sleutels:
        ontbreekt = [k for k in sleutels if k not in reg]
        if ontbreekt:
            raise SystemExit(f"unknown figure key(s): {ontbreekt}\n"
                             f"known: {sorted(reg)}")
        reg = {k: reg[k] for k in sleutels}
    uit = {}
    # Several labels share one function --- the three accusation figures are all
    # `accusation_and_attack` --- and calling it once per label ran the same
    # scan over every run three times. Computed once, reported under each name.
    gedaan: dict = {}
    for sleutel, (sectie, fn) in reg.items():
        hergebruik = fn in gedaan
        print(f"  {sleutel:34} {sectie}.{fn.__name__}()"
              f"{'  [reused]' if hergebruik else ''}", flush=True)
        if not hergebruik:
            gedaan[fn] = fn()
        uit[sleutel] = {"section": sectie,
                        "source": f"figures/{sectie}.py::{fn.__name__}()",
                        "result": gedaan[fn]}
        # Written as we go. A full regeneration walks a 24 GB run set and takes
        # hours; when one was killed part-way it lost everything, because the
        # file was only written at the end. Now a killed run keeps what it
        # finished and the next one picks up the rest.
        if bewaar:
            bewaar(uit)
    return uit


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sec = None
    if "--section" in sys.argv:
        sec = sys.argv[sys.argv.index("--section") + 1]
        args = [a for a in args if a != sec]
    # A full regeneration takes hours on this run set and has been killed
    # part-way more than once. `--resume` builds into a staging file, skips what
    # is already there, and only replaces the real one when every figure is in.
    # A half-finished sweep therefore never becomes the chapter's artefact.
    hervat = "--resume" in sys.argv
    if hervat:
        UIT = HIER / "out" / "figures.staging.json"
        klaar = set()
        if UIT.exists():
            klaar = set(json.loads(UIT.read_text()).get("figures", {}))
        alles = sorted(all_figures())
        args = [k for k in (args or alles) if k not in klaar]
        if not args:
            echt = HIER / "out" / "figures.json"
            staging = json.loads(UIT.read_text())
            echt.write_text(json.dumps(staging, indent=1, ensure_ascii=False))
            print(f"staging is complete ({len(staging['figures'])} figures) "
                  f"-> {echt.name}")
            raise SystemExit(0)
        print(f"{len(klaar)} already done, {len(args)} to go", flush=True)
        sec = None
    def schrijf(deel):
        UIT.parent.mkdir(exist_ok=True)
        oud = json.loads(UIT.read_text()) if UIT.exists() else {"figures": {}}
        oud.setdefault("figures", {}).update(deel)
        oud["generated"] = str(date.today())
        oud["commit"] = commit()
        oud["runset"] = "data/thesis_final/"
        UIT.write_text(json.dumps(oud, indent=1, ensure_ascii=False))
        return oud

    nieuw = run(args or None, sec, bewaar=schrijf)
    oud = schrijf(nieuw)
    print(f"\n{len(nieuw)} figures -> {UIT.relative_to(HIER)} "
          f"({len(oud['figures'])} total, commit {oud['commit']})")
