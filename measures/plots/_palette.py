"""Whether the three payoff colours stay apart for a colourblind reader.

    python3 plots/_palette.py

The design note claimed Okabe-Ito is safe and said outright that nothing had
checked it. This checks it: each pair of colours is simulated under
deuteranopia, protanopia and tritanopia (Brettel/Viénot, as implemented in the
usual LMS matrices) and the distance between them is measured in OKLab, where a
unit is roughly a unit of perceived difference.

The threshold is the one the visualisation guidance uses: a distance of 8 or
more (OKLab times 100) is a pass, 6 to 8 is a floor that needs a second channel
carrying the same distinction, and below 6 the two are one colour. Normal
vision has its own floor of 15, since a pair that is far apart under simulation
and close in the original is a palette that only works for the impaired reader.
"""
from __future__ import annotations

import numpy as np

PALET = {"scarce": "#0072B2", "knife-edge": "#009E73", "abundant": "#D55E00"}

# sRGB -> LMS (Hunt-Pointer-Estevez, normalised to D65), and back
RGB2LMS = np.array([[0.31399022, 0.63951294, 0.04649755],
                    [0.15537241, 0.75789446, 0.08670142],
                    [0.01775239, 0.10944209, 0.87256922]])
LMS2RGB = np.linalg.inv(RGB2LMS)

SIM = {  # Viénot, Brettel & Mollon 1999 --- the standard dichromat projections
    "deuteranopia": np.array([[1.0, 0.0, 0.0],
                              [0.49421, 0.0, 1.24827],
                              [0.0, 0.0, 1.0]]),
    "protanopia": np.array([[0.0, 2.02344, -2.52581],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0]]),
    "tritanopia": np.array([[1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [-0.395913, 0.801109, 0.0]]),
}


def _lineair(c):
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def hex_naar_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def simuleer(rgb, soort):
    lms = RGB2LMS @ _lineair(rgb)
    return np.clip(LMS2RGB @ (SIM[soort] @ lms), 0, 1)


def oklab(lineair_rgb):
    m = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                  [0.2119034982, 0.6806995451, 0.1073969566],
                  [0.0883024619, 0.2817188376, 0.6299787005]])
    lms = np.cbrt(m @ lineair_rgb)
    n = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                  [1.9779984951, -2.4285922050, 0.4505937099],
                  [0.0259040371, 0.7827717662, -0.8086757660]])
    return n @ lms


def afstand(a, b, soort=None):
    """OKLab distance x100 between two hex colours, optionally simulated."""
    ra, rb = hex_naar_rgb(a), hex_naar_rgb(b)
    if soort:
        ra, rb = simuleer(ra, soort), simuleer(rb, soort)
        la, lb = oklab(ra), oklab(rb)          # already linear from simuleer
    else:
        la, lb = oklab(_lineair(ra)), oklab(_lineair(rb))
    return float(np.linalg.norm(la - lb) * 100)


def main() -> int:
    namen = list(PALET)
    fout = 0
    print(f"{'pair':26} {'normal':>7} {'deuter':>7} {'protan':>7} {'tritan':>7}   verdict")
    for i, a in enumerate(namen):
        for b in namen[i + 1:]:
            n = afstand(PALET[a], PALET[b])
            sims = {s: afstand(PALET[a], PALET[b], s) for s in SIM}
            laagste = min(sims.values())
            if laagste >= 8 and n >= 15:
                oordeel = "PASS"
            elif laagste >= 6 and n >= 15:
                oordeel = "FLOOR — needs a second channel"
                fout += 1
            else:
                oordeel = "FAIL"
                fout += 1
            print(f"{a + ' / ' + b:26} {n:7.1f} "
                  f"{sims['deuteranopia']:7.1f} {sims['protanopia']:7.1f} "
                  f"{sims['tritanopia']:7.1f}   {oordeel}")
    print("\nthreshold: >=8 passes, 6-8 is a floor, normal vision must exceed 15")
    return fout


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
