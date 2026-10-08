"""Lettertype van de AAMAS-paper (Linux Libertine, acmart) voor alle figuren (8 okt 2026).
Aanroepen NA de eigen style(), zodat dit het laatste woord heeft."""
from pathlib import Path
from matplotlib import font_manager, rcParams

import os
LIB = Path(os.environ.get("LIBERTINE_DIR", Path.home() / "Library/texmf/fonts/opentype/public/libertine"))
NAME = "Linux Libertine O"
# Greenkeep palette (GitProjects/greenkeep/app/web/tokens.css), checked with the dataviz validator 8 Oct 2026:
# categorical blue/green/orange and the pair green/peach pass CVD and normal-vision separation.
INKGREEN = "#14372A"   # single-series marks
GREEN, PEACH = "#1F7A47", "#E4805B"   # rises/falls, action set/price
GREEN_TINT = "#9CC3AD"   # #1F7A47 mixed ~55% with white: bars that leave room for dark interval lines
PRICE_CELLS = {"scar": "#B98600", "knife": "#2F6BDB", "abund": "#1F7A47"}   # scarce yellow, knife-edge blue, abundant green (validated)

def apply():
    if not (LIB / "LinLibertine_R.otf").exists():
        return   # Linux Libertine not installed: keep matplotlib's default font
    for f in ("LinLibertine_R.otf", "LinLibertine_RB.otf", "LinLibertine_RI.otf", "LinLibertine_RBI.otf"):
        font_manager.fontManager.addfont(str(LIB / f))
    rcParams.update({"font.family": "serif", "font.serif": [NAME], "font.sans-serif": [NAME],
                     "mathtext.fontset": "custom", "mathtext.rm": NAME, "mathtext.it": f"{NAME}:italic",
                     "mathtext.bf": f"{NAME}:bold", "mathtext.sf": NAME, "axes.unicode_minus": False})
# Note: pdf.fonttype 42 with this CFF-based OpenType font gives a font-type mismatch; Type 3 (vector) is kept.
