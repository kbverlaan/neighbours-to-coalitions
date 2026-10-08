"""Where the paper scripts write (AAMAS 2027 submission). Set AAMAS_PAPER_DIR to the paper's
source folder to write tables and figures into it; by default everything goes to ./out."""
import os
from pathlib import Path
HIER = Path(__file__).resolve().parent
PAPER = Path(os.environ.get("AAMAS_PAPER_DIR", HIER / "out"))
(PAPER / "figures").mkdir(parents=True, exist_ok=True)
(PAPER / "sections").mkdir(parents=True, exist_ok=True)
(PAPER / "supplementary").mkdir(parents=True, exist_ok=True)
