"""Hoe het kanaal gebruikt wordt, per trede (AAMAS, 30 sep 2026). Toegevoegde maat, niet geregistreerd.
Per run: aandeel berichten aan één ontvanger, ontvangers per bericht, aandeel berichten dat het
vorige bericht van dezelfde afzender herhaalt (tekst met namen en getallen gemaskeerd), en aandeel
dat al eerder in de run gezegd is. Mediaan over de runs van een trede (alle prijzen).
    /opt/homebrew/bin/python3.10 paper_aamas/channel_use.py
"""
import sys, json, re, statistics as st
from pathlib import Path
HIER = Path(__file__).resolve().parent
for s in ("core", "_shared"): sys.path.insert(0, str(HIER.parent / s))
import runset
def norm(t, names):
    t = re.sub(r"\d+(\.\d+)?", "#", t.lower())
    for n in names: t = re.sub(r"\b" + re.escape(n.lower()) + r"\b", "@", t)
    return re.sub(r"\s+", " ", t).strip()
uit = {}
for r in ("L1", "L2", "L3", "L4"):
    rows = []
    for p in ("scar", "knife", "abund"):
        for f in runset.cel(f"prod_{r}_{p}"):
            R = [json.loads(l) for l in open(f) if l.strip()]
            names = list((R[0].get("agents") or {}).keys()); seen, last = set(), {}
            n = one = rec = own = rep = n30 = own30 = 0
            for d in R:
                for m in d.get("messages") or []:
                    to = m.get("to"); tos = to if isinstance(to, list) else [to]
                    t = norm(m.get("text") or "", names); s = m.get("from"); n += 1
                    one += len(tos) == 1; rec += len(tos); own += last.get(s) == t; rep += t in seen
                    if d.get("round", 0) >= 30: n30 += 1; own30 += last.get(s) == t
                    seen.add(t); last[s] = t
            rows.append({"single": one / n, "recipients": rec / n, "own_repeat": own / n,
                         "own_repeat_late": own30 / n30 if n30 else 0, "run_repeat": rep / n, "messages": n})
    uit[r] = {k: round(st.median([x[k] for x in rows]), 3) for k in rows[0]}
(HIER / "channel_use.json").write_text(json.dumps(uit, indent=1)); print(json.dumps(uit, indent=1))
