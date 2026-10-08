"""Is de scheiding tussen treden in de redeneertraces woordenschat? (AAMAS, 7 okt 2026; verkennend.)

Gestratificeerde steekproef van de traces uit de cache van fig. 8/9 (zelfde verzameling, zelfde
tail van 2.400 tekens). Twee keer embedden met hetzelfde model: (1) ongewijzigd, (2) met de
actiewoorden van alle treden vervangen door één neutraal woord. Scheiding tussen treden in de
768-d ruimte zelf (geen UMAP): afstand tussen centroïden gedeeld door de gepoolde spreiding,
dezelfde maat als fig9_trajectories.separation, plus de nauwkeurigheid van een
nearest-centroid-classificatie op een held-out helft.
  uv run --python 3.12 --with numpy --with sentence-transformers --with einops python paper_aamas/trace_masking.py
"""
import sys, re, json, random
from pathlib import Path
import numpy as np
HIER = Path(__file__).resolve().parent; M = HIER.parent
sys.path.insert(0, str(M / "plots"))
for s in ("core", "_shared"): sys.path.insert(0, str(M / s))
import _reasoning_cache as rc

PER_CELL = 1200                                   # 12 cellen -> 14.400 traces
SEED = 20261007
# actiewoorden en hun directe vormen, alle treden; vervangen door "act"
WORDS = r"""transfer(s|red|ring)? invest(s|ed|ing|ment)? give(s|n)? giving gave gift(s)? hold(s|ing)? held
take(s|n)? taking took attack(s|ed|er|ers|ing)? strike(s)? arm(s|ed|ing)? strengthen(s|ed|ing)?
combat fight(s|ing)? fought raid(s|ed|ing)? predat(e|ion|or|ory)
drop(s|ped|ping)? invite(s|d)? inviting rewir(e|ed|ing) disconnect(s|ed|ing)? connect(s|ed|ing|ion|ions)?
harvest(s|ed|ing)? stock commons quota(s)? regenerat(e|es|ion|ing) cap""".split()
PAT = re.compile(r"\b(" + "|".join(WORDS) + r")\b", re.I)

rows, texts, skipped = rc.collect()
rng = random.Random(SEED)
idx = []
for cell in sorted({r["cell"] for r in rows}):
    ii = [i for i, r in enumerate(rows) if r["cell"] == cell]
    idx += rng.sample(ii, min(PER_CELL, len(ii)))
sub_rows = [rows[i] for i in idx]; sub = [texts[i] for i in idx]
masked = [PAT.sub("act", t[-rc.TAIL_CHARS:]) for t in sub]
share = np.mean([len(PAT.findall(t[-rc.TAIL_CHARS:])) / max(len(t[-rc.TAIL_CHARS:].split()), 1) for t in sub])
print(f"{len(sub)} traces; masked words {share:.1%} of tokens", flush=True)

E0 = rc.embed(sub).astype(np.float32)
E1 = rc.embed(masked).astype(np.float32)
rung = np.array([r["rung"] for r in sub_rows])

def sep(E):
    out = {}
    R = ("L1", "L2", "L3", "L4")
    for i, a in enumerate(R):
        for b in R[i + 1:]:
            pa, pb = E[rung == a], E[rung == b]
            gap = float(np.linalg.norm(pa.mean(0) - pb.mean(0)))
            pooled = float(np.sqrt((pa.var(0).sum() + pb.var(0).sum()) / 2))
            out[f"{a}-{b}"] = round(gap / pooled, 3)
    return out

def nc_acc(E):
    n = len(E); perm = np.random.default_rng(SEED).permutation(n); tr, te = perm[: n // 2], perm[n // 2:]
    R = ("L1", "L2", "L3", "L4")
    C = np.stack([E[tr][rung[tr] == r].mean(0) for r in R])
    pred = np.array(R)[np.argmax(E[te] @ C.T, axis=1)]
    return round(float((pred == rung[te]).mean()), 3)

res = {"n": len(sub), "masked_token_share": round(float(share), 4),
       "separation_raw": sep(E0), "separation_masked": sep(E1),
       "nearest_centroid_acc_raw": nc_acc(E0), "nearest_centroid_acc_masked": nc_acc(E1), "chance": 0.25}
print(json.dumps(res, indent=1))
(HIER / "trace_masking.json").write_text(json.dumps(res, indent=1))
