"""Volgorde bij L3: bericht -> uitnodiging -> gedeelde buurt -> gezamenlijke aanval (AAMAS, 8 okt 2026; verkennend,
op vraag van Debraj). Leest per ronde berichten (zonden in ronde r, gelezen in r+1), uitnodigingen
(rewire_intent.invite, outcome 'added'), drops, de graaf (network.edges, eind van de ronde) en gevechten.

A. Doel van een uitnodiging i->j in ronde t, venster t..t+W: j wordt door i aangevallen ('target'), j valt samen met i
   een derde aan ('partner'), anders 'other'. Kans: zelfde indeling voor alle agents die i in ronde t had kunnen
   uitnodigen (levend, geen buur), gemiddeld per uitnodiging.
B. Bericht vooraf: i en j wisselden een bericht (beide richtingen) in t-3..t-1; of i zond/ontving een bericht dat j
   bij naam noemt in t-3..t-1. Met dezelfde kansbasis.
C. Behoud: nieuwe banden uit uitnodigingen, nog aanwezig W rondes later; bij doelwit-uitnodigingen gevolgd door een
   aanval: binnen 3 rondes na de aanval gedropt?
D. Keten per deelname aan een gezamenlijke aanval (>= 2 aanvallers) op T in ronde s, tegen solo-aanvallen:
   eigen uitnodiging aan T in s-W..s; daarvoor een bericht dat T noemt (gezonden of ontvangen); en een mede-aanvaller
   die T ook in s-W..s uitnodigde (gedeelde buurt, samen gebouwd).
  uv run --python 3.12 python paper_aamas/rewire_sequence.py
"""
import sys, json, re
from collections import defaultdict
from pathlib import Path
HIER = Path(__file__).resolve().parent; M = HIER.parent
for s in ("core", "_shared"): sys.path.insert(0, str(M / s))
import runset
W = 5

def load(p):
    R = {}
    for line in open(p):
        try: d = json.loads(line)
        except json.JSONDecodeError: continue
        R[d["round"]] = d
    return [R[k] for k in sorted(R)]

def names_in(text, names):
    return {n for n in names if re.search(rf"\b{re.escape(n)}\b", text or "")}

def run(p, acc):
    rs = load(p); names = list(rs[0]["agents"]); idx = {d["round"]: d for d in rs}
    edges = {d["round"]: {frozenset(e) for e in d["network"]["edges"]} for d in rs}
    msgs = defaultdict(list)                 # round -> (from, set(to), named)
    for d in rs:
        for m in d.get("messages") or []:
            msgs[d["round"]].append((m.get("from"), set(m.get("to") or []), names_in(m.get("text"), names)))
    invites = []; inv_by = defaultdict(list)  # (t, i, j)
    for d in rs:
        for a, x in d["agents"].items():
            ri = x.get("rewire_intent") or {}
            if ri.get("invite") and ri.get("invite_outcome") == "added":
                invites.append((d["round"], a, ri["invite"])); inv_by[a].append((d["round"], ri["invite"]))
    fights = [(d["round"], set(f["attackers"]), f["defender"]) for d in rs for f in (d.get("combat") or [])]
    last = rs[-1]["round"]
    for d in rs:
        ph = "early" if d["round"] <= 20 else ("mid" if d["round"] <= 40 else "late")
        acc[f"G_{ph}_deg"] += 2 * len(d["network"]["edges"]) / len(d["agents"]); acc[f"G_{ph}_r"] += 1
    for (t, i, j) in invites:
        ph = "early" if t <= 20 else ("mid" if t <= 40 else "late"); acc[f"I_{ph}"] += 1
    def role(i, j, t):
        att = any(i in A and D == j for (s, A, D) in fights if t <= s <= t + W)
        par = any(i in A and j in A for (s, A, D) in fights if t <= s <= t + W)
        return "target" if att else ("partner" if par else "other")
    def contact(i, j, t):
        return any((f == i and j in to) or (f == j and i in to) for s in range(t - 3, t) for (f, to, _) in msgs[s])
    def named(i, j, t):
        return any(j in nm and (f == i or i in to) for s in range(t - 3, t) for (f, to, nm) in msgs[s])
    for (t, i, j) in invites:
        acc["A_n"] += 1; acc["A_" + role(i, j, t)] += 1
        acc["B_contact"] += contact(i, j, t); acc["B_named"] += named(i, j, t)
        prev = edges.get(t - 1, set()); alive = list(idx[t]["agents"])
        cand = [k for k in alive if k != i and frozenset((i, k)) not in prev]
        if cand:
            rr = [role(i, k, t) for k in cand]
            for r in ("target", "partner", "other"): acc["Ac_" + r] += rr.count(r) / len(cand)
            acc["Bc_contact"] += sum(contact(i, k, t) for k in cand) / len(cand)
            acc["Bc_named"] += sum(named(i, k, t) for k in cand) / len(cand)
        if t + W <= last:
            acc["C_n"] += 1; acc["C_kept"] += frozenset((i, j)) in edges[t + W]
    # C2: target-invites followed by an attack: dropped within 3 rounds after the attack?
    for (t, i, j) in invites:
        hits = [s for (s, A, D) in fights if i in A and D == j and t <= s <= t + W]
        if hits:
            s = hits[0]; acc["C2_n"] += 1
            acc["C2_gone"] += any(frozenset((i, j)) not in edges.get(r, set()) for r in range(s + 1, min(s + 4, last + 1)))
    # baseline for C: initial edges kept W rounds later, from round 2
    for e in edges.get(1, set()):
        if 1 + W <= last: acc["C0_n"] += 1; acc["C0_kept"] += e in edges[1 + W]
    # D
    for (s, A, T) in fights:
        k = "joint" if len(A) >= 2 else "solo"
        ph = "early" if s <= 20 else ("mid" if s <= 40 else "late")
        if k == "joint":
            acc[f"P_{ph}_fights"] += 1
            acc[f"P_{ph}_shared"] += any(any(j == T and s - W <= t <= s for (t, j) in inv_by[b]) for b in A)
        for a in A:
            acc[f"D_{k}_n"] += 1
            if k == "joint": acc[f"P_{ph}_n"] += 1
            own = [t for (t, j) in inv_by[a] if j == T and s - W <= t <= s]
            if own:
                acc[f"D_{k}_inv"] += 1; t0 = min(own)
                if any(T in nm and (f == a or a in to) for r in range(t0 - W, t0) for (f, to, nm) in msgs[r]):
                    acc[f"D_{k}_msg_inv"] += 1
                    if k == "joint" and any(any(j == T and s - W <= t <= s for (t, j) in inv_by[b]) for b in A - {a}):
                        acc["D_joint_chain"] += 1; acc[f"P_{ph}_chain"] += 1
            if k == "joint" and any(any(j == T and s - W <= t <= s for (t, j) in inv_by[b]) for b in A):
                acc["D_joint_shared"] += 1

def summ(cel):
    acc = defaultdict(float)
    for p in runset.cel(cel): run(p, acc)
    n = acc["A_n"] or 1; o = {"runs": len(runset.cel(cel)), "invites": int(acc["A_n"])}
    for r in ("target", "partner", "other"): o[f"invite_{r}"] = (acc["A_" + r] / n, acc["Ac_" + r] / n)
    o["msg_contact_before_invite"] = (acc["B_contact"] / n, acc["Bc_contact"] / n)
    o["invitee_named_before_invite"] = (acc["B_named"] / n, acc["Bc_named"] / n)
    o["new_tie_kept_after_W"] = acc["C_kept"] / (acc["C_n"] or 1); o["initial_tie_kept_after_W"] = acc["C0_kept"] / (acc["C0_n"] or 1)
    o["target_tie_dropped_within_3_after_attack"] = (acc["C2_gone"] / (acc["C2_n"] or 1), int(acc["C2_n"]))
    for k in ("joint", "solo"):
        m = acc[f"D_{k}_n"] or 1
        o[f"{k}_participations"] = int(acc[f"D_{k}_n"]); o[f"{k}_own_invite_to_target"] = acc[f"D_{k}_inv"] / m
        o[f"{k}_named_then_invite"] = acc[f"D_{k}_msg_inv"] / m
    jm = acc["D_joint_n"] or 1
    for ph in ("early", "mid", "late"):
        o[f"{ph}: degree / invites / joint fights / any-invited / chain"] = (round(acc[f"G_{ph}_deg"] / (acc[f"G_{ph}_r"] or 1), 1), int(acc[f"I_{ph}"]), int(acc[f"P_{ph}_fights"]), round(acc[f"P_{ph}_shared"] / (acc[f"P_{ph}_fights"] or 1), 2), round(acc[f"P_{ph}_chain"] / (acc[f"P_{ph}_n"] or 1), 2))
    o["joint_any_attacker_invited_target"] = acc["D_joint_shared"] / jm; o["joint_full_chain"] = acc["D_joint_chain"] / jm
    return o

out = {c: summ(c) for c in ("prod_L3_knife", "prod_L3_scar", "prod_L3_abund", "robust_qwen_L3_knife", "prod_L3_knife_nocomm")}
(HIER / "rewire_sequence.json").write_text(json.dumps(out, indent=1))
for c, o in out.items():
    print("==", c)
    for k, v in o.items():
        print(f"  {k:42}", (str(v) if isinstance(v, tuple) and len(v) == 5 else f"{v[0]:.2f} (chance {v[1]:.2f})" if isinstance(v, tuple) and isinstance(v[1], float) else (f"{v[0]:.2f} (n={v[1]})" if isinstance(v, tuple) else (f"{v:.2f}" if isinstance(v, float) else v))))
