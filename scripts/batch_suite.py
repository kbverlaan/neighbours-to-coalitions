import json, sys, math, glob, os
from collections import defaultdict, Counter
import numpy as np, igraph as ig, leidenalg
# communicatie-familie markers (zelfde scripts/-map)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deontic, enforcement

PRIMARY=['transfer','strengthen','take','harvest','drop','invite','hold']
def cat(a):
    if a is None: return 'hold'
    a=str(a).lower()
    if a.startswith('invest') or a=='transfer': return 'transfer'
    if a.startswith('take') or a=='attack': return 'take'
    if 'arm' in a or 'strengthen' in a: return 'strengthen'
    if a.startswith('harvest'): return 'harvest'
    if a.startswith('drop'): return 'drop'
    if a.startswith('invite'): return 'invite'
    return 'hold'
def effective_action(info):
    """Actie zoals gekozen: productie-logs schrijven de rewire-keuze als
    action='hold' + rewire_intent (engine-neutraal); hier terugvertaald."""
    a = info.get('action')
    ri = info.get('rewire_intent') or {}
    if (a in (None, 'hold', 'do_nothing')) and isinstance(ri, dict):
        if ri.get('drop'): return 'drop'
        if ri.get('invite'): return 'invite'
    return a

def gini(xs):
    xs=sorted(x for x in xs if x is not None); n=len(xs); s=sum(xs)
    if n==0 or s<=0: return 0.0
    return (2*sum((i+1)*x for i,x in enumerate(xs)))/(n*s)-(n+1)/n
def ns(X): Y=X/X.max() if X.max()>0 else X; return (Y+Y.T)/2

def analyze(path):
    lines=[json.loads(l) for l in open(path) if l.strip()]
    if len(lines)<2: return None
    agents=set(); T=defaultdict(float); M=defaultdict(float)
    per=[]; attacks=[]; takes=[]; drops=ninv=0; membNA=0
    for d in lines:
        ag=d.get('agents',{}); agents.update(ag.keys())
        cc=Counter(); Rs={}
        bf=d.get('bilateral_flows') or {}
        for aid,info in ag.items():
            c=cat(effective_action(info)); cc[c]+=1
            R=info.get('resources');  Rs[aid]=R if R is not None else Rs.get(aid)
            tg=info.get('target')
            if c=='take' and tg: attacks.append((d.get('round'),aid,tg)); takes.append((aid,tg))
            if not bf and c=='transfer' and tg: T[(aid,tg)]+=(info.get('breakdown') or {}).get('invest_cost',1.0) or 1.0
            ri=info.get('rewire_intent') or {}
            if ri.get('drop'): drops+=1
            if ri.get('invite'): ninv+=1
        for k,v in bf.items():
            if '→' in k: a,b=k.split('→'); T[(a,b)]+=v
        for m in d.get('messages',[]):
            to=m.get('to'); tos=to if isinstance(to,list) else ([] if to in (None,'all') else [to])
            for t in tos: M[(m.get('from'),t)]+=1
        per.append((d.get('round'),cc,sum(x for x in Rs.values() if x is not None),dict(Rs)))
    nr=len(per)
    # survivors: agent telt als 'overlever' zolang final R >= 10% van diens start-R
    # (lethal pot pluk je kaal i.p.v. naar 0; <10%-van-start = effectief uit het spel)
    initR=per[0][3]; finalR=per[-1][3]
    start_n=sum(1 for v in initR.values() if v is not None) or len(agents)
    surv=sum(1 for a,v0 in initR.items()
             if v0 and finalR.get(a) is not None and finalR[a]>=0.10*v0)
    gone=sum(1 for a in initR if finalR.get(a) is None)  # echt verdwenen uit dict (bankrupt)
    surv_rate=surv/start_n if start_n else 0
    R0=per[0][2]; Rn=per[-1][2]
    pr=((Rn/R0)**(1/(nr-1))-1)*100 if R0>0 and Rn>0 else 0
    finalRs=[x for x in per[-1][3].values() if x is not None]
    g=gini(finalRs); top=max(finalRs)/sum(finalRs)*100 if finalRs and sum(finalRs)>0 else 0
    if pr>0.3: reg='HEG' if top>=15 else 'BLOEI'
    elif pr<-0.3: reg='VEROV' if top>=15 else 'NIVEL'
    else: reg='VLAK+' if top>=15 else 'VLAK'
    # drift
    prev=None; drifts=[]
    for r in per:
        n=sum(r[1].values()) or 1; p=np.array([r[1].get(k,0)/n for k in PRIMARY])
        if prev is not None and r[0] and r[0]>=3: drifts.append(0.5*np.abs(p-prev).sum())
        prev=p
    drift=float(np.mean(drifts)) if drifts else 0
    cons=[]
    for r in per:
        n=sum(r[1].values()) or 1
        cons.append(max(r[1].values())/n if r[1] else 0)
    cons_mean=float(np.mean(cons)); cons_std=float(np.std(cons))
    # mob
    brt=defaultdict(lambda: defaultdict(set))
    for rd,a,t in attacks: brt[rd][t].add(a)
    mob=np.mean([max((len(s) for s in tg.values()),default=0) for tg in brt.values()]) if brt else 0
    # subgroups
    agents=sorted(agents); idx={a:i for i,a in enumerate(agents)}; nn=len(agents)
    Tm=np.zeros((nn,nn)); Mm=np.zeros((nn,nn))
    for (a,b),v in T.items():
        if a in idx and b in idx: Tm[idx[a],idx[b]]+=v
    for (a,b),v in M.items():
        if a in idx and b in idx: Mm[idx[a],idx[b]]+=v
    A=0.5*ns(Tm)+0.5*ns(Mm)
    edges=[(i,j) for i in range(nn) for j in range(i+1,nn) if A[i,j]>0]; w=[A[i,j] for i,j in edges]
    Q=0;K=0;coh=0
    if edges:
        gr=ig.Graph(n=nn,edges=edges); gr.es['weight']=w
        part=leidenalg.find_partition(gr,leidenalg.RBConfigurationVertexPartition,weights='weight',seed=42)
        memb=np.array(part.membership); Q=part.modularity; K=len(part)
        share=np.array([[A[i,memb==c].sum() for c in range(K)] for i in range(nn)])
        coh=sum(1 for i in range(nn) if share[i].sum()>0 and share[i][memb[i]]/share[i].sum()>=0.6)/nn
    # take intra
    membmap={agents[i]:int(memb[i]) for i in range(nn)} if K else {}
    intra=sum(1 for s,t in takes if s in membmap and t in membmap and membmap[s]==membmap[t])
    tt=len(takes)
    # communicatie-familie (norm-inhoud): deontische dichtheid + norm->daad enforcement
    try:
        dd=deontic.analyze(path)
        norm_pub=dd['norm']; norm_priv=dd['priv']['norm']; sanc=dd['sanction']
    except Exception: norm_pub=norm_priv=sanc=0.0
    try:
        ee=enforcement.analyze(path)
        enf_neg=ee['neg_rate']; enf_pos=ee['pos_rate']
    except Exception: enf_neg=enf_pos=0.0
    return dict(reg=reg,pr=pr,top=top,g=g,Q=Q,K=K,coh=coh,mob=mob,drift=drift,tt=tt,
                cmean=cons_mean,cstd=cons_std,surv=surv_rate,gone=gone,
                norm_pub=norm_pub,norm_priv=norm_priv,sanc=sanc,enf_neg=enf_neg,enf_pos=enf_pos,
                verr=100*intra/tt if tt else 0,rw=(drops+ninv)/nr,alive=sum(1 for x in finalRs if x>1))

dirs=sys.argv[1:]
for d in dirs:
    files=sorted(glob.glob(os.path.expanduser(d)+'/*_reasoning_live.jsonl'))
    rows=[]
    for f in files:
        try: r=analyze(f)
        except Exception as e: r=None
        if r: rows.append(r)
    if not rows: continue
    name=os.path.basename(d.rstrip('/'))
    import collections
    regs=collections.Counter(r['reg'] for r in rows)
    mean=lambda k: np.mean([r[k] for r in rows])
    print(f"\n### {name}  (n={len(rows)})")
    print(f"  regimes: {dict(regs)}")
    print(f"  Q {mean('Q'):.2f} | mob {mean('mob'):.1f} | cohesie {100*mean('coh'):.0f}% | drift {mean('drift'):.3f} | groei {mean('pr'):+.1f}%/r | top {mean('top'):.0f}% | gini {mean('g'):.2f} | takes {mean('tt'):.0f} (verraad {mean('verr'):.0f}%) | rewire {mean('rw'):.1f}/r")
    print(f"  survivors {100*mean('surv'):.0f}% (final R >= 10% van start) | bankrupt/verdwenen {mean('gone'):.1f}")
    print(f"  consensus: mean {mean('cmean'):.2f} (eensgezindheid) | std {mean('cstd'):.2f} (golf: hoog = eenheid breekt+herstelt)")
    print(f"  COMM: norm-publiek {mean('norm_pub'):.3f} | norm-privé {mean('norm_priv'):.3f} | handhaving {mean('enf_neg'):.3f} | vervulling {mean('enf_pos'):.3f}")
    # spreiding Q
    Qs=sorted(r['Q'] for r in rows)
    print(f"  Q-spreiding: min {Qs[0]:.2f} med {Qs[len(Qs)//2]:.2f} max {Qs[-1]:.2f}")
