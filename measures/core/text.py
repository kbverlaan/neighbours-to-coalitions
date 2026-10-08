"""Finding a phrasing in generated text, on one unit, against a baseline.

Every language figure in the chapter is this module with a different pattern.
The patterns themselves live in `lexicons.py`; none of them is baked in here,
because the moment a detector owns its own regex two figures start disagreeing
about what counts as an accusation.

Three constraints are enforced here rather than repeated per figure.

**The sentence is the unit.** Not the message, not the reasoning block. The
audit found the old measures each picked their own, and the looser
the unit the higher the rate --- the one with the largest unit returned 100 per
cent in all six cells, which is a fact about the detector. A thinking block runs
to hundreds of words and will contain almost any keyword somewhere.

**Public and private never share a denominator.** Messages are addressed to
others and are the medium in which order is negotiated; thinking and memory are
private. Pooling them answers neither question, so a figure must ask for one.

**Every count carries its chance expectation.** A share means nothing until you
know what the same pattern yields on text with no signal in it. For a pattern
that must co-occur with a name, that baseline is the same corpus with the names
shuffled: sentence length, vocabulary and pattern all preserved, only the link
being claimed destroyed.
"""
from __future__ import annotations

import random
import re
from functools import lru_cache
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_shared"))

from logs import rounds        # noqa: E402
from result import Result      # noqa: E402
import runset                  # noqa: E402

# Split on terminal punctuation, newlines, and the bullet markers the models use
# heavily. Deliberately eager: a smaller unit biases against finding an effect,
# which is the safe direction to be wrong in.
SPLIT = re.compile(r"(?<=[.!?;:])\s+|\n+|(?:^|\s)[-*•]\s+")

AGENTS = ("Bronze Rust Silver Sage Indigo Cyan Violet Plum Pearl Mauve Ash Jade "
          "Onyx Storm Slate Olive Coral Dusk Teal Blue Cobalt Gold Green Maroon "
          "Copper Scarlet Amber Crimson Ivory Red").split()


def sentences(t: str) -> list[str]:
    return [s.strip() for s in SPLIT.split(t or "") if s and s.strip()]


# Splitting is the expensive half of every language measure: one run yields
# about 240,000 sentences of private trace and takes half a second, and five to
# eight measures walk the same text. Cached per (run, stream) the way rounds are
# cached per run, which is where the equivalent fix belonged one layer down.
# Bounded, for the same reason the round cache is. A private-trace stream is
# about 240,000 sentences; holding every run's costs gigabytes and puts a
# whole-section run into swap, where it spends its wall time paging rather than
# counting. Four streams covers any single figure's reach; a figure walks a cell one run at a time.
#
# Eviction changes what is resplit, never what is returned.
_ZINNEN_MAX = 2
_ZINNEN: "OrderedDict[tuple, list]" = OrderedDict()


def _cached(sleutel, bouw):
    hit = _ZINNEN.get(sleutel)
    if hit is None:
        hit = _ZINNEN[sleutel] = list(bouw())
        while len(_ZINNEN) > _ZINNEN_MAX:
            _ZINNEN.popitem(last=False)
    else:
        _ZINNEN.move_to_end(sleutel)
    return hit


def clear_sentences():
    _ZINNEN.clear()


def public(paths):
    """(run, round, speaker, sentence) for every public message."""
    for p in paths:
        def bouw(p=p):
            for e in rounds(p):
                for m in (e.get("messages") or []):
                    for s in sentences(m.get("text") or ""):
                        yield p, e.get("round"), m.get("from"), s
        yield from _cached((p, "public"), bouw)


def private(field: str = "thinking"):
    """Stream factory for a private trace. `field` is 'thinking' or 'memory'.

    They are separate records and are never merged; a figure wanting both must
    ask twice and say so.
    """
    if field not in ("thinking", "memory"):
        raise ValueError("field must be 'thinking' or 'memory'")

    def stream(paths):
        for p in paths:
            def bouw(p=p):
                for e in rounds(p):
                    for nm, a in (e.get("agents") or {}).items():
                        for s in sentences(str(a.get(field) or "")):
                            yield p, e.get("round"), nm, s
            yield from _cached((p, field), bouw)
    stream.__name__ = f"private_{field}"
    return stream


def share(paths, stream, pattern, baseline=None) -> Result:
    """Share of sentences in one stream matching `pattern`, with its denominator."""
    hits = total = 0
    for _, _, _, s in stream(paths):
        total += 1
        if pattern.search(s):
            hits += 1
    # No sentences is not a share of zero. The no-channel cells have no public
    # text at all, so any public-text detector returns nothing there whatever
    # the agents do --- reporting that as 0.0 per cent turns a fact about the
    # design into a finding about behaviour, which is exactly what the sixth of
    # these measures was doing before it was caught.
    return Result(value=round(100 * hits / total, 2) if total else None,
                  n=len(list(paths)), denominator=total, unit="sentences",
                  baseline=baseline,
                  note=f"{hits} of {total}" if total
                       else "no sentences in this stream, so the share is "
                            "undefined rather than zero")


def name_baseline(paths, stream, pattern, target_of, draws: int = 200,
                  seed: int = 20260814) -> float:
    """Chance expectation for a pattern that must co-occur with a NAME.

    Shuffles which agent counts as the target and re-runs the same test on the
    same untouched text. `target_of` maps a run to the name the figure cares
    about. The seed is fixed so the baseline is reproducible.
    """
    rng = random.Random(seed)
    uit = []
    for p in paths:
        zinnen = [s for _, _, _, s in stream([p]) if pattern.search(s)]
        echt = target_of(p)
        if not zinnen or not echt:
            continue
        treffers = 0
        for _ in range(draws):
            nep = rng.choice([n for n in AGENTS if n != echt])
            treffers += sum(1 for s in zinnen if re.search(rf"\b{nep}\b", s))
        uit.append(treffers / draws / len(zinnen) * 100)
    return round(sum(uit) / len(uit), 2) if uit else 0.0


def repetition(paths) -> Result:
    """How much of a cell's public messaging is one exact text repeated.

    Exact matching, whitespace-stripped, is the strictest reading of "the same
    sentence": any looser notion can only raise the figure, so this is a lower
    bound on how repetitive a cell is. The baseline is the uniform case --- if
    every message were distinct, the top text would be one of them.
    """
    from collections import Counter
    texts = [" ".join(s.split()) for p, _, _, s in public(paths)]
    volle = []
    for p in paths:
        for e in rounds(p):
            for m in (e.get("messages") or []):
                t = " ".join((m.get("text") or "").split())
                if t:
                    volle.append(t)
    tally = Counter(volle)
    if not volle:
        return Result(value=0.0, n=len(paths), denominator=0, unit="messages")
    top, top_n = tally.most_common(1)[0]
    return Result(value=round(100 * top_n / len(volle), 2), n=len(paths),
                  denominator=len(volle), unit="messages",
                  baseline=round(100 / len(volle), 4),
                  sensitivity={"distinct_texts": len(tally),
                               "top_text": top[:120], "top_count": top_n,
                               "sentences": len(texts)},
                  note="share of all messages that are the single most frequent exact text")


# --- named agreements ------------------------------------------------------
#
# A capitalised multi-word phrase that at least three different agents use is
# treated as a name they have coined for something they share. The three-user
# threshold is what makes it a shared name rather than one agent's turn of
# phrase, and it is the free parameter of this detector.
#
# Ported unchanged from the hand-validated original so that figures using it
# cannot drift from the ones that did. Its known failure modes are stated with
# every use: it cannot tell an institution from a plan, and a phrase coined once
# and echoed twice counts the same as one used for forty rounds.

_W = r"[A-Z][a-zA-Z-]*[a-zA-Z]"
NGRAM = re.compile(rf"\b({_W}(?:\s+(?:of\s+|the\s+)?{_W}){{1,3}})\b")
ABBREV = re.compile(rf"\b({_W}(?:\s+{_W}){{1,3}})\s*\(([A-Z]{{2,5}}\))")
SENT_START = re.compile(r"(?:^|[.!?;:]\s+|\n\s*|[-*]\s+|\"\s*)$")
EDGE_WORDS = {"the", "a", "an", "to", "in", "of", "for", "on", "at", "and", "if"}
MIN_USERS = 3

# A coined name is a noun phrase. Where agents write in capitals, the
# capital-letter rule catches fragments of ordinary sentences instead: "NO ONE
# ELSE HAS RESOURCES" gives ONE ELSE, "I AM DEPLETED" gives IS DEPLETED, and
# each reaches three speakers because the whole cell is shouting the same
# distress. Six such fragments stand among the coined terms of the six L2 and L3
# cells.
#
# A finite verb, a negation, or a pronoun is what separates a sentence from a
# name, so a candidate containing one is not a name. This costs any arrangement
# actually called "We Will Attack"; none is in the corpus, and the alternative
# is a detector that reports the collective's panic as its vocabulary.
SENTENCE_WORDS = {
    "is", "are", "was", "were", "be", "been", "am", "no", "not", "if", "then",
    "i", "we", "you", "he", "she", "they", "it", "my", "your", "his", "her",
    "their", "this", "that", "these", "those", "has", "have", "had", "do",
    "does", "did", "will", "would", "can", "could", "should", "must", "one",
    "else"}


def _trim(c: str) -> str:
    w = c.split()
    while w and w[0].lower() in EDGE_WORDS:
        w = w[1:]
    while w and w[-1].lower() in EDGE_WORDS:
        w = w[:-1]
    return " ".join(w)


# --- one arrangement, several wordings -------------------------------------
#
# The absorption rule below the detector is a substring test, so it merges
# "Defence Pact" into "Mutual Defence Pact" and nothing else. That leaves one
# arrangement counted several times whenever the wording moves: "Core Local
# NAP", "Core NAP" and "Local NAPs" are one pact; "Master List of NAP", "Master
# List NAP", "Master List of IDs", "ID Master List" and "ID-based Master List"
# are one list; "Death Spiral" and "DEATH SPIRAL" are one phrase the detector
# separated on case alone. Counted as written, a collective that cannot settle
# on a wording reads as one that keeps inventing.
#
# Two phrases are the same arrangement when their content words agree on half or
# more, counting a word once regardless of case, plural, or hyphen, and ignoring
# the articles and prepositions that carry no content. Merging is transitive:
# "ID-based Master List" reaches "Master List of NAP" through "ID Master List",
# and stopping short of that would leave the fragment the merge exists to
# remove.
#
# The threshold is the free parameter. At 0.6 the master-list family splits in
# two; at 0.4 phrases that share one common word ("Safe List", "Master List")
# start collapsing into each other. Every merged group is reported in the
# sensitivity block of the figures that use this, so what it did is readable
# rather than trusted.

_STOP = {"the", "of", "a", "an", "and", "for", "to", "our", "we", "all"}
SAME_TERM = 0.5


def _content(naam: str) -> frozenset:
    w = []
    for x in naam.lower().replace("-", " ").split():
        x = "".join(c for c in x if c.isalnum())
        if not x or x in _STOP:
            continue
        if len(x) > 3 and x.endswith("s"):
            x = x[:-1]
        w.append(x)
    return frozenset(w)


def _variant_groups(namen: list[str], drempel: float = SAME_TERM) -> list[list[str]]:
    """Names grouped by arrangement, transitively, at `drempel` overlap."""
    kern = {n: _content(n) for n in namen}
    ouder = {n: n for n in namen}

    def vind(x):
        while ouder[x] != x:
            ouder[x] = ouder[ouder[x]]
            x = ouder[x]
        return x

    for i, a in enumerate(namen):
        for b in namen[i + 1:]:
            vereniging = kern[a] | kern[b]
            if vereniging and len(kern[a] & kern[b]) / len(vereniging) >= drempel:
                ouder[vind(a)] = vind(b)
    groepen = {}
    for n in namen:
        groepen.setdefault(vind(n), []).append(n)
    return list(groepen.values())


def named_agreements(path, min_users: int = MIN_USERS, with_rounds: bool = False,
                     naming_agents: bool = False, merge_variants: bool = True):
    """Coined names in one run, mapped to how many distinct agents used them.

    With `with_rounds`, each name maps to a dict carrying the speaker count and
    the rounds it was used in, which is what a timeline of names needs: when a
    phrase appears, how long it survives, and how many mouths it reaches. The
    default return is unchanged, so nothing that calls this has to know.

    With `naming_agents`, the filter is inverted and the function returns the
    phrases it would otherwise drop: coined terms that name an agent. Those are
    not institutions --- "TAKE ONYX" is a call to strike, not an arrangement ---
    but they are shared vocabulary reaching three or more speakers, and which
    rungs produce them is a finding of its own.

    A phrase that opens a sentence loses its first word, since capitalisation
    there carries no information. Phrases containing an agent's name are
    dropped: "Bronze and Cyan" is not an institution. Phrases containing a
    finite verb, a negation or a pronoun are dropped as sentences rather than
    names. A longer phrase absorbs a shorter one it contains, so "Mutual Defence
    Pact" and "Defence Pact" count once, and wordings of one arrangement are
    folded together, so "Core NAP" and "Core Local NAP" do too.
    """
    rs = rounds(path)
    from collections import defaultdict
    users, afk = defaultdict(set), {}
    wanneer = defaultdict(set)
    per_ronde = defaultdict(lambda: defaultdict(set))   # term -> round -> speakers
    agents = set(AGENTS)
    klein_agents = {a.lower() for a in agents}
    for e in rs:
        for m in (e.get("messages") or []):
            for vol, a in ABBREV.findall(m.get("text") or ""):
                afk.setdefault(a.rstrip(")"), _trim(vol))
    for e in rs:
        for m in (e.get("messages") or []):
            t, wie = m.get("text") or "", m.get("from")
            gevonden = set()
            for mt in NGRAM.finditer(t):
                c = mt.group(1)
                if SENT_START.search(t[:mt.start()]):
                    c = " ".join(c.split()[1:])
                c = _trim(c)
                # Case-insensitively: the rule drops a phrase naming an agent,
                # and it was comparing exact case, so "Take Ash" went and
                # "TAKE ASH" stayed. Eleven strike calls at L3 --- TAKE ONYX,
                # COPPER NOW, VIOLET NOW --- were counted as coined agreements
                # by that gap alone, and none at L2, which is where the two
                # rungs' naming counts were being compared.
                kleine = {w.lower().strip("-") for w in c.split()}
                noemt_agent = bool(kleine & klein_agents)
                if len(c.split()) < 2 or noemt_agent != naming_agents:
                    continue
                if kleine & SENTENCE_WORDS:      # a sentence, not a name
                    continue
                gevonden.add(c)
            for a, vol in afk.items():
                # The abbreviation route bypassed the agent-name filter, so
                # "Non-Aggression Pact" arrived here whichever way the filter
                # was set and appeared in both lists at once.
                if bool({w.lower() for w in vol.split()} & klein_agents) != naming_agents:
                    continue
                if re.search(rf"\b{re.escape(a)}\b", t):
                    gevonden.add(vol)
            for c in gevonden:
                users[c].add(wie)
                wanneer[c].add(e.get("round"))
                per_ronde[c][e.get("round")].add(wie)
    for c in sorted(users, key=lambda x: -len(x.split())):
        if c not in users:
            continue
        gastheer = next((k for k in users if k != c and f" {c} " in f" {k} "), None)
        if gastheer:
            users[gastheer] |= users[c]
            wanneer[gastheer] |= wanneer[c]
            for r, wie in per_ronde[c].items():
                per_ronde[gastheer][r] |= wie
            del users[c]

    # Wordings of one arrangement folded together; see _variant_groups above.
    # The wording kept is the one most agents used, so the name in a figure is
    # the name the collective settled on rather than the first one tried.
    varianten = {}
    if merge_variants:
        for groep in _variant_groups(sorted(users)):
            if len(groep) < 2:
                continue
            kop = max(groep, key=lambda n: (len(users[n]), -len(n), n))
            for n in groep:
                if n == kop:
                    continue
                users[kop] |= users[n]
                wanneer[kop] |= wanneer[n]
                for r, wie in per_ronde[n].items():
                    per_ronde[kop][r] |= wie
                del users[n]
            varianten[kop] = sorted(groep)

    if with_rounds:
        return {c: {"users": len(u),
                    "rounds": sorted(r for r in wanneer[c] if r is not None),
                    # speakers per round: what a term's rise and fall is made of
                    "speakers_by_round": {r: len(w) for r, w in sorted(per_ronde[c].items())
                                          if r is not None},
                    "variants": varianten.get(c, [c])}
                for c, u in users.items() if len(u) >= min_users}
    return {c: len(u) for c, u in users.items() if len(u) >= min_users}


# --- accusations -----------------------------------------------------------
#
# An accusation names someone and says they did wrong, and the two have to be
# grammatically joined for it to be one. The predecessor of this detector
# accepted any violation word within 120 characters of a name and was right in
# roughly six cases of thirty: "Bronze proposed a pact, and someone has broken
# it" is not an accusation of Bronze. The version here requires the name to
# stand as the subject of the violation verb, or to follow it after "by".
#
# Hand-validated at twenty-four of twenty-four on a stratified sample: four hits
# from one randomly chosen run in each of the six L2 and L3 cells, seed 20260815.
#
# Recall is unknown and is known to undercount in one specific way: a sentence
# naming several offenders ("Teal, Maroon, and Violet betrayed the NAP") returns
# only the first. Counts are therefore lower bounds, which is the opposite
# direction from the loose lexicons in this register and the safer one.

# `attack` is deliberately absent. It is a noun at least as often as a verb in
# this corpus, and "the Slate attack" names the target rather than an offender:
# a hand sample of eight hits with it included was right twice. Every verb kept
# here is unambiguously a verb, which costs recall and buys precision.
VIOLATION = (r"brok\w*|broke|breach\w*|violat\w*|betray\w*|defect\w*|cheat\w*|"
             r"lied|reneg\w*|abandon\w*|double-cross\w*|backstab\w*")


def accusation(sentence: str, names) -> str | None:
    """The name accused in one sentence, or None.

    Two grammatical positions count. A name immediately before the verb, with at
    most an auxiliary or an adverb between them, is its subject. A name after
    "by" within a short window is its agent in the passive. Nothing else is
    accepted, and in particular mere proximity is not.
    """
    for nm in names:
        # "Onyx is broken" makes Onyx the patient, not the offender, so the
        # copula forms that mark a passive are excluded from the subject
        # position while the perfect forms that mark an active are kept.
        subject = re.compile(
            rf"\b{re.escape(nm)}\b\s+(?:has|have|had|just|then|also|"
            rf"clearly|repeatedly|again)?\s*(?:{VIOLATION})", re.I)
        passive = re.compile(
            rf"(?:{VIOLATION})[^.!?]{{0,40}}?\bby\s+{re.escape(nm)}\b", re.I)
        if subject.search(sentence) or passive.search(sentence):
            return nm
    return None


def accusations(paths, stream=None):
    """(run, round, accuser, accused) for every accusation in one stream."""
    stream = stream or public
    for p in paths:
        namen = set()
        for e in rounds(p):
            namen |= set((e.get("agents") or {}).keys())
        for _, r, spreker, s in stream([p]):
            beschuldigd = accusation(s, namen)
            if beschuldigd and beschuldigd != spreker:
                yield p, r, spreker, beschuldigd


# --- order of mental-state attribution -------------------------------------
#
# Order 1 is a model of another agent's state: "Bronze will attack." Order 2 is
# that agent modelling the writer: "Bronze thinks I will attack." Order 3 nests
# once more: "Bronze thinks I believe he will attack."
#
# The unit here is the reasoning block and not the sentence, which departs from
# the rule the rest of this module enforces. It is deliberate and it is the
# measure's own definition: an attribution can span two sentences, and splitting
# would cut the nesting apart. The consequence is that these shares are not
# comparable with any sentence-level figure in the chapter.

_MENTAL = r"think\w*|believ\w*|expect\w*|assum\w*|suspect\w*|know\w*|worr\w*|fear\w*"
_SELF = r"\b(?:I|me|my|us|we|our)\b"
_OTHER = r"(?:they|he|she|him|her|them)"


@lru_cache(maxsize=64)
def _order_patterns(names):
    """The three patterns for one agent list, compiled once.

    Compiling inside the classifier meant three regex builds per reasoning
    block, which over half a million blocks dominated the whole run. The cache
    is keyed on the agent set, of which there is one per run.
    """
    wie = (r"(?:" + _OTHER + "|" + "|".join(re.escape(n) for n in sorted(names)) + ")"
           if names else _OTHER)
    return (
        re.compile(rf"\b{wie}\s+(?:will|would|might|may|is likely to|probably|{_MENTAL})\b", re.I),
        re.compile(rf"\b{wie}\s+(?:{_MENTAL})[^.!?]{{0,40}}?{_SELF}"
                   rf"|{_SELF}[^.!?]{{0,30}}\b(?:from|in)\s+(?:his|her|their)\s+"
                   rf"(?:view|perspective|eyes|position)", re.I),
        re.compile(rf"\b{wie}\s+(?:{_MENTAL})[^.!?]{{0,40}}?{_SELF}"
                   rf"[^.!?]{{0,25}}\b(?:{_MENTAL})\b", re.I))


def attribution_order(block: str, names=None) -> int:
    """The highest order of mental-state attribution a block reaches, 0 to 3.

    `names` is the run's agent list. Passing it matters more than it looks: an
    earlier version accepted any capitalised word in the attributing slot, so
    "Therefore we believe we should hold" scored order 2 --- "Therefore" filled
    the slot and the writer's own belief about itself read as an attribution.
    That inflated the one arm which writes in the plural and opens sentences
    with connectives, from a corrected 30 per cent to 68.

    Without `names` the function falls back to pronouns only, which undercounts
    rather than over. That is the safer direction and it is the default.
    """
    if not block:
        return 0
    o1, o2, o3 = _order_patterns(frozenset(names) if names else None)
    if o3.search(block):
        return 3
    if o2.search(block):
        return 2
    if o1.search(block):
        return 1
    return 0


def plural_first_person(block: str) -> bool:
    """Whether the block's first person is plural.

    One arm writes "we" where the others write "I", and counting that "we" as
    somebody else turns its own reasoning into an attribution. The correction
    matters enough to be measured rather than assumed.
    """
    ev = len(re.findall(r"\b(?:we|us|our)\b", block or "", re.I))
    ik = len(re.findall(r"\b(?:I|me|my)\b", block or "", re.I))
    return ev > ik
