"""One register of lexicons for the whole language family.

Every phrasing test in this thesis lives here, in one place, so that two measures
cannot silently disagree about what counts as an accusation or as attack talk.
Before this register each measure carried its own regex and they had drifted.

Each entry states what it is meant to catch and — where it exists — how it was
validated. That second field is the one that matters when reading a result: an
unvalidated lexicon gives an upper bound, not a count.

Validation status is recorded honestly, including the two that fail:

    NAP           unvalidated, but the target phrase is near-formulaic
    MORAL         hand-checked, 20/20 on a sample (welfare-rule work, Aug)
    WELFARE       hand-checked, precision 82.5%
    ATTACK_TALK   UNVALIDATED and broad — catches "let's not attack"
    CEILING       UNVALIDATED — a number near a negation is not a threshold
    SECOND_ORDER  unvalidated
    ROTA          unvalidated
"""
from __future__ import annotations

import re

# Explicit non-aggression language. Public messages only; a pact is spoken.
NAP = re.compile(
    r"non[- ]?aggression|no[nt]?[- ]attack|not attack|won'?t attack|won'?t take from"
    r"|truce|cease[- ]?fire|refrain from (?:attack|violence|conflict)"
    r"|avoid (?:conflict|attacking|violence)|no violence|peace pact|mutual peace"
    r"|agree not to|let'?s not attack|avoid unnecessary conflict", re.I)

# Moral charge: language that frames another agent as having done wrong.
# Hand-checked at 20/20 on a sample.
MORAL = re.compile(
    r"\b(broke|breaking|violat\w*|betray\w*|breach\w*|traitor|defect\w*|cheat\w*|"
    r"liar|lied|lying|dishonest|greedy|selfish|unfair|hoard\w*|freerid\w*|"
    r"free[- ]rid\w*|punish\w*|deserve\w*)\b", re.I)

# A distributive rule with a floor under the poorest. Precision 82.5%.
WELFARE = re.compile(
    r"\b(?:everyone|anyone|all of us|those|whoever|agents?)\b[^.!?]{0,60}"
    r"\b(?:below|under|less than|beneath)\b[^.!?]{0,30}\d"
    r"|\b(?:floor|minimum|safety net|support fund|welfare|guarantee)\b", re.I)

# Talk of attacking. BROAD: it matches the verb regardless of polarity, so
# "let's not attack" counts. Usable for comparing cells, not as a level.
ATTACK_TALK = re.compile(
    r"\b(attack|strike|hit|target|take from|raid|eliminate|remove|take down|"
    r"go after|coordinate against|focus on)\b", re.I)

# A stated attack threshold. BROAD: any number near a negation matches, so
# "I have 45 resources and will not attack" counts as a ceiling.
CEILING = re.compile(
    r"(?:not?\s+attack|don'?t\s+attack|spare|leave\s+alone|immune|protect|below|"
    r"under|above|exceed|threshold|cap|ceiling|limit)[^.!?]{0,80}?(\d{2,4}(?:\.\d)?)"
    r"|(\d{2,4}(?:\.\d)?)[^.!?]{0,80}?(?:or\s+below|or\s+less|and\s+below|"
    r"is\s+safe|are\s+safe|untouchable)", re.I)

# Modelling another agent's mind rather than their balance.
SECOND_ORDER = re.compile(
    r"\b(?:they|he|she|\w+)\s+(?:will|would|might|may|is likely to|probably)\s+"
    r"(?:think|believe|expect|assume|attack|retaliate|respond|reciprocate)"
    r"|\b(?:thinks|believes|expects|assumes)\s+(?:that|I|we)\b"
    r"|\bfrom (?:his|her|their) (?:point of view|perspective)\b", re.I)

# A turn-taking schedule for the shared stock.
ROTA = re.compile(
    r"\b(rotation|rota|take turns|taking turns|turn to harvest|schedule|"
    r"group [abc]\b|round[- ]robin|alternate|every third round)\b", re.I)

# Reciprocal transfer: the arrangement that keeps the L4 economy above decay.
RECIPROCAL = re.compile(
    r"(?:mutual|reciproc\w*|both|each other|swap|loop|exchange)[^.!?]{0,60}transfer"
    r"|transfer[^.!?]{0,60}(?:mutual|reciproc\w*|back|each other|to me|swap|loop)"
    r"|i'?ll transfer to you|you transfer to me", re.I)

VALIDATED = {"MORAL": "20/20 on a hand-coded sample", "WELFARE": "precision 82.5%"}
UNVALIDATED = ("NAP", "ATTACK_TALK", "CEILING", "SECOND_ORDER", "ROTA", "RECIPROCAL")
