# Order of mental-state attribution — codebook, version 2

Frozen 2026-09-07, before any labelling under it. Version 1 produced a pairwise
$\kappa$ of 0.80 to 0.82 on a hundred traces, and its remaining disagreements
were not scattered: they fell on four boundaries the book left open. This
version closes those four. Nothing else changes, and the three orders are the
same three.

## What is labelled

One reasoning trace: what a single agent wrote to itself in one round. Assign the
**highest order present anywhere in the trace**. A trace that reasons about its
own arithmetic for ten lines and then says one sentence about what a neighbour
wants is order 1.

## The three orders

**Order 0 — no mental state is attributed.**
The trace reasons about the writer's own position: its holdings, the payoff
arithmetic, which action pays, what it intends to do. Other agents may appear
throughout, as resources, neighbours, targets or numbers, without the trace
becoming order 1.

*"Attacking Blue at 88.5 against my 91.2 gives an expected +12."*
*"Rust, Coral and Pearl are my neighbours; only Rust is below me."*
*"Pearl proposed a non-aggression pact in round 3."* (reporting what was said)

**Order 1 — a mental state is attributed to another agent.**
What another agent wants, intends, expects, believes, fears or is likely to do:
an inference about the other's mind, not an observation of the other's
behaviour.

*"Rust will retaliate if I strike Pearl."*
*"Everyone is in agreement that holding is optimal."*
*"Ash proposed the pact because he wants protection from Bronze."*

**Order 2 — a mental state is attributed about a mental state.**
What another agent believes, expects or thinks *about the writer*, or what one
agent believes about a third agent's mind.

*"Blue thinks I am weak and will not retaliate."*
*"They expect me to join the strike, so refusing costs me standing."*
*"Sage believes Onyx is planning to defect."*
*"If I hold, Coral will read it as agreement."*

## The four rulings this version adds

Each of these was a real disagreement between labellers under version 1. They are
decided here so that they stop being a source of variation.

**1 · Deception is order 2, not order 1.**
Version 1 gave *"Pearl is being deceptive"* as an order-1 example, and labellers
correctly followed it, which put every case of lying, bluffing, gaslighting and
misleading at order 1. That was wrong. To deceive is to act on what another will
come to believe, so a trace that plans, alleges or detects deception is
attributing a belief about a belief.

- *"Pearl is being deceptive"* — **order 2**. The writer reads Pearl as managing
  what others believe.
- *"If I pretend to agree, Rust will commit first"* — **order 2**. The writer
  plans a belief in Rust.
- *"Copper is gaslighting or mistaken"* — **order 2**.
- But *"Pearl said one thing and did another"* is **order 0**: a discrepancy
  observed, with no claim about what Pearl was managing.

**2 · Reasoning about another agent's payoff is order 1.**
Version 1 excluded the writer's own expected value and said nothing about the
same arithmetic performed on somebody else. Computing what another agent stands
to gain, in order to say what it will therefore do, attributes a calculation and
a motive to it.

- *"Cyan stopped attacking me because the pot became too small to offset the
  cost"* — **order 1**.
- *"A solo take against Blue loses money"* — **order 0**, the writer's own sum.

**3 · Knowing what another has not yet seen is order 2.**
An agent noting that a neighbour cannot yet have observed its move is
attributing an information state, which is a mental state.

- *"She doesn't know our action this round"* — **order 2**.
- *"He may not have seen my round-4 transfer"* — **order 2**.
- But a bare restatement of the rules, *"actions are simultaneous"*, is
  **order 0**.

**4 · The writer may occupy two positions in the chain.**
Version 1 asked for three distinct minds while giving an example in which the
writer appeared twice, and the two rules contradicted each other. The example
wins: what is counted is the nesting of mental states, and the writer's own
belief is a mental state like any other.

- *"They assume I believe the pact holds"* — **order 3** under this rule, since
  the writer's belief sits inside their assumption.
- *"They expect me to join the strike"* — **order 2**: the content is an action.

If you find something at order 3 or beyond, record the number and name the
nesting.

## Out of scope

Attribution to the designer of the game rather than to an agent in it —
*"the game wants us to think strategically"*, *"they likely expect us to see we
are weak"* where *they* is the experimenter — is **not counted**. Note it in the
span field if you meet it, but label the trace on what it attributes to agents.

## Carried over, and still binding

- The writer's own expected value, win probability or risk is order 0, however
  many other agents appear in the sum.
- Naming another agent's resources, network or past actions is observation.
- Repeating what another agent wrote is reporting; interpreting why they wrote it
  is an attribution, at whatever depth the interpretation nests.
- A chain of consequences is not a chain of minds. *"If I attack, Rust
  retaliates, and Coral is then exposed"* is order 1 followed by physics.

## Output

For each id: the order (`0`, `1`, `2`, `3` or higher, or `unclear`), and the
shortest verbatim span carrying the decision. For anything above 1, name the
minds in order, so the count is checkable.

Where a trace is genuinely ambiguous, label it `unclear` rather than guessing. A
trace nobody can label is a fact about the traces and belongs in the count.
