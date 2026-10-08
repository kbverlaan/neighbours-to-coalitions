"""Primitives shared by every reported figure.

The measures this replaces each carried their own round loop, their own idea of
a denominator and their own copy of the Gini. That is where two numbers in one
chapter stop agreeing without anyone noticing. Everything a figure needs to
count, group or test lives here, parameterised, so a new figure is an argument
list rather than a new loop.

Seven rules hold across the package and are checked by `tests/test_standards.py`:

  1. One run set. No module chooses its own files; they all go through `runset`.
  2. Every result carries its n and the denominator it was divided by.
  3. Every count carries what chance yields on the same corpus.
  4. An unreadable source is an error, never an empty one.
  5. Skipped runs are named in the output, never dropped in silence.
  6. Running twice on the same input gives the same answer.
  7. Where a definition has a free parameter, the alternative is computed too.
"""
