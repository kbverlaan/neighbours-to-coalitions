import os
import sys

# Make `src/` importable as top-level packages (e.g. `agents.memory`),
# matching how the runner sets up sys.path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
