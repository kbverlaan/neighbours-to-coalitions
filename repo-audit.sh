#!/usr/bin/env bash
# Repository hygiene audit for the public snapshot.
#   ./repo-audit.sh              structural + secret checks (fast)
#   ./repo-audit.sh --reproduce  also regenerate the tables and diff them
#                                (needs data/thesis_final/ in place; slow)
set -uo pipefail
cd "$(dirname "$0")"

fail=0; warn=0
PASS(){ printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
FAIL(){ printf '  \033[31mFAIL\033[0m  %s\n' "$1"; fail=$((fail+1)); }
WARN(){ printf '  \033[33mWARN\033[0m  %s\n' "$1"; warn=$((warn+1)); }

# Files under version control (or, if not a git repo yet, the tree minus data/
# and bytecode). Never scans the raw-log tree. Portable to bash 3.2 (macOS).
LIST=$(mktemp)
trap 'rm -f "$LIST"' EXIT
if git rev-parse --git-dir >/dev/null 2>&1; then
  git ls-files > "$LIST"; SRC="git-tracked"
else
  find . -type f -not -path './data/*' -not -path './.git/*' \
    -not -path '*/__pycache__/*' -not -name '*.pyc' | sed 's|^\./||' > "$LIST"
  SRC="working-tree (not a git repo yet)"
fi
echo "Auditing $(wc -l < "$LIST" | tr -d ' ') files (${SRC})"
echo

echo "1. No raw data or secrets in the tree"
if grep -qE '^data/' "$LIST"; then
  FAIL "data/ is tracked (raw logs must stay out)"; else PASS "no data/ files tracked"; fi
if grep -qE '(^|/)\.env$' "$LIST"; then
  FAIL ".env is tracked"; else PASS "no .env tracked"; fi

echo
echo "2. No absolute home paths in tracked files"
# (this script itself contains the search patterns by design; exclude it)
hits=$(tr '\n' '\0' < "$LIST" | xargs -0 grep -nI -e '/Users/' -e '/home/[a-z]' 2>/dev/null | grep -v 'repo-audit\.sh:' | head)
if [ -n "$hits" ]; then FAIL "absolute home paths found:"; echo "$hits" | sed 's/^/        /'
else PASS "no /Users or /home paths"; fi

echo
echo "3. No credential-shaped strings"
hits=$(tr '\n' '\0' < "$LIST" | xargs -0 grep -nIE 'sk-[A-Za-z0-9]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY|Bearer [A-Za-z0-9._-]{20,}' 2>/dev/null | grep -v 'repo-audit\.sh:' | head)
if [ -n "$hits" ]; then FAIL "possible secrets:"; echo "$hits" | sed 's/^/        /'
else PASS "no credential patterns"; fi

echo
echo "4. Every path the README names exists"
for p in src config measures tests handlabels requirements.txt \
         measures/registry.py measures/make_tables.py \
         measures/out/figures.json measures/out/scalars.json measures/out/uncertainty.json \
         measures/tables/tab_grid.tex measures/tables/tab_forms.tex measures/tables/tab_channel.tex; do
  [ -e "$p" ] && PASS "$p" || FAIL "missing: $p"
done

echo
echo "5. Artifact provenance stamps"
stamp=$(grep -oE '"commit"[[:space:]]*:[[:space:]]*"[0-9a-f]{7,40}"' measures/out/figures.json 2>/dev/null | grep -oE '[0-9a-f]{7,40}' | head -1)
if [ -z "$stamp" ]; then WARN "no commit stamp in out/figures.json"
elif git rev-parse --git-dir >/dev/null 2>&1 && git cat-file -e "${stamp}^{commit}" 2>/dev/null; then
  PASS "stamp $stamp resolves in this history"
else
  WARN "stamp $stamp is a pre-release commit, not in this squashed history (expected — see README > Provenance)"
fi

if [ "${1:-}" = "--reproduce" ]; then
  echo
  echo "6. Reproduce the LaTeX tables and diff against committed"
  if [ ! -e data/thesis_final/_INDEX.csv ]; then
    WARN "data/thesis_final/ absent — cannot reproduce (request the logs; see README)"
  else
    tmp=$(mktemp); ( cd measures && python3 make_tables.py --stdout ) > "$tmp" 2>/dev/null
    report=$(python3 - "$tmp" <<'PY'
import sys, pathlib
raw = pathlib.Path(sys.argv[1]).read_text().splitlines()
blocks={}; i=0; cur=None
while i < len(raw):
    if set(raw[i].strip())=={"="} and i+2 < len(raw) and raw[i+1].strip().endswith(".tex") and set(raw[i+2].strip())=={"="}:
        cur=raw[i+1].strip(); blocks[cur]=[]; i+=3; continue
    if cur is not None: blocks[cur].append(raw[i])
    i+=1
def norm(lines):
    out=[l for l in lines if not l.lstrip().startswith("%")]
    while out and not out[0].strip(): out.pop(0)
    while out and not out[-1].strip(): out.pop()
    return out
for name in ("tab_grid.tex","tab_forms.tex","tab_channel.tex"):
    old=norm(pathlib.Path("measures/tables",name).read_text().splitlines())
    print(("OK " if norm(blocks.get(name,[]))==old else "DIFF ")+name)
PY
)
    while read -r verdict name; do
      [ "$verdict" = "OK" ] && PASS "$name reproduces (provenance header aside)" || FAIL "$name differs from committed"
    done <<< "$report"
    rm -f "$tmp"
  fi
fi

echo
if [ "$fail" -gt 0 ]; then echo "AUDIT FAILED — $fail failure(s), $warn warning(s)"; exit 1
else echo "AUDIT OK — 0 failures, $warn warning(s)"; fi
