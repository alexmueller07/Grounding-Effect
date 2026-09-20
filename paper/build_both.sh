#!/bin/bash
# Build both submission PDFs from one source, then verify anonymity.
#
# The ONLY difference between them is the \iclrfinalcopy line, which must sit
# immediately after \usepackage{iclr2027_conference}. Placed later, the
# \ificlrfinal test has already run and the build silently emits the code URL
# WITHOUT the author names -- a failure that looks fine until someone checks.
set -euo pipefail

BD="$(cd "$(dirname "$0")" && pwd)"
OUT="/private/tmp/claude-501/-Users-alexandermueller/5c34c5c5-95b9-418b-b19c-55d56bbe516f/scratchpad/iclrbuild2/out"
mkdir -p "$OUT"
cd "$BD"

echo "== regenerating figures =="
python3 make_figs.py

echo
echo "== 1/2  ANONYMOUS (OpenReview) =="
# default state of paper.tex has \iclrfinalcopy commented out
grep -q '^%\\iclrfinalcopy' paper.tex || { echo "FATAL: paper.tex is not in anonymous state"; exit 1; }
tectonic -X compile paper.tex --outdir "$BD" >/dev/null 2>&1
cp paper.pdf "$OUT/ICLR2027_submission_ANONYMOUS.pdf"

echo "== 2/2  NAMED (arXiv / advisor) =="
cp paper.tex paper_named.tex
# uncomment in place, immediately after the package line
perl -0pi -e 's/^%\\iclrfinalcopy/\\iclrfinalcopy/m' paper_named.tex
grep -q '^\\iclrfinalcopy' paper_named.tex || { echo "FATAL: \\iclrfinalcopy not enabled"; exit 1; }
# \iclrfinalcopy turns the running head into "Published as a conference paper at ICLR 2027".
# The paper is not published, so the arXiv build must not say so.
perl -0pi -e 's/\\maketitle/\\maketitle\n\\lhead{Preprint.}/' paper_named.tex
grep -q 'lhead{Preprint.}' paper_named.tex || { echo "FATAL: preprint header not injected"; exit 1; }

tectonic -X compile paper_named.tex --outdir "$BD" >/dev/null 2>&1
cp paper_named.pdf "$OUT/arXiv_NAMED.pdf"

echo
echo "===================== VERIFICATION ====================="

echo "-- page counts --"
for f in "$OUT"/*.pdf; do printf "  %-42s %s\n" "$(basename "$f")" "$(pdfinfo "$f" | awk '/^Pages/{print $2}')"; done

echo
echo "-- ANONYMOUS must leak nothing (expect 0) --"
HITS=$(pdftotext "$OUT/ICLR2027_submission_ANONYMOUS.pdf" - | grep -icE "mueller|park|wisconsin|korea|github|/Users/|alexmueller" || true)
echo "  identifying-marker hits: $HITS"
if [ "$HITS" != "0" ]; then
  echo "  *** FAIL -- would be desk-rejected. Offending lines:"
  pdftotext "$OUT/ICLR2027_submission_ANONYMOUS.pdf" - | grep -inE "mueller|park|wisconsin|korea|github|/Users/|alexmueller" | head
  exit 1
fi

echo
echo "-- NAMED must carry both authors and the code URL (expect >=1 each) --"
for pat in "Mueller" "Park" "Wisconsin" "Korea" "github.com/alexmueller07"; do
  n=$(pdftotext "$OUT/arXiv_NAMED.pdf" - | grep -ic "$pat" || true)
  printf "  %-34s %s\n" "$pat" "$n"
  [ "$n" = "0" ] && { echo "  *** FAIL: '$pat' missing from the named build"; exit 1; }
done

echo
echo "-- NAMED must NOT carry the review stamp (expect 0) --"
echo "  'Under review' hits: $(pdftotext "$OUT/arXiv_NAMED.pdf" - | grep -ic 'Under review' || true)"

echo
echo "-- NEITHER build may claim acceptance (expect 0 each) --"
for f in ICLR2027_submission_ANONYMOUS arXiv_NAMED; do
  n=$(pdftotext "$OUT/$f.pdf" - | grep -ic 'Published as a conference paper' || true)
  echo "  $f: $n"
  [ "$n" = "0" ] || { echo "  *** FAIL: $f claims it is published at ICLR 2027"; exit 1; }
done

echo
echo "-- main text must end by page 9 (References may not start later than p10) --"
python3 - "$OUT/ICLR2027_submission_ANONYMOUS.pdf" <<'PYGATE'
import subprocess,sys,re
pdf=sys.argv[1]
def lines(n):
    t=subprocess.run(["pdftotext","-f",str(n),"-l",str(n),pdf,"-"],capture_output=True,text=True).stdout
    out=[l.strip() for l in t.splitlines()]
    return [l for l in out if l and not l.startswith("Under review") and not re.fullmatch(r"\d+",l)]
# The main text ends where the first end statement begins. The condition is: that heading sits on
# page <= 9, or it is the very first line of page 10 (body filled page 9 exactly).
for p in range(1,13):
    L=lines(p)
    if "AI USE STATEMENT" in L:
        k=L.index("AI USE STATEMENT")
        if p<=9 or (p==10 and k==0):
            print("  main text ends on page %d (first end statement on p%d, line %d)"%(p if k>0 else p-1,p,k+1)); sys.exit(0)
        print("  *** FAIL: main text runs to page %d"%p); sys.exit(1)
print("  *** FAIL: AI USE STATEMENT heading not found"); sys.exit(1)
PYGATE
nref=$(pdftotext "$OUT/ICLR2027_submission_ANONYMOUS.pdf" - 2>/dev/null | grep -c '??' || true)
echo "  unresolved cross-references: $nref"
[ "$nref" = "0" ] || { echo "  *** FAIL: unresolved cross-references"; exit 1; }

echo
echo "All checks passed. Output in $OUT"
