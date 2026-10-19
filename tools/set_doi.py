"""Replace the Zenodo DOI everywhere it is cited, after a NEW draft has been created.

    python tools/set_doi.py 10.5281/zenodo.1234567 --dry-run   # list what would change
    python tools/set_doi.py 10.5281/zenodo.1234567             # apply

Only the DOI string 10.5281/zenodo.<digits> is replaced, in the files listed below. Nothing else is touched.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ["CITATION.cff", "paper/latex/refs.bib", "paper/latex/main.tex", "submission/InformationFusion/refs.bib",
         "submission/InformationFusion/main.tex", "submission/InformationFusion/title_page.tex", "submission/IEEE_TIV/refs.bib"]
PAT = re.compile(r"10\.5281/zenodo\.\d+")


def main():
    new = sys.argv[1]
    if not re.fullmatch(r"10\.5281/zenodo\.\d+", new):
        sys.exit("expected a DOI like 10.5281/zenodo.1234567")
    dry = "--dry-run" in sys.argv
    total = 0
    for rel in FILES:
        p = ROOT / rel
        raw = p.read_bytes().decode("utf-8")
        found = PAT.findall(raw)
        if not found:
            print(f"{rel}: no DOI found (check)")
            continue
        total += len(found)
        print(f"{rel}: {sorted(set(found))} -> {new} ({len(found)} place(s))")
        if not dry:
            p.write_bytes(PAT.sub(new, raw).encode("utf-8"))
    print(("DRY RUN: " if dry else "UPDATED: ") + f"{total} occurrence(s)")


if __name__ == "__main__":
    main()
