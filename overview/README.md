# overview/ — the three-page summary

[`contracted_growth_overview.pdf`](contracted_growth_overview.pdf) states what
the system guarantees and how it scores against published methods, in three
pages and without the mechanisms, for readers who want the results first.

Both files here are generated: `python gen_overview.py` from the project root
reads `results/claims.json`, writes the `.tex`, and builds the PDF with one
pdflatex pass if pdflatex is installed. Edit `gen_overview.py`, not the files
in this folder. Its published anchors are the verified values from
[ANCHORS_PTM.md](../ANCHORS_PTM.md), each with its source table.
