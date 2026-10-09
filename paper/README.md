# paper/ — *Forgetting Is a Certificate*

The draft paper, *Forgetting Is a Certificate: A Verified Developmental
Lifecycle for Continual Learning*, draft v3.6 (2026-09-18), anonymised for
review and unpublished. [`main.pdf`](main.pdf) is the built copy.

Every quantitative claim in it is regenerated from per-seed records by
`claims.py` into `results/claims.json`; its appendix maps each claim to that
file. Results added to the record after v3.6, such as the size and pruning
battery, are in [REPORT.md](../REPORT.md) and not yet in the paper.

To rebuild, from this folder:

    python ../make_figures.py     # fig_lifecycle.pdf and fig_laws.pdf, from results/claims.json
    pdflatex main && bibtex main && pdflatex main && pdflatex main

`neurips_2025.sty` is a stand-in that reproduces the official NeurIPS layout
constants, because the official style file is distributed by the conference.
Replace it with the official file before any submission; the source does not
change.
