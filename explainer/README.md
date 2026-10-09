# explainer/ — the anatomy of one organism

[`anatomy.html`](anatomy.html) is an interactive walk through the life of a
grown organism on split-digits, split-MNIST and CIFAR-100 (evaluation seed 2):
organs being born on what the body declines, vetoes accumulating as new classes
arrive, the gated consolidation and arbiter, and what each step did to
accuracy. It is a single self-contained page; download it and open it in a
browser (it loads its fonts from Google Fonts and needs nothing else).

It is generated: `python trace_anatomy.py` from the project root reads the
result files, writes `anatomy_trace.json`, and inlines it into
`anatomy_src.html` to produce `anatomy.html`. Edit `anatomy_src.html` or the
script, not the built page. Rebuilding needs the CIFAR-100 class names from
torchvision's download in `data/`, and the per-organ sizes on CIFAR-100 come
from the cached organism pickle, which is not tracked; without it the trace is
built without those sizes.
