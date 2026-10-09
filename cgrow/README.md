# cgrow/ — the contracted-growth library

The core every experiment script composes. [TECHNICAL.md](../TECHNICAL.md)
§2 walks through each module function by function; this is the map.

| module | what it holds |
|---|---|
| `organ.py` | `Organ`: a spectrally capped prototype trunk behind its contract — anchors, margin, radius, territory, vetoes — with threshold solving, freezing, monotone tightening and the certified radius |
| `proximity.py` | territories: `PCAProx` (the default, and a data generator), `MixProx`, `KMeansProx`, `AEProx` and the shoot-out contenders; `merge_pca_family` for consolidated organs |
| `organism.py` | `Organism`: the first-accept cascade, the growth gate, tightening against incoming data, and the bitwise retention audit |
| `adolescence.py` | consolidation from contract samples (`consolidate_protos`, `consolidate_mlp`, `consolidate_sheet`) and its audit |
| `gates.py` | the acceptance gate: territory check, real-data escrow, contract check |
| `sheet_organ.py` | `SheetOrgan`: organs made of recurrent units placed on a 2-D sheet |
| `curriculum.py` | class-order policies for the curriculum experiments |
| `baselines.py` | the plain MLP, fine-tuning, EWC and replay used as baselines |
| `data.py` | dataset loading, splitting and scaling, and the class-incremental task split |

Importing the package refuses a conda *base* interpreter (see `__init__.py`);
set `CGROW_ALLOW_ANY_PYTHON=1` to override.
