"""Results manifest: make every number traceable to a file with a hash.

Several result files were nearly clobbered by reruns during this project, and
one battery was found running twice under two interpreters into the same
output. This writes results/MANIFEST.json -- sha256, size and mtime for every
results/*.json, *.npz and *.pkl -- so a paper number can be traced to the exact
artefact that produced it. claims.py embeds the same hashes for the JSON files
it reads under claims["_provenance"].

Run:  python manifest.py
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import time

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build() -> dict:
    files = sorted(glob.glob(os.path.join(RES, "*.json"))
                   + glob.glob(os.path.join(RES, "*.npz"))
                   + glob.glob(os.path.join(RES, "*.pkl")))
    out = {}
    for p in files:
        name = os.path.basename(p)
        if name == "MANIFEST.json":
            continue
        st = os.stat(p)
        out[name] = {"sha256": sha256(p), "bytes": st.st_size,
                     "mtime": time.strftime("%Y-%m-%d %H:%M:%S",
                                            time.localtime(st.st_mtime))}
    return out


def main():
    m = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "files": build()}
    with open(os.path.join(RES, "MANIFEST.json"), "w") as f:
        json.dump(m, f, indent=1)
    print(f"manifest: {len(m['files'])} files -> results/MANIFEST.json")


if __name__ == "__main__":
    main()
