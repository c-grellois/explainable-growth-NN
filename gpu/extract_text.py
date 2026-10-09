"""Frozen sentence-encoder features for TEXT class-incremental streams (B-text).

The pipeline downstream is unchanged: the cache has the same fields as
extract_ptm.py (Xtr/ytr/Xte/yte, n_classes, inc, dataset, backbone, classes),
so every configuration (contracted prototypes, contracted head, certified
lifecycle) runs on it with the ptm_ prefix. Nothing here is tuned: one encoder,
fixed in advance, mean-pooled, no fine-tuning.

  banking77   PolyAI Banking77 intent classification, 77 classes,
              10,003 train / 3,080 test           -> inc 7  (11 tasks)
  clinc150    CLINC150 ("clinc_oos", config "plus"), 150 in-scope intents,
              15,000 train / 4,500 test (the 250 + 100 + 1,000 out-of-scope
              queries are written to a separate OOD cache)  -> inc 15 (10 tasks)

Encoder (default): sentence-transformers/all-mpnet-base-v2 (768-d), loaded
with transformers, mean pooling over the attention mask, fp16 on CUDA.

Needs: pip install datasets transformers  (see setup_env_text.ps1)

Run (gpu/ folder):
  .\\venv\\Scripts\\python.exe extract_text.py --dataset banking77 --out out\\ptm_banking77_mpnet.npz
  .\\venv\\Scripts\\python.exe extract_text.py --dataset clinc150 --out out\\ptm_clinc150_mpnet.npz --ood-out out\\ood_clinc_oos_mpnet.npz
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

ENCODERS = {"mpnet": "sentence-transformers/all-mpnet-base-v2",
            "minilm": "sentence-transformers/all-MiniLM-L12-v2",
            "e5": "intfloat/e5-base-v2"}


def load_dataset_split(name):
    from datasets import load_dataset
    if name == "banking77":
        # 2026-09-14 GPU-box fix: datasets >= 3 refuses script-based Hub repos
        # ("Dataset scripts are no longer supported") and PolyAI/banking77 has no
        # parquet conversion. Read the two CSVs that banking77.py itself downloads
        # and map categories with the script's ClassLabel names (dataset_infos.json),
        # so texts, splits and integer labels are exactly what load_dataset gave.
        import json
        import pandas as pd
        from huggingface_hub import hf_hub_download
        base = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/"
        info = json.load(open(hf_hub_download("PolyAI/banking77", "dataset_infos.json",
                                              repo_type="dataset"), encoding="utf-8"))
        names = list(info.values())[0]["features"]["label"]["names"]
        idx = {n: i for i, n in enumerate(names)}
        tr_df = pd.read_csv(base + "train.csv")
        te_df = pd.read_csv(base + "test.csv")
        tr = (tr_df["text"].tolist(), np.asarray([idx[c] for c in tr_df["category"]]))
        te = (te_df["text"].tolist(), np.asarray([idx[c] for c in te_df["category"]]))
        return tr, te, names, None, 7
    if name == "clinc150":
        ds = load_dataset("clinc/clinc_oos", "plus")
        names = ds["train"].features["intent"].names
        oos = names.index("oos")
        keep = [n for n in names if n != "oos"]
        remap = {names.index(n): i for i, n in enumerate(keep)}

        def split(part):
            txt = np.asarray(ds[part]["text"], dtype=object)
            lab = np.asarray(ds[part]["intent"])
            m = lab != oos
            return list(txt[m]), np.asarray([remap[int(l)] for l in lab[m]]), list(txt[~m])
        tr_t, tr_y, tr_oos = split("train")
        te_t, te_y, te_oos = split("test")
        va_t, va_y, va_oos = split("validation")
        # the pipeline draws its own validation rows from train: fold HF's
        # validation split into train so nothing is wasted
        return (tr_t + va_t, np.concatenate([tr_y, va_y])), (te_t, te_y), keep, tr_oos + va_oos + te_oos, 15
    raise SystemExit(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["banking77", "clinc150"])
    ap.add_argument("--encoder", default="mpnet", help="key in ENCODERS or a HF model id")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ood-out", default=None, help="clinc150 only: where the out-of-scope queries go")
    ap.add_argument("--inc", type=int, default=None, help="override classes per task")
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()
    if os.path.exists(args.out):
        print(f"{args.out} exists -- skipping"); return
    from transformers import AutoModel, AutoTokenizer
    model_id = ENCODERS.get(args.encoder, args.encoder)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    enc = AutoModel.from_pretrained(model_id).eval().to(device)
    if device == "cuda":
        enc.half()
    prefix = "query: " if "e5" in model_id else ""
    print("encoder:", model_id, "| device:", device)

    @torch.no_grad()
    def embed(texts):
        out = []
        t0 = time.time()
        for i in range(0, len(texts), args.batch):
            b = tok([prefix + t for t in texts[i:i + args.batch]], padding=True, truncation=True,
                    max_length=128, return_tensors="pt").to(device)
            h = enc(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
            out.append(pooled.float().cpu().numpy().astype(np.float16))
            if (i // args.batch) % 20 == 0:
                print(f"  {min(i + args.batch, len(texts))}/{len(texts)} ({(i + args.batch) / max(time.time() - t0, 1e-6):.0f} txt/s)", flush=True)
        return np.concatenate(out)

    (tr_t, tr_y), (te_t, te_y), names, oos, inc_default = load_dataset_split(args.dataset)
    inc = args.inc or inc_default
    n_cls = len(names)
    print(f"{args.dataset}: {n_cls} classes, {len(tr_t)} train, {len(te_t)} test, inc {inc} -> {-(-n_cls // inc)} tasks")
    Xtr, Xte = embed(tr_t), embed(te_t)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, Xtr=Xtr, ytr=tr_y.astype(np.int64), Xte=Xte, yte=te_y.astype(np.int64),
                        n_classes=np.int64(n_cls), inc=np.int64(inc), dataset=np.str_(args.dataset),
                        backbone=np.str_(model_id), classes=np.array(names))
    print(f"wrote {args.out}  train {Xtr.shape} test {Xte.shape}")
    if oos and args.ood_out:
        Xo = embed(oos)
        np.savez_compressed(args.ood_out, Xte=Xo, yte=np.full(len(Xo), -1, dtype=np.int64), ood=np.bool_(True),
                            dataset=np.str_("clinc_oos"), backbone=np.str_(model_id))
        print(f"wrote {args.ood_out}  {Xo.shape} out-of-scope queries")


if __name__ == "__main__":
    main()
