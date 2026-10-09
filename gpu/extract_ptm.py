r"""GPU feature extractor for the standard pretrained-model class-IL suite.

Same contract as extract_gpu.py (torch + torchvision + timm + numpy only,
fp16 on CUDA, resumable) but for image-folder datasets laid out as

    <root>/train/<class_name>/*.jpg
    <root>/test/<class_name>/*.jpg

which is the layout distributed with the RevisitingCIL / LAMDA-PILOT code
(ImageNet-R, ImageNet-A, CUB-200, ObjectNet-200, OmniBenchmark, VTAB).
Class ids follow torchvision.datasets.ImageFolder (sorted folder names), and
the test split must use the same folder names as train.

The cache stores, besides Xtr/ytr/Xte/yte, the protocol metadata the CPU side
reads: n_classes, inc (classes per task, B0 protocol), dataset, backbone.

Run (from the gpu/ folder, CUDA venv):
  .\venv\Scripts\python.exe extract_ptm.py --root data\imagenet-r --dataset imr --inc 20 ^
      --backbone vit_base_patch16_224.augreg_in21k --out out\ptm_imr_vitb.npz
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import timm
import torch
import torchvision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="folder holding train/ and test/ ImageFolder trees")
    ap.add_argument("--dataset", required=True,
                    help="short name stored in the cache (imr, ima, cub, obj, omni, vtab)")
    ap.add_argument("--inc", type=int, required=True,
                    help="classes per task in the B0 protocol (20 for the 200-class sets, 30 omni, 10 vtab)")
    ap.add_argument("--backbone", default="vit_base_patch16_224.augreg_in21k")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-class", type=int, default=None,
                    help="optional cap on train images per class (stratified, seed 0)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    if os.path.exists(args.out):
        print(f"{args.out} exists -- skipping")
        return
    for split in ("train", "test"):
        if not os.path.isdir(os.path.join(args.root, split)):
            raise SystemExit(f"missing {os.path.join(args.root, split)} -- expected "
                             f"<root>/train/<class>/... and <root>/test/<class>/...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = timm.create_model(args.backbone, pretrained=True, num_classes=0)
    net.eval().to(device)
    if device == "cuda":
        net.half()
    cfg = timm.data.resolve_model_data_config(net)
    tf = timm.data.create_transform(**cfg, is_training=False)
    print("backbone:", args.backbone, "| device:", device, "| transform:", tf)

    tr = torchvision.datasets.ImageFolder(os.path.join(args.root, "train"), transform=tf)
    te = torchvision.datasets.ImageFolder(os.path.join(args.root, "test"), transform=tf)
    if tr.classes != te.classes:
        missing = sorted(set(tr.classes) ^ set(te.classes))
        raise SystemExit(f"train/test class folders differ ({len(missing)} names): {missing[:10]}")
    n_classes = len(tr.classes)
    if n_classes % args.inc != 0:
        print(f"WARNING: {n_classes} classes not divisible by inc={args.inc}; "
              f"the CPU side will drop the remainder into a final smaller task")
    print(f"{args.dataset}: {n_classes} classes, {len(tr)} train, {len(te)} test, inc {args.inc}")

    targets = np.array(tr.targets)
    if args.per_class is not None:
        rng = np.random.default_rng(0)
        keep = np.concatenate([
            rng.choice(np.flatnonzero(targets == c),
                       min(args.per_class, int((targets == c).sum())), replace=False)
            for c in range(n_classes)])
        tr = torch.utils.data.Subset(tr, sorted(keep.tolist()))
        print(f"train capped to {len(tr)} images ({args.per_class}/class)")

    @torch.no_grad()
    def extract(ds, name):
        loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)
        feats, labels = [], []
        t0 = time.time()
        for i, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            if device == "cuda":
                x = x.half()
            feats.append(net(x).float().cpu().numpy().astype(np.float16))
            labels.append(np.asarray(y).astype(np.int64))
            if i % 20 == 0:
                done = min((i + 1) * args.batch, len(ds))
                print(f"  {name}: {done}/{len(ds)} ({done / max(time.time() - t0, 1e-6):.0f} img/s)",
                      flush=True)
        return np.concatenate(feats), np.concatenate(labels)

    Xtr, ytr = extract(tr, "train")
    Xte, yte = extract(te, "test")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte,
                        n_classes=np.int64(n_classes), inc=np.int64(args.inc),
                        dataset=np.str_(args.dataset), backbone=np.str_(args.backbone),
                        classes=np.array(tr.classes))
    print(f"wrote {args.out}  train {Xtr.shape}  test {Xte.shape}  "
          f"n_classes {n_classes} inc {args.inc}")


if __name__ == "__main__":
    main()
