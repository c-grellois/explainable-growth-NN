"""GPU feature extractor for OUT-OF-DISTRIBUTION test sets (B-ood).

Same backbone / transform / fp16 contract as extract_gpu.py, but the output is
a single split (`Xte`, `yte`) of a set the organism was never trained on, so
the CPU side can measure how the certificate declines it. torchvision
downloads everything on first use into --data.

  svhn     SVHN test split (26,032 digits)              far-OOD for CIFAR-100
  dtd      Describable Textures, all 5,640 images       far-OOD (textures)
  places   Places365 val (small, 36,500) -- optional; ~500 MB download
  cifar10  (use extract_gpu.py --dataset cifar10 for the near-OOD set)

Run (gpu/ folder, CUDA venv):
  .\\venv\\Scripts\\python.exe extract_ood.py --dataset svhn --backbone vit_base_patch16_224.augreg_in21k --out out\\ood_svhn_vitb.npz
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
    ap.add_argument("--dataset", required=True, choices=["svhn", "dtd", "places"])
    ap.add_argument("--backbone", default="vit_base_patch16_224.augreg_in21k")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max", type=int, default=None, help="optional cap on images (first N)")
    args = ap.parse_args()
    if os.path.exists(args.out):
        print(f"{args.out} exists -- skipping"); return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("WARNING: no CUDA")
    net = timm.create_model(args.backbone, pretrained=True, num_classes=0).eval().to(device)
    if device == "cuda":
        net.half()
    cfg = timm.data.resolve_model_data_config(net)
    tf = timm.data.create_transform(**cfg, is_training=False)
    print("backbone:", args.backbone, "| device:", device)

    root = args.data
    if args.dataset == "svhn":
        ds = torchvision.datasets.SVHN(os.path.join(root, "svhn"), split="test", download=True, transform=tf)
    elif args.dataset == "dtd":
        parts = [torchvision.datasets.DTD(os.path.join(root, "dtd"), split=s, download=True, transform=tf)
                 for s in ("train", "val", "test")]
        ds = torch.utils.data.ConcatDataset(parts)
    else:
        ds = torchvision.datasets.Places365(os.path.join(root, "places365"), split="val", small=True,
                                            download=True, transform=tf)
    if args.max is not None:
        ds = torch.utils.data.Subset(ds, list(range(min(args.max, len(ds)))))
    print(f"{args.dataset}: {len(ds)} images")

    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=False,
                                         num_workers=args.workers, pin_memory=True)
    feats, labels = [], []
    t0 = time.time()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            if device == "cuda":
                x = x.half()
            feats.append(net(x).float().cpu().numpy().astype(np.float16))
            labels.append(np.asarray(y).astype(np.int64))
            if i % 20 == 0:
                done = min((i + 1) * args.batch, len(ds))
                print(f"  {done}/{len(ds)} ({done / max(time.time() - t0, 1e-6):.0f} img/s)", flush=True)
    X, y = np.concatenate(feats), np.concatenate(labels)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, Xte=X, yte=y, ood=np.bool_(True),
                        dataset=np.str_(args.dataset), backbone=np.str_(args.backbone))
    print(f"wrote {args.out}  {X.shape}")


if __name__ == "__main__":
    main()
