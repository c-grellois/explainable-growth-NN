"""GPU feature extractor for the contracted-growth benchmarks.

Self-contained (torch + torchvision + timm + numpy only). Extracts frozen
features from a timm backbone for CIFAR-10/100, fp16 on CUDA, resumable
(skips if the output file exists), stratified train subsampling.

Run on the GPU box via setup_and_extract.ps1, or directly:
  python extract_gpu.py --backbone vit_base_patch14_dinov2.lvd142m \
      --dataset cifar100 --per-class 500 --out out/cifar100_dinov2b.npz
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
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--dataset", choices=["cifar100", "cifar10"],
                    default="cifar100")
    ap.add_argument("--per-class", type=int, default=500)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    if os.path.exists(args.out):
        print(f"{args.out} exists -- skipping")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = timm.create_model(args.backbone, pretrained=True, num_classes=0)
    net.eval().to(device)
    if device == "cuda":
        net.half()
    cfg = timm.data.resolve_model_data_config(net)
    tf = timm.data.create_transform(**cfg, is_training=False)
    print("backbone:", args.backbone, "| transform:", tf)

    DS = (torchvision.datasets.CIFAR100 if args.dataset == "cifar100"
          else torchvision.datasets.CIFAR10)
    data_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)),
                            "..", "data")
    tr = DS(data_dir, train=True, download=True, transform=tf)
    te = DS(data_dir, train=False, download=True, transform=tf)

    targets = np.array(tr.targets)
    rng = np.random.default_rng(0)
    n_cls = targets.max() + 1
    keep = np.concatenate([
        rng.choice(np.flatnonzero(targets == c),
                   min(args.per_class, (targets == c).sum()), replace=False)
        for c in range(n_cls)])
    tr = torch.utils.data.Subset(tr, keep.tolist())

    @torch.no_grad()
    def extract(ds, name):
        loader = torch.utils.data.DataLoader(ds, batch_size=args.batch,
                                             num_workers=2, pin_memory=True)
        feats, labels = [], []
        t0 = time.time()
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            if device == "cuda":
                x = x.half()
            feats.append(net(x).float().cpu().numpy().astype(np.float16))
            labels.append(np.asarray(y).astype(np.int64))
            if i % 20 == 0:
                done = (i + 1) * args.batch
                print(f"  {name}: {done}/{len(ds)} "
                      f"({done / (time.time() - t0):.0f} img/s)", flush=True)
        return np.concatenate(feats), np.concatenate(labels)

    Xtr, ytr = extract(tr, "train")
    Xte, yte = extract(te, "test")
    np.savez_compressed(args.out, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    print(f"wrote {args.out}  train {Xtr.shape}  test {Xte.shape}")


if __name__ == "__main__":
    main()
