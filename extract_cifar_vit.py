"""One-time extraction of frozen ViT-B/16 (ImageNet-21k pretrained, timm
`vit_base_patch16_224.augreg_in21k` -- SimpleCIL's backbone) features for
CIFAR-100. Train is stratified-subsampled to 250/class (prototypes and PCA
territories saturate well before that); test is kept whole. Cached to
results/cifar100_vitb.npz.
"""
from __future__ import annotations

import os
import time

import numpy as np
import timm
import torch
import torchvision
from torchvision import transforms

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT = os.path.join(RES, "cifar100_vitb.npz")
PER_CLASS = 250


def main():
    net = timm.create_model("vit_base_patch16_224.augreg_in21k",
                            pretrained=True, num_classes=0)
    net.eval()
    torch.set_grad_enabled(False)
    cfg = timm.data.resolve_model_data_config(net)
    tf = timm.data.create_transform(**cfg, is_training=False)
    print("transform:", tf, flush=True)

    tr = torchvision.datasets.CIFAR100(DATA, train=True, download=True, transform=tf)
    te = torchvision.datasets.CIFAR100(DATA, train=False, download=True, transform=tf)

    # stratified subsample of train
    targets = np.array(tr.targets)
    rng = np.random.default_rng(0)
    keep = np.concatenate([rng.choice(np.flatnonzero(targets == c),
                                      PER_CLASS, replace=False)
                           for c in range(100)])
    tr = torch.utils.data.Subset(tr, keep.tolist())

    def extract(ds, name):
        loader = torch.utils.data.DataLoader(ds, batch_size=64, num_workers=0)
        feats, labels = [], []
        t0 = time.time()
        for i, (x, y) in enumerate(loader):
            feats.append(net(x).numpy().astype(np.float16))
            labels.append(np.asarray(y).astype(np.int64))
            if i % 10 == 0:
                done = (i + 1) * 64
                rate = done / (time.time() - t0)
                eta = (len(ds) - done) / max(rate, 1e-9) / 60
                print(f"  {name}: {done}/{len(ds)} ({rate:.1f} img/s, eta {eta:.0f}m)",
                      flush=True)
        return np.concatenate(feats), np.concatenate(labels)

    Xtr, ytr = extract(tr, "train")
    Xte, yte = extract(te, "test")
    np.savez_compressed(OUT, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    print(f"wrote {OUT}  train {Xtr.shape} test {Xte.shape}")


if __name__ == "__main__":
    main()
