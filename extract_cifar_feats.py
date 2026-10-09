"""One-time extraction of frozen ImageNet-pretrained ResNet-18 features for
CIFAR-100 (the pretrained-backbone class-IL regime). Cached to
results/cifar100_r18.npz; every later experiment loads the cache.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
import torchvision
from torchvision import transforms

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUT = os.path.join(RES, "cifar100_r18.npz")


def main():
    os.makedirs(DATA, exist_ok=True)
    tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    tr = torchvision.datasets.CIFAR100(DATA, train=True, download=True, transform=tf)
    te = torchvision.datasets.CIFAR100(DATA, train=False, download=True, transform=tf)

    net = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    net.eval()
    torch.set_grad_enabled(False)

    def extract(ds, name):
        loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=0)
        feats, labels = [], []
        t0 = time.time()
        for i, (x, y) in enumerate(loader):
            feats.append(net(x).numpy().astype(np.float16))
            labels.append(y.numpy().astype(np.int64))
            if i % 20 == 0:
                done = (i + 1) * 128
                rate = done / (time.time() - t0)
                print(f"  {name}: {done}/{len(ds)}  ({rate:.0f} img/s)", flush=True)
        return np.concatenate(feats), np.concatenate(labels)

    Xtr, ytr = extract(tr, "train")
    Xte, yte = extract(te, "test")
    np.savez_compressed(OUT, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    print(f"wrote {OUT}  train {Xtr.shape} test {Xte.shape}")


if __name__ == "__main__":
    main()
