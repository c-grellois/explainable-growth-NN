"""First-session parameter-efficient tuning ("Phase 1"), then frozen extraction.

APER (Zhou et al. 2023) and RanPAC (McDonnell et al. 2023) tune the ViT once on
the FIRST task's data with a parameter-efficient module, then freeze it for the
rest of the stream. Read in this project's terms it is a birth step: organ 0
also shapes the substrate, once. This script does exactly that and writes a
feature cache the CPU pipeline reads unchanged:

  1. backbone  timm ViT-B/16 (in21k by default), all weights frozen
  2. adapters  a parallel bottleneck adapter (d -> r -> d, ReLU, zero-init up
               projection, scale s) on the MLP of every transformer block
               (AdaptFormer-style; r = 16 as in APER)
  3. task 0    the first task's classes under THIS PROJECT'S protocol for the
               given seed: order = default_rng(seed + 400).permutation(n_classes),
               task 0 = the first `inc` classes. Only the rows that
               exp_cifar.load_feats(seed) would put in TRAIN are used (the 10 %
               validation rows are excluded, so the CPU pipeline's thresholds are
               solved on data the adapters never saw).
  4. training  adapters + a cosine classifier over task-0 classes, cross-entropy,
               AdamW, cosine schedule, mixed precision
  5. extract   features for ALL train and test images with the tuned, frozen
               backbone -> <out>.npz with the same fields as extract_ptm.py plus
               phase1 metadata (seed, task-0 classes, epochs, r, val accuracy)

The cache is seed-specific: use it only with --first-seed <seed> --seeds 1 on
the CPU side (the class order it was born from must be the one evaluated).

CIFAR-100 (torchvision):
  python phase1_extract.py --dataset cifar100 --seed 2 --out out\\cifar100_vitb_p1s2.npz
ImageFolder suite (train/ test/ layout; inc from the table in HANDOFF_GPU2.md):
  python phase1_extract.py --root data\\imagenet-r --dataset imr --inc 20 --seed 2 --out out\\ptm_imr_vitb_p1s2.npz
"""
from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as T


class ParallelAdapter(nn.Module):
    def __init__(self, dim, r=16, scale=0.1):
        super().__init__()
        self.down = nn.Linear(dim, r)
        self.up = nn.Linear(r, dim)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)
        self.act = nn.ReLU()
        self.scale = scale

    def forward(self, x):
        return self.up(self.act(self.down(x))) * self.scale


class MlpWithAdapter(nn.Module):
    def __init__(self, mlp, adapter):
        super().__init__()
        self.mlp, self.adapter = mlp, adapter

    def forward(self, x):
        return self.mlp(x) + self.adapter(x)


class CosineHead(nn.Module):
    def __init__(self, dim, n_cls, scale=16.0):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_cls, dim) * 0.02)
        self.scale = scale

    def forward(self, f):
        return self.scale * nn.functional.normalize(f, dim=1) @ nn.functional.normalize(self.w, dim=1).T


def build_datasets(args, train_tf, eval_tf):
    if args.dataset == "cifar100":
        data_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), "..", "data")
        tr_raw = torchvision.datasets.CIFAR100(data_dir, train=True, download=True, transform=train_tf)
        tr_eval = torchvision.datasets.CIFAR100(data_dir, train=True, download=True, transform=eval_tf)
        te = torchvision.datasets.CIFAR100(data_dir, train=False, download=True, transform=eval_tf)
        targets = np.array(tr_raw.targets); n_cls = 100
        inc = args.inc or 10
    else:
        if not args.root:
            raise SystemExit("--root is required for ImageFolder datasets")
        tr_raw = torchvision.datasets.ImageFolder(os.path.join(args.root, "train"), transform=train_tf)
        tr_eval = torchvision.datasets.ImageFolder(os.path.join(args.root, "train"), transform=eval_tf)
        te = torchvision.datasets.ImageFolder(os.path.join(args.root, "test"), transform=eval_tf)
        if tr_raw.classes != te.classes:
            raise SystemExit("train/test class folders differ")
        targets = np.array(tr_raw.targets); n_cls = len(tr_raw.classes)
        if not args.inc:
            raise SystemExit("--inc is required for ImageFolder datasets")
        inc = args.inc
    return tr_raw, tr_eval, te, targets, n_cls, inc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="cifar100, or a short name for an ImageFolder set")
    ap.add_argument("--root", default=None)
    ap.add_argument("--inc", type=int, default=None)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--backbone", default="vit_base_patch16_224.augreg_in21k")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--adapter-scale", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--per-class-cap", type=int, default=None,
                    help="cap on train images per class for the CIFAR cache (the "
                         "reference cifar100_vitb cache used 250; default keeps all)")
    args = ap.parse_args()
    if os.path.exists(args.out):
        print(f"{args.out} exists -- skipping"); return
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("WARNING: no CUDA; this will be very slow")

    net = timm.create_model(args.backbone, pretrained=True, num_classes=0).to(device)
    cfg = timm.data.resolve_model_data_config(net)
    eval_tf = timm.data.create_transform(**cfg, is_training=False)
    train_tf = T.Compose([T.RandomResizedCrop(cfg["input_size"][-1], scale=(0.7, 1.0)),
                          T.RandomHorizontalFlip(), T.ToTensor(),
                          T.Normalize(cfg["mean"], cfg["std"])])
    tr_raw, tr_eval, te, targets, n_cls, inc = build_datasets(args, train_tf, eval_tf)

    # ---- the protocol's first task, and the rows load_feats(seed) trains on ---
    order = np.random.default_rng(args.seed + 400).permutation(n_cls)
    task0 = sorted(int(c) for c in order[:inc])
    N = len(targets)
    if args.per_class_cap is not None:
        # replicate extract_gpu.py's stratified cap (seed 0) BEFORE the val split,
        # so row order and count match the capped cache
        rng0 = np.random.default_rng(0)
        keep = np.concatenate([rng0.choice(np.flatnonzero(targets == c),
                                           min(args.per_class_cap, int((targets == c).sum())), replace=False)
                               for c in range(n_cls)])
        keep = np.sort(keep)          # extract_gpu.py used the concatenated (class-ordered) list; see note
        row_ids = keep
    else:
        row_ids = np.arange(N)
    perm = np.random.default_rng(args.seed).permutation(len(row_ids))
    n_val = int(0.1 * len(row_ids))
    train_rows = row_ids[perm[n_val:]]
    t0_rows = train_rows[np.isin(targets[train_rows], task0)]
    cls_index = {c: i for i, c in enumerate(task0)}
    print(f"{args.dataset}: {n_cls} classes, inc {inc}, seed {args.seed} -> task 0 = {task0}; "
          f"{len(t0_rows)} training images for Phase 1 (val rows excluded)", flush=True)

    # ---- adapters -------------------------------------------------------------
    for p in net.parameters():
        p.requires_grad = False
    dim = net.embed_dim
    for blk in net.blocks:
        blk.mlp = MlpWithAdapter(blk.mlp, ParallelAdapter(dim, r=args.r, scale=args.adapter_scale).to(device))
    head = CosineHead(dim, len(task0)).to(device)
    params = [p for p in net.parameters() if p.requires_grad] + list(head.parameters())
    n_tuned = sum(p.numel() for p in params)
    print(f"tunable parameters: {n_tuned:,} (backbone {sum(p.numel() for p in net.parameters()):,})", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * math.ceil(len(t0_rows) / args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    sub = torch.utils.data.Subset(tr_raw, t0_rows.tolist())
    loader = torch.utils.data.DataLoader(sub, batch_size=args.batch, shuffle=True,
                                         num_workers=args.workers, pin_memory=True, drop_last=False)
    remap = torch.full((n_cls,), -1, dtype=torch.long)
    for c, i in cls_index.items():
        remap[c] = i
    remap = remap.to(device)
    t_start = time.time()
    for ep in range(args.epochs):
        net.train(); head.train(); tot, correct, seen = 0.0, 0, 0
        for x, y in loader:
            x = x.to(device, non_blocking=True); y = remap[y.to(device)]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                f = net(x); logits = head(f.float())
                loss = nn.functional.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(y); correct += (logits.argmax(1) == y).sum().item(); seen += len(y)
        print(f"  epoch {ep + 1}/{args.epochs}: loss {tot / seen:.3f} train acc {correct / seen:.3f} "
              f"({time.time() - t_start:.0f}s)", flush=True)

    # ---- validation of Phase 1 on the held-out val rows of task 0 --------------
    net.eval()
    val_rows = row_ids[perm[:n_val]]
    v0 = val_rows[np.isin(targets[val_rows], task0)]
    if len(v0):
        vl = torch.utils.data.DataLoader(torch.utils.data.Subset(tr_eval, v0.tolist()), batch_size=128,
                                         num_workers=args.workers)
        c, n = 0, 0
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            for x, y in vl:
                pred = head(net(x.to(device)).float()).argmax(1).cpu()
                c += (pred == remap.cpu()[y]).sum().item(); n += len(y)
        val_acc = c / n
    else:
        val_acc = float("nan")
    print(f"Phase 1 task-0 val accuracy: {val_acc:.3f}", flush=True)

    # ---- extraction with the tuned, frozen backbone -----------------------------
    if device == "cuda":
        net.half()

    @torch.no_grad()
    def extract(ds, rows, name):
        d = torch.utils.data.Subset(ds, rows.tolist()) if rows is not None else ds
        loader = torch.utils.data.DataLoader(d, batch_size=128, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)
        feats, labels = [], []
        t0 = time.time()
        for i, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            if device == "cuda":
                x = x.half()
            feats.append(net(x).float().cpu().numpy().astype(np.float16))
            labels.append(np.asarray(y).astype(np.int64))
            if i % 40 == 0:
                print(f"  {name}: {min((i + 1) * 128, len(d))}/{len(d)} ({(i + 1) * 128 / max(time.time() - t0, 1e-6):.0f} img/s)", flush=True)
        return np.concatenate(feats), np.concatenate(labels)

    Xtr, ytr = extract(tr_eval, row_ids if args.per_class_cap is not None else None, "train")
    Xte, yte = extract(te, None, "test")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte,
                        n_classes=np.int64(n_cls), inc=np.int64(inc),
                        dataset=np.str_(args.dataset), backbone=np.str_(args.backbone),
                        phase1_seed=np.int64(args.seed), phase1_task0=np.asarray(task0),
                        phase1_epochs=np.int64(args.epochs), phase1_r=np.int64(args.r),
                        phase1_val_acc=np.float64(val_acc), phase1_tuned_params=np.int64(n_tuned))
    print(f"wrote {args.out}  train {Xtr.shape}  test {Xte.shape}")


if __name__ == "__main__":
    main()
