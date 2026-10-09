r"""Tabular class-incremental streams from network-intrusion data (B-tabular).

No encoder: the rows themselves are the features, and the pipeline's own
min-max scaling (exp_cifar.load_feats) is the only normalisation. "New attack
categories appear over time" is the class-incremental story the partner
document tells, so the class of a flow is its attack category.

Fixed preprocessing (decided before any run; do not tune):
  * drop identifier-like columns (ids, IPs, ports, timestamps, flow ids)
  * one-hot the categorical columns (protocol, service, state / flag)
  * log1p on every non-negative numeric column (counts and byte totals are
    heavy-tailed); negative or infinite entries -> 0
  * keep a class iff it has >= 300 train rows and >= 30 test rows
  * cap train rows per class at 5,000 (stratified, seed 0) so the CPU chain
    runs in minutes and no class dominates the cascade's birth order
  * inc = 2 (the convention of the project's own tabular benchmarks)

Datasets (pick what the box can fetch; report which):
  nslkdd      KDDTrain+ / KDDTest+ (auto-download from a GitHub mirror);
              attack types present in BOTH splits with enough rows
  cicids2017  CIC-IDS2017 MachineLearningCSV (auto-download attempted from
              cicresearch.ca; else place the 8 CSVs under data\cicids2017\);
              no official split -> stratified 70/30 split, seed 0
  unswnb15    UNSW_NB15_training-set.csv / UNSW_NB15_testing-set.csv placed
              manually under data\unswnb15\ (10 categories)

Run (gpu/ folder; CPU is fine):
  .\\venv\\Scripts\\python.exe extract_tabular.py --dataset nslkdd --out out\\ptm_nslkdd_raw.npz
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import urllib.request
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_TRAIN, MIN_TEST, CAP = 300, 30, 5000

KDD_COLS = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
            "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
            "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
            "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
            "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
            "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
            "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
            "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
            "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
            "label", "difficulty"]
KDD_URLS = {"train": ["https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
                      "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTrain%2B.txt"],
            "test": ["https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt",
                     "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/KDDTest%2B.txt"]}
CIC_URL = "http://cicresearch.ca/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV.zip"


def fetch(urls, dest):
    if os.path.exists(dest):
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for u in urls:
        try:
            print("downloading", u)
            urllib.request.urlretrieve(u, dest)
            return dest
        except Exception as e:  # noqa
            print("  failed:", e)
    raise SystemExit(f"could not download {dest}; place it manually")


def encode(df_tr, df_te, cat_cols, drop_cols, label_col):
    import pandas as pd
    y_tr_raw, y_te_raw = df_tr[label_col].astype(str).values, df_te[label_col].astype(str).values
    Xtr = df_tr.drop(columns=[c for c in drop_cols + [label_col] if c in df_tr.columns])
    Xte = df_te.drop(columns=[c for c in drop_cols + [label_col] if c in df_te.columns])
    both = pd.concat([Xtr, Xte], keys=["tr", "te"])
    both = pd.get_dummies(both, columns=[c for c in cat_cols if c in both.columns], dtype=np.float64)
    num = [c for c in both.columns if not any(c.startswith(k + "_") for k in cat_cols)]
    A = both[num].apply(pd.to_numeric, errors="coerce").values.astype(np.float64)
    A[~np.isfinite(A)] = 0.0
    A = np.log1p(np.maximum(A, 0.0))
    both[num] = A
    Xtr, Xte = both.loc["tr"].values.astype(np.float32), both.loc["te"].values.astype(np.float32)
    return Xtr, y_tr_raw, Xte, y_te_raw, list(both.columns)


def select_classes(Xtr, ytr_raw, Xte, yte_raw):
    names = sorted(set(ytr_raw) & set(yte_raw))
    keep = [n for n in names if (ytr_raw == n).sum() >= MIN_TRAIN and (yte_raw == n).sum() >= MIN_TEST]
    dropped = {n: (int((ytr_raw == n).sum()), int((yte_raw == n).sum()))
               for n in sorted(set(ytr_raw) | set(yte_raw)) if n not in keep}
    idx = {n: i for i, n in enumerate(keep)}
    rng = np.random.default_rng(0)
    tr_rows = []
    for n in keep:
        rows = np.flatnonzero(ytr_raw == n)
        if len(rows) > CAP:
            rows = rng.choice(rows, CAP, replace=False)
        tr_rows.append(np.sort(rows))
    tr_rows = np.concatenate(tr_rows)
    te_rows = np.flatnonzero(np.isin(yte_raw, keep))
    ytr = np.asarray([idx[n] for n in ytr_raw[tr_rows]], dtype=np.int64)
    yte = np.asarray([idx[n] for n in yte_raw[te_rows]], dtype=np.int64)
    return Xtr[tr_rows], ytr, Xte[te_rows], yte, keep, dropped


def load_nslkdd(data):
    import pandas as pd
    tr = fetch(KDD_URLS["train"], os.path.join(data, "nslkdd", "KDDTrain+.txt"))
    te = fetch(KDD_URLS["test"], os.path.join(data, "nslkdd", "KDDTest+.txt"))
    df_tr = pd.read_csv(tr, header=None, names=KDD_COLS)
    df_te = pd.read_csv(te, header=None, names=KDD_COLS)
    return encode(df_tr, df_te, ["protocol_type", "service", "flag"], ["difficulty"], "label")


def load_cicids(data):
    import pandas as pd
    d = os.path.join(data, "cicids2017")
    csvs = [os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs if f.lower().endswith(".csv")]
    if not csvs:
        z = fetch([CIC_URL], os.path.join(d, "MachineLearningCSV.zip"))
        zipfile.ZipFile(z).extractall(d)
        csvs = [os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs if f.lower().endswith(".csv")]
    df = pd.concat([pd.read_csv(c, low_memory=False) for c in sorted(csvs)], ignore_index=True)
    df.columns = [c.strip() for c in df.columns]
    df["Label"] = df["Label"].astype(str).str.replace("�", "-", regex=False).str.strip()
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(df))
    n_te = int(0.3 * len(df))
    df_te, df_tr = df.iloc[perm[:n_te]], df.iloc[perm[n_te:]]
    drop = [c for c in df.columns if any(k in c.lower() for k in ("flow id", " ip", "timestamp", "port"))]
    return encode(df_tr, df_te, [], drop, "Label")


def load_unsw(data):
    import pandas as pd
    d = os.path.join(data, "unswnb15")
    tr = os.path.join(d, "UNSW_NB15_training-set.csv"); te = os.path.join(d, "UNSW_NB15_testing-set.csv")
    if not (os.path.exists(tr) and os.path.exists(te)):
        raise SystemExit(f"place UNSW_NB15_training-set.csv and UNSW_NB15_testing-set.csv under {d}")
    df_tr, df_te = pd.read_csv(tr), pd.read_csv(te)
    for df in (df_tr, df_te):
        df["attack_cat"] = df["attack_cat"].astype(str).str.strip()
    return encode(df_tr, df_te, ["proto", "service", "state"], ["id", "label"], "attack_cat")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["nslkdd", "cicids2017", "unswnb15"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--inc", type=int, default=2)
    args = ap.parse_args()
    if os.path.exists(args.out):
        print(f"{args.out} exists -- skipping"); return
    Xtr, ytr_raw, Xte, yte_raw, cols = {"nslkdd": load_nslkdd, "cicids2017": load_cicids,
                                        "unswnb15": load_unsw}[args.dataset](args.data)
    Xtr, ytr, Xte, yte, names, dropped = select_classes(Xtr, ytr_raw, Xte, yte_raw)
    n_cls = len(names)
    print(f"{args.dataset}: {Xtr.shape[1]} features | {n_cls} classes kept (>= {MIN_TRAIN} train, >= {MIN_TEST} test):")
    for i, n in enumerate(names):
        print(f"  {i:2d} {n:28s} train {int((ytr == i).sum()):6d} test {int((yte == i).sum()):6d}")
    print(f"  dropped ({len(dropped)}): " + ", ".join(f"{k} ({a}/{b})" for k, (a, b) in dropped.items()))
    if n_cls % args.inc:
        print(f"WARNING: {n_cls} classes not divisible by inc {args.inc}: last task is smaller")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, n_classes=np.int64(n_cls),
                        inc=np.int64(args.inc), dataset=np.str_(args.dataset), backbone=np.str_("raw-tabular"),
                        classes=np.array(names), columns=np.array(cols))
    print(f"wrote {args.out}  train {Xtr.shape} test {Xte.shape}  inc {args.inc} -> {-(-n_cls // args.inc)} tasks")


if __name__ == "__main__":
    main()
