"""Build 8x16 log-spectrograms from the Free Spoken Digit Dataset (the real
temporal stream ASDbrain's streams.py deliberately left as a drop-in). Reads the
dataset's GitHub archive (master.zip of Jakobovski/free-spoken-digit-dataset)
directly: FSDD_ZIP if set, otherwise ASDbrain's cached copy at
<ASDBRAIN_DIR or ../ASDbrain>/data/fsdd/master.zip. Caches to
results/fsdd_spec.npz.
"""
from __future__ import annotations

import io
import os
import zipfile

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft

HERE = os.path.dirname(os.path.abspath(__file__))
ASDBRAIN = os.environ.get("ASDBRAIN_DIR", os.path.join(os.path.dirname(HERE), "ASDbrain"))
ZIP = os.environ.get("FSDD_ZIP", os.path.join(ASDBRAIN, "data", "fsdd", "master.zip"))
OUT = os.path.join(HERE, "results", "fsdd_spec.npz")
T_FRAMES, N_BANDS = 8, 16


def spectrogram_8x16(sig: np.ndarray, fs: int) -> np.ndarray:
    sig = sig.astype(np.float64)
    if sig.ndim > 1:
        sig = sig.mean(axis=1)
    sig /= (np.abs(sig).max() + 1e-9)
    nper = max(64, min(256, len(sig) // 8))
    _, _, Z = stft(sig, fs=fs, nperseg=nper, noverlap=nper // 2)
    S = np.log1p(np.abs(Z) * 100.0)            # (freq_bins, frames)
    fb, tb = S.shape
    # pool frequency into N_BANDS bands, time into T_FRAMES frames
    f_edges = np.linspace(0, fb, N_BANDS + 1).astype(int)
    t_edges = np.linspace(0, tb, T_FRAMES + 1).astype(int)
    out = np.zeros((T_FRAMES, N_BANDS))
    for i in range(T_FRAMES):
        for j in range(N_BANDS):
            blk = S[f_edges[j]:max(f_edges[j + 1], f_edges[j] + 1),
                    t_edges[i]:max(t_edges[i + 1], t_edges[i] + 1)]
            out[i, j] = blk.mean() if blk.size else 0.0
    out /= (out.max() + 1e-9)
    return out


def main():
    X, y, speakers = [], [], []
    with zipfile.ZipFile(ZIP) as z:
        wavs = [n for n in z.namelist()
                if n.endswith(".wav") and "recordings/" in n]
        print(f"{len(wavs)} recordings")
        for n in sorted(wavs):
            base = os.path.basename(n)          # {digit}_{speaker}_{idx}.wav
            digit = int(base.split("_")[0])
            fs, sig = wavfile.read(io.BytesIO(z.read(n)))
            X.append(spectrogram_8x16(sig, fs).ravel())
            y.append(digit)
            speakers.append(base.split("_")[1])
    X = np.asarray(X); y = np.asarray(y, dtype=np.int64)
    np.savez_compressed(OUT, X=X, y=y,
                        speakers=np.asarray(speakers))
    print(f"wrote {OUT}  X {X.shape}  classes {np.bincount(y)}")


if __name__ == "__main__":
    main()
