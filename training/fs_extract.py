"""Download Google's ASL Fingerspelling Recognition landmarks one parquet at a time and keep only the hands.

For each train_landmarks/*.parquet (or supplemental_landmarks/*.parquet) this downloads the file, reads only
the 126 hand columns (left and right hand, 21 points x/y/z), saves one compact .npz per parquet
(float16 [frames, 126] for every sequence, NaN = hand not detected) and deletes the download.
Needs a Kaggle token in KAGGLE_API_TOKEN and the competition rules accepted.

Usage: python fs_extract.py --raw DIR --out DIR [--split train_landmarks] [--limit N]
"""
import argparse, os, subprocess, sys, zipfile
import numpy as np, pandas as pd, pyarrow.parquet as pq

COMP = "asl-fingerspelling"
HAND_COLS = [f"{a}_{h}_hand_{i}" for h in ("left", "right") for i in range(21) for a in "xyz"]


def kaggle_get(name, dest):
    subprocess.run(["kaggle", "competitions", "download", "-c", COMP, "-f", name, "-p", dest, "-q"], check=True)
    for base in (os.path.join(dest, name), os.path.join(dest, os.path.basename(name))):
        if os.path.exists(base + ".zip"):
            with zipfile.ZipFile(base + ".zip") as z:
                z.extractall(os.path.dirname(base))
            os.remove(base + ".zip")
        if os.path.exists(base):
            return base
    raise FileNotFoundError(name)


def extract(path, out_path):
    t = pq.read_table(path, columns=["sequence_id", "frame"] + HAND_COLS)
    sid = t.column("sequence_id").to_numpy()
    frame = t.column("frame").to_numpy()
    x = np.stack([t.column(c).to_numpy(zero_copy_only=False) for c in HAND_COLS], 1).astype(np.float16)
    order = np.lexsort((frame, sid))
    sid, x = sid[order], x[order]
    cuts = np.flatnonzero(np.diff(sid)) + 1
    seqs = {str(s[0]): xs for s, xs in zip(np.split(sid, cuts), np.split(x, cuts))}
    np.savez(out_path, **seqs)
    return len(seqs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train_landmarks")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    csv = "train.csv" if a.split == "train_landmarks" else "supplemental_metadata.csv"
    if not os.path.exists(os.path.join(a.raw, csv)):
        kaggle_get(csv, a.raw)
    meta = pd.read_csv(os.path.join(a.raw, csv))
    files = sorted(meta.file_id.unique())
    if a.limit:
        files = files[: a.limit]
    for i, fid in enumerate(files):
        out_path = os.path.join(a.out, f"{a.split}_{fid}.npz")
        if os.path.exists(out_path):
            continue
        local = kaggle_get(f"{a.split}/{fid}.parquet", a.raw)
        n = extract(local, out_path + ".tmp.npz")
        os.replace(out_path + ".tmp.npz", out_path)
        os.remove(local)
        print(f"[{i + 1}/{len(files)}] {fid}: {n} sequences", flush=True)


if __name__ == "__main__":
    sys.exit(main())
