"""Build stand-in fingerspelling takes (stat-capture-v1 JSON) from Google's ASL Fingerspelling data
(Kaggle asl-fingerspelling, CC BY 4.0), to check with tools/eval_recordings.mjs that fingerspelled
sentences are still read as fingerspelling (and not as whole signs).

Uses sentences by the fingerspelling model's held-out signers only. Each take is labeled the way the
evaluator expects spelled words ("M-Y C-A-T"), followed by 1 s with the hands down.
The output is derived from the Kaggle data: keep it on this machine, don't commit it.

Usage: python make_fs_takes.py PARQUET SUPPLEMENTAL_CSV OUT_DIR [--n 60] [--max-words 4]
"""
import argparse, json, os, re
import numpy as np, pandas as pd, pyarrow.parquet as pq

TEST_SIGNERS = {4, 43, 53, 68, 99, 128, 157, 225, 233, 251}  # same split as training/train_fingerspell.py
POSE_IDX = [11, 12, 13, 14, 15, 16, 23, 24]
FRAME_MS = 1000 / 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet"); ap.add_argument("csv"); ap.add_argument("out")
    ap.add_argument("--n", type=int, default=60); ap.add_argument("--max-words", type=int, default=4)
    a = ap.parse_args()
    fid = int(os.path.basename(a.parquet).split(".")[0])
    meta = pd.read_csv(a.csv)
    meta = meta[(meta.file_id == fid) & meta.participant_id.isin(TEST_SIGNERS)]
    ok = meta.phrase.map(lambda p: bool(re.fullmatch(r"[a-z]{2,}( [a-z]{2,})*", p)) and len(p.split()) <= a.max_words)
    meta = meta[ok].head(a.n)
    hand = [f"{c}_{h}_hand_{i}" for h in ("left", "right") for i in range(21) for c in "xyz"]
    pose = [f"{c}_pose_{i}" for i in POSE_IDX for c in "xyz"]
    t = pq.read_table(a.parquet, columns=["sequence_id", "frame"] + hand + pose,
                      filters=[("sequence_id", "in", meta.sequence_id.tolist())]).to_pandas().reset_index()
    os.makedirs(a.out, exist_ok=True)
    n = 0
    for _, m in meta.iterrows():
        d = t[t.sequence_id == m.sequence_id].sort_values("frame")
        if len(d) < 10:
            continue
        H, P = d[hand].to_numpy(np.float32), d[pose].to_numpy(np.float32)
        frames = []
        for i in range(len(d)):
            hs = []
            for k, name in enumerate(("Left", "Right")):
                v = H[i, k * 63:(k + 1) * 63]
                if not np.isnan(v).any():
                    hs.append({"handedness": name, "score": 0.99, "landmarks": v.reshape(21, 3).round(5).tolist()})
            p = P[i].reshape(8, 3)
            frames.append({"t": round(i * FRAME_MS), "hands": hs,
                           "pose": None if np.isnan(p).any() else [[*map(float, q.round(5)), 1.0] for q in p]})
        last = next((f["pose"] for f in reversed(frames) if f["pose"]), None)
        t0 = frames[-1]["t"]
        frames += [{"t": round(t0 + (k + 1) * FRAME_MS), "hands": [], "pose": last} for k in range(30)]
        label = " ".join("-".join(w.upper()) for w in m.phrase.split())
        n += 1
        json.dump({"format": "stat-capture-v1", "signer": f"K{m.participant_id}", "dominantHand": "Right",
                   "type": "phrase", "label": label, "take": n, "videoWidth": 1080, "videoHeight": 1920,
                   "poseIndices": POSE_IDX, "frames": frames}, open(f"{a.out}/FS_{n:03d}.json", "w"))
    print(f"{n} fingerspelling takes written to {a.out}")


if __name__ == "__main__":
    main()
