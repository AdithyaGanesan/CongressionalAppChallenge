"""Build stand-in capture-page takes (stat-capture-v1 JSON) from ASL Citizen landmark clips,
to test tools/eval_recordings.mjs before real recordings exist.

- sign takes: one clip each, from signers held out of training (default P15, P17, P18)
- phrase takes: several clips joined with about 1 s of hands down between them, labeled with
  the signs in order, a rough stand-in for continuous signing

The clips are ASL Citizen data (Microsoft Research license: no redistribution), so the output
stays on this machine; never commit it.

Usage: python make_test_takes.py LANDMARK_DIR CLASSES_JSON OUT_DIR [--signs 60] [--phrases 20]
"""
import argparse, glob, json, os, random
import numpy as np

POSE_IDX = [11, 12, 13, 14, 15, 16, 23, 24]
FRAME_MS = 1000 / 15  # ASL Citizen landmarks are every 2nd frame of 30 fps video


def frames_from_clip(lm, t0):
    """150-float rows -> capture frames (hands with handedness, 8 pose points)."""
    out = []
    for i, v in enumerate(lm.astype(np.float32)):
        hands = []
        for slot, name in ((0, "Left"), (1, "Right")):
            h = v[slot * 63:(slot + 1) * 63]
            if np.abs(h).sum() > 0:
                hands.append({"handedness": name, "score": 0.99, "landmarks": h.reshape(21, 3).round(5).tolist()})
        p = v[126:150].reshape(8, 3)
        pose = [[*map(float, q.round(5)), 1.0] for q in p] if np.abs(p).sum() > 0 else None
        out.append({"t": round(t0 + i * FRAME_MS), "hands": hands, "pose": pose})
    return out


def rest_frames(pose, t0, n):
    """Hands down (not detected) with the body still in view."""
    return [{"t": round(t0 + k * FRAME_MS), "hands": [], "pose": pose} for k in range(n)]


def take(kind, label, frames, w, h, n):
    return {"format": "stat-capture-v1", "signer": "TEST", "dominantHand": "Right", "type": kind,
            "label": label, "take": n, "videoWidth": w, "videoHeight": h, "poseIndices": POSE_IDX,
            "frames": frames}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("landmarks"); ap.add_argument("classes"); ap.add_argument("out")
    ap.add_argument("--signers", default="P15,P17,P18"); ap.add_argument("--signs", type=int, default=60)
    ap.add_argument("--phrases", type=int, default=20); ap.add_argument("--words", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)
    known = set(json.load(open(a.classes)))
    signers = set(a.signers.split(","))
    clips = []
    for p in sorted(glob.glob(f"{a.landmarks}/test__*.npz")):
        d = np.load(p)
        if str(d["participant"]) in signers and str(d["gloss"]) in known and len(d["landmarks"]) >= 8:
            w, h = (int(d["width"]), int(d["height"])) if "width" in d.files else (640, 480)
            clips.append((str(d["gloss"]), d["landmarks"], w, h))
    random.shuffle(clips)
    os.makedirs(a.out, exist_ok=True)
    n = 0
    for gloss, lm, w, h in clips[:a.signs]:
        fr = frames_from_clip(lm, 0)
        n += 1
        json.dump(take("sign", gloss, fr, w, h, n), open(f"{a.out}/TEST_sign_{gloss}_{n:03d}.json", "w"))
    rest = clips[a.signs:]
    for i in range(a.phrases):
        parts = rest[i * a.words:(i + 1) * a.words]
        if len(parts) < a.words:
            break
        frames, t = [], 0.0
        for gloss, lm, w, h in parts:
            fr = frames_from_clip(lm, t)
            frames += fr
            t = fr[-1]["t"] + FRAME_MS
            gap = rest_frames(fr[-1]["pose"], t, 15)  # ~1 s with hands down
            frames += gap
            t = gap[-1]["t"] + FRAME_MS
        label = " ".join(g for g, *_ in parts)
        n += 1
        json.dump(take("phrase", label, frames, parts[0][2], parts[0][3], n),
                  open(f"{a.out}/TEST_phrase_{i + 1:03d}.json", "w"))
    print(f"{n} takes written to {a.out} ({len(clips)} held-out clips available)")


if __name__ == "__main__":
    main()
