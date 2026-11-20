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
    ap.add_argument("--continuous", action="store_true", help="phrases without hands-down rests: each clip trimmed "
                    "to its frames with a hand visible, joined by a short blend, as in continuous signing")
    ap.add_argument("--skip", type=int, default=0, help="skip this many shuffled clips (e.g. to build a tuning set "
                    "that doesn't overlap the default test set: --skip 300)")
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
    clips = clips[a.skip:]
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
        if a.continuous:
            seq = []
            for gloss, lm, w, h in parts:
                hand = np.flatnonzero(np.abs(lm[:, :126]).sum(1) > 0)
                if len(hand):
                    lm = lm[hand[0]:hand[-1] + 1]
                if seq:  # 4 blended frames from the end of one sign to the start of the next
                    a0, b0 = seq[-1], lm[0]
                    for k in range(1, 5):
                        f = k / 5
                        mix = np.where((np.abs(a0) > 0) & (np.abs(b0) > 0), (1 - f) * a0 + f * b0, b0 if f >= .5 else a0)
                        seq.append(mix.astype(np.float32))
                seq += list(lm)
            frames = frames_from_clip(np.stack(seq), 0)
            frames += rest_frames(frames[-1]["pose"], frames[-1]["t"] + FRAME_MS, 15)
        else:
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
