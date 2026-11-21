"""Extract STAT landmarks from ASL Citizen (Microsoft Research) for train_signs.py.

Reads videos straight out of the remote ASL_Citizen.zip (no 46 GB download), runs the same MediaPipe
models and version as the app (HandLandmarker, 2 hands + PoseLandmarker lite, 0.10.14) on every 2nd frame
(15 fps) and saves one .npz per video: landmarks [T,150] float32 (left hand 63, right hand 63,
pose points 11,12,13,14,15,16,23,24), fps, gloss, split, participant, asl_lex.
File names: {split}__{GLOSS}__{video name}.npz. Resumable: finished videos are skipped.

ASL Citizen's license does not allow redistribution: keep the output private, never commit it.

Usage (Python 3.12, mediapipe==0.10.14, opencv-python-headless, remotezip, pandas):
  python extract_citizen.py --splits DIR --out DIR [--first models/sign_classes_v2.json] [--workers 3]
  DIR for --splits holds train.csv, val.csv, test.csv from ASL_Citizen/splits/ (fetched if missing).
"""
import argparse, json, os, sys, tempfile, time, urllib.request
from multiprocessing import get_context

URL = "https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip"
MODELS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}
POSE_IDX = [11, 12, 13, 14, 15, 16, 23, 24]
FRAME_STEP = 2
_z = _hand = _pose = None
WORK = OUT = None


def remote():
    from remotezip import RemoteZip
    return RemoteZip(URL, support_suffix_range=False)


def init(work, out):
    global _z, WORK, OUT
    WORK, OUT = work, out
    _z = remote()


def extract(path):
    import numpy as np, cv2, mediapipe as mp
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    b = lambda f: mpp.BaseOptions(model_asset_path=os.path.join(WORK, f))
    V = vision.RunningMode.VIDEO
    hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=b("hand_landmarker.task"), running_mode=V, num_hands=2))
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(base_options=b("pose_landmarker_lite.task"), running_mode=V, num_poses=1))
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    rows, i = [], 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i % FRAME_STEP == 0:
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            ts = int(i * 1000 / fps)
            hr, pr = hand.detect_for_video(img, ts), pose.detect_for_video(img, ts)
            v = np.zeros(150, np.float32)
            for lm, hd in zip(hr.hand_landmarks, hr.handedness):
                s = 0 if hd[0].category_name == "Left" else 1
                v[s * 63:(s + 1) * 63] = np.array([[l.x, l.y, l.z] for l in lm]).ravel()
            if pr.pose_landmarks:
                lm = pr.pose_landmarks[0]
                v[126:150] = np.array([[lm[j].x, lm[j].y, lm[j].z] for j in POSE_IDX]).ravel()
            rows.append(v)
        i += 1
    cap.release(); hand.close(); pose.close()
    return (np.stack(rows) if rows else np.zeros((0, 150), np.float32)), fps / FRAME_STEP


def out_name(r):
    return os.path.join(OUT, f"{r['split']}__{r['gloss'].replace('/', '_')}__{os.path.splitext(r['video'])[0]}.npz")


def work(r):
    import numpy as np
    dst = out_name(r)
    if os.path.exists(dst):
        return "skip"
    err = ""
    for attempt in range(3):
        try:
            data = _z.read("ASL_Citizen/videos/" + r["video"])
            with tempfile.NamedTemporaryFile(suffix=".mp4", dir=WORK, delete=False) as f:
                f.write(data)
            try:
                lm, fps = extract(f.name)
            finally:
                os.remove(f.name)
            tmp = dst + ".tmp.npz"
            np.savez_compressed(tmp, landmarks=lm, fps=fps, gloss=r["gloss"], split=r["split"],
                                participant=r["participant"], asl_lex=r["asl_lex"])
            os.replace(tmp, dst)
            return "ok"
        except Exception as e:  # network hiccups: retry
            err = str(e); time.sleep(3 * (attempt + 1))
    return f"fail {r['video']}: {err}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=os.path.expanduser("~/stat_citizen_work"))
    ap.add_argument("--first", help="sign_classes JSON: extract these glosses first")
    ap.add_argument("--workers", type=int, default=3); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import pandas as pd
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.work, exist_ok=True)
    for name, url in MODELS.items():
        if not os.path.exists(os.path.join(a.work, name)):
            urllib.request.urlretrieve(url, os.path.join(a.work, name))
    rows = []
    for split in ("train", "val", "test"):
        p = os.path.join(a.splits, f"{split}.csv")
        if not os.path.exists(p):
            open(p, "wb").write(remote().read(f"ASL_Citizen/splits/{split}.csv"))
        for _, x in pd.read_csv(p).iterrows():
            rows.append({"split": split, "gloss": str(x["Gloss"]), "video": x["Video file"],
                         "participant": x["Participant ID"], "asl_lex": str(x["ASL-LEX Code"])})
    first = set(json.load(open(a.first))) if a.first else set()
    # glosses in --first before the rest; within each, alternate glosses so any prefix covers many signs
    seen = {}
    for r in rows:
        seen[r["gloss"]] = seen.get(r["gloss"], 0) + 1
        r["rank"] = (r["gloss"] not in first, seen[r["gloss"]])
    rows.sort(key=lambda r: r["rank"])
    global OUT
    OUT = a.out
    todo = [r for r in rows if not os.path.exists(out_name(r))]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(rows)} videos, {len(rows) - len(todo)} already done, {len(todo)} to do", flush=True)
    t0, done, fails = time.time(), 0, 0
    with get_context("fork").Pool(a.workers, initializer=init, initargs=(a.work, a.out)) as pool:
        for status in pool.imap_unordered(work, todo, chunksize=4):
            done += 1
            if status.startswith("fail"):
                fails += 1; print(status, flush=True)
            if done % 200 == 0 or done == len(todo):
                el = time.time() - t0
                print(f"{done}/{len(todo)} ({fails} failed), {done / el:.2f} videos/s, "
                      f"~{(len(todo) - done) / (done / el) / 3600:.1f} h left", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
