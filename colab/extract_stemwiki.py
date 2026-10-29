# STAT: extract landmarks from ASL STEM Wiki (continuous signing + English sentences).
# One cell. Run the same cell in several Colab notebooks at once, each with a different SLICE
# (0, 1, 2, ...) and the same NUM_SLICES. Safe to re-run after a disconnect: it skips videos
# already saved. Output: one .npz per video in Google Drive (private; never put it on GitHub).
# Settings. If you run this through the short loader cell (see colab/README.md), the values you
# set there are kept; otherwise these defaults apply.
SLICE = globals().get("SLICE", 0)                # this notebook's slice: 0, 1, 2 ... up to NUM_SLICES - 1
NUM_SLICES = globals().get("NUM_SLICES", 3)      # how many notebooks you run in parallel
TARGET_HOURS = globals().get("TARGET_HOURS", 40) # total hours of video across ALL slices (316 h = everything)
WORKERS = globals().get("WORKERS", 2)            # Colab free CPUs; on your own computer use its CPU cores

import os, sys, subprocess, importlib
IN_COLAB = "google.colab" in sys.modules
def pip(*a):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *a], check=False)
pip("remotezip", "pandas")
# the same MediaPipe version as the app (0.10.14; tested with numpy 2). Newer versions need
# graphics libraries (libEGL) that Colab's CPU machines may not have.
pip("mediapipe==0.10.14", "opencv-python-headless")
import mediapipe as mp; print("mediapipe", mp.__version__)

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT = "/content/drive/MyDrive/stat_stemwiki"
    WORK = "/content/stat_work"
else:
    OUT = os.path.expanduser("~/stat_stemwiki")   # on your own computer; upload the folder to Drive later
    WORK = os.path.expanduser("~/stat_work")
os.makedirs(OUT, exist_ok=True); os.makedirs(WORK, exist_ok=True)

import urllib.request
for name, url in {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}.items():
    if not os.path.exists(f"{WORK}/{name}"):
        urllib.request.urlretrieve(url, f"{WORK}/{name}")

# ---- worker code, written to a file so multiprocessing can import it ----
LIB = r'''
import os, time, tempfile, hashlib, numpy as np, cv2, mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
URL = "https://download.microsoft.com/download/4/c/f/4cfec788-7478-4e47-9a15-ace9b6a96198/ASL_STEM_Wiki.zip"
POSE_IDX = [11, 12, 13, 14, 15, 16, 23, 24]          # shoulders, elbows, wrists, hips (same as ASL Citizen)
FACE_IDX = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 33, 133, 159, 145, 362, 263, 386, 374, 1,
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]  # brows, eyes, nose, lips
FRAME_STEP = 2                                        # every 2nd frame, same as ASL Citizen (15 fps)
_z = None; _names = None; WORK = None; OUT = None

def init(work, out):
    global _z, _names, WORK, OUT
    from remotezip import RemoteZip
    WORK, OUT = work, out
    _z = RemoteZip(URL)
    _names = {os.path.basename(n): n for n in _z.namelist() if n.endswith(".mp4")}

def landmarkers():
    b = lambda f: mp_python.BaseOptions(model_asset_path=f"{WORK}/{f}")
    V = vision.RunningMode.VIDEO
    return (vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=b("hand_landmarker.task"), running_mode=V, num_hands=2)),
            vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(base_options=b("pose_landmarker_lite.task"), running_mode=V, num_poses=1)),
            vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=b("face_landmarker.task"), running_mode=V, num_faces=1)))

def extract(path):
    hand, pose, face = landmarkers()
    cap = cv2.VideoCapture(path); fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(3)), int(cap.get(4)); hp, fc, i = [], [], 0
    while True:
        ok, bgr = cap.read()
        if not ok: break
        if i % FRAME_STEP == 0:
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            ts = int(i * 1000 / fps)
            hr, pr, fr = hand.detect_for_video(img, ts), pose.detect_for_video(img, ts), face.detect_for_video(img, ts)
            v = np.zeros(150, np.float32)
            for lm, hd in zip(hr.hand_landmarks, hr.handedness):
                s = 0 if hd[0].category_name == "Left" else 1
                v[s*63:(s+1)*63] = np.array([[l.x, l.y, l.z] for l in lm]).ravel()
            if pr.pose_landmarks:
                lm = pr.pose_landmarks[0]; v[126:150] = np.array([[lm[j].x, lm[j].y, lm[j].z] for j in POSE_IDX]).ravel()
            f = np.zeros(len(FACE_IDX) * 3, np.float32)
            if fr.face_landmarks:
                lm = fr.face_landmarks[0]; f[:] = np.array([[lm[j].x, lm[j].y, lm[j].z] for j in FACE_IDX]).ravel()
            hp.append(v); fc.append(f)
        i += 1
    cap.release(); hand.close(); pose.close(); face.close()
    E = lambda a, n: np.stack(a) if a else np.zeros((0, n), np.float32)
    return E(hp, 150).astype(np.float16), E(fc, len(FACE_IDX) * 3).astype(np.float16), fps / FRAME_STEP, w, h

def work(row):
    fn = row["video filename"]
    out = f"{OUT}/{fn.replace('.mp4', '')}.npz"
    if os.path.exists(out): return "skip", 0.0
    err = ""
    for attempt in range(3):
        try:
            data = _z.read(_names[fn])
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=WORK) as f: f.write(data)
            try: hp, fc, fps, w, h = extract(f.name)
            finally: os.remove(f.name)
            tmp = f"{WORK}/{os.path.basename(out)}.tmp.npz"
            np.savez_compressed(tmp, landmarks=hp, face=fc, fps=fps, width=w, height=h,
                                participant=row["participant"], article=row["article"],
                                section=row["section index"], sentence=row["sentence index"],
                                text=str(row["sentence text"]), face_idx=np.array(FACE_IDX),
                                mediapipe_version=mp.__version__)
            import shutil; shutil.move(tmp, out)
            return "ok", float(row["video length"])
        except Exception as e:
            err = str(e); time.sleep(3 * (attempt + 1))
    return "fail " + fn + " " + err, 0.0
'''
open(f"{WORK}/stemwiki_worker.py", "w").write(LIB)
sys.path.insert(0, WORK)
import stemwiki_worker; importlib.reload(stemwiki_worker)

# ---- which videos this slice handles: balanced across signers, so any prefix is a fair sample ----
import pandas as pd, hashlib, time
from remotezip import RemoteZip
idx_path = f"{WORK}/videos.csv"
if not os.path.exists(idx_path):
    with RemoteZip(stemwiki_worker.URL) as z, z.open("ASL_STEM_Wiki/videos.csv") as f:
        pd.read_csv(f).to_csv(idx_path, index=False)
v = pd.read_csv(idx_path)
v["video length"] = pd.to_numeric(v["video length"], errors="coerce").fillna(0)
v = v[(v["video length"] > 0) & (v["video length"] < 120)]
v["h"] = v["video filename"].map(lambda s: int(hashlib.md5(s.encode()).hexdigest(), 16))
v["slice"] = v["h"] % NUM_SLICES
v = v[v["slice"] == SLICE].sort_values("h")
v["rank"] = v.groupby("participant").cumcount()        # round-robin over signers
v = v.sort_values(["rank", "h"])
v = v[v["video length"].cumsum() <= TARGET_HOURS * 3600 / NUM_SLICES]
rows = [r for _, r in v.iterrows()]
have = sum(os.path.exists(f"{OUT}/{r['video filename'].replace('.mp4', '')}.npz") for r in rows)
print(f"slice {SLICE}/{NUM_SLICES}: {len(rows)} videos, {v['video length'].sum()/3600:.1f} h, "
      f"{v['participant'].nunique()} signers; {have} already saved")

# ---- run ----
from multiprocessing import get_context
t0, done, secs, fails = time.time(), 0, 0.0, 0
todo = [r.to_dict() for r in rows if not os.path.exists(f"{OUT}/{r['video filename'].replace('.mp4', '')}.npz")]
with get_context("fork").Pool(WORKERS, initializer=stemwiki_worker.init, initargs=(WORK, OUT)) as pool:
    for status, s in pool.imap_unordered(stemwiki_worker.work, todo, chunksize=2):
        done += 1; secs += s
        if status.startswith("fail"): fails += 1; print(status)
        if done % 25 == 0 or done == len(todo):
            el = time.time() - t0
            print(f"{done}/{len(todo)} done ({secs/3600:.2f} h of video), {fails} failed, "
                  f"~{(len(todo)-done) * el / done / 3600:.1f} h left", flush=True)
print(f"finished slice {SLICE}: {done} processed, {fails} failed")
