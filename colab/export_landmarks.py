# STAT: package the ASL Citizen landmarks already in Drive so the Claude Code
# session can download them. Paste into a NEW, empty
# Colab notebook (Runtime: CPU is fine) and run this one cell.
# It asks once for Google permission, to mount Drive.
import os, time, zipfile
from google.colab import drive, auth
drive.mount("/content/drive")

SRC = "/content/drive/MyDrive/stat_landmarks_citizen"
files = sorted(f for f in os.listdir(SRC) if f.endswith(".npz") and ".tmp" not in f)
print(f"{len(files)} clips found")

ZIP = "/content/stat_landmarks_citizen.zip"
t0 = time.time()
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_STORED) as z:  # .npz files are already compressed
    for i, f in enumerate(files):
        z.write(os.path.join(SRC, f), f)
        if i % 1000 == 0:
            print(f"  {i}/{len(files)} added ({time.time() - t0:.0f}s)")
print(f"zip ready: {os.path.getsize(ZIP) / 1e6:.0f} MB")

# copy into Drive, then share it by hand: right-click it in Drive -> Share ->
# General access: "Anyone with the link" -> Copy link, and paste the link to Claude.
# (Sharing from code needs a sign-in pop-up that can hang, so it's done by hand.)
import shutil
shutil.copy(ZIP, "/content/drive/MyDrive/stat_landmarks_citizen.zip")
print("Copied to Drive as stat_landmarks_citizen.zip. Share it from Drive and paste the link to Claude.")
