# STAT: package the ASL Citizen landmarks already in Drive so the Claude Code
# session can download them, and print a share link. Paste into a NEW, empty
# Colab notebook (Runtime: CPU is fine) and run this one cell.
# It asks twice for Google permission: once to mount Drive, once to create the link.
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

# upload to Drive and make an "anyone with the link can view" link
auth.authenticate_user()
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
svc = build("drive", "v3")
f = svc.files().create(body={"name": "stat_landmarks_citizen.zip"},
                       media_body=MediaFileUpload(ZIP, mimetype="application/zip", resumable=True),
                       fields="id").execute()
svc.permissions().create(fileId=f["id"], body={"type": "anyone", "role": "reader"}).execute()
print("\nDone. Paste this link to Claude (and keep it private):")
print(f"https://drive.google.com/file/d/{f['id']}/view")
