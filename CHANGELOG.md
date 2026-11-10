# STAT changelog (work after the Congressional App Challenge video)

The video shows the version on `main` (submitted before Oct 26, 2026). Everything below was built
afterwards on the `next` branch. Numbers are measured on signers the models never saw in training.

## 2026-11-10: fingerspelling model v3 (`?fs=v3`)

- A larger fingerspelling model, trained on all of Google's ASL Fingerspelling data (Kaggle, CC BY 4.0,
  100+ signers) at 10-30 frames per second. The app reads frames at up to 30 fps with it.
- Held-out signers, clips where the hand is tracked in 60%+ of frames:
  - letters right (1 - character error rate): v2 86.5% (15 fps) -> v3 88.7% (15 fps), 90.0% (30 fps)
  - English words right with whole-word scoring: v2 71% -> v3 80% (15 fps), 85% (30 fps)
- Full numbers: `training/report_fs_v3.json`. v2 is unchanged and still available with `?fs=v2`.

## 2026-11: earlier work on `next`

- Whole-word scoring for fingerspelling: each spelled word is matched against the dictionary using the
  model's letter probabilities, not just its best guess per letter (v2: 58% -> 71% of words).
- Sign model v2.1 (`?model=v2`): 1,017 signs.
- Hand-tracking meter, a recording page (`capture.html`) and tools to replay recordings through the app.
