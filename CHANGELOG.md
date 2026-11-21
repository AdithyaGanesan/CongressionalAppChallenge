# STAT changelog (work after the Congressional App Challenge video)

The video shows the version on `main` (submitted before Oct 26, 2026). Everything below was built
afterwards on the `next` branch. Numbers are measured on signers the models never saw in training.

## 2026-11-22: sign model v3 (`?model=v3`)

- Extracted hand and body landmarks from all of ASL Citizen (83,393 videos, up from ~36,000) and trained
  the same kind of model on it. Numbered variants of a sign (BABY1, BABY2) are merged, as in v2.
  v3 knows **2,301 signs** (v2: 1,017).
- Held-out signers (P15, P17, P18), clips of v2's 1,017 signs:

  | model | top-1 | top-5 |
  |---|---|---|
  | v2 | 76.7% | 92.2% |
  | v3, choosing among all 2,301 signs | 71.1% | 90.3% |
  | v3, limited to v2's 1,017 signs (`&vocab=v2`) | 78.8% | 93.2% |

  More data made the model better; the extra signs cost accuracy because many look alike
  (CUT / SCISSORS, RAT / HAMSTER). Over all 2,301 signs v3 gets 71.2% top-1, 89.8% top-5.
- Replayed through the app (same test recordings as above, `fs=v3`; note these recordings only use
  v2's signs, and `vocab=v2` was chosen after seeing v3's first app result on them):

  | test | v2 | v3 | v3 + `vocab=v2` |
  |---|---|---|---|
  | single signs, exactly right | 70% | 57% | 70% |
  | phrases with pauses, all signs found | 63% | 55% | 64% |
  | continuous phrases, signs found | 48% | 43% | 53% |
  | fingerspelled sentences, words found | 50% | 50% | 50% |

- Best settings so far: `?model=v3&vocab=v2&fs=v3`. Plain `?model=v3` recognizes more signs, less reliably.
  Full numbers: `training/report_v3.json`.

## 2026-11-21: whole-sign recognition in the app

- Found the main problem: the sign model alone gets ~79% of single signs right, but the app cut one sign
  into 2-4 pieces before reading it, so only 7% came out exactly right.
- New cutting: the app tries the possible cut points and keeps the split the sign model is most sure
  about (`segModel`). Also: a stretch that looks fingerspelled stays a sign when the sign model reads it
  confidently (`signOverSpell`), a longer pause ends a line (1.1 s), and lower-confidence signs are shown.
- Settings were tuned on one set of recordings and measured once on separate, untouched ones
  (replayed through the real app, signers the sign model never saw, `model=v2&fs=v3`):

  | test | before | after |
  |---|---|---|
  | single signs (150), exactly right | 7% | 70% |
  | phrases with pauses (40), all signs found | 43% | 63% |
  | continuous phrases, no pauses (40), signs found | 44% | 48% |
  | fingerspelled sentences (56), words found | 47% | 50% |

- Continuous signing without pauses is still weak (10% of phrases fully right). The old behavior is
  available with `?segModel=0&segMin=3&lineEndMs=700&signConfMin=0.4&signOverSpell=1`.

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
