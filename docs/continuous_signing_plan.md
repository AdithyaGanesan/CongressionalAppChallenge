# Continuous signing: plan (October 26, 2026)

## Why the current approach breaks

Today STAT handles a line of signing like this:

1. **Cut** the line where hand motion dips (`segmentClip`, motion valleys).
2. **Classify** each piece on its own with the isolated-sign model.
3. **Join** the guessed words and apply a few fixed phrase rules.

Each step loses accuracy:

- **Cutting by motion is fragile.** Real signing flows: one sign blends into the next
  (coarticulation), holds and pauses don't line up with sign boundaries, and a wrong cut
  hands the model half a sign or two signs at once.
- **The model never saw transitions.** It learned from clips containing exactly one clean sign,
  so the movement between signs looks like noise to it, and it still has to pick one of its
  1,017 signs for every piece.
- **A list of signs isn't English.** ASL has its own grammar (word order, use of space,
  facial grammar for questions and negation), so even perfectly recognized signs need real
  translation, not word-for-word joining.

## The plan: three stages, each usable on its own

### Stage 1: sign spotting instead of cutting (days)

Instead of cutting first, slide a window (about 1 second, every 0.25 s) over the whole line and
ask the model at every position: *which sign, or "no sign / transition"?* Then keep only
confident peaks and merge repeats.

- **Training data:** ASL Citizen clips as before, plus a new **"no sign"** class built from
  the hands rising and lowering at the start and end of clips, artificial transitions made by
  interpolating from the end of one clip to the start of another, and resting hands.
- **Why it helps:** no hard cuts to get wrong, and the model learns what *not* to label.
- **Cost:** about a day of training here, with the data we already have.

### Stage 2: continuous recognition, with whole lines in and the sign sequence out (1–2 weeks)

Train a sequence model with CTC (the same idea as speech recognition and as the fingerspelling
model being built in the other session) that reads a whole line and outputs the sequence of
signs directly. It learns boundaries and blending itself.

- **Training data:** synthetic sentences made by **stitching** ASL Citizen clips with smooth
  transitions (a known technique in recent research), with sign orders taken from real ASL gloss
  sequences, plus any real continuous recordings we collect (the professional, other signers).
- **Measure:** sign error rate on held-out signers and on real recorded phrases.

### Stage 3: real translation to English (weeks; the actual goal)

A model that goes from the landmarks of a signed sentence straight to an English sentence, with
no word list in between.

- **Data: ASL STEM Wiki (Microsoft).** 64,266 videos, **316 hours**, 37 signers, each video an ASL
  rendering of a sentence from one of 254 English Wikipedia **STEM** articles, paired with the
  English text. For a STEM translator this is the best public dataset there is.
- **Model:** the Stage 2 encoder plus a small English text decoder, trained end to end. Kept
  small (tens of millions of parameters at most, 8-bit) so it can still run in a phone browser.
- **Face:** ASL marks questions, negation and more on the face. Add a subset of MediaPipe face
  points (eyebrows, eyes, mouth), as the top Kaggle solutions did; check the phone frame rate.
- **Expect** partial, gist-level output at first: research systems on similar data are still in
  the high-teens to low-20s BLEU. It improves with data, model size and real-signer fine-tuning.
- **Compute is the bottleneck:**
  - Extracting landmarks from 316 hours of video would take roughly 110 hours on this 4-core
    container. It needs to be split across parallel jobs (several Colab notebooks and/or extra
    cloud sessions), or started on a subset (for example ~40 hours spread over all 37 signers).
  - Training a translation model needs a GPU: free Colab T4 sessions, a few hours per run.

## How every stage is measured

- **Held-out signers** from each dataset (never trained on).
- **Real phone recordings** from the ASL professional and other signers, made on the capture
  page, replayed through the exact app pipeline by an evaluation tool that reports hand-tracking
  quality, line cutting, sign accuracy and letter accuracy separately.
- **Stage 3:** BLEU/chrF on held-out STEM Wiki signers, plus ratings by Deaf signers. Automatic
  scores alone are not enough for sign language translation.

## Order of work

1. Evaluation tool for real recordings (this session).
2. Stage 1 spotting model with a "no sign" class (this session, existing data).
3. Start ASL STEM Wiki extraction on a subset, in parallel jobs.
4. Stage 2 stitched-sentence CTC model.
5. Stage 3 translation model on STEM Wiki, then fine-tune on real signers.
