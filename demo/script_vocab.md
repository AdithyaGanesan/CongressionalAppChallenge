# Script vocabulary check (October 17 shoot, final scene)

Scene: a Deaf lead engineer assigns a new hearing intern's first task, calibrating the lab's sensors.
Checked against `models/sign_classes.json` (506 signs, loose matching: THANKYOU, THANK YOU and so on)
and the merged dictionary (English + STEM + `models/domain_words.json`).

## Whole signs

| Sign | Line | In the model? |
|---|---|---|
| HELLO | 1 | **missing** → fingerspell `H-I` ("Hi.") |
| TODAY | 2 | yes |
| UNDERSTAND | 4 | yes |
| GOOD | 5 | yes |
| THANK-YOU | 5 (backup) | **missing** → use `GREAT` ("Great.") |
| WHAT, FIRST, FINISH | extras | missing |
| HELP, NEXT | extras | yes |

Also in the model: NOW, DAY, WORK, GREAT, RIGHT, START, MEASURE, ENGINEER, TEST.

## Fingerspelled words

Simulated misread letters (`node tools/word_correction_test.mjs --list calibrate,sensors,torque,velocity --reps 100`),
share of tries corrected to the right word:

| Word | Clean | Light noise | Medium noise | J or Z? |
|---|---|---|---|---|
| CALIBRATE | 100% | 97% | 77% | no |
| SENSORS | 100% | 84% | 64% | no |
| TORQUE | 100% | 91% | 70% | no |
| VELOCITY | 100% | 94% | 78% | no |

The letter confusions in that test are a hand-written guess, not measured, so these compare words with each other; they are not real accuracy.

## Lines as STAT reads them

1. `H-I` → "Hi."
2. `TODAY` + `C-A-L-I-B-R-A-T-E` (one line: the sign, then the spelling) → "Today: calibrate", opens **Sensor calibration**
3. `S-E-N-S-O-R-S` (within 3 s of hands-down) → joins line 2: "Today: calibrate sensors"
4. `UNDERSTAND` → "Understand?"
5. `GOOD` → "Good."

## Diagrams

Sensor calibration (main), TORQUE and VELOCITY (backups), in `models/topic_diagrams.json`.

## Expected words (for filming)

Open STAT with `?expect=calibrate,sensors`. A spelling close to an expected word (weighted edit distance at most
half the word's length) becomes that word; a spelling that is already a real word is kept. The raw letters stay
on screen and an "Expecting: calibrate, sensors" note is shown, so the video stays honest about what was read.
Say in the video or description that the lesson's words were given to STAT in advance.

Simulated misreads, 100 tries each (`--expect calibrate,sensors`):

| Word | Light noise | Medium noise | Heavy noise |
|---|---|---|---|
| CALIBRATE (without → with) | 97% → 100% | 77% → 93% | 43% → 62% |
| SENSORS (without → with) | 84% → 99% | 64% → 89% | 37% → 56% |

Other words: 1 wrong snap in 800 tries (heavy noise); accuracy otherwise unchanged.
