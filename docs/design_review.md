# STAT design review (October 26, 2026)

Written after the ASL professional tested `?model=v2` and reported that both signs and
fingerspelling were still inaccurate, even when signing slowly.

## 1. Can the original idea work?

**The original goal, real-time translation of continuous ASL into written English, is an
unsolved research problem, not a student-project gap.**

- Research systems trained on large ASL video corpora score around the high teens to low 20s
  BLEU on the How2Sign benchmark, and papers disagree on how to even measure it.
- Google DeepMind announced SignGemma, an open ASL-to-English model, in May 2025. I could not
  find a model card or downloadable weights as of this review, so it can't be planned around.
- STAT's measured result (76% on single, cleanly recorded webcam signs from new signers) is a
  good *isolated-sign* number, but the real task has three extra failure points that the
  benchmark doesn't test:
  1. **Segmentation:** continuous signing has to be cut into signs before the model sees them.
  2. **Domain:** phone camera, lighting, framing and signer differ from the training videos.
  3. **Language:** ASL grammar uses the face, space and movement; a sign-by-sign word list is
     not English, and STAT doesn't read faces.

**Conclusion:** STAT will not become a general ASL interpreter. It can become a reliable tool for a
narrower job, and that job should be chosen with Deaf users.

## 2. What went wrong in the process

| Problem | Effect |
|---|---|
| Accuracy was measured on the training dataset's own style of video, not on real use | The 76% number didn't predict the professional's experience |
| Deaf users tested late, once, with no structured feedback | We know it failed but not *where* (tracking, cutting, signs or letters) |
| Fingerspelling trained on one person's hand photos | Letters fail for real signers; this was known and never fixed |
| No signer diversity data | ASL Citizen did not record race; we can't check for bias, and Black ASL variants may be missing |
| One-way design | All the communication effort sits on the Deaf person; a common Deaf-community criticism of sign-translation tech |
| Scope grew to fit a demo script | Fixes targeted the video scene, not general use |

## 3. Should we switch to an existing model?

**Partly. Reuse proven designs and data, keep our own pipeline and app.**

What exists:

- **Google's two Kaggle competitions (2023):** isolated signs (250 signs) and fingerspelling
  (3 million+ characters from 100+ Deaf signers, recorded on phone selfie cameras, in MediaPipe
  landmark format, the same input STAT uses). Winning solutions are public: 1D-CNN + transformer
  for signs (the same family as STAT v2), a speech-recognition-style sequence model for
  fingerspelling. The fingerspelling data matches our use case far better than the Kaggle photo
  alphabet we trained on.
- **SignGemma:** possibly the strongest option if it is released with usable weights and license,
  but likely too large for a phone browser, which would need a paid server and break the
  "everything free" rule.
- **Small GitHub projects:** mostly static alphabet classifiers; not better than what we have.

"Combining models" (ensembling) is a small gain. **Better and more diverse data is the big one.**

## 4. Recommended direction

Narrow the product to what current technology does well, and make it two-way:

1. **STEM fingerspelling assistant (core).** Technical terms usually have no standard sign and
   get fingerspelled. Reading fingerspelled terms well, correcting them with a STEM dictionary
   and showing the word plus a diagram is useful, measurable and achievable, *if* the letter
   model is rebuilt on multi-signer phone data.
2. **A limited, honest sign vocabulary.** Keep the 1,017-sign model, but show it as "suggested
   words" with confidence, not as a translation.
3. **The other direction: speech to text for the hearing person.** Browser speech recognition is
   mature, and a two-way conversation screen removes the one-way burden.
4. **Diagrams, not image generation.** Hand-checked Mermaid diagrams are accurate and free;
   free in-browser image generation is slow, heavy and can produce wrong science.
5. **Co-design and measure with Deaf users.** Every change is scored on recordings of real
   signers (several, including the professional and ideally Black signers), with held-out
   takes, before it ships.

## 5. Next steps (on the `next` branch; `main` stays as submitted)

1. **Measure first:** a recording session with the professional on the capture page (signs,
   phrases, letters, words), plus an offline evaluation that replays recordings through the
   exact app pipeline and reports tracking quality, cutting, signs and letters separately.
2. **Rebuild fingerspelling** as a sequence model trained on Google's fingerspelling dataset
   (needs a Kaggle API key saved as environment secrets and the competition rules accepted).
3. **Add Google's isolated-sign data** to the sign model (many signers, phone cameras),
   after checking its license.
4. **Add the hearing-side speech-to-text panel** and redesign the screen for conversation.
5. **Re-check SignGemma's status** and benchmark it if weights become available.

## Sources

- SignGemma announcement: <https://x.com/GoogleDeepMind/status/1927375853551235160>,
  <https://slator.com/google-invites-feedback-for-signgemma-a-new-ai-sign-language-translation-model/>
- Translation benchmarks: <https://arxiv.org/html/2603.13240>, <https://arxiv.org/pdf/2406.11049>,
  <https://arxiv.org/html/2607.09611>
- Kaggle fingerspelling competition and winners: <https://blog.tensorflow.org/2023/05/american-sign-language-fingerspelling-recognition.html>,
  <https://blogs.nvidia.com/blog/2023/10/23/nvidia-life-kaggle-ai-contest-sign-language-recognition/>
- Kaggle isolated-sign competition: <https://www.kaggle.com/competitions/asl-signs>
- ASL Citizen bias study (no race data recorded): <https://arxiv.org/abs/2410.05206>
- Deaf-led critique of sign-language AI research: <https://arxiv.org/pdf/2403.02563>
- Deaf community views on sign-translation tech: <https://cacm.acm.org/magazines/2018/12/232889-technology-for-the-deaf/fulltext>,
  <https://cssh.northeastern.edu/survey-finds-skepticism-of-sign-language-tech-among-deaf-and-hard-of-hearing-community/>
