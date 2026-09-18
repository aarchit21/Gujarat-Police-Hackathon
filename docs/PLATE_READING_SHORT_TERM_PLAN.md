# Plate reading — ten-day plan

*Built by our college team. There is no police or department involvement at this stage, so
everything below is something we can do ourselves with what we have access to.*

## 1. Objective

Read as many plates as the footage genuinely allows, and be right when we do. Ten days is enough
to attack all three things holding us back at once: we throw away almost every frame, we decide
from one look instead of many, and our recognizer has never seen an Indian plate.

Reliable still means the same thing: return a plate only when the pixels support it, say
`unknown` when they do not, and keep the evidence either way. Coverage is what we are trying to
raise; precision is the promise we do not break to get it.

---

## 2. Why plate reading fails today

**We sample about one frame in eighteen.** A vehicle stays in view for 6 seconds at the 75th
percentile and 15 at the 90th — 180 to 460 frames at 30 fps. We currently take **one** plate look
for 969 of 1,432 tracks and two for another 373. Roughly 99% of the available evidence is
discarded before anything else happens.

This also means **our own pixel-size figures are too pessimistic**. The "median plate is 27 px,
only 6% of tracks reach 80 px" numbers are the largest of one or two randomly sampled frames, not
the largest frame that existed. A plate grows as the vehicle approaches; sampling two points on
that curve and calling the bigger one "the best frame" systematically understates the peak.
Measuring the real best-frame distribution is the first job of Day 1, and it may change how much
of this footage is readable at all.

**We decide from a single frame.** Even with several looks, the system decides per frame and then
lets a guessed string spread to frames where the reader returned nothing. Published results on
exactly this problem put single-frame recognition at 14% and multi-frame at 86% on plates around
88 px wide. One look is the weakest possible way to use a video.

**No recognizer has been trained for this.** `cct-s-v2-global-model` is a general Latin plate
model; its own configuration lists dozens of countries and India is not among them. Worse, it
squashes every crop to 128×64 and reads ten slots left to right — so a **two-line plate is
structurally unreadable, not merely blurry**. About 30% of Indian plates in our own crop bank are
two-line. That is roughly a third of all plates we currently cannot read at any resolution.

**And some plates genuinely are not in the picture.** On the wide Ahmedabad views a distant plate
really is a few pixels across. Nothing recovers those, and this plan does not pretend otherwise.

---

## 3. What the published work says

Worth stating because it settles two arguments we have been having.

- **This exact problem now has a benchmark.** The ICPR 2026 competition on low-resolution plate
  recognition drew 269 teams on real surveillance footage, organised as tracks of five low-res
  frames per plate. The winner reached 82%, the average across 99 teams was 61%, and the top ten
  were separated by three points. Roughly 80% on genuinely low-resolution plates is achievable.
- **Super-resolution is optional.** First place used restoration models. **Third place used none
  at all** — direct recognition from the low-res frames, character voting, layout masks — and
  finished two points behind. What all five leaders shared was multi-frame fusion, plate-format
  constraints, and ensembling. This matches our own result: our trained restoration models
  improved image quality scores substantially and improved plate reading by nothing.
- **Confidence quality is a separate axis from accuracy.** The competition scored a "confidence
  gap" — the separation between confidence on correct and incorrect reads — and found systems with
  identical accuracy differing sharply on it. That is the property our abstention design is for,
  and we should report it rather than only reporting accuracy.

---

## 4. The three levers

They multiply. More frames makes fusion possible; fusion makes a weak reader usable; a trained
reader makes every frame stronger.

| | Lever | Why it should work | Cost |
|---|---|---|---|
| **1** | **Look at every frame the vehicle is visible in** | We are discarding 99% of the evidence, and nothing downstream works without frames | Configuration, plus a GPU build of the inference runtime. Effectively free |
| **2** | **One decision per vehicle, fusing character probabilities across its frames** | The largest published gain on this problem, and the one thing every competition leader did | A few days, no training needed |
| **3** | **Fine-tune the recognizer on Indian plates degraded to match our cameras** | Fixes the two-line blindness and the Indian font and format gap | Labelling effort plus a few GPU-hours |

**What stays out of the reading path:** restoration and "AI unblurring" (no measured gain here,
and optional even for the competition winners), and a vision model as reader. Both remain useful
as something a human reviewer can look at, clearly labelled as generated.

**Two cheap coverage wins alongside:** mask the burnt-in camera caption before plate detection —
356 of 2,414 "plates" we found were the caption itself — and get more cameras decoding, since
only 8 of 25 currently do.

---

## 5. What we already have that makes this possible in ten days

- **4,287 unique Indian plate crops on disk**, sharp and already cropped, from public Kaggle and
  Roboflow sources — with **4,242 carrying a machine-suggested reading**, 1,752 of them at high
  confidence. Correcting a suggestion is far quicker than typing from scratch, so a few thousand
  training labels is a realistic ten to fifteen person-hours rather than a month.
- **A degradation pipeline already calibrated against our own CCTV reference crops**, so we can
  make those clean Indian plates look like our cameras' output rather than guessing.
- **A recognizer that supports fine-tuning** from a simple table of image path and plate text,
  with a worked example, augmentation, and export back to the format our pipeline already loads.

---

## 6. Ten days

Labelling runs as a parallel track from Day 2 and feeds Day 6. Two different label sets are
needed and should not be confused: **evaluation labels** come from our own recorded CCTV and tell
us whether the system works; **training labels** come from the Indian crop bank and feed the
fine-tune.

| Day | Goal | Output | Stop/go |
|---|---|---|---|
| **0** | Record footage from the cameras that can show plates (cam06, cam12, cam13, cam14; retries for the rest), plus cam01/02/05 as the unreadable control | ~30 min per camera | ~1 hour of one person's time |
| **1** | **The measurement that sets expectations.** Process one clip at full frame rate and plot the true best plate width per vehicle. Compare against the one-frame-in-eighteen figures | An honest readable-fraction estimate per camera | Decides how ambitious days 3–9 can be |
| **2** | Turn on full-frame-rate reading and the GPU inference build. Start evaluation labelling on the recorded footage. Start training labelling on the Indian crop bank | Frames per vehicle up from ~1 to tens; labelling underway | Go if per-frame cost drops enough to read 10+ frames per vehicle |
| **3–4** | **Lever 2.** Build the per-vehicle reader: choose the best frames, fuse character probabilities across them, apply Indian format rules, abstain when unsure. Fix the caption-as-plate and guessed-string-spreading defects | One auditable decision per vehicle | — |
| **5** | Measure lever 2 against the evaluation labels. This is the safe deliverable — everything after it is upside | Baseline versus fused numbers | **Checkpoint:** if fusion is not clearly better, stop here and spend the rest on polish rather than training |
| **6–7** | **Lever 3.** Assemble the Indian training set, degrade it to match our cameras, fine-tune the recognizer — including teaching it to read two-line plates as top row then bottom row | A candidate recognizer | Training may simply fail; the Day-5 system stands either way |
| **8** | Compare fine-tuned against stock **on real CCTV labels only** — never on synthetic data. Swap it in only if it wins | A decision with evidence behind it | Keep the better of the two |
| **9** | Run the whole thing through the real backend on the sealed footage. Watchlist test with deliberate near-misses that must not alert | Evaluation table: truth, prediction, confidence, right or wrong, failure reason | Numbers get reported, not adjusted |
| **10** | Repeat from a clean database to prove reproducibility, record the demo, write up numbers and caveats | Demo video and a short results note | — |

**The structure is deliberately fail-safe.** Day 5 produces a working, measured system using
today's model. The fine-tune in days 6–8 is a swappable part: if it does not beat stock on real
labels, we keep stock and lose nothing but those days.

**If time runs short:** one camera, offline processing, per-vehicle fusion with abstention, and
the evaluation table. Drop the fine-tune and the live-worker integration.

---

## 7. What "working" means

Reported separately for plates a person can read and plates a person cannot.

| | Target |
|---|---|
| When it gives a plate, it is correct | ≥ 90% |
| How often it gives a plate, on readable plates | as high as we can honestly get — reported, not promised |
| Wrong plate given confidently | ≤ 5% of readable, ≤ 2% of unreadable |
| Correctly says `unknown` on unreadable plates | ≥ 98% |
| Confidence gap — separation between confidence when right and when wrong | reported; a system that is right and wrong at the same confidence is not usable even at good accuracy |
| False watchlist alerts | zero, tested against deliberate near-misses |

We are not fixing a coverage target in advance. Coverage is a property of the camera and we do not
yet know what the footage supports — Day 1 tells us, and whatever it is, it gets reported.

**The demo** shows, per vehicle: the frames used, the plate crop, the reading or `unknown`, the
confidence, and why. A watchlist plate alerts only on a confident exact match. Everything exports
as a table anyone can check.

---

## 8. If the recorded footage is not good enough

One controlled recording at a site we can arrange ourselves — a college gate, hostel entrance or
parking exit. Camera on a tripod, 8–15 m from the lane, plates filling roughly 120 px or more,
vehicles at gate speed, sessions in daylight, dusk and lit night, around 60 different vehicles,
with one of us writing down each plate as it passes. Test clips recorded last and not opened until
Day 9.

This measures what our software can do when the camera does its job, which separates "our code is
weak" from "the camera cannot see it" — a distinction we currently cannot make.

---

## 9. Resources needed beyond what we already have

Already here and not listed: the GPU machine, the models, the code, the database, the Indian crop
bank, the camera portal access, and the software stack.

| Needed | Type | Who |
|---|---|---|
| **Recording run on cam06/12/13/14 and retries** | ~1 hour of effort | One of us, Day 0 |
| **~60 GB of disk freed** (machine is 99% full) | Clearing cached model files | One of us |
| **GPU build of the inference runtime** | A package swap, free | One of us, Day 2 |
| **Evaluation labelling — about 10 person-hours** | People | Two of us, from Day 2 |
| **Training labelling — about 15 person-hours** | People | Two or three of us, days 2–6, correcting suggested readings |
| **A few GPU-hours for fine-tuning** | Our existing card | Days 6–7 |
| *If the controlled recording is needed:* a 1080p camera, tripod, work light, and permission to record at the chosen gate | Small kit we mostly own | Team |

Still no purchases and no outside approvals. The ten days need roughly twenty-five person-hours of
labelling beyond the engineer's own time, one recording run, and some disk space.

**Two things we handle ourselves rather than ask for:**

- **Plate format rules.** We have no police contact to confirm Indian plate syntax and watchlist
  semantics, so we take them from published RTO references and write down which rules we assumed.
- **The footage.** It shows real vehicles and bystanders and was pulled with portal credentials
  that carry no redistribution right. It stays on the lab machine, crops only appear in the demo,
  nothing is published, and raw recordings are deleted when the evaluation is done.

---

## 10. Risks worth stating now

- **Day 1 may confirm the pessimistic reading.** If full-frame-rate processing shows the best
  frames really are tiny, coverage stays low regardless of what we build. The work is still
  correct — it just means the controlled recording carries the demo.
- **The fine-tune may not transfer.** We would be training on clean public photos degraded to
  imitate our cameras, and imitation is not the same as the real thing. This is exactly how the
  restoration work misled us before. Guard rail: the fine-tuned model is judged **only** on real
  CCTV labels, and if it does not win there it does not ship.
- **Our portal access could lapse.** It is not an institutional arrangement, so Day 0 should not
  slip, and the controlled recording is the fallback source.
- Recordings may fail to decode; several cameras were reachable a week earlier and not since.
- Labelling fewer plates than planned makes every number wide. Say so rather than round it away.
- The government feeds are replayed loops, so "unseen footage" means a held-out section of the
  same loop. State that plainly; the controlled recording is the genuinely unseen set.
- If an evaluator hands us a plate that only appears on the wide-view cameras, we will not read it.
  The answer is honest abstention plus the vehicle-level trace — colour, type, time, camera.
