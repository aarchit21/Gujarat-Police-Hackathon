# Plate reading — long-term plan

*Built by our college team. There is no police or department involvement at this stage. This plan
is therefore split into what **we can build and prove ourselves**, and what a real deployment
would additionally need — stated as a specification for whoever eventually runs it, not as
resources we are asking for.*

## 1. Objective

Build a vehicle intelligence pipeline that reads Indian number plates from real CCTV with a
confidence that can be defended — and that says `unknown` rather than guess. Alongside the plate:
vehicle type, colour, camera, time, and the evidence for each, so a vehicle can be traced across
cameras and an alert can be trusted.

The single missing asset is a **recognizer trained on Indian plates as they actually appear on
CCTV**. Everything else in this plan exists to make that model possible, to deploy it only where
it can work, and to keep it honest afterwards.

Our realistic goal is to prove the approach end to end on the cameras we can reach, with numbers
that hold up — not to run a statewide system.

---

## 2. The core gap

The recognizer in use today is a general Latin plate model. India is not in its training regions.
It has not seen Indian formats, yellow commercial plates, or Indian roads at night. It was never
going to work well here, and no threshold tuning changes that.

One part of the gap is not a matter of degree. The model squashes every crop to a fixed
128×64 and reads ten character slots left to right, so a **two-line plate is structurally
unreadable** — not blurry, unreadable by construction. Around 30% of the Indian plates in our own
crop bank are two-line. That is roughly a third of all plates that no amount of better capture
would fix, and only training fixes it.

Training a replacement needs labelled Indian plates. We have **none verified from a real camera** —
the project contains no confirmed plate number attached to a CCTV image. But we do have **4,287
unique Indian plate crops** already on disk from public sources, most with a machine-suggested
reading that a person can correct quickly, plus a degradation pipeline already calibrated against
our own CCTV references. The starting material for a first fine-tune is therefore in hand; what is
missing is the labelling effort and the real-camera evaluation set to judge the result.

But training alone is not enough, because of a second finding: on the cameras surveyed, plates
are typically 27 pixels wide. A perfect model cannot read characters that were never captured.
**So camera quality has to be assessed before model work** — otherwise we train a good model,
deploy it on cameras that can never use it, and blame the model.

That gives the order of work.

---

## 3. The plan, in order

**Step 1 — Work out which cameras can do this at all.**
Sample each camera we can reach for a day and measure how large plates actually appear, how much
glare and blur there is, and whether the timestamp can be trusted. Sort cameras into three groups:
those that can support plate reading, those that can partly, and those that cannot and should be
used for vehicle detection only. This is measurement, not modelling. It is cheap, it is the most
useful single thing this project can produce, and no one appears to have done it — the
80,000-camera figure quoted for the state is a camera count, not a count of cameras that can read
a plate.

**Step 2 — Collect and label real data.**
From the cameras that qualify, plus our own controlled recordings: day and night, different
weather, all plate types, front and rear. Two labellers per image with a third to settle
disagreements, and explicit labels for plates that are unreadable — those teach the model when to
stay silent. Split the data by camera and by day so the model is never tested on a camera it
trained on. This is the slow part and the part that matters most.

**Step 3 — Train the models that are actually missing.**
A plate detector fine-tuned on Indian scenes, trained with the things that currently fool it —
advertising boards, tail-lamps, the burnt-in camera caption. A recognizer fine-tuned on Indian
plates, reading two-line layouts as top row then bottom row, and reporting per-character
confidence. Our plates are often badly skewed as well as small, so the recognizer should correct
geometry internally rather than rely on a separate rectification step. Clean training crops should
be degraded to match each camera class rather than used as they are — imitating our own cameras is
what makes the training transfer, and a competition-placing team used exactly this trick. Vehicle
type and colour models fine-tuned for Indian classes, since the current type model sits at 50%
precision and has no class for an auto-rickshaw.

The guard rail that matters: a fine-tuned model is judged **only on labels from real cameras**,
never on the synthetic data it was trained against. Synthetic scores are how the restoration work
misled this project once already.

**Step 4 — Decide once per vehicle, not once per frame.**
Pick the clearest views of each vehicle, read them together, check the result against Indian plate
format rules, and emit a plate only when the combined evidence passes a calibrated threshold.
Otherwise `unknown`, with a reason.

This is the largest single lever, not a refinement. Published work on exactly this problem reports
single-frame recognition at 14% against 86% for the same plates read across frames, and every
leading entry in the 2026 low-resolution plate competition used multi-frame fusion. It also
depends on a prerequisite we currently fail: the pipeline must actually look at most of the frames
a vehicle is visible in. We sample roughly one frame in eighteen, so there is usually nothing to
fuse.

**Step 5 — Keep a person in the loop, and let it feed back.**
Uncertain reads go to a review queue with their evidence. Corrections become new training labels.
This is how accuracy improves after deployment instead of decaying, and it is how a small team
grows a dataset without a large annotation budget.

**Step 6 — Watch it over time.**
Per-camera tracking: how often it answers, how often a reviewer overturns it, whether plate sizes
have changed because a camera moved or re-zoomed. A camera that drifts gets re-assessed. A small
random sample reviewed blind each week keeps an honest running accuracy figure rather than a
number we quoted once.

---

## 4. Choices and reasoning

| Decision | Reasoning |
|---|---|
| Fine-tune open models rather than use a vendor ANPR | Plate reading must be reproducible and auditable; a closed vendor score cannot be explained or debugged per camera. It is also the only option we can actually afford and inspect |
| Move off SQLite to PostgreSQL on a server | We already hit database write-lock failures with four cameras running at once. The code is written for this — the database URL is configurable and a PostgreSQL schema is already in the repo — so it is a switch, not a rewrite. A modest lab server or a small cloud VM is enough for our scale |
| Design for processing near the camera, prove it on one machine | Sending metadata and a few crops per vehicle is roughly 0.1–0.5 Mbps per camera against 2–4 Mbps for video. We cannot deploy edge boxes, but we can measure per-stream cost on our GPU and show the arithmetic honestly |
| Keep the abstention behaviour as a permanent feature | A system that always answers produces confident wrong plates. Coverage is a property of the camera; precision is the promise we make |
| Report a confidence gap, not just accuracy | The 2026 low-resolution competition scored the separation between confidence on correct and incorrect reads, and found systems with identical accuracy differing sharply on it. A model that is right and wrong at the same confidence cannot support a review queue, however good its headline number |
| Store crops and metadata, not whole video | Already the project's position, and it is what makes the storage claim credible |
| Prefer permissively licensed models | The current detector stack is AGPL. Fine for a student project, a real problem if this is ever adopted as a networked service. Equivalent Apache/MIT models exist and we should move to them before that question is asked |
| Enhancement, "unblurring" and generative restoration stay out of the reading path | We trained these ourselves — PSNR improved from 14.3 to 20.6 dB with no measured gain on a single real plate read. Independently, the team placing third in the 2026 low-resolution competition used no restoration at all and finished two points behind the winner, so it is optional even at the top of the field. Below about 64 px these models invent characters. They may be shown to a human reviewer, clearly labelled as generated, and never used as evidence |
| A vision language model may check, never decide | It can be asked whether it agrees with a reading, and logged as a second opinion. It must not produce a plate number or trigger an alert — it will fabricate a plausible plate when shown none |
| Cameras that cannot see plates are not failures | They still give vehicle type, colour, time and movement, which supports a trace even when the number is unreadable |

---

## 5. Stages

| Stage | Roughly | What it proves | Who does it |
|---|---|---|---|
| **Prototype** | 1–2 months | Cameras assessed; first labelled Indian CCTV dataset; a fine-tuned model beating the off-the-shelf one on held-out cameras | Us |
| **Extended trial** | 3–4 months | The full pipeline running for weeks on the cameras we can reach, on a proper server, with the review queue and monitoring in use; accuracy checked by blind audit rather than our own reporting | Us |
| **Deployment** | — | District roll-out on qualified cameras, versioned models, rollback, per-camera health | Only possible with an official engagement — §7 |

A model version replaces the previous one only if, on held-out cameras it has never seen, it is
more accurate, does not answer confidently when wrong more often, and does not regress on the
things that currently fool the detector.

---

## 6. What we need beyond what we already have

Already available: the GPU workstation, the existing models, the codebase, the portal access to
the current camera catalogue, and the software stack.

| Needed | Type | Notes |
|---|---|---|
| **A server with PostgreSQL** | A lab machine or small cloud VM | Replaces SQLite, which already failed under four concurrent cameras. Configuration change, not a rewrite |
| **Storage — a few TB** | Hardware | The present machine has 17 GB free; footage and training data will not fit |
| **GPU time for training** | Scheduling, or a second card | One shared GPU is workable but will be the bottleneck once fine-tuning starts |
| **Labelling effort — the largest recurring cost** | People, mostly our own time | Target a few thousand labelled plates to begin with; classmates can help if the labelling interface is simple |
| **An annotation tool, self-hosted** | Free software on the server above | |
| **Our own camera and tripod for controlled recordings** | Small kit | Lets us build data that does not depend on portal access continuing |
| **A written data-handling policy we actually follow** | Our own discipline | Real vehicles and bystanders are in this footage; keep it local, publish crops only, delete raw video after evaluation. If the college has an ethics process, use it |

Not on this list, deliberately: edge hardware, district servers, annotation contractors, hiring.
Those belong to a deployment, not to us.

---

## 7. What a real deployment would additionally need

Recorded here so the work is useful to whoever picks it up, and so nobody mistakes it for our
plan. None of it is available to us now.

- **Official feed access** across a representative set of cameras, and camera details — mount
  height, angle, lens, zoom, encoding — which are blank in our records and cannot be measured
  from a video stream alone.
- **Authority and budget to re-aim, re-zoom or add cameras** at priority junctions. This is the
  only remedy for cameras where plates are too small, and it is field work rather than
  engineering. Where it is refused, those sites stay vehicle-only, and our survey shows which.
- **Clock synchronisation across cameras.** Timestamps are currently unverifiable; one feed
  carried a three-month-old on-screen clock.
- **A privacy assessment, legal basis and retention policy**, signed by someone with the standing
  to sign it.
- **Authorised watchlist data.** Everything to date is synthetic, and should stay synthetic until
  it is not.
- **Processing capacity at the edge or district level**, sized from the camera survey — which is
  why the survey comes first.
- **Staff**: engineers to run it, annotators, police reviewers who can confirm plate formats and
  judge alerts, and technicians per district.
- **A licence review** of the AGPL components before procurement.

---

## 8. Decisions for us

1. Which cameras we can still reach, and how long our portal access is likely to last.
2. Where the server and storage come from — a lab machine or a small cloud budget.
3. How much labelling we commit to, and whether classmates help.
4. Whether we move to permissively licensed models now or after the prototype.
5. Whether a locally run vision model is used as a logged second opinion, with no decision power.

---

## 9. What success looks like

At the end of our extended trial, for every camera we can reach, we can state: how often the
system reads a plate, how often it is right when it does, how often it correctly stays silent, how
well separated its confidence is on right and wrong answers, and how many false alerts it raised —
each verified by blind audit rather than asserted. Where a camera cannot support plate reading, we
can say so with the measurement that proves it, and still provide the vehicle trace.

For scale: the best entries in the 2026 low-resolution plate competition reached about 82% on real
degraded surveillance plates, with an average of 61% across ninety-nine teams. That is the band a
serious result sits in — not the 99% figures quoted in product literature, which come from
toll-booth captures where the plate fills the frame.

That result is worth something to a department whether or not they ever adopt our code, because
the camera survey and the labelled dataset outlive the prototype. A higher headline accuracy with
none of those properties would not be.

---

## 10. References consulted

Not a literature review — only the sources that changed a decision in this plan.

- **ICPR 2026 Competition on Low-Resolution License Plate Recognition** (arXiv 2604.22506).
  269 teams on real degraded surveillance plates, organised as multi-frame tracks. Winner 82%,
  mean 61%. Establishes the achievable band, shows restoration is optional, and introduces the
  confidence-gap metric.
- **MF-LPR²: Multi-Frame License Plate Restoration and Recognition** (arXiv 2508.14797).
  86% multi-frame against 14% single-frame at roughly 88 px plate width, fusing aligned frames by
  averaging rather than generating, so it cannot invent characters.
- **fast-plate-ocr** (ankandrew.github.io/fast-plate-ocr). The recognizer we already run;
  documents fine-tuning from a table of image path and plate text, and exports back to the format
  our pipeline loads.
- **Indian Licence Plate Dataset in the Wild** (arXiv 2111.06054). 16,192 images, 21,683 plates.
  Records that models trained on Chinese and Brazilian data transfer poorly to Indian plates. The
  dataset itself is **not released** ("legalities involved in making Indian road data public"), so
  public Kaggle and Roboflow crops plus our own footage remain the practical sources.
- **LPSRGAN / LPDGAN restoration line.** The approach this project already implemented and
  measured; recorded here only to explain why it sits outside the reading path.
