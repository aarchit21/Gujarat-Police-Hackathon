# POC vehicle-attribute evaluation — frozen artifacts

Written for: whoever picks this work up next, including an auditor checking
whether the deployed safeguards match the measurement that justified them.

These files are frozen evidence for the accuracy claims in the root `README.md`
and in `POC_ACCURACY` (`app/services/vehicle_observations.py`). Do not
regenerate them in place; a new measurement belongs in a new dated folder.

| File | What it is |
|---|---|
| `tracks_labels.frozen.jsonl` | 59 ground-truth track labels |
| `predictions_offline.frozen.jsonl` | 35 predictions, `cam01_smoke.mp4` segment |
| `predictions_final.frozen.jsonl` | 26 predictions, `cam01_final.mp4` segment |
| `evaluation_primary.frozen.json` | Headline metrics, both segments |
| `evaluation_policy_detector.frozen.json` | Conflict-policy comparison, offline segment only |

## Headline result

| | Scored tracks | Coverage | Selective precision |
|---|---|---|---|
| Vehicle type | 28 | 64% | **50%** |
| Vehicle colour | 42 | 71% | **83%** |

On the 17 tracks whose true class the model cannot represent (`auto_rickshaw`,
`suv`) it correctly abstained 82% of the time; 3 were false attributions.

**Vehicle type is not fit for operational use.** Only 2 of 15 buses were called
`bus` and 6 became `truck`. Critically, precision *fell* from 50% to 30% as the
confidence floor rose from 0.60 to 0.95 — the errors are confident and
systematic, so no threshold reaches the 95% objective. That measurement is why
`type_deployment_gate()` fails closed and the operational `vehicle_type` is
suppressed to `unknown`.

Colour's dominant error is `white → red` (4 cases), from brake lights and red
signage at night. `silver` is unreachable, so both true silver cars were missed.

## Method

Labels were produced **blind**: `scripts/label_tracks.py` builds contact sheets
from crop files and filenames only and never opens `observations.jsonl`, so no
prediction could reach the annotator. Ground truth uses the full canonical
vocabulary, not the model's four classes — labelling an auto-rickshaw as `car`
to fit the model would have hidden the exact gap being measured.

14 of 59 crops were too dark or blurred to call and are excluded as `unclear`.
The two segments are disjoint captures, giving a split by video.

Evaluation scores the model's **raw candidate** (`type_candidate`), never the
suppressed operational `vehicle_type` — see `_predicted_value()` in
`scripts/vehicle_pipeline.py`. Scoring the suppressed value would measure the
safeguard rather than the model and report a flattering 0% error rate.

Reproduce:

```bash
python scripts/vehicle_pipeline.py --mode evaluate \
    --manifest docs/poc_evaluation/tracks_labels.frozen.jsonl \
    --predictions docs/poc_evaluation/predictions_offline.frozen.jsonl \
                  docs/poc_evaluation/predictions_final.frozen.jsonl
```

## Two caveats that limit these numbers

**Single, non-independent annotator.** The labels were produced by the same
assistant that wrote the code being evaluated. Blind, but not an independent
human annotator and not double-checked. Treat the figures as the right order of
magnitude, not as certified accuracy.

**Small sample.** 28 scored type tracks and 42 colour tracks, one camera, one
night, one lighting condition. Nothing here says anything about daylight
performance, other cameras, or other cities.

## Discarded measurement — contamination note

A first conflict-policy comparison across **both** segments was computed and
**discarded**. It reported `detector` 52.2% vs `abstain` 50.0%.

It was invalid. The `final` labels came from a live RTSP run, but the comparison
re-ran the pipeline against that run's *recording*. The recording contains every
decoded frame while the live run processed every third, so replaying it produced
a different frame subset and different ByteTrack ids: **only 11 of 24 `final`
tracks matched, and 9 labelled tracks were silently dropped** from the scoring.

The comparison was redone on the `offline` segment alone, which is deterministic
from its mp4 and matched 35/35 tracks in every configuration:

| `VATTR_CONFLICT_POLICY` | Coverage | Selective precision |
|---|---|---|
| `detector` (default) | 90.0% | 50.0% |
| `abstain` | 55.0% | 36.4% |
| `classifier` | 85.0% | 29.4% |

This reversed an earlier design assumption: `abstain` had been chosen on the
reasoning that neither unvalidated model should win a conflict, but it was
discarding the cases the COCO detector got right. `detector` dominates on both
axes and is now the default.

**It is still only 50% precise.** Winning a three-way comparison does not make
it a production classifier, and `evaluate` counts unmatched tracks explicitly so
this class of error is visible rather than silent.
