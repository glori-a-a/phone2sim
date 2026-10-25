# Recorded-data results — 22 October 2026

Sixteen applicant-declared independent phone recordings were retained after tracking QA. A fixed source-recording split uses 12 training and four test recordings. Learned and geometric feedback each pass all 36 simulation trials; replay passes none. This supports feedback in this small experiment, not a success-rate advantage for learning.

Artifacts: `runs/real_20261022_v1/{manifest.json,prediction_metrics.json,rollouts.json,summary.json,report.html}` and all `trace_*.npz`. Processed data: `data/processed/collection_20261021_v1/`.

## Data and tracking

The applicant confirmed personally recording 16 independent takes on 21 October 2026 at approximately 23:00; timezone unspecified. These are collector declarations, not independently verified authorship. All takes share one session, scene and object setup.

The measured four-cross rectangle is 400 × 300 mm; measured tool diameter is 23.5 mm. Red-object diameter is estimated as 45 mm from the median of its first five calibrated silhouette measurements per recording, rounded to 1 mm. Initial estimates range from 41.8 to 47.2 mm. This is an unvalidated image estimate of the top silhouette, not a measured contact diameter.

Original videos are preserved with SHA-256 hashes. Proxies are 1280 × 720 at 30 fps; portrait takes 01, 03 and 16 are rotated 90° counterclockwise. Tracking samples at 10 Hz without temporal smoothing. Background ORB/RANSAC homographies compensate for camera motion; SIFT is a fallback with the same geometric thresholds. Acceptance requires at least 18 inliers, spatial coverage, low median reprojection error and a plausible table quadrilateral. Colour, metric size and aspect filters reject distractors.

There are 2,060 sampled times, with 2,056 jointly observed positions (99.81%). Four isolated missing times occur in takes 04 and 15: one rejected camera fit, three marker detection failures. Short linear fills maintain continuous replay paths; transitions touching fills are excluded from fitting and prediction scoring. Long gaps would be rejected. Tracking QA covered sampled overlays, suspect edge frames and all-sample continuity diagnostics, rather than an exhaustive frame-by-frame review. All recordings were retained; no synthetic transitions were added.

Before fitting, a detector error near the table boundary in take 08 confused a finger with the red object. Expanding the detection region by 30 mm and checking marker size/aspect corrected it. Final maximum one-sample displacement is 17.0 mm. Processing-source and raw-track hashes are stored in episode metadata.

## Fixed evaluation protocol

Seed 7 splits original source hashes into 12 training recordings (1,458 fully observed transitions) and four test recordings (558 transitions, 204 moving by the >1 mm/sample criterion). Test IDs are **phone_01, phone_03, phone_07, phone_16**. Source groups do not overlap. Existing model/controller parameters were retained; the test split was not used to retune them.

Test paths are near-straight pushes: right, right, up and left. Curved takes happen to be in training, so these results do not evaluate unseen curves or a held-out downward path. Same-session splitting does not establish cross-session, object or camera generalization. Tracking QA used all footage. The diameter estimate pools initial static frames across all recordings as shared hardware calibration; preprocessing is therefore not described as strictly train-only.

## Prediction

Mean Euclidean displacement errors relative to extracted video trajectories:

| Predictor | All 558 transitions | 204 moving transitions |
|---|---:|---:|
| Learned 12-neighbour state dynamics | 0.60 mm | 0.85 mm |
| Same model, shuffled training actions | 1.01 mm | 1.57 mm |
| Stationary object | 1.27 mm | 2.82 mm |
| Constant object velocity | 1.10 mm | 1.78 mm |
| Copy observed tool displacement | 0.89 mm | 0.90 mm |

The model improves over action shuffling and stationary/velocity baselines. Its moving-sample advantage over tool-copy is only 0.055 mm, about 6%; no statistical significance is claimed. Ten-step conditional endpoint error is 2.03 mm, versus 6.17 mm for shuffled actions. That test receives observed future tool positions; it is not autonomous visual prediction or long-horizon planning.

Both predictions and targets derive from the same tracking pipeline. Correlated camera/marker error, marker height, reference-corner placement and lens distortion lack independent ground truth. Sub-millimetre prediction error does not establish sub-millimetre physical accuracy.

## Simulation

Four test paths × three reset seeds × three conditions × three methods = **108 rollouts**. All methods receive identical references and resets. Success requires final centre error <25 mm and at least 95% of sequential checkpoints reached. Conditions are sliding friction 0.4, a +0.16 m/s y-velocity impulse, and friction increased to 0.7.

| Condition | Replay | Geometry feedback | Learned feedback |
|---|---:|---:|---:|
| Nominal success | 0/12 | 12/12 | 12/12 |
| Impulse success | 0/12 | 12/12 | 12/12 |
| Friction success | 0/12 | 12/12 | 12/12 |
| Mean final error, all conditions | 84.64 mm | 10.64 mm | 7.36 mm |

Learning has lower mean endpoint error here, but geometric feedback also solves every case. Both share a substantial hand-written contact approach planner, checkpoints, inverse kinematics and exact simulator state. Learning only ranks in-contact actions. The two-joint planar arm is a different embodiment from the human tool, not a simulated humanoid.

All 36 impulse trials applied the disturbance. Early-finishing feedback runs wait for the scheduled impulse. This disturbance is mild relative to the 25 mm tolerance; success is not evidence of broad robustness. Wilson intervals are descriptive: rollouts sharing a reference are correlated and there are only four independent test recordings. Median controller timings include waiting/repositioning and are not pure inference benchmarks.

## Failures and limitations

Replay fails all 36 cases, including nominal physics. The preselected example phone_01, impulse seed 0, completes 45% of checkpoints and ends 94.3 mm from the goal. Its trace shows the puck stops progressing while the tool continues; feedback repositions and resumes contact. Initial object offsets, contact differences and projected marker/contact-centre mismatch are plausible causes, not separately isolated mechanisms. Replay failure does not prove learning is needed: geometry also recovers.

The three `*_impulse.mp4` videos use the same preselected recording/seed/condition, not individually selected best cases. Average actual-vs-requested tool displacement error is 0.14 mm for replay, 1.88 mm for geometry, and 2.54 mm for learned control. Observed human displacement at fitting time differs from requested servo displacement at control time. Simulator mass/friction are assumed, not identified from videos.

This is a small action-conditioned state model, not a pretrained VLA, generative video model or foundation-model contribution. It only addresses slow planar pushing of one approximately round object. It does not learn the full policy, language, grasping, stacking, general robot perception or real-robot deployment.

## Verification and next experiment

Twenty tests pass, covering camera compensation/rejection, SIFT fallback, edge-marker distractors, interpolation exclusion, geometry persistence, grouped splits, contact and provenance. `pip check` finds no broken requirements. The real-data audit passes dataset-hash, split and completed-evaluation checks; it does not certify authorship, recruiting eligibility or public-repository access. Versions are in `requirements-lock.txt`.

The next useful collection is a separate session with measured red contact diameter, static-marker calibration repetitions, and held-out curved/downward paths. Freeze this model and results. Predeclare stronger lateral disturbances/contact loss and compare both feedback methods, rather than retuning against the existing test set. A shuffled-model controller ablation would help separate learned dynamics from the shared planner.

## Submission status

The first local recorded-data experiment is complete. No public repo, footage release, recruiting message or application has been submitted. Raw footage remains local.

Earlier `runs/smoke_verified/experiment/` outputs are synthetic checks from an older implementation state, kept separate from these results.
