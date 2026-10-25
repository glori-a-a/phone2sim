# Phone data collection protocol

## Setup

Use your own phone, a fixed support and a plain table. Tape out a measured 40 × 30 cm rectangle. Put its corners in frame with a near-overhead view. Use the rear camera at normal zoom, 1080p/30 fps if available, with stable lighting and exposure. Do not move or zoom the camera after calibration. Avoid an ultra-wide lens, reflective surfaces, red/blue backgrounds, and similarly coloured clothing.

Use one red round puck of about 50 mm diameter and a rigid blue handheld pusher of about 24 mm diameter. A bottle cap fixed to a handle is adequate if it stays upright. The marker centre must represent the **centre of the contacting circular tool**, not the location of the hand holding it. Keep both markers close to the table plane; perspective from different heights is a systematic error. If using a blue fingertip marker instead, measure/explain the offset between that marker and the effective contact centre. The current support prior assumes object radius 25 mm and tool radius 12 mm.

## Record independent takes

Record 12–20 short takes, 5–12 seconds each. Use gentle pushes at about 2–8 cm/s, different directions, some starts/stops and slight curves. Aim for 10–20 cm of object travel and keep path span below 30 × 26 cm. Include tool motion without contact so the model sees that a moving tool need not move the object. Let the tool stay visible at the first and last sample. Do not lift, grasp, occlude, or tip the puck: those violate the planar task assumption.

Vary initial contact angle, speed and location; repeat across more than one session if possible. The held-out split must contain independent source recordings. All clips from one continuous original recording must share `source_group`; separate original takes may use different groups. Keep session notes separately. A recording-level split within one session does not establish cross-session generalization. Keep the original files. Record the collection date and who held the tool/phone. Only declare personal collection when true.

Avoid recording people or unrelated screens. Inspect raw footage before deciding what to publish. Raw videos are ignored by Git by default.

## Prepare and inspect

The current decoder assumes approximately constant video FPS. For variable-frame-rate phone files, convert to a constant 30 fps first using an installed ffmpeg (`ffmpeg -i input.mp4 -vf fps=30 output.mp4`), preserving the original file separately and documenting that preprocessing. Trim long introductions, camera movements, setup changes and periods with missing markers.

Calibrate each take from its first frame. Actual mat dimensions determine metric scale. Choose corners in the order top-left, top-right, bottom-right, bottom-left. If video orientation or dimensions change, recalibrate.

The default OpenCV HSV ranges target saturated red and blue. They are configuration values, not a trained general detector. Inspect every tracking overlay at normal speed and paused during contact. A big red blob elsewhere can win the largest-component detector; the region-of-interest mat limits this risk but does not eliminate it. Look for swaps, flicker, occlusion, and hand-marker offset. Change ranges/re-record instead of accepting bad tracks.

The tracker rejects >10% missing samples, gaps longer than two samples (0.2 s at 10 Hz), missing endpoints and >6 cm jumps per sample. Short gaps are linearly interpolated and counted in provenance. This is an explicit preprocessing approximation; high interpolation counts weaken evidence. The `review` command records your inspection declaration; it does not detect or repair bad tracking.

## Expected files per take

- Original `.mp4` plus calibration JSON (retain privately until reviewed for publication).
- Processed episode `.npz`: uniform timestamps, metric object and tool trajectories, provenance metadata. No arbitrary Python objects/pickle.
- `.tracking.mp4`: the detector overlay you inspect.
- `.provenance.json`: source/calibration SHA-256, source FPS, sample rate, interpolation counts, collector declaration and tracking review status.

If collection changes mid-recording, stop and make a new take. Do not smooth or manually edit trajectories to make evaluation look better. If annotations are corrected, document the correction and keep the original outputs.

