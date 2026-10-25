from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def parser():
    root = argparse.ArgumentParser(description="Phone2Sim: phone tracks -> state world model -> MuJoCo arm.")
    sub = root.add_subparsers(dest="command", required=True)
    cal = sub.add_parser("calibrate", help="Click four measured mat corners in the first frame.")
    cal.add_argument("--video", required=True); cal.add_argument("--out", required=True)
    cal.add_argument("--width-m", type=float, default=.4); cal.add_argument("--height-m", type=float, default=.3)
    track = sub.add_parser("track", help="Track a personally recorded red object and blue fingertip marker.")
    for name in ("video", "calibration", "out", "collector", "collected-at"):
        track.add_argument("--"+name, required=True)
    track.add_argument("--attest-own-recording", action="store_true")
    track.add_argument("--source-group"); track.add_argument("--episode-id")
    review = sub.add_parser("review", help="Record that you inspected the tracking overlay.")
    review.add_argument("--episode", required=True); review.add_argument("--accept-tracks", action="store_true", required=True)
    train = sub.add_parser("train", help="Split by source video, fit dynamics, evaluate held-out predictions.")
    train.add_argument("--data", required=True); train.add_argument("--out", required=True)
    train.add_argument("--seed", type=int, default=7); train.add_argument("--allow-synthetic", action="store_true")
    evaluate = sub.add_parser("evaluate", help="Run replay / geometric / learned controllers on identical cases.")
    evaluate.add_argument("--data", required=True); evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--seeds", type=int, default=3); evaluate.add_argument("--max-test", type=int)
    evaluate.add_argument("--max-steps", type=int, default=500); evaluate.add_argument("--no-video", action="store_true")
    smoke = sub.add_parser("smoke", help="Synthetic engineering check; NOT challenge-compliant data.")
    smoke.add_argument("--out", default="runs/smoke"); smoke.add_argument("--episodes", type=int, default=16)
    smoke.add_argument("--seed", type=int, default=7); smoke.add_argument("--seeds", type=int, default=2)
    smoke.add_argument("--no-video", action="store_true")
    audit = sub.add_parser("audit", help="Check provenance declarations and evaluation artifacts.")
    audit.add_argument("--data", required=True); audit.add_argument("--run", required=True)
    return root


def main():
    args = parser().parse_args()
    try:
        if args.command == "calibrate":
            from .vision import calibrate
            calibrate(args.video, args.out, args.width_m, args.height_m)
            print(f"Calibration saved: {args.out}")
        elif args.command == "track":
            from .vision import track_video
            episode = track_video(args.video, args.calibration, args.out, args.collector, args.collected_at,
                                  args.attest_own_recording, args.source_group, args.episode_id)
            print(f"Saved {len(episode.time)} samples. Inspect {Path(args.out).with_suffix('.tracking.mp4')}")
        elif args.command == "review":
            from .data import Episode, write_json
            episode = Episode.load(args.episode)
            if episode.metadata["origin"] != "human_video":
                raise ValueError("review is for human-video tracking only.")
            episode.metadata["tracking_reviewed"] = True
            episode.save(args.episode)
            write_json(Path(args.episode).with_suffix(".provenance.json"), episode.metadata)
            print("Tracking review recorded. This is your declaration, not automatic validation.")
        elif args.command == "train":
            from .experiment import train
            _, _, manifest = train(args.data, args.out, args.seed, args.allow_synthetic)
            print(f"Model saved: {args.out} | data origin: {manifest['origin']}")
        elif args.command == "evaluate":
            from .experiment import evaluate
            evaluate(args.data, args.run, args.seeds, args.max_test, not args.no_video, args.max_steps)
            print(f"Report: {Path(args.run)/'report.html'}")
        elif args.command == "smoke":
            from .synthetic import generate
            from .experiment import train, evaluate
            output = Path(args.out)
            if output.exists():
                raise ValueError("Smoke output already exists. Use a new --out.")
            print("SYNTHETIC ENGINEERING CHECK - not personally collected data.", flush=True)
            generate(output/"episodes", args.episodes, args.seed)
            train(output/"episodes", output/"experiment", args.seed, allow_synthetic=True)
            evaluate(output/"episodes", output/"experiment", args.seeds, max_test=1, record=not args.no_video)
            print(f"Report: {output/'experiment'/'report.html'}")
        elif args.command == "audit":
            from .experiment import audit
            result = audit(args.data, args.run)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if not result["data_and_evaluation_checks_pass"]:
                return 2
    except (ValueError, OSError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

