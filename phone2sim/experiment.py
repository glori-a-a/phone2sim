"""Reproducible experiments and plain reports; synthetic results are always labelled."""
from __future__ import annotations

from collections import defaultdict
import html
import json
from pathlib import Path
import platform

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

from .control import rollout
from .data import load_episodes, sha256, split_episodes, write_json
from .world_model import WorldModel, prediction_metrics


def train(data_dir, output, seed=7, allow_synthetic=False):
    output = Path(output)
    if output.exists():
        raise ValueError("Run directory already exists. Use a fresh --out to preserve previous experiments.")
    episodes = load_episodes(data_dir)
    origin = episodes[0].metadata["origin"]
    if origin == "synthetic" and not allow_synthetic:
        raise ValueError("Synthetic episodes require --allow-synthetic; they cannot satisfy the challenge.")
    if origin == "human_video" and not all(e.metadata.get("tracking_reviewed") for e in episodes):
        raise ValueError("Review each .tracking.mp4, then run review --episode PATH --accept-tracks.")
    # A duplicate recording must not be assigned to separate split groups.
    video_groups = defaultdict(set)
    for e in episodes:
        if e.metadata.get("source_sha256"):
            video_groups[e.metadata["source_sha256"]].add(e.metadata["source_group"])
    if any(len(groups) > 1 for groups in video_groups.values()):
        raise ValueError("The same source video appears in multiple source groups; use one group per recording/session.")
    train_set, test_set = split_episodes(episodes, seed)
    output.mkdir(parents=True)
    model = WorldModel().fit(train_set, seed=seed)
    model.save(output/"model.npz")
    shuffled = WorldModel().fit(train_set, shuffle_actions=True, seed=seed)
    shuffled.save(output/"model_shuffled.npz")
    package = Path(__file__).parent
    source_files = sorted(package.glob("*.py")) + sorted((package/"assets").glob("*.xml"))
    manifest = {"origin": origin, "challenge_data_requirement_met_by_declaration": origin == "human_video",
                "authorship_note": "Recording authorship is collector-declared, not independently verified.",
                "seed": seed, "dt": episodes[0].dt,
                "train_ids": [e.metadata["episode_id"] for e in train_set],
                "test_ids": [e.metadata["episode_id"] for e in test_set],
                "train_groups": sorted({e.metadata["source_group"] for e in train_set}),
                "test_groups": sorted({e.metadata["source_group"] for e in test_set}),
                "implementation_sha256": {str(p.relative_to(package)): sha256(p) for p in source_files},
                "episodes": {Path(p).name: {"sha256": sha256(p), "metadata": load_episodes_file(p).metadata}
                             for p in sorted(Path(data_dir).glob("*.npz"))},
                "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                            "mujoco": mujoco.__version__, "platform": platform.system()}}
    write_json(output/"manifest.json", manifest)
    prediction = {"learned": prediction_metrics(model, test_set),
                  "action_shuffled": prediction_metrics(shuffled, test_set)}
    write_json(output/"prediction_metrics.json", prediction)
    return model, test_set, manifest


def load_episodes_file(path):
    from .data import Episode
    return Episode.load(path)


def verify_inputs(data_dir, manifest):
    paths = {p.name: p for p in Path(data_dir).glob("*.npz")}
    if set(paths) != set(manifest["episodes"]):
        raise ValueError("Dataset files changed after training. Create a fresh run.")
    for name, provenance in manifest["episodes"].items():
        if sha256(paths[name]) != provenance["sha256"]:
            raise ValueError(f"Dataset file changed after training: {name}. Create a fresh run.")
    episodes = load_episodes(data_dir)
    return [e for e in episodes if e.metadata["episode_id"] in manifest["test_ids"]]


def evaluate(data_dir, run_dir, seeds=3, max_test=None, record=True, max_steps=500):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir/"manifest.json").read_text(encoding="utf-8"))
    model = WorldModel.load(run_dir/"model.npz")
    if model.metadata["train_episode_ids"] != manifest["train_ids"]:
        raise ValueError("Checkpoint and manifest do not match.")
    test_set = verify_inputs(data_dir, manifest)
    package = Path(__file__).parent
    for name, expected in manifest.get("implementation_sha256", {}).items():
        if sha256(package/name) != expected:
            raise ValueError("Implementation changed since training. Use a new run directory to keep results reproducible.")
    if max_test is not None:
        test_set = test_set[:max_test]
    if seeds < 1 or not test_set:
        raise ValueError("Evaluation requires >=1 seed and a held-out episode.")
    rows, videos = [], []
    settings = [("nominal", .4, False), ("impulse", .4, True), ("friction", .7, False)]
    for episode in test_set:
        for condition, friction, perturb in settings:
            for seed in range(seeds):
                for mode in ("replay", "geometry", "learned"):
                    take_video = record and episode is test_set[0] and condition == "impulse" and seed == 0
                    metrics, frames, trace = rollout(episode, model, mode, seed, friction, perturb,
                                                     max_steps=max_steps, record=take_video)
                    metrics.update({"episode_id": episode.metadata["episode_id"], "condition": condition})
                    rows.append(metrics)
                    # IDs are data, not filesystem paths.
                    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(episode.metadata["episode_id"]))
                    stem = f"{safe_id}_{condition}_{seed}_{mode}"
                    np.savez_compressed(run_dir/f"trace_{stem}.npz", **trace)
                    if take_video and frames:
                        label = "SYNTHETIC PIPELINE CHECK" if manifest["origin"] == "synthetic" else "HUMAN VIDEO MODEL / SIMULATION"
                        for frame in frames:
                            cv2.rectangle(frame, (0, 0), (640, 53), (18, 24, 32), -1)
                            cv2.putText(frame, label, (12, 22), 0, .58, (255, 215, 80), 1, cv2.LINE_AA)
                            cv2.putText(frame, f"{mode} | impulse test | not real robot footage", (12, 43), 0, .45, (235, 235, 235), 1, cv2.LINE_AA)
                        video_name = f"{mode}_impulse.mp4"
                        imageio.mimsave(run_dir/video_name, frames, fps=1/(2*episode.dt), macro_block_size=2)
                        imageio.imwrite(run_dir/f"{mode}_preview.png", frames[len(frames)//2])
                        videos.append(video_name)
        print(f"Evaluated {episode.metadata['episode_id']}", flush=True)
    write_json(run_dir/"rollouts.json", rows)
    summary = summarize(rows)
    write_json(run_dir/"summary.json", summary)
    report(run_dir, manifest, summary, videos)
    return summary


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row["mode"])].append(row)
    result = []
    for (condition, mode), values in groups.items():
        n = len(values)
        successes = sum(v["success"] for v in values)
        p, z = successes/n, 1.96
        center = (p+z*z/(2*n))/(1+z*z/n)
        half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
        result.append({"condition": condition, "mode": mode, "n": n, "successes": successes,
                       "success_rate": p, "wilson95_low": center-half, "wilson95_high": center+half,
                       "final_error_mm_mean": float(np.mean([v["final_error_mm"] for v in values])),
                       "path_completion_mean": float(np.mean([v["path_completion"] for v in values])),
                       "controller_ms_p50_mean": float(np.mean([v["controller_ms_p50"] for v in values]))})
    return result


def report(run_dir, manifest, summary, videos):
    prediction = json.loads((run_dir/"prediction_metrics.json").read_text(encoding="utf-8"))
    synthetic = manifest["origin"] == "synthetic"
    title = "Phone2Sim — synthetic engineering check" if synthetic else "Phone2Sim — recorded-data experiment"
    warning = ("SYNTHETIC DATA ONLY. This run does not satisfy the applicant-collected data requirement."
               if synthetic else "Uses collector-declared phone recordings; evaluate tracking and provenance before submission.")
    lines = [f"# {title}", "", f"**{warning}**", "", "## Held-out state prediction", "",
             "Distances are millimetres. Lower is better. Split unit: source recording/session.", "",
             "| Model | One-step | Moving samples | 10-step endpoint |", "|---|---:|---:|---:|"]
    def fmt(x):
        return "N/A" if x is None else f"{x:.2f}"
    for name, m in prediction.items():
        lines.append(f"| {name} | {fmt(m['model_mae_mm'])} | {fmt(m['model_moving_mae_mm'])} | {fmt(m['ten_step_endpoint_mae_mm'])} |")
    base = prediction["learned"]
    for name in ("stationary", "constant_velocity", "copy_tool"):
        lines.append(f"| {name} | {fmt(base[name+'_mae_mm'])} | {fmt(base[name+'_moving_mae_mm'])} | N/A |")
    lines += ["", "## Simulation rollouts", "", "| Condition | Method | Successes | Rate (95% interval) | Final error mm | Path complete | Control ms |",
              "|---|---|---:|---:|---:|---:|---:|"]
    table_rows = []
    for row in summary:
        values = [row["condition"], row["mode"], f"{row['successes']}/{row['n']}",
                  f"{row['success_rate']:.0%} ({row['wilson95_low']:.0%}–{row['wilson95_high']:.0%})",
                  f"{row['final_error_mm_mean']:.1f}", f"{row['path_completion_mean']:.0%}",
                  f"{row['controller_ms_p50_mean']:.2f}"]
        lines.append("| " + " | ".join(values) + " |")
        table_rows.append("<tr>"+"".join(f"<td>{html.escape(v)}</td>" for v in values)+"</tr>")
    lines += ["", "## Interpretation", "",
              "Success: final object centre within 25 mm of the endpoint AND at least 95% of sequential path checkpoints reached.",
              "Replay, geometry and learned controllers receive the same reference and simulator state. Geometry and learned share a hand-coded contact approach planner. Only in-contact action ranking uses learned predictions.",
              "The impulse adds 0.16 m/s to puck y velocity at 45% of demonstration duration (at least step 15). Friction test changes sliding friction from 0.4 to 0.7.",
              "Wilson intervals treat rollouts as trials; runs sharing a demonstration are correlated. These are descriptive, not independent-sample significance claims.",
              "The state world model uses observed tool displacement as a proxy action. Simulator servo commands may not perfectly realize that displacement. Per-rollout servo error is recorded; these experiments do not establish sim-to-real transfer.",
              "The 10-step metric conditions on observed future tool positions. It is not autonomous video generation or a full planning benchmark.",
              "Results may favour the geometric baseline; report them without selecting only favourable seeds. Synthetic data come from that baseline and do not establish human-video transfer.", "",
              "## Reproducibility", "", f"Seed: {manifest['seed']}; timestep: {manifest['dt']:.3f}s.",
              f"Train source groups: {len(manifest['train_groups'])}; test source groups: {len(manifest['test_groups'])}.",
              "Exact episode hashes, collector declarations, split IDs and runtime versions: manifest.json.", "",
              "## Example videos", ""]
    lines += [f"- [{name}]({name})" for name in videos]
    (run_dir/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    markup = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>body{{font:16px/1.6 system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#182838;background:#f5f6f8}}h1{{line-height:1.2}}.notice{{background:#fff0c2;border-left:5px solid #be7b00;padding:16px}}table{{border-collapse:collapse;width:100%;background:white;font-size:14px}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #ddd}}video{{width:100%;max-width:640px}}pre{{white-space:pre-wrap;background:white;padding:24px}}a{{color:#1758a0}}</style>
<h1>{html.escape(title)}</h1><p class="notice">{html.escape(warning)}</p><p>Action-conditioned state dynamics from metric object/tool tracks, evaluated on a physical two-joint MuJoCo arm.</p>
<h2>Simulation results</h2><table><tr><th>Condition</th><th>Method</th><th>Successes</th><th>Rate (95% interval)</th><th>Error mm</th><th>Path</th><th>Control ms</th></tr>{''.join(table_rows)}</table>
<h2>Example rollouts</h2>{''.join(f'<h3>{html.escape(v)}</h3><video controls preload="metadata" poster="{html.escape(v.replace("_impulse.mp4", "_preview.png"))}" src="{html.escape(v)}"></video>' for v in videos)}
<h2>Prediction metrics and experimental notes</h2><pre>{html.escape(chr(10).join(lines))}</pre></html>"""
    (run_dir/"report.html").write_text(markup, encoding="utf-8")


def audit(data_dir, run_dir):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir/"manifest.json").read_text(encoding="utf-8"))
    verify_inputs(data_dir, manifest)
    issues = []
    if manifest["origin"] != "human_video":
        issues.append("Only synthetic data: personally collected videos are still required.")
    if set(manifest["train_groups"]) & set(manifest["test_groups"]):
        issues.append("Source-group leakage between train and test.")
    if not (run_dir/"summary.json").exists():
        issues.append("No completed simulation evaluation.")
    return {"data_and_evaluation_checks_pass": not issues, "issues": issues,
            "still_manual": ["Check raw recordings and tracking overlays.",
                             "Inspect failures and write your own experimental discussion.",
                             "Confirm public repo access and whether late applications are accepted.",
                             "Authorship declarations are not independent verification."]}

