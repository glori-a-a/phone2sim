"""Episode-level storage and splits. No frame from a test video enters training."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class Episode:
    time: np.ndarray
    object_xy: np.ndarray
    tool_xy: np.ndarray
    metadata: dict
    observation_valid: np.ndarray | None = None

    def validate(self):
        n = len(self.time)
        if n < 12 or self.object_xy.shape != (n, 2) or self.tool_xy.shape != (n, 2):
            raise ValueError("An episode needs >=12 time samples and matching (N,2) object/tool tracks.")
        if not all(np.isfinite(x).all() for x in (self.time, self.object_xy, self.tool_xy)):
            raise ValueError("Tracks contain NaN or infinite values; inspect tracking first.")
        if self.observation_valid is not None:
            if self.observation_valid.shape != (n,) or self.observation_valid.dtype != np.bool_:
                raise ValueError("observation_valid must be a boolean array with one entry per sample.")
        dt = np.diff(self.time)
        if np.any(dt <= 0) or not np.allclose(dt, dt[0], rtol=.02, atol=1e-5):
            raise ValueError("Samples must have a uniform, positive timestep.")
        for key in ("episode_id", "origin", "source_group"):
            if not self.metadata.get(key):
                raise ValueError(f"Missing metadata: {key}")
        if self.metadata["origin"] not in ("human_video", "synthetic"):
            raise ValueError("origin must be human_video or synthetic.")
        if self.metadata["origin"] == "human_video":
            for key in ("collector", "collected_at", "source_sha256", "calibration_sha256"):
                if not self.metadata.get(key):
                    raise ValueError(f"Human episode missing provenance field: {key}")
            if self.metadata.get("attested_own_recording") is not True:
                raise ValueError("The collector must attest that they personally recorded this video.")
        return self

    @property
    def dt(self):
        return float(self.time[1] - self.time[0])

    @property
    def valid_observations(self):
        return self.observation_valid if self.observation_valid is not None else np.ones(len(self.time),dtype=bool)

    def save(self, path):
        self.validate()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, time=self.time, object_xy=self.object_xy, tool_xy=self.tool_xy,
                            metadata=np.array(json.dumps(self.metadata)),observation_valid=self.valid_observations)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            episode = cls(archive["time"], archive["object_xy"], archive["tool_xy"],
                          json.loads(str(archive["metadata"])),
                          archive["observation_valid"] if "observation_valid" in archive else None)
        return episode.validate()


def load_episodes(directory):
    paths = sorted(Path(directory).glob("*.npz"))
    if not paths:
        raise ValueError(f"No episode .npz files in {directory}. Run track or smoke first.")
    episodes = [Episode.load(p) for p in paths]
    ids = [e.metadata["episode_id"] for e in episodes]
    if len(ids) != len(set(ids)):
        raise ValueError("Episode IDs must be unique.")
    if len({e.metadata["origin"] for e in episodes}) != 1:
        raise ValueError("Do not mix human and synthetic episodes in one experiment.")
    if not all(np.isclose(e.dt, episodes[0].dt) for e in episodes):
        raise ValueError("All episodes must use the same sample rate.")
    return episodes


def split_episodes(episodes, seed=7):
    groups = sorted({e.metadata["source_group"] for e in episodes})
    if len(groups) < 4:
        raise ValueError("Need at least 4 independent source recordings (prefer 12+).")
    rng = np.random.default_rng(seed)
    groups = list(rng.permutation(groups))
    n_test = max(1, int(round(len(groups) * .25)))
    test_groups = set(groups[:n_test])
    train = [e for e in episodes if e.metadata["source_group"] not in test_groups]
    test = [e for e in episodes if e.metadata["source_group"] in test_groups]
    return train, test


def transitions(episodes):
    # Velocity uses only the preceding interval. No next-state information in inputs.
    objects, tools, velocities, actions, targets = [], [], [], [], []
    for e in episodes:
        observed=e.valid_observations
        valid=observed[:-2]&observed[1:-1]&observed[2:]
        objects.append(e.object_xy[1:-1][valid])
        tools.append(e.tool_xy[1:-1][valid])
        velocities.append((e.object_xy[1:-1] - e.object_xy[:-2])[valid])
        actions.append((e.tool_xy[2:] - e.tool_xy[1:-1])[valid])
        targets.append((e.object_xy[2:] - e.object_xy[1:-1])[valid])
    if not objects or sum(len(x) for x in objects)==0:
        raise ValueError("No fully observed transitions; interpolated samples are excluded from learning/scoring.")
    return tuple(np.concatenate(x) for x in (objects, tools, velocities, actions, targets))


def resample_path(path, count=32):
    path = np.asarray(path, dtype=float)
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    keep = np.r_[True, np.diff(distance) > 1e-6]
    if distance[-1] < .04:
        raise ValueError("Demonstrated object moved less than 4 cm. Use a longer push.")
    targets = np.linspace(0, distance[-1], count)
    return np.column_stack([np.interp(targets, distance[keep], path[keep, i]) for i in range(2)])

