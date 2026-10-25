"""Action-aligned nearest-neighbour state dynamics, not a pretrained VLA.

In an action-aligned frame, rotation and translation disappear. Each neighbour
votes for a puck displacement; disagreement is a heuristic uncertainty score,
not a calibrated probability. A swept-contact gate prevents remote pushes.
"""
from __future__ import annotations

import json
import numpy as np

from .data import transitions


def action_frame(obj, tool, velocity, action):
    obj, tool, velocity, action = [np.atleast_2d(x) for x in (obj, tool, velocity, action)]
    length = np.linalg.norm(action, axis=1)
    direction = action / np.maximum(length[:, None], 1e-8)
    direction[length < 1e-8] = [1., 0.]
    side = np.column_stack([-direction[:, 1], direction[:, 0]])
    rel = tool - obj
    features = np.column_stack([
        np.sum(rel*direction, axis=1)/.04, np.sum(rel*side, axis=1)/.04,
        length/.015, np.sum(velocity*direction, axis=1)/.015,
        np.sum(velocity*side, axis=1)/.015])
    return features, direction, side


class WorldModel:
    def __init__(self, k=12):
        self.k = k

    @property
    def object_radius(self):
        return float(self.metadata.get('object_radius_m',.025))

    @property
    def tool_radius(self):
        return float(self.metadata.get('tool_radius_m',.012))

    def fit(self, episodes, shuffle_actions=False, seed=7):
        sizes=[(float(e.metadata.get('object_diameter_m',.05)),float(e.metadata.get('tool_diameter_m',.024))) for e in episodes]
        if not all(np.allclose(pair,sizes[0],atol=1e-6) for pair in sizes):
            raise ValueError("A dynamics checkpoint requires a consistent object/tool geometry across episodes.")
        obj, tool, velocity, action, delta = transitions(episodes)
        if shuffle_actions:
            action = np.random.default_rng(seed).permutation(action)
        features, direction, side = action_frame(obj, tool, velocity, action)
        target = np.column_stack([np.sum(delta*direction, axis=1), np.sum(delta*side, axis=1)])
        # Bound cost/memory; deterministic sample, retaining stationary examples too.
        if len(features) > 12000:
            keep = np.random.default_rng(seed).choice(len(features), 12000, replace=False)
            features, target = features[keep], target[keep]
        self.features, self.target = features, target
        self.metadata = {"origin": episodes[0].metadata["origin"], "dt": episodes[0].dt,
                         "train_episode_ids": [e.metadata["episode_id"] for e in episodes],
                         "train_source_groups": sorted({e.metadata["source_group"] for e in episodes}),
                         "action_shuffled": shuffle_actions, "seed": seed,
                         "object_radius_m":sizes[0][0]/2,"tool_radius_m":sizes[0][1]/2,
                         "object_size_source":episodes[0].metadata.get('object_diameter_source','synthetic/default geometry')}
        return self

    def predict(self, obj, tool, velocity, action):
        query, direction, side = action_frame(obj, tool, velocity, action)
        predictions, uncertainty, novelty = [], [], []
        # Chunk to cap candidate x transition memory when scoring long videos.
        for start in range(0, len(query), 128):
            part = query[start:start+128]
            distance = np.sum((part[:, None] - self.features[None])**2, axis=-1)
            k = min(self.k, len(self.features))
            nearest = np.argpartition(distance, k-1, axis=1)[:, :k]
            d = np.take_along_axis(distance, nearest, axis=1)
            weights = np.exp(-np.minimum(d - d.min(axis=1, keepdims=True), 80) / .4)
            weights /= weights.sum(axis=1, keepdims=True)
            values = self.target[nearest]
            mean = np.sum(weights[..., None]*values, axis=1)
            std = np.sqrt(np.sum(weights * np.sum((values-mean[:, None])**2, axis=-1), axis=1))
            predictions.append(mean)
            uncertainty.append(std)
            novelty.append(np.sqrt(d.min(axis=1)))
        local = np.concatenate(predictions)
        delta = local[:, :1]*direction + local[:, 1:]*side
        # Physical support prior, calibrated body radii + tracking allowance .008 m.
        obj, tool, action = [np.atleast_2d(x) for x in (obj, tool, action)]
        rel = tool - obj
        t = np.clip(-np.sum(rel*action, axis=1) / np.maximum(np.sum(action**2, axis=1), 1e-12), 0, 1)
        contact = np.linalg.norm(rel + t[:, None]*action, axis=1) < self.object_radius+self.tool_radius+.008
        # Resting puck outside swept contact has zero action-induced displacement.
        delta[~contact] = np.atleast_2d(velocity)[~contact] * .25
        delta = np.clip(delta, -.025, .025)
        return delta, np.concatenate(uncertainty), np.concatenate(novelty)

    def save(self, path):
        np.savez_compressed(path, features=self.features, target=self.target,
                            k=np.array(self.k), metadata=np.array(json.dumps(self.metadata)))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            model = cls(int(archive["k"]))
            model.features = archive["features"]
            model.target = archive["target"]
            model.metadata = json.loads(str(archive["metadata"]))
        return model


def prediction_metrics(model, episodes):
    obj, tool, velocity, action, target = transitions(episodes)
    pred, uncertainty, novelty = model.predict(obj, tool, velocity, action)
    moving = np.linalg.norm(target, axis=1) > .001
    def error(value, mask):
        return float(np.linalg.norm(value[mask]-target[mask], axis=1).mean()*1000) if mask.any() else None
    all_samples = np.ones(len(target), dtype=bool)
    result = {"transitions": len(target), "moving_transitions": int(moving.sum()),
              "mean_uncertainty_mm": float(uncertainty.mean()*1000),
              "mean_feature_distance": float(novelty.mean())}
    for label, value in [("model", pred), ("stationary", np.zeros_like(target)),
                         ("constant_velocity", velocity), ("copy_tool", action)]:
        result[f"{label}_mae_mm"] = error(value, all_samples)
        result[f"{label}_moving_mae_mm"] = error(value, moving)
    # Open-loop 10-step prediction uses observed future tool trajectories, no
    # future object observations. It is conditional prediction, not a policy test.
    errors = []
    for e in episodes:
        for t in range(1, len(e.time)-11, 10):
            if not e.valid_observations[t-1:t+11].all():
                continue
            position = e.object_xy[t].copy()
            vel = e.object_xy[t] - e.object_xy[t-1]
            for h in range(10):
                d, _, _ = model.predict(position, e.tool_xy[t+h], vel,
                                        e.tool_xy[t+h+1]-e.tool_xy[t+h])
                position += d[0]
                vel = d[0]
            errors.append(np.linalg.norm(position-e.object_xy[t+10])*1000)
    result["ten_step_endpoint_mae_mm"] = float(np.mean(errors)) if errors else None
    return result

