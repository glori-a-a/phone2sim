"""A shared geometric contact planner isolates the effect of the learned dynamics."""
from __future__ import annotations

import time
import numpy as np

from .data import resample_path
from .sim import PushEnv


def limit(vector, length=.014):
    vector = np.asarray(vector, dtype=float)
    return vector * min(1., length / max(np.linalg.norm(vector), 1e-9))


def approach_action(obj, tool, direction, contact_radius=.037):
    """Go around a puck rather than push it while moving to the opposite side."""
    desired = obj - (contact_radius+.003)*direction
    rel = tool - obj
    if np.linalg.norm(tool-desired) < .013:
        return None
    behind = np.dot(rel, direction) < -(contact_radius-.007)
    lateral = abs(np.cross(np.r_[direction, 0], np.r_[rel, 0])[2])
    if behind and lateral < contact_radius-.002:
        return limit(desired-tool, .010)
    radius = max(np.linalg.norm(rel), 1e-6)
    angle = np.arctan2(rel[1], rel[0])
    goal_angle = np.arctan2(-direction[1], -direction[0])
    difference = (goal_angle-angle+np.pi) % (2*np.pi)-np.pi
    if radius < contact_radius+.018:
        target = obj + rel/radius*(contact_radius+.028)
    elif abs(difference) > .18:
        angle += np.clip(difference, -.20, .20)
        target = obj + (contact_radius+.028)*np.array([np.cos(angle), np.sin(angle)])
    else:
        target = desired
    return limit(target-tool, .010)


class Controller:
    def __init__(self, path, model=None, mode="learned"):
        self.path = resample_path(path)
        self.progress = 1
        self.model = model
        self.mode = mode
        self.last_uncertainty = 0.
        self.last_novelty = 0.
        self.contact_radius=model.object_radius+model.tool_radius if model is not None else .037

    def update_progress(self, obj):
        while self.progress < len(self.path)-1 and np.linalg.norm(obj-self.path[self.progress]) < .014:
            self.progress += 1

    def act(self, obj, tool, velocity):
        self.update_progress(obj)
        goal = self.path[self.progress]
        direction = goal-obj
        direction /= max(np.linalg.norm(direction), 1e-9)
        approach = approach_action(obj, tool, direction,self.contact_radius)
        if approach is not None:
            return approach
        if self.mode == "geometry":
            return limit(goal + .004*direction-tool, .009)
        angles = np.linspace(-.7, .7, 9)
        rotations = np.array([[[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]] for a in angles])
        directions = rotations @ direction
        actions = np.concatenate([directions*s for s in (.004, .008, .012, .016)])
        n = len(actions)
        pred, uncertainty, novelty = self.model.predict(
            np.tile(obj, (n, 1)), np.tile(tool, (n, 1)), np.tile(velocity, (n, 1)), actions)
        # One-step model predictive control. All methods share contact repositioning.
        desired = goal-obj
        score = np.linalg.norm(pred-desired, axis=1)
        score += .12*uncertainty + .00015*novelty
        # Tie-breaker encourages approaching contact if all predicted displacements are zero.
        score += .03*np.linalg.norm(tool+actions-(obj+.006*direction), axis=1)
        best = int(np.argmin(score))
        self.last_uncertainty, self.last_novelty = float(uncertainty[best]), float(novelty[best])
        return actions[best]


def rollout(episode, model, mode="learned", seed=0, friction=.4, perturb=False,
            max_steps=500, record=False):
    rng = np.random.default_rng(seed)
    # Translate the demonstration into the robot's workspace; preserve metric scale.
    reference = episode.object_xy - episode.object_xy[0]
    reference -= (reference.max(axis=0) + reference.min(axis=0))/2
    if np.any(np.abs(reference) > [.16, .14]):
        raise ValueError("Demonstrated path is too large for the simulated workspace; record within 30x26 cm.")
    offset = reference[0]-episode.object_xy[0]
    replay_tool = episode.tool_xy+offset
    start_tool = replay_tool[0].copy()
    contact_radius=model.object_radius+model.tool_radius
    if np.linalg.norm(start_tool-reference[0]) < contact_radius+.001:
        direction = reference[min(10, len(reference)-1)]-reference[0]
        start_tool = reference[0]-(contact_radius+.023)*direction/max(np.linalg.norm(direction), 1e-9)
    env = PushEnv(dt=episode.dt, friction=friction,object_radius=model.object_radius,tool_radius=model.tool_radius)
    state = env.reset(reference[0]+rng.uniform(-.004, .004, 2), start_tool)
    env.set_goal(reference[-1])
    controller = Controller(reference, model, mode)
    frames, positions, tools, durations = [], [state[0]], [state[1]], []
    uncertainties, novelties, servo_errors = [], [], []
    intervention_at = max(15, int(len(episode.time)*.45))
    intervention_applied = False
    try:
        for step in range(max_steps):
            started = time.perf_counter()
            controller.update_progress(state[0])
            at_end = (controller.progress == len(controller.path)-1 and
                      np.linalg.norm(state[0]-reference[-1]) < .018)
            if at_end and (not perturb or (intervention_applied and step > intervention_at+10)):
                break
            if mode == "replay":
                index = min(step+1, len(replay_tool)-1)
                action = limit(replay_tool[index]-state[1], .018)
            elif at_end:
                # A faster controller waits for the shared disturbance time.
                action = np.zeros(2)
            else:
                action = controller.act(*state)
            durations.append(time.perf_counter()-started)
            previous_tool = state[1].copy()
            state = env.step(action)
            servo_errors.append(np.linalg.norm(state[1]-previous_tool-action))
            if perturb and step == intervention_at:
                env.impulse((0., .16))
                intervention_applied = True
            positions.append(state[0])
            tools.append(state[1])
            uncertainties.append(controller.last_uncertainty)
            novelties.append(controller.last_novelty)
            if record and step % 2 == 0:
                frames.append(env.render())
            if np.any(np.abs(state[0]) > [.26, .24]):
                break
        # Replay progress is evaluated from the same sequential checkpoints offline.
        progress = 1
        for pos in positions:
            while progress < len(controller.path)-1 and np.linalg.norm(pos-controller.path[progress]) < .018:
                progress += 1
        final_error = float(np.linalg.norm(positions[-1]-reference[-1]))
        completed = progress/(len(controller.path)-1)
        tracking = np.min(np.linalg.norm(np.array(positions)[:, None]-controller.path[None], axis=-1), axis=1)
        metrics = {"mode": mode, "seed": seed, "friction": friction, "perturb": perturb,
                   "success": bool(final_error < .025 and completed >= .95 and (not perturb or intervention_applied)),
                   "intervention_applied": intervention_applied,
                   "intervention_step": intervention_at if intervention_applied else None,
                   "final_error_mm": final_error*1000, "path_completion": completed,
                   "mean_path_distance_mm": float(tracking.mean()*1000), "steps": len(positions)-1,
                   "controller_ms_p50": float(np.median(durations)*1000),
                   "controller_ms_p95": float(np.quantile(durations, .95)*1000),
                   "mean_servo_displacement_error_mm": float(np.mean(servo_errors)*1000),
                   "mean_uncertainty_mm": float(np.mean(uncertainties)*1000),
                   "mean_feature_distance": float(np.mean(novelties))}
        trace = {"object_xy": np.array(positions), "tool_xy": np.array(tools), "reference": reference}
        return metrics, frames, trace
    finally:
        env.close()

