"""Physics-generated diagnostics. These are NOT applicant-collected data."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .control import Controller
from .data import Episode, write_json
from .sim import PushEnv


def generate(directory, count=16, seed=7):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if list(directory.glob("*.npz")):
        raise ValueError("Synthetic output directory already contains episodes. Choose a new directory.")
    rng = np.random.default_rng(seed)
    for i in range(count):
        angle = rng.uniform(-np.pi, np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)])
        side = np.array([-direction[1], direction[0]])
        length = rng.uniform(.10, .22)
        t = np.linspace(0, 1, 50)
        reference = (t[:, None]-.5)*length*direction
        reference += np.sin(t[:, None]*np.pi)*rng.uniform(-.03, .03)*side
        env = PushEnv(friction=rng.uniform(.3, .55))
        state = env.reset(reference[0], reference[0]-.052*direction)
        controller = Controller(reference, mode="geometry")
        objects, tools = [state[0]], [state[1]]
        try:
            for step in range(220):
                action = controller.act(*state)
                # Speed variation helps identify action-conditional dynamics.
                action *= rng.uniform(.65, 1.5)
                state = env.step(action)
                objects.append(state[0])
                tools.append(state[1])
                if controller.progress == len(controller.path)-1 and np.linalg.norm(state[0]-reference[-1]) < .012:
                    for _ in range(8):
                        state = env.step(np.zeros(2))
                        objects.append(state[0]); tools.append(state[1])
                    break
            episode = Episode(np.arange(len(objects))*env.dt, np.array(objects), np.array(tools),
                              {"episode_id": f"synthetic_{i:03d}", "source_group": f"sim_{i:03d}",
                               "origin": "synthetic", "seed": seed, "purpose": "pipeline_check_only"})
            episode.save(directory/f"synthetic_{i:03d}.npz")
        finally:
            env.close()
    write_json(directory/"NOTICE.json", {"origin": "synthetic", "applicant_data": False,
                                       "message": "Never submit these as personally collected recordings."})

