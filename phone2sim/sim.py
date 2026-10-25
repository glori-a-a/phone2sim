"""Two revolute joints, position servos, and physical contact; no object teleport during control."""
from __future__ import annotations

from importlib.resources import files
import xml.etree.ElementTree as ET
import numpy as np
import mujoco


class PushEnv:
    def __init__(self, dt=.1, friction=.4, object_radius=.025, tool_radius=.012):
        if not .005 <= object_radius <= .05 or not .003 <= tool_radius <= .04:
            raise ValueError("Object/tool radii are outside the supported table-task range.")
        asset=files("phone2sim").joinpath("assets/push.xml")
        root=ET.fromstring(asset.read_text(encoding='utf-8'))
        root.find(".//geom[@name='puck_geom']").set('size',f'{object_radius} 0.015')
        root.find(".//geom[@name='tip']").set('size',f'{tool_radius} 0.020')
        # Compile after setting sizes so inertias are derived consistently from geometry.
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
        self.data = mujoco.MjData(self.model)
        self.steps = int(round(dt / self.model.opt.timestep))
        self.dt = self.steps * self.model.opt.timestep
        if self.steps < 1 or not np.isclose(self.dt, dt, atol=1e-8, rtol=1e-6):
            raise ValueError("Control timestep must be a positive integer multiple of 0.002 s; use 10 Hz tracking.")
        self.model.geom("puck_geom").friction[0] = friction
        self.model.geom("table").friction[0] = friction
        self.renderer = None
        self.previous_object = np.zeros(2)

    @staticmethod
    def ik(xy):
        rel = np.asarray(xy) - [-.32, -.30]
        c = (rel @ rel - .4**2 - .36**2) / (2 * .4 * .36)
        if abs(c) > 1:
            raise ValueError(f"End-effector target {xy} is unreachable.")
        elbow = np.arccos(np.clip(c, -1, 1))
        shoulder = np.arctan2(rel[1], rel[0]) - np.arctan2(.36*np.sin(elbow), .4+.36*np.cos(elbow))
        return np.array([shoulder, elbow])

    def reset(self, object_xy, tool_xy):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:2] = self.ik(tool_xy)
        self.data.ctrl[:] = self.data.qpos[:2]
        self.data.joint("puck_free").qpos[:3] = [*object_xy, .015]
        mujoco.mj_forward(self.model, self.data)
        self.previous_object = np.array(object_xy, dtype=float)
        return self.state()

    def state(self):
        obj = self.data.body("puck").xpos[:2].copy()
        tool = self.data.site("tip_site").xpos[:2].copy()
        return obj, tool, obj - self.previous_object

    def step(self, action):
        obj, tool, _ = self.state()
        action = np.array(action, dtype=float, copy=True)
        action *= min(1., .018 / max(np.linalg.norm(action), 1e-9))
        target = np.clip(tool + action, [-.22, -.18], [.22, .20])
        self.data.ctrl[:] = self.ik(target)
        self.previous_object = obj.copy()
        for _ in range(self.steps):
            mujoco.mj_step(self.model, self.data)
        return self.state()

    def impulse(self, xy=(0., .10)):
        # Explicit evaluation intervention: velocity impulse, not a hidden state reset.
        self.data.joint("puck_free").qvel[:2] += xy

    def set_goal(self, goal):
        self.model.site("goal").pos[:2] = goal

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.renderer.update_scene(self.data, camera="overview")
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

