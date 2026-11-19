"""Quadrupeds with *different joint structures* that share a keypoint interface.

radial    Ant-like: spherical torso, four legs on the diagonals, 2 joints per
          leg (hip yaw about z + ankle).                       8 actuated joints
bilateral dog-like: elongated torso, legs hanging under its four corners,
          3 joints per leg (hip abduction about x, hip pitch about y, knee
          about y).                                           12 actuated joints

The joint spaces are incompatible, so the two policies are defined in a
shared, morphology-agnostic keypoint space:

observation (37): torso height, gravity and world-x axis in the torso frame,
    torso linear / angular velocity (torso frame), and for each foot
    (FL, FR, RL, RR) its position relative to its nominal stance position and
    its velocity, both in the torso frame.
action (12): per-foot target offset from the nominal stance position, in the
    torso frame, scaled to +-ACT_SCALE metres.

Each body has its own fixed low-level controller that realises foot targets
with task-space PD control through the foot Jacobian,
    tau = J^T Kp (p_target - p_foot) - Kd * qdot,
so all embodiment-specific knowledge lives below the shared interface. A 2-DoF
leg cannot reach every 3D foot target; that is part of the morphological
difference, not a bug.

Reward is identical for both: forward (+x) velocity + 1 (alive)
- 0.05 * |a|^2, episodes end when the torso falls or flips, max 1000 steps.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

ASSETS = Path(__file__).parent / "assets"
ANT_XML = Path(os.path.dirname(gym.__file__)) / "envs/mujoco/assets/ant.xml"
ACT_SCALE = 0.15
FRAME_SKIP = 5
FEET = ("FL", "FR", "RL", "RR")

BILATERAL_XML = """
<mujoco model="quad_bilateral">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option integrator="RK4" timestep="0.01"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom conaffinity="0" condim="3" density="5.0" friction="1 0.5 0.5" margin="0.01"/>
  </default>
  <worldbody>
    <geom conaffinity="1" condim="3" name="floor" pos="0 0 0" size="40 40 40" type="plane"/>
    <body name="torso" pos="0 0 0.62">
      <joint armature="0" damping="0" limited="false" margin="0.01" name="root" pos="0 0 0" type="free"/>
      <geom name="torso_geom" type="capsule" fromto="-0.32 0 0 0.32 0 0" size="0.13"/>
      {legs}
    </body>
  </worldbody>
  <actuator>{motors}</actuator>
</mujoco>
"""

LEG = """
      <body name="{n}_hip" pos="{x} {y} 0">
        <joint name="{n}_abd" axis="1 0 0" range="-30 30" type="hinge"/>
        <joint name="{n}_pitch" axis="0 1 0" range="-60 60" type="hinge"/>
        <geom fromto="0 0 0 0.07 0 -0.26" size="0.06" type="capsule"/>
        <body name="{n}_shin" pos="0.07 0 -0.26">
          <joint name="{n}_knee" axis="0 1 0" range="-70 70" type="hinge"/>
          <geom fromto="0 0 0 -0.07 0 -0.26" size="0.05" type="capsule"/>
          <site name="foot_{n}" pos="-0.07 0 -0.26" size="0.02"/>
        </body>
      </body>"""


def build_bilateral() -> str:
    ASSETS.mkdir(exist_ok=True)
    corners = {"FL": (0.32, 0.17), "FR": (0.32, -0.17), "RL": (-0.32, 0.17), "RR": (-0.32, -0.17)}
    legs = "".join(LEG.format(n=n, x=x, y=y) for n, (x, y) in corners.items())
    motors = "".join(
        f'<motor ctrllimited="true" ctrlrange="-1 1" joint="{n}_{j}" gear="100"/>'
        for n in FEET for j in ("abd", "pitch", "knee"))
    p = ASSETS / "quad_bilateral_kp.xml"
    p.write_text(BILATERAL_XML.format(legs=legs, motors=motors))
    return str(p)


def build_radial() -> str:
    """The stock Ant plus a site at each foot tip, named by quadrant."""
    ASSETS.mkdir(exist_ok=True)
    tree = ET.parse(ANT_XML)
    torso = tree.find(".//body[@name='torso']")
    quadrant = {"front_left_leg": "FL", "right_back_leg": "FR", "front_right_leg": "RL", "back_leg": "RR"}
    for leg, name in quadrant.items():
        ankle = torso.find(f"body[@name='{leg}']/body/body")
        tip = ankle.find("geom").get("fromto").split()[3:]
        ET.SubElement(ankle, "site", {"name": f"foot_{name}", "pos": " ".join(tip), "size": "0.02"})
    p = ASSETS / "ant_radial_kp.xml"
    tree.write(p)
    return str(p)


class KeypointQuad(gym.Env):
    def __init__(self, body: str, kp: float = 600.0, kd: float = 8.0, max_steps: int = 1000):
        xml = build_radial() if body == "radial" else build_bilateral()
        self.body = body
        self.m = mujoco.MjModel.from_xml_path(xml)
        self.d = mujoco.MjData(self.m)
        self.kp, self.kd, self.max_steps = kp, kd, max_steps
        self.sites = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, f"foot_{f}") for f in FEET]
        self.torso = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "torso")
        jnt = self.m.actuator_trnid[:, 0]
        self.act_dof = self.m.jnt_dofadr[jnt]
        self.act_qpos = self.m.jnt_qposadr[jnt]
        self.gear = self.m.actuator_gear[:, 0]
        # nominal pose: joints at the middle of their range (Ant ankles are 30..70 deg)
        self.q0 = self.m.qpos0.copy()
        rng = self.m.jnt_range[jnt]
        self.q0[self.act_qpos] = np.where(self.m.jnt_limited[jnt] > 0, rng.mean(1), 0.0)
        self.d.qpos[:] = self.q0
        mujoco.mj_kinematics(self.m, self.d)
        self.nominal = np.stack([self._foot_body(i) for i in range(4)])
        # drop the torso so the lowest foot touches the ground
        self.q0[2] -= min(self.d.site_xpos[s][2] for s in self.sites) - 0.03
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (37,), np.float64)
        self.action_space = gym.spaces.Box(-1, 1, (12,), np.float64)
        self.dt = self.m.opt.timestep * FRAME_SKIP
        self.np_random = np.random.default_rng()

    @property
    def data(self):
        return self.d

    # -- helpers
    def _R(self):
        return self.d.xmat[self.torso].reshape(3, 3)

    def _foot_body(self, i):
        return self._R().T @ (self.d.site_xpos[self.sites[i]] - self.d.xpos[self.torso])

    def _obs(self):
        R = self._R()
        feet = np.stack([self._foot_body(i) for i in range(4)])
        fvel = (feet - self.prev_feet) / self.dt
        self.prev_feet = feet
        return np.concatenate([
            [self.d.qpos[2]], R.T @ np.array([0, 0, -1.0]), R.T @ np.array([1.0, 0, 0]),
            R.T @ self.d.qvel[:3], self.d.qvel[3:6],
            (feet - self.nominal).ravel(), fvel.ravel()])

    def _torques(self, targets_body):
        R, pt = self._R(), self.d.xpos[self.torso]
        tau = np.zeros(self.m.nv)
        jac = np.zeros((3, self.m.nv))
        for i, s in enumerate(self.sites):
            mujoco.mj_jacSite(self.m, self.d, jac, None, s)
            err = (pt + R @ targets_body[i]) - self.d.site_xpos[s]
            tau += jac.T @ (self.kp * err)
        tau_a = tau[self.act_dof] - self.kd * self.d.qvel[self.act_dof]
        return np.clip(tau_a / self.gear, -1, 1)

    # -- gym API
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:] = self.q0 + self.np_random.uniform(-0.05, 0.05, self.m.nq) * (np.arange(self.m.nq) >= 7)
        self.d.qvel[:] = self.np_random.normal(0, 0.05, self.m.nv)
        mujoco.mj_forward(self.m, self.d)
        self.t = 0
        self.prev_feet = np.stack([self._foot_body(i) for i in range(4)])
        return self._obs(), {}

    def step(self, action):
        a = np.clip(action, -1, 1)
        targets = self.nominal + ACT_SCALE * a.reshape(4, 3)
        x0 = self.d.qpos[0]
        for _ in range(FRAME_SKIP):
            self.d.ctrl[:] = self._torques(targets)
            mujoco.mj_step(self.m, self.d)
        self.t += 1
        vx = (self.d.qpos[0] - x0) / self.dt
        up = self._R()[2, 2]
        healthy = 0.15 <= self.d.qpos[2] <= 1.2 and up > 0.2 and np.isfinite(self.d.qpos).all()
        r = vx + 1.0 - 0.05 * float((a**2).sum())
        return self._obs(), r, not healthy, self.t >= self.max_steps, {"x": self.d.qpos[0]}


GAINS = {"radial": (600.0, 8.0), "bilateral": (300.0, 15.0)}


def make(body):
    kp, kd = GAINS[body]
    return KeypointQuad(body, kp=kp, kd=kd)


if __name__ == "__main__":
    for b in ("radial", "bilateral"):
        e = make(b)
        o, _ = e.reset(seed=0)
        print(b, "nu", e.m.nu, "obs", o.shape, "z0", round(e.q0[2], 3))
        print("  nominal feet (torso frame):\n", np.round(e.nominal, 3))
        R, z = 0.0, []
        for t in range(300):
            o, r, te, tr, info = e.step(np.zeros(12))
            R += r
            z.append(e.d.qpos[2])
            if te:
                print("  fell at", t)
                break
        print(f"  zero action: return {R:.1f}, torso z {np.mean(z[-50:]):.3f}, x drift {info['x']:.3f}")
        # a crude open-loop trot in keypoint space to check that feet respond
        e.reset(seed=1)
        x = 0
        for t in range(300):
            ph = 2 * np.pi * t * e.dt * 2
            a = np.zeros((4, 3))
            for i, s in enumerate([0, np.pi, np.pi, 0]):
                a[i, 0] = np.sin(ph + s)
                a[i, 2] = max(0, np.cos(ph + s))
            o, r, te, tr, info = e.step(a.ravel())
            if te:
                print("  trot fell at", t)
                break
        print(f"  open-loop trot: x {info['x']:.2f}")
