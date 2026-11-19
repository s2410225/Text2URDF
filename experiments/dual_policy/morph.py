"""Two quadruped morphologies with identical joint trees (8 actuated joints).

radial    : the standard MuJoCo Ant. Four legs on the diagonals (45, 135, 225,
            315 deg) of a spherical torso: 4-fold rotational symmetry, no
            preferred heading.
bilateral : same legs, same joint order, but the torso is an elongated capsule
            along x, the legs are attached at its front and back corners and
            rotated so they point sideways (+-y). Only mirror symmetry across
            the sagittal (x-z) plane remains; x is a natural "forward".

Because the kinematic tree and actuator order are identical, observation and
action vectors line up leg-for-leg (leg i <-> leg i), so a policy of one body
can be queried on the other body's observations, but the physics they map to
is different: the hip/ankle axes point in different world directions.
"""

import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import gymnasium as gym

def _atomic_write(path, text):
    """Write via a temp file + rename so parallel workers never see a partial file."""
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".xml")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)


ANT_XML = Path(os.path.dirname(gym.__file__)) / "envs/mujoco/assets/ant.xml"
OUT_DIR = Path(__file__).parent / "assets"

# leg root body -> (rotation about z in degrees, attachment point on torso)
BILATERAL = {
    "front_left_leg": (45, (0.30, 0.10)),     # 45  -> 90  (points +y), front
    "front_right_leg": (-45, (-0.30, 0.10)),  # 135 -> 90  (points +y), rear
    "back_leg": (45, (-0.30, -0.10)),         # 225 -> 270 (points -y), rear
    "right_back_leg": (-45, (0.30, -0.10)),   # 315 -> 270 (points -y), front
}


def bilateral_xml() -> str:
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / "ant_bilateral.xml"
    tree = ET.parse(ANT_XML)
    torso = tree.find(".//body[@name='torso']")
    g = torso.find("geom[@name='torso_geom']")
    g.attrib.update({"type": "capsule", "fromto": "-0.30 0 0 0.30 0 0", "size": "0.15"})
    g.attrib.pop("pos", None)
    for name, (rot, (x, y)) in BILATERAL.items():
        b = torso.find(f"body[@name='{name}']")
        b.set("pos", f"{x} {y} 0")
        b.set("euler", f"0 0 {rot}")
    _atomic_write(path, ET.tostring(tree.getroot(), encoding="unicode"))
    return str(path)


def make(morph: str, **kw):
    kw.setdefault("include_cfrc_ext_in_observation", False)
    if morph == "radial":
        return gym.make("Ant-v5", **kw)
    if morph == "bilateral":
        return gym.make("Ant-v5", xml_file=bilateral_xml(), **kw)
    raise ValueError(morph)


if __name__ == "__main__":
    import numpy as np

    for m in ("radial", "bilateral"):
        e = make(m)
        o, _ = e.reset(seed=0)
        mass = e.unwrapped.model.body_mass.sum()
        print(m, o.shape, e.action_space.shape, f"mass {mass:.2f}")
        R = 0.0
        for _ in range(200):
            o, r, te, tr, _ = e.step(np.zeros(8))
            R += r
        print("  zero-action 200 steps return", round(R, 1), "z", round(e.unwrapped.data.qpos[2], 3))

