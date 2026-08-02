/**
 * The equip-time rules.
 *
 * These are the only rules the browser decides on its own, which makes them the
 * only ones worth testing here: everything else this app claims is the server's
 * answer, and the server has its own suite. What matters is that a part is
 * offered exactly when the spec's section 2.4 pairing rule says it fits, and
 * that fitting one produces an assembly the API would accept.
 */

import { describe, expect, it } from "vitest";

import {
  detach,
  equip,
  figures,
  fitTest,
  jointHandles,
  slots,
  suggestName,
  type Slot,
} from "./assembly";
import type { AssemblySpec, Module, Modules, Port } from "./catalog";

// A miniature catalog: a body with three mounts, a leg, a pad that fits the
// leg's foot, a gripper that fits nothing here, and a part whose only matching
// port is on the wrong link. Enough to reach every branch, small enough to read.
const port = (name: string, role: Port["role"], iface: string, link: string): Port => ({
  name,
  role,
  interface: iface,
  link,
  origin: { xyz: [0, 0, 0], rpy: [0, 0, 0] },
  description: "",
});

const module = (id: string, over: Partial<Module>): Module => ({
  id,
  version: "1.0.0",
  category: "body",
  display_name: id,
  description: "",
  tags: [],
  dof: 0,
  mass: 1,
  root_link: "base",
  links: [{ name: "base", material: null, mass: 1, visual: null }],
  joints: [],
  ports: [],
  ...over,
});

const BODY = module("body_v1", {
  mass: 4,
  ports: [
    port("leg_mount_fl", "female", "MOUNT_A40", "base"),
    port("leg_mount_fr", "female", "MOUNT_A40", "base"),
    port("top", "female", "MOUNT_A40", "base"),
  ],
});

const LEG = module("leg_left_v1", {
  category: "leg",
  mass: 2,
  dof: 2,
  links: [
    { name: "base", material: null, mass: 1, visual: null },
    { name: "shin", material: null, mass: 1, visual: null },
  ],
  joints: [
    {
      name: "hip",
      type: "revolute",
      parent: "base",
      child: "shin",
      origin: { xyz: [0, 0, 0], rpy: [0, 0, 0] },
      axis: [0, 1, 0],
      limit: { lower: -1, upper: 1, effort: 10, velocity: 2 },
      movable: true,
    },
  ],
  ports: [port("base", "male", "MOUNT_A40", "base"), port("foot", "male", "FOOT_C25", "shin")],
});

// Something that fits the leg's foot, so the tests have a three-deep unit.
const PAD = module("foot_pad_v1", {
  category: "foot",
  mass: 0.3,
  dof: 1,
  links: [
    { name: "base", material: null, mass: 0.2, visual: null },
    { name: "sole", material: null, mass: 0.1, visual: null },
  ],
  joints: [
    {
      name: "swivel",
      type: "revolute",
      parent: "base",
      child: "sole",
      origin: { xyz: [0, 0, 0], rpy: [0, 0, 0] },
      axis: [0, 0, 1],
      limit: { lower: -0.5, upper: 0.5, effort: 5, velocity: 1 },
      movable: true,
    },
  ],
  ports: [port("mount", "female", "FOOT_C25", "base")],
});

const GRIPPER = module("gripper_v1", {
  category: "gripper",
  mass: 0.5,
  ports: [port("base", "male", "TOOL_M20", "base")],
});

// A part whose only matching port is on a link the expander cannot attach.
const BADLY_PORTED = module("badly_ported_v1", {
  links: [
    { name: "base", material: null, mass: 1, visual: null },
    { name: "tip", material: null, mass: 0, visual: null },
  ],
  ports: [port("side", "male", "MOUNT_A40", "tip")],
});

const MODULES: Modules = new Map([BODY, LEG, PAD, GRIPPER, BADLY_PORTED].map((m) => [m.id, m]));

const UNIT: AssemblySpec = {
  name: "unit_01",
  root: "body",
  modules: [
    { id: "body", module: "body_v1", version: "1.0.0" },
    { id: "leg_fl", module: "leg_left_v1", version: "1.0.0" },
  ],
  connections: [{ parent: { instance: "body", port: "leg_mount_fl" }, child: { instance: "leg_fl", port: "base" } }],
};

const slotFor = (spec: AssemblySpec, key: string): Slot => {
  const found = slots(spec, MODULES).find((slot) => slot.key === key);
  if (!found) throw new Error(`no hardpoint ${key}`);
  return found;
};

describe("slots", () => {
  it("lists every port of every fitted part, parents before children", () => {
    expect(slots(UNIT, MODULES).map((slot) => slot.key)).toEqual([
      "body.leg_mount_fl",
      "leg_fl.foot",
      "body.leg_mount_fr",
      "body.top",
    ]);
  });

  it("does not offer the port a part already hangs from", () => {
    expect(slots(UNIT, MODULES).map((slot) => slot.key)).not.toContain("leg_fl.base");
  });

  it("nests a hardpoint under the part that carries it", () => {
    const foot = slotFor(UNIT, "leg_fl.foot");
    expect(foot.depth).toBe(1);
    expect(slotFor(UNIT, "body.top").depth).toBe(0);
  });

  it("says what is fitted where", () => {
    const taken = slotFor(UNIT, "body.leg_mount_fl");
    expect(taken.occupied).toBe(true);
    expect(taken.fittedModule).toBe("leg_left_v1");
    expect(slotFor(UNIT, "body.top").occupied).toBe(false);
  });
});

describe("fitTest", () => {
  it("accepts a matching interface with opposite roles", () => {
    expect(fitTest(LEG, slotFor(UNIT, "body.top"))).toEqual({ ok: true, port: "base" });
  });

  it("refuses a different interface, and says which one is needed", () => {
    const fit = fitTest(GRIPPER, slotFor(UNIT, "body.top"));
    expect(fit.ok).toBe(false);
    expect(fit.ok || fit.why).toMatch(/needs MOUNT_A40, part has TOOL_M20/);
  });

  it("refuses two ports of the same role", () => {
    const fit = fitTest(BODY, slotFor(UNIT, "body.top"));
    expect(fit.ok).toBe(false);
    expect(fit.ok || fit.why).toMatch(/Both sides are female/);
  });

  it("refuses a matching port that is not on the root link", () => {
    // Only the root link can be attached to a parent, so a port anywhere else
    // is unusable however well its interface matches.
    const fit = fitTest(BADLY_PORTED, slotFor(UNIT, "body.top"));
    expect(fit.ok).toBe(false);
    expect(fit.ok || fit.why).toMatch(/root link/);
  });
});

describe("equip", () => {
  it("adds the instance and the connection, pinning the catalog version", () => {
    const next = equip(UNIT, MODULES, slotFor(UNIT, "body.top"), "leg_left_v1");
    expect(next.modules).toHaveLength(3);
    expect(next.modules.at(-1)).toEqual({ id: "leg", module: "leg_left_v1", version: "1.0.0" });
    expect(next.connections.at(-1)).toEqual({
      parent: { instance: "body", port: "top" },
      child: { instance: "leg", port: "base" },
    });
  });

  it("refuses a part that does not fit, rather than building something illegal", () => {
    expect(equip(UNIT, MODULES, slotFor(UNIT, "body.top"), "gripper_v1")).toBe(UNIT);
  });

  it("replaces whatever was already fitted", () => {
    const next = equip(UNIT, MODULES, slotFor(UNIT, "body.leg_mount_fl"), "leg_left_v1");
    expect(next.modules.filter((m) => m.module === "leg_left_v1")).toHaveLength(1);
    expect(next.connections).toHaveLength(1);
  });

  it("takes the whole subtree off when it replaces a part", () => {
    const withFoot = equip(UNIT, MODULES, slotFor(UNIT, "leg_fl.foot"), "foot_pad_v1");
    expect(withFoot.modules).toHaveLength(3);

    const replaced = equip(withFoot, MODULES, slotFor(withFoot, "body.leg_mount_fl"), "leg_left_v1");
    expect(replaced.modules.map((m) => m.id).sort()).toEqual(["body", "leg_fl"]);
  });

  it("never reuses an instance name", () => {
    const one = equip(UNIT, MODULES, slotFor(UNIT, "body.top"), "leg_left_v1");
    const two = equip(one, MODULES, slotFor(one, "leg.foot"), "foot_pad_v1");
    const ids = two.modules.map((m) => m.id);
    expect(new Set(ids).size).toBe(ids.length);
  });
});

describe("detach", () => {
  it("leaves an assembly alone when the hardpoint is empty", () => {
    expect(detach(UNIT, "body.top")).toBe(UNIT);
  });

  it("removes the part and everything below it", () => {
    const deep = equip(UNIT, MODULES, slotFor(UNIT, "leg_fl.foot"), "foot_pad_v1");
    const bare = detach(deep, "body.leg_mount_fl");
    expect(bare.modules).toEqual([{ id: "body", module: "body_v1", version: "1.0.0" }]);
    expect(bare.connections).toEqual([]);
  });
});

describe("figures", () => {
  it("sums the catalog, and counts a connection as a joint", () => {
    const sums = figures(UNIT, MODULES);
    expect(sums.mass).toBeCloseTo(6);
    expect(sums.dof).toBe(2);
    expect(sums.links).toBe(3);
    expect(sums.joints).toBe(2); // one inside the leg, one for the connection
    expect(sums.parts).toBe(2);
    expect(sums.open).toBe(3);
  });
});

describe("jointHandles", () => {
  it("carries the limits, and sorts parents before children", () => {
    const deep = equip(UNIT, MODULES, slotFor(UNIT, "leg_fl.foot"), "foot_pad_v1");
    const handles = jointHandles(deep, MODULES);
    expect(handles.map((handle) => handle.key)).toEqual(["leg_fl/hip", "foot_pad/swivel"]);
    expect(handles[0]).toMatchObject({ lower: -1, upper: 1, depth: 1 });
  });
});

describe("suggestName", () => {
  it("names a part after the job it does", () => {
    expect(suggestName("leg_left_v1", slotFor(UNIT, "body.leg_mount_fr"))).toBe("leg_fr");
    expect(suggestName("gripper_v1", slotFor(UNIT, "body.top"))).toBe("gripper");
    expect(suggestName("body_v1", slotFor(UNIT, "body.top"))).toBe("body");
  });
});
