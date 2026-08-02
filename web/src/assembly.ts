/**
 * The equip-time rules, and nothing else.
 *
 * These are the checks that have to be interactive: whether a part fits the
 * hardpoint under the cursor, and what the assembly becomes when it does. Mass
 * properties, interference, kinematic reachability and the URDF round trip are
 * the server's, reached through `POST /api/validate` -- re-implementing them
 * here would be a second validation layer that could disagree with the first.
 *
 * Every function is pure: state in, new state out. Nothing here touches the
 * network, the DOM or three.js, which is what makes it testable without any.
 */

import type { AssemblySpec, Module, Modules, Role } from "./catalog";

export interface Slot {
  /** `instance.port`, unique within an assembly. */
  key: string;
  instance: string;
  port: string;
  role: Role;
  interface: string;
  description: string;
  /** How deep in the assembly tree this hardpoint sits, for indenting. */
  depth: number;
  /** The instance fitted here, if any. */
  fitted: string | null;
  fittedModule: string | null;
  occupied: boolean;
}

export type Fit = { ok: true; port: string } | { ok: false; why: string };

/**
 * Every hardpoint on the unit, depth first, parents before children.
 *
 * The port a part hangs from is skipped: that is how the part is attached, not
 * somewhere you can fit anything.
 */
export function slots(spec: AssemblySpec, modules: Modules): Slot[] {
  const used = new Set<string>();
  const childAt = new Map<string, string>();
  for (const connection of spec.connections) {
    used.add(ref(connection.parent.instance, connection.parent.port));
    used.add(ref(connection.child.instance, connection.child.port));
    childAt.set(ref(connection.parent.instance, connection.parent.port), connection.child.instance);
  }

  const out: Slot[] = [];
  const walk = (instanceId: string, depth: number): void => {
    const instance = spec.modules.find((m) => m.id === instanceId);
    const module = instance && modules.get(instance.module);
    if (!module) return;

    for (const port of module.ports) {
      const key = ref(instanceId, port.name);
      const isMount = spec.connections.some(
        (c) => c.child.instance === instanceId && c.child.port === port.name,
      );
      if (isMount) continue;

      const fitted = childAt.get(key) ?? null;
      out.push({
        key,
        instance: instanceId,
        port: port.name,
        role: port.role,
        interface: port.interface,
        description: port.description,
        depth,
        fitted,
        fittedModule: fitted
          ? (spec.modules.find((m) => m.id === fitted)?.module ?? null)
          : null,
        occupied: used.has(key),
      });
      if (fitted) walk(fitted, depth + 1);
    }
  };

  walk(spec.root, 0);
  return out;
}

/**
 * Whether a module can be fitted to a hardpoint, and why not when it cannot.
 *
 * Spec section 2.4: the interfaces must match and the roles must form a
 * male/female pair. The port also has to sit on the module's root link, since
 * that is the only link the expander will attach to a parent.
 */
export function fitTest(module: Module, slot: Slot): Fit {
  const port = module.ports.find(
    (p) => p.link === module.root_link && p.interface === slot.interface && p.role !== slot.role,
  );
  if (port) return { ok: true, port: port.name };

  const sameInterface = module.ports.filter((p) => p.interface === slot.interface);
  if (sameInterface.length === 0) {
    const offered = [...new Set(module.ports.map((p) => p.interface))].join(", ") || "none";
    return { ok: false, why: `Interface mismatch — needs ${slot.interface}, part has ${offered}` };
  }
  if (sameInterface.some((p) => p.role === slot.role)) {
    return { ok: false, why: `Both sides are ${slot.role} — a joint needs one male and one female` };
  }
  return { ok: false, why: "No mounting port on this part's root link" };
}

/**
 * Fit `moduleId` to `slot`, returning the new assembly.
 *
 * Whatever was already there comes off first, along with everything hanging
 * below it -- a hardpoint carries one part, and that part's subtree goes with
 * it. Returns the assembly unchanged when the part does not fit, so a caller
 * that skipped `fitTest` cannot build something illegal by accident.
 */
export function equip(
  spec: AssemblySpec,
  modules: Modules,
  slot: Slot,
  moduleId: string,
): AssemblySpec {
  const module = modules.get(moduleId);
  if (!module) return spec;
  const fit = fitTest(module, slot);
  if (!fit.ok) return spec;

  const next = detach(spec, slot.key);
  const id = instanceName(next, moduleId, slot);
  return {
    ...next,
    modules: [...next.modules, { id, module: moduleId, version: module.version }],
    connections: [
      ...next.connections,
      { parent: { instance: slot.instance, port: slot.port }, child: { instance: id, port: fit.port } },
    ],
  };
}

/** Remove whatever is fitted at `slotKey`, and everything below it. */
export function detach(spec: AssemblySpec, slotKey: string): AssemblySpec {
  const link = spec.connections.find((c) => ref(c.parent.instance, c.parent.port) === slotKey);
  if (!link) return spec;

  const doomed = new Set([link.child.instance]);
  for (let grew = true; grew; ) {
    grew = false;
    for (const c of spec.connections) {
      if (doomed.has(c.parent.instance) && !doomed.has(c.child.instance)) {
        doomed.add(c.child.instance);
        grew = true;
      }
    }
  }

  return {
    ...spec,
    modules: spec.modules.filter((m) => !doomed.has(m.id)),
    connections: spec.connections.filter(
      (c) => !doomed.has(c.child.instance) && !doomed.has(c.parent.instance),
    ),
  };
}

export interface Figures {
  mass: number;
  dof: number;
  links: number;
  joints: number;
  open: number;
  parts: number;
}

/**
 * What the unit weighs and how many ways it moves.
 *
 * Summed from the catalog, so these are the same numbers the builder will
 * produce -- but they are a preview, not an authority: the figures the server
 * reports after `POST /api/build` are the ones that came out of the expander.
 */
export function figures(spec: AssemblySpec, modules: Modules): Figures {
  let mass = 0;
  let dof = 0;
  let links = 0;
  let joints = 0;
  for (const instance of spec.modules) {
    const module = modules.get(instance.module);
    if (!module) continue;
    mass += module.mass;
    dof += module.dof;
    links += module.links.length;
    joints += module.joints.length;
  }
  return {
    mass,
    dof,
    links,
    joints: joints + spec.connections.length,
    open: slots(spec, modules).filter((s) => !s.occupied).length,
    parts: spec.modules.length,
  };
}

/** Every movable joint on the unit, in tree order, with its instance path. */
export interface JointHandle {
  key: string;
  instance: string;
  joint: string;
  lower: number;
  upper: number;
  depth: number;
}

export function jointHandles(spec: AssemblySpec, modules: Modules): JointHandle[] {
  const depths = new Map<string, number>([[spec.root, 0]]);
  for (const slot of slots(spec, modules)) {
    if (slot.fitted) depths.set(slot.fitted, slot.depth + 1);
  }

  const out: JointHandle[] = [];
  for (const instance of spec.modules) {
    const module = modules.get(instance.module);
    if (!module) continue;
    for (const joint of module.joints) {
      if (!joint.movable) continue;
      out.push({
        key: `${instance.id}/${joint.name}`,
        instance: instance.id,
        joint: joint.name,
        lower: joint.limit?.lower ?? -Math.PI,
        upper: joint.limit?.upper ?? Math.PI,
        depth: depths.get(instance.id) ?? 0,
      });
    }
  }
  return out.sort((a, b) => a.depth - b.depth || a.key.localeCompare(b.key));
}

/** Name a part after the job it does, the way the prompt asks the model to. */
export function suggestName(moduleId: string, slot: Slot): string {
  const side = /_(fl|fr|rl|rr)$/.exec(slot.port);
  if (side) return `leg_${side[1]}`;
  if (moduleId.startsWith("arm")) return "arm_main";
  if (moduleId.startsWith("gripper")) return "gripper";
  if (moduleId.startsWith("leg")) return "leg";
  return moduleId.replace(/_v\d+$/, "");
}

function instanceName(spec: AssemblySpec, moduleId: string, slot: Slot): string {
  const base = suggestName(moduleId, slot);
  let name = base;
  for (let n = 2; spec.modules.some((m) => m.id === name); n++) name = `${base}_${n}`;
  return name;
}

export function ref(instance: string, port: string): string {
  return `${instance}.${port}`;
}
