/**
 * The assembly as a three.js scene graph.
 *
 * The graph mirrors the robot: a group per link, a group per joint, a group per
 * connection. That is not decoration -- it is what makes posing cheap. Moving a
 * joint sets one node's transform and three.js carries the change down the
 * subtree, so the arm follows the shoulder without anything being rebuilt.
 *
 * The transforms are the ones the expander uses (spec section 6.2), so what you
 * see here is what `POST /api/build` will emit:
 *
 *     T_connection = T_parent_port . T_child_port^-1
 *
 * URDF is Z-up and its `rpy` is a fixed-axis XYZ rotation, which is three.js's
 * intrinsic `ZYX` Euler order with the same three numbers.
 */

import * as THREE from "three";

import type { AssemblySpec, Geometry, Module, Modules, Origin, Rgba, Vec3 } from "./catalog";

/** Ports drawn as rings, so you can see where a part could go. */
export interface PortMarker {
  key: string;
  instance: string;
  port: string;
  occupied: boolean;
}

export interface AssemblyModel {
  /** The whole robot, ready to add to a scene. */
  root: THREE.Group;
  /** Move the joints. Keys are `instance/joint`, values radians or metres. */
  pose: (values: Record<string, number>) => void;
  /** Highlight one hardpoint, or none. */
  highlight: (slotKey: string | null) => void;
  /** A sphere that contains the robot, for framing the camera. */
  bounds: THREE.Sphere;
  dispose: () => void;
}

const FALLBACK_COLOUR: Rgba = [0.62, 0.64, 0.67, 1];
const MARKER_RADIUS = 0.022;

export function buildAssembly(
  spec: AssemblySpec,
  modules: Modules,
  materials: Record<string, Rgba>,
): AssemblyModel {
  const root = new THREE.Group();
  root.name = spec.name;

  const disposables: { dispose: () => void }[] = [];
  const linkNodes = new Map<string, Map<string, THREE.Group>>();
  const jointNodes = new Map<string, { node: THREE.Group; joint: Module["joints"][number] }>();
  const markers = new Map<string, THREE.Mesh>();

  const materialFor = (name: string | null): THREE.Material => {
    const [r, g, b, a] = (name && materials[name]) || FALLBACK_COLOUR;
    const material = new THREE.MeshStandardMaterial({
      color: new THREE.Color(r, g, b),
      roughness: 0.62,
      metalness: 0.12,
      transparent: a < 1,
      opacity: a,
    });
    disposables.push(material);
    return material;
  };

  // One subtree per instance, wired up internally by its own joints.
  for (const instance of spec.modules) {
    const module = modules.get(instance.module);
    if (!module) continue;

    const links = new Map<string, THREE.Group>();
    for (const link of module.links) {
      const node = new THREE.Group();
      node.name = `${instance.id}_${link.name}`;
      links.set(link.name, node);

      if (!link.visual) continue;
      const geometry = geometryOf(link.visual.geometry);
      if (!geometry) continue;
      disposables.push(geometry);

      const mesh = new THREE.Mesh(geometry, materialFor(link.visual.material ?? link.material));
      applyOrigin(mesh, link.visual.origin);
      // A cylinder is Z-long in URDF and Y-long in three.js.
      if (link.visual.geometry.shape === "cylinder") mesh.rotateX(Math.PI / 2);
      node.add(mesh);
    }
    linkNodes.set(instance.id, links);

    for (const joint of module.joints) {
      const parent = links.get(joint.parent);
      const child = links.get(joint.child);
      if (!parent || !child) continue;

      const fixed = new THREE.Group();
      applyOrigin(fixed, joint.origin);
      parent.add(fixed);

      // A separate node for the variable part, so posing never disturbs the
      // frame the joint was designed in.
      const moving = new THREE.Group();
      fixed.add(moving);
      moving.add(child);
      if (joint.movable) jointNodes.set(`${instance.id}/${joint.name}`, { node: moving, joint });
    }

    for (const port of module.ports) {
      const owner = links.get(port.link);
      if (!owner) continue;
      const marker = new THREE.Mesh(
        new THREE.TorusGeometry(MARKER_RADIUS, MARKER_RADIUS / 3.2, 8, 20),
        new THREE.MeshBasicMaterial({ color: 0x3d4450, transparent: true, opacity: 0.85 }),
      );
      disposables.push(marker.geometry, marker.material);
      applyOrigin(marker, port.origin);
      marker.name = `${instance.id}.${port.name}`;
      markers.set(marker.name, marker);
      owner.add(marker);
    }
  }

  // Then hang the instances off each other at their mating ports.
  const attached = new Set<string>([spec.root]);
  const rootModule = moduleOf(spec, modules, spec.root);
  const rootLink = rootModule && linkNodes.get(spec.root)?.get(rootModule.root_link);
  if (rootLink) root.add(rootLink);

  for (let progressed = true; progressed; ) {
    progressed = false;
    for (const connection of spec.connections) {
      if (attached.has(connection.child.instance) || !attached.has(connection.parent.instance)) {
        continue;
      }
      const parentModule = moduleOf(spec, modules, connection.parent.instance);
      const childModule = moduleOf(spec, modules, connection.child.instance);
      const parentPort = parentModule?.ports.find((p) => p.name === connection.parent.port);
      const childPort = childModule?.ports.find((p) => p.name === connection.child.port);
      const parentLink = parentPort && linkNodes.get(connection.parent.instance)?.get(parentPort.link);
      const childLink =
        childModule && linkNodes.get(connection.child.instance)?.get(childModule.root_link);
      if (!parentPort || !childPort || !parentLink || !childLink) continue;

      const mount = new THREE.Group();
      mount.matrixAutoUpdate = false;
      mount.matrix.copy(matrixOf(parentPort.origin)).multiply(matrixOf(childPort.origin).invert());
      mount.add(childLink);
      parentLink.add(mount);

      attached.add(connection.child.instance);
      progressed = true;
    }
  }

  const pose = (values: Record<string, number>): void => {
    for (const [key, { node, joint }] of jointNodes) {
      const q = values[key] ?? 0;
      node.position.set(0, 0, 0);
      node.quaternion.identity();
      const axis = new THREE.Vector3(...joint.axis).normalize();
      if (joint.type === "prismatic") {
        node.position.copy(axis.multiplyScalar(q));
      } else {
        node.quaternion.setFromAxisAngle(axis, q);
      }
    }
  };

  const highlight = (slotKey: string | null): void => {
    for (const [key, marker] of markers) {
      const material = marker.material as THREE.MeshBasicMaterial;
      const selected = key === slotKey;
      material.color.setHex(selected ? 0xe06b1f : 0x3d4450);
      material.opacity = selected ? 1 : 0.85;
      marker.scale.setScalar(selected ? 1.7 : 1);
    }
  };

  pose({});
  root.updateMatrixWorld(true);

  const box = new THREE.Box3().setFromObject(root);
  const bounds = box.isEmpty()
    ? new THREE.Sphere(new THREE.Vector3(), 1)
    : box.getBoundingSphere(new THREE.Sphere());

  return {
    root,
    pose,
    highlight,
    bounds,
    dispose: () => {
      for (const item of disposables) item.dispose();
    },
  };
}

/** The ground plane and origin axes, in URDF's Z-up world. */
export function buildStage(radius: number): THREE.Group {
  const stage = new THREE.Group();

  const span = Math.max(1, Math.ceil(radius * 2));
  const grid = new THREE.GridHelper(span * 2, span * 8, 0x2b3038, 0x1d2127);
  grid.rotateX(Math.PI / 2);
  stage.add(grid);

  // Short enough to read as an origin marker rather than a mast through the
  // chassis, which is where the world origin usually sits.
  stage.add(new THREE.AxesHelper(radius * 0.28));
  return stage;
}

function moduleOf(spec: AssemblySpec, modules: Modules, instanceId: string): Module | undefined {
  const instance = spec.modules.find((m) => m.id === instanceId);
  return instance ? modules.get(instance.module) : undefined;
}

function geometryOf(geometry: Geometry): THREE.BufferGeometry | null {
  switch (geometry.shape) {
    case "box":
      return new THREE.BoxGeometry(...geometry.size);
    case "cylinder":
      return new THREE.CylinderGeometry(geometry.radius, geometry.radius, geometry.length, 24);
    case "sphere":
      return new THREE.SphereGeometry(geometry.radius, 24, 16);
    case "mesh":
      // The payload names the mesh but cannot carry it. Draw the bounding cube
      // rather than nothing, so a module with CAD geometry is still visible.
      return new THREE.BoxGeometry(...(geometry.scale as Vec3));
  }
}

function applyOrigin(object: THREE.Object3D, origin: Origin): void {
  object.position.set(...origin.xyz);
  object.setRotationFromEuler(new THREE.Euler(...origin.rpy, "ZYX"));
}

function matrixOf(origin: Origin): THREE.Matrix4 {
  return new THREE.Matrix4().compose(
    new THREE.Vector3(...origin.xyz),
    new THREE.Quaternion().setFromEuler(new THREE.Euler(...origin.rpy, "ZYX")),
    new THREE.Vector3(1, 1, 1),
  );
}
