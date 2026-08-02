/**
 * The 3D viewport.
 *
 * three.js owns the canvas and React owns everything around it, so the two only
 * meet at this boundary. The renderer, the scene and the controls are built once
 * and kept in refs; React re-renders never touch them. What flows in is an
 * assembly to draw, a pose to hold it in, and which hardpoint is selected.
 */

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

import type { AssemblySpec, Modules, Rgba } from "./catalog";
import { buildAssembly, buildStage, type AssemblyModel } from "./scene";

const VOID = 0x07090c;

export interface ViewName {
  name: "iso" | "front" | "side" | "top";
}

/** Camera directions, as a unit vector in URDF's Z-up world. */
const VIEWS: Record<ViewName["name"], [number, number, number]> = {
  iso: [1, -1.15, 0.72],
  front: [1, 0, 0.06],
  side: [0, -1, 0.06],
  top: [0.001, 0, 1],
};

interface Props {
  spec: AssemblySpec;
  modules: Modules;
  materials: Record<string, Rgba>;
  joints: Record<string, number>;
  selected: string | null;
  view: ViewName["name"];
  /** Bumped by the parent to re-frame the camera without changing the view. */
  frameToken: number;
}

export default function Viewer({
  spec,
  modules,
  materials,
  joints,
  selected,
  view,
  frameToken,
}: Props): React.ReactElement {
  const host = useRef<HTMLDivElement>(null);
  const renderer = useRef<THREE.WebGLRenderer>(null);
  const scene = useRef<THREE.Scene>(null);
  const camera = useRef<THREE.PerspectiveCamera>(null);
  const controls = useRef<OrbitControls>(null);
  const model = useRef<AssemblyModel>(null);
  const stage = useRef<THREE.Group>(null);

  // --- the parts that outlive every render ---
  useEffect(() => {
    const element = host.current;
    if (!element) return;

    const gl = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    gl.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    gl.setClearColor(VOID, 1);
    element.appendChild(gl.domElement);

    const world = new THREE.Scene();
    world.add(new THREE.HemisphereLight(0xdfe7f2, 0x1b1f26, 1.5));
    const key = new THREE.DirectionalLight(0xffffff, 2.1);
    key.position.set(2.4, -3.1, 3.6);
    world.add(key);
    const fill = new THREE.DirectionalLight(0x9fb4d0, 0.7);
    fill.position.set(-2.6, 2.2, 1.1);
    world.add(fill);

    const eye = new THREE.PerspectiveCamera(42, 1, 0.01, 200);
    eye.up.set(0, 0, 1); // URDF is Z-up; three.js defaults to Y.

    const orbit = new OrbitControls(eye, gl.domElement);
    orbit.enableDamping = true;
    orbit.dampingFactor = 0.09;
    orbit.rotateSpeed = 0.85;

    renderer.current = gl;
    scene.current = world;
    camera.current = eye;
    controls.current = orbit;

    let running = true;
    const tick = (): void => {
      if (!running) return;
      requestAnimationFrame(tick);
      orbit.update();
      gl.render(world, eye);
    };
    tick();

    const resize = new ResizeObserver(() => {
      const { clientWidth, clientHeight } = element;
      if (!clientWidth || !clientHeight) return;
      gl.setSize(clientWidth, clientHeight, false);
      eye.aspect = clientWidth / clientHeight;
      eye.updateProjectionMatrix();
    });
    resize.observe(element);

    return () => {
      running = false;
      resize.disconnect();
      orbit.dispose();
      gl.dispose();
      element.removeChild(gl.domElement);
      renderer.current = null;
    };
  }, []);

  // --- the robot, rebuilt when the assembly changes ---
  useEffect(() => {
    const world = scene.current;
    if (!world) return;

    const built = buildAssembly(spec, modules, materials);
    world.add(built.root);
    model.current = built;

    const ground = buildStage(built.bounds.radius);
    world.add(ground);
    stage.current = ground;

    return () => {
      world.remove(built.root);
      world.remove(ground);
      built.dispose();
      model.current = null;
    };
  }, [spec, modules, materials]);

  // A pose is a transform update, not a rebuild -- which is the whole reason
  // the scene graph mirrors the robot.
  useEffect(() => {
    model.current?.pose(joints);
  }, [joints, spec]);

  useEffect(() => {
    model.current?.highlight(selected);
  }, [selected, spec]);

  // --- framing ---
  useEffect(() => {
    const eye = camera.current;
    const orbit = controls.current;
    const built = model.current;
    if (!eye || !orbit || !built) return;

    const { center, radius } = built.bounds;
    const distance = (radius || 0.5) * 3.1;
    const direction = new THREE.Vector3(...VIEWS[view]).normalize();

    orbit.target.copy(center);
    eye.position.copy(center).addScaledVector(direction, distance);
    eye.near = Math.max(0.01, distance / 400);
    eye.far = distance * 40;
    eye.updateProjectionMatrix();
    orbit.update();
  }, [view, frameToken, spec]);

  return <div className="stage" ref={host} />;
}
