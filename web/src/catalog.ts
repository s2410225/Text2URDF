/**
 * The shape of `GET /api/catalog`, and of the `AssemblySpec` the API takes back.
 *
 * These types are written by hand against `urdfgen.catalog.payload`, which is
 * the price of not running a code generator over the OpenAPI document. What
 * keeps them honest is `PAYLOAD_VERSION`: the payload carries a version, the
 * app refuses a version it was not written for, and a change to the Python side
 * that matters is required to bump it.
 */

export const SUPPORTED_PAYLOAD_VERSION = 1;

export type Vec3 = [number, number, number];
export type Rgba = [number, number, number, number];

export interface Origin {
  xyz: Vec3;
  rpy: Vec3;
}

export type Geometry =
  | { shape: "box"; size: Vec3 }
  | { shape: "cylinder"; radius: number; length: number }
  | { shape: "sphere"; radius: number }
  | { shape: "mesh"; mesh: string; scale: Vec3 };

export interface Visual {
  origin: Origin;
  material: string | null;
  geometry: Geometry;
}

export interface Link {
  name: string;
  material: string | null;
  mass: number | null;
  visual: Visual | null;
}

export interface Limit {
  lower: number;
  upper: number;
  effort: number;
  velocity: number;
}

export interface Joint {
  name: string;
  type: string;
  parent: string;
  child: string;
  origin: Origin;
  axis: Vec3;
  limit: Limit | null;
  movable: boolean;
}

export type Role = "male" | "female";

export interface Port {
  name: string;
  role: Role;
  interface: string;
  link: string;
  origin: Origin;
  description: string;
}

export interface Module {
  id: string;
  version: string;
  category: string;
  display_name: string;
  description: string;
  tags: string[];
  dof: number;
  mass: number;
  root_link: string;
  links: Link[];
  joints: Joint[];
  ports: Port[];
}

export interface InterfaceDef {
  id: string;
  description: string;
}

export interface CatalogPayload {
  version: number;
  interfaces: InterfaceDef[];
  materials: Record<string, Rgba>;
  modules: Module[];
}

/** The catalog indexed by module id, which is how every caller wants it. */
export type Modules = ReadonlyMap<string, Module>;

export function indexModules(payload: CatalogPayload): Modules {
  return new Map(payload.modules.map((module) => [module.id, module]));
}

// --- the assembly IR (urdfgen.ir.assembly) --------------------------------

export interface PortRef {
  instance: string;
  port: string;
}

export interface ModuleInstance {
  id: string;
  module: string;
  version?: string | null;
  mirror?: "none" | "x" | "y";
}

export interface Connection {
  parent: PortRef;
  child: PortRef;
}

export interface AssemblySpec {
  name: string;
  root: string;
  modules: ModuleInstance[];
  connections: Connection[];
  notes?: string | null;
}
