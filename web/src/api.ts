/**
 * The typed client for `urdfgen.api`.
 *
 * Every request goes to a same-origin `/api/...`: uvicorn serves this app, and
 * the dev server proxies. There is no base URL to configure and no CORS.
 */

import type { AssemblySpec, CatalogPayload } from "./catalog";

export interface Finding {
  code: string;
  severity: "error" | "warning";
  message: string;
  path: string;
}

export interface Report {
  findings: Finding[];
  errors: number;
  warnings: number;
  ok: boolean;
}

export interface ModuleUse {
  id: string;
  version: string;
}

export interface Summary {
  name: string;
  root_link: string;
  links: number;
  joints: number;
  movable_joints: number;
  mass: number;
  unconnected_ports: string[];
  modules_used: ModuleUse[];
}

export interface BuildResponse {
  urdf: string;
  summary: Summary;
  report: Report;
}

export interface AssembleResponse {
  spec: AssemblySpec | null;
  report: Report;
  attempts: number;
  succeeded: boolean;
  schema_enforced: boolean;
}

export interface LLMStatus {
  configured: boolean;
  model: string | null;
  base_url: string | null;
  detail: string | null;
}

export interface Health {
  status: "ok";
  payload_version: number;
  modules: number;
  catalog_root: string | null;
  web_app: boolean;
  llm: LLMStatus;
}

/**
 * A request the server refused, carrying what it said.
 *
 * The API answers a rejected build with the validation report rather than a
 * bare status, so the screen can show the findings that caused it.
 */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly report: Report | null;

  constructor(status: number, code: string, message: string, report: Report | null = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.report = report;
  }
}

export const health = (): Promise<Health> => get<Health>("/api/health");
export const catalog = (): Promise<CatalogPayload> => get<CatalogPayload>("/api/catalog");

export const defaultAssembly = async (): Promise<AssemblySpec> =>
  (await get<{ spec: AssemblySpec }>("/api/assembly/default")).spec;

export const validate = (spec: AssemblySpec, signal?: AbortSignal): Promise<Report> =>
  post<Report>("/api/validate", { spec }, signal);

export const build = (spec: AssemblySpec, packageName?: string): Promise<BuildResponse> =>
  post<BuildResponse>("/api/build", {
    spec,
    ...(packageName ? { package_name: packageName } : {}),
  });

export const assemble = (text: string): Promise<AssembleResponse> =>
  post<AssembleResponse>("/api/assemble", { text });

async function get<T>(path: string): Promise<T> {
  return unwrap<T>(await fetch(path, { headers: { accept: "application/json" } }));
}

async function post<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  return unwrap<T>(
    await fetch(path, {
      method: "POST",
      headers: { "content-type": "application/json", accept: "application/json" },
      body: JSON.stringify(body),
      ...(signal ? { signal } : {}),
    }),
  );
}

async function unwrap<T>(response: Response): Promise<T> {
  if (response.ok) return (await response.json()) as T;

  let code = `http_${response.status}`;
  let message = response.statusText || "the request failed";
  let report: Report | null = null;
  try {
    const detail: unknown = (await response.json()).detail;
    if (typeof detail === "string") {
      message = detail;
    } else if (Array.isArray(detail)) {
      // FastAPI's own schema rejection, which means this client sent something
      // the API does not accept -- a bug here, not a bad assembly.
      code = "bad_request";
      message = "the request body was rejected by the API";
    } else if (detail && typeof detail === "object") {
      const body = detail as { error?: string; message?: string; report?: Report };
      code = body.error ?? code;
      message = body.message ?? message;
      report = body.report ?? null;
    }
  } catch {
    // A non-JSON error body is still an error; the status is what matters.
  }
  throw new ApiError(response.status, code, message, report);
}
