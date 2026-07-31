# Text2URDF

Assemble URDF robot descriptions from a catalog of pre-designed modules
(bodies, legs, arms, grippers), driven by natural-language instructions.

## Design principles

Two constraints define the architecture and must not be weakened:

1. **The LLM never writes URDF/XML.** Its only job is natural language →
   structured intermediate representation (`AssemblySpec`). XML assembly,
   coordinate transforms and consistency checking are done by deterministic
   Python code.
2. **The LLM never invents geometry or mass.** Link shapes, meshes, masses and
   inertia tensors all come from the module definitions. The LLM only chooses
   *which* module, *how many*, and *which port* it attaches to.

```
module catalog (statically designed)
      │
natural-language text
      ↓  1. LLM assembly layer (tool use, catalog-ID enum constraints)
  AssemblySpec (module instances + port connections)
      ↓  2. validation layer (deterministic, no LLM)
  validated AssemblySpec  ──┐ on failure, errors are fed back to 1 (max 3 attempts)
      ↓  3. expansion + builder (deterministic)
  URDF XML
      ↓  4. web 3D preview
```

## Status

| Phase | Scope | State |
|---|---|---|
| 0 | Catalog foundation, single module → URDF | implemented |
| 1 | Expansion engine (transforms, namespacing, multi-module assembly) | implemented |
| 2 | Validation layer | implemented |
| 3 | LLM assembly layer | not started |
| 4 | Web UI | not started |
| 5 | Advanced features (tree editing, ROS 2 export, catalog search) | not started |

Phases 0–2 cover the entire deterministic core: everything works and is fully
tested without an API key. `src/urdfgen/llm/`, `src/urdfgen/api/`,
`builder/ros_package.py` and `web/` arrive in phases 3–5.

Mirroring (`ModuleInstance.mirror`) and parametric modules
(`ModuleDef.parameters`) are listed under phase 5 in the spec, but the fields
exist in the IR from phase 1 onward, so they are implemented here rather than
silently ignored.

## Layout

```
modules/                      module catalog
  interfaces.yaml             mechanical interface definitions
  materials.yaml              shared appearance palette
  <module_id>/module.yaml     one directory per module
src/urdfgen/
  catalog/schema.py           Pydantic models for module.yaml
  catalog/loader.py           the only file-I/O boundary
  catalog/digest.py           LLM-facing catalog summary
  ir/assembly.py              AssemblySpec
  validation/                 pure check functions, return ValidationError lists
  builder/transform.py        homogeneous transforms, RPY, mirroring
  builder/expand.py           module expansion and namespacing
  builder/expression.py       arithmetic evaluator for parametric modules
  builder/inertia.py          primitive-shape inertia fallback
  builder/urdf.py             URDF XML construction
  cli.py
tests/
  fixtures/modules/           minimal catalog for edge cases
  fixtures/specs/             hand-written AssemblySpec documents
  fixtures/expected/          golden URDF output
```

The shipped catalog is five modules: a quadruped chassis, a left and a right
3-DOF leg, a 3-DOF arm and a parallel-jaw gripper. Together they cover every
row of the spec's acceptance table.

## Usage

```bash
uv sync --dev

# browse the catalog
uv run urdfgen list
uv run urdfgen show leg_3dof_left_v1
uv run urdfgen digest              # the summary the LLM layer will be given

# build a single module into URDF
uv run urdfgen build-module leg_3dof_left_v1 -o leg.urdf

# validate and build an AssemblySpec
uv run urdfgen validate tests/fixtures/specs/quadruped_a.json
uv run urdfgen build tests/fixtures/specs/quadruped_a.json -o quadruped.urdf
```

`validate` exits non-zero when it reports an error, and `build` refuses to
write a URDF for a spec that fails validation. `--skip-validation` overrides
that when you want to see what a rejected spec would have produced.

## Port role convention

A connection is legal when both ends share the same `interface` **and** the two
`role`s form a `male`/`female` pair.

The convention used by this catalog: the *receiving* side is `female` (body
`leg_mount_*` and `top`, arm `tool`), and the side that plugs in is `male`
(leg `base`, arm `base`, gripper `base`).

> The design spec is internally inconsistent here: section 2.3 gives the leg's
> `base` port `role: female` while section 4.2 shows the body's `leg_mount_fl`
> as `female` too, which their own pairing rule rejects. The convention above
> resolves it.

## Catalog notes

* Inertia values are always written explicitly in `module.yaml`. They are never
  estimated from meshes.
* The catalog currently ships primitive-shape modules only (`box`, `cylinder`,
  `sphere`); no CAD/mesh assets exist yet. The masses and inertia tensors are
  still authored values in the YAML, computed once from the formulas in
  appendix A of the spec.
* `builder/inertia.py` implements the appendix A fallback, which the spec
  permits *only* for primitive-shape modules that omit `inertial`.
* Left/right asymmetric parts ship as separate modules (`_left` / `_right`).
  Mirroring exists but emits a warning and is a last resort. `leg_3dof_right_v1`
  is the exact Y mirror of `leg_3dof_left_v1`, and a test derives one from the
  other to keep the mirroring rules and the catalog honest.
* A link may omit `inertial.inertia` (but never `inertial.mass`) when all its
  geometry is primitive; the tensor is then computed from the shape. Anything
  with a mesh must declare its tensor.
* Materials are named in `module.yaml` and defined once in `materials.yaml`, so
  the generated URDF never repeats a `<material>` definition.

## Deviations from the spec

Two places where the implementation had to decide something the spec left open
or contradicted itself:

* **Port roles.** Resolved as described above; the spec's own example violates
  its own pairing rule.
* **Connection joint types.** `Connection.joint` accepts `fixed` (the default),
  `continuous`, `floating` and `planar`. `revolute` and `prismatic` are
  rejected, because URDF requires limits for them and a connection has no
  source of limits. Articulation belongs inside a module, which is what the
  spec says anyway.

One place where work was pulled forward from a later phase: `mirror` and
`parameters` are fields of the phase-1 IR, so they are implemented rather than
silently ignored, even though the spec schedules them for phase 5.

## Development

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -q
```

`catalog`, `validation` and `builder` are written as side-effect-free pure
functions. File I/O lives in `catalog/loader.py`; network access will live in
`llm/`. Validation failures are returned as `ValidationError` lists, never
raised.
