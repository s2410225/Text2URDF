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
      ↓  1. assembly layer (local LLM, catalog-ID enum constraints)
  AssemblySpec (module instances + port connections)
      ↓  2. validation layer (deterministic, no LLM)
  validated AssemblySpec  ──┐ on failure, errors are fed back to 1 (max 3 attempts)
      ↓  3. expansion + builder (deterministic)
  URDF XML
      ↓  4. web 3D preview
```

The model runs **locally**. There is no cloud API and no vendor SDK: the
assembly layer speaks the OpenAI chat-completions protocol over plain HTTP, so
Ollama, LM Studio, vLLM and llama.cpp's server all work by pointing an
environment variable at them.

## Status

| Phase | Scope | State |
|---|---|---|
| 0 | Catalog foundation, single module → URDF | implemented |
| 1 | Expansion engine (transforms, namespacing, multi-module assembly) | implemented |
| 2 | Validation layer | implemented |
| 3 | Assembly layer (local LLM, two-stage generation, repair loop) | implemented |
| 4 | Web UI | partial — a generated assembly page, no React yet |
| 5 | Advanced features (tree editing, ROS 2 export, catalog search) | not started |

Everything except the web UI works from the command line today. Phases 0–2 need
nothing but Python; phase 3 needs a local model running somewhere.
`src/urdfgen/api/`, `builder/ros_package.py` and `web/` arrive in phases 4–5.

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
  llm/config.py               endpoint settings, read from the environment
  llm/client.py               the only network boundary
  llm/schema_gen.py           enum-constrained JSON Schemas built from the catalog
  llm/prompts.py              system prompt, per-stage instructions, error feedback
  llm/assemble.py             two-stage generation
  llm/repair.py               validate, feed errors back, retry
  catalog/payload.py          the catalog as JSON, for a UI or the API
  preview/scene.py            projects a built URDF into a 2D scene
  preview/svg.py              scene to SVG
  preview/raster.py           scene to PNG
  ui/page.py                  generates the standalone assembly screen
  ui/assets/assembly.html     its template
  cli.py
tests/
  stub_llm.py                 an OpenAI-compatible stub server
  fixtures/modules/           minimal catalog for edge cases
  fixtures/specs/             hand-written AssemblySpec documents
  fixtures/expected/          golden URDF output
```

The shipped catalog is five modules: a quadruped chassis, a left and a right
3-DOF leg, a 3-DOF arm and a parallel-jaw gripper. Together they cover every
row of the spec's acceptance table.

## Usage

These need no model:

```bash
uv sync --dev

# browse the catalog
uv run urdfgen list
uv run urdfgen show leg_3dof_left_v1
uv run urdfgen digest              # exactly what the model is shown

# build a single module into URDF
uv run urdfgen build-module leg_3dof_left_v1 -o leg.urdf

# validate and build an AssemblySpec
uv run urdfgen validate tests/fixtures/specs/quadruped_a.json
uv run urdfgen build tests/fixtures/specs/quadruped_a.json -o quadruped.urdf

# look at the result
uv run urdfgen preview quadruped.urdf -o quadruped.png
uv run urdfgen preview quadruped.urdf -o crouched.svg --view side \
    --joint leg_fl_knee_pitch=-1.2 --joint leg_fr_knee_pitch=1.2
```

`validate` exits non-zero when it reports an error, and `build` refuses to
write a URDF for a spec that fails validation. `--skip-validation` overrides
that when you want to see what a rejected spec would have produced.

`preview` draws a URDF with no browser, GPU or display involved, so it works
over SSH and in CI. The output extension picks the format — `.png` to look at
somewhere that previews images, `.svg` to keep it sharp at any zoom and diffable
as text — and `--format` overrides it. Both come from the same projection, so
they always show the same thing. `--view` takes `iso`, `front`, `side` or `top`;
the last three are exactly axis-aligned, which means a robot that is symmetric
about a plane must project symmetrically — an asymmetry in the picture is an
asymmetry in the robot. `--joint NAME=VALUE` is repeatable and moves the robot
into a pose, which is the fastest way to check that an assembly articulates the
way it should. This is a complement to the interactive three.js viewer coming
in phase 4, not a replacement for it.

The assembly screen — pick a hardpoint, fit a part, watch the numbers move:

```bash
uv run urdfgen ui -o assembly.html   # then open it
```

These need a local model (see below):

```bash
uv run urdfgen assemble "4脚ロボットを作って" -o spec.json
uv run urdfgen generate "a four-legged robot with an arm on top" -o robot.urdf
```

`assemble` stops at the `AssemblySpec` so you can read and edit it; `generate`
goes straight to URDF. Both accept Japanese or English. If the model cannot
produce something that validates in three attempts, `assemble` still writes the
best spec it got — with a non-zero exit — so you can fix the last mile by hand.

## The assembly screen

`urdfgen ui` writes one self-contained HTML file: no CDN, no build step, no
server. It opens from a `file://` URL.

The interaction is a parts garage. The left rail lists every free hardpoint on
the machine — which is just every unused port on every fitted part. Selecting
one shows the whole catalog, with the parts that cannot attach struck through
and told why: `Interface mismatch — needs TOOL_M20, part has MOUNT_A40` or
`Both sides are female`. Those are the section 2.4 pairing rules, not a
decoration. Hovering a part previews what fitting it would do to the mass and
degrees of freedom before you commit.

Two boundaries are deliberate:

* **The page never writes URDF.** It produces an `AssemblySpec`, exactly like
  the LLM layer does, and `urdfgen build` turns that into XML. The same rule
  applies for the same reason.
* **The page is generated from the catalog, never hand-written.** Copying the
  module data into the HTML would fork the single source of truth.

It also does not re-implement the validation layer. What it checks live is the
equip-time rules — interface, role, port occupancy — because those have to be
interactive. Mass, inertia, interference and the URDF round trip stay in Python;
copy the spec out and run `urdfgen validate` on it.

This is the phase 4 web UI in a form that works today. The React and three.js
viewer the spec describes will supersede it, and the catalog payload it consumes
(`urdfgen`'s `catalog_payload`) is the shape `GET /api/catalog` will return.

## Running a local model

Set the endpoint and model, then use `assemble` or `generate`:

```bash
# Ollama
ollama serve && ollama pull qwen2.5:7b
export LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen2.5:7b

# LM Studio (start the local server from the UI)
export LLM_BASE_URL=http://localhost:1234/v1 LLM_MODEL=qwen2.5-7b-instruct

# vLLM
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
export LLM_BASE_URL=http://localhost:8000/v1 LLM_MODEL=Qwen/Qwen2.5-7B-Instruct

# llama.cpp server
llama-server -m model.gguf --port 8080
export LLM_BASE_URL=http://localhost:8080/v1 LLM_MODEL=local
```

| Variable | Default | Meaning |
|---|---|---|
| `LLM_MODEL` | **required** | Model name. Never hardcoded, so there is no fallback. |
| `LLM_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible endpoint. |
| `LLM_API_KEY` | `local` | Sent as a bearer token; most local servers ignore it. |
| `LLM_TIMEOUT` | `120` | Seconds. Raise it for CPU inference. |
| `LLM_TEMPERATURE` | `0.0` | Zero so the same request gives the same robot. |
| `LLM_MAX_TOKENS` | `2048` | An `AssemblySpec` is a few dozen lines. |
| `LLM_STRUCTURED_OUTPUT` | `auto` | `auto`, `json_schema` or `json_object`. |

`--model` and `--base-url` override the first two per invocation.

### How the model is constrained

Generation happens in two calls (spec section 4.3). The first picks modules,
with `module` restricted to a JSON Schema `enum` built from the catalog at
runtime. The second is then given the exact list of ports those modules have —
`body.leg_mount_fl`, `leg_fl.base`, … — and picks connections from that enum.
Neither call can name a part or a port that does not exist.

That is a strong constraint, not a guarantee. A small model can ignore an enum,
so every field is re-checked afterwards, and the result goes through the same
validation layer as a hand-written spec. Failures are fed back with the
checker's own messages and retried up to three times.

Local runtimes vary in how well they enforce `response_format`. If yours
rejects schema-constrained generation the client falls back to plain JSON mode,
moves the schema into the prompt, and warns you — because in that mode the
catalog enum was a suggestion rather than a constraint. Set
`LLM_STRUCTURED_OUTPUT=json_schema` to make an unsupporting server an error
instead.

### Choosing a model

Anything that reliably emits JSON works. The schemas are kept deliberately flat
and the prompts short, so 7B-class instruction-tuned models are enough; the
model is choosing from a menu, not designing a robot. Larger models mostly buy
better instance naming and better `notes`.

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

* **Local model instead of the Anthropic API.** The spec names the `anthropic`
  SDK and Tool Use. This runs against a local OpenAI-compatible server over
  `httpx` instead, with no vendor SDK. The constraints the spec cares about are
  unchanged: the model still only emits an `AssemblySpec`, `module` is still an
  enum over the catalog, port references are still restricted to real ports,
  and everything still goes through the same validation and repair loop.
  Two consequences of running locally are visible in the code — schema-
  constrained generation is negotiated rather than assumed, and responses are
  parsed defensively, because small models wrap JSON in prose and code fences.
* **Port roles.** Resolved as described above; the spec's own example violates
  its own pairing rule.
* **Connection joint types.** `Connection.joint` accepts `fixed` (the default),
  `continuous`, `floating` and `planar`. `revolute` and `prismatic` are
  rejected, because URDF requires limits for them and a connection has no
  source of limits. Articulation belongs inside a module, which is what the
  spec says anyway.
* **Port references as single strings.** The IR keeps `PortRef` as an object,
  but the model is asked for `"body.leg_mount_fl"` and the string is split
  deterministically. One enum of legal values constrains far better than two
  independent free-text fields, which matters more on a small model.

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

The whole suite runs without a model. Tests of the assembly layer start a small
OpenAI-compatible stub server (`tests/stub_llm.py`) and serve real HTTP, so the
request bodies — including whether the enum constraints actually carry the
catalog — are covered, not mocked away.

To check against a real model instead:

```bash
LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen2.5:7b \
    uv run pytest -m integration -q
```

Those are deselected by default and skip when `LLM_MODEL` is unset.

`catalog`, `validation` and `builder` are written as side-effect-free pure
functions. File I/O lives in `catalog/loader.py` and network access in
`llm/client.py`; nothing else touches either. Validation failures are returned
as `ValidationError` lists, never raised.
