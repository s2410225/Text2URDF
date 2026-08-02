/**
 * The assembly screen.
 *
 * Two boundaries decide the shape of this file, and they are the same two the
 * standalone page keeps:
 *
 *   * It never writes URDF. What it produces is an `AssemblySpec`, exactly like
 *     the LLM layer does, and `POST /api/build` turns that into XML.
 *   * It never re-implements the validation layer. Whether a part fits a
 *     hardpoint is decided here, because that has to be interactive; everything
 *     else is `POST /api/validate`, debounced, on the committed unit.
 *
 * The API is what the React version buys over the standalone page: the equip
 * check is now the real validation layer rather than a JS impression of it, the
 * URDF comes from the real builder, and a sentence of natural language can
 * reach the real assembly layer.
 */

import { useCallback, useEffect, useMemo, useState } from "react";

import * as api from "./api";
import { ApiError, type Report, type Summary } from "./api";
import {
  equip,
  figures,
  fitTest,
  jointHandles,
  slots as slotsOf,
  type Figures,
  type Slot,
} from "./assembly";
import type { AssemblySpec, CatalogPayload, Module, Modules } from "./catalog";
import { SUPPORTED_PAYLOAD_VERSION, indexModules } from "./catalog";
import { Confirm, Findings, Hardpoints, Joints, Meters, Parts } from "./panels";
import Viewer, { type ViewName } from "./Viewer";

const VALIDATE_DEBOUNCE_MS = 250;
const VIEWS: ViewName["name"][] = ["iso", "front", "side", "top"];

/** A part chosen but not yet installed. */
interface Pending {
  slotKey: string;
  moduleId: string;
  spec: AssemblySpec;
}

interface Loaded {
  payload: CatalogPayload;
  modules: Modules;
  start: AssemblySpec;
  llm: api.LLMStatus;
}

export default function App(): React.ReactElement {
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const [bootError, setBootError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    void (async () => {
      try {
        const [payload, start, health] = await Promise.all([
          api.catalog(),
          api.defaultAssembly(),
          api.health(),
        ]);
        if (!live) return;
        if (payload.version !== SUPPORTED_PAYLOAD_VERSION) {
          throw new Error(
            `this app speaks catalog payload v${SUPPORTED_PAYLOAD_VERSION}, the server ` +
              `serves v${payload.version}. Rebuild the web app against this server.`,
          );
        }
        setLoaded({ payload, modules: indexModules(payload), start, llm: health.llm });
      } catch (error) {
        if (live) setBootError(error instanceof Error ? error.message : String(error));
      }
    })();
    return () => {
      live = false;
    };
  }, []);

  if (bootError) {
    return (
      <div className="boot">
        <div className="brand">
          urdfgen <span>· assembly</span>
        </div>
        <div>the api did not answer</div>
        <div className="why">{bootError}</div>
        <div className="why">
          Start it with <code>urdfgen serve</code>, or run <code>npm run dev</code> with the
          server on port 8000.
        </div>
      </div>
    );
  }

  if (!loaded) {
    return (
      <div className="boot">
        <div className="brand">
          urdfgen <span>· assembly</span>
        </div>
        <div>reading the catalog…</div>
      </div>
    );
  }

  return <Garage {...loaded} />;
}

function Garage({ payload, modules, start, llm }: Loaded): React.ReactElement {
  const [unit, setUnit] = useState<AssemblySpec>(start);
  const [pending, setPending] = useState<Pending | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [listOpen, setListOpen] = useState(false);
  const [joints, setJoints] = useState<Record<string, number>>({});
  const [view, setView] = useState<ViewName["name"]>("iso");
  const [frameToken, setFrameToken] = useState(0);

  const [report, setReport] = useState<Report | null>(null);
  const [built, setBuilt] = useState<Summary | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [busy, setBusy] = useState<"" | "build" | "assemble">("");
  const [text, setText] = useState("");

  // What is on screen: the pending unit while previewing, the committed one
  // otherwise. The spec panel and the sliders deliberately keep showing the
  // committed unit, so what you can send to the builder is what you built.
  const shown = pending ? pending.spec : unit;

  const slots = useMemo(() => slotsOf(unit, modules), [unit, modules]);
  const shownSlots = useMemo(
    () => (pending ? slotsOf(shown, modules) : slots),
    [pending, shown, slots, modules],
  );
  const handles = useMemo(() => jointHandles(unit, modules), [unit, modules]);
  const now = useMemo<Figures>(() => figures(unit, modules), [unit, modules]);
  const preview = useMemo<Figures | null>(
    () => (pending ? figures(pending.spec, modules) : null),
    [pending, modules],
  );

  const slot = slots.find((entry) => entry.key === selected) ?? null;
  const specJson = useMemo(() => JSON.stringify(unit, null, 2), [unit]);

  // --- the equip check, from the server -----------------------------------
  useEffect(() => {
    const abort = new AbortController();
    const timer = setTimeout(() => {
      api
        .validate(unit, abort.signal)
        .then(setReport)
        .catch((error: unknown) => {
          if (!abort.signal.aborted) {
            setNote(error instanceof Error ? error.message : "validation failed");
          }
        });
    }, VALIDATE_DEBOUNCE_MS);

    return () => {
      clearTimeout(timer);
      abort.abort();
    };
  }, [unit]);

  // A part removed by an install takes its joint values with it.
  useEffect(() => {
    setJoints((current) => {
      const live = new Set(handles.map((handle) => handle.key));
      const next = Object.fromEntries(
        Object.entries(current).filter(([key]) => live.has(key)),
      );
      return Object.keys(next).length === Object.keys(current).length ? current : next;
    });
  }, [handles]);

  const moduleOf = useCallback(
    (instanceId: string): string | null =>
      unit.modules.find((entry) => entry.id === instanceId)?.module ?? null,
    [unit],
  );

  const onPreview = useCallback(
    (module: Module) => {
      if (!slot || !fitTest(module, slot).ok) return;
      setPending({ slotKey: slot.key, moduleId: module.id, spec: equip(unit, modules, slot, module.id) });
    },
    [slot, unit, modules],
  );

  const onInstall = useCallback(() => {
    if (!pending) return;
    setUnit(pending.spec);
    setPending(null);
    setSelected(null);
    setListOpen(false);
    setBuilt(null);
    setNote(null);
  }, [pending]);

  const onSelect = useCallback((next: Slot) => {
    setSelected(next.key);
    setListOpen(false);
    setPending(null);
  }, []);

  const onBuild = useCallback(async () => {
    setBusy("build");
    setNote(null);
    try {
      const result = await api.build(unit);
      setBuilt(result.summary);
      download(`${unit.name}.urdf`, result.urdf);
      setNote(
        `${result.summary.links} links, ${result.summary.joints} joints, ` +
          `${result.summary.movable_joints} movable — downloaded.`,
      );
    } catch (error) {
      setBuilt(null);
      setNote(
        error instanceof ApiError
          ? `${error.message} (${error.report?.errors ?? 0} error(s))`
          : String(error),
      );
    } finally {
      setBusy("");
    }
  }, [unit]);

  const onAssemble = useCallback(async () => {
    if (!text.trim()) return;
    setBusy("assemble");
    setNote(null);
    try {
      const result = await api.assemble(text);
      if (!result.spec) {
        setNote("the model produced nothing that parsed as an assembly");
        return;
      }
      setUnit(result.spec);
      setPending(null);
      setSelected(null);
      setBuilt(null);
      setFrameToken((token) => token + 1);
      setNote(
        result.succeeded
          ? `assembled in ${result.attempts} attempt(s)` +
              (result.schema_enforced ? "" : " — the server did not enforce the schema")
          : `assembled with ${result.report.errors} error(s); edit it here`,
      );
    } catch (error) {
      setNote(error instanceof ApiError ? error.message : String(error));
    } finally {
      setBusy("");
    }
  }, [text]);

  return (
    <div className="app">
      <div className="topbar">
        <div className="brand">
          urdfgen <span>· assembly</span>
        </div>
        <div className="stat-inline">
          unit <b>{unit.name}</b>
        </div>
        <div className="stat-inline">
          parts <b>{now.parts}</b>
        </div>
        <div className="stat-inline">
          mass <b>{now.mass.toFixed(3)}</b> kg
        </div>
        <div className="stat-inline">
          dof <b>{now.dof}</b>
        </div>
        <div className="spacer" />
        <div className="stat-inline">
          catalog <b>{payload.modules.length}</b> modules
        </div>
      </div>

      <div className="cols">
        <div className="col scroll">
          <Hardpoints
            slots={slots}
            selected={selected}
            open={listOpen}
            pendingAt={pending?.slotKey ?? null}
            onToggle={() => setListOpen((open) => !open)}
            onSelect={onSelect}
          />
          {pending && (
            <Confirm
              moduleId={pending.moduleId}
              slotKey={pending.slotKey}
              massDelta={(preview?.mass ?? now.mass) - now.mass}
              onInstall={onInstall}
              onCancel={() => setPending(null)}
            />
          )}
          <Parts
            slot={slot}
            modules={modules}
            pendingModule={pending?.moduleId ?? null}
            onPreview={onPreview}
          />
        </div>

        <div className="col">
          <div className="viewport">
            <Viewer
              spec={shown}
              modules={modules}
              materials={payload.materials}
              joints={joints}
              selected={selected}
              view={view}
              frameToken={frameToken}
            />
            <span className="bracket tl" />
            <span className="bracket tr" />
            <span className="bracket bl" />
            <span className="bracket br" />
            <div className="scan" />
            <div className="viewtag">
              view <em>{view}</em> · {shownSlots.filter((s) => !s.occupied).length} open
            </div>
          </div>

          <div className="controls">
            {VIEWS.map((name) => (
              <button
                type="button"
                key={name}
                className="btn"
                aria-pressed={view === name}
                onClick={() => setView(name)}
              >
                {name}
              </button>
            ))}
            <button type="button" className="btn" onClick={() => setFrameToken((t) => t + 1)}>
              Re-frame
            </button>
            <button type="button" className="btn" onClick={() => setJoints({})}>
              Zero pose
            </button>
            <div className="spacer" />
            <button
              type="button"
              className="btn"
              disabled={busy !== ""}
              onClick={() => {
                setUnit(start);
                setPending(null);
                setSelected(null);
                setBuilt(null);
                setJoints({});
                setNote(null);
              }}
            >
              Reset unit
            </button>
          </div>

          <div className="dock">
            <div className="dock-pane">
              <div className="panel-head">
                <span>Assembly spec</span>
                <span style={{ display: "flex", gap: "0.35rem" }}>
                  <button
                    type="button"
                    className="btn"
                    onClick={() => void navigator.clipboard?.writeText(specJson)}
                  >
                    Copy
                  </button>
                  <button
                    type="button"
                    className="btn install"
                    disabled={busy !== ""}
                    onClick={() => void onBuild()}
                  >
                    {busy === "build" ? "Building…" : "Build URDF"}
                  </button>
                </span>
              </div>
              <div className="spec">
                <pre>{specJson}</pre>
              </div>
              {note && <div className="prompt-note">{note}</div>}
            </div>

            <div className="dock-pane">
              <div className="panel-head">
                <span>Unit status</span>
                <b style={{ color: statusColour(report) }}>{statusFlag(report)}</b>
              </div>
              <Meters now={now} preview={preview} />
              <div className="panel-head">
                <span title="urdfgen validate, run on the server">Equip check</span>
                <b>{report ? report.findings.length : "—"}</b>
              </div>
              <Findings report={report} stale={pending !== null} />
            </div>
          </div>
        </div>

        <div className="col">
          <div className="panel-head">
            <span title="Natural language to AssemblySpec, through the local model">
              Describe it
            </span>
            <b>{llm.configured ? (llm.model ?? "llm") : "offline"}</b>
          </div>
          <div className="prompt">
            <textarea
              value={text}
              disabled={!llm.configured || busy !== ""}
              placeholder={
                llm.configured
                  ? "a four-legged robot with an arm and a gripper on top"
                  : "no local model configured"
              }
              onChange={(event) => setText(event.target.value)}
            />
            <div className="prompt-row">
              <span className="prompt-note">
                {llm.configured
                  ? "The model only picks modules and ports. It never writes URDF."
                  : "Set LLM_MODEL and restart the server."}
              </span>
              <div className="spacer" />
              <button
                type="button"
                className="btn"
                disabled={!llm.configured || busy !== "" || !text.trim()}
                onClick={() => void onAssemble()}
              >
                {busy === "assemble" ? "Thinking…" : "Assemble"}
              </button>
            </div>
          </div>

          <div className="panel-head">
            <span title="Each joint sits under the joint that carries it">
              Joints · nested by driver
            </span>
            <b>{handles.length}</b>
          </div>
          <Joints
            handles={handles}
            values={joints}
            moduleOf={moduleOf}
            onChange={(key, value) => setJoints((current) => ({ ...current, [key]: value }))}
          />

          {built && (
            <div className="panel-head">
              <span>Last build</span>
              <b>
                {built.links}L / {built.joints}J
              </b>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function statusFlag(report: Report | null): string {
  if (!report) return "···";
  if (report.errors > 0) return "FAULT";
  return report.warnings > 0 ? "CHECK" : "NOMINAL";
}

function statusColour(report: Report | null): string {
  if (!report) return "var(--dim)";
  if (report.errors > 0) return "var(--bad)";
  return report.warnings > 0 ? "var(--amber)" : "var(--good)";
}

function download(name: string, text: string): void {
  const url = URL.createObjectURL(new Blob([text], { type: "application/xml" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  URL.revokeObjectURL(url);
}
