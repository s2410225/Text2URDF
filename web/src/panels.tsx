/**
 * The panels around the viewport. Presentational: they take state and call
 * back, and hold nothing of their own.
 */

import { Fragment } from "react";

import type { Figures, JointHandle, Slot } from "./assembly";
import { fitTest } from "./assembly";
import type { Report } from "./api";
import type { Module, Modules } from "./catalog";

// --- hardpoints -----------------------------------------------------------

interface HardpointsProps {
  slots: Slot[];
  selected: string | null;
  open: boolean;
  pendingAt: string | null;
  onToggle: () => void;
  onSelect: (slot: Slot) => void;
}

export function Hardpoints({
  slots,
  selected,
  open,
  pendingAt,
  onToggle,
  onSelect,
}: HardpointsProps): React.ReactElement {
  const current = slots.find((slot) => slot.key === selected) ?? null;
  const free = slots.filter((slot) => !slot.occupied).length;

  return (
    <>
      <div className="panel-head">
        <span title="Every port on every fitted part, nested by what carries it">
          Hardpoints
        </span>
        <b>{slots.length}</b>
      </div>

      <button
        type="button"
        className="slot-current"
        aria-expanded={open}
        data-empty={String(!current || !current.occupied)}
        onClick={onToggle}
      >
        {current ? (
          <span>
            <span className="port">{current.key}</span>
            <br />
            <span className="fitted">{current.fittedModule ?? "— empty —"}</span>
          </span>
        ) : (
          <span>
            <span className="port">Select a hardpoint</span>
            <br />
            <span className="fitted">
              {slots.length} on this unit · {free} open
            </span>
          </span>
        )}
        {current ? <span className="iface">{current.interface}</span> : <span />}
        <span className="chev">{open ? "▲" : "▼"}</span>
      </button>

      {open && (
        <div className="slots">
          {slots.map((slot) => (
            <button
              type="button"
              key={slot.key}
              className={`slot${pendingAt === slot.key ? " pulse" : ""}`}
              style={{ "--d": slot.depth } as React.CSSProperties}
              aria-selected={selected === slot.key}
              data-empty={String(!slot.occupied)}
              onClick={() => onSelect(slot)}
            >
              <span>
                <span className="port">
                  <span className="tree">{slot.depth ? "└" : "▪"}</span>
                  {slot.key}
                </span>
                <br />
                <span className="fitted">{slot.fittedModule ?? "— empty —"}</span>
              </span>
              <span className="iface">{slot.interface}</span>
            </button>
          ))}
        </div>
      )}
    </>
  );
}

// --- parts ----------------------------------------------------------------

interface PartsProps {
  slot: Slot | null;
  modules: Modules;
  pendingModule: string | null;
  onPreview: (module: Module) => void;
}

export function Parts({ slot, modules, pendingModule, onPreview }: PartsProps): React.ReactElement {
  if (!slot) {
    return (
      <>
        <div className="panel-head">
          <span>Parts</span>
          <b>—</b>
        </div>
        <p className="hint">
          Pick a hardpoint to see what fits it. A part is only offered when its
          interface matches and the two roles form a male/female pair — the
          pairing rule from section 2.4, not a decoration.
        </p>
      </>
    );
  }

  const parts = [...modules.values()];
  return (
    <>
      <div className="panel-head">
        <span>Parts for {slot.key}</span>
        <b>{slot.interface}</b>
      </div>
      <div className="parts">
        {parts.map((module) => {
          const fit = fitTest(module, slot);
          return (
            <button
              type="button"
              key={module.id}
              className="part"
              disabled={!fit.ok}
              aria-current={pendingModule === module.id}
              onClick={() => onPreview(module)}
            >
              <span className="part-id">{module.id}</span>
              <span className="part-figs">
                {module.mass.toFixed(2)} kg · {module.dof} dof
              </span>
              {fit.ok ? (
                <span className="part-desc">{module.description}</span>
              ) : (
                <span className="part-why">{fit.why}</span>
              )}
            </button>
          );
        })}
      </div>
    </>
  );
}

// --- the pending install --------------------------------------------------

interface ConfirmProps {
  moduleId: string;
  slotKey: string;
  massDelta: number;
  onInstall: () => void;
  onCancel: () => void;
}

export function Confirm({
  moduleId,
  slotKey,
  massDelta,
  onInstall,
  onCancel,
}: ConfirmProps): React.ReactElement {
  const sign = massDelta >= 0 ? "+" : "−";
  return (
    <div className="confirm">
      <span className="confirm-text">
        <b>{moduleId}</b>
        <span>
          → {slotKey} · {sign}
          {Math.abs(massDelta).toFixed(3)} kg
        </span>
      </span>
      <button type="button" className="btn install" onClick={onInstall}>
        Install
      </button>
      <button type="button" className="btn" onClick={onCancel}>
        Cancel
      </button>
    </div>
  );
}

// --- unit status ----------------------------------------------------------

const METERS = [
  { key: "mass", name: "Total mass", unit: "kg", full: 12, digits: 3 },
  { key: "dof", name: "Degrees of freedom", unit: "", full: 24, digits: 0 },
  { key: "links", name: "Links", unit: "", full: 40, digits: 0 },
  { key: "joints", name: "Joints", unit: "", full: 40, digits: 0 },
  { key: "open", name: "Open hardpoints", unit: "", full: 12, digits: 0 },
] as const;

export function Meters({
  now,
  preview,
}: {
  now: Figures;
  preview: Figures | null;
}): React.ReactElement {
  return (
    <div className="meters">
      {METERS.map((meter) => {
        const value = now[meter.key];
        const after = preview ? preview[meter.key] : value;
        const delta = after - value;
        const shown = preview ? after : value;
        return (
          <div className="meter" key={meter.key}>
            <div className="meter-name">{meter.name}</div>
            <div className="meter-value">
              {shown.toFixed(meter.digits)}
              <u>{meter.unit}</u>
              {Math.abs(delta) > 1e-9 && (
                <span className={`meter-delta ${delta > 0 ? "up" : "down"}`}>
                  {delta > 0 ? "▲" : "▼"}
                  {Math.abs(delta).toFixed(meter.digits)}
                </span>
              )}
            </div>
            <div className={`bar${preview ? " ghost" : ""}`}>
              <i style={{ width: `${Math.min(100, (shown / meter.full) * 100).toFixed(1)}%` }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

// --- equip check ----------------------------------------------------------

/**
 * The validation layer's findings, as returned by the server.
 *
 * Nothing here is computed in the browser. The one thing the app decides for
 * itself is whether a part fits a hardpoint, and that is in `assembly.ts`.
 */
export function Findings({
  report,
  stale,
}: {
  report: Report | null;
  stale: boolean;
}): React.ReactElement {
  if (!report) {
    return (
      <div className="findings">
        <div className="finding">
          <span className="tag">···</span>
          <span>Waiting for the first answer from urdfgen validate.</span>
        </div>
      </div>
    );
  }

  if (report.findings.length === 0) {
    return (
      <div className="findings">
        <div className="finding">
          <span className="tag ok">clear</span>
          <span>
            The validation layer found nothing{stale ? " on the committed unit" : ""}: catalog,
            connections, mass properties, interference and the URDF round trip all pass.
          </span>
        </div>
      </div>
    );
  }

  return (
    <div className="findings">
      {report.findings.map((finding, index) => (
        <div className="finding" key={`${finding.code}-${finding.path}-${index}`}>
          <span className={`tag ${finding.severity === "error" ? "bad" : "warn"}`}>
            {finding.severity}
          </span>
          <span>
            {finding.message}
            {finding.path && <span className="finding-path">{finding.path}</span>}
          </span>
        </div>
      ))}
    </div>
  );
}

// --- joints ---------------------------------------------------------------

interface JointsProps {
  handles: JointHandle[];
  values: Record<string, number>;
  moduleOf: (instance: string) => string | null;
  onChange: (key: string, value: number) => void;
}

export function Joints({ handles, values, moduleOf, onChange }: JointsProps): React.ReactElement {
  let lastInstance = "";
  return (
    <div className="joints">
      {handles.map((handle) => {
        const heading = handle.instance !== lastInstance;
        lastInstance = handle.instance;
        const value = values[handle.key] ?? 0;
        return (
          <Fragment key={handle.key}>
            {heading && (
              <div className="joint-part" style={{ "--d": handle.depth } as React.CSSProperties}>
                <span className="tree">{handle.depth ? "└" : "▪"}</span>
                <b>{handle.instance}</b>
                <span className="mod">{moduleOf(handle.instance) ?? ""}</span>
              </div>
            )}
            <div className="joint" style={{ "--d": handle.depth } as React.CSSProperties}>
              <label htmlFor={handle.key}>{handle.joint}</label>
              <input
                id={handle.key}
                type="range"
                min={handle.lower}
                max={handle.upper}
                step={0.01}
                value={value}
                onChange={(event) => onChange(handle.key, Number(event.target.value))}
              />
              <output>{value.toFixed(2)}</output>
            </div>
          </Fragment>
        );
      })}
    </div>
  );
}
