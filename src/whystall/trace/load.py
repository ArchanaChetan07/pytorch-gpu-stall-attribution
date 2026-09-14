"""Load a Kineto trace into a TraceView, stripping ground-truth labels as it goes.

The one non-obvious thing this module does is refuse to hand back a TraceView
that still contains WHYSTALL annotations. Ground-truth labels and the classifier
never meet, because the loader physically separates them. See
whystall.validate.firewall for why that is a type-level property here rather
than a convention.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import re
from pathlib import Path
from typing import Any

from .schema import (
    CPU_CATEGORIES,
    GPU_CATEGORIES,
    LABEL_PREFIX,
    RUNTIME_CATEGORIES,
    Event,
    Kind,
    Label,
    TraceView,
)

__all__ = ["load_trace", "load_trace_dict", "memcpy_kind", "PROFILER_STEP_RE"]

PROFILER_STEP_RE = re.compile(r"^ProfilerStep\s*#?\s*(\d+)")

# Kineto spells memcpy events like "Memcpy HtoD (Pageable -> Device)". The
# pageable/pinned half matters: a pageable host source makes the copy
# synchronous, which is a different finding from a slow but async pinned copy.
_MEMCPY_RE = re.compile(r"Memcpy\s+(\w+)\s*(?:\(([^)]*)\))?", re.IGNORECASE)


def memcpy_kind(name: str) -> tuple[str | None, str | None]:
    """Return (direction, host_memory_kind) for a memcpy event name.

    direction is like HtoD / DtoH / DtoD; host_memory_kind is pageable, pinned
    or None when the trace did not say.
    """
    m = _MEMCPY_RE.search(name)
    if not m:
        return None, None
    direction = m.group(1)
    detail = (m.group(2) or "").lower()
    if "pageable" in detail:
        return direction, "pageable"
    if "pinned" in detail:
        return direction, "pinned"
    return direction, None


def _arg(args: dict[str, Any], *names: str) -> Any:
    for n in names:
        if n in args:
            return args[n]
    return None


def _as_int(value: Any) -> int | None:
    """Round, never truncate.

    Chrome-format ts and dur are floats in microseconds and Kineto emits
    fractional values. int() truncates, which biases every boundary downwards by
    up to a microsecond; round() halves the error and leaves it unbiased.
    """
    if value is None:
        return None
    try:
        return round(float(value))
    except (TypeError, ValueError):
        return None


def _duration_ticks(value: Any) -> tuple[int, bool]:
    """Event duration in whole microseconds, and whether it was quantised.

    A positive duration is never allowed to round to zero. Timestamps here are
    integer microseconds (see the module note in schema.py), so a 0.4 us kernel
    would otherwise become dur=0 and be dropped from the busy timeline entirely
    by busy_intervals -- manufacturing idle out of real work, on precisely the
    dispatch-bound traces this tool exists to diagnose. Rounding it up to 1 us
    overstates busy time by under a microsecond per event, which is a bounded
    and reportable error rather than a silent one.
    """
    if value is None:
        return 0, False
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return 0, False
    if raw <= 0:
        return 0, False
    ticks = round(raw)
    if ticks == 0:
        return 1, True
    return ticks, raw != ticks


def _classify_category(cat: str) -> int:
    c = cat.lower()
    if c in GPU_CATEGORIES:
        return Kind.GPU
    if c in RUNTIME_CATEGORIES:
        return Kind.RUNTIME
    if c in CPU_CATEGORIES:
        return Kind.CPU
    return Kind.OTHER


def load_trace(
    path: str | os.PathLike[str], *, keep_labels: bool = False
) -> tuple[TraceView, list[Label]]:
    """Read a .json or .json.gz Kineto trace.

    Returns the view and the labels that were removed from it. Pass
    keep_labels=True only from tooling that deliberately wants the annotations
    left in the view -- no analysis path does, and the firewall test asserts it.
    """
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rb") as fh:  # type: ignore[operator]
        doc = json.load(io.TextIOWrapper(fh, encoding="utf-8"))
    view, labels = load_trace_dict(doc, keep_labels=keep_labels)
    view.source = str(p)
    return view, labels


def load_trace_dict(
    doc: dict[str, Any], *, keep_labels: bool = False
) -> tuple[TraceView, list[Label]]:
    """Same as load_trace, for an already-parsed document."""
    view = TraceView()
    view.meta = {k: v for k, v in doc.items() if k != "traceEvents"}
    labels: list[Label] = []
    quantised = 0
    rescued = 0

    for raw in doc.get("traceEvents", ()):
        if raw.get("ph") != "X":
            continue
        name = raw.get("name") or ""
        ts = _as_int(raw.get("ts"))
        if ts is None:
            continue
        raw_dur = raw.get("dur")
        dur, was_quantised = _duration_ticks(raw_dur)
        if was_quantised:
            quantised += 1
            try:
                if 0 < float(raw_dur) < 0.5:
                    rescued += 1
            except (TypeError, ValueError):
                pass
        pid = _as_int(raw.get("pid")) or 0
        tid = _as_int(raw.get("tid")) or 0

        if name.startswith(LABEL_PREFIX):
            lab = Label.parse(name, pid, tid, ts, dur)
            if lab is not None:
                labels.append(lab)
            if not keep_labels:
                continue

        cat = raw.get("cat") or ""
        kind = _classify_category(cat)
        if kind == Kind.OTHER:
            continue

        args = raw.get("args") or {}
        ev = Event(
            name=name,
            cat=cat,
            kind=kind,
            pid=pid,
            tid=tid,
            ts=ts,
            dur=dur,
            correlation=_as_int(_arg(args, "correlation", "Correlation")),
            external_id=_as_int(_arg(args, "External id", "external id", "External Id")),
            stream=_as_int(_arg(args, "stream", "Stream")),
            device=_as_int(_arg(args, "device", "Device")),
            nbytes=_as_int(_arg(args, "bytes", "Bytes")),
        )

        if kind == Kind.GPU:
            if ev.device is None:
                # Kineto puts the device in pid for device rows when it does not
                # repeat it in args.
                ev.device = pid
            if ev.stream is None:
                ev.stream = tid
            view.gpu.append(ev)
        elif kind == Kind.RUNTIME:
            view.runtime.append(ev)
            if ev.correlation is not None:
                # Keep the earliest runtime call for a correlation id: that is
                # the launch, not a later synchronise that reuses the id.
                prev = view.launch_by_correlation.get(ev.correlation)
                if prev is None or ev.ts < prev.ts:
                    view.launch_by_correlation[ev.correlation] = ev
        else:
            view.cpu.append(ev)
            if PROFILER_STEP_RE.match(name):
                view.steps.append(ev)

    view.gpu.sort(key=lambda e: (e.ts, e.end))
    view.runtime.sort(key=lambda e: e.ts)
    view.steps.sort(key=lambda e: e.ts)
    view.index_threads()
    if quantised:
        # Carried so that every report derived from this trace can state its own
        # resolution rather than implying more precision than it has.
        view.meta["whystall_quantisation"] = {
            "resolution_us": 1,
            "events_rounded": quantised,
            "events_below_half_us": rescued,
        }
    return view, labels
