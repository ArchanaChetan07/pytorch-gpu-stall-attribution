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
    NS_PER_US,
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
    """Parse a plain integer field (pid, tid, correlation, bytes)."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _us_to_ns(value: Any) -> int | None:
    """Convert a microsecond field to integer nanoseconds.

    Chrome format gives ts and dur as floats in microseconds and Kineto emits
    fractional values, so this is where the document's unit becomes this
    package's. Scaling by 1000 before rounding keeps everything the trace
    recorded: a 0.4 us kernel becomes 400 ns rather than being truncated to
    zero and dropped from the busy timeline, which is how an earlier version
    fabricated idle out of real work.
    """
    if value is None:
        return None
    try:
        return round(float(value) * NS_PER_US)
    except (TypeError, ValueError):
        return None


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
    sub_us = 0

    for raw in doc.get("traceEvents", ()):
        if raw.get("ph") != "X":
            continue
        name = raw.get("name") or ""
        ts_ns = _us_to_ns(raw.get("ts"))
        if ts_ns is None:
            continue
        dur_ns = _us_to_ns(raw.get("dur")) or 0
        if 0 < dur_ns < NS_PER_US:
            sub_us += 1
        pid = _as_int(raw.get("pid")) or 0
        tid = _as_int(raw.get("tid")) or 0

        if name.startswith(LABEL_PREFIX):
            lab = Label.parse(name, pid, tid, ts_ns, dur_ns)
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
            ts_ns=ts_ns,
            dur_ns=dur_ns,
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
                if prev is None or ev.ts_ns < prev.ts_ns:
                    view.launch_by_correlation[ev.correlation] = ev
        else:
            view.cpu.append(ev)
            if PROFILER_STEP_RE.match(name):
                view.steps.append(ev)

    view.gpu.sort(key=lambda e: (e.ts_ns, e.end_ns))
    view.runtime.sort(key=lambda e: e.ts_ns)
    view.steps.sort(key=lambda e: e.ts_ns)
    view.index_threads()
    if sub_us:
        # Informational now rather than a caveat: at nanosecond resolution these
        # events are represented exactly. Kept because a trace full of
        # sub-microsecond kernels is itself a finding -- that shape is what a
        # dispatch-bound workload looks like.
        view.meta["whystall_resolution"] = {
            "unit": "ns",
            "sub_microsecond_events": sub_us,
        }
    return view, labels
