"""Event model for Kineto / Chrome-trace-format profiler output.

The PyTorch profiler writes a Chrome Trace Event JSON document. Only a small
part of it matters here, so the loader projects each event onto a slotted
Event and drops the rest -- a realistic training trace runs to millions of
events, and retaining every 'args' dict is the difference between a tool that
opens a 400 MB trace and one that does not.

Timestamps are whole microseconds. Interval arithmetic on integers is exact and
gap boundaries are compared for equality in the tests, which is why the source
document's floats are rounded here rather than carried.

That rounding is a known, bounded approximation and not a free choice. Chrome
format gives ts and dur as floats in microseconds and Kineto emits fractional
values, so loading quantises to 1 us; load.py never lets a positive duration
round to zero, because a vanished kernel becomes fabricated idle. The correct
fix is integer nanoseconds throughout -- float microseconds cannot work, since
Kineto timestamps are epoch-scale and float64 spacing there is already coarser
than a nanosecond -- and that is a unit change across the whole package. Until
it happens, every report states its own resolution; see the known limits in the
README.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass, field

__all__ = [
    "LABEL_PREFIX",
    "Kind",
    "Event",
    "Label",
    "TraceView",
    "GPU_CATEGORIES",
    "CPU_CATEGORIES",
    "RUNTIME_CATEGORIES",
]

#: Every annotation this package emits into a trace starts with this. It is the
#: single string the firewall strips, and the reason the firewall can be
#: structural rather than a flag -- see whystall.validate.firewall.
LABEL_PREFIX = "WHYSTALL/"


class Kind:
    """Normalised event kinds. Plain ints; compared in hot loops."""

    GPU = 0
    CPU = 1
    RUNTIME = 2
    OTHER = 3


# Kineto has renamed these across releases; accept both spellings. Matching is
# done on the lowercased category, so "Kernel" and "kernel" both land.
GPU_CATEGORIES = frozenset(
    {"kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation", "memcpy", "memset"}
)
CPU_CATEGORIES = frozenset(
    {"cpu_op", "user_annotation", "operator", "python_function", "cpu_instant_event"}
)
RUNTIME_CATEGORIES = frozenset({"cuda_runtime", "cuda_driver", "runtime"})


@dataclass(slots=True)
class Event:
    """One complete (ph == X) trace event, projected to what we use."""

    name: str
    cat: str
    kind: int
    pid: int
    tid: int
    ts: int
    dur: int
    #: CUPTI correlation id. Links a runtime launch to the device op it produced.
    #: This is the primary linkage; ac2g flow events are the fallback.
    correlation: int | None = None
    #: Links a runtime event to the enclosing cpu_op.
    external_id: int | None = None
    stream: int | None = None
    device: int | None = None
    nbytes: int | None = None
    #: Nesting depth on its own thread, computed at load time. Depth is how the
    #: blame resolver finds the leaf of the call stack covering a window.
    depth: int = 0

    @property
    def end(self) -> int:
        return self.ts + self.dur

    def overlap(self, start: int, end: int) -> int:
        """Microseconds of this event inside [start, end)."""
        lo = self.ts if self.ts > start else start
        hi = self.end if self.end < end else end
        return hi - lo if hi > lo else 0

    def __repr__(self) -> str:
        return f"Event({self.name!r}, ts={self.ts}, dur={self.dur}, tid={self.tid})"


@dataclass(slots=True, frozen=True)
class Label:
    """A WHYSTALL annotation, stripped out of the trace before classification.

    The 'kind' is the first path segment (inject, clock, run) and 'parts' the
    remainder, so callers apply their own grammar without re-parsing the name.
    """

    kind: str
    parts: tuple[str, ...]
    pid: int
    tid: int
    ts: int
    dur: int

    @property
    def end(self) -> int:
        return self.ts + self.dur

    @classmethod
    def parse(cls, ev_name: str, pid: int, tid: int, ts: int, dur: int) -> Label | None:
        if not ev_name.startswith(LABEL_PREFIX):
            return None
        segs = tuple(s for s in ev_name[len(LABEL_PREFIX):].split("/") if s)
        if not segs:
            return None
        return cls(kind=segs[0], parts=segs[1:], pid=pid, tid=tid, ts=ts, dur=dur)

    def overlap(self, start: int, end: int) -> int:
        lo = self.ts if self.ts > start else start
        hi = self.end if self.end < end else end
        return hi - lo if hi > lo else 0


@dataclass(slots=True)
class TraceView:
    """A loaded trace, indexed for the queries the classifier makes.

    A TraceView never contains WHYSTALL events. The loader removes them and
    returns them separately, which is what makes the ground-truth firewall a
    property of the type rather than a discipline the caller has to maintain.
    """

    #: Device ops (kernels, memcpys, memsets), sorted by ts.
    gpu: list[Event] = field(default_factory=list)
    #: Host ops, sorted by ts.
    cpu: list[Event] = field(default_factory=list)
    #: CUDA runtime/driver calls, sorted by ts.
    runtime: list[Event] = field(default_factory=list)
    #: correlation id -> the runtime call that launched it.
    launch_by_correlation: dict[int, Event] = field(default_factory=dict)
    #: tid -> that thread's host events, sorted by ts.
    cpu_by_tid: dict[int, list[Event]] = field(default_factory=dict)
    #: tid -> that thread's CUDA runtime calls, sorted by ts. Kept apart from
    #: cpu_by_tid because a blocking cudaStreamSynchronize is a runtime event,
    #: not a cpu_op, and the host-sync rule depends on finding it.
    runtime_by_tid: dict[int, list[Event]] = field(default_factory=dict)
    #: ProfilerStep annotations, in order.
    steps: list[Event] = field(default_factory=list)
    #: Everything the document said about itself.
    meta: dict = field(default_factory=dict)
    #: Provenance, for the report header.
    source: str = "<memory>"
    #: Cached ts keys per tid so lookups bisect without rebuilding.
    ts_by_tid: dict[int, list[int]] = field(default_factory=dict, repr=False)
    runtime_ts_by_tid: dict[int, list[int]] = field(default_factory=dict, repr=False)
    #: Running maximum of end times, per thread. max_end[i] is the latest end
    #: among events 0..i, so a backward scan can stop the moment it drops below
    #: the window: nothing earlier can still be open. Replaces a fixed-length
    #: heuristic that was both slow on long traces and able to miss a genuinely
    #: long enclosing frame.
    max_end_by_tid: dict[int, list[int]] = field(default_factory=dict, repr=False)
    runtime_max_end_by_tid: dict[int, list[int]] = field(default_factory=dict, repr=False)

    def devices(self) -> list[int]:
        return sorted({ev.device for ev in self.gpu if ev.device is not None})

    def index_threads(self) -> None:
        """Build the per-thread indices and nesting depths. Idempotent."""
        by_tid: dict[int, list[Event]] = {}
        for ev in self.cpu:
            by_tid.setdefault(ev.tid, []).append(ev)
        self.ts_by_tid = {}
        self.max_end_by_tid = {}
        for tid, evs in by_tid.items():
            # Sort so an enclosing event precedes what it contains: ascending
            # start, then descending duration.
            evs.sort(key=lambda e: (e.ts, -e.dur))
            _assign_depths(evs)
            self.ts_by_tid[tid] = [e.ts for e in evs]
            self.max_end_by_tid[tid] = _running_max_end(evs)
        self.cpu_by_tid = by_tid

        rt_by_tid: dict[int, list[Event]] = {}
        for ev in self.runtime:
            rt_by_tid.setdefault(ev.tid, []).append(ev)
        self.runtime_ts_by_tid = {}
        self.runtime_max_end_by_tid = {}
        for tid, evs in rt_by_tid.items():
            evs.sort(key=lambda e: (e.ts, -e.dur))
            self.runtime_ts_by_tid[tid] = [e.ts for e in evs]
            self.runtime_max_end_by_tid[tid] = _running_max_end(evs)
        self.runtime_by_tid = rt_by_tid

    def runtime_on_thread(self, tid: int, start: int, end: int) -> list[Event]:
        """CUDA runtime calls on tid overlapping [start, end), in time order."""
        return _scan_back(
            self.runtime_by_tid.get(tid),
            self.runtime_ts_by_tid.get(tid),
            self.runtime_max_end_by_tid.get(tid),
            start,
            end,
        )

    def events_on_thread(self, tid: int, start: int, end: int) -> list[Event]:
        """Host events on tid overlapping [start, end), outermost first.

        Events on a thread form a proper stack, but an enclosing frame can start
        arbitrarily far back, so bisecting on start time alone would miss it.
        The scan walks backwards and stops at the first index whose running
        maximum end time has fallen to or below the window start: at that point
        no earlier event can still be open, which is exact rather than a
        heuristic and costs one comparison per step.
        """
        return _scan_back(
            self.cpu_by_tid.get(tid),
            self.ts_by_tid.get(tid),
            self.max_end_by_tid.get(tid),
            start,
            end,
        )

    def bounds(self) -> tuple[int, int]:
        """The window the analysis is confined to.

        Starts at the first ProfilerStep, because idle before profiling began is
        warm-up rather than starvation. Ends at the later of the last step and
        the last device op: the host always finishes a step before the device
        does, and closing the window at the host would silently discard the
        trailing device work -- along with any gap inside it.
        """
        gpu_end = max((e.end for e in self.gpu), default=0)
        if self.steps:
            return self.steps[0].ts, max(max(s.end for s in self.steps), gpu_end)
        if self.gpu:
            return self.gpu[0].ts, gpu_end
        return 0, 0


def _running_max_end(evs: list[Event]) -> list[int]:
    """Prefix maximum of end times, so a backward scan knows when to stop."""
    out: list[int] = []
    best = 0
    for ev in evs:
        end = ev.end
        if end > best:
            best = end
        out.append(best)
    return out


def _scan_back(
    evs: list[Event] | None,
    keys: list[int] | None,
    max_end: list[int] | None,
    start: int,
    end: int,
) -> list[Event]:
    """Events overlapping [start, end), outermost first.

    Exact: the prefix-maximum test stops the scan only when no earlier event can
    still be open, so a long enclosing frame is always found however many short
    events sit between it and the window.
    """
    if not evs or keys is None or max_end is None:
        return []
    hi = bisect_right(keys, end)
    out: list[Event] = []
    for i in range(hi - 1, -1, -1):
        if max_end[i] <= start:
            break
        ev = evs[i]
        if ev.end > start:
            out.append(ev)
    out.reverse()
    return out


def _assign_depths(evs: list[Event]) -> None:
    """Depth via a tolerant stack: malformed nesting must not raise."""
    stack: list[Event] = []
    for ev in evs:
        while stack and stack[-1].end <= ev.ts:
            stack.pop()
        ev.depth = len(stack)
        stack.append(ev)


def sort_events(events: Iterable[Event]) -> list[Event]:
    return sorted(events, key=lambda e: (e.ts, -e.dur))
