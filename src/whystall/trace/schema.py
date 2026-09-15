"""Event model for Kineto / Chrome-trace-format profiler output.

The PyTorch profiler writes a Chrome Trace Event JSON document. Only a small
part of it matters here, so the loader projects each event onto a slotted
Event and drops the rest -- a realistic training trace runs to millions of
events, and retaining every 'args' dict is the difference between a tool that
opens a 400 MB trace and one that does not.

Time is integer nanoseconds throughout, and every time-valued name carries an
_ns suffix so a unit error has to be written on purpose.

The source document is not in nanoseconds: Chrome format gives ts and dur as
*floats* in microseconds, and Kineto emits fractional values. Two units were
therefore wrong before this one. Integer microseconds silently dropped every
sub-microsecond kernel, because a 0.4 us duration truncated to zero and vanished
from the busy timeline, fabricating idle out of real work. Float microseconds
cannot work either: Kineto timestamps are epoch-scale, around 1.7e15 us, where
float64 spacing is already coarser than a nanosecond. Scaling to integer
nanoseconds at load keeps interval arithmetic exact, keeps gap boundaries
comparable for equality in the tests, and loses nothing the trace recorded.

User-facing knobs stay in microseconds, because that is the unit people think
in; they convert at the boundary.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass, field

__all__ = [
    "LABEL_PREFIX",
    "NS_PER_US",
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

#: Scale factor between the document's unit and this package's.
NS_PER_US = 1000


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
    ts_ns: int
    dur_ns: int
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
    def end_ns(self) -> int:
        return self.ts_ns + self.dur_ns

    def overlap_ns(self, start_ns: int, end_ns: int) -> int:
        """Nanoseconds of this event inside [start_ns, end_ns)."""
        lo = self.ts_ns if self.ts_ns > start_ns else start_ns
        hi = self.end_ns if self.end_ns < end_ns else end_ns
        return hi - lo if hi > lo else 0

    def __repr__(self) -> str:
        return f"Event({self.name!r}, ts_ns={self.ts_ns}, dur_ns={self.dur_ns}, tid={self.tid})"


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
    ts_ns: int
    dur_ns: int

    @property
    def end_ns(self) -> int:
        return self.ts_ns + self.dur_ns

    @classmethod
    def parse(cls, ev_name: str, pid: int, tid: int, ts_ns: int, dur_ns: int) -> Label | None:
        if not ev_name.startswith(LABEL_PREFIX):
            return None
        segs = tuple(s for s in ev_name[len(LABEL_PREFIX):].split("/") if s)
        if not segs:
            return None
        return cls(kind=segs[0], parts=segs[1:], pid=pid, tid=tid, ts_ns=ts_ns, dur_ns=dur_ns)

    def overlap_ns(self, start_ns: int, end_ns: int) -> int:
        lo = self.ts_ns if self.ts_ns > start_ns else start_ns
        hi = self.end_ns if self.end_ns < end_ns else end_ns
        return hi - lo if hi > lo else 0


@dataclass(slots=True)
class TraceView:
    """A loaded trace, indexed for the queries the classifier makes.

    A TraceView never contains WHYSTALL events. The loader removes them and
    returns them separately, which is what makes the ground-truth firewall a
    property of the type rather than a discipline the caller has to maintain.
    """

    #: Device ops (kernels, memcpys, memsets), sorted by ts_ns.
    gpu: list[Event] = field(default_factory=list)
    #: Host ops, sorted by ts_ns.
    cpu: list[Event] = field(default_factory=list)
    #: CUDA runtime/driver calls, sorted by ts_ns.
    runtime: list[Event] = field(default_factory=list)
    #: correlation id -> the runtime call that launched it.
    launch_by_correlation: dict[int, Event] = field(default_factory=dict)
    #: tid -> that thread's host events, sorted by ts_ns.
    cpu_by_tid: dict[int, list[Event]] = field(default_factory=dict)
    #: tid -> that thread's CUDA runtime calls, sorted by ts_ns. Kept apart from
    #: cpu_by_tid because a blocking cudaStreamSynchronize is a runtime event,
    #: not a cpu_op, and the host-sync rule depends on finding it.
    runtime_by_tid: dict[int, list[Event]] = field(default_factory=dict)
    #: ProfilerStep annotations, in order.
    steps: list[Event] = field(default_factory=list)
    #: Everything the document said about itself.
    meta: dict = field(default_factory=dict)
    #: Provenance, for the report header.
    source: str = "<memory>"
    #: Cached ts_ns keys per tid so lookups bisect without rebuilding.
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
            evs.sort(key=lambda e: (e.ts_ns, -e.dur_ns))
            _assign_depths(evs)
            self.ts_by_tid[tid] = [e.ts_ns for e in evs]
            self.max_end_by_tid[tid] = _running_max_end(evs)
        self.cpu_by_tid = by_tid

        rt_by_tid: dict[int, list[Event]] = {}
        for ev in self.runtime:
            rt_by_tid.setdefault(ev.tid, []).append(ev)
        self.runtime_ts_by_tid = {}
        self.runtime_max_end_by_tid = {}
        for tid, evs in rt_by_tid.items():
            evs.sort(key=lambda e: (e.ts_ns, -e.dur_ns))
            self.runtime_ts_by_tid[tid] = [e.ts_ns for e in evs]
            self.runtime_max_end_by_tid[tid] = _running_max_end(evs)
        self.runtime_by_tid = rt_by_tid

    def events_on_thread(self, tid: int, start_ns: int, end_ns: int) -> list[Event]:
        """Host events on tid overlapping [start_ns, end_ns), outermost first.

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
            start_ns,
            end_ns,
        )

    def runtime_on_thread(self, tid: int, start_ns: int, end_ns: int) -> list[Event]:
        """CUDA runtime calls on tid overlapping the window, in time order."""
        return _scan_back(
            self.runtime_by_tid.get(tid),
            self.runtime_ts_by_tid.get(tid),
            self.runtime_max_end_by_tid.get(tid),
            start_ns,
            end_ns,
        )

    def bounds_ns(self) -> tuple[int, int]:
        """The window the analysis is confined to.

        Starts at the first ProfilerStep, because idle before profiling began is
        warm-up rather than starvation. Ends at the later of the last step and
        the last device op: the host always finishes a step before the device
        does, and closing the window at the host would silently discard the
        trailing device work -- along with any gap inside it.
        """
        gpu_end = max((e.end_ns for e in self.gpu), default=0)
        if self.steps:
            return self.steps[0].ts_ns, max(max(s.end_ns for s in self.steps), gpu_end)
        if self.gpu:
            return self.gpu[0].ts_ns, gpu_end
        return 0, 0


def _running_max_end(evs: list[Event]) -> list[int]:
    """Prefix maximum of end times, so a backward scan knows when to stop."""
    out: list[int] = []
    best = 0
    for ev in evs:
        end = ev.end_ns
        if end > best:
            best = end
        out.append(best)
    return out


def _scan_back(
    evs: list[Event] | None,
    keys: list[int] | None,
    max_end: list[int] | None,
    start_ns: int,
    end_ns: int,
) -> list[Event]:
    """Events overlapping the window, outermost first.

    Exact: the prefix-maximum test stops the scan only when no earlier event can
    still be open, so a long enclosing frame is always found however many short
    events sit between it and the window.
    """
    if not evs or keys is None or max_end is None:
        return []
    hi = bisect_right(keys, end_ns)
    out: list[Event] = []
    for i in range(hi - 1, -1, -1):
        if max_end[i] <= start_ns:
            break
        ev = evs[i]
        if ev.end_ns > start_ns:
            out.append(ev)
    out.reverse()
    return out


def _assign_depths(evs: list[Event]) -> None:
    """Depth via a tolerant stack: malformed nesting must not raise."""
    stack: list[Event] = []
    for ev in evs:
        while stack and stack[-1].end_ns <= ev.ts_ns:
            stack.pop()
        ev.depth = len(stack)
        stack.append(ev)


def sort_events(events: Iterable[Event]) -> list[Event]:
    return sorted(events, key=lambda e: (e.ts_ns, -e.dur_ns))
