"""Resolve an idle window to the host activity that held it open.

The walk is short and the whole method rests on it:

1. take the device op that ends the gap;
2. follow its CUPTI correlation id back to the runtime call that launched it;
3. if that launch happened before the gap opened, the work was already queued
   and the GPU was waiting on the device -- stop, this is not starvation;
4. otherwise, look at what occupied the launching thread between the moment the
   device went idle and the moment it was finally fed. That is the blame span.

Step 3 is the one that is usually skipped, and skipping it inflates every
host-wait number in the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .schema import Event, TraceView
from .timeline import Gap

__all__ = ["Blame", "resolve_blame", "resolve_all"]


@dataclass(slots=True)
class Blame:
    """Everything the classifier is allowed to reason from, for one gap."""

    gap: Gap
    #: The runtime call that launched the op ending the gap.
    launch: Event | None = None
    launcher_tid: int | None = None
    t_launch: int | None = None
    #: True when the launch preceded the gap: a device-side dependency.
    queued_before_gap: bool = False
    #: The interval on the launching thread that the gap is charged against.
    window: tuple[int, int] = (0, 0)
    #: Host events covering the window, outermost first.
    covering: list[Event] = field(default_factory=list)
    #: CUDA runtime calls on the launching thread inside the window.
    runtime_spans: list[Event] = field(default_factory=list)
    #: Deepest covering event with real overlap: the leaf of the call stack.
    leaf: Event | None = None
    #: Share of the window the leaf accounts for. Doubles as confidence for
    #: leaf-driven verdicts.
    leaf_fraction: float = 0.0

    @property
    def window_us(self) -> int:
        return max(0, self.window[1] - self.window[0])

    @property
    def resolved(self) -> bool:
        return bool(self.covering or self.runtime_spans)

    def top_names(self, n: int = 3) -> list[tuple[str, int]]:
        """The names accounting for most of the window, deepest frames first.

        This is what keeps 'other host compute' from being a dustbin: the bucket
        always arrives with the operators that produced it.
        """
        lo, hi = self.window
        totals: dict[str, int] = {}
        for ev in self.covering:
            # Exclude step annotations by name, matching _select_leaf. Excluding
            # by depth instead would drop a genuine top-level cause on any thread
            # that carries no step annotation at all.
            if _is_step(ev.name):
                continue
            ov = ev.overlap(lo, hi)
            if ov > 0:
                totals[ev.name] = max(totals.get(ev.name, 0), ov)
        for ev in self.runtime_spans:
            ov = ev.overlap(lo, hi)
            if ov > 0:
                totals[ev.name] = max(totals.get(ev.name, 0), ov)
        return sorted(totals.items(), key=lambda kv: -kv[1])[:n]


def resolve_blame(view: TraceView, gap: Gap) -> Blame:
    """Resolve one gap. Never raises; an unresolvable gap comes back empty."""
    blame = Blame(gap=gap, window=(gap.start, gap.end))

    nxt = gap.next_gpu
    if nxt is None or nxt.correlation is None:
        return blame

    launch = view.launch_by_correlation.get(nxt.correlation)
    if launch is None:
        return blame

    blame.launch = launch
    blame.launcher_tid = launch.tid
    blame.t_launch = launch.ts

    if launch.ts <= gap.start:
        # Enqueued before the device went idle: the GPU was not waiting on us.
        blame.queued_before_gap = True
        blame.window = (gap.start, gap.start)
        return blame

    lo = gap.start
    hi = min(launch.ts, gap.end)
    blame.window = (lo, hi)
    if hi <= lo:
        return blame

    blame.covering = view.events_on_thread(launch.tid, lo, hi)
    blame.runtime_spans = view.runtime_on_thread(launch.tid, lo, hi)

    span = hi - lo
    best, best_ov = _select_leaf(blame.covering, blame.runtime_spans, lo, hi)
    blame.leaf = best
    blame.leaf_fraction = (best_ov / span) if best is not None and span > 0 else 0.0
    return blame


#: A candidate must retain at least this share of the best overlap to be
#: considered. Below it, going deeper means describing a detail rather than the
#: thing that held the thread.
_LEAF_COVERAGE_TOLERANCE = 0.8

#: Runtime calls nest inside the operator that made them, but carry no depth of
#: their own, so they are ranked as deeper than any host frame. A blocking
#: cudaStreamSynchronize is a more precise answer than the aten::item around it.
_RUNTIME_DEPTH = 10_000


def _select_leaf(
    covering: list[Event], runtime_spans: list[Event], lo: int, hi: int
) -> tuple[Event | None, int]:
    """Pick the frame that actually held the thread.

    Two failure modes to avoid, and they pull in opposite directions.

    Ranking by depth alone picks the innermost frame even when it covers a
    sliver: a 100 us detail inside a 2 ms convolution is not what the GPU was
    waiting for. Ranking by overlap alone picks the outermost frame, because the
    enclosing annotation always covers the most.

    So: go as deep as possible without materially losing coverage. Step
    annotations are excluded outright -- a step is a container, never a cause,
    and letting one win produces evidence that reads "ProfilerStep#7".
    """
    candidates: list[tuple[int, int, Event]] = []
    best_ov = 0
    for ev in covering:
        if _is_step(ev.name):
            continue
        ov = ev.overlap(lo, hi)
        if ov > 0:
            candidates.append((ev.depth, ov, ev))
            best_ov = max(best_ov, ov)
    for ev in runtime_spans:
        ov = ev.overlap(lo, hi)
        if ov > 0:
            candidates.append((_RUNTIME_DEPTH, ov, ev))
            best_ov = max(best_ov, ov)

    if not candidates:
        return None, 0
    floor = best_ov * _LEAF_COVERAGE_TOLERANCE
    viable = [c for c in candidates if c[1] >= floor]
    depth, ov, ev = max(viable, key=lambda c: (c[0], c[1]))
    return ev, ov


def _is_step(name: str) -> bool:
    return name.startswith("ProfilerStep")


def resolve_all(view: TraceView, gaps: list[Gap]) -> list[Blame]:
    return [resolve_blame(view, g) for g in gaps]
