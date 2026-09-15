"""Device timelines: busy intervals, and the idle windows between them.

One rule matters more than the rest of this module: a device is busy if *any*
of its streams is busy. Computing idle per stream and summing is the commonest
way to manufacture a starvation problem that does not exist -- the compute
stream sitting quiet while the NCCL stream runs a collective is not an idle GPU.

Times are nanoseconds; see the unit note in schema.py.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass

from .schema import NS_PER_US, Event, TraceView

__all__ = ["Interval", "Gap", "busy_intervals", "find_gaps", "GapSet"]

Interval = tuple[int, int]


@dataclass(slots=True)
class Gap:
    """One window in which a device had no work executing."""

    device: int
    start_ns: int
    end_ns: int
    #: First device op at or after end_ns on this device. None only at the tail.
    next_gpu: Event | None = None
    #: Last device op ending at or before start_ns.
    prev_gpu: Event | None = None

    @property
    def dur_ns(self) -> int:
        return self.end_ns - self.start_ns

    @property
    def dur_us(self) -> float:
        return self.dur_ns / NS_PER_US

    def __repr__(self) -> str:
        return f"Gap(dev={self.device}, {self.start_ns}..{self.end_ns}, dur_ns={self.dur_ns})"


@dataclass(slots=True)
class GapSet:
    """Gaps for one device, plus the accounting for what was filtered out."""

    device: int
    gaps: list[Gap]
    #: Analysis window, from TraceView.bounds_ns().
    window_ns: Interval
    #: Total idle nanoseconds, including gaps below the classification floor.
    idle_total_ns: int
    #: Idle in gaps too short to classify individually. Reported as an aggregate
    #: rather than dropped, because on a launch-bound workload it is the answer.
    idle_subthreshold_ns: int
    #: The threshold that produced the split, so a report can state it.
    min_gap_ns: int
    #: Idle after the last device op, when the host outlived the device. This is
    #: teardown, not starvation, and it is kept separate: folding it into the
    #: sub-threshold total labels seconds of shutdown as launch overhead, which
    #: is exactly the kind of quietly wrong number this tool exists to object to.
    idle_trailing_ns: int = 0

    @property
    def busy_ns(self) -> int:
        return (self.window_ns[1] - self.window_ns[0]) - self.idle_total_ns

    @property
    def utilisation(self) -> float:
        span = self.window_ns[1] - self.window_ns[0]
        return (span - self.idle_total_ns) / span if span > 0 else 0.0


def busy_intervals(view: TraceView, device: int) -> list[Interval]:
    """Merged busy intervals for one device, across every stream."""
    spans = [(ev.ts_ns, ev.end_ns) for ev in view.gpu if ev.device == device and ev.dur_ns > 0]
    if not spans:
        return []
    spans.sort()
    merged: list[Interval] = []
    cur_start, cur_end = spans[0]
    for start, end in spans[1:]:
        if start <= cur_end:
            if end > cur_end:
                cur_end = end
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))
    return merged


def find_gaps(
    view: TraceView,
    device: int,
    *,
    min_gap_us: float = 20,
    window_ns: Interval | None = None,
) -> GapSet:
    """Idle windows on 'device', bounded to the profiled region.

    min_gap_us is in microseconds because that is the unit the threshold is
    reasoned about in; everything downstream of it is nanoseconds. Gaps shorter
    than it are counted but not returned individually: at that scale the cause
    is launch overhead rather than anything attributable, and classifying tens
    of thousands of them one by one buys nothing.
    """
    min_gap_ns = round(min_gap_us * NS_PER_US)
    lo, hi = window_ns if window_ns is not None else view.bounds_ns()
    busy = busy_intervals(view, device)

    gaps: list[Gap] = []
    idle_total = 0
    idle_sub = 0
    idle_trailing = 0
    cursor = lo

    for start, end in busy:
        if end <= lo or start >= hi:
            continue
        start = max(start, lo)
        if start > cursor:
            idle = start - cursor
            idle_total += idle
            if idle >= min_gap_ns:
                gaps.append(Gap(device=device, start_ns=cursor, end_ns=start))
            else:
                idle_sub += idle
        if end > cursor:
            cursor = min(end, hi)
    if cursor < hi:
        # Trailing idle after the last device op is teardown, not a stall, and
        # not sub-threshold either. Its own bucket.
        idle_trailing = hi - cursor
        idle_total += idle_trailing

    _attach_neighbours(view, device, gaps)
    return GapSet(
        device=device,
        gaps=gaps,
        window_ns=(lo, hi),
        idle_total_ns=idle_total,
        idle_subthreshold_ns=idle_sub,
        min_gap_ns=min_gap_ns,
        idle_trailing_ns=idle_trailing,
    )


def _attach_neighbours(view: TraceView, device: int, gaps: list[Gap]) -> None:
    """Attach the device ops bracketing each gap.

    next_gpu is the op that ends the gap, and it is the single strongest piece
    of evidence the classifier has: an NCCL kernel or an HtoD memcpy names the
    cause on its own.
    """
    dev_ops = [ev for ev in view.gpu if ev.device == device and ev.dur_ns > 0]
    if not dev_ops:
        return
    starts = [ev.ts_ns for ev in dev_ops]
    ends_sorted = sorted(dev_ops, key=lambda e: e.end_ns)
    end_keys = [ev.end_ns for ev in ends_sorted]

    for gap in gaps:
        i = bisect_left(starts, gap.end_ns)
        # Several ops can begin at the same instant; prefer the longest, which
        # is the one that actually reoccupied the device.
        best: Event | None = None
        while i < len(dev_ops) and dev_ops[i].ts_ns == gap.end_ns:
            if best is None or dev_ops[i].dur_ns > best.dur_ns:
                best = dev_ops[i]
            i += 1
        gap.next_gpu = best if best is not None else (dev_ops[i] if i < len(dev_ops) else None)

        j = bisect_left(end_keys, gap.start_ns + 1) - 1
        gap.prev_gpu = ends_sorted[j] if j >= 0 else None
