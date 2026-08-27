"""Device timelines: busy intervals, and the idle windows between them.

One rule matters more than the rest of this module: a device is busy if *any*
of its streams is busy. Computing idle per stream and summing is the commonest
way to manufacture a starvation problem that does not exist -- the compute
stream sitting quiet while the NCCL stream runs a collective is not an idle GPU.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass

from .schema import Event, TraceView

__all__ = ["Interval", "Gap", "busy_intervals", "find_gaps", "GapSet"]

Interval = tuple[int, int]


@dataclass(slots=True)
class Gap:
    """One window in which a device had no work executing."""

    device: int
    start: int
    end: int
    #: First device op at or after 'end' on this device. None only at the tail.
    next_gpu: Event | None = None
    #: Last device op ending at or before 'start'.
    prev_gpu: Event | None = None

    @property
    def dur(self) -> int:
        return self.end - self.start

    def __repr__(self) -> str:
        return f"Gap(dev={self.device}, {self.start}..{self.end}, dur={self.dur})"


@dataclass(slots=True)
class GapSet:
    """Gaps for one device, plus the accounting for what was filtered out."""

    device: int
    gaps: list[Gap]
    #: Analysis window, from TraceView.bounds().
    window: Interval
    #: Total idle microseconds, including gaps below the classification floor.
    idle_total_us: int
    #: Idle in gaps too short to classify individually. Reported as an aggregate
    #: rather than dropped, because on a launch-bound workload it is the answer.
    idle_subthreshold_us: int
    #: The threshold that produced the split, so a report can state it.
    min_gap_us: int

    @property
    def busy_us(self) -> int:
        return (self.window[1] - self.window[0]) - self.idle_total_us

    @property
    def utilisation(self) -> float:
        span = self.window[1] - self.window[0]
        return (span - self.idle_total_us) / span if span > 0 else 0.0


def busy_intervals(view: TraceView, device: int) -> list[Interval]:
    """Merged busy intervals for one device, across every stream."""
    spans = [(ev.ts, ev.end) for ev in view.gpu if ev.device == device and ev.dur > 0]
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
    min_gap_us: int = 20,
    window: Interval | None = None,
) -> GapSet:
    """Idle windows on 'device', bounded to the profiled region.

    Gaps shorter than min_gap_us are counted but not returned individually: at
    that scale the cause is launch overhead rather than anything attributable,
    and classifying tens of thousands of them one by one buys nothing.
    """
    lo, hi = window if window is not None else view.bounds()
    busy = busy_intervals(view, device)

    gaps: list[Gap] = []
    idle_total = 0
    idle_sub = 0
    cursor = lo

    for start, end in busy:
        if end <= lo or start >= hi:
            continue
        start = max(start, lo)
        if start > cursor:
            idle = start - cursor
            idle_total += idle
            if idle >= min_gap_us:
                gaps.append(Gap(device=device, start=cursor, end=start))
            else:
                idle_sub += idle
        if end > cursor:
            cursor = min(end, hi)
    if cursor < hi:
        idle = hi - cursor
        idle_total += idle
        # Trailing idle after the last kernel is profiler teardown, not a stall.
        idle_sub += idle

    _attach_neighbours(view, device, gaps)
    return GapSet(
        device=device,
        gaps=gaps,
        window=(lo, hi),
        idle_total_us=idle_total,
        idle_subthreshold_us=idle_sub,
        min_gap_us=min_gap_us,
    )


def _attach_neighbours(view: TraceView, device: int, gaps: list[Gap]) -> None:
    """Attach the device ops bracketing each gap.

    next_gpu is the op that ends the gap, and it is the single strongest piece
    of evidence the classifier has: an NCCL kernel or an HtoD memcpy names the
    cause on its own.
    """
    dev_ops = [ev for ev in view.gpu if ev.device == device and ev.dur > 0]
    if not dev_ops:
        return
    starts = [ev.ts for ev in dev_ops]
    ends_sorted = sorted(dev_ops, key=lambda e: e.end)
    end_keys = [ev.end for ev in ends_sorted]

    for gap in gaps:
        i = bisect_left(starts, gap.end)
        # Several ops can begin at the same microsecond; prefer the longest,
        # which is the one that actually reoccupied the device.
        best: Event | None = None
        while i < len(dev_ops) and dev_ops[i].ts == gap.end:
            if best is None or dev_ops[i].dur > best.dur:
                best = dev_ops[i]
            i += 1
        gap.next_gpu = best if best is not None else (dev_ops[i] if i < len(dev_ops) else None)

        j = bisect_left(end_keys, gap.start + 1) - 1
        gap.prev_gpu = ends_sorted[j] if j >= 0 else None
