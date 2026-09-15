"""Align the sidecar sampler's clock with the trace's clock.

This is the failure that corrupts results silently. Kineto timestamps are
microseconds on a boot-derived clock; the sampler reads CLOCK_MONOTONIC. Get the
mapping wrong and nothing raises -- you simply attribute every gap to whatever
the workers were doing some tens of milliseconds earlier, and every number in
the report is quietly false.

The fix is to make the trace carry the answer. The training harness emits
WHYSTALL/clock/<monotonic_ns> annotations at the start and end of profiling. The
trace records where each one landed in trace time; the name records where it
landed in sampler time. Two points give offset and drift, and drift is worth
fitting rather than assuming: the two clocks are usually the same source, but
"usually" is not a thing to build a measurement on.

Refuse to guess. If the sync points are missing, this module raises rather than
assuming an offset of zero, because a silently wrong alignment is far worse than
a run that fails loudly.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..trace.schema import Label

__all__ = ["ClockMap", "ClockSyncError", "fit_clock", "sync_label_name"]


class ClockSyncError(RuntimeError):
    """Not enough information to align the sampler with the trace."""


def sync_label_name(monotonic_ns: int) -> str:
    """The annotation a harness should emit at profiler start and stop."""
    return f"WHYSTALL/clock/{monotonic_ns}"


@dataclass(frozen=True, slots=True)
class ClockMap:
    """Affine map from sampler nanoseconds to trace microseconds."""

    #: trace_ns = slope * (mono_ns - mono0_ns) + trace0_ns
    slope: float
    mono0_ns: int
    trace0_ns: float
    n_points: int
    #: Largest absolute residual across the fit points, microseconds. A report
    #: should quote this: it is the honest error bar on every join the sampler
    #: contributes to.
    max_residual_ns: float

    def to_trace_ns(self, monotonic_ns: int) -> float:
        return self.slope * (monotonic_ns - self.mono0_ns) + self.trace0_ns

    def to_monotonic_ns(self, trace_ns: float) -> float:
        return (trace_ns - self.trace0_ns) / self.slope + self.mono0_ns

    def summary(self) -> dict[str, float | int]:
        return {
            "points": self.n_points,
            "slope": self.slope,
            "drift_ppm": (self.slope - 1.0) * 1e6,
            "max_residual_us": round(self.max_residual_ns / 1000, 3),
        }


def fit_clock(labels: list[Label], *, max_residual_us: float = 500.0) -> ClockMap:
    """Fit the sampler-to-trace mapping from WHYSTALL/clock annotations.

    Raises unless at least two usable sync points exist and the fit residual is
    within max_residual_us. A loose fit means the two clocks disagree in a way
    this model does not capture, and joining on it would be guesswork.
    """
    points: list[tuple[int, int]] = []
    for lab in labels:
        if lab.kind != "clock" or not lab.parts:
            continue
        try:
            mono_ns = int(lab.parts[0])
        except ValueError:
            continue
        points.append((mono_ns, lab.ts_ns))

    if len(points) < 2:
        raise ClockSyncError(
            f"found {len(points)} clock sync point(s); need at least 2. "
            "Emit whystall.enrich.clock.sync_label_name(time.clock_gettime_ns("
            "time.CLOCK_MONOTONIC)) via record_function at profiler start and stop."
        )

    points.sort()
    mono0, _trace0 = points[0]
    n = len(points)
    xs = [float(m - mono0) for m, _ in points]
    ys = [float(t) for _, t in points]
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        raise ClockSyncError("all clock sync points share a timestamp; cannot fit drift")
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    slope = sxy / sxx
    intercept = my - slope * mx

    resid = max(abs(y - (slope * x + intercept)) for x, y in zip(xs, ys, strict=True))
    if resid > max_residual_us * 1000:
        raise ClockSyncError(
            f"clock fit residual {resid / 1000:.1f} us exceeds {max_residual_us:.1f} us "
            f"over {n} points; the sampler and the trace are not on compatible clocks"
        )

    return ClockMap(
        slope=slope,
        mono0_ns=mono0,
        trace0_ns=intercept,
        n_points=n,
        max_residual_ns=resid,
    )
