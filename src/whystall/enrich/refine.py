"""Split dataloader waits into I/O and CPU using the sidecar.

This is the separation the buyers care about and the hardest one in the
taxonomy. Both cases present identically in a trace, so the decision rests
entirely on what the sampler saw during the same window.

The rule is deliberately a small set of named thresholds rather than a fitted
score. Two reasons. A threshold can be published, argued with, and recalibrated
against someone else's hardware; a learned score cannot be checked without the
training data. And the honest output here is frequently "both", which a
continuous score tends to paper over by picking whichever side is marginally
ahead.

Four outcomes, and three of them are answers:

DATALOADER_IO
    Real block-device traffic, or workers parked in uninterruptible sleep.
DATALOADER_CPU
    Workers burning core time with the disk quiet. Storage cannot fix this.
DATALOADER_MIXED
    Both signals present. Reported as such.
DATALOADER
    Left unrefined. Too few samples landed in the window to decide, and
    guessing from two observations is how a tool earns a reputation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..analysis import Attribution
from ..taxonomy import StallClass
from .clock import ClockMap
from .sampler import SamplerLog, WindowStats

__all__ = ["RefineConfig", "RefineOutcome", "refine_attribution", "refine_window"]


@dataclass(frozen=True, slots=True)
class RefineConfig:
    """Published thresholds for the I/O-versus-CPU decision.

    Defaults are starting points, not results. Calibrate them against an
    injection sweep on your own hardware and publish what you calibrated to --
    the defaults here are sized for a single NVMe device and eight workers, and
    a machine with sixty-four workers on a parallel filesystem needs different
    ones.
    """

    #: Block-device read rate above which real storage traffic is present.
    io_read_mb_per_s: float = 20.0
    #: Share of worker observations in uninterruptible sleep implying I/O wait.
    io_frac_uninterruptible: float = 0.5
    #: Per-worker core occupancy above which decode or augmentation dominates.
    cpu_per_worker: float = 0.55
    #: Share of worker observations runnable, as a corroborating CPU signal.
    cpu_frac_runnable: float = 0.4
    #: Fewer samples than this in the window means no decision is made.
    #:
    #: Watch the interaction with the sampling rate, because it decides which
    #: stalls are refinable at all: at H hertz, a window only reaches this many
    #: observations if it lasts min_samples/H seconds. At the default 200 Hz
    #: that is 15 ms, so shorter dataloader waits come back unrefined -- which
    #: is correct behaviour but easy to mistake for the sampler not working.
    #: If the stalls you care about are single-digit milliseconds, raise the
    #: sampler to 500-1000 Hz rather than lowering this.
    min_samples: int = 3

    def as_dict(self) -> dict[str, Any]:
        return {
            "io_read_mb_per_s": self.io_read_mb_per_s,
            "io_frac_uninterruptible": self.io_frac_uninterruptible,
            "cpu_per_worker": self.cpu_per_worker,
            "cpu_frac_runnable": self.cpu_frac_runnable,
            "min_samples": self.min_samples,
        }


@dataclass(frozen=True, slots=True)
class RefineOutcome:
    cls: StallClass
    confidence: float
    reason: str
    evidence: dict[str, Any]


def refine_window(stats: WindowStats, cfg: RefineConfig | None = None) -> RefineOutcome:
    """Decide I/O versus CPU for one window of sampler data."""
    cfg = cfg or RefineConfig()
    ev: dict[str, Any] = {"sampler": stats.summary(), "thresholds": cfg.as_dict()}

    if stats.samples < cfg.min_samples:
        needed_hz = (
            cfg.min_samples / (stats.window_us / 1e6) if stats.window_us > 0 else float("inf")
        )
        return RefineOutcome(
            StallClass.DATALOADER,
            0.0,
            f"only {stats.samples} sampler observations in a "
            f"{stats.window_us/1000:.1f} ms window (need {cfg.min_samples}); "
            f"not refined -- resolving a window this short needs about "
            f"{needed_hz:.0f} Hz",
            ev,
        )

    io_rate = stats.read_mb_per_s >= cfg.io_read_mb_per_s
    io_state = stats.frac_uninterruptible >= cfg.io_frac_uninterruptible
    cpu_busy = stats.cpu_per_worker >= cfg.cpu_per_worker
    cpu_state = stats.frac_runnable >= cfg.cpu_frac_runnable

    io = io_rate or io_state
    cpu = cpu_busy or cpu_state
    ev["signals"] = {
        "io_rate": io_rate,
        "io_state": io_state,
        "cpu_busy": cpu_busy,
        "cpu_state": cpu_state,
    }
    if stats.read_bytes_unavailable:
        # Without read_bytes the I/O case rests on process state alone, which
        # aliases. Say so rather than presenting the verdict at full strength.
        ev["caveat"] = "read_bytes unavailable; I/O evidence is process-state only"

    io_margin = max(
        _margin(stats.read_mb_per_s, cfg.io_read_mb_per_s),
        _margin(stats.frac_uninterruptible, cfg.io_frac_uninterruptible),
    )
    cpu_margin = max(
        _margin(stats.cpu_per_worker, cfg.cpu_per_worker),
        _margin(stats.frac_runnable, cfg.cpu_frac_runnable),
    )

    if io and not cpu:
        conf = _confidence(io_margin, penalty=0.25 if stats.read_bytes_unavailable else 0.0)
        return RefineOutcome(StallClass.DATALOADER_IO, conf, "block-device traffic dominant", ev)
    if cpu and not io:
        return RefineOutcome(
            StallClass.DATALOADER_CPU, _confidence(cpu_margin), "worker cores saturated", ev
        )
    if io and cpu:
        return RefineOutcome(
            StallClass.DATALOADER_MIXED,
            _confidence(min(io_margin, cpu_margin)),
            "both signals above threshold",
            ev,
        )
    return RefineOutcome(
        StallClass.DATALOADER,
        0.0,
        "neither signal reached its threshold; workers were idle-waiting "
        "(a queue, a lock, or upstream starvation)",
        ev,
    )


def refine_attribution(
    att: Attribution,
    log: SamplerLog,
    clock: ClockMap,
    *,
    config: RefineConfig | None = None,
) -> int:
    """Refine every dataloader verdict in place. Returns how many were split.

    The original trace-only verdict is preserved in the evidence under
    trace_only_class, so a report can show what the upstreamable half concluded
    on its own and what the sampler added.
    """
    cfg = config or RefineConfig()
    refined = 0
    for dev in att.devices:
        for verdict, blame in zip(dev.verdicts, dev.blames, strict=True):
            if verdict.cls is not StallClass.DATALOADER:
                continue
            lo_us, hi_us = blame.window
            if hi_us <= lo_us:
                lo_us, hi_us = verdict.gap.start, verdict.gap.end
            stats = log.stats_ns(clock.to_monotonic_ns(lo_us), clock.to_monotonic_ns(hi_us))
            outcome = refine_window(stats, cfg)
            verdict.evidence["trace_only_class"] = StallClass.DATALOADER.value
            verdict.evidence["refine_reason"] = outcome.reason
            verdict.evidence.update(outcome.evidence)
            if outcome.cls is not StallClass.DATALOADER:
                verdict.cls = outcome.cls
                verdict.rule = "sampler_refined"
                verdict.confidence = outcome.confidence
                refined += 1
    return refined


def _margin(value: float, threshold: float) -> float:
    """How far past its threshold a signal is, normalised to 0..1."""
    if threshold <= 0:
        return 1.0 if value > 0 else 0.0
    return max(0.0, min(1.0, (value - threshold) / threshold))


def _confidence(margin: float, *, penalty: float = 0.0) -> float:
    return max(0.0, min(1.0, 0.6 + 0.4 * margin - penalty))
