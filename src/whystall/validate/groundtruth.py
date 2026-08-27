"""Turn injected annotations into per-gap ground truth.

The association rule is the methodological heart of the validation, so it is
stated here rather than buried: an injected region labels a gap when it overlaps
either

* the blame window -- the interval on the launching thread the gap is charged
  against -- which is the right test for any stall the host experiences; or
* the gap itself, which is the right test for a stall the host never sees, such
  as a device-side dependency, where the host is not blocked at all.

Whichever overlap is larger decides, and when several regions match, the largest
overlap wins. Regions never label a gap on a different process.

Note what is *not* the rule: the injected magnitude is not ground truth for how
much idle it produced. A pipeline with prefetch depth absorbs part of any
injection, so a 5 ms delay routinely yields 2 ms of idle. Ground truth is the
labelled gap; the magnitude is only the sweep coordinate.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..taxonomy import StallClass
from ..trace.blame import Blame
from ..trace.schema import Label

__all__ = [
    "InjectedRegion",
    "injected_regions",
    "label_gaps",
    "Granularity",
    "collapse",
]


@dataclass(frozen=True, slots=True)
class InjectedRegion:
    """One deliberately injected stall, as recorded by the injector itself."""

    cls: StallClass
    #: The dial setting, microseconds. The sweep coordinate, not the answer.
    magnitude_us: int
    pid: int
    tid: int
    start: int
    end: int

    def overlap(self, start: int, end: int) -> int:
        lo = self.start if self.start > start else start
        hi = self.end if self.end < end else end
        return hi - lo if hi > lo else 0


def injected_regions(labels: list[Label]) -> list[InjectedRegion]:
    """Parse WHYSTALL/inject/<class>/<magnitude_us> annotations.

    Unknown class names are skipped rather than raising: a trace from a newer
    injector must still be scorable by an older reader for the classes it does
    understand.
    """
    out: list[InjectedRegion] = []
    for lab in labels:
        if lab.kind != "inject" or not lab.parts:
            continue
        try:
            cls = StallClass(lab.parts[0])
        except ValueError:
            continue
        magnitude = 0
        if len(lab.parts) > 1:
            try:
                magnitude = int(lab.parts[1])
            except ValueError:
                magnitude = 0
        out.append(
            InjectedRegion(
                cls=cls,
                magnitude_us=magnitude,
                pid=lab.pid,
                tid=lab.tid,
                start=lab.ts,
                end=lab.end,
            )
        )
    return out


def label_gaps(
    blames: list[Blame],
    regions: list[InjectedRegion],
) -> list[InjectedRegion | None]:
    """Ground truth per gap, aligned with the blames list. None means unlabelled.

    An unlabelled gap in an injected run is not an error. It is a gap the
    injection did not cause, and it belongs in the matrix as such -- collapsing
    it into the injected class is how recall gets inflated.
    """
    out: list[InjectedRegion | None] = []
    for b in blames:
        gap_lo, gap_hi = b.gap.start, b.gap.end
        win_lo, win_hi = b.window
        best: InjectedRegion | None = None
        best_ov = 0
        for r in regions:
            ov = max(r.overlap(win_lo, win_hi), r.overlap(gap_lo, gap_hi))
            if ov > best_ov:
                best_ov, best = ov, r
        out.append(best)
    return out


class Granularity:
    """How finely a prediction has to match to count as correct.

    STRICT compares classes as they are. COARSE first collapses the dataloader
    refinements into the unsplit class, which is the only fair way to score a
    trace-only classifier that is structurally incapable of emitting them.

    Both belong in the report. Scoring the trace-only half coarsely and the
    enriched half strictly is what makes the architecture visible in the results
    rather than just in the prose.
    """

    STRICT = "strict"
    COARSE = "coarse"


_COARSE_MAP = {
    StallClass.DATALOADER_IO: StallClass.DATALOADER,
    StallClass.DATALOADER_CPU: StallClass.DATALOADER,
    StallClass.DATALOADER_MIXED: StallClass.DATALOADER,
}


def collapse(cls: StallClass | None, granularity: str) -> StallClass | None:
    if cls is None or granularity == Granularity.STRICT:
        return cls
    return _COARSE_MAP.get(cls, cls)
