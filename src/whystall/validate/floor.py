"""Detection floors: the smallest injected stall a class is reliably found at.

More useful to anyone running the tool than the confusion matrix, and almost
never reported. A matrix says "recall 0.94"; a floor says "recall 0.94 above
3 ms, and below 800 us this class is indistinguishable from noise", which is
what an operator actually needs in order to trust a verdict.

The floor is read off the recall-versus-magnitude curve by linear interpolation
between the bracketing sweep points. That is an estimate, and it is reported
with the bracketing points beside it so the reader can see how wide the bracket
was rather than taking the interpolation on faith.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..taxonomy import INJECTABLE_CLASSES, StallClass
from .score import Sweep

__all__ = ["DetectionFloor", "detection_floor", "all_floors"]


@dataclass(frozen=True, slots=True)
class DetectionFloor:
    cls: StallClass
    target: float
    #: Interpolated magnitude in microseconds at which recall reaches target.
    #: None when the curve never reaches it, or when there is not enough data.
    floor_us: float | None
    #: The last sweep point below target and the first at or above it.
    below: tuple[int, float] | None
    above: tuple[int, float] | None
    #: The full curve, so a report can draw it instead of just quoting a number.
    curve: tuple[tuple[int, float, int], ...]
    note: str = ""

    def summary(self) -> dict[str, Any]:
        return {
            "class": self.cls.value,
            "target_recall": self.target,
            "floor_us": None if self.floor_us is None else round(self.floor_us, 1),
            "below": list(self.below) if self.below else None,
            "above": list(self.above) if self.above else None,
            "curve": [
                {"magnitude_us": m, "recall": None if r != r else round(r, 4), "support": s}
                for m, r, s in self.curve
            ],
            "note": self.note,
        }


def detection_floor(
    sweep: Sweep,
    cls: StallClass,
    *,
    target: float = 0.9,
    min_support: int = 5,
) -> DetectionFloor:
    """Smallest magnitude at which recall for 'cls' reaches 'target'."""
    curve = [
        (m, r, s) for m, r, s in sweep.recall_curve(cls) if s >= min_support and r == r
    ]
    curve.sort()
    if not curve:
        return DetectionFloor(
            cls, target, None, None, None, tuple(curve),
            note=f"no sweep point with support >= {min_support}",
        )
    if len(curve) == 1:
        m, r, _ = curve[0]
        reached = r >= target
        return DetectionFloor(
            cls, target,
            float(m) if reached else None,
            None if reached else (m, r),
            (m, r) if reached else None,
            tuple(curve),
            note="single sweep point; floor is not bracketed",
        )

    below: tuple[int, float] | None = None
    for i, (m, r, _s) in enumerate(curve):
        if r >= target:
            if i == 0:
                return DetectionFloor(
                    cls, target, float(m), None, (m, r), tuple(curve),
                    note="target already met at the smallest magnitude swept; "
                         "the true floor is lower and was not bracketed",
                )
            pm, pr = below if below else curve[i - 1][:2]
            span = r - pr
            frac = 0.0 if span <= 0 else (target - pr) / span
            return DetectionFloor(
                cls, target, pm + frac * (m - pm), (pm, pr), (m, r), tuple(curve)
            )
        below = (m, r)

    return DetectionFloor(
        cls, target, None, below, None, tuple(curve),
        note="recall never reached the target anywhere in the swept range",
    )


def all_floors(
    sweep: Sweep, *, target: float = 0.9, min_support: int = 5
) -> list[DetectionFloor]:
    out = []
    for cls in INJECTABLE_CLASSES:
        if sweep.magnitudes(cls):
            out.append(detection_floor(sweep, cls, target=target, min_support=min_support))
    return out
