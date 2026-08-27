"""Confusion matrix over labelled gaps.

Two conventions, both chosen so the result cannot flatter itself:

* Unlabelled gaps in an injected run get the truth key None and are kept. They
  are the false positives, and dropping them is the single easiest way to
  publish a recall number that does not survive replication.
* Counts are per gap. Microseconds are tracked alongside, because a classifier
  can be right about most gaps and wrong about most of the idle time, and the
  reader is entitled to see both.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..taxonomy import CLASS_INFO, StallClass

__all__ = ["ConfusionMatrix", "ClassScore"]

#: Truth key for a gap the injection did not cause.
NO_TRUTH = "(none)"


@dataclass(frozen=True, slots=True)
class ClassScore:
    cls: StallClass
    support: int
    predicted: int
    hits: int

    @property
    def recall(self) -> float:
        return self.hits / self.support if self.support else float("nan")

    @property
    def precision(self) -> float:
        return self.hits / self.predicted if self.predicted else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if math.isnan(p) or math.isnan(r) or (p + r) == 0:
            return float("nan")
        return 2 * p * r / (p + r)


@dataclass
class ConfusionMatrix:
    """Rows are ground truth, columns are predictions."""

    counts: dict[tuple[Any, StallClass], int] = field(default_factory=dict)
    micros: dict[tuple[Any, StallClass], int] = field(default_factory=dict)
    granularity: str = "strict"

    def add(self, truth: StallClass | None, pred: StallClass, *, us: int = 0) -> None:
        key = (truth if truth is not None else NO_TRUTH, pred)
        self.counts[key] = self.counts.get(key, 0) + 1
        self.micros[key] = self.micros.get(key, 0) + us

    def extend(self, rows: Iterable[tuple[StallClass | None, StallClass, int]]) -> None:
        for truth, pred, us in rows:
            self.add(truth, pred, us=us)

    # --- axes ----------------------------------------------------------------

    def truths(self) -> list[Any]:
        seen = {t for t, _ in self.counts}
        ordered = [c for c in StallClass if c in seen]
        if NO_TRUTH in seen:
            ordered.append(NO_TRUTH)
        return ordered

    def predictions(self) -> list[StallClass]:
        seen = {p for _, p in self.counts}
        return [c for c in StallClass if c in seen]

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    # --- scores --------------------------------------------------------------

    def score(self, cls: StallClass) -> ClassScore:
        support = sum(n for (t, _), n in self.counts.items() if t == cls)
        predicted = sum(n for (_, p), n in self.counts.items() if p == cls)
        hits = self.counts.get((cls, cls), 0)
        return ClassScore(cls=cls, support=support, predicted=predicted, hits=hits)

    def scores(self) -> list[ClassScore]:
        classes = {t for t, _ in self.counts if isinstance(t, StallClass)}
        classes |= {p for _, p in self.counts}
        return [self.score(c) for c in StallClass if c in classes]

    @property
    def macro_recall(self) -> float:
        vals = [s.recall for s in self.scores() if s.support]
        return sum(vals) / len(vals) if vals else float("nan")

    @property
    def accuracy(self) -> float:
        labelled = sum(n for (t, _), n in self.counts.items() if isinstance(t, StallClass))
        if not labelled:
            return float("nan")
        hits = sum(n for (t, p), n in self.counts.items() if isinstance(t, StallClass) and t == p)
        return hits / labelled

    @property
    def unlabelled_blame_rate(self) -> float:
        """Share of unlabelled gaps that were nonetheless given a named cause.

        This is deliberately NOT called a false-positive rate, because it is not
        one. A training run stalls for real reasons the injection did not
        create -- the first fetch of an epoch, a genuine device dependency
        behind a copy stream -- and a correct verdict on one of those gaps lands
        here, indistinguishable from a mistake, because nothing labelled it.

        Read it as an upper bound on the error, and get the real answer from the
        paired control run: what matters is how the class distribution *shifts*
        between control and treatment, not how many gaps got a name.

        The same caveat applies to precision throughout this matrix, which is
        likewise a lower bound.
        """
        unlabelled = sum(n for (t, _), n in self.counts.items() if t == NO_TRUTH)
        if not unlabelled:
            return float("nan")
        blamed = sum(
            n
            for (t, p), n in self.counts.items()
            if t == NO_TRUTH and CLASS_INFO[p].injectable
        )
        return blamed / unlabelled

    def confusions(self, limit: int = 8) -> list[tuple[StallClass, StallClass, int]]:
        """The off-diagonal cells, largest first. The part worth publishing."""
        rows = [
            (t, p, n)
            for (t, p), n in self.counts.items()
            if isinstance(t, StallClass) and t != p
        ]
        rows.sort(key=lambda r: -r[2])
        return rows[:limit]

    # --- output --------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "granularity": self.granularity,
            "total": self.total,
            "accuracy": _round(self.accuracy),
            "macro_recall": _round(self.macro_recall),
            "unlabelled_blame_rate": _round(self.unlabelled_blame_rate),
            "cells": [
                {
                    "truth": t.value if isinstance(t, StallClass) else t,
                    "pred": p.value,
                    "n": n,
                    "us": self.micros.get((t, p), 0),
                }
                for (t, p), n in sorted(
                    self.counts.items(),
                    key=lambda kv: (str(kv[0][0]), str(kv[0][1])),
                )
            ],
            "per_class": [
                {
                    "class": s.cls.value,
                    "support": s.support,
                    "predicted": s.predicted,
                    "hits": s.hits,
                    "recall": _round(s.recall),
                    "precision": _round(s.precision),
                    "f1": _round(s.f1),
                }
                for s in self.scores()
            ],
            "top_confusions": [
                {"truth": t.value, "pred": p.value, "n": n} for t, p, n in self.confusions()
            ],
        }

    def render_text(self, width: int = 11) -> str:
        preds = self.predictions()
        if not preds:
            return "(empty matrix)"
        head = "truth \\ pred".ljust(24) + "".join(_abbr(p).rjust(width) for p in preds)
        lines = [head, "-" * len(head)]
        for t in self.truths():
            label = t.value if isinstance(t, StallClass) else t
            row = label.ljust(24)
            for p in preds:
                n = self.counts.get((t, p), 0)
                row += (str(n) if n else ".").rjust(width)
            lines.append(row)
        lines.append("")
        lines.append("class".ljust(24) + "support".rjust(9) + "recall".rjust(9) + "precision".rjust(11))
        for s in self.scores():
            lines.append(
                s.cls.value.ljust(24)
                + str(s.support).rjust(9)
                + _fmt(s.recall).rjust(9)
                + _fmt(s.precision).rjust(11)
            )
        lines.append("")
        lines.append(f"accuracy {_fmt(self.accuracy)}   macro recall {_fmt(self.macro_recall)}")
        lines.append(
            f"unlabelled gaps given a named cause {_fmt(self.unlabelled_blame_rate)} "
            "(upper bound on error, not a false-positive rate -- see the control baseline)"
        )
        return "\n".join(lines)


def _abbr(cls: StallClass) -> str:
    parts = cls.value.split("_")
    if len(parts) == 1:
        return parts[0][:9]
    return (parts[0][:4] + "." + parts[-1][:4])[:10]


def _fmt(x: float) -> str:
    return "-" if x != x else f"{x:.3f}"


def _round(x: float) -> float | None:
    return None if x != x else round(x, 6)
