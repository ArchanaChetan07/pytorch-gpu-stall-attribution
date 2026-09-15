"""Score attributed runs against their injected ground truth."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..analysis import Attribution, attribute_trace
from ..taxonomy import StallClass
from ..trace.classify import ClassifierConfig
from .firewall import assert_clean, assert_evidence_clean
from .groundtruth import Granularity, collapse, injected_regions, label_gaps
from .matrix import ConfusionMatrix

__all__ = ["RunScore", "Sweep", "score_attribution", "score_traces"]


@dataclass(slots=True)
class RunScore:
    """One trace, scored."""

    name: str
    #: The dominant injected class in this run, or None for a control run.
    injected: StallClass | None
    #: The dial setting for this run, microseconds.
    magnitude_us: int
    matrix: ConfusionMatrix
    gaps: int
    labelled_gaps: int
    idle_ns: int
    unattributed_share: float

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "injected": self.injected.value if self.injected else None,
            "magnitude_us": self.magnitude_us,
            "gaps": self.gaps,
            "labelled_gaps": self.labelled_gaps,
            "idle_ns": self.idle_ns,
            "unattributed_share": round(self.unattributed_share, 6),
            "matrix": self.matrix.to_dict(),
        }


def score_attribution(
    att: Attribution,
    *,
    granularity: str = Granularity.COARSE,
    name: str | None = None,
) -> RunScore:
    """Build a confusion matrix for one attributed trace.

    The firewall assertions run here rather than in the test suite alone: any
    path that produces a score also proves the classifier never saw the answers.
    """
    assert_clean(att.view)
    assert_evidence_clean(att.verdicts)

    regions = injected_regions(att.labels)
    matrix = ConfusionMatrix(granularity=granularity)
    gaps = labelled = 0
    idle_ns = 0

    for dev in att.devices:
        truths = label_gaps(dev.blames, regions)
        for verdict, region in zip(dev.verdicts, truths, strict=True):
            truth = collapse(region.cls if region else None, granularity)
            pred = collapse(verdict.cls, granularity)
            matrix.add(truth, pred, ns=verdict.dur_ns)
            gaps += 1
            idle_ns += verdict.dur_ns
            if region is not None:
                labelled += 1

    dominant: StallClass | None = None
    magnitude = 0
    if regions:
        by_class: dict[StallClass, int] = {}
        for r in regions:
            by_class[r.cls] = by_class.get(r.cls, 0) + (r.end_ns - r.start_ns)
        dominant = max(by_class, key=lambda k: by_class[k])
        mags = [r.magnitude_us for r in regions if r.cls is dominant]
        magnitude = max(set(mags), key=mags.count) if mags else 0

    unattr = sum(
        v.dur_ns for v in att.verdicts if v.cls is StallClass.UNATTRIBUTED
    )
    return RunScore(
        name=name or att.view.source,
        injected=dominant,
        magnitude_us=magnitude,
        matrix=matrix,
        gaps=gaps,
        labelled_gaps=labelled,
        idle_ns=idle_ns,
        unattributed_share=(unattr / idle_ns) if idle_ns else 0.0,
    )


@dataclass
class Sweep:
    """A set of scored runs, pooled and sliceable by magnitude."""

    runs: list[RunScore] = field(default_factory=list)
    granularity: str = Granularity.COARSE

    def add(self, run: RunScore) -> None:
        self.runs.append(run)

    def pooled(self) -> ConfusionMatrix:
        """Every gap from every run in one matrix. The headline number."""
        out = ConfusionMatrix(granularity=self.granularity)
        for run in self.runs:
            for key, n in run.matrix.counts.items():
                out.counts[key] = out.counts.get(key, 0) + n
                out.nanos[key] = out.nanos.get(key, 0) + run.matrix.nanos.get(key, 0)
        return out

    def magnitudes(self, cls: StallClass) -> list[int]:
        return sorted({r.magnitude_us for r in self.runs if r.injected is cls})

    def recall_at(self, cls: StallClass, magnitude: int) -> tuple[float, int]:
        """(recall, support) for one class at one dial setting."""
        hits = support = 0
        for run in self.runs:
            if run.injected is not cls or run.magnitude_us != magnitude:
                continue
            s = run.matrix.score(cls)
            hits += s.hits
            support += s.support
        return ((hits / support) if support else float("nan")), support

    def recall_curve(self, cls: StallClass) -> list[tuple[int, float, int]]:
        return [(m, *self.recall_at(cls, m)) for m in self.magnitudes(cls)]

    def unattributed_share(self) -> float:
        idle = sum(r.idle_ns for r in self.runs)
        if not idle:
            return 0.0
        return sum(r.unattributed_share * r.idle_ns for r in self.runs) / idle

    # --- control comparison --------------------------------------------------

    def control_runs(self) -> list[RunScore]:
        return [r for r in self.runs if r.injected is None]

    def treatment_runs(self) -> list[RunScore]:
        return [r for r in self.runs if r.injected is not None]

    def _distribution(self, runs: list[RunScore]) -> dict[StallClass, float]:
        totals: dict[StallClass, int] = {}
        n = 0
        for run in runs:
            for (_t, p), count in run.matrix.counts.items():
                totals[p] = totals.get(p, 0) + count
                n += count
        return {k: v / n for k, v in totals.items()} if n else {}

    def control_baseline(self) -> dict[StallClass, float]:
        """Class shares in the runs where nothing was injected.

        The number to compare a treatment run against. A tool that reports 12%
        dataloader stalls under injection is only interesting if it did not
        already report 12% without it -- and precision alone cannot tell you,
        because a correct verdict on an uninjected stall counts against it.
        """
        return self._distribution(self.control_runs())

    def treatment_baseline(self) -> dict[StallClass, float]:
        return self._distribution(self.treatment_runs())

    def shift(self) -> dict[StallClass, tuple[float, float, float]]:
        """(control share, treatment share, delta) per class."""
        control = self.control_baseline()
        treat = self.treatment_baseline()
        keys = sorted(set(control) | set(treat), key=lambda c: c.value)
        return {k: (control.get(k, 0.0), treat.get(k, 0.0),
                    treat.get(k, 0.0) - control.get(k, 0.0)) for k in keys}

    def summary(self) -> dict[str, Any]:
        return {
            "granularity": self.granularity,
            "runs": [r.summary() for r in self.runs],
            "pooled": self.pooled().to_dict(),
            "unattributed_share": round(self.unattributed_share(), 6),
            "control_runs": len(self.control_runs()),
            "class_shift": {
                k.value: {
                    "control": round(c, 6),
                    "treatment": round(t, 6),
                    "delta": round(d, 6),
                }
                for k, (c, t, d) in self.shift().items()
            },
        }


def score_traces(
    paths: Iterable[str | os.PathLike[str]],
    *,
    granularity: str = Granularity.COARSE,
    config: ClassifierConfig | None = None,
) -> Sweep:
    """Attribute and score every trace in a sweep directory."""
    sweep = Sweep(granularity=granularity)
    for path in paths:
        att = attribute_trace(path, config=config)
        sweep.add(score_attribution(att, granularity=granularity, name=str(path)))
    return sweep
