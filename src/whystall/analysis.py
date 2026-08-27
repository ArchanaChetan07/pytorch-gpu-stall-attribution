"""The one entry point most callers want: trace in, attributed gaps out."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .taxonomy import CLASS_INFO, StallClass
from .trace.blame import Blame, resolve_all
from .trace.classify import ClassifierConfig, Verdict, classify_all
from .trace.load import load_trace
from .trace.schema import Label, TraceView
from .trace.timeline import GapSet, find_gaps

__all__ = ["Attribution", "DeviceResult", "attribute", "attribute_trace"]


@dataclass(slots=True)
class DeviceResult:
    """Attributed idle for one device."""

    device: int
    gapset: GapSet
    blames: list[Blame]
    verdicts: list[Verdict]

    def by_class_us(self) -> dict[StallClass, int]:
        out: dict[StallClass, int] = {}
        for v in self.verdicts:
            out[v.cls] = out.get(v.cls, 0) + v.dur
        return out

    def by_class_count(self) -> dict[StallClass, int]:
        out: dict[StallClass, int] = {}
        for v in self.verdicts:
            out[v.cls] = out.get(v.cls, 0) + 1
        return out

    @property
    def classified_us(self) -> int:
        return sum(v.dur for v in self.verdicts)

    @property
    def unattributed_us(self) -> int:
        return sum(v.dur for v in self.verdicts if v.cls is StallClass.UNATTRIBUTED)

    @property
    def unattributed_share(self) -> float:
        """Share of classified idle that no rule explained.

        Published as-is. It is the cheapest available evidence that the rest of
        the numbers were measured rather than tuned.
        """
        total = self.classified_us
        return self.unattributed_us / total if total else 0.0

    @property
    def host_wait_us(self) -> int:
        return sum(v.dur for v in self.verdicts if CLASS_INFO[v.cls].host_wait)

    def summary(self) -> dict[str, Any]:
        span = self.gapset.window[1] - self.gapset.window[0]
        return {
            "device": self.device,
            "window_us": span,
            "busy_us": self.gapset.busy_us,
            "utilisation": round(self.gapset.utilisation, 6),
            "idle_total_us": self.gapset.idle_total_us,
            "idle_subthreshold_us": self.gapset.idle_subthreshold_us,
            "idle_trailing_us": self.gapset.idle_trailing_us,
            "min_gap_us": self.gapset.min_gap_us,
            "gaps": len(self.gapset.gaps),
            "classified_us": self.classified_us,
            "host_wait_us": self.host_wait_us,
            "unattributed_us": self.unattributed_us,
            "unattributed_share": round(self.unattributed_share, 6),
            "by_class_us": {k.value: v for k, v in sorted(self.by_class_us().items())},
            "by_class_count": {k.value: v for k, v in sorted(self.by_class_count().items())},
        }


@dataclass(slots=True)
class Attribution:
    """Result of attributing every device in one trace."""

    view: TraceView
    config: ClassifierConfig
    devices: list[DeviceResult] = field(default_factory=list)
    #: Ground-truth labels stripped from the trace. Empty for ordinary analysis;
    #: populated only when the trace came from an injected run. The classifier
    #: never saw these -- see whystall.validate.firewall.
    labels: list[Label] = field(default_factory=list)

    def device(self, device: int) -> DeviceResult:
        for d in self.devices:
            if d.device == device:
                return d
        raise KeyError(f"no such device in trace: {device}")

    @property
    def verdicts(self) -> list[Verdict]:
        return [v for d in self.devices for v in d.verdicts]

    def summary(self) -> dict[str, Any]:
        return {
            "source": self.view.source,
            "config": self.config.as_dict(),
            "devices": [d.summary() for d in self.devices],
            "labels": len(self.labels),
        }


def attribute(
    view: TraceView,
    *,
    config: ClassifierConfig | None = None,
    labels: list[Label] | None = None,
    devices: list[int] | None = None,
) -> Attribution:
    """Attribute every idle window on every device in a loaded trace."""
    config = config or ClassifierConfig()
    result = Attribution(view=view, config=config, labels=labels or [])
    for dev in devices if devices is not None else view.devices():
        gapset = find_gaps(view, dev, min_gap_us=config.min_gap_us)
        blames = resolve_all(view, gapset.gaps)
        verdicts = classify_all(blames, config)
        result.devices.append(DeviceResult(dev, gapset, blames, verdicts))
    return result


def attribute_trace(
    path: str | os.PathLike[str],
    *,
    config: ClassifierConfig | None = None,
) -> Attribution:
    """Load a trace from disk and attribute it.

    Ground-truth labels are separated by the loader and carried alongside the
    result; they are never part of the view the classifier reads.
    """
    view, labels = load_trace(path)
    return attribute(view, config=config, labels=labels)
