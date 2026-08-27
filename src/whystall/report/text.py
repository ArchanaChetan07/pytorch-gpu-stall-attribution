"""Terminal summaries."""

from __future__ import annotations

from ..analysis import Attribution
from ..taxonomy import CLASS_INFO
from ..validate.floor import DetectionFloor
from ..validate.score import Sweep

__all__ = ["render_attribution", "render_sweep"]


def _bar(fraction: float, width: int = 28) -> str:
    filled = int(round(fraction * width))
    return "#" * filled + "." * (width - filled)


def render_attribution(att: Attribution) -> str:
    lines: list[str] = []
    lines.append(f"source   {att.view.source}")
    cfg = att.config.as_dict()
    lines.append("config   " + "  ".join(f"{k}={v}" for k, v in cfg.items()))
    lines.append("")

    for dev in att.devices:
        gs = dev.gapset
        span = gs.window[1] - gs.window[0]
        lines.append(f"device {dev.device}")
        lines.append(
            f"  window {span/1000:.1f} ms   busy {gs.busy_us/1000:.1f} ms   "
            f"utilisation {gs.utilisation:.1%}"
        )
        lines.append(
            f"  idle {gs.idle_total_us/1000:.1f} ms in {len(gs.gaps)} gaps "
            f"(+{gs.idle_subthreshold_us/1000:.1f} ms below the {gs.min_gap_us} us floor)"
        )
        lines.append("")

        total = dev.classified_us or 1
        by_us = dev.by_class_us()
        by_n = dev.by_class_count()
        lines.append(
            "  " + "class".ljust(22) + "idle".rjust(11) + "share".rjust(8)
            + "  gaps".rjust(7) + "  " + "distribution"
        )
        for cls, us in sorted(by_us.items(), key=lambda kv: -kv[1]):
            share = us / total
            lines.append(
                "  "
                + CLASS_INFO[cls].label.ljust(22)
                + f"{us/1000:.1f} ms".rjust(11)
                + f"{share:.1%}".rjust(8)
                + f"{by_n[cls]}".rjust(7)
                + "  "
                + _bar(share)
            )
        lines.append("")
        lines.append(
            f"  unattributed {dev.unattributed_share:.1%} of classified idle"
            "   <- publish this number"
        )
        lines.append("")
    return "\n".join(lines)


def render_sweep(sweep: Sweep, floors: list[DetectionFloor]) -> str:
    lines = [sweep.pooled().render_text(), ""]
    lines.append(f"pooled unattributed share  {sweep.unattributed_share():.1%}")
    lines.append(f"runs scored                {len(sweep.runs)} "
                 f"({len(sweep.control_runs())} control)")
    lines.append("")

    shift = sweep.shift()
    if sweep.control_runs() and shift:
        lines.append("control vs treatment (share of gaps by class)")
        lines.append("  " + "class".ljust(22) + "control".rjust(9) + "treatment".rjust(11)
                     + "shift".rjust(11))
        for cls, (c, t, d) in sorted(shift.items(), key=lambda kv: -abs(kv[1][2])):
            lines.append(
                "  " + CLASS_INFO[cls].label.ljust(22)
                + f"{c:.1%}".rjust(9) + f"{t:.1%}".rjust(11) + f"{d*100:+.1f} pp".rjust(11)
            )
        lines.append("")
    if floors:
        lines.append("detection floors")
        lines.append(
            "  " + "class".ljust(22) + "floor".rjust(12) + "  bracket"
        )
        for f in floors:
            floor = "-" if f.floor_us is None else f"{f.floor_us/1000:.2f} ms"
            bracket = ""
            if f.below and f.above:
                bracket = (
                    f"{f.below[0]/1000:.1f} ms r={f.below[1]:.2f}  ->  "
                    f"{f.above[0]/1000:.1f} ms r={f.above[1]:.2f}"
                )
            elif f.note:
                bracket = f.note
            lines.append("  " + CLASS_INFO[f.cls].label.ljust(22) + floor.rjust(12) + "  " + bracket)
    return "\n".join(lines)
