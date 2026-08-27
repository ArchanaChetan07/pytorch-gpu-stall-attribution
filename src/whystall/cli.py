"""Command line interface.

    whystall analyze  trace.json [--samples s.jsonl] [--out report.html]
    whystall validate runs/ [--granularity coarse|strict] [--out report.html]
    whystall sample   --pid PID --hz 200 --out s.jsonl
    whystall synth    dataloader --out t.json [--stall-us 5000]
    whystall demo     --out demo/          full synthetic sweep, end to end
    whystall selftest                      firewall and taxonomy invariants
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .analysis import attribute, attribute_trace
from .taxonomy import CLASS_INFO, INJECTABLE_CLASSES, StallClass
from .trace.classify import ClassifierConfig
from .trace.load import load_trace_dict

__all__ = ["main"]


def _config(args: argparse.Namespace) -> ClassifierConfig:
    base = ClassifierConfig()
    return ClassifierConfig(
        min_gap_us=getattr(args, "min_gap_us", None) or base.min_gap_us,
        launch_bound_max_us=getattr(args, "launch_bound_max_us", None)
        or base.launch_bound_max_us,
        leaf_dominance=getattr(args, "leaf_dominance", None) or base.leaf_dominance,
        launch_dominance=base.launch_dominance,
        weak_annotation_min=base.weak_annotation_min,
    )


# --- analyze -----------------------------------------------------------------


def cmd_analyze(args: argparse.Namespace) -> int:
    from .report.html import write_report
    from .report.text import render_attribution

    att = attribute_trace(args.trace, config=_config(args))

    if args.samples:
        from .enrich.clock import ClockSyncError, fit_clock
        from .enrich.refine import RefineConfig, refine_attribution
        from .enrich.sampler import SamplerLog

        try:
            clock = fit_clock(att.labels)
        except ClockSyncError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        log = SamplerLog.load(args.samples)
        n = refine_attribution(att, log, clock, config=RefineConfig())
        print(
            f"clock fit over {clock.n_points} points, max residual "
            f"{clock.max_residual_us:.1f} us; refined {n} dataloader gaps "
            f"from {len(log)} samples\n"
        )

    print(render_attribution(att))

    if args.out:
        path = write_report(args.out, att, title=f"whystall - {Path(args.trace).name}")
        print(f"wrote {path}")
        print(f"wrote {path.with_suffix('.json')}")
    if args.json:
        Path(args.json).write_text(json.dumps(att.summary(), indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


# --- validate ----------------------------------------------------------------


def _collect_traces(targets: Sequence[str]) -> list[Path]:
    out: list[Path] = []
    for t in targets:
        p = Path(t)
        if p.is_dir():
            out.extend(sorted(p.glob("*.trace.json")))
            out.extend(sorted(p.glob("*.json.gz")))
        elif any(ch in t for ch in "*?["):
            out.extend(sorted(Path().glob(t)))
        else:
            out.append(p)
    return [p for p in out if p.is_file()]


def cmd_validate(args: argparse.Namespace) -> int:
    from .report.html import write_report
    from .report.text import render_sweep
    from .validate.floor import all_floors
    from .validate.score import score_traces

    traces = _collect_traces(args.targets)
    if not traces:
        print("error: no traces found", file=sys.stderr)
        return 2
    print(f"scoring {len(traces)} trace(s) at {args.granularity} granularity\n")

    sweep = score_traces(traces, granularity=args.granularity, config=_config(args))
    floors = all_floors(sweep, target=args.target, min_support=args.min_support)
    print(render_sweep(sweep, floors))

    if args.out:
        path = write_report(
            args.out, None, sweep=sweep, floors=floors, title="whystall validation"
        )
        print(f"\nwrote {path}")
    return 0


# --- sample ------------------------------------------------------------------


def cmd_sample(args: argparse.Namespace) -> int:
    from .enrich import sampler as sampler_mod

    if not sampler_mod.available():
        print("error: the sidecar sampler requires Linux /proc", file=sys.stderr)
        return 2
    path = sampler_mod.run(
        args.pid, args.out, hz=args.hz, duration_s=args.duration
    )
    print(f"wrote {path}")
    return 0


# --- synth -------------------------------------------------------------------


def cmd_synth(args: argparse.Namespace) -> int:
    from .synth import build_scenario

    doc = build_scenario(
        args.kind, steps=args.steps, stall_us=args.stall_us, label=not args.no_labels
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc), encoding="utf-8")
    print(f"wrote {out}  ({args.kind}, {args.steps} steps, {args.stall_us} us)")
    return 0


# --- demo --------------------------------------------------------------------

_DEMO_MAGNITUDES = (400, 900, 2000, 4500, 10000)
_DEMO_KINDS = (
    "dataloader",
    "h2d",
    "host_sync",
    "collective",
    "other_host",
    "device_dependency",
)


def cmd_demo(args: argparse.Namespace) -> int:
    """A complete sweep on synthetic traces: the whole pipeline, no GPU.

    This is what to run first. It exercises loading, gap extraction, blame
    resolution, classification, the firewall, ground-truth association, the
    confusion matrix and the detection floors, and leaves a report behind.
    """
    from .report.html import write_report
    from .report.text import render_sweep
    from .synth import build_scenario
    from .validate.floor import all_floors
    from .validate.score import Sweep, score_attribution

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep = Sweep(granularity=args.granularity)

    for kind in _DEMO_KINDS:
        for mag in _DEMO_MAGNITUDES:
            doc = build_scenario(kind, steps=args.steps, stall_us=mag)
            view, labels = load_trace_dict(doc)
            att = attribute(view, labels=labels, config=_config(args))
            sweep.add(score_attribution(att, granularity=args.granularity,
                                        name=f"{kind}@{mag}us"))
    for i in range(3):
        doc = build_scenario("clean", steps=args.steps)
        view, labels = load_trace_dict(doc)
        att = attribute(view, labels=labels, config=_config(args))
        sweep.add(score_attribution(att, granularity=args.granularity, name=f"control#{i}"))
    doc = build_scenario("launch", steps=args.steps, stall_us=34)
    view, labels = load_trace_dict(doc)
    att = attribute(view, labels=labels, config=_config(args))
    sweep.add(score_attribution(att, granularity=args.granularity, name="launch@34us"))

    floors = all_floors(sweep, target=args.target, min_support=3)
    print(render_sweep(sweep, floors))

    report = write_report(
        out_dir / "demo-report.html",
        att,
        sweep=sweep,
        floors=floors,
        title="whystall demo - synthetic sweep",
    )
    print(f"\nwrote {report}")
    print(f"wrote {report.with_suffix('.json')}")
    print(
        "\nThese are synthetic traces. The numbers show the pipeline works end to end; "
        "they are not a hardware result and must never be quoted as one."
    )
    return 0


# --- selftest ----------------------------------------------------------------


def cmd_selftest(args: argparse.Namespace) -> int:
    from .synth import build_scenario
    from .trace.load import load_trace
    from .validate.firewall import FirewallBreach, assert_clean, scan

    failures: list[str] = []

    doc = build_scenario("dataloader", steps=4, stall_us=4000)
    view, labels = load_trace_dict(doc)
    if scan(view):
        failures.append("loader left ground-truth annotations in the view")
    if not labels:
        failures.append("loader did not extract ground-truth annotations")
    try:
        assert_clean(view)
    except FirewallBreach as exc:
        failures.append(f"firewall rejected a clean view: {exc}")

    dirty, _ = load_trace_dict(doc, keep_labels=True)
    try:
        assert_clean(dirty)
        failures.append("firewall failed to catch a view holding ground truth")
    except FirewallBreach:
        pass

    for cls, info in CLASS_INFO.items():
        if info.needs_sampler and info.refines is None:
            failures.append(f"{cls.value} needs the sampler but refines nothing")
    for cls in INJECTABLE_CLASSES:
        if cls is StallClass.UNATTRIBUTED:
            failures.append("unattributed must not be injectable")

    _ = load_trace  # referenced so the import is obviously intentional

    if failures:
        for f in failures:
            print(f"FAIL  {f}", file=sys.stderr)
        return 1
    print("selftest passed: firewall structural, taxonomy consistent")
    return 0


# --- parser ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="whystall",
        description="Validated attribution of GPU idle windows to their causes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--min-gap-us", type=int, help="ignore idle windows shorter than this")
    p.add_argument("--launch-bound-max-us", type=int, help="upper bound for the launch-bound rule")
    p.add_argument("--leaf-dominance", type=float, help="share one frame must cover to decide")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="attribute the idle time in one trace")
    a.add_argument("trace")
    a.add_argument("--samples", help="sidecar JSONL, to split dataloader waits")
    a.add_argument("--out", help="write a standalone HTML report here")
    a.add_argument("--json", help="write the machine-readable summary here")
    a.set_defaults(func=cmd_analyze)

    v = sub.add_parser("validate", help="score injected runs against their ground truth")
    v.add_argument("targets", nargs="+", help="trace files, globs, or a run directory")
    v.add_argument(
        "--granularity",
        choices=("coarse", "strict"),
        default="coarse",
        help="coarse collapses the dataloader refinements; use it for trace-only results",
    )
    v.add_argument("--target", type=float, default=0.9, help="recall target for detection floors")
    v.add_argument("--min-support", type=int, default=5)
    v.add_argument("--out", help="write a standalone HTML report here")
    v.set_defaults(func=cmd_validate)

    s = sub.add_parser("sample", help="run the /proc sidecar next to a training process")
    s.add_argument("--pid", type=int, required=True)
    s.add_argument("--hz", type=int, default=200)
    s.add_argument("--out", required=True)
    s.add_argument("--duration", type=float, default=None)
    s.set_defaults(func=cmd_sample)

    y = sub.add_parser("synth", help="write a synthetic trace with a known stall")
    y.add_argument("kind")
    y.add_argument("--out", required=True)
    y.add_argument("--steps", type=int, default=16)
    y.add_argument("--stall-us", type=int, default=5000)
    y.add_argument("--no-labels", action="store_true")
    y.set_defaults(func=cmd_synth)

    d = sub.add_parser("demo", help="run the whole pipeline on synthetic traces")
    d.add_argument("--out", default="demo")
    d.add_argument("--steps", type=int, default=14)
    d.add_argument("--granularity", choices=("coarse", "strict"), default="coarse")
    d.add_argument("--target", type=float, default=0.9)
    d.set_defaults(func=cmd_demo)

    t = sub.add_parser("selftest", help="check the firewall and taxonomy invariants")
    t.set_defaults(func=cmd_selftest)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
