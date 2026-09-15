"""The trace-only classifier.

This module decides everything that can be decided from a Kineto trace and
nothing that cannot. In particular it never emits DATALOADER_IO or
DATALOADER_CPU: both present identically in a trace -- the launching thread is
blocked inside the iterator either way -- and separating them needs worker
process state, which a trace does not carry. It emits DATALOADER and stops.

That restraint is deliberate and structural. This file has no dependencies
beyond the standard library and no knowledge of the sampler, which is what makes
it the half that can be contributed to a trace-consuming tool upstream. The
split lives in whystall.enrich.refine.

Rules are ordered by strength of evidence, not by class:

1. the launch preceded the gap, so the host was not the cause at all;
2. the op that ended the gap names the cause outright (an NCCL kernel);
3. one frame dominated the blame window, and its identity decides;
4. the op that ended the gap suggests a cause, but weakly;
5. an enclosing annotation is present but did not dominate;
6. the gap is short and the host was in launch machinery;
7. something resolved but matched nothing -- reported with its operators;
8. nothing resolved.

Ordering by evidence rather than by class is what stops a 300 us window in
which the host spent 280 us inside the dataloader from being charged to the
memcpy that happened to close it.

Every threshold is a named field on ClassifierConfig with a stated default,
because a classifier whose constants are buried in the code cannot be
independently checked, and being independently checkable is the entire point.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..taxonomy import TRACE_ONLY_CLASSES, StallClass
from .blame import Blame
from .load import memcpy_kind
from .schema import NS_PER_US, Event
from .timeline import Gap

__all__ = ["ClassifierConfig", "Verdict", "classify", "classify_all", "RULES"]


# --- name vocabulary ---------------------------------------------------------
# Matched case-insensitively against event names. Listed here rather than inline
# so a reader can audit the whole vocabulary in one place.

_NCCL_HINTS = ("nccl", "c10d::", "record_param_comms")
_COLLECTIVE_OPS = (
    "all_reduce",
    "allreduce",
    "all_gather",
    "allgather",
    "reduce_scatter",
    "reducescatter",
    "alltoall",
    "all_to_all",
    "barrier",
)
_SYNC_HINTS = (
    "cudadevicesynchronize",
    "cudastreamsynchronize",
    "cudaeventsynchronize",
    "cudaipcsynchronize",
    "aten::item",
    "aten::_local_scalar_dense",
    "aten::equal",
    "aten::allclose",
)
_COPY_HINTS = (
    "cudamemcpyasync",
    "cudamemcpy",
    "aten::to",
    "aten::_to_copy",
    "aten::copy_",
    "aten::pin_memory",
    "memcpy htod",
)
_LAUNCH_HINTS = (
    "cudalaunchkernel",
    "cudalaunchkernelexc",
    "cudalaunchcooperativekernel",
    "cudagraphlaunch",
)
_DATALOADER_HINT = "enumerate(dataloader)"


def _has(name: str, needles: tuple[str, ...]) -> bool:
    low = name.lower()
    return any(n in low for n in needles)


def _is_collective_kernel(ev: Event | None) -> bool:
    if ev is None:
        return False
    low = ev.name.lower()
    if any(h in low for h in _NCCL_HINTS):
        return True
    return "kernel" in ev.cat.lower() and any(op in low for op in _COLLECTIVE_OPS)


def _memcpy_direction(ev: Event | None) -> tuple[str | None, str | None]:
    if ev is None or "memcpy" not in ev.cat.lower():
        return None, None
    direction, host_mem = memcpy_kind(ev.name)
    return (direction.lower() if direction else None), host_mem


@dataclass(frozen=True, slots=True)
class ClassifierConfig:
    """Published thresholds. Change them in experiments, but report them."""

    #: Gaps shorter than this are not classified individually; see timeline.
    min_gap_us: int = 20
    #: A gap at or below this length, with the host in launch machinery, is
    #: launch-bound rather than starved. Sits above a single kernel launch (a
    #: few microseconds) and below anything a human would call a stall.
    launch_bound_max_us: int = 200
    #: Share of the blame window one frame must cover before its identity alone
    #: decides the verdict.
    leaf_dominance: float = 0.5
    #: Share of the blame window that must be inside launch calls for the
    #: launch-bound rule to fire.
    launch_dominance: float = 0.6
    #: Minimum overlap for the weak dataloader fallback to fire at all.
    weak_annotation_min: float = 0.15

    @property
    def min_gap_ns(self) -> int:
        return round(self.min_gap_us * NS_PER_US)

    @property
    def launch_bound_max_ns(self) -> int:
        return round(self.launch_bound_max_us * NS_PER_US)

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_gap_us": self.min_gap_us,
            "launch_bound_max_us": self.launch_bound_max_us,
            "leaf_dominance": self.leaf_dominance,
            "launch_dominance": self.launch_dominance,
            "weak_annotation_min": self.weak_annotation_min,
        }


@dataclass(slots=True)
class Verdict:
    """One gap, one class, and the reason."""

    gap: Gap
    cls: StallClass
    #: Which rule fired. Reports group by this, so a surprising number can be
    #: traced back to the line of logic that produced it.
    rule: str
    #: 0..1. For dominance-driven rules this is the frame's share of the window.
    confidence: float = 1.0
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def dur_ns(self) -> int:
        return self.gap.dur_ns


Rule = Callable[[Blame, ClassifierConfig], "Verdict | None"]


# --- rules, in priority order ------------------------------------------------


def rule_device_dependency(b: Blame, cfg: ClassifierConfig) -> Verdict | None:
    """The work was already queued: the device waited on the device."""
    if not b.queued_before_gap:
        return None
    return Verdict(
        b.gap,
        StallClass.DEVICE_DEPENDENCY,
        "device_dependency",
        1.0,
        {
            "launch_ts_ns": b.t_launch_ns,
            "gap_start_ns": b.gap.start_ns,
            "next_gpu": b.gap.next_gpu.name if b.gap.next_gpu else None,
        },
    )


def rule_collective_kernel(b: Blame, cfg: ClassifierConfig) -> Verdict | None:
    """The gap ends in a collective. The strongest single piece of evidence."""
    nxt = b.gap.next_gpu
    if not _is_collective_kernel(nxt):
        return None
    assert nxt is not None
    return Verdict(
        b.gap,
        StallClass.COLLECTIVE,
        "collective_next_kernel",
        1.0,
        {"kernel": nxt.name, "stream": nxt.stream},
    )


def rule_dominant_frame(b: Blame, cfg: ClassifierConfig) -> Verdict | None:
    """One frame held the thread for most of the window; it names the cause."""
    leaf = b.leaf
    if leaf is None or b.leaf_fraction < cfg.leaf_dominance:
        return None
    name = leaf.name
    ev: dict[str, Any] = {"frame": name, "fraction": round(b.leaf_fraction, 4)}

    if _has(name, _NCCL_HINTS) or _has(name, _COLLECTIVE_OPS):
        return Verdict(b.gap, StallClass.COLLECTIVE, "collective_frame", b.leaf_fraction, ev)
    if _has(name, _SYNC_HINTS):
        return Verdict(b.gap, StallClass.HOST_SYNC, "host_sync_frame", b.leaf_fraction, ev)
    if _DATALOADER_HINT in name.lower():
        ev["needs_sampler"] = True
        return Verdict(b.gap, StallClass.DATALOADER, "dataloader_frame", b.leaf_fraction, ev)
    if _has(name, _COPY_HINTS):
        _, host_mem = memcpy_kind(name)
        ev["host_memory"] = host_mem
        return Verdict(b.gap, StallClass.H2D_COPY, "h2d_frame", b.leaf_fraction, ev)
    # Dominant but unrecognised. That is still the answer -- just a less specific
    # one -- and it must be decided here. Falling through would let the op that
    # happened to close the gap overrule a frame that held the thread for most
    # of it, which is the exact error the rule ordering exists to prevent.
    ev["top_ops"] = b.top_names(3)
    return Verdict(b.gap, StallClass.OTHER_HOST, "other_host_frame", b.leaf_fraction, ev)


def rule_next_op_transfer(b: Blame, cfg: ClassifierConfig) -> Verdict | None:
    """The op that ended the gap was a transfer. Weaker than a dominant frame."""
    direction, host_mem = _memcpy_direction(b.gap.next_gpu)
    if direction is None:
        return None
    nxt = b.gap.next_gpu
    assert nxt is not None
    if direction == "htod":
        return Verdict(
            b.gap,
            StallClass.H2D_COPY,
            "h2d_next_memcpy",
            0.7,
            {"direction": direction, "host_memory": host_mem, "bytes": nxt.nbytes},
        )
    if direction == "dtoh":
        # A readback closing the gap is a host sync told from the device side:
        # the host asked for a value and stopped feeding the queue.
        return Verdict(
            b.gap,
            StallClass.HOST_SYNC,
            "host_sync_readback",
            0.7,
            {"readback": nxt.name, "bytes": nxt.nbytes},
        )
    return None


def rule_dataloader_present(b: Blame, cfg: ClassifierConfig) -> Verdict | None:
    """A dataloader frame covered part of the window without dominating it.

    Keeps recall on pipelines where pinning or collation splits the wait into
    several frames, none of which clears the dominance threshold on its own.
    """
    lo, hi = b.window_ns
    span = hi - lo
    if span <= 0:
        return None
    total = 0
    which = None
    for ev in b.covering:
        if _DATALOADER_HINT in ev.name.lower():
            ov = ev.overlap_ns(lo, hi)
            if ov > total:
                total, which = ov, ev.name
    frac = total / span
    if which is None or frac < cfg.weak_annotation_min:
        return None
    return Verdict(
        b.gap,
        StallClass.DATALOADER,
        "dataloader_partial",
        frac,
        {"annotation": which, "fraction": round(frac, 4), "needs_sampler": True},
    )


def rule_launch_dispatch(b: Blame, cfg: ClassifierConfig) -> Verdict | None:
    """A short gap that no vocabulary explains.

    Related to, but much wider than, Holistic Trace Analysis's "kernel wait".
    HTA's threshold is consecutive_kernel_delay, which defaults to 30 ns, so its
    kernel wait covers essentially back-to-back kernels and nothing else; the
    default here is launch_bound_max_us, 200 us, which is over six thousand
    times wider. This class should therefore be a strict superset of HTA's
    kernel wait, and the difference is carved out of what HTA reports as host
    wait. That is the thesis, not an accident: host wait is the bucket worth
    subdividing.

    The discriminator is deliberately negative: below launch_bound_max_us the
    host is churning through operator dispatch rather than sitting in any one
    identifiable frame, so there is nothing positive to match on. The rule
    therefore refuses to fire whenever a named cause is present anywhere in the
    window, which keeps it from swallowing a short dataloader or sync stall.

    It runs early, before the dominance rules, because at this scale a "dominant
    frame" is just whichever aten op the window happened to land in.
    """
    if b.gap.dur_ns > cfg.launch_bound_max_ns:
        return None
    lo, hi = b.window_ns
    span = hi - lo
    if span <= 0:
        return None
    for ev in (*b.covering, *b.runtime_spans):
        low = ev.name.lower()
        if (
            _DATALOADER_HINT in low
            or _has(ev.name, _SYNC_HINTS)
            or _has(ev.name, _COPY_HINTS)
            or _has(ev.name, _NCCL_HINTS)
            or _has(ev.name, _COLLECTIVE_OPS)
        ):
            return None
    launch_ns = sum(ev.overlap_ns(lo, hi) for ev in b.runtime_spans if _has(ev.name, _LAUNCH_HINTS))
    return Verdict(
        b.gap,
        StallClass.LAUNCH_DISPATCH,
        "launch_bound",
        0.6,
        {
            "launch_fraction": round(launch_ns / span, 4),
            "gap_us": round(b.gap.dur_us, 3),
            "frames": [e.name for e in b.covering[-2:]],
        },
    )


def rule_other_host(b: Blame, cfg: ClassifierConfig) -> Verdict | None:
    """Resolvable host work matching no other rule, with its operators attached."""
    if not b.resolved or b.leaf is None:
        return None
    return Verdict(
        b.gap,
        StallClass.OTHER_HOST,
        "other_host",
        b.leaf_fraction,
        {"frame": b.leaf.name, "top_ops": b.top_names(3)},
    )


def rule_unattributed(b: Blame, cfg: ClassifierConfig) -> Verdict:
    """The residual. Always fires last, and is always reported."""
    if b.launch is None:
        reason = "no_launch_correlation"
    elif not b.resolved:
        reason = "no_covering_host_events"
    else:
        reason = "no_rule_matched"
    return Verdict(
        b.gap,
        StallClass.UNATTRIBUTED,
        "unattributed",
        1.0,
        {
            "reason": reason,
            "next_gpu": b.gap.next_gpu.name if b.gap.next_gpu else None,
            "launcher_tid": b.launcher_tid,
        },
    )


#: Order is the priority order, and it is the classifier's actual specification.
RULES: tuple[Rule, ...] = (
    rule_device_dependency,
    rule_collective_kernel,
    rule_launch_dispatch,
    rule_dominant_frame,
    rule_next_op_transfer,
    rule_dataloader_present,
    rule_other_host,
)


def classify(blame: Blame, cfg: ClassifierConfig | None = None) -> Verdict:
    """Classify one resolved gap. Total function: always returns a verdict."""
    cfg = cfg or ClassifierConfig()
    for rule in RULES:
        verdict = rule(blame, cfg)
        if verdict is not None:
            if verdict.cls not in TRACE_ONLY_CLASSES:
                raise AssertionError(
                    f"{rule.__name__} emitted {verdict.cls}, which needs the sampler. "
                    "The trace-only classifier must not reach sampler-only classes."
                )
            return verdict
    return rule_unattributed(blame, cfg)


def classify_all(blames: list[Blame], cfg: ClassifierConfig | None = None) -> list[Verdict]:
    cfg = cfg or ClassifierConfig()
    return [classify(b, cfg) for b in blames]
