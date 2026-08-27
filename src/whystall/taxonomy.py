"""The stall taxonomy.

Frozen 2026-08-24, before any measurement existed. See ``TAXONOMY.md`` for the
rationale and the commit that fixed it. A taxonomy reverse-engineered from what a
classifier happens to detect is not a taxonomy; this module is the record that
this one was not.

Two properties carry the architecture:

``needs_sampler``
    True for the classes that cannot be decided from a Kineto trace alone. The
    trace-only classifier in :mod:`whystall.trace.classify` is structurally
    incapable of emitting them -- it emits :data:`StallClass.DATALOADER` and
    leaves the split to :mod:`whystall.enrich.refine`. That boundary is what
    keeps the trace-only half upstreamable into a trace-consuming tool.

``injectable``
    True for the classes with a matching fault injector in :mod:`whystall.inject`.
    A class that cannot be injected cannot be validated, and an unvalidatable
    class does not belong in the table.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = [
    "StallClass",
    "ClassInfo",
    "CLASS_INFO",
    "TRACE_ONLY_CLASSES",
    "ENRICHED_CLASSES",
    "INJECTABLE_CLASSES",
]


class StallClass(str, Enum):
    """The verdict attached to a single GPU idle window."""

    #: Host wait inside the dataloader iterator, before the sampler splits it.
    #: This is the most specific verdict a trace-only classifier can reach.
    DATALOADER = "dataloader"

    #: Dataloader wait dominated by block-device reads. The only class a storage
    #: vendor can actually fix, and therefore the whole commercial question.
    DATALOADER_IO = "dataloader_io"

    #: Dataloader wait dominated by decode and augmentation on the worker cores.
    DATALOADER_CPU = "dataloader_cpu"

    #: Dataloader wait with neither signal dominant. Reported honestly rather
    #: than forced into one of the two above.
    DATALOADER_MIXED = "dataloader_mixed"

    #: Host-to-device transfer. Pageable sources are synchronous and are their
    #: own finding; the evidence records which was used.
    H2D_COPY = "h2d_copy"

    #: Waiting on a collective. Sub-labelled straggler vs transfer when per-rank
    #: entry timestamps are available.
    COLLECTIVE = "collective"

    #: A host-side blocking synchronise or scalar readback drained the queue.
    #: The most actionable verdict the tool can return.
    HOST_SYNC = "host_sync"

    #: Short, repeated gaps where the host is in dispatch machinery. Equivalent
    #: to Holistic Trace Analysis's "kernel wait", and cross-checked against it.
    LAUNCH_DISPATCH = "launch_dispatch"

    #: Host wait with a resolvable cause matching none of the above. Always
    #: carries the top operator names, so the bucket stays actionable instead of
    #: becoming a dustbin.
    OTHER_HOST = "other_host"

    #: The next kernel was already enqueued before the window opened. The GPU was
    #: waiting on the device, not the host. NOT starvation, and counting it as
    #: such inflates every result.
    DEVICE_DEPENDENCY = "device_dependency"

    #: No blame span resolved. A first-class row in every report, never folded
    #: into "other". Its size on unmodified workloads is a published number.
    UNATTRIBUTED = "unattributed"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class ClassInfo:
    """Static metadata about a stall class."""

    label: str
    summary: str
    host_wait: bool
    needs_sampler: bool
    injector: str | None
    refines: StallClass | None = None

    @property
    def injectable(self) -> bool:
        return self.injector is not None


CLASS_INFO: dict[StallClass, ClassInfo] = {
    StallClass.DATALOADER: ClassInfo(
        label="Dataloader",
        summary="Blocked in the dataloader iterator; storage-vs-CPU not yet decided.",
        host_wait=True,
        needs_sampler=False,
        injector=None,
    ),
    StallClass.DATALOADER_IO: ClassInfo(
        label="Dataloader - I/O",
        summary="Workers blocked on block-device reads.",
        host_wait=True,
        needs_sampler=True,
        injector="dataloader_io",
        refines=StallClass.DATALOADER,
    ),
    StallClass.DATALOADER_CPU: ClassInfo(
        label="Dataloader - CPU",
        summary="Workers saturated decoding and augmenting; storage is not the problem.",
        host_wait=True,
        needs_sampler=True,
        injector="dataloader_cpu",
        refines=StallClass.DATALOADER,
    ),
    StallClass.DATALOADER_MIXED: ClassInfo(
        label="Dataloader - mixed",
        summary="Both I/O and CPU present; neither dominant at the calibrated threshold.",
        host_wait=True,
        needs_sampler=True,
        injector=None,
        refines=StallClass.DATALOADER,
    ),
    StallClass.H2D_COPY: ClassInfo(
        label="Host-to-device copy",
        summary="Waiting on a transfer to the device; pinned vs pageable recorded.",
        host_wait=True,
        needs_sampler=False,
        injector="h2d",
    ),
    StallClass.COLLECTIVE: ClassInfo(
        label="Collective wait",
        summary="Waiting on a collective; straggler vs transfer sub-labelled where possible.",
        host_wait=True,
        needs_sampler=False,
        injector="collective",
    ),
    StallClass.HOST_SYNC: ClassInfo(
        label="Host sync",
        summary="A blocking synchronise or scalar readback drained the queue.",
        host_wait=True,
        needs_sampler=False,
        injector="host_sync",
    ),
    StallClass.LAUNCH_DISPATCH: ClassInfo(
        label="Launch / dispatch",
        summary="Host in launch and dispatch machinery; too many small kernels.",
        host_wait=True,
        needs_sampler=False,
        injector="launch",
    ),
    StallClass.OTHER_HOST: ClassInfo(
        label="Other host compute",
        summary="Resolvable host work matching no other rule; top operators attached.",
        host_wait=True,
        needs_sampler=False,
        injector="other_host",
    ),
    StallClass.DEVICE_DEPENDENCY: ClassInfo(
        label="Device dependency",
        summary="Next kernel was already queued; the GPU waited on the device, not the host.",
        host_wait=False,
        needs_sampler=False,
        # Injectable after all: record an event on a second stream, make the
        # compute stream wait on it, and delay the event. The work is enqueued
        # on time and still cannot run, which is the whole shape of the class.
        injector="device_dependency",
    ),
    StallClass.UNATTRIBUTED: ClassInfo(
        label="Unattributed",
        summary="No blame span resolved. Reported, never hidden.",
        host_wait=False,
        needs_sampler=False,
        injector=None,
    ),
}

#: What the dependency-free, trace-only classifier is allowed to emit.
TRACE_ONLY_CLASSES: tuple[StallClass, ...] = tuple(
    c for c, info in CLASS_INFO.items() if not info.needs_sampler
)

#: What only the sampler-enriched classifier can emit.
ENRICHED_CLASSES: tuple[StallClass, ...] = tuple(
    c for c, info in CLASS_INFO.items() if info.needs_sampler
)

#: Classes with a matching injector: the rows of the confusion matrix that can
#: carry a ground-truth label.
INJECTABLE_CLASSES: tuple[StallClass, ...] = tuple(
    c for c, info in CLASS_INFO.items() if info.injectable
)
