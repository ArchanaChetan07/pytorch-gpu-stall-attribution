"""Injector protocol and the label emission that makes runs self-scoring.

Every injector wraps its effect in a WHYSTALL/inject annotation carrying its
class and its dial setting. That annotation rides inside the same trace, on the
same clock, as the thing it labels -- which is what makes per-gap ground truth
possible at all, and why the confusion matrix here is a real one rather than an
aggregate error bar.

Where the label goes matters. A dataloader injector perturbs a worker process,
and worker processes are not profiled, so its label is emitted on the main
thread around the fetch instead. The ground truth being asserted is "this gap
was caused by the injected condition", not "this gap happened inside the code I
edited", and the paired control run is what makes that assertion checkable.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any

from ..taxonomy import StallClass
from ..trace.schema import LABEL_PREFIX

__all__ = ["Injector", "LabelScope", "inject_label", "emit_clock_sync", "monotonic_ns"]


class LabelScope:
    """Where in the step the harness should place this injector's label."""

    FETCH = "fetch"
    TRANSFER = "transfer"
    COMPUTE = "compute"
    STEP_END = "step_end"
    NONE = "none"


def monotonic_ns() -> int:
    """The clock the sidecar sampler stamps its samples with."""
    try:
        return time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    except AttributeError:  # pragma: no cover - non-POSIX
        return time.monotonic_ns()


def _record_function(name: str) -> AbstractContextManager[Any]:
    try:
        from torch.profiler import record_function
    except ImportError:
        return nullcontext()
    return record_function(name)


@contextmanager
def inject_label(stall_class: StallClass, magnitude_us: int) -> Iterator[None]:
    """Annotate a region as deliberately perturbed."""
    with _record_function(f"{LABEL_PREFIX}inject/{stall_class.value}/{int(magnitude_us)}"):
        yield


def emit_clock_sync(*, settle_us: int = 300) -> int:
    """Drop a clock sync point into the trace. Call at profiler start and stop.

    The short settle is not decoration: a zero-width range gives the fit nothing
    to bracket, and two sync points at indistinguishable timestamps cannot
    resolve drift.
    """
    now = monotonic_ns()
    with _record_function(f"{LABEL_PREFIX}clock/{now}"):
        time.sleep(settle_us / 1e6)
    return now


class Injector:
    """Base class. Subclasses override only the hooks they need.

    An injector must perturb exactly one thing. Combination runs are built by
    composing two injectors at the harness level, not by writing an injector
    that does two jobs -- otherwise the labels cannot say which caused what.
    """

    #: Short identifier, used in filenames and manifests.
    name: str = "none"
    #: The ground-truth class this injector produces.
    stall_class: StallClass = StallClass.UNATTRIBUTED
    #: Where the harness places the label.
    label_scope: str = LabelScope.NONE

    def __init__(self, magnitude_us: int = 0) -> None:
        self.magnitude_us = int(magnitude_us)

    # --- hooks ---------------------------------------------------------------

    def wrap_dataset(self, dataset: Any) -> Any:
        """Return a dataset that carries the perturbation. Default: unchanged."""
        return dataset

    def wrap_loader_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Adjust DataLoader construction, e.g. to turn off pin_memory."""
        return kwargs

    def on_fetch(self) -> AbstractContextManager[Any]:
        return nullcontext()

    def on_transfer(self) -> AbstractContextManager[Any]:
        return nullcontext()

    def on_compute(self) -> AbstractContextManager[Any]:
        return nullcontext()

    def on_step_end(self) -> AbstractContextManager[Any]:
        return nullcontext()

    # --- reporting -----------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "class": self.stall_class.value,
            "magnitude_us": self.magnitude_us,
            "label_scope": self.label_scope,
        }

    def __repr__(self) -> str:
        return f"{type(self).__name__}(magnitude_us={self.magnitude_us})"


class NullInjector(Injector):
    """The control. Perturbs nothing and emits no labels."""

    name = "control"
    stall_class = StallClass.UNATTRIBUTED
    label_scope = LabelScope.NONE
