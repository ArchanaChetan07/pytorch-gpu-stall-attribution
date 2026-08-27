"""The seven injectors, one per validatable class.

Each perturbs exactly one thing and does so with a dial, because the deliverable
is a detection floor per class and a floor needs a sweep.

On the I/O injector specifically: the honest instrument is kernel-level, either
dm-delay for latency or cgroup v2 io.max for bandwidth, and both need privileges
that most rented containers withhold. The portable fallback here does the real
read and then sleeps, which reproduces the timing but not the kernel's queueing
behaviour. Where you have privileges even once, run the same sweep both ways and
publish the comparison -- that single cross-check is what keeps the fallback
defensible. See docs/injection.md.
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from ..taxonomy import StallClass
from .base import Injector, LabelScope

__all__ = [
    "DataloaderIO",
    "DataloaderCPU",
    "HostToDevice",
    "HostSync",
    "CollectiveStraggler",
    "LaunchBound",
    "OtherHostCompute",
    "DeviceDependency",
    "REGISTRY",
    "build",
]


class _Spinner:
    """Busy-wait calibrated to wall time.

    A sleep would be wrong here: sleeping parks the process and looks exactly
    like the I/O case to the sampler, which is the one distinction the whole
    project is trying to make.
    """

    def __init__(self) -> None:
        self._per_iter_s = self._calibrate()

    @staticmethod
    def _calibrate(target_s: float = 0.01) -> float:
        n = 0
        start = time.perf_counter()
        while time.perf_counter() - start < target_s:
            for _ in range(256):
                pass
            n += 256
        elapsed = time.perf_counter() - start
        return elapsed / max(n, 1)

    def spin(self, micros: float) -> None:
        if micros <= 0:
            return
        deadline = time.perf_counter() + micros / 1e6
        while time.perf_counter() < deadline:
            for _ in range(64):
                pass


_SPINNER: _Spinner | None = None


def _spinner() -> _Spinner:
    global _SPINNER
    if _SPINNER is None:
        _SPINNER = _Spinner()
    return _SPINNER


class _WrappedDataset:
    """Applies a per-item perturbation inside the worker process."""

    def __init__(self, inner: Any, hook: Any) -> None:
        self._inner = inner
        self._hook = hook

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, index: Any) -> Any:
        item = self._inner[index]
        self._hook(index)
        return item

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)


class DataloaderIO(Injector):
    """Delay every item read, as a slower storage tier would.

    jitter_frac spreads the delay because a constant per-item latency is not
    what a real device produces, and a classifier tuned on a constant is tuned
    on an artefact.
    """

    name = "dataloader_io"
    stall_class = StallClass.DATALOADER_IO
    label_scope = LabelScope.FETCH

    def __init__(self, magnitude_us: int = 2000, *, jitter_frac: float = 0.25) -> None:
        super().__init__(magnitude_us)
        self.jitter_frac = jitter_frac

    def wrap_dataset(self, dataset: Any) -> Any:
        micros = self.magnitude_us
        jitter = self.jitter_frac
        rng = random.Random(0xD10)

        def hook(_index: Any) -> None:
            delay = micros * (1.0 + rng.uniform(-jitter, jitter))
            if delay > 0:
                time.sleep(delay / 1e6)

        return _WrappedDataset(dataset, hook)


class DataloaderCPU(Injector):
    """Burn worker cores, as heavier decode or augmentation would."""

    name = "dataloader_cpu"
    stall_class = StallClass.DATALOADER_CPU
    label_scope = LabelScope.FETCH

    def wrap_dataset(self, dataset: Any) -> Any:
        micros = self.magnitude_us

        def hook(_index: Any) -> None:
            _spinner().spin(micros)

        return _WrappedDataset(dataset, hook)


class HostToDevice(Injector):
    """Make transfers synchronous and larger.

    Turning pin_memory off is the realistic form: a pageable host source makes
    the copy synchronous, which is a different failure from a merely slow one
    and shows up in the trace as a pageable memcpy.
    """

    name = "h2d"
    stall_class = StallClass.H2D_COPY
    label_scope = LabelScope.TRANSFER

    def __init__(self, magnitude_us: int = 2000, *, pad_mb: int = 64) -> None:
        super().__init__(magnitude_us)
        self.pad_mb = pad_mb
        self._pad: Any = None

    def wrap_loader_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        kwargs = dict(kwargs)
        kwargs["pin_memory"] = False
        return kwargs

    @contextmanager
    def on_transfer(self) -> Iterator[None]:
        import torch

        if self._pad is None and self.pad_mb:
            # Pageable, so the copy is synchronous by construction.
            self._pad = torch.empty(self.pad_mb * 1024 * 1024 // 4, dtype=torch.float32)
        if self._pad is not None and torch.cuda.is_available():
            self._pad.to("cuda", non_blocking=False)
        yield


class HostSync(Injector):
    """Drain the queue from the host, the way a stray .item() does."""

    name = "host_sync"
    stall_class = StallClass.HOST_SYNC
    label_scope = LabelScope.STEP_END

    @contextmanager
    def on_step_end(self) -> Iterator[None]:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _spinner().spin(self.magnitude_us)
        yield


class CollectiveStraggler(Injector):
    """Delay one rank before it reaches the collective.

    Only the straggler rank perturbs; every other rank runs clean and waits.
    That asymmetry is the point -- it is what a slow node looks like.
    """

    name = "collective"
    stall_class = StallClass.COLLECTIVE
    label_scope = LabelScope.COMPUTE

    def __init__(self, magnitude_us: int = 2000, *, straggler_rank: int = 0) -> None:
        super().__init__(magnitude_us)
        self.straggler_rank = straggler_rank

    def _is_straggler(self) -> bool:
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        return rank == self.straggler_rank

    @contextmanager
    def on_compute(self) -> Iterator[None]:
        if self._is_straggler():
            _spinner().spin(self.magnitude_us)
        yield


class LaunchBound(Injector):
    """Replace fused work with an unfused chain of tiny kernels."""

    name = "launch"
    stall_class = StallClass.LAUNCH_DISPATCH
    label_scope = LabelScope.COMPUTE

    def __init__(self, magnitude_us: int = 0, *, kernels: int = 400) -> None:
        super().__init__(magnitude_us or kernels)
        self.kernels = kernels

    @contextmanager
    def on_compute(self) -> Iterator[None]:
        import torch

        if torch.cuda.is_available():
            x = torch.ones(64, device="cuda")
            for _ in range(self.kernels):
                x = x + 1.0
        yield


class OtherHostCompute(Injector):
    """Host-side work between steps: metrics, logging, checkpoint bookkeeping."""

    name = "other_host"
    stall_class = StallClass.OTHER_HOST
    label_scope = LabelScope.STEP_END

    @contextmanager
    def on_step_end(self) -> Iterator[None]:
        _spinner().spin(self.magnitude_us)
        yield


class DeviceDependency(Injector):
    """Enqueue work on time that still cannot run.

    A second stream records an event; the compute stream waits on it. The launch
    precedes the gap, so a correct classifier must not call this host wait.
    """

    name = "device_dependency"
    stall_class = StallClass.DEVICE_DEPENDENCY
    label_scope = LabelScope.COMPUTE

    @contextmanager
    def on_compute(self) -> Iterator[None]:
        import torch

        if not torch.cuda.is_available():
            yield
            return
        side = torch.cuda.Stream()
        event = torch.cuda.Event()
        with torch.cuda.stream(side):
            blocker = torch.ones(2048, 2048, device="cuda")
            iters = max(1, self.magnitude_us // 50)
            for _ in range(iters):
                blocker = blocker @ blocker.T
                blocker = blocker / blocker.norm()
            event.record(side)
        torch.cuda.current_stream().wait_event(event)
        yield


REGISTRY: dict[str, type[Injector]] = {
    cls.name: cls
    for cls in (
        DataloaderIO,
        DataloaderCPU,
        HostToDevice,
        HostSync,
        CollectiveStraggler,
        LaunchBound,
        OtherHostCompute,
        DeviceDependency,
    )
}


def build(name: str, magnitude_us: int, **kwargs: Any) -> Injector:
    """Construct an injector by name, as the CLI does."""
    if name in ("none", "control"):
        from .base import NullInjector

        return NullInjector(0)
    try:
        cls = REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown injector {name!r}; expected one of "
            f"{sorted(REGISTRY) + ['control']}"
        ) from None
    return cls(magnitude_us, **kwargs)


