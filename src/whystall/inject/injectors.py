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


def spin(micros: float) -> None:
    """Busy-wait for micros microseconds.

    A sleep would be wrong here: sleeping parks the process and looks exactly
    like the I/O case to the sampler, which is the one distinction the whole
    project is trying to make. The loop is a deadline check, so it needs no
    calibration and cannot drift.
    """
    if micros <= 0:
        return
    deadline = time.perf_counter() + micros / 1e6
    while time.perf_counter() < deadline:
        for _ in range(64):
            pass


class _SpinHook:
    """Burn worker CPU per item. A module-level class so it survives pickling."""

    __slots__ = ("micros",)

    def __init__(self, micros: float) -> None:
        self.micros = micros

    def __call__(self, _index: Any) -> None:
        spin(self.micros)


class _SleepHook:
    """Delay per item, as a slower storage tier would.

    The RNG is seeded lazily from the process id rather than at construction:
    the hook is built in the parent and copied into every worker, so a seed
    fixed up front would give all eight workers an identical delay sequence and
    a suspiciously periodic access pattern.
    """

    __slots__ = ("micros", "jitter_frac", "_rng", "_rng_pid")

    def __init__(self, micros: float, jitter_frac: float) -> None:
        self.micros = micros
        self.jitter_frac = jitter_frac
        self._rng: random.Random | None = None
        self._rng_pid: int | None = None

    def __call__(self, _index: Any) -> None:
        pid = os.getpid()
        if self._rng is None or self._rng_pid != pid:
            self._rng = random.Random(0xD10 ^ pid)
            self._rng_pid = pid
        delay = self.micros * (1.0 + self._rng.uniform(-self.jitter_frac, self.jitter_frac))
        if delay > 0:
            time.sleep(delay / 1e6)

    def __getstate__(self) -> tuple[float, float]:
        # random.Random does not survive spawn cleanly and must not be shared
        # across workers anyway; rebuild it on first use in the child.
        return self.micros, self.jitter_frac

    def __setstate__(self, state: tuple[float, float]) -> None:
        self.micros, self.jitter_frac = state
        self._rng = None
        self._rng_pid = None


class _WrappedDataset:
    """Applies a per-item perturbation inside the worker process.

    Deliberately not a closure. DataLoader with num_workers > 0 pickles the
    dataset under the spawn and forkserver start methods, and a closure is not
    picklable -- which fails at the first fetch, on the rented box, mid-sweep.
    """

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
        # __getattr__ runs for anything not found normally, including the
        # dunders pickle probes for before __init__ has restored _inner. Without
        # this guard that is unbounded recursion during unpickling in a worker.
        if item.startswith("__") or item in ("_inner", "_hook"):
            raise AttributeError(item)
        return getattr(self._inner, item)

    def __getstate__(self) -> dict[str, Any]:
        return {"_inner": self._inner, "_hook": self._hook}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)


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
        return _WrappedDataset(dataset, _SleepHook(self.magnitude_us, self.jitter_frac))


class DataloaderCPU(Injector):
    """Burn worker cores, as heavier decode or augmentation would."""

    name = "dataloader_cpu"
    stall_class = StallClass.DATALOADER_CPU
    label_scope = LabelScope.FETCH

    def wrap_dataset(self, dataset: Any) -> Any:
        return _WrappedDataset(dataset, _SpinHook(self.magnitude_us))


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
        spin(self.magnitude_us)
        yield


class CollectiveStraggler(Injector):
    """Delay one rank before it reaches the collective.

    Only the straggler rank perturbs; every other rank runs clean and waits.
    That asymmetry is the point -- it is what a slow node looks like.

    The label goes on the ranks that *wait*, not the one that causes the wait.
    The straggler's own idle is its own spinning -- host compute, correctly not a
    collective stall -- so labelling its trace `collective` would assert a ground
    truth that is false on that rank and would score every honest verdict there
    as a miss. The victim ranks are the ones blocked inside the collective, and
    they are the only ranks whose gaps this injector is entitled to claim.

    This is why label placement is a property rather than a class attribute: it
    is the one injector whose ground truth is rank-dependent.
    """

    name = "collective"
    stall_class = StallClass.COLLECTIVE

    def __init__(self, magnitude_us: int = 2000, *, straggler_rank: int = 0) -> None:
        super().__init__(magnitude_us)
        self.straggler_rank = straggler_rank

    @property
    def label_scope(self) -> str:  # type: ignore[override]
        return LabelScope.NONE if self._is_straggler() else LabelScope.COMPUTE

    def _is_straggler(self) -> bool:
        rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        return rank == self.straggler_rank

    @contextmanager
    def on_compute(self) -> Iterator[None]:
        if self._is_straggler():
            spin(self.magnitude_us)
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
        spin(self.magnitude_us)
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


