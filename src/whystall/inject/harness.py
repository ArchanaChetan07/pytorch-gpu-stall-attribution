"""Run one injected training run and leave behind everything needed to score it.

A run produces three artefacts and needs all three:

  <run>.trace.json     the Kineto trace, with injection and clock labels in it
  <run>.samples.jsonl  the sidecar output, if the host has /proc
  <run>.manifest.json  what was injected, at what dial, on what machine

The manifest is not bookkeeping. Cache state, device model, injector settings
and code revision are what make a number reproducible, and a sweep whose runs
cannot be told apart six weeks later is a sweep that has to be re-run.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..taxonomy import StallClass
from .base import Injector, LabelScope, emit_clock_sync, inject_label

__all__ = ["HarnessConfig", "Harness"]


@dataclass
class HarnessConfig:
    out_dir: Path = Path("runs")
    run_name: str = ""
    #: Steps skipped before profiling starts, so allocator and autotuner warm-up
    #: does not land in the trace and get attributed as a stall.
    warmup_steps: int = 8
    #: Steps actually profiled.
    active_steps: int = 40
    sampler: bool = True
    sampler_hz: int = 200
    #: Recorded, never enforced -- the harness cannot verify what you did to the
    #: page cache, and a wrong value here is worse than an absent one.
    cache_state: str = "unknown"
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class Harness:
    """Context manager owning the profiler, the sampler and the labels."""

    def __init__(self, injector: Injector, config: HarnessConfig | None = None) -> None:
        self.injector = injector
        self.config = config or HarnessConfig()
        self.step_index = 0
        self._prof: Any = None
        self._sampler: subprocess.Popen[bytes] | None = None
        self._clock_points: list[int] = []
        self._t0 = 0.0

        name = self.config.run_name or f"{injector.name}-{injector.magnitude_us}us"
        self.out_dir = Path(self.config.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.out_dir / f"{name}.trace.json"
        self.samples_path = self.out_dir / f"{name}.samples.jsonl"
        self.manifest_path = self.out_dir / f"{name}.manifest.json"
        self.run_name = name

    # --- dataset and loader --------------------------------------------------

    def wrap_dataset(self, dataset: Any) -> Any:
        return self.injector.wrap_dataset(dataset)

    def loader_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        return self.injector.wrap_loader_kwargs(kwargs)

    # --- lifecycle -----------------------------------------------------------

    def __enter__(self) -> Harness:
        if self.config.sampler:
            self._start_sampler()
        self._start_profiler()
        self._t0 = time.time()
        self._clock_points.append(emit_clock_sync())
        return self

    def __exit__(self, *exc: Any) -> None:
        self._clock_points.append(emit_clock_sync())
        self._stop_profiler()
        self._stop_sampler()
        self._write_manifest()

    def _start_profiler(self) -> None:
        import torch.profiler as tp

        self._prof = tp.profile(
            activities=[tp.ProfilerActivity.CPU, tp.ProfilerActivity.CUDA],
            schedule=tp.schedule(
                wait=0, warmup=self.config.warmup_steps, active=self.config.active_steps, repeat=1
            ),
            on_trace_ready=lambda p: p.export_chrome_trace(str(self.trace_path)),
            record_shapes=False,
            with_stack=False,
            profile_memory=False,
        )
        self._prof.start()

    def _stop_profiler(self) -> None:
        if self._prof is not None:
            self._prof.stop()

    def _start_sampler(self) -> None:
        from ..enrich import sampler as sampler_mod

        if not sampler_mod.available():
            self.config.extra["sampler_skipped"] = "no /proc on this host"
            return
        self._sampler = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "whystall.cli",
                "sample",
                "--pid",
                str(os.getpid()),
                "--hz",
                str(self.config.sampler_hz),
                "--out",
                str(self.samples_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        # Give it a moment to reach steady state so the first profiled step is
        # already covered by samples.
        time.sleep(0.2)

    def _stop_sampler(self) -> None:
        if self._sampler is None:
            return
        self._sampler.terminate()
        try:
            self._sampler.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover
            self._sampler.kill()

    # --- per-step scopes -----------------------------------------------------

    @contextmanager
    def step(self) -> Iterator[int]:
        idx = self.step_index
        try:
            yield idx
        finally:
            self.step_index += 1
            if self._prof is not None:
                self._prof.step()

    def _scope(self, scope: str):
        @contextmanager
        def cm() -> Iterator[None]:
            active = self.injector.label_scope == scope
            hook = {
                LabelScope.FETCH: self.injector.on_fetch,
                LabelScope.TRANSFER: self.injector.on_transfer,
                LabelScope.COMPUTE: self.injector.on_compute,
                LabelScope.STEP_END: self.injector.on_step_end,
            }[scope]
            if active:
                with (
                    inject_label(self.injector.stall_class, self.injector.magnitude_us),
                    hook(),
                ):
                    yield
            else:
                yield

        return cm()

    def fetch(self):
        """Wrap the batch fetch. Labelled for the dataloader injectors."""
        return self._scope(LabelScope.FETCH)

    def transfer(self):
        """Wrap the host-to-device move."""
        return self._scope(LabelScope.TRANSFER)

    def compute(self):
        """Wrap forward, backward and the optimiser."""
        return self._scope(LabelScope.COMPUTE)

    def step_end(self):
        """Wrap the host-side epilogue."""
        return self._scope(LabelScope.STEP_END)

    # --- manifest ------------------------------------------------------------

    def _write_manifest(self) -> None:
        manifest = {
            "run": self.run_name,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "wall_s": round(time.time() - self._t0, 3),
            "injector": self.injector.describe(),
            "ground_truth_class": (
                self.injector.stall_class.value
                if self.injector.stall_class is not StallClass.UNATTRIBUTED
                else None
            ),
            "config": {
                "warmup_steps": self.config.warmup_steps,
                "active_steps": self.config.active_steps,
                "sampler_hz": self.config.sampler_hz if self.config.sampler else None,
                "cache_state": self.config.cache_state,
                "notes": self.config.notes,
            },
            "artifacts": {
                "trace": self.trace_path.name,
                "samples": self.samples_path.name if self.samples_path.exists() else None,
            },
            "clock_sync_points_ns": self._clock_points,
            "host": _host_info(),
            "extra": self.config.extra,
        }
        self.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _host_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "revision": _git_revision(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda"] = torch.version.cuda
            info["devices"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        info["torch"] = None
    return info


def _git_revision() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None
