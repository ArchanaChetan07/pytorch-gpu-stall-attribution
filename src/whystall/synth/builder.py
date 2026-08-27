"""Build Kineto-format traces programmatically.

This exists so the classifier can be developed and regression-tested without a
GPU, which is most of the project's calendar time. It is not a mock: the builder
models the thing that actually produces idle windows, which is that a kernel
starts at max(stream free, launch time + launch latency). Feed the host timeline
a delay and the device gap appears on its own, with the same shape it has in a
real trace.

Traces produced here are valid Chrome Trace Event documents and open in
chrome://tracing and Perfetto, which is a cheap way to eyeball a scenario.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..trace.schema import LABEL_PREFIX

__all__ = ["TraceBuilder", "DeviceConfig"]


@dataclass(frozen=True, slots=True)
class DeviceConfig:
    device: int = 0
    compute_stream: int = 7
    copy_stream: int = 20
    nccl_stream: int = 21
    #: Microseconds between a launch returning on the host and the kernel
    #: starting on the device, when the stream is already free.
    launch_latency_us: int = 4
    #: How far ahead of the device the host is allowed to run before a launch
    #: blocks. Real training loops are bounded this way -- by queue depth, by the
    #: caching allocator, by the backward pass needing a result -- and without
    #: the bound a synthetic host sprints arbitrarily far ahead and no injected
    #: stall ever reaches the device.
    max_queue_us: int = 2500


@dataclass
class TraceBuilder:
    """Accumulates trace events on a host timeline and one device."""

    pid: int = 1000
    tid: int = 1000
    dev: DeviceConfig = field(default_factory=DeviceConfig)
    #: Host clock cursor, microseconds. Advanced by the emit helpers.
    t: int = 1_000_000

    _events: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _corr: int = field(default=1, repr=False)
    _ext: int = field(default=1, repr=False)
    _stream_free: dict[int, int] = field(default_factory=dict, repr=False)

    # --- primitives ----------------------------------------------------------

    def _next_corr(self) -> int:
        self._corr += 1
        return self._corr

    def _next_ext(self) -> int:
        self._ext += 1
        return self._ext

    def _emit(self, ev: dict[str, Any]) -> dict[str, Any]:
        self._events.append(ev)
        return ev

    def cpu_op(self, name: str, ts: int, dur: int, *, cat: str = "cpu_op") -> dict[str, Any]:
        return self._emit(
            {
                "ph": "X",
                "cat": cat,
                "name": name,
                "pid": self.pid,
                "tid": self.tid,
                "ts": ts,
                "dur": dur,
                "args": {"External id": self._next_ext()},
            }
        )

    def annotation(self, name: str, ts: int, dur: int) -> dict[str, Any]:
        return self.cpu_op(name, ts, dur, cat="user_annotation")

    def label(self, kind: str, ts: int, dur: int, *parts: str) -> dict[str, Any]:
        """Emit a WHYSTALL annotation. Only validation runs contain these."""
        name = LABEL_PREFIX + "/".join((kind, *parts))
        return self.cpu_op(name, ts, dur, cat="user_annotation")

    def runtime(self, name: str, ts: int, dur: int, *, corr: int | None = None) -> int:
        corr = corr if corr is not None else self._next_corr()
        self._emit(
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": name,
                "pid": self.pid,
                "tid": self.tid,
                "ts": ts,
                "dur": dur,
                "args": {"correlation": corr, "External id": self._ext},
            }
        )
        return corr

    def device_op(
        self,
        name: str,
        ts: int,
        dur: int,
        corr: int,
        *,
        stream: int | None = None,
        cat: str = "kernel",
        nbytes: int | None = None,
    ) -> dict[str, Any]:
        stream = stream if stream is not None else self.dev.compute_stream
        args: dict[str, Any] = {
            "correlation": corr,
            "device": self.dev.device,
            "stream": stream,
        }
        if nbytes is not None:
            args["bytes"] = nbytes
        self._stream_free[stream] = ts + dur
        return self._emit(
            {
                "ph": "X",
                "cat": cat,
                "name": name,
                "pid": self.dev.device,
                "tid": stream,
                "ts": ts,
                "dur": dur,
                "args": args,
            }
        )

    # --- the model -----------------------------------------------------------

    def enqueue(
        self,
        name: str,
        *,
        gpu_us: int,
        host_us: int = 3,
        stream: int | None = None,
        cat: str = "kernel",
        launch_name: str = "cudaLaunchKernel",
        nbytes: int | None = None,
        op_name: str | None = None,
        start_no_earlier_than: int | None = None,
    ) -> int:
        """Launch one device op from the current host cursor.

        The device op starts at max(stream free, launch end + latency), which is
        the whole reason a late host produces a visible gap. Returns the device
        op's start time.

        start_no_earlier_than models a cross-stream event wait: the work is
        enqueued on time but cannot run yet. That is how a device dependency is
        produced, and it is the case that must not be counted as starvation.
        """
        stream = stream if stream is not None else self.dev.compute_stream
        # Block the host if it has run too far ahead of the device.
        backlog = self._stream_free.get(stream, self.t) - self.t
        if backlog > self.dev.max_queue_us:
            self.t = self._stream_free[stream] - self.dev.max_queue_us
        if op_name:
            self.cpu_op(op_name, self.t, host_us)
        # The launch sits at the end of the operator's host work, as it does in a
        # real trace: dispatch happens first, the launch call last. That ordering
        # is what puts dispatch inside the blame window and the launch itself at
        # its boundary.
        launch_dur = min(host_us, 3)
        corr = self.runtime(launch_name, self.t + host_us - launch_dur, launch_dur)
        launch_end = self.t + host_us
        earliest = launch_end + self.dev.launch_latency_us
        start = max(self._stream_free.get(stream, 0), earliest)
        if start_no_earlier_than is not None:
            start = max(start, start_no_earlier_than)
        self.device_op(name, start, gpu_us, corr, stream=stream, cat=cat, nbytes=nbytes)
        self.t = launch_end
        return start

    def host_work(self, name: str, dur: int, *, cat: str = "cpu_op") -> tuple[int, int]:
        """Occupy the host for dur microseconds. Returns (start, end)."""
        start = self.t
        self.cpu_op(name, start, dur, cat=cat)
        self.t += dur
        return start, self.t

    def synchronize(self, *, wakeup_us: int = 30, stream: int | None = None) -> tuple[int, int]:
        """Block the host until the device drains, plus a wakeup cost.

        The wakeup tail is what makes a host sync visible as a stall: the device
        is already idle while the host is still inside the call.
        """
        stream = stream if stream is not None else self.dev.compute_stream
        start = self.t
        drain = max(self._stream_free.get(stream, start), start)
        dur = (drain - start) + wakeup_us
        self.runtime("cudaStreamSynchronize", start, dur)
        self.t = start + dur
        return start, self.t

    def device_busy_until(self, stream: int | None = None) -> int:
        stream = stream if stream is not None else self.dev.compute_stream
        return self._stream_free.get(stream, self.t)

    # --- output --------------------------------------------------------------

    def build(self, **meta: Any) -> dict[str, Any]:
        doc: dict[str, Any] = {
            "schemaVersion": 1,
            "deviceProperties": [
                {"id": self.dev.device, "name": "synthetic", "totalGlobalMem": 1 << 34}
            ],
            "traceEvents": sorted(self._events, key=lambda e: (e["ts"], -e.get("dur", 0))),
        }
        doc.update(meta)
        return doc

    def write(self, path: str | os.PathLike[str], **meta: Any) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.build(**meta)), encoding="utf-8")
        return p
