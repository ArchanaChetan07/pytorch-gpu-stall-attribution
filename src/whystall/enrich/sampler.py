"""The /proc sidecar: what the workers were doing while the GPU was idle.

A trace says the launching thread was blocked inside the dataloader. It cannot
say whether the workers behind that iterator were on the disk or on the CPU,
because nothing in the trace observes them. This does, at a couple of hundred
hertz, from outside the training process.

Three signals, in descending order of usefulness:

read_bytes
    From /proc/<pid>/io. Counts bytes the process actually fetched from a block
    device, so it distinguishes a warm page cache from a real disk read. This is
    the signal that separates "storage can fix this" from "storage cannot", and
    it is the reason the sidecar exists.
process state
    D is uninterruptible sleep, which in practice means waiting on I/O; R is
    runnable. The classic discriminator, but it aliases: D also covers some
    non-disk waits, and a worker decoding JPEGs in a thread pool can look
    briefly idle.
pressure stall information
    /proc/pressure/io, when the kernel provides it. System-wide rather than
    per-process, so it corroborates rather than decides.

Linux only, by construction -- this module imports and its data types work
anywhere, so the refinement logic stays testable off Linux, but run() will
refuse to start without /proc.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "WorkerSample",
    "Sample",
    "SamplerLog",
    "WindowStats",
    "available",
    "discover_workers",
    "parse_ppid",
    "parse_stat",
    "sample_once",
    "run",
]

_PROC = Path("/proc")


def available() -> bool:
    """True when this host exposes the /proc interfaces the sampler needs."""
    return sys.platform.startswith("linux") and _PROC.is_dir()


@dataclass(frozen=True, slots=True)
class WorkerSample:
    pid: int
    #: Single-letter process state: R runnable, S sleeping, D uninterruptible.
    state: str
    utime: int
    stime: int
    #: Bytes fetched from a block device since process start. -1 when unreadable
    #: (it needs matching uid or CAP_SYS_PTRACE).
    read_bytes: int = -1
    #: Bytes read through the read() syscall family, cache hits included.
    rchar: int = -1


@dataclass(frozen=True, slots=True)
class Sample:
    t_mono_ns: int
    workers: tuple[WorkerSample, ...]
    psi_io_some: float | None = None
    psi_cpu_some: float | None = None

    def to_json(self) -> dict:
        return {
            "t": self.t_mono_ns,
            "w": [[w.pid, w.state, w.utime, w.stime, w.read_bytes, w.rchar] for w in self.workers],
            "psi_io": self.psi_io_some,
            "psi_cpu": self.psi_cpu_some,
        }

    @classmethod
    def from_json(cls, d: dict) -> Sample:
        return cls(
            t_mono_ns=int(d["t"]),
            workers=tuple(
                WorkerSample(
                    pid=int(w[0]),
                    state=str(w[1]),
                    utime=int(w[2]),
                    stime=int(w[3]),
                    read_bytes=int(w[4]),
                    rchar=int(w[5]),
                )
                for w in d.get("w", ())
            ),
            psi_io_some=d.get("psi_io"),
            psi_cpu_some=d.get("psi_cpu"),
        )


# --- reading /proc -----------------------------------------------------------


def parse_stat(raw: str) -> tuple[str, int, int] | None:
    """(state, utime, stime) from the contents of /proc/<pid>/stat.

    Split on the LAST close paren, never on whitespace: the comm field is
    parenthesised and may itself contain spaces and parens. A process named
    "python (worker 3)" breaks every naive parser, and dataloader workers are
    exactly the processes people rename.
    """
    close = raw.rfind(")")
    if close < 0:
        return None
    fields = raw[close + 2 :].split()
    if len(fields) < 13:
        return None
    try:
        return fields[0], int(fields[11]), int(fields[12])
    except (ValueError, IndexError):
        return None


def parse_ppid(raw: str) -> int | None:
    """Parent pid from the contents of /proc/<pid>/stat."""
    close = raw.rfind(")")
    if close < 0:
        return None
    fields = raw[close + 2 :].split()
    try:
        return int(fields[1])
    except (ValueError, IndexError):
        return None


def _read_stat(pid: int) -> tuple[str, int, int] | None:
    try:
        return parse_stat((_PROC / str(pid) / "stat").read_text())
    except (OSError, ValueError):
        return None


def _read_io(pid: int) -> tuple[int, int]:
    """(read_bytes, rchar), or (-1, -1) when not permitted or absent."""
    read_bytes = rchar = -1
    try:
        for line in (_PROC / str(pid) / "io").read_text().splitlines():
            key, _, value = line.partition(":")
            if key == "read_bytes":
                read_bytes = int(value)
            elif key == "rchar":
                rchar = int(value)
    except (OSError, ValueError):
        pass
    return read_bytes, rchar


def _read_psi(resource: str) -> float | None:
    try:
        for line in (_PROC / "pressure" / resource).read_text().splitlines():
            if line.startswith("some"):
                for tok in line.split():
                    if tok.startswith("avg10="):
                        return float(tok[6:])
    except (OSError, ValueError):
        pass
    return None


def discover_workers(parent_pid: int, *, include_parent: bool = True) -> list[int]:
    """Direct children of a process, which for a DataLoader are its workers.

    Reads /proc/<pid>/task/*/children, which lists children per thread. Falls
    back to scanning every process's ppid when that interface is unavailable.
    """
    pids: set[int] = {parent_pid} if include_parent else set()
    found = False
    try:
        for task in (_PROC / str(parent_pid) / "task").iterdir():
            try:
                kids = (task / "children").read_text().split()
            except OSError:
                continue
            found = True
            pids.update(int(k) for k in kids)
    except OSError:
        pass

    if not found:
        for entry in _PROC.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                ppid = parse_ppid((entry / "stat").read_text())
            except (OSError, ValueError):
                continue
            if ppid == parent_pid:
                pids.add(int(entry.name))
    return sorted(pids)


def sample_once(pids: Sequence[int], *, psi: bool = True) -> Sample:
    """One observation of every pid. Processes that have exited are skipped."""
    workers = []
    for pid in pids:
        st = _read_stat(pid)
        if st is None:
            continue
        read_bytes, rchar = _read_io(pid)
        workers.append(
            WorkerSample(
                pid=pid, state=st[0], utime=st[1], stime=st[2],
                read_bytes=read_bytes, rchar=rchar,
            )
        )
    return Sample(
        t_mono_ns=time.clock_gettime_ns(time.CLOCK_MONOTONIC),
        workers=tuple(workers),
        psi_io_some=_read_psi("io") if psi else None,
        psi_cpu_some=_read_psi("cpu") if psi else None,
    )


def run(
    parent_pid: int,
    out_path: str | os.PathLike[str],
    *,
    hz: int = 200,
    duration_s: float | None = None,
    rediscover_every: int = 200,
) -> Path:
    """Sample until the parent exits or duration_s elapses. Writes JSONL.

    Workers are rediscovered periodically because a DataLoader with
    persistent_workers=False respawns them every epoch, and a sampler pinned to
    the first generation quietly stops observing anything.
    """
    if not available():
        raise RuntimeError("the sidecar sampler needs Linux /proc; this host has none")

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    period = 1.0 / hz
    deadline = None if duration_s is None else time.monotonic() + duration_s
    pids = discover_workers(parent_pid)
    n = 0

    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": {"hz": hz, "parent": parent_pid}}) + "\n")
        next_at = time.monotonic()
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                break
            if not (_PROC / str(parent_pid)).exists():
                break
            if n % rediscover_every == 0:
                pids = discover_workers(parent_pid)
            fh.write(json.dumps(sample_once(pids).to_json(), separators=(",", ":")) + "\n")
            n += 1
            next_at += period
            sleep = next_at - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                # Fell behind; resynchronise rather than accumulating debt.
                next_at = time.monotonic()
    return path


# --- windowed queries --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WindowStats:
    """What the workers were doing during one trace-time window."""

    samples: int
    observations: int
    #: Length of the window asked about, microseconds. Carried so that a refusal
    #: to decide can say how long the window was and how many samples that
    #: implies at the sampling rate in use.
    window_us: float
    frac_uninterruptible: float
    frac_runnable: float
    frac_sleeping: float
    #: Block-device bytes read across all workers during the window.
    read_bytes: int
    read_mb_per_s: float
    #: Fraction of a full core, per worker, spent on CPU during the window.
    cpu_per_worker: float
    psi_io_some: float | None
    #: True when no worker exposed read_bytes, so the I/O signal is absent
    #: rather than zero. The distinction matters and must not be silently lost.
    read_bytes_unavailable: bool

    def summary(self) -> dict:
        return {
            "samples": self.samples,
            "window_us": round(self.window_us, 1),
            "frac_D": round(self.frac_uninterruptible, 4),
            "frac_R": round(self.frac_runnable, 4),
            "read_bytes": self.read_bytes,
            "read_mb_per_s": round(self.read_mb_per_s, 3),
            "cpu_per_worker": round(self.cpu_per_worker, 4),
            "psi_io_some": self.psi_io_some,
            "read_bytes_unavailable": self.read_bytes_unavailable,
        }


@dataclass(slots=True)
class SamplerLog:
    """Loaded sidecar output, queryable by trace-time window."""

    samples: list[Sample]
    hz: int = 0
    meta: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.meta is None:
            self.meta = {}
        self.samples.sort(key=lambda s: s.t_mono_ns)

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> SamplerLog:
        samples: list[Sample] = []
        meta: dict = {}
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if "meta" in d:
                    meta = d["meta"]
                    continue
                samples.append(Sample.from_json(d))
        return cls(samples=samples, hz=int(meta.get("hz", 0)), meta=meta)

    def between_ns(self, lo_ns: float, hi_ns: float) -> list[Sample]:
        return [s for s in self.samples if lo_ns <= s.t_mono_ns <= hi_ns]

    def stats_ns(self, lo_ns: float, hi_ns: float, *, clock_ticks: int = 100) -> WindowStats:
        """Aggregate worker state across a window given in sampler nanoseconds.

        Deltas need a bracketing sample on each side, so the window is widened by
        one sample before differencing counters. Without that, a short window
        that contains a single sample reports zero I/O regardless of the truth.
        """
        window_us = max(0.0, (hi_ns - lo_ns) / 1000.0)
        window = self.between_ns(lo_ns, hi_ns)
        if not window:
            return WindowStats(0, 0, window_us, 0.0, 0.0, 0.0, 0, 0.0, 0.0, None, True)

        states: dict[str, int] = {}
        obs = 0
        for s in window:
            for w in s.workers:
                states[w.state] = states.get(w.state, 0) + 1
                obs += 1

        first, last = self._bracket(lo_ns, hi_ns)
        read_bytes = 0
        cpu_ticks = 0
        any_read = False
        if first is not None and last is not None and last.t_mono_ns > first.t_mono_ns:
            before = {w.pid: w for w in first.workers}
            for w in last.workers:
                prev = before.get(w.pid)
                if prev is None:
                    continue
                if w.read_bytes >= 0 and prev.read_bytes >= 0:
                    any_read = True
                    read_bytes += max(0, w.read_bytes - prev.read_bytes)
                cpu_ticks += max(0, (w.utime + w.stime) - (prev.utime + prev.stime))
            span_s = (last.t_mono_ns - first.t_mono_ns) / 1e9
        else:
            span_s = max((hi_ns - lo_ns) / 1e9, 1e-9)

        n_workers = max(1, max(len(s.workers) for s in window))
        psi = [s.psi_io_some for s in window if s.psi_io_some is not None]

        return WindowStats(
            samples=len(window),
            observations=obs,
            window_us=window_us,
            frac_uninterruptible=states.get("D", 0) / obs if obs else 0.0,
            frac_runnable=states.get("R", 0) / obs if obs else 0.0,
            frac_sleeping=states.get("S", 0) / obs if obs else 0.0,
            read_bytes=read_bytes,
            read_mb_per_s=(read_bytes / 1e6 / span_s) if span_s > 0 else 0.0,
            cpu_per_worker=(cpu_ticks / clock_ticks / span_s / n_workers) if span_s > 0 else 0.0,
            psi_io_some=(sum(psi) / len(psi)) if psi else None,
            read_bytes_unavailable=not any_read,
        )

    def _bracket(self, lo_ns: float, hi_ns: float) -> tuple[Sample | None, Sample | None]:
        first = last = None
        for s in self.samples:
            if s.t_mono_ns <= lo_ns:
                first = s
            if s.t_mono_ns >= hi_ns and last is None:
                last = s
        inside = self.between_ns(lo_ns, hi_ns)
        if first is None and inside:
            first = inside[0]
        if last is None and inside:
            last = inside[-1]
        return first, last

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)
