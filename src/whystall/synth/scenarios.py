"""Synthetic training runs with one known stall class injected.

Each scenario builds a plausible step -- fetch, transfer, forward, backward,
optimiser -- and perturbs exactly one part of it. The perturbed region carries a
WHYSTALL/inject annotation, so the resulting trace is self-labelling in the same
way a real injected run is, and the same scoring code works on both.

Two properties are deliberate:

* The GPU is given enough queued work to absorb a small stall. Injecting 5 ms
  does not produce 5 ms of idle, exactly as on real hardware, which is why
  ground truth is the labelled gap and never the injected magnitude.
* The clean scenario injects nothing. It is the control run, and the classes it
  produces are the false-positive rate.
"""

from __future__ import annotations

from typing import Any

from .builder import DeviceConfig, TraceBuilder

__all__ = ["STALL_KINDS", "build_scenario", "GROUND_TRUTH_CLASS"]

#: Scenario name -> the stall class a correct classifier should return. The
#: dataloader variants share a trace-level shape and differ only in what the
#: sampler sees, which is the point of the split.
GROUND_TRUTH_CLASS: dict[str, str | None] = {
    "clean": None,
    "dataloader": "dataloader",
    "dataloader_io": "dataloader_io",
    "dataloader_cpu": "dataloader_cpu",
    "h2d": "h2d_copy",
    "host_sync": "host_sync",
    "collective": "collective",
    "launch": "launch_dispatch",
    "other_host": "other_host",
    "device_dependency": "device_dependency",
}

STALL_KINDS: tuple[str, ...] = tuple(GROUND_TRUTH_CLASS)

_DATALOADER_ANNOTATION = "enumerate(DataLoader)#_MultiProcessingDataLoaderIter.__next__"

# Baseline costs, microseconds. Sized so a clean step keeps the device busy with
# roughly 1.4 ms of runway -- enough that a small injection is absorbed.
_DL_BASE = 200
_COPY_HOST_BASE = 60
_COPY_GPU = 120
_FWD_KERNELS = 6
_FWD_GPU = 300
_FWD_HOST = 12
_BWD_KERNELS = 6
_BWD_GPU = 340
_BWD_HOST = 14
_OPT_GPU = 180
_OPT_HOST = 20
# Dispatch-bound shape: host cost per operator far exceeds the kernel it starts.
_LAUNCH_KERNELS = 90
_LAUNCH_GPU = 2
_LAUNCH_HOST = 34


def build_scenario(
    kind: str,
    *,
    steps: int = 8,
    stall_us: int = 5000,
    label: bool = True,
    every: int = 1,
    distributed: bool | None = None,
) -> dict[str, Any]:
    """Build a trace for one scenario.

    every=2 injects on alternate steps, which is how a paired within-run control
    is produced when two separate runs are not available.
    """
    if kind not in GROUND_TRUTH_CLASS:
        raise ValueError(f"unknown scenario {kind!r}; expected one of {STALL_KINDS}")
    if distributed is None:
        distributed = kind == "collective"

    b = TraceBuilder(dev=DeviceConfig())
    gt = GROUND_TRUTH_CLASS[kind]

    for n in range(steps):
        injecting = label and gt is not None and (n % every == 0)
        _step(b, n, kind, stall_us, injecting, distributed)

    return b.build(
        whystall={
            "scenario": kind,
            "steps": steps,
            "stall_us": stall_us,
            "every": every,
            "ground_truth": gt,
            "synthetic": True,
        }
    )


def _step(
    b: TraceBuilder,
    n: int,
    kind: str,
    stall_us: int,
    injecting: bool,
    distributed: bool,
) -> None:
    step_start = b.t
    gt = GROUND_TRUTH_CLASS[kind] or ""

    def mark(start: int, end: int) -> None:
        if injecting and end > start:
            b.label("inject", start, end - start, gt, str(stall_us))

    # 1. Fetch the batch.
    dl_extra = stall_us if kind.startswith("dataloader") else 0
    dl_start, dl_end = b.t, b.t + _DL_BASE + dl_extra
    b.annotation(_DATALOADER_ANNOTATION, dl_start, dl_end - dl_start)
    b.t = dl_end
    if dl_extra:
        mark(dl_start, dl_end)

    # 2. Move it to the device. A pageable source makes the copy synchronous,
    #    which is why the injected variant also changes the memcpy's name.
    copy_extra = stall_us if kind == "h2d" else 0
    pinned = "Pageable -> Device" if copy_extra else "Pinned -> Device"
    copy_start = b.t
    b.enqueue(
        f"Memcpy HtoD ({pinned})",
        gpu_us=_COPY_GPU + (copy_extra // 4),
        host_us=_COPY_HOST_BASE + copy_extra,
        stream=b.dev.copy_stream,
        cat="gpu_memcpy",
        launch_name="cudaMemcpy" if copy_extra else "cudaMemcpyAsync",
        nbytes=48 * 1024 * 1024,
        op_name="aten::to",
    )
    if copy_extra:
        mark(copy_start, b.t)

    # 3. Forward and backward. The launch-bound variant replaces the fused
    #    kernels with an unfused chain rather than appending one: a model that
    #    is dispatch-bound is dispatch-bound throughout, and appending tiny
    #    kernels behind 2 ms of queued work just hides them in the backlog.
    if kind == "launch":
        launch_start = b.t
        for _ in range(_LAUNCH_KERNELS):
            b.enqueue(
                "void at::native::vectorized_elementwise_kernel<4, float>",
                gpu_us=_LAUNCH_GPU,
                host_us=_LAUNCH_HOST,
                op_name="aten::add",
            )
        mark(launch_start, b.t)
    else:
        for i in range(_FWD_KERNELS):
            b.enqueue(
                "void at::native::vectorized_elementwise_kernel<4, float>"
                if i % 2
                else "void cutlass::Kernel<cutlass_80_tensorop_s1688gemm>",
                gpu_us=_FWD_GPU,
                host_us=_FWD_HOST,
                op_name="aten::conv2d" if i % 2 == 0 else "aten::relu",
            )

    if kind == "device_dependency":
        # Enqueued on time from a host that is running ahead, but unable to run
        # until a cross-stream event fires. The launch precedes the gap, so this
        # must not be counted as host wait.
        blocked_until = b.device_busy_until() + stall_us
        b.enqueue(
            "void at::native::cross_stream_consumer_kernel",
            gpu_us=_FWD_GPU,
            host_us=8,
            op_name="aten::mul",
            start_no_earlier_than=blocked_until,
        )
        if injecting:
            b.label("inject", b.t, max(1, stall_us), gt, str(stall_us))

    if kind != "launch":
        for _ in range(_BWD_KERNELS):
            b.enqueue(
                "void at::native::batch_norm_backward_kernel<float>",
                gpu_us=_BWD_GPU,
                host_us=_BWD_HOST,
                op_name="autograd::engine::evaluate_function: ConvolutionBackward0",
            )

    # 4. Gradient all-reduce.
    if distributed:
        straggler = stall_us if kind == "collective" else 0
        if straggler:
            c0 = b.t
            b.host_work("record_param_comms", straggler)
            mark(c0, b.t)
        b.enqueue(
            "ncclDevKernel_AllReduce_Sum_f32_RING_LL(ncclDevComm*)",
            gpu_us=400,
            host_us=10,
            stream=b.dev.nccl_stream,
            op_name="c10d::allreduce_",
        )

    # 5. Optimiser.
    b.enqueue(
        "void at::native::multi_tensor_apply_kernel<FusedAdam>",
        gpu_us=_OPT_GPU,
        host_us=_OPT_HOST,
        op_name="aten::_fused_adam_",
    )

    # 6. Host-side epilogue.
    if kind == "other_host":
        o0 = b.t
        b.host_work("EpochMetrics.update", stall_us)
        mark(o0, b.t)

    if kind == "host_sync":
        s0 = b.t
        b.cpu_op("aten::item", s0, 0)
        b.synchronize(wakeup_us=stall_us)
        b.cpu_op("aten::item", s0, b.t - s0)
        mark(s0, b.t)

    b.annotation(f"ProfilerStep#{n}", step_start, max(1, b.t - step_start))
