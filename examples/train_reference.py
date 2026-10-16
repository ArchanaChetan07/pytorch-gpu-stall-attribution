"""A complete injected run, ready to point at a real model and dataset.

Run one configuration:

    python examples/train_reference.py --injector dataloader_io --magnitude-us 4000

Run the whole sweep the plan calls for, controls included:

    python examples/train_reference.py --sweep --out runs/

Then score it:

    whystall validate runs/ --out validation.html

Everything below the SWEEP section is generic. To use a real workload, replace
build_dataset and build_model; nothing else changes, because the harness only
needs the four scopes to be placed around the right parts of the step.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# The magnitudes to sweep. Five points spanning clearly-undetectable to
# obviously-dominant, so the recall curve has something to interpolate between.
# A floor read off two points is not a floor.
MAGNITUDES_US = (500, 1500, 4000, 10_000, 25_000)

INJECTORS = (
    "dataloader_io",
    "dataloader_cpu",
    "h2d",
    "host_sync",
    "other_host",
    "device_dependency",
)


class SyntheticImages:
    """A synthetic image corpus.

    Module scope, not a local class inside build_dataset, for the same reason
    _WrappedDataset is not a closure: DataLoader pickles the dataset when
    num_workers > 0 under the spawn and forkserver start methods, and a class
    defined inside a function cannot be pickled by qualified name. Under fork it
    works and hides, so this only bites on Windows and macOS -- or on Linux the
    moment anyone sets a start method, which CUDA safety regularly makes people
    do, since initialising CUDA before forking workers is its own hazard.

    Duck-typed rather than subclassing torch.utils.data.Dataset, which keeps
    torch out of this module's import path so --help works without it.

    torch is imported inside __getitem__ rather than held on the instance: a
    module attribute on self is not picklable and would reintroduce the failure
    this class exists to avoid. The import is a cached sys.modules lookup.
    """

    def __init__(self, n: int, px: int) -> None:
        self.n = n
        self.px = px

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        import torch

        g = torch.Generator().manual_seed(i)
        return (
            torch.rand(3, self.px, self.px, generator=g),
            int(torch.randint(0, 1000, (1,), generator=g).item()),
        )


def build_dataset(size: int = 4096, image_px: int = 224):
    """A synthetic image corpus.

    Synthetic on purpose for the sweep: it makes decode cost and bytes-per-item
    independently dialable, which is exactly what calibrating the I/O-versus-CPU
    threshold needs, and it keeps dataset download off the rental clock.

    Swap in torchvision.datasets.ImageFolder for the field number -- ImageNet is
    what the storage vendors publish against and the comparison has to be
    like-for-like.
    """
    return SyntheticImages(size, image_px)


def build_model():
    import torch

    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 7, stride=2, padding=3),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(64, 128, 3, padding=1),
        torch.nn.BatchNorm2d(128),
        torch.nn.ReLU(inplace=True),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(128, 1000),
    )


def dist_context():
    """Rank, world size and local rank, whether or not torchrun launched us.

    Returns (rank, world_size, local_rank). Outside torchrun this is (0, 1, 0)
    and every distributed path below turns itself off.
    """
    import os

    return (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
        int(os.environ.get("LOCAL_RANK", "0")),
    )


def run_one(
    injector_name: str,
    magnitude_us: int,
    *,
    out_dir: Path,
    steps: int,
    batch_size: int,
    num_workers: int,
    cache_state: str,
    sampler_hz: int,
    run_name: str | None = None,
) -> Path:
    import torch
    from torch.utils.data import DataLoader

    from whystall.inject import Harness, HarnessConfig, build

    rank, world_size, local_rank = dist_context()
    distributed = world_size > 1

    if torch.cuda.is_available():
        device = f"cuda:{local_rank}" if distributed else "cuda"
        torch.cuda.set_device(local_rank if distributed else 0)
    else:
        device = "cpu"
        print("warning: no CUDA device; the trace will contain no device ops",
              file=sys.stderr)

    if distributed:
        import torch.distributed as dist

        if not dist.is_initialized():
            # nccl on GPUs, gloo otherwise -- gloo is what makes the whole
            # distributed path rehearsable on a laptop with no CUDA at all.
            dist.init_process_group(
                backend="nccl" if torch.cuda.is_available() else "gloo"
            )

    injector = build(injector_name, magnitude_us)
    # Every rank writes its own files. Without the suffix all ranks race for one
    # path and you end up with one surviving trace out of N, which looks like a
    # successful run right up until you score it.
    base_name = run_name or f"{injector_name}-{magnitude_us}us"
    config = HarnessConfig(
        out_dir=out_dir,
        run_name=f"{base_name}-rank{rank}" if distributed else base_name,
        warmup_steps=max(4, steps // 8),
        active_steps=steps,
        sampler_hz=sampler_hz,
        cache_state=cache_state,
        notes=(
            f"reference script, batch={batch_size}, workers={num_workers}"
            + (f", world_size={world_size}, rank={rank}" if distributed else "")
        ),
        extra={"rank": rank, "world_size": world_size} if distributed else {},
    )

    model = build_model().to(device)
    if distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP

        # DDP is not decoration here: the gradient allreduce during backward is
        # the only real collective in this script, and it is the thing the
        # straggler makes the other ranks wait inside. Without it the collective
        # injector has nothing to stall on and the second GPU buys nothing.
        model = DDP(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
        )
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    harness = Harness(injector, config)
    dataset = harness.wrap_dataset(build_dataset())
    loader = DataLoader(
        dataset,
        **harness.loader_kwargs(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2 if num_workers > 0 else None,
        ),
    )

    with harness:
        it = iter(loader)
        for _ in range(config.warmup_steps + config.active_steps):
            with harness.step():
                with harness.fetch():
                    try:
                        images, targets = next(it)
                    except StopIteration:
                        it = iter(loader)
                        images, targets = next(it)

                with harness.transfer():
                    images = images.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)

                with harness.compute():
                    optimiser.zero_grad(set_to_none=True)
                    loss = loss_fn(model(images), targets)
                    loss.backward()
                    optimiser.step()

                with harness.step_end():
                    pass

    if distributed:
        # Barrier before anyone returns: a rank that exits while another is
        # still inside the allreduce takes the whole job down with a timeout
        # that looks like a hang, several runs later and nowhere near the cause.
        import torch.distributed as dist

        dist.barrier()

    print(f"wrote {harness.trace_path}")
    return harness.trace_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--injector", default="control",
                   help=f"one of {', '.join(INJECTORS)}, or control")
    p.add_argument("--magnitude-us", type=int, default=4000)
    p.add_argument("--sweep", action="store_true",
                   help="run every injector at every magnitude, plus controls")
    p.add_argument("--injectors", default=None,
                   help="comma-separated subset to sweep, e.g. 'collective'. "
                        "Use this to run the collective phase on its own, so the "
                        "other classes are not measured under DDP")
    p.add_argument("--no-controls", action="store_true",
                   help="skip the control runs (phase 2 reuses phase 1's)")
    p.add_argument("--out", default="runs")
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--sampler-hz", type=int, default=500,
                   help="raise this if the stalls you care about are under 15 ms")
    p.add_argument("--cache-state", default="unknown",
                   choices=("cold", "warm", "unknown"),
                   help="recorded in the manifest, never verified -- say what you did")
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    common = dict(
        out_dir=out_dir, steps=args.steps, batch_size=args.batch_size,
        num_workers=args.num_workers, cache_state=args.cache_state,
        sampler_hz=args.sampler_hz,
    )

    if not args.sweep:
        run_one(args.injector, args.magnitude_us, **common)
        return 0

    # Which classes this phase sweeps.
    #
    # The sweep runs in two phases on purpose, and they must not be merged.
    # Wrapping the model in DDP puts a gradient allreduce in every step, so a
    # dataloader or h2d run measured under torchrun is no longer the run the
    # plan specifies -- it carries a collective the single-process baseline
    # never had. Phase 1 therefore sweeps the six single-process classes with no
    # DDP at all; phase 2 adds only `collective`, under torchrun, and reuses
    # phase 1's controls.
    _, world_size, _ = dist_context()
    if args.injectors:
        injectors = tuple(n.strip() for n in args.injectors.split(",") if n.strip())
        unknown = [n for n in injectors if n not in INJECTORS + ("collective",)]
        if unknown:
            print(f"error: unknown injector(s) {unknown}; known: "
                  f"{sorted(INJECTORS + ('collective',))}", file=sys.stderr)
            return 2
    else:
        injectors = INJECTORS
    if "collective" in injectors and world_size == 1:
        print("error: the collective class needs more than one rank. Launch it as\n"
              "    torchrun --nproc_per_node=2 examples/train_reference.py \\\n"
              "        --sweep --injectors collective --no-controls --out runs/",
              file=sys.stderr)
        return 2
    if world_size > 1 and any(n != "collective" for n in injectors):
        print("warning: running non-collective classes under torchrun wraps the "
              "model in DDP, which adds a gradient allreduce to every step. Those "
              "runs will not match a single-process baseline. Phase 1 is meant to "
              "run without torchrun.", file=sys.stderr)

    # Controls first. A sweep without them cannot be interpreted: precision is a
    # lower bound in this setting, and the control distribution is the only
    # thing that turns it back into a measurement.
    #
    # Each control gets its own run_name up front. Renaming the files afterwards
    # would leave every control writing to the same sampler log and silently
    # overwriting the previous one.
    if not args.no_controls:
        for i in range(3):
            run_one("control", 0, run_name=f"control-{i}", **common)

    total = len(injectors) * len(MAGNITUDES_US)
    done = 0
    for name in injectors:
        for magnitude in MAGNITUDES_US:
            done += 1
            print(f"[{done}/{total}] {name} at {magnitude} us")
            run_one(name, magnitude, **common)

    print(f"\ndone. score it with:\n    whystall validate {out_dir} --out validation.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
