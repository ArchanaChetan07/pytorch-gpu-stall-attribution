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


def build_dataset(size: int = 4096, image_px: int = 224):
    """A synthetic image corpus.

    Synthetic on purpose for the sweep: it makes decode cost and bytes-per-item
    independently dialable, which is exactly what calibrating the I/O-versus-CPU
    threshold needs, and it keeps dataset download off the rental clock.

    Swap in torchvision.datasets.ImageFolder for the field number -- ImageNet is
    what the storage vendors publish against and the comparison has to be
    like-for-like.
    """
    import torch
    from torch.utils.data import Dataset

    class SyntheticImages(Dataset):
        def __init__(self, n: int, px: int) -> None:
            self.n = n
            self.px = px

        def __len__(self) -> int:
            return self.n

        def __getitem__(self, i: int):
            g = torch.Generator().manual_seed(i)
            return (
                torch.rand(3, self.px, self.px, generator=g),
                int(torch.randint(0, 1000, (1,), generator=g).item()),
            )

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
) -> Path:
    import torch
    from torch.utils.data import DataLoader

    from whystall.inject import Harness, HarnessConfig, build

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("warning: no CUDA device; the trace will contain no device ops",
              file=sys.stderr)

    injector = build(injector_name, magnitude_us)
    config = HarnessConfig(
        out_dir=out_dir,
        run_name=f"{injector_name}-{magnitude_us}us",
        warmup_steps=max(4, steps // 8),
        active_steps=steps,
        sampler_hz=sampler_hz,
        cache_state=cache_state,
        notes=f"reference script, batch={batch_size}, workers={num_workers}",
    )

    model = build_model().to(device)
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

    # Controls first. A sweep without them cannot be interpreted: precision is a
    # lower bound in this setting, and the control distribution is the only
    # thing that turns it back into a measurement.
    for i in range(3):
        run_one("control", 0, **{**common, "out_dir": out_dir})
        (out_dir / "control-0us.manifest.json").rename(
            out_dir / f"control-{i}.manifest.json"
        )
        (out_dir / "control-0us.trace.json").rename(out_dir / f"control-{i}.trace.json")

    total = len(INJECTORS) * len(MAGNITUDES_US)
    done = 0
    for name in INJECTORS:
        for magnitude in MAGNITUDES_US:
            done += 1
            print(f"[{done}/{total}] {name} at {magnitude} us")
            run_one(name, magnitude, **common)

    print(f"\ndone. score it with:\n    whystall validate {out_dir} --out validation.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
