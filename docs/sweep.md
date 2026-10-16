# The experiment plan

Write this before renting anything. A sweep improvised on a running meter is a
sweep you pay for twice.

## Rehearse the sweep on CPU before you rent anything

The orchestration does not need a GPU to be wrong. Run the whole sweep on your
laptop first, small and fast:

```bash
python examples/train_reference.py --sweep     --steps 6 --batch-size 4 --num-workers 2 --out /tmp/rehearse
```

Every run will warn that there is no CUDA device and the traces will contain no
device ops, so `whystall analyze` will refuse them -- that is correct and is not
what this is testing. What it does test is everything that is not the GPU: that
all 33 runs start and finish, that the dataset survives pickling to the worker
processes, that run names do not collide, that both clock sync points land
inside the profiled window, and that each manifest comes out complete.

**Use `--num-workers 2` or more, and do it on Windows or macOS if you have one.**
Those platforms default to the `spawn` start method, which pickles the dataset;
Linux defaults to `fork`, which does not, and therefore hides this entire class
of bug until you are paying for it. This rehearsal is how the
`SyntheticImages`-as-a-local-class bug was found -- a defect that would have
failed at the first fetch of the first run on any spawn host, and that the
library's own `_WrappedDataset` had been hardened against while the example
script quietly had not.

Check a manifest before you stop:

```bash
cat /tmp/rehearse/control-0.manifest.json
```

`clock_sync_points_ns` must hold **two** entries. One, or none, means the loop
ran fewer than `warmup_steps + active_steps` iterations and the sampler could
never have been aligned -- a failure that is otherwise invisible until analysis,
long after the box is released.

## Machine acceptance test, before anything else

Run these on every fresh rental. Each has cost someone a whole run.

```bash
# 1. Is the "SSD" a real local device or network block storage?
fio --name=r --rw=randread --bs=4k --iodepth=32 --numjobs=4 --size=2G \
    --runtime=30 --time_based --group_reporting --filename=/scratch/fio.test
```

Local NVMe: hundreds of thousands of IOPS, p99 latency in the hundreds of
microseconds. Network-backed: an order of magnitude worse with a completely
different tail. Every I/O result is contaminated if you get this wrong, and the
contamination is invisible in the numbers.

```bash
# 2. Can you inject at all?
dmsetup targets | grep delay || echo "no dm-delay: fallback injector only"
test -w /proc/sys/vm/drop_caches || echo "cannot drop caches: size past 2x RAM"
test -d /sys/fs/cgroup/cgroup.controllers && echo "cgroup v2 present"

# 3. Is the GPU the one you are paying for, and is it power-limited?
nvidia-smi --query-gpu=name,power.limit,power.max_limit,pcie.link.gen.current,pcie.link.width.current --format=csv
```

A card at a reduced power limit, or wired at PCIe gen3 x8 when you assumed gen4
x16, changes the H2D results specifically. Record all of it — `HarnessConfig`
writes host details into every manifest automatically.

## The sweep

| axis | values | why |
|---|---|---|
| class | 6 injectors | one per validatable class the box can produce |
| magnitude | 500, 1500, 4000, 10000, 25000 µs | five points, spanning undetectable to dominant |
| controls | 3 runs, no injection | the only thing that makes precision interpretable |
| repeats | 1 initially | add repeats only where the curve is ragged |

33 runs in phase 1, plus 5 more in phase 2. At 40 profiled steps each, that is
well under an hour of GPU time.

### Run it in two phases, and do not merge them

Wrapping the model in DDP puts a gradient allreduce in **every** step. A
dataloader or h2d run measured under `torchrun` is therefore not the run this
plan specifies -- it carries a collective the single-process baseline never had,
and comparing the two is comparing different workloads. So:

```bash
# Phase 1 -- the six single-process classes, no DDP, 33 runs.
python examples/train_reference.py --sweep --out runs/

# Phase 2 -- collective only, two ranks, reusing phase 1's controls. 5 runs x 2 ranks.
torchrun --nproc_per_node=2 examples/train_reference.py     --sweep --injectors collective --no-controls --out runs/

# Score both together.
whystall validate runs/ --out validation.html
```

`--no-controls` on phase 2 is deliberate: controls exist to characterise the
uninjected distribution, phase 1 already produced three, and a DDP control would
not be comparable to them anyway.

**Two GPUs for phase 2 only.** Collective wait cannot be injected or validated on
a single device, and it is the class that carries the tool beyond the storage
vendors. One node, two GPUs, no RDMA needed. Phase 1 runs on one.

The script refuses the ways this goes wrong, **before it runs anything**:

- `--injectors collective` without a second rank exits 2 and prints the torchrun
  line, rather than discovering it 33 runs later;
- an unknown injector name exits 2 and lists the valid ones;
- non-collective classes under `torchrun` warn that DDP has changed the workload.

All three fire ahead of the first run, so a mistyped phase costs nothing.

Two consequences of the straggler design, both of which show up in the output:

- **Each rank writes its own run**, suffixed `-rank0`, `-rank1`. Without the
  suffix every rank races for one path and you keep one trace out of N, which
  looks like a successful run until you try to score it.
- **Only the waiting ranks carry ground truth.** The straggler causes the stall
  but does not experience it -- its idle is its own spinning, which is host
  compute -- so its manifest reports `ground_truth_class: null` and its trace
  carries no inject label. That is correct. A straggler trace labelled
  `collective` would score every honest verdict on that rank as a miss.

The distributed path is rehearsable without any GPU at all: the backend falls
back to gloo when CUDA is absent, so two CPU processes exercise DDP, the
allreduce, the rank-suffixed naming and the asymmetric labelling.

## What each phase has to clear

The gates are the point. A gate you pass by lowering the bar was not a gate.

### Gate 1 — does blame resolution work at all

Two easiest injectors only: `host_sync` and `dataloader_cpu`.

**Pass:** recall ≥ 0.90 on both, and unattributed < 25% on the control run.

**Fail:** if unattributed exceeds half on an ordinary training loop, resolution
from Kineto is not strong enough and the project reshapes onto Nsight Systems,
which sees the blocking syscalls Kineto does not. Reshaping here costs a week.
Discovering it after the full sweep costs the project.

```bash
whystall validate runs/ --granularity coarse --out gate1.html
```

### Gate 2 — does the split that matters work

**Pass:** `dataloader_io` and `dataloader_cpu` separate at recall ≥ 0.85 above
the detection floor, scored `--granularity strict`.

**Fail:** the field number cannot be computed and the commercial argument goes
with it. That is survivable and it changes the product: publish a validated
six-class tool and state plainly that the split the vendors care about is not
achievable from host telemetry alone. Do not paper over it. A negative result
stated crisply still gets read; a soft claim that fails under replication does
not get a second look.

### Gate 3 — does it work on code you did not write

Three unmodified codebases, no edits to the target script — profiler attach plus
sidecar, nothing else:

- the torchvision ImageNet reference script (what the storage vendors benchmark);
- nanoGPT or torchtitan (compute-bound — the negative case, where the correct
  answer is *nothing is starving*);
- a HuggingFace `Trainer` run (different dataloader idioms).

**Pass:** all three run clean, and the unattributed share is reported per
codebase.

A tool that requires instrumenting the training loop is a tool nobody runs.

## Sampling rate

The sidecar defaults to 200 Hz. Refinement needs `min_samples` (3) observations
inside the blame window, so at 200 Hz nothing shorter than ~15 ms is refinable —
correct behaviour, easily mistaken for the sampler being broken.

If the stalls you care about are single-digit milliseconds, run at 500–1000 Hz
rather than lowering `min_samples`. The reports say which windows were skipped
and what rate would have resolved them.

## What to publish

Three method numbers and one field number. The method numbers make it
defensible; the field number is what travels.

1. **The confusion matrix**, including the unattributed row and the pairs it
   confuses.
2. **Detection floors** per class, with the bracketing sweep points shown so the
   width of the bracket is visible rather than implied.
3. **Unattributed share** on the three unmodified workloads.
4. **The field number** — on a standard ResNet-50 ImageNet pipeline with a cold
   cache, X% of GPU idle is block-device I/O and Y% is CPU decode, so at most X%
   of the utilisation gap is a storage problem.

Report the field number with its confidence interval and cache state attached,
every single time it is quoted. It is the sentence that gets pasted into a Slack
channel, and it will be quoted without its caveats unless the caveats are in the
sentence.

## Release the labelled traces

```bash
whystall pack runs/ --out whystall-stalls-v1/ --license CC-BY-4.0
```

Every injected run plus its ground-truth annotations. It lets anyone else's
classifier be scored against yours on identical inputs, which is the fastest
route from "a repo" to "the benchmark everyone reports on".

Three things the packer does that are worth understanding before you publish:

- **It splits by run, not by gap**, grouping on class and dealing magnitudes
  alternately, so both halves see every class. Calibrate thresholds on the
  calibration half and report on test. With one repeat per dial setting there is
  no other honest way to get a calibration set at all.
- **It ships this tool's own baseline** on the test split, so a competing
  classifier has a number to beat rather than having to rerun your sweep.
- **It stamps provenance per run** and refuses to describe a package mixing
  synthetic and hardware traces as a measurement. Check the card says what you
  expect before releasing.

Decide the licence deliberately: the traces may carry model and dataset shapes,
hostnames, and file paths from wherever they were collected. Skim one before
publishing.
