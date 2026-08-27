# Prior art

Written 2026-08-24, before any code, and updated as the survey continued. It
exists because the obvious framing for this project — "nobody attributes GPU idle
time" — is false, and shipping under it would have been dismissed in the first
reply.

## What already exists

### Holistic Trace Analysis (Meta)

The closest prior work by a wide margin, and the one this project extends rather
than replaces.

HTA consumes PyTorch profiler (Kineto) traces and provides an **idle time
breakdown** that splits GPU idle into three categories:

- **host wait** — the GPU is idle because the CPU is not enqueueing kernels fast
  enough;
- **kernel wait** — the short overhead between consecutive kernel launches;
- **other wait** — idle that could not be attributed, typically cross-stream
  synchronisation via CUDA events.

It ships in a Meta-maintained repository, is documented in the official PyTorch
tutorials, and is the tool an engineer at a storage or networking vendor is most
likely to already have run.

Two things it does not do, and both are the opening:

1. **Host wait is a single bucket.** Knowing the host was late tells an operator
   nothing about whether to buy storage, buy CPU cores, or fix their own training
   loop. Every commercially interesting question lives inside that one category.
2. **The attribution is asserted, not measured.** HTA publishes an algorithm. It
   does not publish a recall number, a confusion matrix, a detection floor, or
   any evaluation against known ground truth. There is no way to tell how often
   it is right.

The documentation now carries **TraceInsight** branding, which suggests a rename
or a move. Confirm the canonical repository and its external-PR cadence before
treating it as an upstream target.

### `torch_tb_profiler` (PyTorch TensorBoard plugin)

Goes further than HTA in one direction: its step-time breakdown does carry a
**DataLoader** category alongside Kernel, Memcpy, Memset, Communication, Runtime
and CPU Exec.

Three differences that matter:

- It categorises **CPU-side step time**, not individual GPU idle windows. It
  answers "where did the step go", not "why was the device idle from t0 to t1".
- Attribution is by **operator-name matching**, without following the launch
  correlation back from the kernel that ended the gap — so it cannot distinguish
  a dataloader wait that actually delayed the GPU from one the prefetch absorbed.
- The plugin is effectively unmaintained.

### Nsight Systems

Sees strictly more than Kineto: OS runtime tracing captures blocking syscalls
(`read`, `pread64`, `futex`), which is exactly the evidence a Kineto trace lacks.
It presents them on a timeline for a human to interpret and performs no automatic
per-gap attribution.

Relevant as the fallback substrate if blame resolution from Kineto proves too
weak — that is the documented reshaping path behind Gate 1 in the project plan.

### DALI, `torchdata`, and the dataloading literature

These address the *cause* — faster decode, better prefetch, GPU-side
augmentation — and are generally evaluated by end-to-end throughput. None
provides per-gap causal attribution, and none publishes an attribution accuracy
number, because attribution is not what they are for.

### Vendor material

Storage and interconnect vendors publish GPU-utilisation improvements routinely.
The measurements are end-to-end throughput or aggregate utilisation before and
after. The causal chain from an idle GPU to a specific stall class is asserted,
never demonstrated, and never independently reproducible.

### Fault injection with ground truth, in adjacent fields

Controlled fault injection to produce labelled ground truth for an attribution
system is well established elsewhere — distributed-systems failure diagnosis,
and recently agent-trajectory failure attribution, which reports a confusion
matrix over injected error types. The method is standard. Nobody appears to have
applied it to GPU stall attribution.

## The delta

1. **Subdivide host wait** into dataloader I/O, dataloader CPU, host-to-device
   copy, collective wait, host sync, dispatch overhead, and other host compute —
   the split that determines what an operator should actually buy.
2. **Measure the attribution.** Inject each class deliberately, sweep its
   magnitude, and publish per-class recall, the pairs the classifier confuses,
   the detection floor per class, and the share of idle it cannot explain.
3. **Release the labelled traces** so a competing classifier can be scored on
   identical inputs.

Item 2 is the contribution. Items 1 and 3 are what make it usable.

## What this means for the design

Two consequences, both load-bearing, both visible in the code:

**The trace-only classifier must stay dependency-free and sampler-unaware.** A
trace-consuming tool upstream has no sidecar and will not accept one, so the
half that could be contributed has to be separable from the half that cannot.
That is why `whystall.trace` decides seven classes and stops, and why the split
between dataloader I/O and dataloader CPU lives in `whystall.enrich`. The
boundary was drawn before the classifier was written; retrofitting it would have
meant a rewrite.

**Cross-check against HTA rather than ignoring it.** The `launch_dispatch` class
is deliberately defined to correspond to HTA's *kernel wait*, and
`device_dependency` to a large part of its *other wait*. Running both on the same
trace and reporting the agreement is a validation result; reporting a
disagreement is a more interesting one. Either way it demonstrates the prior tool
was understood rather than overlooked, which is the difference between an
extension and a reinvention.

## Claims still to verify against full text

- The exact current definition of HTA's *kernel wait* boundary, so the
  `launch_dispatch` correspondence is stated precisely rather than approximately.
- Whether the TraceInsight rename changes the canonical repository, and what the
  recent external-contribution cadence looks like there.
- Whether any of the DALI or MosaicML `streaming` benchmark suites report a
  stall-class breakdown that has been missed here.
