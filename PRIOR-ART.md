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
- **kernel wait** — the short overhead between consecutive kernel launches,
  where "short" means below `consecutive_kernel_delay`, **default 30 ns**;
- **other wait** — the residual, and only the residual: the default label kept by
  any gap that matched neither test above. Cross-stream synchronisation via CUDA
  events is what tends to land there; it is not what defines it.

It ships in a Meta-maintained repository, is documented in the official PyTorch
tutorials, and is the tool an engineer at a storage or networking vendor is most
likely to already have run.

That 30 ns default is worth pausing on, because it decides how much room this
project actually has. A threshold that tight means kernel wait captures kernels
which are effectively back to back and essentially nothing else. **Every gap a
human would call a stall lands in host wait or other wait.** The subdivision
opportunity is therefore larger than a three-way split suggests: it is almost
the whole of the interesting idle time.

Two things HTA does not do, and both are the opening:

1. **Host wait is a single bucket.** Knowing the host was late tells an operator
   nothing about whether to buy storage, buy CPU cores, or fix their own training
   loop. Every commercially interesting question lives inside that one category.
2. **The attribution is asserted, not measured.** HTA publishes an algorithm. It
   does not publish a recall number, a confusion matrix, a detection floor, or
   any evaluation against known ground truth. There is no way to tell how often
   it is right.

**Repository status, checked 2026-09-14.** `facebookresearch/HolisticTraceAnalysis`
is alive and actively developed: 555 stars, 98 forks, commits through
2026-09-11, and pull requests numbered into the 360s. Commits are co-authored by
`meta-codesync[bot]`, so the GitHub repo is synced with an internal one — but PRs
are the merge vehicle, not a formality. `CONTRIBUTING.md` asks contributors to
**open an issue and discuss a feature before building it**, which is both the
documented process and, conveniently, the plan's own first move.

The **TraceInsight** branding appears only on the rendered documentation, not on
the repository, which is not archived and carries no rename notice. Treat it as a
docs-side label, not a move.

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

**Cross-check against HTA rather than ignoring it.** Read against HTA's actual
source — `_analyze_idle_time_for_stream` in `hta/analyzers/breakdown_analysis.py`,
checked 2026-10-14 — the relation is not a loose overlap. It is an exact set
identity, and that is far better, because an identity is falsifiable to the
nanosecond.

HTA assigns each gap in three steps: every gap starts as `OTHER`; it becomes
`HOST_WAIT` when `ts_runtime > prev_end_ts`; it becomes `KERNEL_WAIT` when it is
*not* host wait **and** the gap is shorter than `consecutive_kernel_delay`.
Whatever matches neither test keeps the default. So:

| HTA category | condition |
|---|---|
| `HOST_WAIT` | `ts_runtime > prev_end_ts` |
| `KERNEL_WAIT` | `ts_runtime <= prev_end_ts` ∧ gap < 30 ns |
| `OTHER` | `ts_runtime <= prev_end_ts` ∧ gap ≥ 30 ns |

`ts_runtime <= prev_end_ts` is the same predicate as this project's
`queued_before_gap` (`launch.ts_ns <= gap.start_ns`, `trace/blame.py`): the launch
was already enqueued when the device went idle. Same evidence, same test, two
names. And `rule_device_dependency` is rule **one**, so it consumes every gap
that satisfies it. Therefore, over the same gaps:

- **`device_dependency` ≡ `KERNEL_WAIT` ∪ `OTHER`**, exactly. The 30 ns threshold
  only decides which of HTA's two names a gap gets; it does not change the union.
- **HTA's `host_wait` ≡ everything else this tool emits** — the dataloader split,
  `h2d`, `collective`, `host_sync`, `launch_dispatch`, `other_host`, and the
  unattributed remainder.

The second line is the thesis stated as an equation. HTA's one undifferentiated
bucket is precisely the set this tool subdivides into six, which is a cleaner
claim than "host wait is worth splitting" and can be checked rather than argued.

**A correction, in the opposite direction from the last one.** An earlier draft
called `launch_dispatch` a strict *superset* of HTA's kernel wait. It is not:
they are **disjoint**. Kernel wait requires `~is_host_wait`, and any gap meeting
that condition is taken by `rule_device_dependency` before `rule_launch_dispatch`
is ever reached. A gap HTA calls kernel wait cannot be classified
`launch_dispatch` by this tool, at any threshold. The 200 µs–versus–30 ns
comparison was comparing two rules that never see the same gap.

**What genuinely is not comparable is the gap set, not the rules.** HTA computes
idle *per stream* and sums; this tool treats a device as busy when *any* stream is
busy (`trace/timeline.py`). On a multi-stream trace the two do not share a
denominator and HTA's total idle will be the larger, because per-stream summing
counts a quiet compute stream as idle while the NCCL stream runs. That is a real
methodological disagreement and worth stating as one.

It gives the cross-check a sharp form: **on a single-stream trace, scored with
`min_gap_us=0`, the identities above must hold exactly.** Any discrepancy is a
bug in one of the two tools. On multi-stream traces, report both totals and the
stream-merging difference that explains the gap between them.

## Claims still to verify against full text

Resolved 2026-09-14: the kernel-wait boundary (`consecutive_kernel_delay`,
default 30 ns), the repository's status and contribution process, and the
TraceInsight question. All three are folded into the sections above.

Resolved 2026-10-14, against the source rather than the docs: HTA's *other wait*
is a **residual, not a positive classification** — the default label for any gap
that is neither host wait nor kernel wait. The documentation describes it as
"synchronization among CUDA streams using CUDA events and delays in launching
kernels", which is an interpretation of what tends to land there, not the rule
that puts it there. Reading the rule is what turned a loose comparison into the
exact identity recorded above, and what exposed the backwards `launch_dispatch`
claim. The docs would not have done it; they were in fact what the earlier wrong
claims were built on.

Still open:

- Whether any of the DALI or MosaicML `streaming` benchmark suites report a
  stall-class breakdown that has been missed here.
