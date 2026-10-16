# GPU Stall Attribution

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![Tests](https://img.shields.io/badge/tests-230%20passing-brightgreen) ![Coverage](https://img.shields.io/badge/coverage-88%25-green) ![Lint](https://img.shields.io/badge/ruff-clean-success) ![License](https://img.shields.io/badge/license-BSD--3--Clause-lightgrey)

> **Training GPUs sit idle, and nobody can say why. This tool says why —
> and then proves how often it is right.**

Idle GPUs are the most expensive problem in machine learning infrastructure.
Existing profilers report *how much* idle time a training job has, and at best
whether the CPU was late. They do not say **which** part of the pipeline caused
it — and none of them publish how often their answer is correct.

This tool reads a PyTorch profiler (Kineto) trace, follows CUDA/CUPTI
correlation IDs from each idle window back to the host code responsible, and
assigns one of nine causes: dataloader I/O, CPU decode, host-to-device copy,
NCCL collective wait, host synchronisation, kernel launch overhead, other host
compute, device-side dependency, or an explicit unattributed residual.

Then it **validates itself.** Every cause has a matching fault injector, runs
are swept across magnitudes, and the recovery is scored as a confusion matrix,
a per-class detection floor, and an honest unattributed share.

## Example output

```
truth \ predicted      dataload  h2d  collect  h.sync  launch  other  dev.dep
dataloader                   31    .        .       .       .      .        .
h2d_copy                      .   31        .       .       .      .        .
collective                    .    .       28       .       .      .        .
host_sync                     .    .        .      65       .      .        .
launch_dispatch               1   13        .       .    1204      .        .
other_host                    .    .        .       .       .     26        .
device_dependency             .    .        .       .       .      .       29
(not injected)               23    .        .       .       .      .       41

accuracy 0.990      macro recall 0.998      unattributed 0.0%
```

*Synthetic traces — this demonstrates the pipeline, not hardware. See
[Project status](#project-status).*

## What this project demonstrates

- **GPU performance engineering** — CUDA/CUPTI correlation IDs, stream-level
  timeline reconstruction, NCCL collective detection, PCIe transfer analysis,
  pinned vs pageable memory
- **Systems programming** — Linux `/proc` and PSI telemetry, process-state
  sampling at up to 1 kHz, nanosecond clock alignment across two independent
  time sources
- **Experimental design** — fault injection, confusion matrices, detection
  floors, paired control/treatment comparison, a taxonomy frozen before any
  measurement was taken
- **Production Python** — 230 tests, 88% coverage, zero-dependency core,
  slotted dataclasses, CI across six interpreter and OS combinations
- **Engineering judgement** — the reasoning behind every design decision, and
  every correction to an earlier mistake, is recorded in the source

## Quick start — no GPU required

```bash
pip install -e ".[dev]"
whystall demo --out demo/
```

Builds synthetic training traces with a known stall injected into each, runs
the full pipeline, and writes a standalone HTML report.

```bash
whystall analyze trace.json --out report.html      # a real trace
whystall validate runs/ --out validation.html      # score a sweep
whystall pack runs/ --out dataset/                 # release the traces
whystall selftest                                  # check the firewall
```

## How a gap gets a name

1. Take the device op that ends the idle window.
2. Follow its CUPTI correlation ID back to the runtime call that launched it.
3. If that launch happened **before** the window opened, the work was already
   queued and the GPU was waiting on the device, not the host. Stop — this is
   not starvation, and counting it as such inflates every host-wait number.
4. Otherwise look at what occupied the launching thread between the device
   going idle and finally being fed. That is the blame span, and it decides.

Rules are ordered by **strength of evidence**, not by class, so a 300 µs
window where the host spent 280 µs inside the dataloader is not charged to the
memcpy that happened to close it. The ordering in
[`classify.py`](src/whystall/trace/classify.py) is the specification.

## Why the validation is the contribution

Anyone can write rules that name a stall. Almost nobody publishes where their
rules fail. Injected runs **self-label inside the trace**, which makes per-gap
ground truth possible:

| Output | What it answers |
|---|---|
| Confusion matrix | Per-class recall, precision, the confused pairs |
| Detection floor | Smallest stall each class is reliably found at |
| Unattributed share | The fraction of idle the tool cannot explain |
| Control vs treatment | Injection effect, free of labelling bias |

### The firewall

The classifier must never see the ground-truth annotations, or the matrix
proves nothing. The defence is **structural**: `load_trace` strips every
`WHYSTALL/` annotation before a `TraceView` exists, so the classifier cannot
read one even by accident. Asserted on every scoring path, not only in tests.

## Architecture

|  | `whystall.trace` | `whystall.enrich` |
|---|---|---|
| Input | Kineto trace | `/proc` sampling, 200–1000 Hz |
| Dependencies | none | none |
| Decides | 7 classes + residual | dataloader I/O vs CPU |
| Portable upstream | yes | no |

A trace cannot tell you whether workers behind a blocked iterator were on the
disk or on the CPU — both look identical from the launching thread. So the
trace-only classifier emits `dataloader` and stops; it is *structurally
incapable* of emitting `dataloader_io`, and the test suite asserts it.

That boundary is not tidiness. Trace-consuming tools upstream have no sidecar
and will not take one, so the trace-only half is the part that can be
contributed, and the enrichment layer is what this adds on top.

Time is **integer nanoseconds** throughout. Integer microseconds dropped
sub-microsecond kernels entirely, fabricating idle on exactly the
dispatch-bound traces this targets; float microseconds cannot work either,
because Kineto timestamps are epoch-scale where float64 spacing already
exceeds a nanosecond.

## Tech stack

**Languages** Python 3.10+

**GPU & systems** CUDA · CUPTI correlation IDs · NCCL · PyTorch profiler
(Kineto) · NVML · Linux `/proc`, PSI · cgroup v2 · `dm-delay`

**ML** PyTorch · `DataLoader` · distributed training · DALI-style pipelines

**Methods** fault injection · confusion matrices · detection floors ·
paired control/treatment design · reproducible dataset packaging

**Engineering** pytest · ruff · GitHub Actions · zero-dependency core ·
self-contained HTML reporting

## Project status

**No hardware results yet.** Everything here has run on synthetic traces only,
and `whystall demo` says so in its own output. The pipeline, the validation
harness and the dataset packaging are complete and tested; what remains is a
two-GPU box and the sweep in [`docs/sweep.md`](docs/sweep.md). Numbers will
land in [`RESULTS.md`](RESULTS.md) with the machine, page-cache state and
thresholds attached.

This is deliberate. Publishing an unvalidated number is the thing the project
exists to argue against.

## Known limits

- **Precision is a lower bound**, and "unlabelled, named" is an upper bound on
  error rather than a false-positive rate. Real runs stall for reasons the
  injection did not create, and a correct verdict on one of those is
  indistinguishable from a mistake. Read the control-versus-treatment shift.
- **Prior art exists and is close.** Holistic Trace Analysis already splits
  GPU idle three ways. This does not replace it — it subdivides the one bucket
  that matters, and reports how well. See [`PRIOR-ART.md`](PRIOR-ART.md).
- **Multi-rank collectives are single-rank scored.** The straggler-versus-
  fabric sub-label needs per-rank timestamps joined across traces.
- **`read_bytes` needs permission.** Without a matching uid or
  `CAP_SYS_PTRACE`, the I/O case rests on process state alone.
- **The portable I/O injector sleeps** rather than throttling the device; the
  honest instrument is `dm-delay` or cgroup v2 `io.max`. See
  [`docs/injection.md`](docs/injection.md).
- **Refinement needs samples in the window.** At 200 Hz nothing shorter than
  ~15 ms can be split; run the sampler at 500–1000 Hz for short stalls.

## Documentation

| File | Contents |
|---|---|
| [`TAXONOMY.md`](TAXONOMY.md) | The nine classes, frozen before measurement |
| [`PRIOR-ART.md`](PRIOR-ART.md) | What exists, and what the delta is |
| [`docs/sweep.md`](docs/sweep.md) | The experiment plan and its three gates |
| [`docs/injection.md`](docs/injection.md) | Injecting stalls honestly |
| [`RESULTS.md`](RESULTS.md) | Results, when hardware runs happen |

## Author

**Archana Suresh Patil** — AI Engineer, LLM and semiconductor automation.
Sunnyvale, CA. Open to roles in GPU performance, ML infrastructure and
inference systems.

[LinkedIn](https://www.linkedin.com/in/archana-suresh-patil-792213245) ·
[GitHub](https://github.com/ArchanaChetan07)

## Licence

BSD-3-Clause.
