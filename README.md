# GPU Stall Attribution

**Per-gap attribution of GPU idle time in PyTorch training — and a harness
that measures how often the attribution is right.**

GPUs in training jobs sit idle. Existing tools report *how much* idle time
there is, and at best whether the host was late. They do not say **why**,
and none of them publish how often their answer is correct.

This tool takes a PyTorch profiler (Kineto) trace, walks CUDA/CUPTI
correlation IDs from each idle window back to the host code that caused it,
and assigns one of nine causes: dataloader I/O, CPU decode, host-to-device
copy, NCCL collective wait, host synchronisation, kernel launch overhead,
other host compute, device-side dependency, or an explicit unattributed
residual.

It then **validates itself**: each cause is deliberately injected at a swept
magnitude, and the recovery is scored as a per-class confusion matrix, a
detection floor per class, and an honest unattributed share.

Installs as `whystall`. The core has **zero dependencies**.

```
Python 3.10+  ·  214 tests  ·  88% coverage  ·  ruff clean  ·  BSD-3-Clause
```

## Quick start — no GPU required

```bash
pip install -e ".[dev]"
whystall demo --out demo/
```

Builds synthetic training traces with a known stall injected into each, runs
the full pipeline — gap extraction, blame resolution, classification, the
ground-truth firewall, confusion matrix, detection floors — and writes a
standalone HTML report.

## Analyse a real trace

```bash
whystall analyze trace.json --out report.html
```

Add the `/proc` sidecar log to split dataloader waits into I/O versus CPU:

```bash
whystall analyze trace.json --samples run.samples.jsonl --out report.html
```

## How a gap gets a name

1. Take the device op that ends the idle window.
2. Follow its CUPTI correlation ID back to the runtime call that launched it.
3. If that launch happened **before** the window opened, the work was already
   queued and the GPU was waiting on the device, not the host. Stop — this is
   not starvation, and counting it as such inflates every host-wait number.
4. Otherwise look at what occupied the launching thread between the device
   going idle and finally being fed. That is the blame span, and its identity
   decides the verdict.

Rules are ordered by **strength of evidence**, not by class, so a 300 µs
window in which the host spent 280 µs inside the dataloader is not charged to
the memcpy that happened to close it. The ordering in
[`classify.py`](src/whystall/trace/classify.py) is the actual specification.

## Validation — the part that is the contribution

Anyone can write rules that name a stall. Almost nobody publishes where their
rules fail. Every class in the taxonomy has a matching fault injector, and
injected runs **self-label inside the trace**, which makes per-gap ground
truth possible:

| Output | What it answers |
|---|---|
| Confusion matrix | Per-class recall, precision, and the confused pairs |
| Detection floor | The smallest stall each class is reliably found at |
| Unattributed share | The fraction of idle the tool cannot explain |
| Control vs treatment | Injection effect, free of labelling bias |

```bash
whystall validate runs/ --out validation.html   # score a magnitude sweep
whystall pack runs/ --out dataset/              # release the labelled traces
```

### The firewall

The classifier must never see the ground-truth annotations, or the matrix
proves nothing. The defence is **structural**: `load_trace` strips every
`WHYSTALL/` annotation before a `TraceView` exists and returns them
separately, so the classifier cannot read one even by accident. Asserted on
every scoring path, not only in tests.

```bash
whystall selftest
```

## Architecture

|  | `whystall.trace` | `whystall.enrich` |
|---|---|---|
| Input | Kineto trace | `/proc` sampling at 200–1000 Hz |
| Dependencies | none | none |
| Decides | 7 classes + residual | dataloader I/O vs CPU |
| Portable upstream | yes | no |

A trace cannot tell you whether workers behind a blocked iterator were on the
disk or on the CPU — both look identical from the launching thread. So the
trace-only classifier emits `dataloader` and stops; it is *structurally
incapable* of emitting `dataloader_io`, and the test suite asserts it. The
split needs worker process state, which only the sidecar sees.

That boundary is not tidiness. Trace-consuming tools upstream have no sidecar
and will not take one, so the trace-only half is the part that can be
contributed and the enrichment layer is what this tool adds on top.

Time is **integer nanoseconds** throughout. Integer microseconds dropped
sub-microsecond kernels entirely, fabricating idle on exactly the
dispatch-bound traces this targets; float microseconds cannot work either,
because Kineto timestamps are epoch-scale where float64 spacing already
exceeds a nanosecond.

## Tech stack

- **Languages** — Python 3.10+
- **GPU / systems** — CUDA, CUPTI correlation IDs, NCCL collectives,
  PyTorch profiler (Kineto), NVML, Linux `/proc` and PSI telemetry
- **ML** — PyTorch, `DataLoader`, distributed training, DALI-style pipelines
- **Methods** — fault injection, confusion matrices, detection floors,
  paired control/treatment design, reproducible dataset packaging
- **Engineering** — pytest, ruff, GitHub Actions CI, zero-dependency core,
  self-contained HTML reporting

## Project status

**No hardware results yet.** Everything here has been exercised on synthetic
traces only, and `whystall demo` says so in its own output. The pipeline,
the validation harness and the dataset packaging are complete and tested;
what remains is renting a two-GPU box and running the sweep in
[`docs/sweep.md`](docs/sweep.md). Real numbers will land in
[`RESULTS.md`](RESULTS.md) with the machine, page-cache state and thresholds
attached.

This is deliberate. Publishing an unvalidated number is the thing the project
exists to argue against.

## Known limits

- **Precision is a lower bound**, and "unlabelled, named" is an upper bound
  on error rather than a false-positive rate. Real runs stall for reasons the
  injection did not create, and a correct verdict on one of those is
  indistinguishable from a mistake. Read the control-versus-treatment shift.
- **Prior art exists and is close.** Holistic Trace Analysis already splits
  GPU idle three ways. This does not replace it — it subdivides the one
  bucket that matters, and unlike anything released, reports how well.
  See [`PRIOR-ART.md`](PRIOR-ART.md).
- **Multi-rank collectives are single-rank scored.** The straggler-versus-
  fabric sub-label needs per-rank timestamps joined across traces.
- **`read_bytes` needs permission.** Without a matching uid or
  `CAP_SYS_PTRACE`, the I/O case rests on process state alone.
- **The portable I/O injector sleeps** rather than throttling the device.
  The honest instrument is `dm-delay` or cgroup v2 `io.max`; see
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

## Licence

BSD-3-Clause.
