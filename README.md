# whystall

Attribute each GPU idle window to the thing that caused it — dataloader I/O,
CPU decode, host-to-device copy, collective wait, host sync, dispatch overhead,
or a device-side dependency — and **measure how often that attribution is right**
by injecting each cause deliberately and scoring the recovery.

## Read this before the numbers

- **The classifier is not the contribution. The validation harness is.** Anyone
  can write rules that name a stall. Almost nobody publishes a confusion matrix
  showing where their rules fail, and nothing in this space has ever published
  a detection floor. That absence is why this exists.
- **Prior art exists and is close.** Holistic Trace Analysis already splits GPU
  idle into *host wait*, *kernel wait* and *other wait*, and ships in a
  Meta-maintained repository. `whystall` does not replace it. It subdivides the
  one bucket that matters commercially — host wait — and, unlike anything
  released, it reports how well it does so. See [`PRIOR-ART.md`](PRIOR-ART.md).
- **Precision in the reports is a lower bound**, and the "unlabelled, named"
  figure is an upper bound on error rather than a false-positive rate. Real runs
  stall for reasons the injection did not create, and a correct verdict on one
  of those is indistinguishable from a mistake when nothing labelled it. Read
  the control-versus-treatment shift instead; every report prints it.
- **No hardware results yet.** Everything in this repository has been exercised
  on synthetic traces only. `whystall demo` says so in its own output. When real
  numbers land they will be in [`RESULTS.md`](RESULTS.md) with the machine, the
  page-cache state, and the sweep plan attached.

## Install

```bash
pip install -e ".[dev]"
```

The core has **no dependencies**. `torch` is needed only to run the injectors,
and only on the machine with the GPU.

## Try it without a GPU

```bash
whystall demo --out demo/
```

Builds a sweep of synthetic training traces with a known stall injected into
each, runs the whole pipeline — gap extraction, blame resolution,
classification, the ground-truth firewall, the confusion matrix, the detection
floors — and writes a standalone HTML report. This is also the fastest way to
see the shape of the output before spending anything on a rental.

## Analyse a real trace

```bash
whystall analyze trace.json --out report.html
```

To split dataloader waits into I/O and CPU, the sidecar sampler has to have been
running alongside training:

```bash
whystall analyze trace.json --samples run.samples.jsonl --out report.html
```

## Collect a run

```python
from torch.utils.data import DataLoader
from whystall.inject import DataloaderIO, Harness, HarnessConfig

harness = Harness(DataloaderIO(magnitude_us=4000),
                  HarnessConfig(out_dir="runs", cache_state="cold"))

with harness:
    ds = harness.wrap_dataset(dataset)
    loader = DataLoader(ds, **harness.loader_kwargs(batch_size=64, num_workers=8))
    it = iter(loader)
    for _ in range(48):
        with harness.step():
            with harness.fetch():
                batch = next(it)
            with harness.transfer():
                batch = batch.to("cuda", non_blocking=True)
            with harness.compute():
                train_one_step(batch)
            with harness.step_end():
                pass
```

That leaves a trace, a sampler log and a manifest in `runs/`. Sweep the
magnitude, add a control run with `NullInjector`, then:

```bash
whystall validate runs/ --out validation.html
```

See [`examples/train_reference.py`](examples/train_reference.py) for a complete
script and [`docs/sweep.md`](docs/sweep.md) for the experiment plan to write
*before* renting anything.

## How a gap gets a name

1. Take the device op that ends the idle window.
2. Follow its CUPTI correlation id back to the runtime call that launched it.
3. If that launch happened **before** the window opened, the work was already
   queued and the GPU was waiting on the device, not the host. Stop. This is not
   starvation, and counting it as such inflates every host-wait number.
4. Otherwise look at what occupied the launching thread between the device going
   idle and finally being fed. That is the blame span, and its identity decides.

Rules are ordered by strength of evidence rather than by class, so a 300 µs
window in which the host spent 280 µs inside the dataloader is not charged to
the memcpy that happened to close it. The full ordering is the docstring of
[`whystall/trace/classify.py`](src/whystall/trace/classify.py) and it is the
classifier's actual specification.

## The two halves, and why they are separate

| | `whystall.trace` | `whystall.enrich` |
|---|---|---|
| Input | a Kineto trace | `/proc` sampling at ~200 Hz |
| Dependencies | none | none |
| Decides | 7 classes plus the residual | dataloader I/O vs CPU |
| Portable upstream | yes | no |

A trace cannot tell you whether the workers behind a blocked iterator were on
the disk or on the CPU: both look identical from the launching thread. So the
trace-only classifier emits `dataloader` and stops — it is structurally
incapable of emitting `dataloader_io`, and the test suite asserts it. The split
needs worker process state, which only the sidecar sees.

That boundary is not tidiness. Trace-consuming tools upstream have no sidecar
and will not take one, so the trace-only half is the part that can be
contributed and the enrichment layer is what this tool adds on top.

## The firewall

The classifier must never see the ground-truth annotations, or the confusion
matrix proves nothing. The defence here is structural: `load_trace` strips every
`WHYSTALL/` annotation before a `TraceView` exists and returns them separately,
so the classifier cannot read one even by accident.
`whystall.validate.firewall` asserts the structure held, and it runs on every
scoring path, not only in tests.

```bash
whystall selftest
```

If you take one idea from this repository into your own harness, take that one.

## The taxonomy

Nine classes, frozen before any measurement existed — see
[`TAXONOMY.md`](TAXONOMY.md). Every class that can be injected has an injector,
because a class that cannot be injected cannot be validated, and an
unvalidatable class does not belong in the table.

## Known limits

- **Multi-rank collectives are single-rank scored.** The straggler-versus-fabric
  sub-label needs per-rank entry timestamps joined across traces; the class is
  detected, the sub-label is not yet computed.
- **`read_bytes` needs permission.** Without a matching uid or `CAP_SYS_PTRACE`,
  the I/O case rests on process state alone, which aliases. The reports say so
  when it happens rather than quietly downgrading.
- **The portable I/O injector sleeps.** The honest instrument is `dm-delay` or
  cgroup v2 `io.max`, both of which need privileges most rented containers
  withhold. Where you have them even once, run the sweep both ways and publish
  the comparison — see [`docs/injection.md`](docs/injection.md).
- **Gaps below 20 µs are aggregated, not classified.** At that scale the cause
  is launch overhead; the total is reported separately so it is not lost, as is
  trailing idle after the last device op, which is teardown rather than a stall.
- **Refinement needs samples in the window.** A verdict needs `min_samples` (3)
  observations inside the blame span, so at the default 200 Hz nothing shorter
  than ~15 ms can be split. Run the sampler at 500–1000 Hz when short stalls
  matter; the reports say which windows were skipped and what rate would have
  resolved them.
- **Python function events are used when present but not required.** Traces
  collected without `with_stack` attribute slightly more coarsely.
- **Timestamps are whole microseconds, and traces are not.** Chrome-format `ts`
  and `dur` are floats in microseconds and Kineto emits fractional values, so
  durations are rounded on load — never down to zero, because dropping a real
  kernel from the busy timeline fabricates idle out of work. Busy time is
  therefore overstated by under a microsecond per rounded event and idle
  understated by the same, which matters only for gaps near the 20 µs
  classification floor. Every report that loads such a trace says so and gives
  the count. **The proper fix is integer nanoseconds throughout**; float
  microseconds is not an option, because Kineto timestamps are epoch-scale
  (~1.7 × 10¹⁵ µs) where float64 spacing is already coarser than a nanosecond.
  That is a unit change across ~300 sites and is deliberately not yet done.

## Licence

BSD-3-Clause.
