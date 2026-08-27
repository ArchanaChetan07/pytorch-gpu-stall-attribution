# The stall taxonomy

**Frozen 2026-08-24, before any measurement existed.** This file is the record
that it was fixed in advance. A taxonomy reverse-engineered from what a
classifier happens to detect is not a taxonomy, and a reviewer is entitled to
ask which came first.

The machine-readable form is `whystall/taxonomy.py`; this is the reasoning.

## The rules the table has to satisfy

1. **Mechanically decidable.** Every class must be reachable from evidence the
   tool actually collects — a trace, plus `/proc` sampling. No class may depend
   on the analyst's judgement.
2. **Injectable, or excluded.** A class that cannot be deliberately produced
   cannot be validated, and an unvalidatable class does not belong in the table.
   Only the two residuals are exempt, and they are residuals precisely because
   they represent the absence of a finding.
3. **Mutually exclusive, with a stated priority.** Real gaps have more than one
   contributing cause. The classifier assigns exactly one class and records the
   evidence, and the rule ordering is published so a verdict can be argued with.

## The nine classes

| Class | Fires when | Injector | Needs sampler |
|---|---|---|---|
| `dataloader` | The launching thread was inside the dataloader iterator. The most specific verdict available from a trace alone. | — | no |
| `dataloader_io` | Same, and the workers were fetching from a block device. | `dataloader_io` | **yes** |
| `dataloader_cpu` | Same, and the workers were saturating cores on decode or augmentation. | `dataloader_cpu` | **yes** |
| `dataloader_mixed` | Both signals above threshold. Reported rather than resolved. | — | **yes** |
| `h2d_copy` | The gap ends in a host-to-device transfer, or a copy frame dominated the window. Pinned versus pageable recorded. | `h2d` | no |
| `collective` | The gap ends in an NCCL kernel, or a collective frame dominated. | `collective` | no |
| `host_sync` | A blocking synchronise or scalar readback drained the queue. | `host_sync` | no |
| `launch_dispatch` | A short gap that no vocabulary explains; the host was churning through operator dispatch. Corresponds to HTA's *kernel wait*. | `launch` | no |
| `other_host` | Resolvable host work matching no other rule. Always carries its top operator names. | `other_host` | no |
| `device_dependency` | The next kernel was launched **before** the window opened. The GPU waited on the device, not the host. | `device_dependency` | no |
| `unattributed` | No blame span resolved. | — | no |

## Why the dataloader split is separate

`dataloader_io` and `dataloader_cpu` present **identically** in a trace: the
launching thread is blocked inside `enumerate(DataLoader)#...__next__` either
way, and the trace has no visibility into the worker processes behind it.

So the trace-only classifier emits `dataloader` and stops. It is structurally
incapable of emitting the refinements — `whystall.trace.classify` raises if a
rule tries, and `tests/test_classify.py` asserts it across every scenario.

This is the taxonomy encoding an architectural constraint. The trace-only half
is the piece that could be contributed to a trace-consuming tool upstream; the
sampler enrichment is what this tool adds. Collapsing the two would make the
first half unmergeable.

## Why `device_dependency` is not host wait

It is the class most often silently mishandled, and getting it wrong inflates
every other number.

When a kernel is enqueued *before* the device goes idle and still cannot run —
because it waits on an event from another stream — the host did everything right.
Counting that window as starvation manufactures a host-side problem out of a
device-side dependency. The classifier checks this first, before any host-wait
rule runs, and the check is the first item in the rule ordering for that reason.

## Why `unattributed` is a first-class row

Because the alternative is to fold it into `other_host` and report a tool that
explains everything.

Its size on unmodified workloads is one of the three published method numbers.
Every report prints it, `DeviceResult.unattributed_share` computes it, and the
HTML report flags it above 25%.

## Scoring granularity

Because the trace-only classifier cannot reach the refinements, scoring it
strictly against `dataloader_io` ground truth would report recall 0 for a
classifier that was right about everything it is able to say.

Two granularities, both meaningful:

- **coarse** — collapse `dataloader_io`, `dataloader_cpu` and `dataloader_mixed`
  to `dataloader` before comparing. The fair way to score the trace-only half.
- **strict** — compare as-is. The way to score the sampler-enriched half.

Publish both. The gap between them is the measured value of the sidecar, which
is a more useful thing to report than either number alone.

## Changes since freezing

- **2026-08-24** — `device_dependency` moved from non-injectable to injectable.
  Rule 2 says an unvalidatable class does not belong in the table, and it turns
  out this one is injectable after all: record an event on a second stream, make
  the compute stream wait on it, delay the event. The class was kept and an
  injector written, rather than the rule being bent.
