# Draft: HTA issue

To file at `facebookresearch/HolisticTraceAnalysis/issues`. Their
`CONTRIBUTING.md` asks for an issue before a feature PR, so this is the required
first step, not an optional courtesy.

**Before posting**, do these three things — the whole point of asking first is
that the answer might change what gets built:

1. Search open and closed issues for "idle", "host wait", "dataloader". If this
   has been proposed and declined, the reason matters more than the proposal.
2. Run the containment cross-check on one real trace and put the actual numbers
   in the issue. A proposal with a measurement attached reads differently from
   one without, and it is a day of work you were going to do anyway.
3. Replace the placeholders marked `<>`.

Keep it short. A maintainer deciding whether to engage reads the first
paragraph and the ask.

---

**Title:** Sub-classifying host wait in the idle time breakdown

---

`get_idle_time_breakdown()` splits GPU idle into host wait, kernel wait and
other wait. With `consecutive_kernel_delay` at its 30 ns default, kernel wait
covers kernels that are effectively back to back, so in practice almost all
idle a user cares about lands in **host wait**.

Host wait answers "the CPU was late" but not "late doing what", and the
follow-up question is usually the actionable one: a dataloader blocked on
storage, decode saturating the worker cores, a pageable H2D copy, a stray
`.item()` draining the queue, and operator dispatch overhead all present as host
wait and have completely different fixes.

I have been working on a per-gap classifier for this and would like to know
whether a sub-classification of host wait is something you would take, and in
what shape.

**The approach.** For each idle window, follow the CUPTI correlation id from the
device op that ends it back to the runtime call that launched it. If that launch
preceded the window, the work was already queued and the GPU was waiting on the
device rather than the host — not host wait at all. Otherwise, look at what
occupied the launching thread between the device going idle and being fed, and
classify on that frame. Rules are ordered by strength of evidence, so a window
where the host sat mostly inside the dataloader is not charged to the memcpy
that happened to close it.

Classes reachable from a trace alone: dataloader, host-to-device copy,
collective wait, host sync, launch/dispatch, other host compute, device
dependency, and an explicit unattributed residual.

**What I think is worth discussing before any code:**

- **Scope.** Would you rather see this as extra categories inside
  `get_idle_time_breakdown()`, an opt-in argument to it, or a separate analyzer
  that consumes the same trace? I have no attachment to any of these.
- **Boundaries against existing categories.** Reading
  `_analyze_idle_time_for_stream`, the correspondence looks exact rather than
  approximate. Your `~is_host_wait` (`ts_runtime <= prev_end_ts`) is the same
  predicate I use to decide that the launch was already enqueued when the device
  went idle, and the class I call *device dependency* is the only thing I emit
  for it. If that reading is right, then over the same gaps:

      device dependency  ==  kernel wait  union  other wait
      host wait          ==  every other class I emit

  — with `consecutive_kernel_delay` only deciding which of your two names a gap
  in that union receives, not the size of the union. Which would mean the
  proposal is precisely "subdivide host wait" and touches neither of the other
  two categories. On a real trace `<N>` I measured `<...>`.

  Two things I would want your read on. First, whether *other wait* is intended
  as a residual — it is the default label in the code, while the docs describe it
  as cross-stream synchronisation, and those are different claims. Second,
  idle is computed per stream and summed here, whereas I treat a device as busy
  when any stream is busy; on multi-stream traces that makes the totals differ by
  construction, and I would rather match your convention than argue for mine.
- **Validation.** Each class has a matching fault injector, and injected runs
  self-label inside the trace, which gives a per-gap confusion matrix, a
  detection floor per class, and an honest unattributed share. If that kind of
  evaluation is useful to you beyond this change, the harness generalises.

**One thing that does not fit here.** Separating a dataloader wait caused by
disk from one caused by CPU decode is not decidable from a trace — the launching
thread is blocked inside the iterator either way — and needs worker process
state sampled alongside. That part stays outside HTA; I mention it only so the
boundary is clear. The trace-only classifier is deliberately built to be
dependency-free and unaware of the sampler for exactly this reason.

Happy to put up a small PR against whichever shape you prefer, or to drop it if
this is already in progress internally.

`<link to whystall, or omit on a first post — the question stands without it>`

---

## Notes to self, not for posting

- **Do not lead with the repo.** The question is whether they want the feature.
  A link up front reads as promotion and invites "no thanks" before the idea has
  been considered.
- **The identity is the strongest opening**, stronger than the 30 ns observation
  it replaced. It is derived from their own source, it is checkable in one
  reading, and it reframes the proposal from "your tool is incomplete" to "your
  host wait is exactly the set I subdivide, and I touch nothing else." It also
  demonstrates the code was read rather than the docs skimmed, which is the
  difference between a contributor and a drive-by.
- **Lead with the question, not the certainty.** The identity is stated as a
  reading to be confirmed, because it is one — and because if it is wrong, being
  told so in the first reply is the cheapest possible outcome.
- **Offer to align vocabulary.** The likeliest objection is a second set of
  category names diverging from theirs. Conceding that in advance costs nothing.
- **Expect no answer.** Meta repos sync from an internal one and issues can sit.
  Two weeks of silence is not a no; it means build the standalone tool and the
  dataset, which are the parts that do not depend on anyone else. Re-ping once,
  politely, then move to the DALI fallback.
- **If they say yes**, the trace-only module is already the separable half.
  Nothing in `whystall.trace` imports the sampler, and the test suite asserts it.
