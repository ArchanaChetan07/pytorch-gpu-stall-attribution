"""One test per fixed bug. Every case here failed before the fix it names.

Grouped by how the bug would have surfaced, because that is what makes them
worth keeping: most of these were invisible to the rest of the suite and would
have been discovered on the rented box, after the money was spent.
"""

from __future__ import annotations

import pickle
import time

import pytest

from whystall.cli import build_parser, main
from whystall.enrich.sampler import Sample, SamplerLog, WorkerSample
from whystall.inject import DataloaderCPU, DataloaderIO
from whystall.synth import TraceBuilder, build_scenario
from whystall.trace.blame import resolve_blame
from whystall.trace.load import load_trace_dict
from whystall.trace.timeline import find_gaps


def _sample(t_ns, states, read_bytes=0, utime=0):
    return Sample(
        t_mono_ns=t_ns,
        workers=tuple(
            WorkerSample(pid=i, state=s, utime=utime, stime=0, read_bytes=read_bytes, rchar=0)
            for i, s in enumerate(states)
        ),
    )


class TestSamplerLookupCost:
    """Window queries scanned the whole log, three times, per gap.

    A 60 s run at 500 Hz against a dataloader-bound trace is 30k samples and a
    few thousand gaps. The scan turned refinement from milliseconds into minutes
    -- slow enough to look like a hang, fast enough on test-sized inputs that
    nothing caught it.
    """

    def test_window_queries_do_not_scan_the_whole_log(self):
        log = SamplerLog(samples=[_sample(i * 1_000_000, "DD") for i in range(40_000)])
        start = time.perf_counter()
        for i in range(2_000):
            log.stats_ns(i * 1_000_000, (i + 4) * 1_000_000)
        elapsed = time.perf_counter() - start
        # Linear scanning is ~80M sample visits here and takes tens of seconds.
        assert elapsed < 2.0, f"{elapsed:.1f}s for 2000 windows over 40k samples"

    def test_bisect_returns_the_same_windows_as_a_scan(self):
        samples = [_sample(t, "DR") for t in (10, 20, 30, 40, 50)]
        log = SamplerLog(samples=list(samples))
        for lo, hi in ((0, 100), (15, 45), (20, 40), (41, 49), (50, 50)):
            expected = [s for s in samples if lo <= s.t_mono_ns <= hi]
            assert log.between_ns(lo, hi) == expected

    def test_counters_are_still_differenced_across_a_bracketing_pair(self):
        log = SamplerLog(samples=[
            _sample(0, "DD", read_bytes=0),
            _sample(1_000_000_000, "DD", read_bytes=5_000_000),
            _sample(2_000_000_000, "DD", read_bytes=9_000_000),
        ])
        # A window holding only the middle sample must still see the change
        # either side of it, or a short stall reports zero I/O however much the
        # workers actually read.
        stats = log.stats_ns(900_000_000, 1_100_000_000)
        assert stats.samples == 1
        assert stats.read_bytes == 2 * 9_000_000


class TestDatasetWrappingSurvivesMultiprocessing:
    """The wrapper held a closure, which DataLoader cannot pickle under spawn.

    On Linux fork it works, so it would have passed every local check and failed
    at the first fetch on a box configured for spawn or forkserver.
    """

    @pytest.mark.parametrize("injector", [DataloaderIO(1000), DataloaderCPU(1000)])
    def test_wrapped_dataset_pickles(self, injector):
        ds = injector.wrap_dataset(list(range(8)))
        revived = pickle.loads(pickle.dumps(ds))
        assert len(revived) == 8
        assert revived[3] == 3

    def test_unpickling_does_not_recurse_through_getattr(self):
        """__getattr__ ran before _inner existed and recursed without bound."""
        ds = DataloaderCPU(0).wrap_dataset(list(range(4)))
        blob = pickle.dumps(ds)
        revived = pickle.loads(blob)
        assert revived[0] == 0

    def test_missing_attributes_still_raise_attribute_error(self):
        ds = DataloaderCPU(0).wrap_dataset(list(range(4)))
        with pytest.raises(AttributeError):
            _ = ds.no_such_attribute

    def test_delay_sequences_differ_between_worker_processes(self):
        """One seed fixed in the parent gave all eight workers one pattern."""
        hook = DataloaderIO(1000, jitter_frac=0.5).wrap_dataset([0])._hook
        revived = pickle.loads(pickle.dumps(hook))
        assert revived._rng is None, "the RNG must be rebuilt per process"


class TestTrailingIdleIsNotSubThreshold:
    """Idle after the last device op was bucketed as launch overhead.

    On a dispatch-bound run the host outlives the device by a wide margin, so
    this reported seconds of teardown as sub-20-microsecond launch gaps.
    """

    def test_trailing_idle_has_its_own_bucket(self):
        b = TraceBuilder()
        b.device_op("k", 1000, 500, 1, stream=7)
        view, _ = load_trace_dict(b.build())
        gs = find_gaps(view, 0, min_gap_us=20, window=(1000, 9000))
        assert gs.idle_trailing_us == 7500
        assert gs.idle_subthreshold_us == 0
        assert gs.idle_total_us == 7500

    def test_short_gaps_still_land_in_the_sub_threshold_bucket(self):
        b = TraceBuilder()
        b.device_op("a", 0, 100, 1, stream=7)
        b.device_op("b", 105, 100, 2, stream=7)
        view, _ = load_trace_dict(b.build())
        gs = find_gaps(view, 0, min_gap_us=20, window=(0, 205))
        assert gs.idle_subthreshold_us == 5
        assert gs.idle_trailing_us == 0

    def test_the_three_buckets_account_for_all_idle(self):
        view, _ = load_trace_dict(build_scenario("launch", steps=6, stall_us=34))
        gs = find_gaps(view, 0, min_gap_us=20)
        classified = sum(g.dur for g in gs.gaps)
        assert (
            classified + gs.idle_subthreshold_us + gs.idle_trailing_us == gs.idle_total_us
        )


class TestZeroIsAValidThreshold:
    """`value or default` silently replaced --min-gap-us 0 with 20."""

    def test_zero_min_gap_is_honoured(self, tmp_path, capsys):
        trace = tmp_path / "t.json"
        main(["synth", "launch", "--out", str(trace), "--steps", "4", "--stall-us", "34"])
        main(["--min-gap-us", "0", "analyze", str(trace)])
        assert "min_gap_us=0" in capsys.readouterr().out

    def test_omitted_flags_still_take_the_default(self, tmp_path, capsys):
        trace = tmp_path / "t.json"
        main(["synth", "clean", "--out", str(trace), "--steps", "4"])
        main(["analyze", str(trace)])
        assert "min_gap_us=20" in capsys.readouterr().out


class TestEmptyTraceIsAnError:
    """A trace with no device activity produced a confident, empty report."""

    def test_reports_the_likely_cause_instead(self, tmp_path, capsys):
        trace = tmp_path / "cpu-only.json"
        trace.write_text(
            '{"traceEvents": [{"ph": "X", "cat": "cpu_op", "name": "aten::add",'
            ' "pid": 1, "tid": 1, "ts": 0, "dur": 10, "args": {}}]}',
            encoding="utf-8",
        )
        assert main(["analyze", str(trace)]) == 2
        assert "ProfilerActivity.CUDA" in capsys.readouterr().err


class TestBlameNamesExcludeStepsConsistently:
    """top_names dropped depth-0 frames; _select_leaf dropped them by name.

    On a thread with no ProfilerStep annotation the depth rule discarded the
    genuine top-level cause, so 'other host compute' arrived with no operators.
    """

    def test_top_level_causes_survive_when_no_step_annotation_exists(self):
        """Two depth-0 siblings and no enclosing step.

        The old rule dropped every depth-0 frame whenever more than one frame
        covered the window, so on a thread PyTorch never annotated -- a custom
        loop, a worker, a non-standard harness -- 'other host compute' arrived
        with no operators at all, which is the one thing that bucket is for.
        """
        b = TraceBuilder()
        b.device_op("prev", 0, 10, 1, stream=7)
        b.cpu_op("EpochMetrics.update", 10, 1400)
        b.cpu_op("Checkpoint.maybe_save", 1420, 1400)
        corr = b.runtime("cudaLaunchKernel", 2880, 3)
        b.device_op("next", 2900, 50, corr, stream=7)
        view, _ = load_trace_dict(b.build())
        gap = find_gaps(view, 0, min_gap_us=20).gaps[0]
        blame = resolve_blame(view, gap)
        assert len(blame.covering) > 1
        assert all(ev.depth == 0 for ev in blame.covering)
        names = [n for n, _ in blame.top_names(3)]
        assert "EpochMetrics.update" in names
        assert "Checkpoint.maybe_save" in names

    def test_step_annotations_are_never_reported_as_a_cause(self):
        view, _ = load_trace_dict(build_scenario("other_host", steps=4, stall_us=5000))
        for gap in find_gaps(view, 0).gaps:
            names = [n for n, _ in resolve_blame(view, gap).top_names(3)]
            assert not any(n.startswith("ProfilerStep") for n in names)


class _FakeProfiler:
    """Models torch.profiler's schedule closely enough to place the syncs.

    With wait=0, warmup=W, active=A, iteration i runs under the action for step
    i: below W it is WARMUP and discarded, at or above W+A it is NONE. Only
    RECORD reaches the trace.
    """

    def __init__(self, warmup: int, active: int) -> None:
        self.warmup, self.active = warmup, active
        self.step_num = 0
        self.stopped = False

    def action(self) -> str:
        if self.step_num < self.warmup:
            return "WARMUP"
        if self.step_num < self.warmup + self.active:
            return "RECORD"
        return "NONE"

    def step(self) -> None:
        self.step_num += 1

    def stop(self) -> None:
        self.stopped = True


class TestClockSyncLandsInsideTheTrace:
    """Both sync points were emitted outside the profiler's recording window.

    __enter__ ran during WARMUP and __exit__ after the schedule had completed,
    so zero sync points reached the trace, fit_clock raised, and the sampler
    could never be aligned. Every real run would have lost the I/O-versus-CPU
    split -- the one result the project exists for -- and nothing would have
    said so until analysis.
    """

    def _run(self, monkeypatch, tmp_path, *, warmup, active, iterations):
        from whystall.inject import harness as harness_mod
        from whystall.inject.base import NullInjector

        emitted: list[str] = []
        prof = _FakeProfiler(warmup, active)

        counter = iter(range(1, 1000))
        def fake_sync(**_kw):
            emitted.append(prof.action())
            return next(counter)

        monkeypatch.setattr(harness_mod, "emit_clock_sync", fake_sync)
        h = harness_mod.Harness(
            NullInjector(0),
            harness_mod.HarnessConfig(
                out_dir=tmp_path, warmup_steps=warmup, active_steps=active, sampler=False
            ),
        )
        monkeypatch.setattr(h, "_start_profiler", lambda: setattr(h, "_prof", prof))
        with h:
            for _ in range(iterations):
                with h.step():
                    pass
        return h, emitted

    def test_both_sync_points_are_recorded(self, monkeypatch, tmp_path):
        h, emitted = self._run(monkeypatch, tmp_path, warmup=3, active=5, iterations=8)
        assert emitted == ["RECORD", "RECORD"]
        assert len(h._clock_points) == 2

    def test_they_bracket_the_recorded_window(self, monkeypatch, tmp_path):
        from whystall.inject import harness as harness_mod
        from whystall.inject.base import NullInjector

        seen: list[int] = []
        prof = _FakeProfiler(3, 5)
        monkeypatch.setattr(
            harness_mod, "emit_clock_sync", lambda **_k: seen.append(prof.step_num) or 1
        )
        h = harness_mod.Harness(
            NullInjector(0),
            harness_mod.HarnessConfig(
                out_dir=tmp_path, warmup_steps=3, active_steps=5, sampler=False
            ),
        )
        monkeypatch.setattr(h, "_start_profiler", lambda: setattr(h, "_prof", prof))
        with h:
            for _ in range(8):
                with h.step():
                    pass
        assert seen == [3, 7], "first and last recorded steps"

    def test_no_warmup_still_syncs_on_the_first_iteration(self, monkeypatch, tmp_path):
        _, emitted = self._run(monkeypatch, tmp_path, warmup=0, active=4, iterations=4)
        assert emitted == ["RECORD", "RECORD"]

    def test_a_short_loop_warns_rather_than_failing_at_analysis(
        self, monkeypatch, tmp_path, capsys
    ):
        from whystall.inject import harness as harness_mod
        from whystall.inject.base import NullInjector

        prof = _FakeProfiler(3, 5)
        monkeypatch.setattr(harness_mod, "emit_clock_sync", lambda **_k: 1)
        h = harness_mod.Harness(
            NullInjector(0),
            harness_mod.HarnessConfig(
                out_dir=tmp_path, warmup_steps=3, active_steps=5, sampler=True
            ),
        )
        monkeypatch.setattr(h, "_start_profiler", lambda: setattr(h, "_prof", prof))
        monkeypatch.setattr(h, "_start_sampler", lambda: setattr(h, "_sampler", object()))
        monkeypatch.setattr(h, "_stop_sampler", lambda: None)
        with h:
            for _ in range(4):  # stops during warmup
                with h.step():
                    pass
        assert "clock sync" in capsys.readouterr().err
        assert "clock_sync_warning" in h.config.extra

    def test_the_manifest_records_the_sync_points(self, monkeypatch, tmp_path):
        import json

        h, _ = self._run(monkeypatch, tmp_path, warmup=1, active=3, iterations=4)
        manifest = json.loads(h.manifest_path.read_text(encoding="utf-8"))
        assert len(manifest["clock_sync_points_ns"]) == 2


class TestSamplerLogSurvivesBeingKilled:
    """The harness SIGTERMs the sampler; the default action loses the tail.

    Buffered samples covering the end of the profiled window went missing, and
    a half-written final line made json.loads raise, taking the whole run's
    sampler data with it.
    """

    def test_a_truncated_final_line_does_not_lose_the_rest(self, tmp_path):
        import json

        p = tmp_path / "s.jsonl"
        good = [_sample(t * 1_000_000, "DR", read_bytes=t) for t in range(50)]
        with p.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({"meta": {"hz": 500}}) + "\n")
            for s in good:
                fh.write(json.dumps(s.to_json()) + "\n")
            fh.write('{"t": 51000000, "w": [[0, "D", 0, 0')  # killed mid-write
        log = SamplerLog.load(p)
        assert len(log) == 50
        assert log.hz == 500
        assert log.meta["unreadable_records"] == 1

    def test_a_structurally_wrong_record_is_counted_not_fatal(self, tmp_path):
        import json

        p = tmp_path / "s.jsonl"
        with p.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({"meta": {"hz": 200}}) + "\n")
            fh.write(json.dumps({"t": 1, "w": [["not-a-pid"]]}) + "\n")
            fh.write(json.dumps(_sample(2_000_000, "DD").to_json()) + "\n")
        log = SamplerLog.load(p)
        assert len(log) == 1
        assert log.meta["unreadable_records"] == 1

    def test_run_installs_a_sigterm_handler(self):
        """Verified by inspection: the default action skips the flush."""
        import inspect

        from whystall.enrich import sampler as sampler_mod

        src = inspect.getsource(sampler_mod.run)
        assert "signal.SIGTERM" in src
        assert "fh.flush()" in src


class TestValidateFindsTracesInARunDirectory:
    """A directory of real runs reported 'no traces found'.

    The harness writes <run>.trace.json, but anyone using --out from synth gets
    plain .json, and the directory glob only matched the former.
    """

    def test_plain_json_traces_are_found(self, tmp_path, capsys):
        main(["synth", "host_sync", "--out", str(tmp_path / "a.json"),
              "--steps", "6", "--stall-us", "6000"])
        assert main(["validate", str(tmp_path)]) == 0
        assert "truth \\ pred" in capsys.readouterr().out

    def test_manifests_and_reports_are_not_mistaken_for_traces(self, tmp_path):
        from whystall.cli import _collect_traces

        for name in ("run.trace.json", "run.manifest.json", "demo-report.json",
                     "run.samples.jsonl"):
            (tmp_path / name).write_text("{}", encoding="utf-8")
        found = {p.name for p in _collect_traces([str(tmp_path)])}
        assert found == {"run.trace.json"}

    def test_each_trace_is_returned_once(self, tmp_path):
        from whystall.cli import _collect_traces

        (tmp_path / "a.trace.json").write_text("{}", encoding="utf-8")
        assert len(_collect_traces([str(tmp_path)])) == 1


class TestThreadScanIsExactAndCheap:
    """The backward scan used a fixed 4096-event tolerance.

    It could miss a genuinely long enclosing frame, and it walked 4096 events
    per gap regardless -- on a dispatch-bound trace with tens of thousands of
    gaps that is hundreds of millions of iterations.
    """

    def test_finds_an_enclosing_frame_beyond_the_old_limit(self):
        events = [{"ph": "X", "cat": "user_annotation", "name": "long_step",
                   "pid": 1, "tid": 1, "ts": 0, "dur": 10_000_000, "args": {}}]
        for i in range(6000):  # more than the old 4096 tolerance
            events.append({"ph": "X", "cat": "cpu_op", "name": f"op{i}", "pid": 1,
                           "tid": 1, "ts": 10 + i * 4, "dur": 2, "args": {}})
        view, _ = load_trace_dict({"traceEvents": events})
        covering = view.events_on_thread(1, 9_000_000, 9_000_100)
        assert [e.name for e in covering] == ["long_step"]

    def test_stops_immediately_once_nothing_can_still_be_open(self):
        events = []
        for i in range(60_000):
            events.append({"ph": "X", "cat": "cpu_op", "name": f"op{i}", "pid": 1,
                           "tid": 1, "ts": i * 10, "dur": 2, "args": {}})
        view, _ = load_trace_dict({"traceEvents": events})
        start = time.perf_counter()
        for i in range(3_000):
            view.events_on_thread(1, 500_000 + i * 10, 500_000 + i * 10 + 5)
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0, f"{elapsed:.1f}s for 3000 lookups"

    def test_returns_frames_outermost_first(self):
        events = [
            {"ph": "X", "cat": "user_annotation", "name": "outer", "pid": 1,
             "tid": 1, "ts": 0, "dur": 1000, "args": {}},
            {"ph": "X", "cat": "cpu_op", "name": "middle", "pid": 1,
             "tid": 1, "ts": 100, "dur": 500, "args": {}},
            {"ph": "X", "cat": "cpu_op", "name": "inner", "pid": 1,
             "tid": 1, "ts": 200, "dur": 100, "args": {}},
        ]
        view, _ = load_trace_dict({"traceEvents": events})
        assert [e.name for e in view.events_on_thread(1, 250, 260)] == [
            "outer", "middle", "inner"
        ]


class TestSubMicrosecondEventsAreNotDropped:
    """int() truncation turned a 0.4 us kernel into dur=0.

    busy_intervals filters dur > 0, so the kernel vanished from the busy
    timeline and its slot became idle -- fabricating a stall out of real work,
    on exactly the dispatch-bound traces this tool is built to diagnose. Chrome
    ts and dur are floats in microseconds and Kineto emits fractional values, so
    this is the normal case, not an edge one.
    """

    def _trace(self, durations):
        events = []
        t = 1000.0
        for i, d in enumerate(durations):
            events.append({"ph": "X", "cat": "kernel", "name": f"k{i}", "pid": 0,
                           "tid": 7, "ts": t, "dur": d,
                           "args": {"correlation": i, "device": 0, "stream": 7}})
            t += d + 0.5
        return {"traceEvents": events}

    def test_a_sub_microsecond_kernel_survives(self):
        view, _ = load_trace_dict(self._trace([0.4, 0.3, 0.45]))
        assert len(view.gpu) == 3
        assert all(e.dur >= 1 for e in view.gpu), "a real kernel must not vanish"

    def test_the_rounding_is_recorded_not_silent(self):
        view, _ = load_trace_dict(self._trace([0.4, 0.3, 12.0]))
        q = view.meta["whystall_quantisation"]
        assert q["resolution_us"] == 1
        assert q["events_below_half_us"] == 2

    def test_a_clean_integer_trace_records_nothing(self):
        view, _ = load_trace_dict(self._trace([4.0, 8.0]))
        assert "whystall_quantisation" not in view.meta

    def test_timestamps_round_rather_than_truncate(self):
        doc = {"traceEvents": [
            {"ph": "X", "cat": "kernel", "name": "k", "pid": 0, "tid": 7,
             "ts": 1000.7, "dur": 5.0,
             "args": {"correlation": 1, "device": 0, "stream": 7}},
        ]}
        view, _ = load_trace_dict(doc)
        assert view.gpu[0].ts == 1001

    def test_zero_duration_events_stay_zero(self):
        """An instant marker is not a kernel and must not be inflated."""
        doc = {"traceEvents": [
            {"ph": "X", "cat": "cpu_op", "name": "marker", "pid": 1, "tid": 1,
             "ts": 10, "dur": 0, "args": {}},
        ]}
        view, _ = load_trace_dict(doc)
        assert view.cpu[0].dur == 0

    def test_the_report_states_its_own_resolution(self):
        from whystall.analysis import attribute
        from whystall.report.text import render_attribution

        view, labels = load_trace_dict(self._trace([0.4] * 6 + [500.0]))
        text = render_attribution(attribute(view, labels=labels))
        assert "sub-microsecond" in text


class TestCLIParserContract:
    def test_threshold_flags_default_to_none_not_a_value(self):
        """_config distinguishes 'not given' from 'given as zero'."""
        args = build_parser().parse_args(["selftest"])
        assert args.min_gap_us is None
        assert args.launch_bound_max_us is None
        assert args.leaf_dominance is None
