"""Loading, timelines and blame resolution."""

from __future__ import annotations

import gzip
import json

import pytest

from whystall.synth import TraceBuilder, build_scenario
from whystall.trace.blame import resolve_blame
from whystall.trace.load import load_trace, load_trace_dict, memcpy_kind
from whystall.trace.schema import Event, Kind
from whystall.trace.timeline import busy_intervals, find_gaps


def _ev(name, ts, dur, tid=1, kind=Kind.CPU, cat="cpu_op"):
    return Event(name=name, cat=cat, kind=kind, pid=1, tid=tid, ts=ts, dur=dur)


class TestEvent:
    def test_overlap_clips_both_ends(self):
        e = _ev("x", 100, 50)
        assert e.overlap(0, 1000) == 50
        assert e.overlap(120, 130) == 10
        assert e.overlap(0, 100) == 0
        assert e.overlap(150, 200) == 0
        assert e.overlap(90, 110) == 10

    def test_end_is_exclusive_upper(self):
        assert _ev("x", 10, 5).end == 15


class TestLoad:
    def test_categories_are_case_insensitive(self):
        doc = {
            "traceEvents": [
                {"ph": "X", "cat": "Kernel", "name": "k", "pid": 0, "tid": 7,
                 "ts": 10, "dur": 5, "args": {"correlation": 1, "device": 0, "stream": 7}},
                {"ph": "X", "cat": "Runtime", "name": "cudaLaunchKernel", "pid": 1,
                 "tid": 1, "ts": 4, "dur": 2, "args": {"correlation": 1}},
                {"ph": "X", "cat": "Operator", "name": "aten::add", "pid": 1,
                 "tid": 1, "ts": 3, "dur": 4, "args": {}},
            ]
        }
        view, _ = load_trace_dict(doc)
        assert len(view.gpu) == 1 and len(view.runtime) == 1 and len(view.cpu) == 1

    def test_non_complete_events_are_dropped(self):
        doc = {"traceEvents": [
            {"ph": "s", "cat": "ac2g", "name": "flow", "pid": 1, "tid": 1, "ts": 1},
            {"ph": "M", "name": "process_name", "pid": 1, "tid": 1, "args": {}},
        ]}
        view, labels = load_trace_dict(doc)
        assert not view.gpu and not view.cpu and not labels

    def test_device_falls_back_to_pid_and_stream_to_tid(self):
        doc = {"traceEvents": [
            {"ph": "X", "cat": "kernel", "name": "k", "pid": 3, "tid": 9,
             "ts": 0, "dur": 1, "args": {"correlation": 1}},
        ]}
        view, _ = load_trace_dict(doc)
        assert view.gpu[0].device == 3
        assert view.gpu[0].stream == 9

    def test_earliest_runtime_call_wins_a_correlation(self):
        doc = {"traceEvents": [
            {"ph": "X", "cat": "cuda_runtime", "name": "cudaStreamSynchronize",
             "pid": 1, "tid": 1, "ts": 900, "dur": 5, "args": {"correlation": 42}},
            {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel",
             "pid": 1, "tid": 1, "ts": 100, "dur": 5, "args": {"correlation": 42}},
        ]}
        view, _ = load_trace_dict(doc)
        assert view.launch_by_correlation[42].name == "cudaLaunchKernel"

    def test_roundtrip_through_gzip(self, tmp_path):
        doc = build_scenario("clean", steps=2)
        p = tmp_path / "t.json.gz"
        with gzip.open(p, "wt", encoding="utf-8") as fh:
            json.dump(doc, fh)
        view, _ = load_trace(p)
        assert view.gpu and view.source.endswith("t.json.gz")

    @pytest.mark.parametrize(
        "name,direction,host",
        [
            ("Memcpy HtoD (Pageable -> Device)", "HtoD", "pageable"),
            ("Memcpy HtoD (Pinned -> Device)", "HtoD", "pinned"),
            ("Memcpy DtoH (Device -> Pageable)", "DtoH", "pageable"),
            ("Memcpy DtoD (Device -> Device)", "DtoD", None),
            ("void some_kernel<float>", None, None),
        ],
    )
    def test_memcpy_kind(self, name, direction, host):
        assert memcpy_kind(name) == (direction, host)


class TestThreadIndex:
    def test_depth_follows_nesting(self):
        doc = {"traceEvents": [
            {"ph": "X", "cat": "user_annotation", "name": "outer", "pid": 1, "tid": 1,
             "ts": 0, "dur": 100, "args": {}},
            {"ph": "X", "cat": "cpu_op", "name": "mid", "pid": 1, "tid": 1,
             "ts": 10, "dur": 50, "args": {}},
            {"ph": "X", "cat": "cpu_op", "name": "leaf", "pid": 1, "tid": 1,
             "ts": 20, "dur": 10, "args": {}},
            {"ph": "X", "cat": "cpu_op", "name": "sibling", "pid": 1, "tid": 1,
             "ts": 70, "dur": 5, "args": {}},
        ]}
        view, _ = load_trace_dict(doc)
        depths = {e.name: e.depth for e in view.cpu}
        assert depths == {"outer": 0, "mid": 1, "leaf": 2, "sibling": 1}

    def test_events_on_thread_finds_a_distant_enclosing_frame(self):
        events = [
            {"ph": "X", "cat": "user_annotation", "name": "step", "pid": 1, "tid": 1,
             "ts": 0, "dur": 10_000, "args": {}},
        ]
        for i in range(500):
            events.append({"ph": "X", "cat": "cpu_op", "name": f"op{i}", "pid": 1,
                           "tid": 1, "ts": 10 + i * 4, "dur": 2, "args": {}})
        view, _ = load_trace_dict({"traceEvents": events})
        covering = view.events_on_thread(1, 9_000, 9_100)
        assert covering and covering[0].name == "step"

    def test_malformed_nesting_does_not_raise(self):
        doc = {"traceEvents": [
            {"ph": "X", "cat": "cpu_op", "name": "a", "pid": 1, "tid": 1,
             "ts": 0, "dur": 50, "args": {}},
            {"ph": "X", "cat": "cpu_op", "name": "b", "pid": 1, "tid": 1,
             "ts": 25, "dur": 100, "args": {}},
        ]}
        view, _ = load_trace_dict(doc)
        assert len(view.cpu) == 2


class TestTimeline:
    def test_busy_is_the_union_across_streams(self):
        b = TraceBuilder()
        b.device_op("k1", 100, 100, 1, stream=7)
        b.device_op("k2", 150, 100, 2, stream=20)
        b.device_op("k3", 400, 50, 3, stream=7)
        view, _ = load_trace_dict(b.build())
        assert busy_intervals(view, 0) == [(100, 250), (400, 450)]

    def test_a_gap_on_one_stream_is_not_idle_if_another_runs(self):
        b = TraceBuilder()
        b.device_op("compute", 0, 100, 1, stream=7)
        b.device_op("nccl", 100, 200, 2, stream=21)
        b.device_op("compute2", 300, 100, 3, stream=7)
        view, _ = load_trace_dict(b.build())
        gs = find_gaps(view, 0, min_gap_us=1, window=(0, 400))
        assert gs.gaps == []

    def test_short_gaps_are_counted_but_not_returned(self):
        b = TraceBuilder()
        b.device_op("k1", 0, 100, 1, stream=7)
        b.device_op("k2", 105, 100, 2, stream=7)
        b.device_op("k3", 500, 100, 3, stream=7)
        view, _ = load_trace_dict(b.build())
        gs = find_gaps(view, 0, min_gap_us=20, window=(0, 600))
        assert [g.dur for g in gs.gaps] == [295]
        assert gs.idle_subthreshold_us == 5
        assert gs.idle_total_us == 300

    def test_neighbours_bracket_each_gap(self):
        b = TraceBuilder()
        b.device_op("before", 0, 100, 1, stream=7)
        b.device_op("after", 500, 100, 2, stream=7)
        view, _ = load_trace_dict(b.build())
        gap = find_gaps(view, 0, min_gap_us=20, window=(0, 600)).gaps[0]
        assert gap.prev_gpu.name == "before"
        assert gap.next_gpu.name == "after"

    def test_utilisation_matches_busy_over_window(self):
        b = TraceBuilder()
        b.device_op("k", 0, 250, 1, stream=7)
        view, _ = load_trace_dict(b.build())
        gs = find_gaps(view, 0, window=(0, 1000))
        assert gs.utilisation == pytest.approx(0.25)


class TestBlame:
    def test_launch_before_the_gap_is_a_device_dependency(self):
        b = TraceBuilder()
        b.device_op("first", 1000, 100, 1, stream=7)
        corr = b.runtime("cudaLaunchKernel", 500, 3)
        b.device_op("blocked", 5000, 100, corr, stream=7)
        view, _ = load_trace_dict(b.build())
        gap = find_gaps(view, 0, min_gap_us=20, window=(1000, 5100)).gaps[0]
        blame = resolve_blame(view, gap)
        assert blame.queued_before_gap
        assert blame.window_us == 0

    def test_leaf_is_the_deepest_covering_frame(self):
        b = TraceBuilder()
        b.device_op("first", 1000, 10, 1, stream=7)
        b.annotation("ProfilerStep#0", 900, 3000)
        b.cpu_op("aten::conv2d", 1000, 2000)
        b.cpu_op("inner_detail", 1500, 100)
        corr = b.runtime("cudaLaunchKernel", 3000, 3)
        b.device_op("next", 3010, 10, corr, stream=7)
        view, _ = load_trace_dict(b.build())
        gap = find_gaps(view, 0, min_gap_us=20, window=(1010, 3020)).gaps[0]
        blame = resolve_blame(view, gap)
        assert blame.leaf.name == "aten::conv2d"
        assert blame.leaf_fraction > 0.9

    def test_a_blocking_runtime_call_outranks_the_op_enclosing_it(self):
        b = TraceBuilder()
        b.device_op("first", 1000, 10, 1, stream=7)
        b.cpu_op("aten::item", 1010, 1900)
        b.runtime("cudaStreamSynchronize", 1010, 1890)
        corr = b.runtime("cudaLaunchKernel", 2950, 3)
        b.device_op("next", 2960, 10, corr, stream=7)
        view, _ = load_trace_dict(b.build())
        gap = find_gaps(view, 0, min_gap_us=20, window=(1010, 2970)).gaps[0]
        blame = resolve_blame(view, gap)
        assert blame.leaf.name == "cudaStreamSynchronize"

    def test_missing_correlation_yields_an_unresolved_blame(self):
        b = TraceBuilder()
        b.device_op("first", 0, 10, 1, stream=7)
        b.device_op("orphan", 5000, 10, 999, stream=7)
        view, _ = load_trace_dict(b.build())
        gap = find_gaps(view, 0, min_gap_us=20, window=(10, 5010)).gaps[0]
        blame = resolve_blame(view, gap)
        assert blame.launch is None and not blame.resolved

    def test_top_names_skips_the_outermost_step_frame(self):
        att_view, _ = load_trace_dict(build_scenario("other_host", steps=3, stall_us=4000))
        gs = find_gaps(att_view, 0)
        blames = [resolve_blame(att_view, g) for g in gs.gaps]
        interesting = [b for b in blames if b.leaf and "Metrics" in b.leaf.name]
        assert interesting
        names = [n for n, _ in interesting[0].top_names(3)]
        assert not any(n.startswith("ProfilerStep") for n in names)


class TestBounds:
    def test_window_extends_past_the_last_host_step(self):
        """The host always finishes a step before the device does."""
        view, _ = load_trace_dict(build_scenario("clean", steps=4))
        lo, hi = view.bounds()
        assert hi >= max(e.end for e in view.gpu)
        assert lo == view.steps[0].ts
