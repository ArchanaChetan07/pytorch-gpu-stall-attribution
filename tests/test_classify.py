"""Classification rules, their ordering, and the class each scenario produces."""

from __future__ import annotations

import pytest

from whystall.synth import TraceBuilder
from whystall.synth.scenarios import GROUND_TRUTH_CLASS
from whystall.taxonomy import ENRICHED_CLASSES, TRACE_ONLY_CLASSES, StallClass
from whystall.trace.blame import Blame, resolve_blame
from whystall.trace.classify import RULES, ClassifierConfig, classify
from whystall.trace.load import load_trace_dict
from whystall.trace.timeline import Gap, find_gaps


def _one_gap_verdict(builder: TraceBuilder, cfg=None):
    """Classify the single idle window in a hand-built trace.

    The analysis window comes from the trace's own bounds, so the op that closes
    the gap is inside it -- passing an explicit window that ends at the gap turns
    the gap into trailing idle, which is not what any of these cases are about.
    """
    view, _ = load_trace_dict(builder.build())
    gaps = find_gaps(view, 0, min_gap_us=20).gaps
    assert len(gaps) == 1, f"expected one gap, got {gaps}"
    return classify(resolve_blame(view, gaps[0]), cfg)


def _dominant_class(att, device=0):
    by_us = att.device(device).by_class_ns()
    return max(by_us, key=lambda k: by_us[k])


class TestScenarioAttribution:
    """The end-to-end contract: each scenario's idle is dominated by its class."""

    @pytest.mark.parametrize(
        "kind",
        [k for k, v in GROUND_TRUTH_CLASS.items() if v is not None and k != "launch"],
    )
    def test_dominant_class_matches_ground_truth(self, kind, make_attribution):
        att = make_attribution(kind, steps=8, stall_us=6000)
        expected = GROUND_TRUTH_CLASS[kind]
        got = _dominant_class(att)
        # Trace-only classification cannot split the dataloader refinements.
        if expected.startswith("dataloader"):
            assert got is StallClass.DATALOADER
        else:
            assert got.value == expected

    def test_launch_bound_scenario(self, make_attribution):
        att = make_attribution("launch", steps=6, stall_us=34)
        assert _dominant_class(att) is StallClass.LAUNCH_DISPATCH

    def test_clean_run_is_almost_entirely_busy(self, make_attribution):
        att = make_attribution("clean", steps=10)
        assert att.device(0).gapset.utilisation > 0.95

    def test_no_gap_is_left_unclassified(self, make_attribution):
        att = make_attribution("dataloader", steps=6, stall_us=5000)
        dev = att.device(0)
        assert len(dev.verdicts) == len(dev.gapset.gaps)

    def test_injection_raises_idle_above_control(self, make_attribution):
        clean = make_attribution("clean", steps=8)
        stalled = make_attribution("dataloader", steps=8, stall_us=6000)
        assert stalled.device(0).gapset.idle_total_ns > clean.device(0).gapset.idle_total_ns * 5


class TestRuleOrdering:
    def test_device_dependency_runs_before_every_host_rule(self):
        assert RULES[0].__name__ == "rule_device_dependency"

    def test_a_dominant_dataloader_frame_beats_the_memcpy_that_closes_the_gap(self):
        """The exact error the ordering exists to prevent."""
        b = TraceBuilder()
        b.device_op("prev", 0, 10, 1, stream=7)
        b.annotation("enumerate(DataLoader)#_MultiProcessingDataLoaderIter.__next__", 10, 2800)
        b.cpu_op("aten::to", 2810, 80)
        corr = b.runtime("cudaMemcpyAsync", 2880, 3)
        b.device_op("Memcpy HtoD (Pinned -> Device)", 2900, 50, corr,
                    stream=20, cat="gpu_memcpy")
        v = _one_gap_verdict(b)
        assert v.cls is StallClass.DATALOADER
        assert v.rule == "dataloader_frame"

    def test_a_dominant_copy_frame_beats_a_brief_dataloader_frame(self):
        b = TraceBuilder()
        b.device_op("prev", 0, 10, 1, stream=7)
        b.annotation("enumerate(DataLoader)#_MultiProcessingDataLoaderIter.__next__", 10, 90)
        b.cpu_op("aten::to", 100, 2800)
        corr = b.runtime("cudaMemcpy", 2890, 3)
        b.device_op("Memcpy HtoD (Pageable -> Device)", 2910, 50, corr,
                    stream=20, cat="gpu_memcpy")
        v = _one_gap_verdict(b)
        assert v.cls is StallClass.H2D_COPY

    def test_collective_kernel_decides_regardless_of_the_host_frame(self):
        b = TraceBuilder()
        b.device_op("prev", 0, 10, 1, stream=7)
        b.cpu_op("EpochMetrics.update", 10, 2800)
        corr = b.runtime("cudaLaunchKernel", 2880, 3)
        b.device_op("ncclDevKernel_AllReduce_Sum_f32_RING_LL", 2900, 400, corr, stream=21)
        v = _one_gap_verdict(b)
        assert v.cls is StallClass.COLLECTIVE
        assert v.rule == "collective_next_kernel"

    def test_launch_bound_refuses_to_fire_when_a_named_cause_is_present(self):
        """A short dataloader stall stays a dataloader stall."""
        b = TraceBuilder()
        b.device_op("prev", 0, 10, 1, stream=7)
        b.annotation("enumerate(DataLoader)#_MultiProcessingDataLoaderIter.__next__", 10, 120)
        corr = b.runtime("cudaLaunchKernel", 130, 3)
        b.device_op("k", 140, 50, corr, stream=7)
        v = _one_gap_verdict(b)
        assert v.cls is StallClass.DATALOADER

    def test_a_dominant_unrecognised_frame_is_other_host_not_the_closing_op(self):
        b = TraceBuilder()
        b.device_op("prev", 0, 10, 1, stream=7)
        b.cpu_op("EpochMetrics.update", 10, 2800)
        corr = b.runtime("cudaMemcpyAsync", 2880, 3)
        b.device_op("Memcpy HtoD (Pinned -> Device)", 2900, 50, corr,
                    stream=20, cat="gpu_memcpy")
        v = _one_gap_verdict(b)
        assert v.cls is StallClass.OTHER_HOST
        assert v.rule == "other_host_frame"

    def test_readback_closing_a_gap_reads_as_host_sync(self):
        b = TraceBuilder()
        b.device_op("prev", 0, 10, 1, stream=7)
        b.cpu_op("many_small_things", 10, 400)
        b.cpu_op("and_another", 420, 400)
        corr = b.runtime("cudaMemcpyAsync", 870, 3)
        b.device_op("Memcpy DtoH (Device -> Pageable)", 900, 20, corr,
                    stream=20, cat="gpu_memcpy")
        v = _one_gap_verdict(b)
        assert v.cls is StallClass.HOST_SYNC


class TestArchitecturalBoundary:
    def test_trace_only_classifier_never_emits_a_sampler_class(self, make_attribution):
        for kind in ("dataloader", "dataloader_io", "dataloader_cpu", "h2d", "clean"):
            att = make_attribution(kind, steps=6, stall_us=5000)
            for v in att.verdicts:
                assert v.cls in TRACE_ONLY_CLASSES
                assert v.cls not in ENRICHED_CLASSES

    def test_io_and_cpu_scenarios_are_indistinguishable_in_the_trace(
        self, make_attribution
    ):
        """The premise of the whole sampler: a trace cannot tell these apart."""
        io = make_attribution("dataloader_io", steps=8, stall_us=5000)
        cpu = make_attribution("dataloader_cpu", steps=8, stall_us=5000)
        assert io.device(0).by_class_ns() == cpu.device(0).by_class_ns()


class TestConfig:
    def test_raising_the_gap_floor_drops_short_gaps(self, make_view):
        view, _ = make_view("launch", steps=4, stall_us=34)
        few = find_gaps(view, 0, min_gap_us=1000)
        many = find_gaps(view, 0, min_gap_us=20)
        assert len(few.gaps) < len(many.gaps)
        assert few.idle_total_ns == many.idle_total_ns

    def test_thresholds_are_reported(self):
        cfg = ClassifierConfig(min_gap_us=33)
        assert cfg.as_dict()["min_gap_us"] == 33
        assert set(cfg.as_dict()) >= {
            "min_gap_us", "launch_bound_max_us", "leaf_dominance", "launch_dominance"
        }



class TestHTACorrespondence:
    """Pins the exact identity against Holistic Trace Analysis.

    HTA's `_analyze_idle_time_for_stream` labels every gap OTHER, promotes it to
    HOST_WAIT when `ts_runtime > prev_end_ts`, then to KERNEL_WAIT when it is not
    host wait and the gap is under `consecutive_kernel_delay` (default 30 ns).
    Its `~is_host_wait` is the same predicate as `queued_before_gap` here, so:

        device_dependency == KERNEL_WAIT union OTHER
        HTA host_wait     == every other class this tool emits

    PRIOR-ART.md argues from that identity, so it is pinned here rather than left
    to prose. An earlier draft had the launch_dispatch relation backwards and
    nothing caught it, because nothing tested it.
    """

    def _blame(self, *, queued: bool, dur_ns: int) -> Blame:
        gap = Gap(device=0, start_ns=1_000_000, end_ns=1_000_000 + dur_ns)
        b = Blame(gap=gap, window_ns=(gap.start_ns, gap.start_ns))
        b.queued_before_gap = queued
        return b

    @pytest.mark.parametrize("dur_ns", [10, 30, 1_000, 199_000, 10_000_000])
    def test_queued_before_gap_is_always_device_dependency(self, dur_ns):
        """Every gap HTA would call KERNEL_WAIT or OTHER, at any duration."""
        verdict = classify(self._blame(queued=True, dur_ns=dur_ns))
        assert verdict.cls is StallClass.DEVICE_DEPENDENCY

    @pytest.mark.parametrize("dur_ns", [10, 29, 30, 1_000])
    def test_launch_dispatch_is_disjoint_from_hta_kernel_wait(self, dur_ns):
        """The correction: kernel wait can never reach rule_launch_dispatch.

        These durations straddle HTA's 30 ns default and sit far below this
        tool's 200 us launch bound, so a superset relation would show up here.
        It does not, because rule_device_dependency runs first.
        """
        verdict = classify(self._blame(queued=True, dur_ns=dur_ns))
        assert verdict.rule != "launch_dispatch"
        assert verdict.cls is not StallClass.LAUNCH_DISPATCH

    def test_rule_order_is_what_makes_the_identity_hold(self):
        """device_dependency must precede launch_dispatch, or the identity breaks."""
        names = [r.__name__ for r in RULES]
        assert names.index("rule_device_dependency") < names.index("rule_launch_dispatch")
        assert names[0] == "rule_device_dependency"

    def test_not_queued_never_yields_device_dependency(self, make_attribution):
        """The other half: HTA's host_wait is everything else we emit.

        No verdict on a trace may be device_dependency unless its blame said the
        launch was already enqueued.
        """
        for kind in ("dataloader", "h2d", "launch", "clean"):
            att = make_attribution(kind, steps=6, stall_us=5000)
            for v in att.verdicts:
                if v.cls is StallClass.DEVICE_DEPENDENCY:
                    assert v.rule == "device_dependency"
