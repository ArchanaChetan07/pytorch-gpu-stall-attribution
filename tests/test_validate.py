"""The firewall, ground-truth association, the matrix, and detection floors."""

from __future__ import annotations

import pytest

from whystall.analysis import attribute
from whystall.synth import build_scenario
from whystall.taxonomy import StallClass
from whystall.trace.load import load_trace_dict
from whystall.trace.schema import LABEL_PREFIX, Label
from whystall.validate.firewall import FirewallBreach, assert_clean, assert_evidence_clean, scan
from whystall.validate.floor import detection_floor
from whystall.validate.groundtruth import (
    Granularity,
    collapse,
    injected_regions,
    label_gaps,
)
from whystall.validate.matrix import NO_TRUTH, ConfusionMatrix
from whystall.validate.score import Sweep, score_attribution


class TestFirewall:
    """If any test in this file matters most, it is this class."""

    def test_loader_removes_ground_truth_from_the_view(self):
        doc = build_scenario("dataloader", steps=4, stall_us=4000)
        view, labels = load_trace_dict(doc)
        assert labels, "the scenario should have emitted ground truth"
        assert scan(view) == []
        assert_clean(view)

    def test_ground_truth_is_present_in_the_raw_document(self):
        """Proves the previous test is removing something real."""
        doc = build_scenario("dataloader", steps=4, stall_us=4000)
        raw = [e for e in doc["traceEvents"] if e["name"].startswith(LABEL_PREFIX)]
        assert raw

    def test_firewall_catches_a_view_that_kept_the_labels(self):
        doc = build_scenario("dataloader", steps=4, stall_us=4000)
        dirty, _ = load_trace_dict(doc, keep_labels=True)
        with pytest.raises(FirewallBreach):
            assert_clean(dirty)

    def test_firewall_catches_ground_truth_cited_as_evidence(self):
        class FakeVerdict:
            gap = None
            evidence = {"top_ops": [(f"{LABEL_PREFIX}inject/host_sync/900", 12)]}

        with pytest.raises(FirewallBreach):
            assert_evidence_clean([FakeVerdict()])

    def test_no_verdict_in_a_scored_run_cites_ground_truth(self):
        doc = build_scenario("host_sync", steps=6, stall_us=5000)
        view, labels = load_trace_dict(doc)
        att = attribute(view, labels=labels)
        assert_evidence_clean(att.verdicts)

    def test_scoring_asserts_the_firewall_itself(self):
        """A path that produces a score also proves the classifier was blind."""
        doc = build_scenario("h2d", steps=4, stall_us=5000)
        view, labels = load_trace_dict(doc)
        att = attribute(view, labels=labels)
        att.view.cpu.append(att.view.cpu[0].__class__(
            name=f"{LABEL_PREFIX}inject/h2d/5000", cat="user_annotation", kind=1,
            pid=1, tid=1, ts=0, dur=1,
        ))
        with pytest.raises(FirewallBreach):
            score_attribution(att)


class TestGroundTruth:
    def test_parses_class_and_magnitude(self):
        labels = [Label("inject", ("host_sync", "4200"), 1, 1, 100, 50)]
        (region,) = injected_regions(labels)
        assert region.cls is StallClass.HOST_SYNC
        assert region.magnitude_us == 4200

    def test_unknown_classes_are_skipped_not_fatal(self):
        labels = [
            Label("inject", ("from_the_future", "10"), 1, 1, 0, 1),
            Label("inject", ("host_sync", "10"), 1, 1, 0, 1),
        ]
        assert len(injected_regions(labels)) == 1

    def test_non_inject_labels_are_ignored(self):
        assert injected_regions([Label("clock", ("123",), 1, 1, 0, 1)]) == []

    def test_association_uses_the_blame_window(self):
        doc = build_scenario("host_sync", steps=6, stall_us=6000)
        view, labels = load_trace_dict(doc)
        att = attribute(view, labels=labels)
        dev = att.devices[0]
        truths = label_gaps(dev.blames, injected_regions(labels))
        matched = [t for t in truths if t is not None]
        assert matched, "injected regions should have labelled at least one gap"
        assert all(t.cls is StallClass.HOST_SYNC for t in matched)

    def test_unlabelled_gaps_stay_unlabelled(self):
        """The control gaps must not be swept into the injected class."""
        doc = build_scenario("host_sync", steps=6, stall_us=6000)
        view, labels = load_trace_dict(doc)
        att = attribute(view, labels=labels)
        dev = att.devices[0]
        truths = label_gaps(dev.blames, injected_regions(labels))
        assert any(t is None for t in truths)

    def test_a_control_run_has_no_ground_truth_at_all(self):
        doc = build_scenario("clean", steps=6)
        _, labels = load_trace_dict(doc)
        assert injected_regions(labels) == []

    @pytest.mark.parametrize(
        "cls,expected",
        [
            (StallClass.DATALOADER_IO, StallClass.DATALOADER),
            (StallClass.DATALOADER_CPU, StallClass.DATALOADER),
            (StallClass.DATALOADER_MIXED, StallClass.DATALOADER),
            (StallClass.HOST_SYNC, StallClass.HOST_SYNC),
            (None, None),
        ],
    )
    def test_coarse_collapses_only_the_dataloader_family(self, cls, expected):
        assert collapse(cls, Granularity.COARSE) is expected
        if cls is not None:
            assert collapse(cls, Granularity.STRICT) is cls


class TestConfusionMatrix:
    def _matrix(self):
        m = ConfusionMatrix()
        for _ in range(8):
            m.add(StallClass.HOST_SYNC, StallClass.HOST_SYNC, us=1000)
        for _ in range(2):
            m.add(StallClass.HOST_SYNC, StallClass.H2D_COPY, us=500)
        for _ in range(5):
            m.add(StallClass.H2D_COPY, StallClass.H2D_COPY, us=300)
        for _ in range(3):
            m.add(None, StallClass.DATALOADER, us=100)
        return m

    def test_recall_and_precision(self):
        m = self._matrix()
        sync = m.score(StallClass.HOST_SYNC)
        assert sync.support == 10 and sync.hits == 8
        assert sync.recall == pytest.approx(0.8)
        assert sync.precision == pytest.approx(1.0)
        copy = m.score(StallClass.H2D_COPY)
        assert copy.precision == pytest.approx(5 / 7)

    def test_f1_is_the_harmonic_mean(self):
        m = self._matrix()
        s = m.score(StallClass.HOST_SYNC)
        assert s.f1 == pytest.approx(2 * 1.0 * 0.8 / 1.8)

    def test_accuracy_ignores_unlabelled_rows(self):
        m = self._matrix()
        assert m.accuracy == pytest.approx(13 / 15)

    def test_unlabelled_gaps_are_kept_as_a_row(self):
        m = self._matrix()
        assert NO_TRUTH in m.truths()
        assert m.total == 18

    def test_unlabelled_blame_rate_counts_only_injectable_classes(self):
        m = ConfusionMatrix()
        m.add(None, StallClass.DEVICE_DEPENDENCY)
        m.add(None, StallClass.UNATTRIBUTED)
        assert m.unlabelled_blame_rate == pytest.approx(0.5)

    def test_top_confusions_are_off_diagonal_and_sorted(self):
        m = self._matrix()
        conf = m.confusions()
        assert conf == [(StallClass.HOST_SYNC, StallClass.H2D_COPY, 2)]

    def test_microseconds_tracked_alongside_counts(self):
        m = self._matrix()
        assert m.micros[(StallClass.HOST_SYNC, StallClass.HOST_SYNC)] == 8000

    def test_renders_without_raising_when_empty(self):
        assert "empty" in ConfusionMatrix().render_text()

    def test_dict_round_trips_the_headline_numbers(self):
        d = self._matrix().to_dict()
        assert d["total"] == 18
        assert {c["class"] for c in d["per_class"]} >= {"host_sync", "h2d_copy"}


class TestDetectionFloor:
    def _sweep_with(self, points):
        """Build a sweep whose recall curve is exactly 'points'."""
        sweep = Sweep()
        for magnitude, hits, misses in points:
            m = ConfusionMatrix()
            for _ in range(hits):
                m.add(StallClass.HOST_SYNC, StallClass.HOST_SYNC, us=10)
            for _ in range(misses):
                m.add(StallClass.HOST_SYNC, StallClass.OTHER_HOST, us=10)
            from whystall.validate.score import RunScore

            sweep.add(
                RunScore(
                    name=f"m{magnitude}", injected=StallClass.HOST_SYNC,
                    magnitude_us=magnitude, matrix=m, gaps=hits + misses,
                    labelled_gaps=hits + misses, idle_us=(hits + misses) * 10,
                    unattributed_share=0.0,
                )
            )
        return sweep

    def test_interpolates_between_bracketing_points(self):
        sweep = self._sweep_with([(1000, 4, 6), (2000, 10, 0)])
        f = detection_floor(sweep, StallClass.HOST_SYNC, target=0.9, min_support=5)
        # recall 0.4 -> 1.0 across 1000 -> 2000 us; 0.9 sits five sixths along.
        assert f.floor_us == pytest.approx(1000 + (0.5 / 0.6) * 1000)
        assert f.below == (1000, pytest.approx(0.4))
        assert f.above == (2000, 1.0)

    def test_reports_when_the_target_is_never_reached(self):
        sweep = self._sweep_with([(1000, 1, 9), (2000, 3, 7)])
        f = detection_floor(sweep, StallClass.HOST_SYNC, target=0.9, min_support=5)
        assert f.floor_us is None
        assert "never reached" in f.note

    def test_flags_an_unbracketed_floor(self):
        sweep = self._sweep_with([(1000, 10, 0), (2000, 10, 0)])
        f = detection_floor(sweep, StallClass.HOST_SYNC, target=0.9, min_support=5)
        assert f.floor_us == 1000
        assert "not bracketed" in f.note

    def test_low_support_points_are_dropped(self):
        sweep = self._sweep_with([(1000, 1, 0), (2000, 10, 0)])
        f = detection_floor(sweep, StallClass.HOST_SYNC, target=0.9, min_support=5)
        assert [m for m, _, _ in f.curve] == [2000]

    def test_no_usable_points_is_reported_not_guessed(self):
        sweep = self._sweep_with([(1000, 1, 0)])
        f = detection_floor(sweep, StallClass.HOST_SYNC, target=0.9, min_support=5)
        assert f.floor_us is None and "support" in f.note


class TestSweep:
    def _demo_sweep(self):
        sweep = Sweep(granularity=Granularity.COARSE)
        for mag in (2000, 6000):
            for kind in ("host_sync", "h2d"):
                doc = build_scenario(kind, steps=8, stall_us=mag)
                view, labels = load_trace_dict(doc)
                sweep.add(
                    score_attribution(attribute(view, labels=labels), name=f"{kind}@{mag}")
                )
        doc = build_scenario("clean", steps=8)
        view, labels = load_trace_dict(doc)
        sweep.add(score_attribution(attribute(view, labels=labels), name="control"))
        return sweep

    def test_control_and_treatment_are_separated(self):
        sweep = self._demo_sweep()
        assert len(sweep.control_runs()) == 1
        assert len(sweep.treatment_runs()) == 4

    def test_shift_moves_toward_the_injected_class(self):
        sweep = self._demo_sweep()
        shift = sweep.shift()
        assert shift[StallClass.HOST_SYNC][2] > 0
        assert shift[StallClass.DATALOADER][2] < 0

    def test_pooled_matrix_sums_every_run(self):
        sweep = self._demo_sweep()
        pooled = sweep.pooled()
        assert pooled.total == sum(r.matrix.total for r in sweep.runs)

    def test_summary_is_json_safe(self):
        import json

        json.dumps(self._demo_sweep().summary())
