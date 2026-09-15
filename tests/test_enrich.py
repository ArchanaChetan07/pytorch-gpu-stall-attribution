"""Clock alignment, /proc parsing, and the I/O-versus-CPU decision.

None of this needs Linux. The sampler's data types are plain and its parsing is
pure, which is the point: the decision that carries the project has to be
testable on the machine the analysis is written on.
"""

from __future__ import annotations

import json

import pytest

from whystall.analysis import attribute
from whystall.enrich.clock import ClockSyncError, fit_clock, sync_label_name
from whystall.enrich.refine import RefineConfig, refine_attribution, refine_window
from whystall.enrich.sampler import (
    Sample,
    SamplerLog,
    WindowStats,
    WorkerSample,
    parse_ppid,
    parse_stat,
)
from whystall.synth import build_scenario
from whystall.taxonomy import StallClass
from whystall.trace.load import load_trace_dict
from whystall.trace.schema import Label


class TestClock:
    def test_fits_offset_and_drift(self):
        labels = [
            Label("clock", ("1000000000",), 1, 1, 500_000, 1),
            Label("clock", ("2000000000",), 1, 1, 1_500_000_000, 1),
        ]
        cm = fit_clock(labels)
        assert cm.to_trace_ns(1_000_000_000) == pytest.approx(500_000)
        assert cm.to_trace_ns(1_500_000_000) == pytest.approx(750_250_000)
        # Both clocks are nanoseconds now, so a healthy fit sits at slope 1.
        assert cm.slope == pytest.approx(1.4995, rel=1e-3)

    def test_inverse_round_trips(self):
        labels = [
            Label("clock", ("1000000000",), 1, 1, 500_000, 1),
            Label("clock", ("2000000000",), 1, 1, 1_000_500_000, 1),
        ]
        cm = fit_clock(labels)
        assert cm.to_monotonic_ns(cm.to_trace_ns(1_234_567_890)) == pytest.approx(
            1_234_567_890, rel=1e-9
        )

    def test_refuses_to_guess_with_one_point(self):
        with pytest.raises(ClockSyncError, match="at least 2"):
            fit_clock([Label("clock", ("1000",), 1, 1, 0, 1)])

    def test_refuses_to_guess_with_no_points(self):
        """Silently assuming zero offset is worse than failing loudly."""
        with pytest.raises(ClockSyncError):
            fit_clock([Label("inject", ("host_sync", "10"), 1, 1, 0, 1)])

    def test_rejects_a_fit_that_does_not_hold(self):
        labels = [
            Label("clock", ("1000000000",), 1, 1, 0, 1),
            Label("clock", ("2000000000",), 1, 1, 1_000_000_000, 1),
            Label("clock", ("3000000000",), 1, 1, 9_000_000_000, 1),
        ]
        with pytest.raises(ClockSyncError, match="residual"):
            fit_clock(labels)

    def test_sync_label_name_is_parseable_by_the_loader(self):
        name = sync_label_name(123456789)
        lab = Label.parse(name, 1, 1, 10, 1)
        assert lab is not None and lab.kind == "clock" and lab.parts == ("123456789",)


class TestProcParsing:
    def test_handles_spaces_and_parens_in_the_process_name(self):
        # state ppid pgrp session tty tpgid flags minflt cminflt majflt cmajflt
        # utime stime -- utime and stime are fields 14 and 15, index 11 and 12
        # once the parenthesised comm has been stripped.
        raw = (
            "4242 (python (worker 3)) D 4200 4242 4242 0 -1 4194560 "
            + " ".join(["0"] * 4)
            + " 137 219 "
            + " ".join(["0"] * 30)
        )
        assert parse_stat(raw) == ("D", 137, 219)
        assert parse_ppid(raw) == 4200

    def test_returns_none_on_garbage(self):
        assert parse_stat("not a stat line") is None
        assert parse_ppid("nope") is None

    def test_returns_none_on_a_truncated_line(self):
        assert parse_stat("1 (x) R 2 3") is None


def _log(samples):
    return SamplerLog(samples=list(samples))


def _sample(t_ns, states, read_bytes, utime=0):
    return Sample(
        t_mono_ns=t_ns,
        workers=tuple(
            WorkerSample(pid=i, state=s, utime=utime, stime=0, read_bytes=read_bytes, rchar=0)
            for i, s in enumerate(states)
        ),
        psi_io_some=None,
    )


class TestSamplerLog:
    def test_window_selects_by_time(self):
        log = _log([_sample(t, "RR", 0) for t in (100, 200, 300, 400)])
        assert [s.t_mono_ns for s in log.between_ns(150, 350)] == [200, 300]

    def test_read_bytes_is_differenced_across_the_window(self):
        one_second = 1_000_000_000
        log = _log([
            _sample(0, "DD", 1_000_000),
            _sample(one_second, "DD", 3_000_000),
            _sample(2 * one_second, "DD", 9_000_000),
        ])
        stats = log.stats_ns(0, 2 * one_second)
        # Two workers, each climbing 1 MB -> 9 MB over two seconds.
        assert stats.read_bytes == 2 * 8_000_000
        assert stats.read_mb_per_s == pytest.approx(8.0)
        assert stats.window_us == pytest.approx(2_000_000)

    def test_state_fractions(self):
        log = _log([_sample(0, "DDDR", 0), _sample(1_000_000, "DDRR", 0)])
        stats = log.stats_ns(0, 1_000_000)
        assert stats.frac_uninterruptible == pytest.approx(5 / 8)
        assert stats.frac_runnable == pytest.approx(3 / 8)

    def test_empty_window_reports_zero_samples_not_zero_io(self):
        """Absence of evidence must be distinguishable from evidence of absence."""
        log = _log([_sample(0, "DD", 0)])
        stats = log.stats_ns(5_000_000, 6_000_000)
        assert stats.samples == 0
        assert stats.read_bytes_unavailable

    def test_unreadable_read_bytes_is_flagged(self):
        log = _log([_sample(0, "DD", -1), _sample(1_000_000, "DD", -1)])
        assert log.stats_ns(0, 1_000_000).read_bytes_unavailable

    def test_jsonl_round_trip(self, tmp_path):
        p = tmp_path / "s.jsonl"
        with p.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({"meta": {"hz": 200, "parent": 7}}) + "\n")
            for s in (_sample(0, "DR", 10), _sample(1_000_000, "DR", 20)):
                fh.write(json.dumps(s.to_json()) + "\n")
        log = SamplerLog.load(p)
        assert len(log) == 2 and log.hz == 200
        assert log.samples[0].workers[0].state == "D"


def _stats(**kw):
    base = dict(
        samples=20, observations=160, window_us=100_000.0, frac_uninterruptible=0.0,
        frac_runnable=0.0, frac_sleeping=1.0, read_bytes=0, read_mb_per_s=0.0,
        cpu_per_worker=0.0, psi_io_some=None, read_bytes_unavailable=False,
    )
    base.update(kw)
    return WindowStats(**base)


class TestRefine:
    def test_disk_traffic_reads_as_io(self):
        out = refine_window(_stats(read_mb_per_s=180.0, frac_uninterruptible=0.9))
        assert out.cls is StallClass.DATALOADER_IO

    def test_saturated_cores_read_as_cpu(self):
        out = refine_window(_stats(cpu_per_worker=0.95, frac_runnable=0.9))
        assert out.cls is StallClass.DATALOADER_CPU

    def test_both_signals_report_mixed_rather_than_picking(self):
        out = refine_window(
            _stats(read_mb_per_s=180.0, frac_uninterruptible=0.7,
                   cpu_per_worker=0.95, frac_runnable=0.6)
        )
        assert out.cls is StallClass.DATALOADER_MIXED

    def test_neither_signal_leaves_the_class_unrefined(self):
        out = refine_window(_stats(frac_sleeping=1.0))
        assert out.cls is StallClass.DATALOADER
        assert "threshold" in out.reason

    def test_too_few_samples_refuses_to_decide(self):
        out = refine_window(_stats(samples=1, window_us=4000.0, read_mb_per_s=500.0))
        assert out.cls is StallClass.DATALOADER
        assert "sampler observations" in out.reason

    def test_the_refusal_says_what_rate_would_have_worked(self):
        """A window shorter than min_samples/hz is unrefinable, not broken."""
        out = refine_window(_stats(samples=1, window_us=4000.0))
        assert "750 Hz" in out.reason

    def test_missing_read_bytes_lowers_confidence_and_says_so(self):
        with_io = refine_window(_stats(frac_uninterruptible=0.9, read_mb_per_s=180.0))
        without = refine_window(
            _stats(frac_uninterruptible=0.9, read_bytes_unavailable=True)
        )
        assert without.cls is StallClass.DATALOADER_IO
        assert without.confidence < with_io.confidence
        assert "read_bytes unavailable" in without.evidence["caveat"]

    def test_thresholds_travel_with_the_verdict(self):
        out = refine_window(_stats(read_mb_per_s=180.0))
        assert out.evidence["thresholds"]["io_read_mb_per_s"] == 20.0

    def test_config_is_honoured(self):
        cfg = RefineConfig(io_read_mb_per_s=1000.0)
        assert refine_window(_stats(read_mb_per_s=180.0), cfg).cls is StallClass.DATALOADER


class TestRefineAttribution:
    def _prepare(self, states, read_bytes):
        """Attribute a dataloader scenario and fabricate matching sampler data."""
        doc = build_scenario("dataloader_io", steps=6, stall_us=6000)
        # Two clock sync points spanning the trace, on a 1:1000 ns->us clock.
        lo = min(e["ts"] for e in doc["traceEvents"])
        hi = max(e["ts"] + e.get("dur", 0) for e in doc["traceEvents"])
        for ts, mono in ((lo, 10_000_000_000), (hi, 10_000_000_000 + (hi - lo) * 1000)):
            doc["traceEvents"].append({
                "ph": "X", "cat": "user_annotation",
                "name": sync_label_name(mono), "pid": 1000, "tid": 1000,
                "ts": ts, "dur": 1, "args": {},
            })
        view, labels = load_trace_dict(doc)
        att = attribute(view, labels=labels)
        clock = fit_clock(labels)
        # 1 kHz. At the 200 Hz default a 4 ms gap gets under one observation and
        # nothing is refinable, which is a real property of the method and is
        # covered by test_the_refusal_says_what_rate_would_have_worked.
        period_ns = 1_000_000
        samples = [
            _sample(mono, states, read_bytes * i)
            for i, mono in enumerate(
                range(10_000_000_000, 10_000_000_000 + (hi - lo) * 1000, period_ns)
            )
        ]
        return att, SamplerLog(samples=samples), clock

    def test_io_signal_splits_dataloader_into_io(self):
        att, log, clock = self._prepare("DDDDDDDD", 40_000_000)
        n = refine_attribution(att, log, clock)
        assert n > 0
        classes = {v.cls for v in att.verdicts}
        assert StallClass.DATALOADER_IO in classes
        assert StallClass.DATALOADER not in classes

    def test_cpu_signal_splits_dataloader_into_cpu(self):
        att, log, clock = self._prepare("RRRRRRRR", 0)
        refine_attribution(att, log, clock)
        assert StallClass.DATALOADER_CPU in {v.cls for v in att.verdicts}

    def test_the_trace_only_verdict_is_preserved(self):
        att, log, clock = self._prepare("DDDDDDDD", 40_000_000)
        refine_attribution(att, log, clock)
        refined = [v for v in att.verdicts if v.cls is StallClass.DATALOADER_IO]
        assert refined
        assert refined[0].evidence["trace_only_class"] == "dataloader"
        assert refined[0].rule == "sampler_refined"

    def test_refinement_never_touches_a_non_dataloader_verdict(self):
        doc = build_scenario("host_sync", steps=6, stall_us=6000)
        view, labels = load_trace_dict(doc)
        att = attribute(view, labels=labels)
        before = [v.cls for v in att.verdicts]
        clock_labels = [
            Label("clock", ("0",), 1, 1, 0, 1),
            Label("clock", ("1000000000",), 1, 1, 1_000_000, 1),
        ]
        refine_attribution(att, SamplerLog(samples=[]), fit_clock(clock_labels))
        assert [v.cls for v in att.verdicts] == before
