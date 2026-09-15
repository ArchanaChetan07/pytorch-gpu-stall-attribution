"""CLI commands, report rendering, and the injectors that do not need a GPU."""

from __future__ import annotations

import json

import pytest

from whystall.analysis import attribute, attribute_trace
from whystall.cli import main
from whystall.inject import DataloaderCPU, DataloaderIO, NullInjector, build, inject_label
from whystall.inject.base import LabelScope, monotonic_ns
from whystall.report.html import render, write_report
from whystall.report.text import render_attribution, render_sweep
from whystall.synth import build_scenario
from whystall.taxonomy import StallClass
from whystall.trace.load import load_trace_dict
from whystall.validate.floor import all_floors
from whystall.validate.score import Sweep, score_attribution


class TestCLI:
    def test_selftest_passes(self):
        assert main(["selftest"]) == 0

    def test_synth_writes_a_loadable_trace(self, tmp_path):
        out = tmp_path / "t.json"
        assert main(["synth", "host_sync", "--out", str(out), "--steps", "4"]) == 0
        att = attribute_trace(out)
        assert att.devices and att.labels

    def test_synth_no_labels_produces_a_control(self, tmp_path):
        out = tmp_path / "t.json"
        main(["synth", "dataloader", "--out", str(out), "--no-labels"])
        assert attribute_trace(out).labels == []

    def test_synth_rejects_an_unknown_scenario(self, tmp_path):
        with pytest.raises(ValueError, match="unknown scenario"):
            main(["synth", "nonsense", "--out", str(tmp_path / "x.json")])

    def test_analyze_writes_html_and_json(self, tmp_path, capsys):
        trace = tmp_path / "t.json"
        main(["synth", "dataloader", "--out", str(trace), "--steps", "6"])
        report = tmp_path / "r.html"
        assert main(["analyze", str(trace), "--out", str(report)]) == 0
        assert report.exists() and report.with_suffix(".json").exists()
        payload = json.loads(report.with_suffix(".json").read_text(encoding="utf-8"))
        assert payload["attribution"]["devices"][0]["by_class_ns"]
        assert "utilisation" in capsys.readouterr().out

    def test_analyze_reports_missing_clock_sync_rather_than_guessing(self, tmp_path, capsys):
        trace = tmp_path / "t.json"
        main(["synth", "dataloader", "--out", str(trace), "--steps", "4"])
        samples = tmp_path / "s.jsonl"
        samples.write_text('{"meta": {"hz": 200}}\n', encoding="utf-8")
        assert main(["analyze", str(trace), "--samples", str(samples)]) == 2
        assert "clock sync point" in capsys.readouterr().err

    def test_validate_scores_a_directory(self, tmp_path, capsys):
        for kind in ("host_sync", "h2d"):
            main(["synth", kind, "--out", str(tmp_path / f"{kind}.trace.json"),
                  "--steps", "8", "--stall-us", "6000"])
        report = tmp_path / "v.html"
        assert main(["validate", str(tmp_path), "--out", str(report)]) == 0
        out = capsys.readouterr().out
        assert "truth \\ pred" in out
        assert report.exists()

    def test_validate_errors_when_nothing_matches(self, tmp_path, capsys):
        assert main(["validate", str(tmp_path / "nope")]) == 2
        assert "no traces found" in capsys.readouterr().err

    def test_demo_runs_the_whole_pipeline(self, tmp_path, capsys):
        assert main(["demo", "--out", str(tmp_path), "--steps", "6"]) == 0
        out = capsys.readouterr().out
        assert "detection floors" in out
        assert "synthetic" in out
        assert (tmp_path / "demo-report.html").exists()

    def test_global_threshold_flags_reach_the_classifier(self, tmp_path, capsys):
        trace = tmp_path / "t.json"
        main(["synth", "launch", "--out", str(trace), "--steps", "4", "--stall-us", "34"])
        main(["--min-gap-us", "5000", "analyze", str(trace)])
        assert "min_gap_us=5000" in capsys.readouterr().out


class TestHTMLReport:
    def _sweep(self):
        sweep = Sweep()
        for kind in ("host_sync", "h2d"):
            for mag in (2000, 6000):
                view, labels = load_trace_dict(
                    build_scenario(kind, steps=6, stall_us=mag)
                )
                sweep.add(score_attribution(attribute(view, labels=labels),
                                            name=f"{kind}@{mag}"))
        view, labels = load_trace_dict(build_scenario("clean", steps=6))
        sweep.add(score_attribution(attribute(view, labels=labels), name="control"))
        return sweep

    def test_renders_a_complete_document(self):
        view, labels = load_trace_dict(build_scenario("dataloader", steps=6, stall_us=5000))
        html = render(attribute(view, labels=labels))
        assert html.startswith("<!doctype html>")
        assert html.rstrip().endswith("</html>")
        assert html.count("<table") == html.count("</table>")

    def test_defines_both_themes_at_token_level(self):
        """A colour defined only inside a media query breaks the unstamped case."""
        html = render(_attributed("clean"))
        assert "prefers-color-scheme:dark" in html
        assert ':root[data-theme="dark"]' in html
        assert ":root{" in html

    def test_wide_tables_scroll_inside_their_own_container(self):
        html = render(_attributed("dataloader"))
        assert "overflow-x:auto" in html

    def test_unattributed_share_is_shown(self):
        html = render(_attributed("dataloader"))
        assert "unattributed" in html.lower()

    def test_validation_section_states_the_precision_caveat(self):
        html = render(None, sweep=self._sweep(), floors=[])
        assert "lower bound" in html
        assert "control" in html.lower()

    def test_control_shift_table_appears(self):
        html = render(None, sweep=self._sweep(), floors=[])
        assert "Control versus treatment" in html

    def test_warns_loudly_when_a_sweep_has_no_control(self):
        sweep = Sweep()
        view, labels = load_trace_dict(build_scenario("h2d", steps=6, stall_us=5000))
        sweep.add(score_attribution(attribute(view, labels=labels), name="only"))
        html = render(None, sweep=sweep, floors=[])
        assert "No control runs" in html

    def test_synthetic_traces_carry_a_provenance_banner(self):
        """A report travels without the terminal that produced it."""
        html = render(_attributed("dataloader"))
        assert "synthetic traces" in html
        assert "never be quoted" in html

    def test_explicit_provenance_overrides_detection(self):
        html = render(_attributed("clean"), provenance="collected on an H100, cold cache")
        assert "collected on an H100" in html

    def test_demo_report_says_it_is_synthetic(self, tmp_path):
        main(["demo", "--out", str(tmp_path), "--steps", "6"])
        html = (tmp_path / "demo-report.html").read_text(encoding="utf-8")
        assert "must never be quoted as a hardware result" in html

    def test_escapes_content(self):
        view, labels = load_trace_dict(build_scenario("clean", steps=3))
        html = render(attribute(view, labels=labels), title="<script>x</script>")
        assert "<script>x</script>" not in html
        assert "&lt;script&gt;" in html

    def test_write_report_emits_both_files(self, tmp_path):
        sweep = self._sweep()
        p = write_report(tmp_path / "r.html", None, sweep=sweep,
                         floors=all_floors(sweep, min_support=3))
        assert p.exists()
        payload = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        assert "validation" in payload


class TestTextReport:
    def test_attribution_summary_lists_every_class(self):
        text = render_attribution(_attributed("host_sync"))
        assert "utilisation" in text and "unattributed" in text

    def test_sweep_summary_includes_the_control_shift(self):
        sweep = Sweep()
        for kind in ("host_sync",):
            view, labels = load_trace_dict(build_scenario(kind, steps=6, stall_us=6000))
            sweep.add(score_attribution(attribute(view, labels=labels), name=kind))
        view, labels = load_trace_dict(build_scenario("clean", steps=6))
        sweep.add(score_attribution(attribute(view, labels=labels), name="control"))
        text = render_sweep(sweep, all_floors(sweep, min_support=3))
        assert "control vs treatment" in text


class TestInjectors:
    def test_registry_builds_every_named_injector(self):
        for name in ("dataloader_io", "dataloader_cpu", "h2d", "host_sync",
                     "collective", "launch", "other_host", "device_dependency"):
            inj = build(name, 1000)
            assert inj.name == name
            assert inj.stall_class is not StallClass.UNATTRIBUTED

    def test_control_builds_and_injects_nothing(self):
        inj = build("control", 0)
        assert isinstance(inj, NullInjector)
        assert inj.label_scope == LabelScope.NONE
        sentinel = object()
        assert inj.wrap_dataset(sentinel) is sentinel

    def test_unknown_injector_names_the_alternatives(self):
        with pytest.raises(ValueError, match="unknown injector"):
            build("teleporter", 10)

    def test_every_injector_describes_itself_for_the_manifest(self):
        d = build("dataloader_io", 4000).describe()
        assert d == {
            "name": "dataloader_io",
            "class": "dataloader_io",
            "magnitude_us": 4000,
            "label_scope": "fetch",
        }

    def test_io_injector_delays_item_reads(self):
        import time

        inner = list(range(4))
        ds = DataloaderIO(magnitude_us=3000, jitter_frac=0.0).wrap_dataset(inner)
        start = time.perf_counter()
        assert ds[2] == 2
        assert (time.perf_counter() - start) >= 0.002

    def test_cpu_injector_burns_time_without_sleeping(self):
        """Sleeping would look exactly like I/O to the sampler."""
        import time

        ds = DataloaderCPU(magnitude_us=3000).wrap_dataset(list(range(4)))
        start = time.perf_counter()
        ds[1]
        assert (time.perf_counter() - start) >= 0.001

    def test_wrapped_dataset_preserves_length_and_attributes(self):
        class Inner(list):
            classes = ("a", "b")

        ds = DataloaderCPU(magnitude_us=0).wrap_dataset(Inner([1, 2, 3]))
        assert len(ds) == 3
        assert ds.classes == ("a", "b")

    def test_h2d_injector_turns_off_pinned_memory(self):
        kwargs = build("h2d", 1000).wrap_loader_kwargs({"batch_size": 8, "pin_memory": True})
        assert kwargs["pin_memory"] is False
        assert kwargs["batch_size"] == 8

    def test_label_context_is_inert_without_torch(self):
        with inject_label(StallClass.HOST_SYNC, 1234):
            pass

    def test_monotonic_clock_is_monotonic(self):
        assert monotonic_ns() <= monotonic_ns()


def _attributed(kind: str, **kwargs):
    view, labels = load_trace_dict(build_scenario(kind, steps=6, stall_us=5000, **kwargs))
    return attribute(view, labels=labels)
