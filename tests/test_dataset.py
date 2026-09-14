"""Packaging injected runs into a releasable dataset."""

from __future__ import annotations

import gzip
import hashlib
import json

import pytest

from whystall.cli import main
from whystall.dataset import CALIBRATION, TEST, pack
from whystall.trace.load import load_trace
from whystall.trace.schema import LABEL_PREFIX
from whystall.validate.firewall import scan


@pytest.fixture
def runs(tmp_path):
    """A small sweep on disk: three classes, two magnitudes, plus controls."""
    src = tmp_path / "runs"
    src.mkdir()
    for kind in ("host_sync", "h2d", "other_host"):
        for mag in (2000, 6000):
            main(["synth", kind, "--out", str(src / f"{kind}-{mag}.trace.json"),
                  "--steps", "6", "--stall-us", str(mag)])
    for i in range(2):
        main(["synth", "clean", "--out", str(src / f"control-{i}.trace.json"),
              "--steps", "6", "--no-labels"])
    return src


class TestPackaging:
    def test_writes_the_expected_layout(self, runs, tmp_path):
        out = tmp_path / "ds"
        pack(runs, out)
        assert (out / "MANIFEST.json").is_file()
        assert (out / "README.md").is_file()
        assert list((out / "traces").glob("*.trace.json.gz"))

    def test_every_run_is_indexed(self, runs, tmp_path):
        out = tmp_path / "ds"
        m = pack(runs, out)
        assert m.counts["runs"] == 8
        assert m.counts["controls"] == 2
        assert set(m.counts["classes"]) == {"host_sync", "h2d_copy", "other_host"}

    def test_traces_are_readable_after_packing(self, runs, tmp_path):
        out = tmp_path / "ds"
        m = pack(runs, out)
        packed = out / m.runs[0].trace
        view, labels = load_trace(packed)
        assert view.gpu

    def test_ground_truth_ships_inside_the_traces(self, runs, tmp_path):
        """The whole point of releasing them. Without labels it is just traces."""
        out = tmp_path / "ds"
        m = pack(runs, out)
        labelled = [r for r in m.runs if r.stall_class]
        raw = gzip.decompress((out / labelled[0].trace).read_bytes()).decode("utf-8")
        assert LABEL_PREFIX in raw

    def test_loading_a_packed_trace_still_strips_ground_truth(self, runs, tmp_path):
        """Shipping labels must not weaken the firewall for whoever reads them."""
        out = tmp_path / "ds"
        m = pack(runs, out)
        labelled = [r for r in m.runs if r.stall_class]
        view, labels = load_trace(out / labelled[0].trace)
        assert labels, "labels should be extracted"
        assert scan(view) == [], "and removed from the view"

    def test_checksums_match_the_source_bytes(self, runs, tmp_path):
        out = tmp_path / "ds"
        m = pack(runs, out)
        entry = m.runs[0]
        source = runs / f"{entry.name}.trace.json"
        assert hashlib.sha256(source.read_bytes()).hexdigest() == entry.sha256
        assert source.stat().st_size == entry.bytes

    def test_repacking_is_byte_identical(self, runs, tmp_path):
        """mtime=0 in the gzip header, so a rebuild is verifiable."""
        a, b = tmp_path / "a", tmp_path / "b"
        pack(runs, a)
        pack(runs, b)
        name = sorted(p.name for p in (a / "traces").iterdir())[0]
        assert (a / "traces" / name).read_bytes() == (b / "traces" / name).read_bytes()

    def test_manifest_is_json_serialisable(self, runs, tmp_path):
        out = tmp_path / "ds"
        pack(runs, out)
        loaded = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
        assert loaded["runs"] and "sha256" in loaded["runs"][0]

    def test_labelled_gaps_never_exceeds_gaps(self, runs, tmp_path):
        """The card once read '177 idle windows, 240 of them labelled'.

        labelled_gaps was counting injected regions, not gaps a region resolves
        to, which made the headline count arithmetically impossible.
        """
        m = pack(runs, tmp_path / "ds")
        for r in m.runs:
            assert r.labelled_gaps <= r.gaps, r.name
        assert m.counts["labelled_gaps"] <= m.counts["gaps"]

    def test_injected_regions_is_reported_separately(self, runs, tmp_path):
        """Regions exceed labelled gaps when the prefetch absorbs a stall."""
        m = pack(runs, tmp_path / "ds")
        assert m.counts["injected_regions"] >= m.counts["labelled_gaps"]
        card = (tmp_path / "ds" / "README.md").read_text(encoding="utf-8")
        assert "injected regions were emitted" in card

    def test_the_card_is_ascii(self, runs, tmp_path):
        """It is read by strangers on unknown terminals."""
        out = tmp_path / "ds"
        pack(runs, out)
        card = (out / "README.md").read_text(encoding="utf-8")
        assert card.isascii(), [c for c in card if not c.isascii()][:5]

    def test_the_card_lists_only_directories_that_exist(self, runs, tmp_path):
        out = tmp_path / "ds"
        pack(runs, out)
        card = (out / "README.md").read_text(encoding="utf-8")
        assert "traces/" in card
        assert "samples/" not in card, "no sidecar logs were packed"
        assert "manifests/" not in card

    def test_rejects_a_directory_with_no_traces(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            pack(empty, tmp_path / "ds")

    def test_reports_and_manifests_are_not_packed_as_traces(self, runs, tmp_path):
        (runs / "x.manifest.json").write_text("{}", encoding="utf-8")
        (runs / "demo-report.json").write_text("{}", encoding="utf-8")
        m = pack(runs, tmp_path / "ds")
        assert m.counts["runs"] == 8


class TestSplit:
    def test_both_splits_are_populated(self, runs, tmp_path):
        m = pack(runs, tmp_path / "ds")
        assert m.counts[CALIBRATION] > 0
        assert m.counts[TEST] > 0

    def test_a_lone_run_in_its_group_goes_to_test(self, tmp_path):
        """Calibrating on the only evidence for a class is worse than not."""
        src = tmp_path / "runs"
        src.mkdir()
        main(["synth", "host_sync", "--out", str(src / "only.trace.json"),
              "--steps", "6", "--stall-us", "5000"])
        m = pack(src, tmp_path / "ds")
        assert m.runs[0].split == TEST

    def test_the_split_is_deterministic(self, runs, tmp_path):
        a = pack(runs, tmp_path / "a")
        b = pack(runs, tmp_path / "b")
        assert {r.name: r.split for r in a.runs} == {r.name: r.split for r in b.runs}

    def test_classes_are_spread_across_both_splits(self, runs, tmp_path):
        """Grouping on (class, magnitude) left calibration empty on a real sweep.

        The recommended sweep runs one repeat per dial setting, so every group
        was a singleton and everything landed in test.
        """
        m = pack(runs, tmp_path / "ds")
        by_split = {CALIBRATION: set(), TEST: set()}
        for r in m.runs:
            if r.stall_class:
                by_split[r.split].add(r.stall_class)
        assert by_split[CALIBRATION] == by_split[TEST]

    def test_controls_reach_both_splits(self, runs, tmp_path):
        """Precision is uninterpretable in a split with no control runs."""
        m = pack(runs, tmp_path / "ds")
        controls = {r.split for r in m.runs if r.stall_class is None}
        assert controls == {CALIBRATION, TEST}

    def test_one_run_per_dial_setting_still_splits(self, tmp_path):
        """The shape docs/sweep.md actually produces: no repeats."""
        src = tmp_path / "runs"
        src.mkdir()
        for mag in (500, 1500, 4000, 10000):
            main(["synth", "host_sync", "--out", str(src / f"hs-{mag}.trace.json"),
                  "--steps", "6", "--stall-us", str(mag)])
        m = pack(src, tmp_path / "ds")
        assert m.counts[CALIBRATION] == 2
        assert m.counts[TEST] == 2

    def test_the_split_rule_is_published(self, runs, tmp_path):
        out = tmp_path / "ds"
        m = pack(runs, out)
        assert any("Split rule:" in n for n in m.notes)
        assert "dealt alternately" in (out / "README.md").read_text(encoding="utf-8")


class TestBaselineAndCard:
    def test_baseline_is_scored_on_the_test_split_only(self, runs, tmp_path):
        m = pack(runs, tmp_path / "ds")
        assert m.baseline["split"] == TEST
        assert m.baseline["accuracy"] is not None

    def test_baseline_names_how_to_reproduce_it(self, runs, tmp_path):
        m = pack(runs, tmp_path / "ds")
        assert "whystall validate" in m.baseline["reproduce"]

    def test_card_states_synthetic_provenance(self, runs, tmp_path):
        out = tmp_path / "ds"
        m = pack(runs, out)
        card = (out / "README.md").read_text(encoding="utf-8")
        assert m.hardware is False
        assert "not a measurement of any real system" in card

    def test_card_warns_loudly_on_mixed_provenance(self, runs, tmp_path):
        """Synthetic and hardware traces must never be quietly combined."""
        doc = json.loads((runs / "h2d-2000.trace.json").read_text(encoding="utf-8"))
        doc["whystall"]["synthetic"] = False
        (runs / "h2d-2000.trace.json").write_text(json.dumps(doc), encoding="utf-8")
        out = tmp_path / "ds"
        m = pack(runs, out)
        assert "MIXED PROVENANCE" in " ".join(m.notes)
        assert "MIXED PROVENANCE" in (out / "README.md").read_text(encoding="utf-8")

    def test_card_explains_that_magnitude_is_not_ground_truth(self, runs, tmp_path):
        out = tmp_path / "ds"
        pack(runs, out)
        card = (out / "README.md").read_text(encoding="utf-8")
        assert "injected magnitude is not the answer" in card
        assert "Unlabelled gaps are real" in card

    def test_card_states_the_precision_caveat(self, runs, tmp_path):
        out = tmp_path / "ds"
        pack(runs, out)
        assert "lower bound" in (out / "README.md").read_text(encoding="utf-8")


class TestPackCLI:
    def test_pack_command_runs(self, runs, tmp_path, capsys):
        out = tmp_path / "ds"
        assert main(["pack", str(runs), "--out", str(out)]) == 0
        printed = capsys.readouterr().out
        assert "packed 8 runs" in printed
        assert "calibration" in printed
        assert (out / "MANIFEST.json").is_file()

    def test_pack_honours_the_licence_flag(self, runs, tmp_path):
        out = tmp_path / "ds"
        main(["pack", str(runs), "--out", str(out), "--license", "MIT"])
        assert json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))["license"] == "MIT"

    def test_packed_dataset_can_be_validated_directly(self, runs, tmp_path, capsys):
        out = tmp_path / "ds"
        main(["pack", str(runs), "--out", str(out)])
        assert main(["validate", str(out / "traces")]) == 0
        assert "truth \\ pred" in capsys.readouterr().out
