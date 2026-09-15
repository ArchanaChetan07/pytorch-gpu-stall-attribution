"""Package injected runs into a releasable, scorable dataset.

The traces are arguably a longer-lived artifact than the classifier that
produced them. A rule set can be rebuilt in a week by anyone who reads the
paper; a corpus of GPU traces with per-gap ground truth cannot be reconstructed
without redoing the injection work, and it lets a competing classifier be scored
on identical inputs instead of on its author's own runs.

Three decisions shape what comes out:

Ground truth ships inside the traces.
    The WHYSTALL annotations stay. That is the whole point of releasing them,
    and it does not weaken the firewall: stripping happens at load time, so a
    scorer reading these files is protected by the same mechanism we are.

The split is by run, and it is balanced rather than random.
    Anyone calibrating a threshold needs somewhere to calibrate that is not
    where they report. Runs are grouped by class and dealt alternately, so both
    halves see every class -- with the handful of runs a sweep produces, random
    assignment would routinely leave a class absent from one side. See
    _assign_splits for why the grouping is on class alone.

Provenance is stamped everywhere, not just in the README.
    Synthetic traces and hardware traces must never be confusable, so the flag
    rides in the manifest, per run, and the dataset card refuses to describe a
    mixed package as a measurement.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .analysis import attribute
from .trace.load import load_trace
from .validate.groundtruth import Granularity, injected_regions, label_gaps
from .validate.score import Sweep, score_attribution

__all__ = ["RunEntry", "DatasetManifest", "pack", "CALIBRATION", "TEST"]

CALIBRATION = "calibration"
TEST = "test"

_CHUNK = 1 << 20


@dataclass
class RunEntry:
    """One run in the package."""

    name: str
    split: str
    trace: str
    samples: str | None
    manifest: str | None
    #: Ground-truth class dominating this run, or None for a control.
    stall_class: str | None
    magnitude_us: int
    synthetic: bool
    gaps: int
    #: Idle windows a ground-truth region actually resolves to. Always <= gaps.
    #: Distinct from injected_regions: the injector marks a region per step,
    #: and a region only labels a gap if the injection produced one.
    labelled_gaps: int
    #: Annotations the injector emitted. Typically one per step, so normally
    #: larger than labelled_gaps -- a stall the prefetch absorbed leaves a
    #: region with no gap to label.
    injected_regions: int
    idle_ns: int
    sha256: str
    bytes: int


@dataclass
class DatasetManifest:
    name: str
    version: str
    created: str
    license: str
    #: True only if every run in the package came from real hardware.
    hardware: bool
    counts: dict[str, Any] = field(default_factory=dict)
    runs: list[RunEntry] = field(default_factory=list)
    baseline: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["runs"] = [asdict(r) if not isinstance(r, dict) else r for r in self.runs]
        return d


def _sha256_and_size(path: Path) -> tuple[str, int]:
    """Checksum the source bytes, not the archive.

    Compression settings are not part of the data, so a package recompressed
    with a different gzip level must still verify.
    """
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def _gzip_into(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0 so repacking the same inputs produces identical bytes.
    with (
        src.open("rb") as fin,
        dest.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as fout,
    ):
        shutil.copyfileobj(fin, fout, _CHUNK)


def _stem(path: Path) -> str:
    name = path.name
    for suffix in (".trace.json.gz", ".trace.json", ".json.gz", ".json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


#: Stated in the manifest and the dataset card, because a split nobody can
#: describe is a split nobody will respect.
SPLIT_RULE = (
    "Runs are grouped by injected class (controls forming their own group), "
    "ordered by magnitude then name, and dealt alternately between test and "
    "calibration. A class represented by a single run goes to test."
)


def _assign_splits(rows: list[dict[str, Any]]) -> None:
    """Deal each class alternately between the two splits.

    Grouped by class and not by (class, magnitude): a sweep runs one repeat per
    dial setting, so grouping on both would make every group a singleton and
    leave calibration empty -- which is what the recommended sweep in
    docs/sweep.md would have produced.

    Dealing magnitudes alternately within a class is also the more demanding
    arrangement. Thresholds here are magnitude-independent by construction, so
    calibrating at some dial settings and reporting at others tests that they
    generalise rather than merely that they fit.

    Controls are dealt the same way, because a split without controls cannot
    interpret precision at all.
    """
    groups: dict[str | None, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["stall_class"], []).append(row)
    for key in sorted(groups, key=lambda k: str(k)):
        members = sorted(groups[key], key=lambda r: (r["magnitude_us"], r["name"]))
        for i, row in enumerate(members):
            # A lone run for a class goes to test: a calibration set that is the
            # only evidence for a class is worse than no calibration set.
            row["split"] = TEST if len(members) == 1 else (CALIBRATION if i % 2 else TEST)


def pack(
    source: str | os.PathLike[str],
    out_dir: str | os.PathLike[str],
    *,
    name: str = "whystall-stalls",
    version: str = "v1",
    license_id: str = "CC-BY-4.0",
    granularity: str = Granularity.COARSE,
) -> DatasetManifest:
    """Build a releasable dataset from a directory of injected runs."""
    src = Path(source)
    out = Path(out_dir)
    if not src.is_dir():
        raise NotADirectoryError(f"{src} is not a directory of runs")

    traces = sorted(
        p
        for p in src.iterdir()
        if p.is_file()
        and (p.name.endswith(".json") or p.name.endswith(".json.gz"))
        and not p.name.endswith((".manifest.json", ".report.json", "-report.json"))
    )
    if not traces:
        raise FileNotFoundError(f"no traces found in {src}")

    (out / "traces").mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    any_synthetic = False
    any_hardware = False

    for trace_path in traces:
        stem = _stem(trace_path)
        view, labels = load_trace(trace_path)
        att = attribute(view, labels=labels)
        regions = injected_regions(labels)

        dominant = None
        magnitude = 0
        if regions:
            by_class: dict[str, int] = {}
            for r in regions:
                by_class[r.cls.value] = by_class.get(r.cls.value, 0) + (r.end_ns - r.start_ns)
            dominant = max(by_class, key=lambda k: by_class[k])
            mags = [r.magnitude_us for r in regions if r.cls.value == dominant]
            magnitude = max(set(mags), key=mags.count) if mags else 0

        meta = view.meta.get("whystall")
        synthetic = bool(isinstance(meta, dict) and meta.get("synthetic"))
        any_synthetic |= synthetic
        any_hardware |= not synthetic

        digest, size = _sha256_and_size(trace_path)
        packed = f"traces/{stem}.trace.json.gz"
        _gzip_into(trace_path, out / packed)

        samples_rel = None
        samples_src = src / f"{stem}.samples.jsonl"
        if samples_src.is_file():
            samples_rel = f"samples/{stem}.samples.jsonl.gz"
            _gzip_into(samples_src, out / samples_rel)

        manifest_rel = None
        manifest_src = src / f"{stem}.manifest.json"
        if manifest_src.is_file():
            manifest_rel = f"manifests/{stem}.manifest.json"
            (out / manifest_rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(manifest_src, out / manifest_rel)

        rows.append(
            {
                "name": stem,
                "split": TEST,
                "trace": packed,
                "samples": samples_rel,
                "manifest": manifest_rel,
                "stall_class": dominant,
                "magnitude_us": magnitude,
                "synthetic": synthetic,
                "gaps": sum(len(d.gapset.gaps) for d in att.devices),
                "labelled_gaps": sum(
                    1
                    for d in att.devices
                    for truth in label_gaps(d.blames, regions)
                    if truth is not None
                ),
                "injected_regions": len(regions),
                "idle_ns": sum(d.classified_ns for d in att.devices),
                "sha256": digest,
                "bytes": size,
            }
        )

    _assign_splits(rows)
    entries = [RunEntry(**row) for row in rows]

    baseline = _baseline_scores(src, entries, granularity=granularity)

    notes: list[str] = []
    if any_synthetic and any_hardware:
        notes.append(
            "MIXED PROVENANCE: this package contains both synthetic and hardware "
            "traces. Filter on the per-run synthetic flag before quoting any number."
        )
    elif any_synthetic:
        notes.append(
            "Every trace here is synthetic, generated by whystall.synth. The package "
            "is valid for comparing classifiers against each other on identical "
            "inputs; it is not a measurement of any real system."
        )
    notes.append(
        "Ground-truth annotations are present inside the traces, prefixed WHYSTALL/. "
        "Strip them before classification -- whystall.trace.load.load_trace does it "
        "structurally, and whystall.validate.firewall asserts it."
    )
    notes.append("Split rule: " + SPLIT_RULE)

    manifest = DatasetManifest(
        name=name,
        version=version,
        created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        license=license_id,
        hardware=any_hardware and not any_synthetic,
        counts={
            "runs": len(entries),
            "calibration": sum(1 for e in entries if e.split == CALIBRATION),
            "test": sum(1 for e in entries if e.split == TEST),
            "controls": sum(1 for e in entries if e.stall_class is None),
            "gaps": sum(e.gaps for e in entries),
            "labelled_gaps": sum(e.labelled_gaps for e in entries),
            "injected_regions": sum(e.injected_regions for e in entries),
            "classes": sorted({e.stall_class for e in entries if e.stall_class}),
        },
        runs=entries,
        baseline=baseline,
        notes=notes,
    )

    (out / "MANIFEST.json").write_text(
        json.dumps(manifest.to_dict(), indent=2), encoding="utf-8"
    )
    present = {d for d in ("traces", "samples", "manifests") if (out / d).is_dir()}
    (out / "README.md").write_text(_dataset_card(manifest, present), encoding="utf-8")
    return manifest


def _baseline_scores(
    src: Path, entries: list[RunEntry], *, granularity: str
) -> dict[str, Any]:
    """Score the test split with this package's own classifier.

    Shipped so a competing classifier has something to beat on identical inputs.
    Scored on the test split only, for the same reason anyone else should.
    """
    sweep = Sweep(granularity=granularity)
    for entry in entries:
        if entry.split != TEST:
            continue
        candidates = [src / f"{entry.name}.trace.json", src / f"{entry.name}.json"]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            continue
        view, labels = load_trace(path)
        sweep.add(
            score_attribution(
                attribute(view, labels=labels), granularity=granularity, name=entry.name
            )
        )
    if not sweep.runs:
        return {}
    pooled = sweep.pooled()
    return {
        "tool": "whystall",
        "split": TEST,
        "granularity": granularity,
        "accuracy": pooled.to_dict()["accuracy"],
        "macro_recall": pooled.to_dict()["macro_recall"],
        "unattributed_share": round(sweep.unattributed_share(), 6),
        "per_class": pooled.to_dict()["per_class"],
        "top_confusions": pooled.to_dict()["top_confusions"],
        "reproduce": "whystall validate <dataset>/traces --granularity " + granularity,
    }


def _dataset_card(m: DatasetManifest, present: set[str] | None = None) -> str:
    """A dataset card, in the shape people expect to be able to cite.

    ASCII only. This file is read by strangers on unknown terminals, and a
    mojibake em dash in the first screen is a bad first impression for a project
    whose pitch is carefulness.
    """
    c = m.counts
    present = present if present is not None else {"traces"}
    classes = ", ".join(f"`{x}`" for x in c.get("classes", [])) or "none"
    lines = [
        f"# {m.name} {m.version}",
        "",
        "GPU training traces with **per-gap ground truth** for stall attribution:",
        "each trace carries annotations recording which stall class was deliberately",
        "injected and at what magnitude, so a classifier can be scored per idle",
        "window rather than in aggregate.",
        "",
    ]
    for note in m.notes:
        lines += [f"> **{note}**", ""]
    lines += [
        "## What is in it",
        "",
        f"- **{c['runs']} runs**, {c['controls']} of them controls with nothing injected.",
        f"- **{c['gaps']:,} classified idle windows**, of which "
        f"**{c['labelled_gaps']:,} carry a ground-truth label**.",
        f"- {c.get('injected_regions', 0):,} injected regions were emitted. That is the"
        " larger number on purpose:",
        "  an injector marks a region every step, and a stall the prefetch absorbed",
        "  leaves a region with no idle window to label.",
        f"- Classes present: {classes}.",
        f"- Splits: **{c['calibration']} calibration**, **{c['test']} test**.",
        "",
        "```",
        "MANIFEST.json   every run: split, injected class, magnitude, provenance, sha256",
        "traces/         the traces, gzipped, ground truth included",
    ]
    if "samples" in present:
        lines.append("samples/        /proc sidecar logs, where one was running")
    if "manifests" in present:
        lines.append("manifests/      per-run collection metadata: host, cache state, settings")
    lines += [
        "```",
        "",
        "## How to use it",
        "",
        "Calibrate any threshold on the **calibration** split and report on **test**.",
        "Reporting a number tuned on the test split is the failure this split exists",
        "to prevent.",
        "",
        f"The split is by run: {SPLIT_RULE[0].lower() + SPLIT_RULE[1:]} Both halves",
        "therefore see every class, at different dial settings, which also tests that",
        "a threshold generalises across magnitudes rather than merely fitting one.",
        "",
        "```bash",
        "pip install whystall",
        f"whystall validate traces/ --granularity {m.baseline.get('granularity', 'coarse')}"
        " --out report.html",
        "```",
        "",
        "## Ground truth",
        "",
        "Annotations are `record_function` ranges named",
        "`WHYSTALL/inject/<class>/<magnitude_us>`, emitted by the injector around the",
        "region it perturbed. A region labels a gap when it overlaps either the blame",
        "window -- the interval on the launching thread the gap is charged against --",
        "or the gap itself, whichever overlap is larger.",
        "",
        "**The injected magnitude is not the answer.** Prefetch depth absorbs part of",
        "any injection, so 5 ms of delay routinely yields 2 ms of idle. Ground truth is",
        "the labelled gap; the magnitude is the sweep coordinate.",
        "",
        "**Unlabelled gaps are real.** A run stalls for reasons the injection did not",
        "create. They are kept, and a scorer must keep them too: dropping them is the",
        "easiest way to publish a recall number that does not replicate.",
        "",
    ]
    if m.baseline:
        b = m.baseline
        lines += [
            "## Baseline",
            "",
            f"`whystall` on the test split, `{b['granularity']}` granularity:",
            "",
            f"- accuracy **{b['accuracy']}**",
            f"- macro recall **{b['macro_recall']}**",
            f"- unattributed share **{b['unattributed_share']}**",
            "",
            "Shipped so a competing classifier has something to beat on identical",
            "inputs. Reproduce with:",
            "",
            "```bash",
            f"{b['reproduce']}",
            "```",
            "",
        ]
    lines += [
        "## Limits",
        "",
        "- **Precision against this ground truth is a lower bound.** A correct verdict",
        "  on a stall the injection did not create scores as a false positive, because",
        "  nothing labelled it. Compare control runs against treatment runs instead.",
        "- **One injector per run.** Simultaneous causes are not represented unless the",
        "  manifest says otherwise.",
        "- **The portable I/O injector sleeps** rather than throttling the device, so",
        "  it reproduces the timing of slow storage but not the kernel's queueing",
        "  behaviour or its process-state signature.",
        "",
        "## Licence",
        "",
        f"{m.license}. Cite the dataset name and version.",
        "",
    ]
    return "\n".join(lines)
