"""Self-contained HTML report.

No dependencies, no external assets, no network. One file you can attach to an
issue or hand to someone who does not have the tool installed, which is most of
the audience that matters.

The layout puts the limits next to the results on purpose: the unattributed
share sits beside the class breakdown, and the confused pairs sit beside the
recall numbers, because a report that makes its own weaknesses hard to find is
the genre this project exists to argue with.
"""

from __future__ import annotations

import html
import json
import os
import time
from pathlib import Path
from typing import Any

from ..analysis import Attribution
from ..taxonomy import CLASS_INFO, StallClass
from ..validate.floor import DetectionFloor
from ..validate.matrix import ConfusionMatrix
from ..validate.score import Sweep

__all__ = ["render", "write_report"]

_CSS = """
:root{
  --ground:#F1F3F5; --surface:#FFFFFF; --surface-2:#E7EAEE;
  --ink:#12161C; --ink-2:#39424D; --muted:#69727E;
  --rule:#D4D9E0; --rule-strong:#B6BDC7;
  --accent:#A4501A; --accent-ink:#8A4315; --accent-soft:#F3E3D6;
  --warn:#96302E; --warn-soft:#F6E0DF; --good:#2F6B4F;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#0E1115; --surface:#151920; --surface-2:#1D232B;
    --ink:#E7EAEF; --ink-2:#B2BAC5; --muted:#818B97;
    --rule:#282F38; --rule-strong:#3A434E;
    --accent:#E28D50; --accent-ink:#F0A971; --accent-soft:#2C2018;
    --warn:#E08B87; --warn-soft:#2E1A19; --good:#7FBF9C;
  }
}
:root[data-theme="dark"]{
  --ground:#0E1115; --surface:#151920; --surface-2:#1D232B;
  --ink:#E7EAEF; --ink-2:#B2BAC5; --muted:#818B97;
  --rule:#282F38; --rule-strong:#3A434E;
  --accent:#E28D50; --accent-ink:#F0A971; --accent-soft:#2C2018;
  --warn:#E08B87; --warn-soft:#2E1A19; --good:#7FBF9C;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font:15px/1.6 "IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif}
.wrap{max-width:1040px;margin:0 auto;padding:0 24px 80px}
h1,h2,h3{margin:0;font-weight:600;letter-spacing:-.01em;text-wrap:balance}
h1{font-size:34px}
h2{font-size:21px}
h3{font-size:15px}
code,.mono{font-family:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,monospace}
header{border-bottom:1px solid var(--rule-strong);padding:44px 0 24px}
header .rule{width:56px;height:4px;background:var(--accent);margin-bottom:20px}
header .meta{display:flex;flex-wrap:wrap;gap:6px 26px;margin-top:16px;
  font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--muted)}
header .meta b{color:var(--ink-2);font-weight:500}
section{padding-top:44px}
section>h2{border-bottom:1px solid var(--rule);padding-bottom:10px;margin-bottom:20px}
p{margin:0 0 14px;color:var(--ink-2)}
.cards{display:flex;flex-wrap:wrap;gap:14px;margin:0 0 22px}
.card{flex:1 1 170px;background:var(--surface);border:1px solid var(--rule);
  border-radius:3px;padding:16px 18px}
.card .n{font-family:"IBM Plex Mono",monospace;font-size:26px;font-weight:600;
  color:var(--accent);font-variant-numeric:tabular-nums;line-height:1.15}
.card .l{font-family:"IBM Plex Mono",monospace;font-size:10.5px;letter-spacing:.09em;
  text-transform:uppercase;color:var(--muted);margin-top:5px}
.card.warn{border-left:3px solid var(--warn);background:var(--warn-soft)}
.card.warn .n{color:var(--warn)}
.scroll{overflow-x:auto;border:1px solid var(--rule);border-radius:3px;
  background:var(--surface);margin:0 0 22px}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{padding:9px 13px;border-bottom:1px solid var(--rule);text-align:right;
  font-variant-numeric:tabular-nums;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
thead th{font-family:"IBM Plex Mono",monospace;font-size:10.5px;letter-spacing:.08em;
  text-transform:uppercase;color:var(--muted);background:var(--surface-2);
  border-bottom:1px solid var(--rule-strong)}
tbody tr:last-child td{border-bottom:none}
td.diag{background:var(--accent-soft);color:var(--ink);font-weight:600}
td.zero{color:var(--muted)}
td.miss{color:var(--warn);font-weight:600}
.bar{display:block;height:7px;border-radius:2px;background:var(--accent);min-width:2px}
.bartrack{background:var(--surface-2);border-radius:2px;width:150px;display:block}
.note{background:var(--surface);border:1px solid var(--rule);border-left:3px solid var(--accent);
  border-radius:3px;padding:16px 18px;margin:0 0 22px}
.note.warn{border-left-color:var(--warn);background:var(--warn-soft)}
.note h3{font-family:"IBM Plex Mono",monospace;font-size:10.5px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--muted);margin-bottom:8px}
.note.warn h3{color:var(--warn)}
.note p:last-child{margin-bottom:0}
dl.kv{display:grid;grid-template-columns:200px 1fr;gap:0;margin:0 0 22px;
  border-top:1px solid var(--rule)}
dl.kv dt{font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--muted);
  padding:9px 12px 9px 0;border-bottom:1px solid var(--rule)}
dl.kv dd{margin:0;padding:9px 0;border-bottom:1px solid var(--rule);font-size:13.5px;
  font-family:"IBM Plex Mono",monospace;color:var(--ink-2)}
footer{margin-top:56px;padding-top:18px;border-top:1px solid var(--rule);
  font-family:"IBM Plex Mono",monospace;font-size:11px;color:var(--muted)}
@media (max-width:680px){dl.kv{grid-template-columns:1fr}dl.kv dt{border-bottom:none;padding-bottom:0}}
"""


def _e(x: Any) -> str:
    return html.escape(str(x))


def _ms(ns: int | float) -> str:
    return f"{ns/1e6:,.1f}"


def _pct(x: float) -> str:
    return "-" if x != x else f"{x*100:.1f}%"


def _num(x: float | None) -> str:
    return "-" if x is None or x != x else f"{x:.3f}"


def _card(value: str, label: str, *, warn: bool = False) -> str:
    cls = "card warn" if warn else "card"
    return f'<div class="{cls}"><div class="n">{_e(value)}</div><div class="l">{_e(label)}</div></div>'


def _attribution_section(att: Attribution) -> str:
    out: list[str] = []
    for dev in att.devices:
        gs = dev.gapset
        total = dev.classified_ns or 1
        out.append(f"<h3>Device {dev.device}</h3>")
        out.append('<div class="cards">')
        out.append(_card(_pct(gs.utilisation), "utilisation"))
        out.append(_card(_ms(gs.idle_total_ns) + " ms", "idle total"))
        out.append(_card(str(len(gs.gaps)), "gaps classified"))
        out.append(
            _card(_ms(gs.idle_subthreshold_ns) + " ms", f"below {gs.min_gap_ns/1000:g} us floor")
        )
        out.append(_card(_ms(gs.idle_trailing_ns) + " ms", "trailing teardown"))
        out.append(
            _card(_pct(dev.unattributed_share), "unattributed", warn=dev.unattributed_share > 0.25)
        )
        out.append("</div>")

        by_ns = dev.by_class_ns()
        by_n = dev.by_class_count()
        rows = []
        for cls, ns in sorted(by_ns.items(), key=lambda kv: -kv[1]):
            share = ns / total
            info = CLASS_INFO[cls]
            rows.append(
                "<tr>"
                f"<td>{_e(info.label)}"
                f'<div style="font-size:11.5px;color:var(--muted);white-space:normal">'
                f"{_e(info.summary)}</div></td>"
                f"<td>{_ms(ns)}</td>"
                f"<td>{_pct(share)}</td>"
                f"<td>{by_n[cls]}</td>"
                f'<td><span class="bartrack"><span class="bar" '
                f'style="width:{share*100:.1f}%"></span></span></td>'
                "</tr>"
            )
        out.append('<div class="scroll"><table><thead><tr>')
        out.append(
            "<th>class</th><th>idle (ms)</th><th>share</th><th>gaps</th><th>&nbsp;</th>"
        )
        out.append("</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")

        host_share = dev.host_wait_ns / total
        out.append(
            f'<p>Of {_ms(total)} ms of classified idle, {_pct(host_share)} is host wait. '
            f"The rest is device-side or unresolved and no amount of faster storage, "
            f"more workers, or bigger batches will recover it.</p>"
        )
    return "".join(out)


def _matrix_table(matrix: ConfusionMatrix) -> str:
    preds = matrix.predictions()
    if not preds:
        return "<p>No gaps were scored.</p>"
    head = "".join(f"<th>{_e(CLASS_INFO[p].label)}</th>" for p in preds)
    rows = []
    for t in matrix.truths():
        label = CLASS_INFO[t].label if isinstance(t, StallClass) else "not injected"
        cells = []
        for p in preds:
            n = matrix.counts.get((t, p), 0)
            if n == 0:
                cells.append('<td class="zero">.</td>')
            elif isinstance(t, StallClass) and t == p:
                cells.append(f'<td class="diag">{n}</td>')
            elif isinstance(t, StallClass):
                cells.append(f'<td class="miss">{n}</td>')
            else:
                cells.append(f"<td>{n}</td>")
        rows.append(f"<tr><td>{_e(label)}</td>{''.join(cells)}</tr>")
    return (
        '<div class="scroll"><table><thead><tr><th>truth \\ predicted</th>'
        + head
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _scores_table(matrix: ConfusionMatrix) -> str:
    rows = []
    for s in matrix.scores():
        rows.append(
            "<tr>"
            f"<td>{_e(CLASS_INFO[s.cls].label)}</td>"
            f"<td>{s.support}</td><td>{s.predicted}</td><td>{s.hits}</td>"
            f"<td>{_num(s.recall)}</td><td>{_num(s.precision)}</td><td>{_num(s.f1)}</td>"
            "</tr>"
        )
    return (
        '<div class="scroll"><table><thead><tr>'
        "<th>class</th><th>support</th><th>predicted</th><th>hits</th>"
        "<th>recall</th><th>precision</th><th>F1</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _shift_table(sweep: Sweep) -> str:
    """Control versus treatment class shares: the honest effect measurement."""
    controls = sweep.control_runs()
    if not controls:
        return (
            '<div class="note warn"><h3>No control runs</h3><p>This sweep contains no '
            "uninjected run, so there is no baseline to compare against and precision "
            "cannot be interpreted. Add a control at every configuration you sweep.</p>"
            "</div>"
        )
    rows = []
    for cls, (c, t, d) in sorted(sweep.shift().items(), key=lambda kv: -abs(kv[1][2])):
        arrow = "&uarr;" if d > 0.005 else ("&darr;" if d < -0.005 else "&middot;")
        rows.append(
            "<tr>"
            f"<td>{_e(CLASS_INFO[cls].label)}</td>"
            f"<td>{_pct(c)}</td><td>{_pct(t)}</td>"
            f"<td>{arrow} {d*100:+.1f} pp</td>"
            "</tr>"
        )
    return (
        f"<h3>Control versus treatment ({len(controls)} control runs)</h3>"
        "<p>Share of gaps assigned to each class, with nothing injected and with. The "
        "shift is the measured effect of the injection; it is immune to the labelling "
        "problem above because both columns are computed the same way.</p>"
        '<div class="scroll"><table><thead><tr>'
        "<th>class</th><th>control</th><th>treatment</th><th>shift</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def _floors_table(floors: list[DetectionFloor]) -> str:
    if not floors:
        return ""
    rows = []
    for f in floors:
        floor = "-" if f.floor_us is None else f"{f.floor_us/1000:.2f}"
        bracket = (
            f"{f.below[0]/1000:.1f} ms r={f.below[1]:.2f} &rarr; "
            f"{f.above[0]/1000:.1f} ms r={f.above[1]:.2f}"
            if f.below and f.above
            else _e(f.note)
        )
        curve = ", ".join(f"{m/1000:.1f}ms:{r:.2f}" for m, r, _ in f.curve)
        rows.append(
            "<tr>"
            f"<td>{_e(CLASS_INFO[f.cls].label)}</td>"
            f"<td>{floor}</td>"
            f'<td style="text-align:left;white-space:normal">{bracket}</td>'
            f'<td class="mono" style="text-align:left;white-space:normal;font-size:11.5px">'
            f"{_e(curve)}</td>"
            "</tr>"
        )
    return (
        '<div class="scroll"><table><thead><tr>'
        "<th>class</th><th>floor (ms)</th><th>bracketing points</th><th>recall curve</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def is_synthetic(att: Attribution | None) -> bool:
    """True when the trace came from whystall.synth rather than a real run."""
    if att is None:
        return False
    meta = att.view.meta.get("whystall")
    return bool(isinstance(meta, dict) and meta.get("synthetic"))


def render(
    att: Attribution | None = None,
    *,
    sweep: Sweep | None = None,
    floors: list[DetectionFloor] | None = None,
    title: str = "whystall report",
    provenance: str | None = None,
) -> str:
    """Render a complete standalone HTML document.

    The provenance banner is not optional politeness. A report travels without
    the terminal that produced it, and a synthetic-trace number that reaches
    someone as an HTML file with no warning attached is exactly the kind of
    unfalsifiable claim this project exists to argue against.
    """
    parts: list[str] = []
    parts.append("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">")
    parts.append('<meta name="viewport" content="width=device-width,initial-scale=1">')
    parts.append(f"<title>{_e(title)}</title>")
    parts.append(
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        "family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600"
        '&display=swap">'
    )
    parts.append(f"<style>{_CSS}</style></head><body><div class=\"wrap\">")

    source = att.view.source if att else (sweep.runs[0].name if sweep and sweep.runs else "-")
    parts.append('<header><div class="rule"></div>')
    parts.append(f"<h1>{_e(title)}</h1>")
    parts.append('<div class="meta">')
    parts.append(f"<span>GENERATED <b>{time.strftime('%Y-%m-%d %H:%M')}</b></span>")
    parts.append(f"<span>SOURCE <b>{_e(Path(str(source)).name)}</b></span>")
    if sweep:
        parts.append(f"<span>RUNS <b>{len(sweep.runs)}</b></span>")
        parts.append(f"<span>SCORING <b>{_e(sweep.granularity)}</b></span>")
    parts.append("</div></header>")

    if att is not None:
        r = att.view.meta.get("whystall_resolution")
        if isinstance(r, dict) and r.get("sub_microsecond_events"):
            parts.append(
                '<section style="padding-top:26px"><div class="note">'
                "<h3>Resolution</h3><p>This trace contains "
                f"{r['sub_microsecond_events']:,} events shorter than a microsecond. "
                "They are represented exactly: times here are integer nanoseconds, "
                "taken from the trace's own fractional-microsecond values. A trace "
                "with many such events is dispatch-bound by shape, which is itself "
                "a finding.</p></div></section>"
            )

    banner = provenance
    if banner is None and is_synthetic(att):
        banner = (
            "These are synthetic traces from whystall.synth, not hardware. The numbers "
            "below show the pipeline runs end to end; they are a property of the trace "
            "model, not a measurement of anything real, and must never be quoted as one."
        )
    if banner:
        parts.append(
            '<section style="padding-top:26px"><div class="note warn">'
            f"<h3>Provenance</h3><p>{_e(banner)}</p></div></section>"
        )

    if att is not None:
        parts.append("<section><h2>Where the idle time went</h2>")
        parts.append(_attribution_section(att))
        parts.append("</section>")

    if sweep is not None and sweep.runs:
        pooled = sweep.pooled()
        parts.append("<section><h2>Validation against injected ground truth</h2>")
        parts.append('<div class="cards">')
        parts.append(_card(_num(pooled.accuracy), "accuracy"))
        parts.append(_card(_num(pooled.macro_recall), "macro recall"))
        parts.append(_card(str(pooled.total), "gaps scored"))
        ubr = pooled.unlabelled_blame_rate
        parts.append(_card(_pct(ubr), "unlabelled, named"))
        parts.append(
            _card(
                _pct(sweep.unattributed_share()),
                "unattributed",
                warn=sweep.unattributed_share() > 0.25,
            )
        )
        parts.append("</div>")
        parts.append(
            '<div class="note"><h3>How to read this</h3><p>Rows are what was injected, '
            "columns are what the classifier said. Off-diagonal cells are the interesting "
            "ones. The <em>not injected</em> row is every gap the injection did not cause: "
            "it is kept rather than discarded, because dropping it is the easiest way to "
            "publish a recall number that does not survive replication.</p>"
            "<p>Precision here is a <em>lower bound</em>, and the "
            "<em>unlabelled, named</em> figure is an upper bound on error rather than a "
            "false-positive rate. A run stalls for real reasons the injection did not "
            "create &mdash; the first fetch of an epoch, a genuine device dependency behind "
            "a copy stream &mdash; and a correct verdict on one of those is "
            "indistinguishable from a mistake here, because nothing labelled it. The "
            "control comparison below is the number that actually answers the question."
            "</p></div>"
        )
        parts.append(_matrix_table(pooled))
        parts.append(_scores_table(pooled))
        parts.append(_shift_table(sweep))

        confusions = pooled.confusions()
        if confusions:
            items = "".join(
                f"<li>{_e(CLASS_INFO[t].label)} read as {_e(CLASS_INFO[p].label)} "
                f"&mdash; {n} gaps</li>"
                for t, p, n in confusions
            )
            parts.append(
                '<div class="note warn"><h3>What it confuses</h3>'
                f"<ul style=\"margin:0;padding-left:20px;color:var(--ink-2)\">{items}</ul></div>"
            )
        if floors:
            parts.append("<h3>Detection floors</h3>")
            parts.append(
                "<p>The smallest injected stall at which recall reaches the target. "
                "Interpolated between the bracketing sweep points, which are shown so the "
                "width of the bracket is visible rather than implied.</p>"
            )
            parts.append(_floors_table(floors))
        parts.append("</section>")

    parts.append("<section><h2>Thresholds in force</h2>")
    parts.append(
        "<p>Every constant the classifier used, so the numbers above can be reproduced "
        "or contested on other hardware.</p>"
    )
    cfg: dict[str, Any] = att.config.as_dict() if att else {}
    parts.append('<dl class="kv">')
    for k, v in cfg.items():
        parts.append(f"<dt>{_e(k)}</dt><dd>{_e(v)}</dd>")
    parts.append("</dl></section>")

    parts.append(
        "<footer>whystall &middot; validated GPU stall attribution &middot; "
        "the unattributed share and the confused pairs are part of the result, not a caveat"
        "</footer>"
    )
    parts.append("</div></body></html>")
    return "".join(parts)


def write_report(
    path: str | os.PathLike[str],
    att: Attribution | None = None,
    *,
    sweep: Sweep | None = None,
    floors: list[DetectionFloor] | None = None,
    title: str = "whystall report",
    provenance: str | None = None,
    also_json: bool = True,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        render(att, sweep=sweep, floors=floors, title=title, provenance=provenance),
        encoding="utf-8",
    )
    if also_json:
        payload: dict[str, Any] = {}
        if att is not None:
            payload["attribution"] = att.summary()
        if sweep is not None:
            payload["validation"] = sweep.summary()
        if floors:
            payload["detection_floors"] = [f.summary() for f in floors]
        p.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p
