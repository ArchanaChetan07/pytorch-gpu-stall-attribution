"""The guard that makes the confusion matrix mean anything.

A classifier that can see the ground-truth annotations scores perfectly and
proves nothing. The defence here is structural rather than procedural: the
loader removes every WHYSTALL annotation before a TraceView exists, so the
classifier cannot read one even by accident. This module is the assertion that
the structure held, and it runs in the test suite and in the CLI.

If you take one thing from this package into your own harness, take this. It is
the first thing a serious reviewer looks for, and its absence is the reason most
published attribution numbers cannot be checked.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..trace.schema import LABEL_PREFIX, TraceView

__all__ = ["FirewallBreach", "assert_clean", "assert_evidence_clean", "scan"]


class FirewallBreach(AssertionError):
    """Ground-truth data reached something that must not see it."""


def scan(view: TraceView) -> list[str]:
    """Return the names of any ground-truth annotations left in the view."""
    found: list[str] = []
    for bucket in (view.gpu, view.cpu, view.runtime, view.steps):
        for ev in bucket:
            if ev.name.startswith(LABEL_PREFIX):
                found.append(ev.name)
    return found


def assert_clean(view: TraceView) -> None:
    """Raise unless the view is free of ground-truth annotations."""
    found = scan(view)
    if found:
        sample = ", ".join(sorted(set(found))[:5])
        raise FirewallBreach(
            f"{len(found)} ground-truth annotation(s) reached the classifier: {sample}. "
            "Load traces with whystall.trace.load.load_trace, which strips them."
        )


def _walk(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _walk(k)
            yield from _walk(v)
    elif isinstance(value, (list, tuple, set)):
        for v in value:
            yield from _walk(v)


def assert_evidence_clean(verdicts: Iterable[Any]) -> None:
    """Raise if any verdict cites a ground-truth annotation as evidence.

    Belt to assert_clean's braces: catches a path that reintroduced labels after
    loading, which a check on the view alone would miss.
    """
    for v in verdicts:
        for s in _walk(getattr(v, "evidence", None)):
            if s.startswith(LABEL_PREFIX):
                raise FirewallBreach(
                    f"verdict for gap at {getattr(v, 'gap', None)} cites ground truth: {s}"
                )
