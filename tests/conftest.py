from __future__ import annotations

import pytest

from whystall.analysis import attribute
from whystall.synth import build_scenario
from whystall.trace.load import load_trace_dict


@pytest.fixture
def make_attribution():
    """Build, load and attribute a synthetic scenario in one call."""

    def _make(kind: str, **kwargs):
        doc = build_scenario(kind, **kwargs)
        view, labels = load_trace_dict(doc)
        return attribute(view, labels=labels)

    return _make


@pytest.fixture
def make_view():
    def _make(kind: str, **kwargs):
        return load_trace_dict(build_scenario(kind, **kwargs))

    return _make
