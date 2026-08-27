"""Scoring injected runs: ground truth, confusion matrix, detection floors."""

from .firewall import FirewallBreach, assert_clean, assert_evidence_clean
from .floor import DetectionFloor, all_floors, detection_floor
from .groundtruth import Granularity, InjectedRegion, collapse, injected_regions, label_gaps
from .matrix import ClassScore, ConfusionMatrix
from .score import RunScore, Sweep, score_attribution, score_traces

__all__ = [
    "ClassScore",
    "ConfusionMatrix",
    "DetectionFloor",
    "FirewallBreach",
    "Granularity",
    "InjectedRegion",
    "RunScore",
    "Sweep",
    "all_floors",
    "assert_clean",
    "assert_evidence_clean",
    "collapse",
    "detection_floor",
    "injected_regions",
    "label_gaps",
    "score_attribution",
    "score_traces",
]
