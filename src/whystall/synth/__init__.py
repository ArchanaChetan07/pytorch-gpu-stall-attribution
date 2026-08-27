"""Synthetic Kineto traces, for development and regression tests without a GPU."""
from .builder import DeviceConfig, TraceBuilder
from .scenarios import GROUND_TRUTH_CLASS, STALL_KINDS, build_scenario

__all__ = ["DeviceConfig", "TraceBuilder", "build_scenario", "STALL_KINDS", "GROUND_TRUTH_CLASS"]
