"""Fault injectors and the run harness.

Importing this package does not require torch. Individual injectors import it
lazily inside the hooks that actually touch a device, so a manifest can be read
and a sweep planned on a laptop that will never run one.
"""

from .base import Injector, LabelScope, NullInjector, emit_clock_sync, inject_label
from .harness import Harness, HarnessConfig
from .injectors import (
    REGISTRY,
    CollectiveStraggler,
    DataloaderCPU,
    DataloaderIO,
    DeviceDependency,
    HostSync,
    HostToDevice,
    LaunchBound,
    OtherHostCompute,
    build,
)

__all__ = [
    "REGISTRY",
    "CollectiveStraggler",
    "DataloaderCPU",
    "DataloaderIO",
    "DeviceDependency",
    "Harness",
    "HarnessConfig",
    "HostSync",
    "HostToDevice",
    "Injector",
    "LabelScope",
    "LaunchBound",
    "NullInjector",
    "OtherHostCompute",
    "build",
    "emit_clock_sync",
    "inject_label",
]
