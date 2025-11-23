"""Mechanistic Watchdog package."""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", message=r"`torch_dtype` is deprecated! Use `dtype` instead!")

from .config import WatchdogConfig
from .runtime import WatchdogRuntime, WatchdogResult

__all__ = [
    "WatchdogConfig",
    "WatchdogRuntime",
    "WatchdogResult",
]

