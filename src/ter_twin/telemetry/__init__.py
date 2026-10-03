"""TeR-Twin telemetry: ingestion, decoding, buffering, virtual channels and lap analysis.

Heavy / optional dependencies (jax, cantools, python-can, asammdf) are imported lazily inside the
functions that need them, so ``import ter_twin.telemetry`` is cheap and headless-safe.
"""
from .can_decoder import CanIngest, CanSignalDecoder, FrameAssembler, ReplayIngest, TimestampTracker
from .channel_definitions import (CHANNELS, CORNERS, POWER_LIMIT_KW, RAW_CHANNELS, ChannelSpec, get_spec,
                                  resolve_channel_name)
from .lap_analyzer import Lap, LapAnalyzer, format_lap_time
from .log_loader import LogData, load_log, make_demo_log
from .math_channels import compute_math_channels, friction_ellipses_g
from .ring_buffer import RingBuffer, decimate_indices, lttb_indices, minmax_indices

__all__ = [
    "CanIngest", "CanSignalDecoder", "FrameAssembler", "ReplayIngest", "TimestampTracker", "CHANNELS", "CORNERS",
    "POWER_LIMIT_KW", "RAW_CHANNELS", "ChannelSpec", "get_spec", "resolve_channel_name", "Lap", "LapAnalyzer",
    "format_lap_time", "LogData", "load_log", "make_demo_log", "compute_math_channels", "friction_ellipses_g",
    "RingBuffer", "decimate_indices", "lttb_indices", "minmax_indices",
]