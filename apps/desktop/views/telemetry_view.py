"""apps/desktop/views/telemetry_view.py
Realtime telemetry bridge module (stub): ingest, ring buffer, observer, WebSocket publisher.
"""
from __future__ import annotations

from typing import ClassVar

from apps.desktop.views.stub_base import SpecRow, StubSpec, StubView

__all__ = ["TelemetryView"]


class TelemetryView(StubView):
    SPEC: ClassVar[StubSpec] = StubSpec(
        title="Realtime Telemetry",
        subtitle="CAN / UDP / serial ingest · ring buffer · state observer · pit-wall WebSocket",
        summary=(
            "Live link between the car and the digital twin. Frames are decoded against the DBC, buffered, "
            "fed to a state observer and published to the pit-wall dashboards. Not implemented yet: the "
            "modules under src/ter_twin/realtime/ were empty skeletons when this view was written."
        ),
        inputs=(
            SpecRow("UDP stream", "Datagram receiver for the on-car logger / radio gateway (udp_receiver.py)."),
            SpecRow("Serial link", "pyserial receiver for a direct or telemetry-radio connection (serial_receiver.py)."),
            SpecRow("CAN database", "DBC file decoded with cantools (can_decoder.py); signal scaling and units."),
            SpecRow("Vehicle config", "config/vehicles/<veh>/vehicle.yaml for the twin running in lockstep."),
            SpecRow("Offline logs", "data/raw/telemetry/*.mf4 | *.log | *.bin for replay."),
        ),
        outputs=(
            SpecRow("Ring buffer", "Fixed-size preallocated numpy buffer (state/ring_buffer.py)."),
            SpecRow("State estimates", "Observer output (state/observer.py): body velocity, slip angles, yaw rate."),
            SpecRow("WebSocket feed", "Payload schema in api/payloads.py, served by api/websocket_server.py."),
            SpecRow("Link health", "Packet rate, loss, decode errors and end-to-end latency to the status bar."),
            SpecRow("Twin residuals", "Measured vs simulated channels from pipeline/live_runner.py."),
        ),
        stack=(
            SpecRow("Concurrency",
                    "Receiver threads / asyncio tasks writing a single-producer ring buffer; the GUI polls via "
                    "AppState and never blocks on I/O."),
            SpecRow("Timing", "Monotonic timestamps with an estimated device-clock offset; late and duplicate "
                              "frames flagged, not dropped silently."),
            SpecRow("Observer", "Estimator choice (EKF / UKF vs. complementary filter) is open; interface fixed "
                                "by the ring buffer and the payload schema."),
            SpecRow("Entry point", "scripts/run_telemetry_bridge.py; unit tests in tests/unit/test_can_decoding.py."),
        ),
        roadmap=(
            "DBC decoder with unit tests against recorded frames.",
            "UDP / serial receivers with reconnect handling.",
            "Ring buffer and offline replay from data/raw/telemetry.",
            "State observer and twin residuals.",
            "WebSocket server and payload schema.",
            "Link-health panel and latency budget in this view.",
        ),
        repo_paths=(
            "src/ter_twin/realtime/ingest/can_decoder.py",
            "src/ter_twin/realtime/ingest/udp_receiver.py",
            "src/ter_twin/realtime/ingest/serial_receiver.py",
            "src/ter_twin/realtime/state/ring_buffer.py",
            "src/ter_twin/realtime/state/observer.py",
            "src/ter_twin/realtime/api/websocket_server.py",
            "src/ter_twin/realtime/pipeline/live_runner.py",
            "scripts/run_telemetry_bridge.py",
        ),
        state_keys=("active_vehicle",),
    )