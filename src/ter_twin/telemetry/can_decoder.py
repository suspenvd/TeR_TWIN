"""Live CAN ingestion: DBC decoding, channel remap, timestamp health, 200 Hz resampling into a RingBuffer.

Sources
-------
``socketcan``  python-can ``interface='socketcan'`` (Linux), ``channel='can0'``.
``serial``     python-can ``interface='serial'`` (USB-UART CAN bridge), ``channel='/dev/ttyUSB0'|'COM3'``.
``udp``        raw UDP datagrams, each = k frames of ``struct '<IIB8s'`` (device_ms, can_id, dlc, data[8]),
               produced by the STM32 gateway. ``device_ms`` is the board tick (HAL_GetTick) and is used as
               the session clock, so host jitter never contaminates the time base.
``replay``     (:class:`ReplayIngest`) real-time replay of arrays, for demos and tests without hardware.

Every decoded signal name goes through ``resolve_channel_name`` => the RL_Temp<->FR_Temp harness fix is
applied on ingestion. CAN frames are asynchronous; :class:`FrameAssembler` emits one row per 5 ms using
zero-order hold of the last value of each channel (the VCU loop runs at 200 Hz).
"""
from __future__ import annotations

import logging
import socket
import struct
import threading
import time
from collections import deque
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .channel_definitions import resolve_channel_name
from .ring_buffer import RingBuffer

LOG = logging.getLogger("telemetry.can")
GRID_HZ = 200.0
_UDP_FRAME = struct.Struct("<IIB8s")


class TimestampTracker:
    """Detects duplicate / late frames and estimates device-vs-host clock drift and arrival jitter."""

    def __init__(self, window: int = 256) -> None:
        self._pairs: deque[tuple[float, float]] = deque(maxlen=window)
        self.last_dev: float | None = None
        self.duplicates = 0
        self.late = 0
        self.n = 0

    def update(self, dev_s: float, host_s: float) -> None:
        self.n += 1
        if self.last_dev is not None:
            if dev_s == self.last_dev:
                self.duplicates += 1
            elif dev_s < self.last_dev:
                self.late += 1
        self.last_dev = dev_s if self.last_dev is None else max(self.last_dev, dev_s)
        self._pairs.append((dev_s, host_s))

    def stats(self) -> dict[str, float]:
        if len(self._pairs) < 32:
            return {"drift_ppm": 0.0, "jitter_ms": 0.0, "offset_s": 0.0}
        a = np.asarray(self._pairs)
        d = a[:, 1] - a[:, 0]
        x = a[:, 0] - a[0, 0]
        if np.ptp(x) < 1e-6:
            return {"drift_ppm": 0.0, "jitter_ms": 0.0, "offset_s": float(d.mean())}
        k, c = np.polyfit(x, d, 1)
        return {"drift_ppm": float(k * 1e6), "jitter_ms": float(np.std(d - (k * x + c)) * 1e3),
                "offset_s": float(c)}


class CanSignalDecoder:
    """cantools wrapper returning ``{canonical_channel: value}`` per frame."""

    def __init__(self, dbc_path: str | Path) -> None:
        import cantools  # lazy: optional dependency at import time

        self.db = cantools.database.load_file(str(dbc_path))
        self._msgs = {m.frame_id: m for m in self.db.messages}
        self._names = {(m.frame_id, s.name): resolve_channel_name(s.name) for m in self.db.messages for s in m.signals}
        self.unknown_ids = 0
        self.decode_errors = 0

    def decode(self, frame_id: int, data: bytes) -> dict[str, float]:
        msg = self._msgs.get(frame_id)
        if msg is None:
            self.unknown_ids += 1
            return {}
        try:
            raw = msg.decode(bytes(data), decode_choices=False, scaling=True, allow_truncated=True)
        except Exception:  # noqa: BLE001  # malformed frame must not kill the stream
            self.decode_errors += 1
            return {}
        out = {}
        for sig, v in raw.items():
            try:
                out[self._names[(frame_id, sig)]] = float(v)
            except (TypeError, ValueError, KeyError):
                continue
        return out


class FrameAssembler:
    """Zero-order-hold resampler: asynchronous frames -> fixed-rate rows."""

    def __init__(self, names: Sequence[str], hz: float = GRID_HZ, max_gap_s: float = 0.5) -> None:
        self.names = tuple(names)
        self._ix = {n: i for i, n in enumerate(self.names)}
        self.state = np.full(len(self.names), np.nan)
        self.dt = 1.0 / hz
        self.max_gap = max_gap_s
        self.next_t: float | None = None
        self.gaps = 0

    def push(self, t: float, values: Mapping[str, float], out_t: list, out_rows: list) -> None:
        if self.next_t is None:
            self.next_t = t
        if t - self.next_t > self.max_gap:  # link dropout: resync instead of flooding stale rows
            self.gaps += 1
            self.next_t = t
        while self.next_t < t:
            out_t.append(self.next_t)
            out_rows.append(self.state.copy())
            self.next_t += self.dt
        for k, v in values.items():
            i = self._ix.get(k)
            if i is not None:
                self.state[i] = v


class _IngestBase(threading.Thread):
    def __init__(self, buffer: RingBuffer, name: str) -> None:
        super().__init__(daemon=True, name=name)
        self.buffer = buffer
        self._stop_evt = threading.Event()
        self.error: Exception | None = None
        self.connected = False
        self.rows = 0
        self.frames = 0
        self._hist: deque[tuple[float, int]] = deque(maxlen=128)

    def stop(self) -> None:
        self._stop_evt.set()

    def _commit(self, t: np.ndarray, block: Mapping[str, np.ndarray]) -> None:
        self.buffer.append_block(t, block)
        self.rows += len(t)
        self._hist.append((time.perf_counter(), self.rows))

    def hz(self) -> float:
        if len(self._hist) < 2:
            return 0.0
        now = time.perf_counter()
        recent = [h for h in self._hist if now - h[0] <= 1.5]
        if len(recent) < 2 or recent[-1][0] - recent[0][0] < 1e-3:
            return 0.0
        return (recent[-1][1] - recent[0][1]) / (recent[-1][0] - recent[0][0])

    def stats(self) -> dict:
        return {"connected": self.connected and self.is_alive(), "hz": self.hz(), "frames": self.frames,
                "rows": self.rows, "dropped": 0, "error": repr(self.error) if self.error else "",
                "fill": self.buffer.fill_fraction}

    def run(self) -> None:
        try:
            self.connected = True
            self._run()
        except Exception as exc:  # noqa: BLE001
            LOG.exception("ingest thread failed")
            self.error = exc
        finally:
            self.connected = False

    def _run(self) -> None:  # pragma: no cover
        raise NotImplementedError


class CanIngest(_IngestBase):
    def __init__(self, buffer: RingBuffer, source: str, channel: str, dbc_path: str | Path,
                 serial_baud: int = 115200, bitrate: int = 1_000_000) -> None:
        super().__init__(buffer, f"can-ingest-{source}")
        self.source, self.channel, self.dbc_path = source, channel, dbc_path
        self.serial_baud, self.bitrate = serial_baud, bitrate
        self.tracker = TimestampTracker()
        self.asm = FrameAssembler(buffer.names)
        self.decoder: CanSignalDecoder | None = None
        self._t0: float | None = None
        self._ts: list[float] = []
        self._rows: list[np.ndarray] = []
        self._last_flush = time.perf_counter()
        self._wrap_last: int | None = None
        self._wrap_off = 0

    def stats(self) -> dict:
        s = super().stats()
        dec = self.decoder
        s.update(self.tracker.stats())
        s["dropped"] = self.tracker.late + self.tracker.duplicates + self.asm.gaps
        s["unknown_ids"] = dec.unknown_ids if dec else 0
        s["decode_errors"] = dec.decode_errors if dec else 0
        return s

    def _flush(self, force: bool = False) -> None:
        now = time.perf_counter()
        if not self._ts or not (force or len(self._ts) >= 8 or now - self._last_flush > 0.02):
            return
        arr = np.asarray(self._rows).T
        self._commit(np.asarray(self._ts), {n: arr[i] for i, n in enumerate(self.asm.names)})
        self._ts.clear()
        self._rows.clear()
        self._last_flush = now

    def _handle(self, t_dev: float, frame_id: int, payload: bytes, host_now: float) -> None:
        self.tracker.update(t_dev, host_now)
        if self._t0 is None:
            self._t0 = t_dev
        self.frames += 1
        vals = self.decoder.decode(frame_id, payload) if self.decoder else {}
        if vals:
            self.asm.push(t_dev - self._t0, vals, self._ts, self._rows)

    def _run(self) -> None:
        self.decoder = CanSignalDecoder(self.dbc_path)
        if self.source == "udp":
            self._run_udp()
        else:
            self._run_can()

    def _run_can(self) -> None:
        try:
            import can  # python-can
        except ImportError as exc:
            raise RuntimeError("python-can is required for socketcan/serial sources (pip install python-can)") from exc
        kw = {"channel": self.channel}
        if self.source == "serial":
            kw["baudrate"] = self.serial_baud
        else:
            kw["bitrate"] = self.bitrate
        bus = can.Bus(interface=self.source, **kw)
        try:
            while not self._stop_evt.is_set():
                msg = bus.recv(timeout=0.05)
                if msg is None:
                    self._flush(force=True)
                    continue
                ts = msg.timestamp or time.time()
                self._handle(ts, msg.arbitration_id, bytes(msg.data), time.perf_counter())
                self._flush()
        finally:
            bus.shutdown()

    def _run_udp(self) -> None:
        host, _, port = self.channel.rpartition(":")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host or "0.0.0.0", int(port or 5005)))
        sock.settimeout(0.05)
        try:
            while not self._stop_evt.is_set():
                try:
                    data, _ = sock.recvfrom(65535)
                except socket.timeout:
                    self._flush(force=True)
                    continue
                now = time.perf_counter()
                for off in range(0, len(data) - _UDP_FRAME.size + 1, _UDP_FRAME.size):
                    ts_ms, cid, dlc, payload = _UDP_FRAME.unpack_from(data, off)
                    if self._wrap_last is not None and ts_ms < self._wrap_last - 2 ** 31:
                        self._wrap_off += 2 ** 32  # uint32 tick wrap
                    self._wrap_last = ts_ms
                    self._handle((ts_ms + self._wrap_off) * 1e-3, cid, payload[:dlc], now)
                self._flush()
        finally:
            sock.close()


class ReplayIngest(_IngestBase):
    """Feeds a buffer in real time from arrays (demo / test source). Loops forever."""

    def __init__(self, buffer: RingBuffer, t: np.ndarray, channels: Mapping[str, np.ndarray], speed: float = 1.0) -> None:
        super().__init__(buffer, "replay-ingest")
        self.t = np.asarray(t, float)
        self.channels = {k: np.asarray(v, float) for k, v in channels.items() if k in buffer.names}
        self.fs_src = 1.0 / float(np.median(np.diff(self.t)))
        self.speed = speed

    def _run(self) -> None:
        n = self.t.size
        t0 = time.perf_counter()
        sent = 0
        while not self._stop_evt.is_set():
            target = int((time.perf_counter() - t0) * self.speed * GRID_HZ)
            if target > sent:
                g = np.arange(sent, target)
                src = ((g / GRID_HZ) * self.fs_src).astype(int) % n
                self._commit(g / GRID_HZ, {k: v[src] for k, v in self.channels.items()})
                self.frames += len(g)
                sent = target
            time.sleep(0.01)