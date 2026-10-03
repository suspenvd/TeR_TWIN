"""Strongly-typed telemetry channel metadata for TeR-Twin.

Conventions
-----------
* Canonical channel names are lower-case snake_case (``vx``, ``wheel_speed_fl`` ...). Wheel order is
  FL, FR, RL, RR everywhere (same as ``ter_twin.models.vehicle``).
* Units are the *display/log* units of this module (km/h, g, deg, bar, kW ...). The vehicle model uses SI;
  ``math_channels`` converts.
* ay > 0 is a LEFT turn (ISO 8855, same as the vehicle model); ax > 0 is acceleration.
* Damper travel > 0 = compression. ``steer_angle`` is the STEERING-WHEEL angle, positive to the left.

Hardware wiring correction (CRITICAL)
-------------------------------------
Because of the harness pin-out the logged IR tyre-temperature signals are transposed:

    logged ``RL_Temp`` -> physical Front-Right   (``tire_temp_fr``)
    logged ``FR_Temp`` -> physical Rear-Left     (``tire_temp_rl``)
    logged ``FL_Temp`` -> physical Front-Left    (``tire_temp_fl``)   (unchanged)
    logged ``RR_Temp`` -> physical Rear-Right    (``tire_temp_rr``)   (unchanged)

The correction lives in ONE place: the ``dbc_signal_name`` of each ``tire_temp_*`` spec holds the name the
logger/DBC uses for that physical sensor, and ``resolve_channel_name`` maps every incoming name through
this table. CAN decoder and every log loader go through ``resolve_channel_name``, so the remap is applied
automatically on ingestion. Names that are already physical (``tire_temp_fr`` ...) are NOT swapped again,
which keeps the remap idempotent for files exported by this suite.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

CORNERS: tuple[str, ...] = ("fl", "fr", "rl", "rr")
CORNER_COLORS: dict[str, str] = {"fl": "#58a6ff", "fr": "#f78166", "rl": "#3fb950", "rr": "#bc8cff"}
POWER_LIMIT_KW = 80.0
SUBSYSTEMS: tuple[str, ...] = ("POWERTRAIN", "CHASSIS", "DRIVER", "TIRES", "SUSPENSION", "BATTERY",
                               "POSITION", "VIRTUAL")

# physical corner -> name used by the logger / DBC for the IR sensor mounted there
TIRE_TEMP_LOGGED_NAME: dict[str, str] = {"fl": "FL_Temp", "fr": "RL_Temp", "rl": "FR_Temp", "rr": "RR_Temp"}


@dataclass(frozen=True, slots=True)
class ChannelSpec:
    name: str
    dbc_signal_name: str
    units: str
    display_name: str
    min_val: float
    max_val: float
    color_hex: str
    precision: int = 2
    subsystem: str = "CHASSIS"
    virtual: bool = False
    linestyle: str = "-"
    known: bool = True


def _mix(hex_color: str, w: float) -> str:
    """Blend a colour towards white by fraction ``w``."""
    c = hex_color.lstrip("#")
    rgb = [int(c[i:i + 2], 16) for i in (0, 2, 4)]
    return "#%02x%02x%02x" % tuple(int(v + (255 - v) * w) for v in rgb)


CHANNELS: dict[str, ChannelSpec] = {}


def _add(name: str, units: str, display: str, lo: float, hi: float, color: str, prec: int = 2,
         sub: str = "CHASSIS", dbc: str | None = None, virtual: bool = False, ls: str = "-") -> None:
    CHANNELS[name] = ChannelSpec(name, dbc or name, units, display, float(lo), float(hi), color, prec, sub,
                                 virtual, ls)


# ---- speeds, accelerations, driver inputs ------------------------------------------------------------
_add("vx", "km/h", "Vehicle speed", 0, 120, "#39c5cf", 1)
_add("ax", "g", "Long. accel", -2.5, 2.0, "#e6edf3", 2)
_add("ay", "g", "Lat. accel", -2.5, 2.5, "#58a6ff", 2)
_add("yaw_rate", "deg/s", "Yaw rate", -120, 120, "#db61a2", 1)
_add("steer_angle", "deg", "Steering angle", -120, 120, "#f0883e", 1, "DRIVER")
_add("throttle_pct", "%", "Throttle", 0, 100, "#3fb950", 1, "DRIVER")
_add("brake_press_front", "bar", "Brake press. front", 0, 100, "#f85149", 1, "DRIVER")
_add("brake_press_rear", "bar", "Brake press. rear", 0, 100, "#ff7b72", 1, "DRIVER", ls=":")

# ---- per-corner channels -----------------------------------------------------------------------------
for _c in CORNERS:
    _u, _col = _c.upper(), CORNER_COLORS[_c]
    _add(f"wheel_speed_{_c}", "km/h", f"Wheel speed {_u}", 0, 120, _col, 1, "POWERTRAIN")
    _add(f"motor_torque_{_c}", "N·m", f"Motor torque {_u}", -200, 280, _col, 1, "POWERTRAIN")
    _add(f"motor_temp_{_c}", "°C", f"Motor temp {_u}", 20, 140, _col, 1, "POWERTRAIN")
    _add(f"inverter_temp_{_c}", "°C", f"Inverter temp {_u}", 20, 110, _mix(_col, 0.35), 1, "POWERTRAIN", ls="--")
    _add(f"damper_travel_{_c}", "mm", f"Damper travel {_u}", 0, 60, _col, 1, "SUSPENSION")
    _add(f"damper_velocity_{_c}", "mm/s", f"Damper velocity {_u}", -400, 400, _mix(_col, 0.35), 0,
         "SUSPENSION", ls="--")
    _add(f"tire_temp_{_c}", "°C", f"Tire temp {_u}", 20, 110, _col, 1, "TIRES",
         dbc=TIRE_TEMP_LOGGED_NAME[_c], ls="-.")
    # virtual (computed by math_channels)
    _add(f"slip_ratio_{_c}", "%", f"Slip ratio {_u}", -30, 30, _mix(_col, 0.35), 1, "VIRTUAL", virtual=True, ls="--")
    _add(f"slip_angle_{_c}", "deg", f"Slip angle {_u}", -15, 15, _col, 2, "VIRTUAL", virtual=True)
    _add(f"fz_{_c}", "N", f"Dynamic Fz {_u}", 0, 2500, _col, 0, "VIRTUAL", virtual=True)
    _add(f"tire_util_{_c}", "-", f"Tire utilisation {_u}", 0, 1.4, _col, 2, "VIRTUAL", virtual=True)

# ---- battery / position / misc raw -------------------------------------------------------------------
_add("battery_voltage", "V", "Battery voltage", 300, 450, "#e3b341", 1, "BATTERY")
_add("battery_current", "A", "Battery current", -150, 300, "#d29922", 1, "BATTERY")
_add("battery_power", "kW", "Battery power (BMS)", -40, 100, "#e3b341", 1, "BATTERY")
_add("soc", "%", "State of charge", 0, 100, "#56d364", 1, "BATTERY")
_add("min_cell_voltage", "V", "Min cell voltage", 2.8, 4.25, "#a5d6ff", 3, "BATTERY")
_add("gps_lat", "deg", "GPS latitude", -90, 90, "#8b949e", 7, "POSITION")
_add("gps_lon", "deg", "GPS longitude", -180, 180, "#8b949e", 7, "POSITION")
_add("lap_beacon", "-", "Lap beacon", 0, 1, "#8b949e", 0, "POSITION")

# ---- other virtual channels --------------------------------------------------------------------------
_add("battery_power_kw", "kW", "Accumulator power", -40, 100, "#e3b341", 1, "VIRTUAL", virtual=True)
_add("power_limit_delta", "kW", "Power - 80 kW limit", -100, 30, "#f85149", 1, "VIRTUAL", virtual=True)
_add("distance", "m", "Distance", 0, 5000, "#8b949e", 1, "VIRTUAL", virtual=True)
_add("vy_est", "m/s", "Lateral speed (est.)", -5, 5, "#8b949e", 2, "VIRTUAL", virtual=True)
_add("roll_angle", "deg", "Roll angle", -4, 4, "#8b949e", 2, "VIRTUAL", virtual=True)
_add("g_total", "g", "Total accel", 0, 3, "#8b949e", 2, "VIRTUAL", virtual=True)
_add("tv_yaw_moment", "N·m", "TV yaw moment", -1500, 1500, "#bc8cff", 0, "VIRTUAL", virtual=True)
_add("track_x", "m", "Track X (east)", -1000, 1000, "#8b949e", 1, "VIRTUAL", virtual=True)
_add("track_y", "m", "Track Y (north)", -1000, 1000, "#8b949e", 1, "VIRTUAL", virtual=True)

RAW_CHANNELS: tuple[str, ...] = tuple(n for n, s in CHANNELS.items() if not s.virtual)
VIRTUAL_CHANNELS: tuple[str, ...] = tuple(n for n, s in CHANNELS.items() if s.virtual)

# ---- name resolution ---------------------------------------------------------------------------------
_EXTRA_ALIASES = {
    "speed": "vx", "vehicle_speed": "vx", "yawrate": "yaw_rate", "steering": "steer_angle",
    "steer": "steer_angle", "throttle": "throttle_pct", "tps": "throttle_pct", "latitude": "gps_lat",
    "lat": "gps_lat", "longitude": "gps_lon", "lon": "gps_lon", "beacon": "lap_beacon",
    "accel_x": "ax", "accel_y": "ay", "brake_front": "brake_press_front", "brake_rear": "brake_press_rear",
}


def normalize_name(raw: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(raw).strip().lower()).strip("_")


_ALIASES: dict[str, str] = {}
for _s in CHANNELS.values():
    for _k in (_s.name, _s.dbc_signal_name, _s.display_name):
        _ALIASES.setdefault(normalize_name(_k), _s.name)
for _k, _v in _EXTRA_ALIASES.items():
    _ALIASES.setdefault(_k, _v)


def resolve_channel_name(raw: str) -> str:
    """Map a logged/DBC signal name to the canonical (physical) channel name. Applies the tyre-temp remap."""
    n = normalize_name(raw)
    return _ALIASES.get(n, n)


def get_spec(name: str) -> ChannelSpec:
    sp = CHANNELS.get(name)
    if sp is not None:
        return sp
    return ChannelSpec(name, name, "", name, 0.0, 1.0, "#8b949e", 3, "VIRTUAL", True, "-", False)