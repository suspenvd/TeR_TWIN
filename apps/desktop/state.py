"""apps/desktop/state.py
Reactive, thread-safe application state for TeR-Twin Studio.

The singleton AppState holds the current vehicle context, tyre selection,
and setup metadata.  External modules subscribe to state-change events via
the observer pattern; changing a value via ``set()`` automatically notifies
all registered listeners.
"""
from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any, Callable

_LOCK = threading.Lock()


class AppState:
    """Singleton reactive application state.

    Usage:
        state = AppState.instance()
        state.subscribe("active_vehicle", my_callback)
        state.set("active_vehicle", "TeR27-4WD")
    """

    _inst: AppState | None = None

    def __init__(self) -> None:
        self._data: dict[str, Any] = {
            # ---------------------------------------------------------------
            # Vehicle context
            # ---------------------------------------------------------------
            "active_vehicle": "TeR27-4WD",
            # ---------------------------------------------------------------
            # Tyre selection — populated by TiresView, consumed by LTSView etc.
            # Each value is either None or an mf.MF61Params instance.
            # ---------------------------------------------------------------
            "front_tyre_params": None,
            "rear_tyre_params": None,
            # ---------------------------------------------------------------
            # Active tyre slugs currently checked in TiresView
            # ---------------------------------------------------------------
            "active_tyre_slugs": [],
            # ---------------------------------------------------------------
            # Setup metadata
            # ---------------------------------------------------------------
            "nominal_fz": [445.0, 667.0, 1112.0],  # [N]
            "aero_balance": 0.45,                   # front fraction [-]
            "roll_stiffness_dist": 0.50,             # front fraction [-]
            # ---------------------------------------------------------------
            # Dataset path (index.json used by TiresView)
            # ---------------------------------------------------------------
            "index_path": None,
            # ---------------------------------------------------------------
            # Background compute latency (ms) — updated by active views
            # ---------------------------------------------------------------
            "last_draw_ms": 0.0,
            # ---------------------------------------------------------------
            # Status message shown in the status bar
            # ---------------------------------------------------------------
            "status_text": "Listo",
        }
        self._listeners: dict[str, list[Callable[[str, Any], None]]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Singleton accessor
    # ------------------------------------------------------------------
    @classmethod
    def instance(cls) -> "AppState":
        if cls._inst is None:
            with _LOCK:
                if cls._inst is None:
                    cls._inst = cls()
        return cls._inst

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        with _LOCK:
            return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with _LOCK:
            self._data[key] = value
            listeners = list(self._listeners.get(key, []))
        # Fire callbacks outside the lock to prevent deadlocks
        for cb in listeners:
            try:
                cb(key, value)
            except Exception:  # noqa: BLE001
                pass

    def subscribe(self, key: str, callback: Callable[[str, Any], None]) -> None:
        """Register *callback* to be called whenever *key* changes."""
        with _LOCK:
            self._listeners[key].append(callback)

    def unsubscribe(self, key: str, callback: Callable[[str, Any], None]) -> None:
        with _LOCK:
            try:
                self._listeners[key].remove(callback)
            except ValueError:
                pass

    def update_status(self, text: str) -> None:
        """Convenience helper: update status_text and last_draw_ms together."""
        self.set("status_text", text)

    def __repr__(self) -> str:
        return f"AppState({list(self._data.keys())})"
