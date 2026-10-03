"""apps/desktop/views/base_view.py
Abstract BaseView lifecycle interface — every module panel inherits from this.
"""
from __future__ import annotations

import tkinter as tk
from abc import ABC, abstractmethod
from tkinter import ttk
from typing import Any

from apps.desktop.state import AppState


class BaseView(ttk.Frame, ABC):
    """Abstract base class for all TeR-Twin Studio view panels.

    Subclasses must implement :meth:`on_mount`.  All other lifecycle hooks
    have sensible no-op defaults so stubs remain minimal.

    Parameters
    ----------
    parent:
        Parent Tk widget (the dynamic content container in the shell).
    app_state:
        The shared :class:`~apps.desktop.state.AppState` singleton.
    **kwargs:
        Forwarded to ``ttk.Frame.__init__``.
    """

    def __init__(
        self,
        parent: tk.Widget,
        app_state: AppState,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, **kwargs)
        self._app_state = app_state
        self._pending_afters: list[str] = []
        self._mounted = False

    # ------------------------------------------------------------------
    # Lifecycle hooks (to be overridden by subclasses)
    # ------------------------------------------------------------------

    @abstractmethod
    def on_mount(self) -> None:
        """Called exactly once, immediately after the view is instantiated.

        Build all child widgets and perform one-time initialization here.
        """

    def on_activate(self) -> None:
        """Called each time the user navigates *into* this view.

        Use to trigger lazy renders, canvas resize reconciliation, or resume
        paused background workers.
        """

    def on_deactivate(self) -> None:
        """Called when the user navigates *away* from this view.

        Cancel pending ``after`` callbacks, pause workers, etc.
        """
        self._cancel_all_afters()

    def on_state_change(self, key: str, value: Any) -> None:
        """Called via the observer pattern when global ``AppState`` changes.

        Override to react to specific keys.  The default implementation is a
        no-op so stub views do not have to implement it.
        """

    # ------------------------------------------------------------------
    # Helpers for subclasses
    # ------------------------------------------------------------------

    def _schedule(self, delay_ms: int, fn: Any, *args: Any) -> str:
        """Schedule *fn* via ``after``, tracking the ID for cancellation."""
        after_id = self.after(delay_ms, fn, *args)
        self._pending_afters.append(after_id)
        return after_id

    def _cancel_after(self, after_id: str) -> None:
        try:
            self.after_cancel(after_id)
        except Exception:  # noqa: BLE001
            pass
        try:
            self._pending_afters.remove(after_id)
        except ValueError:
            pass

    def _cancel_all_afters(self) -> None:
        for aid in list(self._pending_afters):
            try:
                self.after_cancel(aid)
            except Exception:  # noqa: BLE001
                pass
        self._pending_afters.clear()

    def _post_status(self, text: str) -> None:
        """Push a status message to the global AppState."""
        self._app_state.update_status(text)

    # ------------------------------------------------------------------
    # Internal plumbing called by the shell
    # ------------------------------------------------------------------

    def _do_mount(self) -> None:
        """Called by the shell once; wraps on_mount with guard."""
        if not self._mounted:
            self._mounted = True
            self.on_mount()
