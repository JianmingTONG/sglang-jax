"""Frozen runtime-event collector for evaluator-owned backend probes.

The model evaluator instruments the exact backend callable named by a pending
manifest.  This module intentionally exposes no editable-code self-report path:
every captured event comes from that frozen probe.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar


_EVENTS: ContextVar[list[dict] | None] = ContextVar("akt_runtime_events", default=None)
_SCOPE: ContextVar[dict | None] = ContextVar("akt_runtime_scope", default=None)


@contextmanager
def capture_runtime_events():
    events: list[dict] = []
    token = _EVENTS.set(events)
    try:
        yield events
    finally:
        _EVENTS.reset(token)


@contextmanager
def runtime_scope(model_id: str, callsite: str):
    token = _SCOPE.set({"model": model_id, "callsite": callsite})
    try:
        yield
    finally:
        _SCOPE.reset(token)


def _report_probed_backend_execution(
    *,
    capability: str,
    control: str,
    value,
    backend: str,
) -> None:
    """Record a call observed by the frozen evaluator's backend wrapper."""

    events = _EVENTS.get()
    scope = _SCOPE.get()
    if events is None or scope is None:
        return
    events.append(
        {
            **scope,
            "capability": capability,
            "control": control,
            "value": value,
            "backend": backend,
            # Kept for the existing frozen evaluator/result schema. Since the
            # self-report API no longer exists, every recorded event is verified.
            "verified_backend_probe": True,
        }
    )


__all__ = ["capture_runtime_events", "runtime_scope"]
