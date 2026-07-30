"""Frozen runtime-event collector for evaluator-owned backend probes.

The model evaluator instruments the exact backend callable named by a pending
manifest.  This module intentionally exposes no editable-code self-report path:
every captured event comes from that frozen probe, and the reporter AUTHENTICATES
its caller by CODE-OBJECT IDENTITY — only functions whose code object was
registered by the frozen evaluator (``_register_probe_code``, itself restricted to
frozen-tree callers) may record an event.  A direct call from editable
runner/kernel code raises instead of recording, and a ``compile()`` filename spoof
does not help at report time because the check is object identity, not a path
string.

Honest residual (documented in HARDENING.md): CPython offers no true in-process
privilege boundary — code that deliberately spoofs ``co_filename`` via ``compile``
could still self-REGISTER.  That residual is mitigated one layer up: the gate's
static pre-check scans every manifest-declared edited file for references to this
module and rejects the round, so a forgery would have to hide from both layers.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import sys


_EVENTS: ContextVar[list[dict] | None] = ContextVar("akt_runtime_events", default=None)
_SCOPE: ContextVar[dict | None] = ContextVar("akt_runtime_scope", default=None)

# Registration trust boundary: only frames originating inside the FROZEN harness
# (akt/benchmark/**) may register probe code objects.
_FROZEN_REPORTER_ROOT = Path(__file__).resolve().parent
_PROBE_CODES: set = set()


def _register_probe_code(code) -> None:
    """Register an evaluator-installed probe wrapper's code object.

    Called by ``model_eval._install_backend_probes`` for each wrapper it creates
    (and by frozen-tree tests). Restricted to frozen-tree callers."""
    caller = Path(sys._getframe(1).f_code.co_filename)
    try:
        caller.resolve().relative_to(_FROZEN_REPORTER_ROOT)
    except ValueError:
        raise PermissionError(
            "probe registration is reserved for the frozen benchmark harness "
            f"(akt/benchmark/**); rejected caller: {caller}"
        ) from None
    _PROBE_CODES.add(code)


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
    """Record a call observed by the frozen evaluator's backend wrapper.

    Fails closed on an unregistered caller: a forged report from editable code is
    a loud PermissionError (failing the surrounding eval and rejecting the round),
    never a silently-recorded ``verified_backend_probe`` event. The check is code-
    object IDENTITY (see module docstring), so spoofing a filename via ``compile``
    does not authenticate a report.
    """

    events = _EVENTS.get()
    scope = _SCOPE.get()
    if events is None or scope is None:
        return
    caller_code = sys._getframe(1).f_code
    if caller_code not in _PROBE_CODES:
        raise PermissionError(
            "backend execution events are accepted only from evaluator-registered "
            "frozen probes; rejected caller: "
            f"{caller_code.co_filename}:{caller_code.co_name}"
        )
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
