from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, TypeVar

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """Raised when a circuit is open and calls should fail fast."""


@dataclass(slots=True)
class CircuitBreaker:
    """3-state circuit breaker: CLOSED → OPEN → HALF_OPEN → CLOSED.

    CLOSED  : requests pass through; failures are counted.
    OPEN    : fail fast (CircuitOpenError) until reset_timeout elapses.
    HALF_OPEN: one probe request allowed; success closes, failure re-opens.
    """

    name: str
    failure_threshold: int
    reset_timeout_seconds: float
    success_threshold: int = 1
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    opened_at: float | None = None
    transition_log: list[dict[str, str | float]] = field(default_factory=list)

    def allow_request(self) -> bool:
        """Return True if a request should be attempted.

        OPEN: deny until reset_timeout elapses, then transition to HALF_OPEN.
        HALF_OPEN / CLOSED: allow.
        """
        if self.state == CircuitState.OPEN:
            if self.opened_at is not None and time.monotonic() - self.opened_at >= self.reset_timeout_seconds:
                self._transition(CircuitState.HALF_OPEN, "reset_timeout_elapsed")
                # Reset counters so probe starts clean
                self.failure_count = 0
                self.success_count = 0
                return True
            return False
        return True

    def call(self, fn: Callable[..., T], *args: object, **kwargs: object) -> T:
        """Call a function through the circuit breaker."""
        if not self.allow_request():
            raise CircuitOpenError(f"circuit {self.name} is open")
        try:
            result = fn(*args, **kwargs)
        except Exception:
            self.record_failure()
            raise
        self.record_success()
        return result

    def record_success(self) -> None:
        """Record a successful call.

        Resets failure_count. In HALF_OPEN: close circuit when success_threshold
        consecutive successes have been observed.
        """
        self.failure_count = 0
        self.success_count += 1
        if self.state == CircuitState.HALF_OPEN and self.success_count >= self.success_threshold:
            self._transition(CircuitState.CLOSED, "probe_success")
            self.success_count = 0
            self.failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call.

        HALF_OPEN: immediately re-open (probe failed).
        CLOSED: increment failure_count and open when threshold is reached.
        """
        self.success_count = 0
        if self.state == CircuitState.HALF_OPEN:
            # Probe failed — re-open immediately; reset failure_count for next half-open cycle
            self.failure_count = 0
            self._transition(CircuitState.OPEN, "probe_failure")
            self.opened_at = time.monotonic()
        else:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self._transition(CircuitState.OPEN, "failure_threshold")
                self.opened_at = time.monotonic()

    def _transition(self, new_state: CircuitState, reason: str) -> None:
        if self.state == new_state:
            return
        self.transition_log.append(
            {"from": self.state.value, "to": new_state.value, "reason": reason, "ts": time.time()}
        )
        self.state = new_state
