"""Structured logs and basic run metrics (project_spec.md section 12).

`get_logger` emits one JSON object per line so logs are greppable and
machine-readable. `Timer` records wall-clock latency per named stage;
`RunMetrics` collects stage timings, counters and data-freshness
information into a dict that is written next to every experiment's
results.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if any(isinstance(h.formatter, JsonFormatter) for h in root.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.handlers[:] = [handler]
    root.setLevel(level)


def log_event(logger: logging.Logger, msg: str, level: int = logging.INFO, **fields) -> None:
    logger.log(level, msg, extra={"fields": fields})


class RunMetrics:
    """Stage latencies (seconds), counters and free-form values for one run."""

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}
        self.counters: dict[str, float] = {}
        self.values: dict[str, object] = {}

    @contextmanager
    def timer(self, stage: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.timings[stage] = self.timings.get(stage, 0.0) + time.perf_counter() - start

    def incr(self, name: str, amount: float = 1.0) -> None:
        self.counters[name] = self.counters.get(name, 0.0) + amount

    def set(self, name: str, value: object) -> None:
        self.values[name] = value

    def to_dict(self) -> dict:
        return {
            "timings_sec": {k: round(v, 4) for k, v in self.timings.items()},
            "counters": self.counters,
            "values": self.values,
        }
