"""Optional model-serving endpoint (project_spec.md section 12).

    python -m mltrading.models.serve --port 8000

A deliberately small stdlib HTTP server that demonstrates serving and
health/metadata reporting, not a production service.

    GET /health                      model ids, feature version, data freshness
    GET /predict?model=ridge&top=10  latest cross-sectional scores + model_id
    GET /metrics                     request counts and latency summary

Returns 503 with a JSON error if the data is stale (see models/infer.py).
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from mltrading.features.build import FEATURE_TABLE_PATH
from mltrading.models.infer import FeatureVersionMismatch, StaleDataError, score_latest
from mltrading.models.registry import ARTIFACT_DIR
from mltrading.observability import configure_logging, log_event

logger = logging.getLogger(__name__)
KNOWN_MODELS = ("mean", "reversal_5d", "ols", "ridge", "hgb")


class Service:
    def __init__(self, features_path: Path = FEATURE_TABLE_PATH, artifact_dir: Path = ARTIFACT_DIR, max_age_days: int = 7):
        self.features_path, self.artifact_dir, self.max_age_days = features_path, artifact_dir, max_age_days
        self.features = pd.read_parquet(features_path)
        self.model_cache: dict = {}
        self.lock = threading.Lock()
        self.latencies_ms: list[float] = []
        self.counts = {"requests": 0, "errors": 0}

    def predict(self, model: str, top: int | None):
        result = score_latest(self.features, model, self.artifact_dir, max_age_days=self.max_age_days, model_cache=self.model_cache)
        preds = result.predictions if not top else result.predictions.head(top)
        return {"health": result.health, "predictions": json.loads(preds.assign(date=preds["date"].astype(str)).to_json(orient="records"))}

    def health(self):
        last = self.features["date"].max()
        from datetime import UTC, datetime

        age = (datetime.now(UTC).date() - last.date()).days
        return {"status": "ok" if age <= self.max_age_days else "stale", "last_bar_date": str(last.date()), "age_days": age,
                "n_tickers": int(self.features["ticker"].nunique())}

    def metrics(self):
        lat = self.latencies_ms
        return {**self.counts, "latency_ms_p50": statistics.median(lat) if lat else None,
                "latency_ms_max": max(lat) if lat else None}


def make_handler(service: Service):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            started = time.perf_counter()
            url = urlparse(self.path)
            query = parse_qs(url.query)
            with service.lock:
                service.counts["requests"] += 1
            try:
                if url.path == "/health":
                    self._send(200, service.health())
                elif url.path == "/metrics":
                    self._send(200, service.metrics())
                elif url.path == "/predict":
                    model = query.get("model", ["ridge"])[0]
                    if model not in KNOWN_MODELS:
                        self._send(400, {"error": f"unknown model {model!r}", "known": list(KNOWN_MODELS)})
                        return
                    top = int(query["top"][0]) if "top" in query else None
                    self._send(200, service.predict(model, top))
                else:
                    self._send(404, {"error": "not found"})
            except StaleDataError as exc:
                self._error(503, "stale_data", exc)
            except FeatureVersionMismatch as exc:
                self._error(409, "feature_version_mismatch", exc)
            except FileNotFoundError as exc:
                self._error(404, "model_not_trained", exc)
            except Exception as exc:  # noqa: BLE001 - last-resort handler, logged and reported as 500
                self._error(500, "internal_error", exc)
            finally:
                with service.lock:
                    service.latencies_ms.append((time.perf_counter() - started) * 1000)

        def _error(self, code: int, kind: str, exc: Exception) -> None:
            with service.lock:
                service.counts["errors"] += 1
            log_event(logger, "request failed", level=logging.WARNING, kind=kind, detail=str(exc))
            self._send(code, {"error": kind, "detail": str(exc)})

        def log_message(self, format, *args):
            pass

    return Handler


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-age-days", type=int, default=7)
    args = parser.parse_args()
    service = Service(max_age_days=args.max_age_days)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(service))
    log_event(logger, "serving", port=args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
