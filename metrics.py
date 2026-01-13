from __future__ import annotations

import os
import threading
from typing import Optional

from dist_utils import get_distributed_context, is_truthy_env

_PROM_AVAILABLE = False
try:
    from prometheus_client import Counter, Histogram, Info, start_http_server  # type: ignore

    _PROM_AVAILABLE = True
except Exception:
    Counter = None  # type: ignore
    Histogram = None  # type: ignore
    Info = None  # type: ignore
    start_http_server = None  # type: ignore


_LOCK = threading.Lock()
_HTTP_STARTED = False


# A small, stable set of metrics for this demo project.
if _PROM_AVAILABLE:
    PROCESS_INFO = Info(
        "news_agent_process_info",
        "News-agent process metadata.",
    )

    NEWSAPI_REQUESTS_TOTAL = Counter(
        "news_agent_newsapi_requests_total",
        "Total NewsAPI HTTP requests.",
        labelnames=["result"],
    )
    NEWSAPI_REQUEST_LATENCY_SECONDS = Histogram(
        "news_agent_newsapi_request_latency_seconds",
        "Latency of NewsAPI HTTP requests.",
    )
    NEWSAPI_ARTICLES_RETURNED_TOTAL = Counter(
        "news_agent_newsapi_articles_returned_total",
        "Number of articles returned by NewsAPI.",
    )

    HF_CHAT_REQUESTS_TOTAL = Counter(
        "news_agent_hf_chat_completion_requests_total",
        "Total Hugging Face Inference chat_completion calls.",
        labelnames=["result"],
    )
    HF_CHAT_LATENCY_SECONDS = Histogram(
        "news_agent_hf_chat_completion_latency_seconds",
        "Latency of Hugging Face Inference chat_completion calls.",
    )
    ARTICLES_SUMMARIZED_TOTAL = Counter(
        "news_agent_articles_summarized_total",
        "Number of articles attempted to be summarized.",
        labelnames=["result"],
    )

    EXCEPTIONS_TOTAL = Counter(
        "news_agent_exceptions_total",
        "Unhandled exceptions (best-effort).",
        labelnames=["where"],
    )
else:
    PROCESS_INFO = None
    NEWSAPI_REQUESTS_TOTAL = None
    NEWSAPI_REQUEST_LATENCY_SECONDS = None
    NEWSAPI_ARTICLES_RETURNED_TOTAL = None
    HF_CHAT_REQUESTS_TOTAL = None
    HF_CHAT_LATENCY_SECONDS = None
    ARTICLES_SUMMARIZED_TOTAL = None
    EXCEPTIONS_TOTAL = None


def _should_start_http_server() -> bool:
    if not _PROM_AVAILABLE:
        return False

    # Allow disabling metrics completely.
    if not is_truthy_env("NEWS_AGENT_METRICS_ENABLED", default=True):
        return False

    # Avoid port conflicts when multiple local ranks share one pod.
    ctx = get_distributed_context()
    if ctx.local_rank is not None and int(ctx.local_rank) != 0:
        return False

    return True


def setup() -> None:
    """Initialize metrics for the current process.

    - Starts an HTTP /metrics server (default :8000) when enabled.
    - Publishes a small 'process info' metric with rank/world_size metadata.

    Safe to call multiple times.
    """

    global _HTTP_STARTED

    if not _PROM_AVAILABLE:
        return

    # Always set info (best-effort) even if HTTP is disabled; callers might push metrics in the future.
    _set_process_info()

    if not _should_start_http_server():
        return

    with _LOCK:
        if _HTTP_STARTED:
            return

        port = int(os.getenv("NEWS_AGENT_METRICS_PORT", "8000"))
        addr = os.getenv("NEWS_AGENT_METRICS_ADDR", "0.0.0.0")

        try:
            # starts a background thread
            start_http_server(port, addr=addr)  # type: ignore[misc]
            _HTTP_STARTED = True
        except Exception:
            # Best-effort only: do not fail the job if metrics can't bind.
            _HTTP_STARTED = False


def _set_process_info() -> None:
    if PROCESS_INFO is None:
        return

    ctx = get_distributed_context()

    PROCESS_INFO.info(
        {
            "hostname": os.getenv("HOSTNAME", ""),
            "rank": str(ctx.rank),
            "world_size": str(ctx.world_size),
            "local_rank": "" if ctx.local_rank is None else str(ctx.local_rank),
            "distributed_mode": (os.getenv("NEWS_AGENT_DISTRIBUTED_MODE") or "").strip().lower(),
            "distributed_backend": (os.getenv("NEWS_AGENT_DISTRIBUTED_BACKEND") or "").strip().lower(),
        }
    )


def observe_newsapi_request(*, ok: bool, duration_seconds: float, articles: int = 0) -> None:
    if NEWSAPI_REQUEST_LATENCY_SECONDS is not None:
        try:
            NEWSAPI_REQUEST_LATENCY_SECONDS.observe(float(duration_seconds))
        except Exception:
            pass

    if NEWSAPI_REQUESTS_TOTAL is not None:
        try:
            NEWSAPI_REQUESTS_TOTAL.labels(result="ok" if ok else "error").inc()
        except Exception:
            pass

    if ok and articles and NEWSAPI_ARTICLES_RETURNED_TOTAL is not None:
        try:
            NEWSAPI_ARTICLES_RETURNED_TOTAL.inc(int(articles))
        except Exception:
            pass


def observe_hf_chat_completion(*, ok: bool, duration_seconds: float) -> None:
    if HF_CHAT_LATENCY_SECONDS is not None:
        try:
            HF_CHAT_LATENCY_SECONDS.observe(float(duration_seconds))
        except Exception:
            pass

    if HF_CHAT_REQUESTS_TOTAL is not None:
        try:
            HF_CHAT_REQUESTS_TOTAL.labels(result="ok" if ok else "error").inc()
        except Exception:
            pass


def inc_articles_summarized(*, ok: bool) -> None:
    if ARTICLES_SUMMARIZED_TOTAL is None:
        return
    try:
        ARTICLES_SUMMARIZED_TOTAL.labels(result="ok" if ok else "error").inc()
    except Exception:
        pass


def inc_exception(where: str) -> None:
    if EXCEPTIONS_TOTAL is None:
        return
    try:
        EXCEPTIONS_TOTAL.labels(where=str(where)).inc()
    except Exception:
        pass
