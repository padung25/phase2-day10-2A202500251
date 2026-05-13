from __future__ import annotations

import concurrent.futures
import copy
import json
import random
import threading
from pathlib import Path

from reliability_lab.cache import ResponseCache, SharedRedisCache
from reliability_lab.circuit_breaker import CircuitBreaker
from reliability_lab.config import LabConfig, ScenarioConfig
from reliability_lab.gateway import ReliabilityGateway
from reliability_lab.metrics import RunMetrics
from reliability_lab.providers import FakeLLMProvider


def load_queries(path: str | Path = "data/sample_queries.jsonl") -> list[str]:
    queries: list[str] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        queries.append(json.loads(line)["query"])
    return queries


def build_gateway(
    config: LabConfig,
    provider_overrides: dict[str, float] | None = None,
    cache_enabled_override: bool | None = None,
) -> ReliabilityGateway:
    """Build a ReliabilityGateway from config.

    cache_enabled_override: when not None, takes precedence over config.cache.enabled.
    Useful for circuit-breaker scenarios that need real provider calls (not cache hits).
    """
    providers = []
    for p in config.providers:
        fail_rate = provider_overrides.get(p.name, p.fail_rate) if provider_overrides else p.fail_rate
        providers.append(FakeLLMProvider(p.name, fail_rate, p.base_latency_ms, p.cost_per_1k_tokens))
    breakers = {
        p.name: CircuitBreaker(
            name=p.name,
            failure_threshold=config.circuit_breaker.failure_threshold,
            reset_timeout_seconds=config.circuit_breaker.reset_timeout_seconds,
            success_threshold=config.circuit_breaker.success_threshold,
        )
        for p in config.providers
    }

    effective_cache_enabled = (
        cache_enabled_override if cache_enabled_override is not None else config.cache.enabled
    )

    cache: ResponseCache | SharedRedisCache | None = None
    if effective_cache_enabled:
        if config.cache.backend == "redis":
            cache = SharedRedisCache(
                config.cache.redis_url,
                config.cache.ttl_seconds,
                config.cache.similarity_threshold,
            )
        else:
            cache = ResponseCache(config.cache.ttl_seconds, config.cache.similarity_threshold)

    return ReliabilityGateway(providers, breakers, cache)


def calculate_recovery_time_ms(gateway: ReliabilityGateway) -> float | None:
    """Derive recovery time from circuit breaker transition logs.

    Recovery time = elapsed time between a circuit opening and its next close.
    Returns the average across all breakers, or None if no full cycle occurred.
    """
    recovery_times: list[float] = []
    for breaker in gateway.breakers.values():
        open_ts: float | None = None
        for entry in breaker.transition_log:
            if entry["to"] == "open" and open_ts is None:
                open_ts = float(entry["ts"])
            elif entry["to"] == "closed" and open_ts is not None:
                recovery_times.append((float(entry["ts"]) - open_ts) * 1000)
                open_ts = None
    if not recovery_times:
        return None
    return sum(recovery_times) / len(recovery_times)


def run_scenario(config: LabConfig, queries: list[str], scenario: ScenarioConfig) -> RunMetrics:
    """Run a single named chaos scenario and return its metrics.

    Requests are dispatched concurrently using ThreadPoolExecutor with
    config.load_test.concurrency workers.  Results are collected into a list
    and merged into RunMetrics after all requests complete, which avoids
    shared-state races on the counters.

    Note: the CircuitBreaker state machine itself is mutated from multiple
    threads.  Python's GIL makes simple int increments safe in CPython, but
    the ordering of concurrent state transitions may vary between runs.
    """
    cache_override: bool | None = scenario.cache_enabled
    gateway = build_gateway(config, scenario.provider_overrides or None, cache_override)

    request_count = config.load_test.requests
    concurrency = config.load_test.concurrency

    def single_request(_: int) -> tuple[bool, str, float, float, bool]:
        """Returns (cache_hit, route, latency_ms, estimated_cost, is_error)."""
        prompt = random.choice(queries)
        result = gateway.complete(prompt)
        is_error = result.route == "static_fallback"
        return result.cache_hit, result.route, result.latency_ms, result.estimated_cost, is_error

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(single_request, range(request_count)))

    metrics = RunMetrics()
    for cache_hit, route, latency_ms, estimated_cost, is_error in results:
        metrics.total_requests += 1
        metrics.estimated_cost += estimated_cost

        if cache_hit:
            metrics.cache_hits += 1
            metrics.estimated_cost_saved += estimated_cost if estimated_cost else 0.001

        if route == "static_fallback":
            metrics.static_fallbacks += 1
            metrics.failed_requests += 1
        elif route.startswith("fallback:"):
            metrics.fallback_successes += 1
            metrics.successful_requests += 1
        else:
            # "primary:...", "cache_hit:..." — all successful
            metrics.successful_requests += 1

        if latency_ms:
            metrics.latencies_ms.append(latency_ms)

    metrics.circuit_open_count = sum(
        1 for breaker in gateway.breakers.values() for t in breaker.transition_log if t["to"] == "open"
    )
    metrics.recovery_time_ms = calculate_recovery_time_ms(gateway)
    return metrics


def _scenario_passed(name: str, result: RunMetrics) -> bool:
    """Return True if the scenario met its expected outcome criteria."""
    if name == "primary_timeout_100":
        # Primary is 100% fail-rate → circuit must open; backup (fallback) serves most requests.
        return result.circuit_open_count > 0 and result.fallback_success_rate > 0.8
    if name == "primary_flaky_50":
        # 50% fail-rate → circuit oscillates; mix of primary and fallback.
        return result.circuit_open_count > 0 and result.successful_requests > 0
    if name == "all_healthy":
        # Both providers healthy → high availability, no circuit opens expected.
        return result.availability > 0.9
    if name == "cache_stale_candidate":
        # Cache active; repeated queries across 100 requests produce some hits.
        return result.availability > 0.8
    return result.successful_requests > 0


def run_simulation(config: LabConfig, queries: list[str]) -> RunMetrics:
    """Run all named scenarios and aggregate metrics.

    Pass/fail criteria are scenario-specific.  A cache-vs-no-cache comparison
    is appended as a synthetic scenario so the hit-rate difference is visible
    in metrics.json without requiring a separate command.
    """
    if not config.scenarios:
        default_scenario = ScenarioConfig(name="default", description="baseline run")
        metrics = run_scenario(config, queries, default_scenario)
        metrics.scenarios = {"default": "pass" if metrics.successful_requests > 0 else "fail"}
        return metrics

    combined = RunMetrics()

    for scenario in config.scenarios:
        result = run_scenario(config, queries, scenario)
        combined.scenarios[scenario.name] = "pass" if _scenario_passed(scenario.name, result) else "fail"

        combined.total_requests += result.total_requests
        combined.successful_requests += result.successful_requests
        combined.failed_requests += result.failed_requests
        combined.fallback_successes += result.fallback_successes
        combined.static_fallbacks += result.static_fallbacks
        combined.cache_hits += result.cache_hits
        combined.circuit_open_count += result.circuit_open_count
        combined.estimated_cost += result.estimated_cost
        combined.estimated_cost_saved += result.estimated_cost_saved
        combined.latencies_ms.extend(result.latencies_ms)
        if result.recovery_time_ms is not None:
            if combined.recovery_time_ms is None:
                combined.recovery_time_ms = result.recovery_time_ms
            else:
                combined.recovery_time_ms = (combined.recovery_time_ms + result.recovery_time_ms) / 2

    # --- cache vs no-cache comparison (default provider rates, memory backend) ---
    no_cache_cfg = config.model_copy(deep=True)
    no_cache_cfg.cache.enabled = False
    no_cache_result = run_scenario(
        no_cache_cfg, queries, ScenarioConfig(name="_cache_off", description="no-cache baseline")
    )

    with_cache_cfg = config.model_copy(deep=True)
    with_cache_cfg.cache.enabled = True
    with_cache_cfg.cache.backend = "memory"
    with_cache_result = run_scenario(
        with_cache_cfg, queries, ScenarioConfig(name="_cache_on", description="with-cache comparison")
    )

    cache_passed = with_cache_result.cache_hit_rate > no_cache_result.cache_hit_rate
    combined.scenarios["cache_vs_no_cache"] = "pass" if cache_passed else "fail"

    return combined
